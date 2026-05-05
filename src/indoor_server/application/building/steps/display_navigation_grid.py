"""Display navigation grid — polygon-contained 2D routing graph.

This step turns a finished 2D footprint polygon into the graph that the user
actually sees and routes over on the 2D map. It is intentionally separate from
the real-world RTABMap graph:

* corridor nodes are dense tile centers inside the footprint;
* adjacent tiles are connected only when the segment stays inside the polygon;
* POI/stair/elevator nodes are projected into the polygon and attached to
  nearby tile nodes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from uuid import UUID, uuid5

from shapely.affinity import rotate
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    mapping,
    shape,
)
from shapely.ops import nearest_points, unary_union

from indoor_server.application.building.arkit_to_rtabmap_transform import (
    ArkitToRtabmapTransform,
)
from indoor_server.application.routing.vertical_connectors import make_connector_key
from indoor_server.domain.building.enums import EdgeType, NodeType
from indoor_server.domain.building.models import MapEdgeVO, MapNodeVO
from indoor_server.domain.scan.models import POIMarkRow

Point2D = tuple[float, float]
GridKey = tuple[int, int]


@dataclass(frozen=True)
class DisplayNavigationGridParams:
    """Display-routing graph knobs.

    cell_m controls visual/routing granularity. clearance_m shrinks the tile
    node substrate away from walls, while POI nodes still project into the full
    footprint and then attach back to the safe substrate.
    """

    cell_m: float = 0.45
    clearance_m: float = 0.30
    connectivity: int = 8
    poi_attach_k: int = 4
    poi_attach_radius_m: float | None = None
    component_bridge_enabled: bool = True
    component_bridge_max_gap_m: float = 2.10
    component_bridge_width_m: float | None = None
    min_edge_length_m: float = 1e-6


@dataclass(frozen=True)
class DisplayProjection:
    """world/VPS point projected onto the 2D display graph."""

    raw_xyz: tuple[float, float, float]
    world_xyz: tuple[float, float, float]
    display_xyz: tuple[float, float, float]
    nearest_node_id: UUID | None
    distance_to_polygon_m: float
    distance_to_nearest_node_m: float | None
    source: str


@dataclass(frozen=True)
class DisplayNavigationGridResult:
    nodes: list[MapNodeVO]
    edges: list[MapEdgeVO]
    poi_world_poses: dict[int, tuple[float, float, float]]
    poi_position_metadata: dict[int, dict[str, object]]
    footprint_geojson: dict[str, object]
    metadata: dict[str, object]


@dataclass
class _TileGraph:
    nodes: list[Point2D] = field(default_factory=list)
    grid_to_node: dict[GridKey, int] = field(default_factory=dict)
    edges: list[tuple[int, int]] = field(default_factory=list)
    adjacency: dict[int, list[int]] = field(default_factory=dict)
    candidate_neighbor_pairs: int = 0
    blocked_neighbor_pairs_outside_polygon: int = 0


class DisplayNavigationGridStep:
    """Build dense display-routing graph and attach semantic destination nodes."""

    def __init__(
        self,
        scan_id: UUID,
        build_job_id: UUID,
        *,
        params: DisplayNavigationGridParams | None = None,
        arkit_to_rtabmap_transform: ArkitToRtabmapTransform | None = None,
    ) -> None:
        self._scan_id = scan_id
        self._build_job_id = build_job_id
        self._params = params or DisplayNavigationGridParams()
        self._transform = arkit_to_rtabmap_transform

    def run(
        self,
        *,
        footprint_geojson: dict[str, object],
        floor_z: float,
        pois: list[POIMarkRow],
    ) -> DisplayNavigationGridResult:
        """Build graph nodes/edges in the same local metric frame as footprint."""
        footprint = _load_polygonal_footprint(footprint_geojson)
        if footprint.is_empty:
            return DisplayNavigationGridResult(
                nodes=[],
                edges=[],
                poi_world_poses={},
                poi_position_metadata={},
                footprint_geojson=dict(footprint_geojson),
                metadata={
                    "graph_source": "display_navigation_grid",
                    "accepted": False,
                    "reason": "empty_footprint",
                },
            )

        raw_component_count = _polygon_component_count(footprint)
        footprint, bridge_metadata = _bridge_nearby_polygon_components(
            footprint,
            params=self._params,
        )
        dominant_angle_deg = _dominant_axis_angle_deg(footprint)
        graph, clearance_used_m, clearance_retry = _build_connected_aligned_tile_graph(
            footprint,
            dominant_angle_deg=dominant_angle_deg,
            params=self._params,
        )

        nodes = self._tile_nodes_to_map_nodes(graph, floor_z)
        edges = self._tile_edges_to_map_edges(graph, nodes, floor_z)

        poi_world_poses: dict[int, tuple[float, float, float]] = {}
        poi_metadata: dict[int, dict[str, object]] = {}
        poi_nodes, poi_edges = self._build_poi_nodes_and_edges(
            pois=pois,
            footprint=footprint,
            tile_nodes=nodes,
            floor_z=floor_z,
            poi_world_poses=poi_world_poses,
            poi_metadata=poi_metadata,
        )

        all_nodes = nodes + poi_nodes
        all_edges = edges + poi_edges
        valid_neighbor_pairs = (
            graph.candidate_neighbor_pairs
            - graph.blocked_neighbor_pairs_outside_polygon
        )
        connected_ratio = (
            len(graph.edges) / valid_neighbor_pairs if valid_neighbor_pairs else 0.0
        )

        return DisplayNavigationGridResult(
            nodes=all_nodes,
            edges=all_edges,
            poi_world_poses=poi_world_poses,
            poi_position_metadata=poi_metadata,
            footprint_geojson=mapping(footprint),
            metadata={
                "graph_source": "display_navigation_grid",
                "accepted": bool(nodes),
                "cell_m": self._params.cell_m,
                "clearance_m": self._params.clearance_m,
                "clearance_used_m": clearance_used_m,
                "clearance_retry": clearance_retry,
                "connectivity": self._params.connectivity,
                "dominant_angle_deg": dominant_angle_deg,
                "raw_polygon_component_count": raw_component_count,
                "polygon_component_bridge": bridge_metadata,
                "corridor_nodes": len(nodes),
                "corridor_edges": len(edges),
                "connected_component_count": _component_count(graph),
                "poi_nodes": len(poi_nodes),
                "poi_attach_edges": len(poi_edges),
                "candidate_neighbor_pairs": graph.candidate_neighbor_pairs,
                "valid_neighbor_pairs_inside_polygon": valid_neighbor_pairs,
                "blocked_neighbor_pairs_outside_polygon": (
                    graph.blocked_neighbor_pairs_outside_polygon
                ),
                "valid_neighbor_pair_connection_ratio": connected_ratio,
                "polygon_area_m2": float(footprint.area),
                "nodes_per_m2": float(len(nodes) / footprint.area)
                if footprint.area > 0
                else 0.0,
            },
        )

    def project_world_to_display(
        self,
        *,
        footprint_geojson: dict[str, object],
        tile_nodes: list[MapNodeVO],
        world_xyz: tuple[float, float, float],
    ) -> DisplayProjection:
        """Project a VPS/RTABMap world point into the footprint and nearest tile."""
        footprint = _load_polygonal_footprint(footprint_geojson)
        display_xy, source, distance_to_polygon = _project_xy_into_polygon(
            (world_xyz[0], world_xyz[1]),
            footprint,
        )
        nearest = _nearest_corridor_node(display_xy, tile_nodes)
        distance_to_node = (
            _distance_2d(display_xy, (nearest.x, nearest.y)) if nearest else None
        )
        z = nearest.z if nearest is not None else world_xyz[2]
        display_xyz = (
            nearest.x if nearest is not None else display_xy[0],
            nearest.y if nearest is not None else display_xy[1],
            z,
        )
        return DisplayProjection(
            raw_xyz=world_xyz,
            world_xyz=world_xyz,
            display_xyz=display_xyz,
            nearest_node_id=nearest.node_id if nearest else None,
            distance_to_polygon_m=distance_to_polygon,
            distance_to_nearest_node_m=distance_to_node,
            source=source,
        )

    def _tile_nodes_to_map_nodes(
        self,
        graph: _TileGraph,
        floor_z: float,
    ) -> list[MapNodeVO]:
        nodes: list[MapNodeVO] = []
        for index, (x, y) in enumerate(graph.nodes):
            nodes.append(
                MapNodeVO(
                    node_id=uuid5(self._build_job_id, f"display-grid-node:{index}"),
                    scan_id=self._scan_id,
                    build_job_id=self._build_job_id,
                    node_type=NodeType.CORRIDOR,
                    x=float(x),
                    y=float(y),
                    z=float(floor_z),
                    source_ref={
                        "graph_source": "display_navigation_grid",
                        "role": "tile",
                        "tile_index": index,
                    },
                )
            )
        return nodes

    def _tile_edges_to_map_edges(
        self,
        graph: _TileGraph,
        nodes: list[MapNodeVO],
        floor_z: float,
    ) -> list[MapEdgeVO]:
        edges: list[MapEdgeVO] = []
        for edge_index, (a, b) in enumerate(graph.edges):
            first = nodes[a]
            second = nodes[b]
            length = _distance_2d((first.x, first.y), (second.x, second.y))
            if length <= self._params.min_edge_length_m:
                continue
            edges.append(
                MapEdgeVO(
                    edge_id=uuid5(
                        self._build_job_id,
                        f"display-grid-edge:{edge_index}:{a}:{b}",
                    ),
                    scan_id=self._scan_id,
                    build_job_id=self._build_job_id,
                    from_node_id=first.node_id,
                    to_node_id=second.node_id,
                    edge_type=EdgeType.SKELETON,
                    polyline=[
                        (float(first.x), float(first.y), float(floor_z)),
                        (float(second.x), float(second.y), float(floor_z)),
                    ],
                    length_m=length,
                )
            )
        return edges

    def _build_poi_nodes_and_edges(
        self,
        *,
        pois: list[POIMarkRow],
        footprint: Polygon | MultiPolygon,
        tile_nodes: list[MapNodeVO],
        floor_z: float,
        poi_world_poses: dict[int, tuple[float, float, float]],
        poi_metadata: dict[int, dict[str, object]],
    ) -> tuple[list[MapNodeVO], list[MapEdgeVO]]:
        poi_nodes: list[MapNodeVO] = []
        poi_edges: list[MapEdgeVO] = []
        transform_high = (
            self._transform is not None and self._transform.confidence == "high"
        )
        attach_radius = (
            self._params.poi_attach_radius_m
            if self._params.poi_attach_radius_m is not None
            else max(self._params.cell_m * 1.8, 0.75)
        )

        for poi in pois:
            raw_xyz = (float(poi.tx), float(poi.ty), float(poi.tz))
            if transform_high:
                assert self._transform is not None
                world_xyz = self._transform.apply(raw_xyz)
                position_source = "arkit_to_rtabmap_transform"
            else:
                world_xyz = raw_xyz
                position_source = "polygon_projection_fallback"

            display_xy, projection_source, distance_to_polygon = (
                _project_xy_into_polygon((world_xyz[0], world_xyz[1]), footprint)
            )
            attach_nodes = _nearest_attach_nodes(
                display_xy,
                tile_nodes,
                footprint,
                max_count=self._params.poi_attach_k,
                max_radius_m=attach_radius,
            )
            if attach_nodes:
                # If the projection landed on a fragile boundary, use the nearest
                # reachable tile coordinate as the final display coordinate.
                nearest = attach_nodes[0]
                if not footprint.buffer(1e-6).covers(
                    LineString([display_xy, (nearest.x, nearest.y)])
                ):
                    display_xy = (nearest.x, nearest.y)
                    projection_source = "nearest_reachable_tile"

            display_xyz = (display_xy[0], display_xy[1], float(floor_z))
            poi_world_poses[int(poi.id)] = (
                float(world_xyz[0]),
                float(world_xyz[1]),
                float(world_xyz[2]),
            )

            facility_type = _infer_facility_type(poi.label)
            connector_key = make_connector_key(poi.label, facility_type)
            poi_node = MapNodeVO(
                node_id=uuid5(self._build_job_id, f"display-poi-node:{poi.id}"),
                scan_id=self._scan_id,
                build_job_id=self._build_job_id,
                node_type=NodeType.POI,
                x=float(display_xyz[0]),
                y=float(display_xyz[1]),
                z=float(display_xyz[2]),
                label=poi.label,
                poi_mark_id=poi.id,
                source_ref={
                    "graph_source": "display_navigation_grid",
                    "role": "semantic_destination",
                    "facility_type": facility_type,
                    "connector_key": connector_key,
                    "position_source": position_source,
                    "projection_source": projection_source,
                    "raw_arkit_xyz": [float(v) for v in raw_xyz],
                    "world_xyz": [float(v) for v in world_xyz],
                    "display_xyz": [float(v) for v in display_xyz],
                    "distance_to_polygon_m": float(distance_to_polygon),
                    "transform_confidence": (
                        self._transform.confidence
                        if self._transform is not None
                        else "no_transform"
                    ),
                    "poi_source": poi.source.value,
                },
            )
            poi_nodes.append(poi_node)

            attach_ids: list[str] = []
            for attach_index, tile in enumerate(attach_nodes):
                length = _distance_2d((poi_node.x, poi_node.y), (tile.x, tile.y))
                if length <= self._params.min_edge_length_m:
                    continue
                attach_ids.append(str(tile.node_id))
                poi_edges.append(
                    MapEdgeVO(
                        edge_id=uuid5(
                            self._build_job_id,
                            f"display-poi-spur:{poi.id}:{attach_index}:{tile.node_id}",
                        ),
                        scan_id=self._scan_id,
                        build_job_id=self._build_job_id,
                        from_node_id=poi_node.node_id,
                        to_node_id=tile.node_id,
                        edge_type=EdgeType.POI_SPUR,
                        polyline=[
                            (poi_node.x, poi_node.y, poi_node.z),
                            (tile.x, tile.y, tile.z),
                        ],
                        length_m=length,
                    )
                )

            poi_metadata[int(poi.id)] = {
                "poi_position_source": position_source,
                "projection_source": projection_source,
                "facility_type": facility_type,
                "connector_key": connector_key,
                "raw_arkit_xyz": [float(v) for v in raw_xyz],
                "final_world_xyz": [float(v) for v in world_xyz],
                "display_xyz": [float(v) for v in display_xyz],
                "distance_to_polygon_m": float(distance_to_polygon),
                "attached_tile_count": len(attach_ids),
                "attached_tile_node_ids": attach_ids,
                "transform_confidence": (
                    self._transform.confidence
                    if self._transform is not None
                    else "no_transform"
                ),
            }

        return poi_nodes, poi_edges


def _build_connected_aligned_tile_graph(
    footprint: Polygon | MultiPolygon,
    *,
    dominant_angle_deg: float,
    params: DisplayNavigationGridParams,
) -> tuple[_TileGraph, float, list[dict[str, object]]]:
    best_graph: _TileGraph | None = None
    best_clearance = params.clearance_m
    attempts: list[dict[str, object]] = []
    for clearance_m in _clearance_candidates(params.clearance_m):
        graph = _build_aligned_tile_graph(
            footprint,
            dominant_angle_deg=dominant_angle_deg,
            params=params,
            clearance_m=clearance_m,
        )
        components = _component_count(graph)
        attempts.append(
            {
                "clearance_m": float(clearance_m),
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
                "components": components,
            }
        )
        if best_graph is None or _graph_score(graph) > _graph_score(best_graph):
            best_graph = graph
            best_clearance = clearance_m
        if graph.nodes and components <= 1:
            return graph, clearance_m, attempts
    assert best_graph is not None
    return best_graph, best_clearance, attempts


def _bridge_nearby_polygon_components(
    footprint: Polygon | MultiPolygon,
    *,
    params: DisplayNavigationGridParams,
) -> tuple[Polygon | MultiPolygon, dict[str, object]]:
    """Bridge small segmentation gaps between nearby display footprint parts.

    Floor segmentation can leave a narrow unobserved gap in an otherwise single
    corridor. For the user-facing 2D map, a tiny bridge is preferable to a graph
    that cannot route across a visibly continuous hallway. The bridge is capped
    by a short distance gate and is only used for polygon components that are
    already very close.
    """
    polygons = _polygon_parts(footprint)
    if (
        not params.component_bridge_enabled
        or len(polygons) <= 1
        or params.component_bridge_max_gap_m <= 0
    ):
        return footprint, {
            "enabled": bool(params.component_bridge_enabled),
            "components_before": len(polygons),
            "bridges_added": 0,
            "components_after": len(polygons),
            "max_gap_m": float(params.component_bridge_max_gap_m),
        }

    width = (
        params.component_bridge_width_m
        if params.component_bridge_width_m is not None
        else max(params.cell_m, params.clearance_m * 2.0, 0.60)
    )
    width = max(width, params.cell_m)
    connected = polygons[0]
    remaining = polygons[1:]
    bridges: list[dict[str, object]] = []

    while remaining:
        best_index = -1
        best_gap = float("inf")
        best_points: tuple[Point, Point] | None = None
        for index, candidate in enumerate(remaining):
            first, second = nearest_points(connected, candidate)
            gap = float(first.distance(second))
            if gap < best_gap:
                best_index = index
                best_gap = gap
                best_points = (first, second)

        if (
            best_index < 0
            or best_points is None
            or best_gap > params.component_bridge_max_gap_m
        ):
            break

        first, second = best_points
        bridge_line = LineString([(first.x, first.y), (second.x, second.y)])
        if bridge_line.length <= 1e-9:
            bridge = bridge_line.buffer(width * 0.5)
        else:
            bridge = bridge_line.buffer(
                width * 0.5,
                cap_style=2,
                join_style=2,
            )
        candidate = remaining.pop(best_index)
        connected = _polygonal_part(unary_union([connected, candidate, bridge]))
        bridges.append(
            {
                "gap_m": best_gap,
                "width_m": float(width),
                "from_xy": [float(first.x), float(first.y)],
                "to_xy": [float(second.x), float(second.y)],
            }
        )

    if remaining:
        connected = _polygonal_part(unary_union([connected, *remaining]))

    return connected, {
        "enabled": True,
        "components_before": len(polygons),
        "bridges_added": len(bridges),
        "components_after": _polygon_component_count(connected),
        "max_gap_m": float(params.component_bridge_max_gap_m),
        "bridges": bridges,
    }


def _build_aligned_tile_graph(
    footprint: Polygon | MultiPolygon,
    *,
    dominant_angle_deg: float,
    params: DisplayNavigationGridParams,
    clearance_m: float,
) -> _TileGraph:
    if params.connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if params.cell_m <= 0:
        raise ValueError("cell_m must be > 0")
    if clearance_m < 0:
        raise ValueError("clearance_m must be >= 0")

    rotated_footprint = rotate(
        footprint,
        -dominant_angle_deg,
        origin=(0.0, 0.0),
        use_radians=False,
    ).buffer(0)
    safe = (
        rotated_footprint.buffer(-clearance_m)
        if clearance_m > 0
        else rotated_footprint
    )
    safe = _polygonal_part(safe)
    if safe.is_empty:
        safe = _polygonal_part(rotated_footprint)

    xmin, ymin, xmax, ymax = rotated_footprint.bounds
    col_min = math.floor(xmin / params.cell_m) - 1
    col_max = math.ceil(xmax / params.cell_m) + 1
    row_min = math.floor(ymin / params.cell_m) - 1
    row_max = math.ceil(ymax / params.cell_m) + 1

    graph = _TileGraph()
    for row in range(row_min, row_max + 1):
        y = (row + 0.5) * params.cell_m
        for col in range(col_min, col_max + 1):
            x = (col + 0.5) * params.cell_m
            if safe.covers(Point(x, y)):
                node_id = len(graph.nodes)
                graph.grid_to_node[(row, col)] = node_id
                original = rotate(
                    Point(x, y),
                    dominant_angle_deg,
                    origin=(0.0, 0.0),
                    use_radians=False,
                )
                graph.nodes.append((float(original.x), float(original.y)))

    neighbor_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if params.connectivity == 8:
        neighbor_offsets.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])

    edge_set: set[tuple[int, int]] = set()
    candidate_pairs: set[tuple[int, int]] = set()
    inside_guard = footprint.buffer(1e-6)
    for (row, col), node_id in graph.grid_to_node.items():
        x, y = graph.nodes[node_id]
        for dr, dc in neighbor_offsets:
            other_id = graph.grid_to_node.get((row + dr, col + dc))
            if other_id is None:
                continue
            a, b = sorted((node_id, other_id))
            if (a, b) in candidate_pairs:
                continue
            candidate_pairs.add((a, b))
            ox, oy = graph.nodes[other_id]
            if not inside_guard.covers(LineString([(x, y), (ox, oy)])):
                graph.blocked_neighbor_pairs_outside_polygon += 1
                continue
            edge_set.add((a, b))

    graph.candidate_neighbor_pairs = len(candidate_pairs)
    graph.edges = sorted(edge_set)
    graph.adjacency = {idx: [] for idx in range(len(graph.nodes))}
    for a, b in graph.edges:
        graph.adjacency[a].append(b)
        graph.adjacency[b].append(a)
    for neighbors in graph.adjacency.values():
        neighbors.sort()
    return graph


def _clearance_candidates(requested: float) -> list[float]:
    values = [requested, requested * 0.75, requested * 0.5, requested * 0.25, 0.0]
    out: list[float] = []
    for value in values:
        candidate = round(max(0.0, value), 6)
        if candidate not in out:
            out.append(candidate)
    return out


def _component_count(graph: _TileGraph) -> int:
    if not graph.nodes:
        return 0
    remaining = set(range(len(graph.nodes)))
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            current = stack.pop()
            for neighbor in graph.adjacency.get(current, []):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return count


def _polygon_parts(geom: Polygon | MultiPolygon) -> list[Polygon]:
    if isinstance(geom, Polygon):
        return [geom]
    return sorted(
        [part for part in geom.geoms if isinstance(part, Polygon) and not part.is_empty],
        key=lambda part: part.area,
        reverse=True,
    )


def _polygon_component_count(geom: Polygon | MultiPolygon) -> int:
    return len(_polygon_parts(geom))


def _graph_score(graph: _TileGraph) -> tuple[int, int, int]:
    connected = 1 if graph.nodes and _component_count(graph) <= 1 else 0
    return connected, len(graph.nodes), len(graph.edges)


def _load_polygonal_footprint(data: dict[str, object]) -> Polygon | MultiPolygon:
    if data.get("type") == "FeatureCollection":
        features_obj = data.get("features")
        features = features_obj if isinstance(features_obj, list) else []
        geoms = [
            _polygonal_part(shape(feature["geometry"]))
            for feature in features
            if isinstance(feature, dict) and feature.get("geometry") is not None
        ]
        return _polygonal_part(unary_union([geom for geom in geoms if not geom.is_empty]))
    if data.get("type") == "Feature":
        geom_data = data.get("geometry")
        if not isinstance(geom_data, dict):
            return Polygon()
        return _polygonal_part(shape(geom_data))
    return _polygonal_part(shape(data))


def _polygonal_part(geom: object) -> Polygon | MultiPolygon:
    if isinstance(geom, Polygon):
        return geom.buffer(0)
    if isinstance(geom, MultiPolygon):
        return geom.buffer(0)
    if isinstance(geom, GeometryCollection):
        polys = [
            part
            for part in geom.geoms
            if isinstance(part, Polygon | MultiPolygon) and not part.is_empty
        ]
        if not polys:
            return Polygon()
        return _polygonal_part(unary_union(polys))
    buffer_func = getattr(geom, "buffer", None)
    if callable(buffer_func):
        try:
            cleaned = buffer_func(0)
        except Exception:
            return Polygon()
        if isinstance(cleaned, Polygon | MultiPolygon):
            return cleaned
    return Polygon()


def _dominant_axis_angle_deg(geom: Polygon | MultiPolygon) -> float:
    vectors: list[tuple[float, float]] = []
    polygons = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
    for polygon in polygons:
        rings = [polygon.exterior, *polygon.interiors]
        for ring in rings:
            coords = list(ring.coords)
            for first, second in zip(coords, coords[1:], strict=False):
                dx = float(second[0] - first[0])
                dy = float(second[1] - first[1])
                length = math.hypot(dx, dy)
                if length < 0.20:
                    continue
                angle = math.atan2(dy, dx)
                folded = ((angle + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0
                vectors.append((folded, length))
    if not vectors:
        return 0.0
    sum_sin = sum(math.sin(2.0 * angle) * length for angle, length in vectors)
    sum_cos = sum(math.cos(2.0 * angle) * length for angle, length in vectors)
    theta = 0.5 * math.atan2(sum_sin, sum_cos)
    return float(math.degrees(theta))


def _project_xy_into_polygon(
    xy: Point2D,
    geom: Polygon | MultiPolygon,
) -> tuple[Point2D, str, float]:
    point = Point(xy)
    guard = geom.buffer(1e-6)
    if guard.covers(point):
        return (float(xy[0]), float(xy[1])), "inside_polygon", 0.0
    if geom.is_empty:
        return (float(xy[0]), float(xy[1])), "empty_polygon_raw", 0.0
    projected, _ = nearest_points(geom, point)
    distance = float(point.distance(projected))
    return (float(projected.x), float(projected.y)), "nearest_polygon_point", distance


def _nearest_corridor_node(
    xy: Point2D,
    nodes: list[MapNodeVO],
) -> MapNodeVO | None:
    candidates = [node for node in nodes if node.node_type == NodeType.CORRIDOR]
    if not candidates:
        return None
    return min(candidates, key=lambda node: _distance_2d(xy, (node.x, node.y)))


def _nearest_attach_nodes(
    xy: Point2D,
    nodes: list[MapNodeVO],
    footprint: Polygon | MultiPolygon,
    *,
    max_count: int,
    max_radius_m: float,
) -> list[MapNodeVO]:
    if max_count <= 0:
        return []
    sorted_nodes = sorted(
        (node for node in nodes if node.node_type == NodeType.CORRIDOR),
        key=lambda node: _distance_2d(xy, (node.x, node.y)),
    )
    guard = footprint.buffer(1e-6)
    selected: list[MapNodeVO] = []
    for node in sorted_nodes:
        dist = _distance_2d(xy, (node.x, node.y))
        if selected and dist > max_radius_m:
            break
        if not guard.covers(LineString([xy, (node.x, node.y)])):
            continue
        selected.append(node)
        if len(selected) >= max_count:
            break
    if not selected and sorted_nodes:
        selected.append(sorted_nodes[0])
    return selected


def _infer_facility_type(label: str | None) -> str:
    if label is None:
        return "poi"
    value = label.strip().lower()
    if value.startswith("stair") or "계단" in value:
        return "stair"
    if value.startswith("elev") or "엘리베이터" in value or "엘베" in value:
        return "elevator"
    return "poi"


def _distance_2d(first: Point2D, second: Point2D) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])
