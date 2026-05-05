"""DisplayNavigationGridStep 단위 테스트 (Sprint 61).

목표:
- footprint 내부 dense tile graph를 만든다.
- POI/계단/엘리베이터는 polygon 내부에 배치되고 주변 tile과 연결된다.
- VPS/world 좌표는 2D display graph reachable node로 변환된다.
"""
from __future__ import annotations

import struct
from uuid import uuid4

import numpy as np
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, mapping

from indoor_server.application.building.arkit_to_rtabmap_transform import (
    ArkitToRtabmapTransform,
)
from indoor_server.application.building.steps.display_navigation_grid import (
    DisplayNavigationGridParams,
    DisplayNavigationGridStep,
)
from indoor_server.application.routing.snap import snap_coordinate_to_node
from indoor_server.domain.building.enums import EdgeType, NodeType
from indoor_server.domain.poi.enums import POISource
from indoor_server.domain.routing.models import MapNodeRow
from indoor_server.domain.scan.models import POIMarkRow


def _l_polygon() -> Polygon:
    return Polygon(
        [
            (0.0, 0.0),
            (7.0, 0.0),
            (7.0, 2.0),
            (3.0, 2.0),
            (3.0, 7.0),
            (0.0, 7.0),
            (0.0, 0.0),
        ]
    )


def _poi(
    poi_id: int,
    *,
    label: str,
    x: float,
    y: float,
    z: float = 0.0,
) -> POIMarkRow:
    identity = struct.pack("<16f", *[1.0 if i % 5 == 0 else 0.0 for i in range(16)])
    return POIMarkRow(
        id=poi_id,
        scan_id=str(uuid4()),
        keyframe_seq=1,
        created_at=0,
        pose_matrix=identity,
        tx=x,
        ty=y,
        tz=z,
        track_id=None,
        label=label,
        source=POISource.MANUAL,
    )


def _step(scan_id: object, build_job_id: object) -> DisplayNavigationGridStep:
    return DisplayNavigationGridStep(
        scan_id=scan_id,  # type: ignore[arg-type]
        build_job_id=build_job_id,  # type: ignore[arg-type]
        params=DisplayNavigationGridParams(
            cell_m=0.5,
            clearance_m=0.0,
            connectivity=8,
            poi_attach_k=4,
        ),
    )


def test_dense_tile_nodes_and_edges_are_inside_polygon() -> None:
    scan_id = uuid4()
    build_job_id = uuid4()
    poly = _l_polygon()

    result = _step(scan_id, build_job_id).run(
        footprint_geojson=mapping(poly),
        floor_z=0.0,
        pois=[],
    )

    guard = poly.buffer(1e-6)
    corridor_nodes = [n for n in result.nodes if n.node_type == NodeType.CORRIDOR]
    skeleton_edges = [e for e in result.edges if e.edge_type == EdgeType.SKELETON]

    assert corridor_nodes
    assert skeleton_edges
    assert result.metadata["valid_neighbor_pair_connection_ratio"] == pytest.approx(1.0)
    assert all(guard.covers(Point(n.x, n.y)) for n in corridor_nodes)
    assert all(
        guard.covers(
            LineString(
                [
                    (e.polyline[0][0], e.polyline[0][1]),
                    (e.polyline[1][0], e.polyline[1][1]),
                ]
            )
        )
        for e in skeleton_edges
    )


def test_nearby_polygon_components_are_bridged_for_display_routing() -> None:
    scan_id = uuid4()
    build_job_id = uuid4()
    # Real scan floor segmentation can leave a short unobserved slit between
    # hallway pieces. The user-facing graph should bridge only that small gap.
    poly = MultiPolygon(
        [
            Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]),
            Polygon([(0.0, 2.8), (2.0, 2.8), (2.0, 4.0), (0.0, 4.0)]),
        ]
    )

    result = _step(scan_id, build_job_id).run(
        footprint_geojson=mapping(poly),
        floor_z=0.0,
        pois=[],
    )

    bridge_meta = result.metadata["polygon_component_bridge"]
    assert result.metadata["raw_polygon_component_count"] == 2
    assert bridge_meta["bridges_added"] == 1
    assert bridge_meta["components_after"] == 1
    assert result.metadata["connected_component_count"] == 1
    assert result.footprint_geojson["type"] == "Polygon"


def test_poi_stair_elevator_nodes_project_inside_and_attach_to_tiles() -> None:
    scan_id = uuid4()
    build_job_id = uuid4()
    poly = _l_polygon()
    pois = [
        _poi(1, label="302호 강의실", x=1.0, y=6.5),
        _poi(2, label="STAIR_A 동쪽 계단", x=6.7, y=0.5),
        _poi(3, label="ELEV_CENTER 중앙 엘리베이터", x=2.8, y=2.0),
    ]

    result = _step(scan_id, build_job_id).run(
        footprint_geojson=mapping(poly),
        floor_z=0.0,
        pois=pois,
    )

    guard = poly.buffer(1e-6)
    poi_nodes = [n for n in result.nodes if n.node_type == NodeType.POI]
    poi_edges = [e for e in result.edges if e.edge_type == EdgeType.POI_SPUR]
    facility_by_id = {
        n.poi_mark_id: (n.source_ref or {}).get("facility_type")
        for n in poi_nodes
    }
    connector_by_id = {
        n.poi_mark_id: (n.source_ref or {}).get("connector_key")
        for n in poi_nodes
    }

    assert len(poi_nodes) == 3
    assert len(poi_edges) >= 3
    assert facility_by_id == {1: "poi", 2: "stair", 3: "elevator"}
    assert connector_by_id == {1: None, 2: "stair:a", 3: "elevator:center"}
    assert all(guard.covers(Point(n.x, n.y)) for n in poi_nodes)
    assert all(
        result.poi_position_metadata[int(n.poi_mark_id or 0)]["attached_tile_count"] >= 1
        for n in poi_nodes
    )


def test_high_confidence_transform_places_poi_in_rtabmap_world_frame() -> None:
    scan_id = uuid4()
    build_job_id = uuid4()
    poly = _l_polygon()
    transform = ArkitToRtabmapTransform(
        rotation=np.eye(3),
        translation=np.array([1.0, 2.0, 0.0]),
        pair_count=8,
        residual_rms_m=0.02,
        confidence="high",
    )
    step = DisplayNavigationGridStep(
        scan_id=scan_id,
        build_job_id=build_job_id,
        params=DisplayNavigationGridParams(cell_m=0.5, clearance_m=0.0),
        arkit_to_rtabmap_transform=transform,
    )

    result = step.run(
        footprint_geojson=mapping(poly),
        floor_z=0.0,
        pois=[_poi(7, label="301호 강의실", x=0.5, y=0.5)],
    )

    assert result.poi_world_poses[7] == pytest.approx((1.5, 2.5, 0.0))
    assert result.poi_position_metadata[7]["poi_position_source"] == (
        "arkit_to_rtabmap_transform"
    )


def test_world_coordinate_projects_to_nearest_reachable_display_node() -> None:
    scan_id = uuid4()
    build_job_id = uuid4()
    poly = _l_polygon()
    step = _step(scan_id, build_job_id)
    result = step.run(
        footprint_geojson=mapping(poly),
        floor_z=0.0,
        pois=[],
    )
    corridor_nodes = [n for n in result.nodes if n.node_type == NodeType.CORRIDOR]

    projection = step.project_world_to_display(
        footprint_geojson=mapping(poly),
        tile_nodes=corridor_nodes,
        world_xyz=(9.0, 1.0, 0.0),
    )

    assert projection.nearest_node_id is not None
    assert poly.buffer(1e-6).covers(Point(projection.display_xyz[0], projection.display_xyz[1]))
    assert projection.distance_to_polygon_m > 0


def test_coordinate_snap_prefers_corridor_over_nearby_poi() -> None:
    corridor_id = uuid4()
    poi_id = uuid4()
    node_lookup = {
        poi_id: MapNodeRow(
            node_id=poi_id,
            x=0.0,
            y=0.0,
            z=0.0,
            node_type="poi",
            label="301호 강의실",
            poi_mark_id=1,
        ),
        corridor_id: MapNodeRow(
            node_id=corridor_id,
            x=0.2,
            y=0.0,
            z=0.0,
            node_type="corridor",
            label=None,
            poi_mark_id=None,
            source_ref={"graph_source": "display_navigation_grid"},
        ),
    }

    snapped, _ = snap_coordinate_to_node((0.0, 0.0, 0.0), node_lookup, 5.0)

    assert snapped == corridor_id
