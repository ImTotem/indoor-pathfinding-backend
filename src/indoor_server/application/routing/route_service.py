"""route_service.py — A* 경로 탐색 서비스."""
from __future__ import annotations

from uuid import UUID

import networkx as nx

from indoor_server.application.routing.graph_loader import (
    GraphLoader,
    heuristic_2d,
    heuristic_3d,
)
from indoor_server.application.routing.manhattan_path import (
    manhattanize_route_polyline,
    polyline_length,
)
from indoor_server.application.routing.snap import snap_coordinate_to_graph
from indoor_server.domain.routing.errors import (
    EndpointNodeNotFoundError,
    PathNotFoundError,
    POINodeNotFoundError,
)
from indoor_server.domain.routing.models import MapNodeRow, RoutePath, SnapInfo
from indoor_server.interfaces.api.schemas import RouteEndpoint


class RouteService:
    """A* 경로 탐색 서비스.

    GraphLoader → snap → astar_path → RoutePath 반환.
    """

    def __init__(
        self,
        graph_loader: GraphLoader,
        snap_threshold_m: float = 5.0,
        heuristic_use_z: bool = True,
    ) -> None:
        self._loader = graph_loader
        self._snap_threshold_m = snap_threshold_m
        self._heuristic_use_z = heuristic_use_z

    async def compute(
        self,
        *,
        scan_id: str,
        scan_ids: list[str] | None = None,
        merge_overlaps: bool = False,
        start: RouteEndpoint,
        goal: RouteEndpoint,
        vertical_preference: str | None = None,
    ) -> RoutePath:
        """경로 탐색 진입점.

        Raises:
            GraphNotReadyError: build 미완료.
            SnapDistanceExceededError: 좌표 snap 거리 초과.
            EndpointNodeNotFoundError: node_id 미존재.
            POINodeNotFoundError: poi_mark_id 미존재.
            PathNotFoundError: A* 경로 없음.
        """
        effective_scan_ids = _effective_scan_ids(scan_id, scan_ids)
        if len(effective_scan_ids) == 1:
            g, node_lookup, build_job_id = await self._loader.load(
                scan_id=effective_scan_ids[0]
            )
            build_job_ids = [build_job_id]
            route_metadata: dict[str, object] = {
                "route_scope": "single_scan",
                "merge_overlaps": False,
            }
        else:
            g, node_lookup, build_job_ids, merge_report = await self._loader.load_many(
                scan_ids=effective_scan_ids,
                merge_overlaps=merge_overlaps,
                vertical_preference=vertical_preference,
            )
            build_job_id = build_job_ids[0]
            route_metadata = {
                "route_scope": "multi_scan",
                "merge_overlaps": merge_overlaps,
                "scan_count": len(effective_scan_ids),
                "vertical_edge_count": int(g.graph.get("vertical_edge_count", 0)),
                "vertical_preference": vertical_preference,
                "merge_report": (
                    {
                        "fragment_count": merge_report.fragment_count,
                        "merged_node_count": merge_report.merged_node_count,
                        "merged_edge_count": merge_report.merged_edge_count,
                        "overlap_bridge_count": merge_report.overlap_bridge_count,
                        "transforms": merge_report.transforms,
                    }
                    if merge_report is not None
                    else None
                ),
            }

        start_id, start_snap_dist = self._resolve_endpoint(start, g, node_lookup, side="start")
        goal_id, goal_snap_dist = self._resolve_endpoint(goal, g, node_lookup, side="goal")

        # A* 경로 탐색
        if self._heuristic_use_z:
            _heuristic = lambda u, v: heuristic_3d(u, v, g)  # noqa: E731
        else:
            _heuristic = lambda u, v: heuristic_2d(u, v, g)  # noqa: E731
        try:
            path_ids: list[UUID] = nx.astar_path(
                g,
                start_id,
                goal_id,
                heuristic=_heuristic,
                weight="length_m",
            )
        except nx.NetworkXNoPath:
            raise PathNotFoundError(str(start_id), str(goal_id)) from None
        except nx.NodeNotFound as e:
            raise PathNotFoundError(str(start_id), str(goal_id)) from e

        # 경로 메타 구성
        nodes_in_order = [node_lookup[nid] for nid in path_ids]
        raw_polyline = [(n.x, n.y, n.z) for n in nodes_in_order]
        if _is_display_navigation_grid_path(nodes_in_order):
            # Sprint 61: dense display graph edges are already polygon-contained.
            # Applying display-time Manhattan compression here can create shortcut
            # segments outside the footprint, so keep the graph path itself.
            polyline = _dedupe_polyline(raw_polyline)
        else:
            polyline = manhattanize_route_polyline(raw_polyline)
        # Use graph edge weights as the authoritative cost. Display polyline
        # simplification can hide vertical-transition or overlap-bridge costs.
        length_m = _path_length(g, path_ids) or polyline_length(polyline)

        snap = SnapInfo(
            start_snap_distance_m=start_snap_dist,
            goal_snap_distance_m=goal_snap_dist,
        )

        return RoutePath(
            nodes_in_order=nodes_in_order,
            polyline=polyline,
            length_m=length_m,
            snap=snap,
            build_job_id=build_job_id,
            scan_ids=[UUID(value) for value in effective_scan_ids],
            build_job_ids=build_job_ids,
            route_metadata=route_metadata,
        )

    # ── private helpers ───────────────────────────────────────────────────────

    def _resolve_endpoint(
        self,
        endpoint: RouteEndpoint,
        g: nx.Graph,
        node_lookup: dict[UUID, MapNodeRow],
        side: str,
    ) -> tuple[UUID, float | None]:
        """endpoint → (node_id, snap_distance_m | None).

        coordinate: snap.
        node_id: 직접 조회.
        poi_mark_id: map_node.poi_mark_id 역탐색.

        Returns snap_distance_m=None when node_id/poi_mark_id 직접 지정.
        """
        if endpoint.coordinate is not None:
            nid, dist = snap_coordinate_to_graph(
                endpoint.coordinate,
                g,
                node_lookup,
                self._snap_threshold_m,
                side=side,
            )
            return nid, dist

        if endpoint.node_id is not None:
            nid = endpoint.node_id
            if nid not in node_lookup:
                raise EndpointNodeNotFoundError(str(nid), side)
            return nid, None

        # poi_mark_id fallback (R-4 canonical 미구현 대응)
        assert endpoint.poi_mark_id is not None
        poi_id = endpoint.poi_mark_id
        matched = [
            node for node in node_lookup.values()
            if node.poi_mark_id == poi_id
        ]
        if not matched:
            raise POINodeNotFoundError(poi_id, side)
        # 여러 개면 첫 번째 선택 (poi_canonical 미구현 단계)
        return matched[0].node_id, None


def _path_length(g: nx.Graph, path: list[UUID]) -> float:
    """path 노드 시퀀스의 총 edge length_m 합계."""
    total = 0.0
    for i in range(len(path) - 1):
        edge_data = g.get_edge_data(path[i], path[i + 1])
        total += edge_data["length_m"] if edge_data else 0.0
    return total


def _is_display_navigation_grid_path(nodes: list[MapNodeRow]) -> bool:
    return any(
        (node.source_ref or {}).get("graph_source") == "display_navigation_grid"
        for node in nodes
    )


def _dedupe_polyline(
    polyline: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    for point in polyline:
        if out and point == out[-1]:
            continue
        out.append(point)
    return out


def _effective_scan_ids(scan_id: str, scan_ids: list[str] | None) -> list[str]:
    values = scan_ids or [scan_id]
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
    if scan_id not in seen:
        out.insert(0, scan_id)
    return out
