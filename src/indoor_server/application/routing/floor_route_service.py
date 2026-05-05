"""Floor-scoped routing service — corridor-only Dijkstra + leaf edge 허용.

Sprint 78 A-7: GET /api/v1/floors/{floorId}/route?from=&to= 구현.

알고리즘:
1. map_node / map_edge 로드 (non-stale, same scan).
2. networkx 그래프 구성 (전체).
3. start/end kind 판정.
4. POI/passage → corridor subgraph에서 Dijkstra 후 leaf edge prepend/append.
5. POI/passage가 corridor subgraph에서 중간 경유점이 되지 않도록 corridor-only subgraph 사용.
"""
from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

import networkx as nx
from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.infrastructure.db.repositories.map_graph_repo import MapGraphRepository

logger = logging.getLogger(__name__)

# kind 판별
_POI_KINDS = {"POI", "poi"}
_PASSAGE_KINDS = {"PASSAGE", "passage"}  # vertical_connector_stop role → passage kind
_LEAF_KINDS = {"POI", "POI_ATTACH", "poi", "passage"}


def _is_leaf_node_type(node_type: str | None, source_ref: dict[str, Any] | None) -> bool:
    """poi 또는 passage (vertical_connector_stop) 노드이면 True."""
    if node_type and node_type.upper() in ("POI",):
        return True
    if isinstance(source_ref, dict) and source_ref.get("role") == "vertical_connector_stop":
        return True
    return False


class FloorRouteError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class FloorRouteService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def find_route(
        self,
        floor_id: UUID,
        from_node_id: UUID,
        to_node_id: UUID,
    ) -> dict[str, Any]:
        """corridor-only Dijkstra + leaf edge 허용 라우팅."""
        # 1. active scan 조회
        from indoor_server.application.api_v1.building_floor_service import BuildingFloorService
        from indoor_server.domain.building.enums import BuildState
        from indoor_server.infrastructure.db.repositories.build_job_repo import BuildJobRepository

        svc = BuildingFloorService(self._session)
        active = await svc.get_active_scan_for_floor(floor_id)
        if active is None:
            raise FloorRouteError("FLOOR_NOT_FOUND", "active scan not found", 404)

        build_job = await BuildJobRepository(self._session).get_latest(active.scan_id)
        if build_job is None or build_job.state != BuildState.SUCCEEDED:
            raise FloorRouteError("FLOOR_NOT_FOUND", "no succeeded build for floor", 404)

        # 2. map_node / map_edge 로드
        nodes_rows, edges_rows, _ = await MapGraphRepository(self._session).load_graph_for_routing(
            active.scan_id
        )

        # node dict
        node_by_id: dict[str, Any] = {}
        for n in nodes_rows:
            node_by_id[str(n.node_id)] = n

        from_str = str(from_node_id)
        to_str = str(to_node_id)

        if from_str not in node_by_id:
            raise FloorRouteError("NODE_NOT_FOUND", f"from node {from_str} not found", 404)
        if to_str not in node_by_id:
            raise FloorRouteError("NODE_NOT_FOUND", f"to node {to_str} not found", 404)

        # 3. networkx 그래프 구성
        graph: nx.Graph = nx.Graph()
        for n in nodes_rows:
            nid = str(n.node_id)
            graph.add_node(
                nid,
                node_type=n.node_type,
                source_ref=n.source_ref,
                x=n.x,
                y=n.y,
                label=n.label,
            )

        edge_map: dict[tuple[str, str], str] = {}  # (from, to) → edge_id
        for e in edges_rows:
            fid = str(e.from_node_id)
            tid = str(e.to_node_id)
            eid = str(e.edge_id)
            weight = float(e.length_m) if e.length_m else 1.0
            graph.add_edge(fid, tid, weight=weight, edge_id=eid)
            edge_map[(fid, tid)] = eid
            edge_map[(tid, fid)] = eid

        # 4. leaf node 판별
        from_node_data = graph.nodes[from_str]
        to_node_data = graph.nodes[to_str]
        from_is_leaf = _is_leaf_node_type(
            from_node_data.get("node_type"),
            from_node_data.get("source_ref"),
        )
        to_is_leaf = _is_leaf_node_type(
            to_node_data.get("node_type"),
            to_node_data.get("source_ref"),
        )

        # 5. corridor-only subgraph
        corridor_nodes = {
            nid for nid, data in graph.nodes(data=True)
            if not _is_leaf_node_type(data.get("node_type"), data.get("source_ref"))
        }
        graph_corridor = graph.subgraph(corridor_nodes)

        # leaf → corridor neighbor (spur edge 1개 기준)
        def get_corridor_neighbor(leaf_id: str) -> str | None:
            for nbr in graph.neighbors(leaf_id):
                if nbr in corridor_nodes:
                    return str(nbr)
            return None

        # 실제 corridor start/end 결정
        if from_is_leaf:
            start_corridor = get_corridor_neighbor(from_str)
            if start_corridor is None:
                raise FloorRouteError("NO_PATH", "from leaf node has no corridor neighbor", 422)
        else:
            start_corridor = from_str

        if to_is_leaf:
            end_corridor = get_corridor_neighbor(to_str)
            if end_corridor is None:
                raise FloorRouteError("NO_PATH", "to leaf node has no corridor neighbor", 422)
        else:
            end_corridor = to_str

        # 6. corridor-only Dijkstra
        if start_corridor == end_corridor:
            mid_path: list[str] = [start_corridor]
        else:
            if not nx.has_path(graph_corridor, start_corridor, end_corridor):
                raise FloorRouteError("NO_PATH", "no corridor path between nodes", 422)
            try:
                mid_path = nx.shortest_path(
                    graph_corridor, start_corridor, end_corridor, weight="weight"
                )
            except nx.NetworkXNoPath as exc:
                raise FloorRouteError("NO_PATH", "no corridor path between nodes", 422) from exc

        # 7. 최종 path 구성 (leaf 포함)
        final_nodes: list[str] = []
        if from_is_leaf:
            final_nodes.append(from_str)
        final_nodes.extend(mid_path)
        if to_is_leaf and to_str not in final_nodes:
            final_nodes.append(to_str)

        # 8. edge 목록 구성
        final_edges: list[str] = []
        for i in range(len(final_nodes) - 1):
            a, b = final_nodes[i], final_nodes[i + 1]
            eid_val = edge_map.get((a, b)) or edge_map.get((b, a))
            if eid_val:
                final_edges.append(eid_val)

        # 9. total length 계산
        total_length_m = 0.0
        for i in range(len(final_nodes) - 1):
            a, b = final_nodes[i], final_nodes[i + 1]
            data = graph.get_edge_data(a, b)
            if data:
                total_length_m += float(data.get("weight", 0.0))

        logger.info(
            "floor_route: floor_id=%s from=%s to=%s nodes=%d length=%.2fm",
            floor_id, from_node_id, to_node_id, len(final_nodes), total_length_m,
        )

        return {
            "floorId": str(floor_id),
            "from": from_str,
            "to": to_str,
            "nodes": final_nodes,
            "edges": final_edges,
            "totalLengthM": round(total_length_m, 3),
            "nodeCount": len(final_nodes),
        }
