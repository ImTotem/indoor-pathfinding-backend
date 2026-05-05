"""graph_loader.py — DB → networkx Graph 빌드."""
from __future__ import annotations

import math
from uuid import UUID

import networkx as nx
from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.application.routing.scan_graph_merge import (
    ScanGraphFragment,
    ScanGraphMerger,
    ScanMergeReport,
)
from indoor_server.application.routing.vertical_connectors import (
    VerticalEdge,
    add_vertical_edges,
)
from indoor_server.domain.building.enums import BuildState
from indoor_server.domain.routing.errors import GraphNotReadyError, ScanNotFoundError
from indoor_server.domain.routing.models import MapNodeRow
from indoor_server.infrastructure.db.repositories.build_job_repo import BuildJobRepository
from indoor_server.infrastructure.db.repositories.map_graph_repo import MapGraphRepository
from indoor_server.infrastructure.db.repositories.scan_ingest_repo import ScanIngestRepository


def _euclidean_3d(u_data: dict[str, object], v_data: dict[str, object]) -> float:
    """A* heuristic — 3D euclidean 거리."""
    ux, uy, uz = float(u_data["x"]), float(u_data["y"]), float(u_data["z"])  # type: ignore[arg-type]
    vx, vy, vz = float(v_data["x"]), float(v_data["y"]), float(v_data["z"])  # type: ignore[arg-type]
    dx, dy, dz = ux - vx, uy - vy, uz - vz
    return math.sqrt(dx * dx + dy * dy + dz * dz)


class GraphLoader:
    """DB → networkx Graph 변환기.

    매 요청마다 rebuild (Sprint 23 범위 — cache 없음).
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def load(
        self,
        *,
        scan_id: str,
    ) -> tuple[nx.Graph, dict[UUID, MapNodeRow], UUID]:
        """scan_id의 non-stale graph를 networkx로 로드.

        Returns:
            (graph, node_lookup, build_job_id)
            graph edges: weight="length_m".
            node 속성: x, y, z, node_type, label, poi_mark_id.

        Raises:
            GraphNotReadyError: build 미완료 또는 scan_id 미존재 시.
        """
        fragment = await self._load_fragment(scan_id=scan_id)
        g, node_lookup = _graph_from_fragment(fragment)
        return g, node_lookup, fragment.build_job_id

    async def load_many(
        self,
        *,
        scan_ids: list[str],
        merge_overlaps: bool = False,
    ) -> tuple[nx.Graph, dict[UUID, MapNodeRow], list[UUID], ScanMergeReport | None]:
        """Load multiple scan graphs as a route-time union graph.

        `merge_overlaps=False` is the safe default for multi-floor routing. When
        callers know the scans are overlapping same-floor RTABMap scans, setting
        it to true estimates route-time overlap bridges.
        """
        if not scan_ids:
            raise ScanNotFoundError("")
        fragments = [
            await self._load_fragment(scan_id=scan_id)
            for scan_id in _dedupe_scan_ids(scan_ids)
        ]
        if len(fragments) == 1:
            g, node_lookup = _graph_from_fragment(fragments[0])
            return g, node_lookup, [fragments[0].build_job_id], None

        merger = ScanGraphMerger()
        merge_result = merger.merge(fragments, merge_overlaps=merge_overlaps)
        graph_repo = MapGraphRepository(self._session)
        explicit_rows = await graph_repo.load_explicit_vertical_edges(
            [fragment.scan_id for fragment in fragments]
        )
        explicit_edges = [
            VerticalEdge(
                from_node_id=from_node_id,
                to_node_id=to_node_id,
                length_m=length_m,
                source_ref=source_ref,
            )
            for from_node_id, to_node_id, length_m, source_ref in explicit_rows
        ]
        vertical_edge_count = add_vertical_edges(
            merge_result.graph,
            merge_result.node_lookup,
            explicit_edges=explicit_edges,
        )
        merge_result.graph.graph["vertical_edge_count"] = vertical_edge_count
        return (
            merge_result.graph,
            merge_result.node_lookup,
            [fragment.build_job_id for fragment in fragments],
            merge_result.report,
        )

    async def _load_fragment(
        self,
        *,
        scan_id: str,
    ) -> ScanGraphFragment:
        # scan 존재 확인
        scan_repo = ScanIngestRepository(self._session)
        existing = await scan_repo.find_existing(scan_id)
        if existing is None:
            raise ScanNotFoundError(scan_id)

        # build 완료 확인
        build_repo = BuildJobRepository(self._session)
        latest_job = await build_repo.get_latest(scan_id)
        if latest_job is None or latest_job.state != BuildState.SUCCEEDED:
            state_str = latest_job.state.value if latest_job else "not_started"
            job_id_str = str(latest_job.build_job_id) if latest_job else None
            raise GraphNotReadyError(scan_id, state_str, job_id_str)

        # nodes + edges 로드
        graph_repo = MapGraphRepository(self._session)
        nodes, edges, build_job_id = await graph_repo.load_graph_for_routing(scan_id)
        result_job_id = build_job_id if build_job_id is not None else latest_job.build_job_id

        return ScanGraphFragment(
            scan_id=scan_id,
            build_job_id=result_job_id,
            nodes=nodes,
            edges=edges,
        )


def heuristic_3d(u: UUID, v: UUID, g: nx.Graph) -> float:
    """A* 용 3D euclidean heuristic 팩토리.

    networkx.astar_path heuristic 시그니처: heuristic(u, v) → float.
    g를 클로저로 캡처.
    """
    u_data = g.nodes[u]
    v_data = g.nodes[v]
    return _euclidean_3d(u_data, v_data)


def heuristic_2d(u: UUID, v: UUID, g: nx.Graph) -> float:
    """A* 용 2D (xy 평면) euclidean heuristic.

    z 좌표를 무시하므로 단층 그래프에서 heuristic admissibility가 더 강하다.
    """
    u_data = g.nodes[u]
    v_data = g.nodes[v]
    ux, uy = float(u_data["x"]), float(u_data["y"])
    vx, vy = float(v_data["x"]), float(v_data["y"])
    dx, dy = ux - vx, uy - vy
    return math.sqrt(dx * dx + dy * dy)


def _add_node(
    g: nx.Graph,
    node_lookup: dict[UUID, MapNodeRow],
    node: MapNodeRow,
) -> None:
    g.add_node(
        node.node_id,
        x=node.x,
        y=node.y,
        z=node.z,
        node_type=node.node_type,
        label=node.label,
        poi_mark_id=node.poi_mark_id,
        source_ref=node.source_ref,
        scan_id=str(node.scan_id) if node.scan_id is not None else None,
        level_id=node.level_id,
    )
    node_lookup[node.node_id] = node


def _graph_from_fragment(
    fragment: ScanGraphFragment,
) -> tuple[nx.Graph, dict[UUID, MapNodeRow]]:
    g: nx.Graph = nx.Graph()
    node_lookup: dict[UUID, MapNodeRow] = {}

    for node in fragment.nodes:
        _add_node(g, node_lookup, node)

    for edge in fragment.edges:
        g.add_edge(
            edge.from_node_id,
            edge.to_node_id,
            length_m=edge.length_m,
            edge_id=edge.edge_id,
            source_ref={"edge_kind": "scan_internal"},
        )

    return g, node_lookup


def _dedupe_scan_ids(scan_ids: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for scan_id in scan_ids:
        if scan_id in seen:
            continue
        out.append(scan_id)
        seen.add(scan_id)
    return out
