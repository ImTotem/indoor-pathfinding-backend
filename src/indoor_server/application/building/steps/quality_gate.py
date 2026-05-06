"""QualityGateStep — walkable coverage + graph connectivity 검증."""
from __future__ import annotations

import logging

import numpy as np
from pydantic import BaseModel, ConfigDict

from indoor_server.domain.building.enums import BuildFailureReason
from indoor_server.domain.building.models import MapEdgeVO, MapNodeVO, WalkableGrid

logger = logging.getLogger(__name__)

# ASSUMPTION A-8
# Sprint 22/63 알려진 이슈: grid bbox가 trajectory/RTABMap evidence 전체라
# 분모가 커서 ratio가 낮게 보고된다. Floor-pointcloud/RTABMap 경로에서는
# 충분한 polygon/graph가 생성돼도 0.30 미만인 실 scan이 있어, coverage는
# 완전 실패 탐지용 하한으로만 둔다. 지도 품질은 graph connectivity와
# downstream route 검증에서 다시 확인한다.
_DEFAULT_MIN_COVERAGE = 0.10
_DEFAULT_MAX_COMPONENTS = 3


class QualityReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    walkable_coverage: float
    connected_components: int
    passed: bool
    failure_reason: BuildFailureReason | None = None


class QualityGateStep:
    def __init__(
        self,
        min_coverage: float = _DEFAULT_MIN_COVERAGE,
        max_components: int = _DEFAULT_MAX_COMPONENTS,
    ) -> None:
        self._min_coverage = min_coverage
        self._max_components = max_components

    def evaluate(
        self,
        grid: WalkableGrid,
        nodes: list[MapNodeVO],
        edges: list[MapEdgeVO],
    ) -> QualityReport:
        coverage = self._compute_coverage(grid)
        components = self._count_components(nodes, edges)

        logger.info(
            "quality_gate coverage=%.3f components=%d min_coverage=%.2f",
            coverage,
            components,
            self._min_coverage,
        )

        if coverage < self._min_coverage:
            return QualityReport(
                walkable_coverage=coverage,
                connected_components=components,
                passed=False,
                failure_reason=BuildFailureReason.WALKABLE_COVERAGE_LOW,
            )

        # Sprint 25: components 1개 강제는 너무 엄격. POI projection이 추가하는
        # spur 노드가 main skeleton과 분리될 수 있어 ≤ 3까지 허용.
        # 실제 navigation에서 끊긴 component는 /route 422 PATH_NOT_FOUND로 처리됨.
        if components > self._max_components:
            return QualityReport(
                walkable_coverage=coverage,
                connected_components=components,
                passed=False,
                failure_reason=BuildFailureReason.GRAPH_DISCONNECTED,
            )

        return QualityReport(
            walkable_coverage=coverage,
            connected_components=components,
            passed=True,
        )

    def _compute_coverage(self, grid: WalkableGrid) -> float:
        """walkable cell 수 / 총 cell 수."""
        total = grid.mask.size
        if total == 0:
            return 0.0
        return float(np.sum(grid.mask)) / float(total)

    def _count_components(
        self,
        nodes: list[MapNodeVO],
        edges: list[MapEdgeVO],
    ) -> int:
        """skeleton 노드/엣지 기준 connected component 수 (poi/poi_attach 제외)."""
        if not nodes:
            return 0

        import networkx as nx

        from indoor_server.domain.building.enums import NodeType

        skeleton_node_ids = {
            str(n.node_id)
            for n in nodes
            if n.node_type not in (NodeType.POI, NodeType.POI_ATTACH)
        }

        if not skeleton_node_ids:
            return 0

        graph: nx.Graph = nx.Graph()
        graph.add_nodes_from(skeleton_node_ids)

        from indoor_server.domain.building.enums import EdgeType

        for e in edges:
            if e.edge_type == EdgeType.SKELETON:
                fn = str(e.from_node_id)
                tn = str(e.to_node_id)
                if fn in skeleton_node_ids and tn in skeleton_node_ids:
                    graph.add_edge(fn, tn)

        return int(nx.number_connected_components(graph))
