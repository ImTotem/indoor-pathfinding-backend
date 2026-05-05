"""QualityGateStep 단위 테스트."""
from __future__ import annotations

from uuid import uuid4

import numpy as np

from indoor_server.application.building.steps.quality_gate import QualityGateStep
from indoor_server.domain.building.enums import BuildFailureReason, EdgeType, NodeType
from indoor_server.domain.building.models import (
    GridOrigin,
    MapEdgeVO,
    MapNodeVO,
    WalkableGrid,
)


def _make_grid(walkable_fraction: float = 0.5) -> WalkableGrid:
    h, w = 10, 10
    mask = np.zeros((h, w), dtype=bool)
    n_walkable = int(h * w * walkable_fraction)
    mask.flat[:n_walkable] = True
    origin = GridOrigin(x0=0.0, y0=0.0, z0=0.0, cell_size=0.10, w=w, h=h)
    obs = np.zeros((h, w), dtype=np.uint16)
    return WalkableGrid(origin=origin, mask=mask, observation_count=obs)


def _make_skeleton_node(sid: object, bjid: object) -> MapNodeVO:
    return MapNodeVO(
        node_id=uuid4(),
        scan_id=sid,
        build_job_id=bjid,
        node_type=NodeType.JUNCTION,
        x=0.0, y=0.0, z=0.0,
    )


def _make_skeleton_edge(sid: object, bjid: object, fn: object, tn: object) -> MapEdgeVO:
    return MapEdgeVO(
        edge_id=uuid4(),
        scan_id=sid,
        build_job_id=bjid,
        from_node_id=fn,
        to_node_id=tn,
        edge_type=EdgeType.SKELETON,
        polyline=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        length_m=1.0,
    )


def test_low_coverage_fails():
    gate = QualityGateStep(min_coverage=0.4)
    grid = _make_grid(walkable_fraction=0.2)  # 20% < 40%
    report = gate.evaluate(grid, nodes=[], edges=[])
    assert not report.passed
    assert report.failure_reason == BuildFailureReason.WALKABLE_COVERAGE_LOW


def test_disconnected_fails():
    """Sprint 25 이후: components ≤ 3은 허용(POI spur 고려). 4 이상이어야 실패."""
    gate = QualityGateStep(min_coverage=0.1)
    grid = _make_grid(walkable_fraction=0.5)
    sid, bjid = uuid4(), uuid4()

    # 4개의 분리된 노드 (edge 없음) → components=4 > 3 → 실패
    n1 = _make_skeleton_node(sid, bjid)
    n2 = _make_skeleton_node(sid, bjid)
    n3 = _make_skeleton_node(sid, bjid)
    n4 = _make_skeleton_node(sid, bjid)
    report = gate.evaluate(grid, nodes=[n1, n2, n3, n4], edges=[])
    assert not report.passed
    assert report.failure_reason == BuildFailureReason.GRAPH_DISCONNECTED


def test_passing_gate():
    gate = QualityGateStep(min_coverage=0.4)
    grid = _make_grid(walkable_fraction=0.5)
    sid, bjid = uuid4(), uuid4()
    n1 = _make_skeleton_node(sid, bjid)
    n2 = _make_skeleton_node(sid, bjid)
    e = _make_skeleton_edge(sid, bjid, n1.node_id, n2.node_id)
    report = gate.evaluate(grid, nodes=[n1, n2], edges=[e])
    assert report.passed
    assert report.failure_reason is None


def test_default_gate_accepts_sparse_rtabmap_floor_grid():
    """RTABMap/floor-pointcloud bbox coverage는 sparse해도 graph가 좋으면 통과한다."""
    grid = _make_grid(walkable_fraction=0.12)
    sid, bjid = uuid4(), uuid4()
    n1 = _make_skeleton_node(sid, bjid)
    n2 = _make_skeleton_node(sid, bjid)
    e = _make_skeleton_edge(sid, bjid, n1.node_id, n2.node_id)
    report = QualityGateStep().evaluate(grid, nodes=[n1, n2], edges=[e])
    assert report.passed
    assert report.failure_reason is None
