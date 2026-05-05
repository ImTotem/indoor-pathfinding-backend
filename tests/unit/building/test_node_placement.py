"""NodePlacementStep 단위 테스트 (Sprint 23 트랙 A — densify 회귀).

주요 검증:
  - 직선 2점 polyline + total_len=10m → max_edge_length_m=3m → mid-node ≥ 3개 생성
  - polyline 여러 점 + total_len=6m → mid-node 생성 확인
  - total_len ≤ max_edge_length_m → mid-node 없음
  - junction/endpoint/corridor 분류
"""
from __future__ import annotations

import math
from uuid import UUID

from indoor_server.application.building.steps.node_placement import (
    NodePlacementStep,
    _polyline_length,
)
from indoor_server.application.building.steps.skeletonize import (
    SkeletonEdge,
    SkeletonGraph,
    SkeletonNode,
)
from indoor_server.domain.building.enums import NodeType
from indoor_server.domain.building.models import GridOrigin

_SCAN_ID = UUID("00000000-0000-0000-0000-000000000001")
_JOB_ID = UUID("00000000-0000-0000-0000-000000000002")

_ORIGIN = GridOrigin(x0=0.0, y0=0.0, z0=0.0, cell_size=1.0, w=200, h=200)


def _make_step(max_edge_length_m: float = 3.0) -> NodePlacementStep:
    return NodePlacementStep(
        scan_id=_SCAN_ID,
        build_job_id=_JOB_ID,
        max_edge_length_m=max_edge_length_m,
    )


def _make_skel(
    u_px: tuple[int, int],
    v_px: tuple[int, int],
    polyline_px: list[tuple[int, int]],
    u_degree: int = 1,
    v_degree: int = 1,
) -> SkeletonGraph:
    """단일 edge를 가진 SkeletonGraph 생성."""
    node_u = SkeletonNode(pixel_x=u_px[0], pixel_y=u_px[1], degree=u_degree)
    node_v = SkeletonNode(pixel_x=v_px[0], pixel_y=v_px[1], degree=v_degree)
    edge = SkeletonEdge(u=u_px, v=v_px, polyline_px=polyline_px)
    return SkeletonGraph(
        nodes=[node_u, node_v], edges=[edge], skeleton_pixel_count=len(polyline_px)
    )


# ── densify 핵심 케이스 ───────────────────────────────────────────────────────


def test_densify_straight_2pt_polyline_10m() -> None:
    """직선 2점 polyline + total_len=10m → mid-node ≥ 3개 생성 (Sprint 23 회귀 테스트).

    cell_size=1.0m, u=(0,0), v=(10,0) → polyline_px = [(0,0),(10,0)]
    total_len = 10m, max_edge_length_m=3m
    num_samples = int(10/3) - 1 = 2 → corridor node 2개 삽입
    chain = [u, mid1, mid2, v] → split_edges 3개
    """
    step = _make_step(max_edge_length_m=3.0)
    skel = _make_skel(
        u_px=(0, 0),
        v_px=(10, 0),
        polyline_px=[(0, 0), (10, 0)],  # 2점 직선
    )
    nodes, edges = step.run(skel, _ORIGIN)

    # u, v 제외 corridor 노드만 필터
    corridor_nodes = [n for n in nodes if n.node_type == NodeType.CORRIDOR]
    assert len(corridor_nodes) >= 2, (
        f"10m 직선에서 mid-node ≥ 2개 기대, 실제 {len(corridor_nodes)}개"
    )
    # edge 수는 corridor_nodes + 1 = chain segments
    assert len(edges) >= 3, f"split_edges ≥ 3 기대, 실제 {len(edges)}개"


def test_densify_multi_pt_polyline_6m() -> None:
    """여러 점 polyline + total_len=6m → mid-node 생성 확인.

    u=(0,0), 중간점 (3,0), v=(6,0) → total_len=6m, max=3m
    num_samples = int(6/3) - 1 = 1 → mid-node 1개
    """
    step = _make_step(max_edge_length_m=3.0)
    skel = _make_skel(
        u_px=(0, 0),
        v_px=(6, 0),
        polyline_px=[(0, 0), (3, 0), (6, 0)],
    )
    nodes, edges = step.run(skel, _ORIGIN)
    corridor_nodes = [n for n in nodes if n.node_type == NodeType.CORRIDOR]
    assert len(corridor_nodes) >= 1


def test_no_densify_short_edge() -> None:
    """total_len ≤ max_edge_length_m → mid-node 없음."""
    step = _make_step(max_edge_length_m=3.0)
    skel = _make_skel(
        u_px=(0, 0),
        v_px=(2, 0),
        polyline_px=[(0, 0), (2, 0)],
    )
    nodes, edges = step.run(skel, _ORIGIN)
    corridor_nodes = [n for n in nodes if n.node_type == NodeType.CORRIDOR]
    assert len(corridor_nodes) == 0
    assert len(edges) == 1


def test_densify_preserves_total_length() -> None:
    """densify 후 split_edge 길이 합계 == original total_len (오차 ≤ 1mm)."""
    step = _make_step(max_edge_length_m=3.0)
    skel = _make_skel(
        u_px=(0, 0),
        v_px=(10, 0),
        polyline_px=[(0, 0), (10, 0)],
    )
    nodes, edges = step.run(skel, _ORIGIN)
    total_split = sum(e.length_m for e in edges)
    assert abs(total_split - 10.0) < 0.001, f"길이 합계={total_split:.4f}m, 기대 10.0m"


# ── 노드 분류 ─────────────────────────────────────────────────────────────────


def test_junction_node_classification() -> None:
    """degree≥3 → JUNCTION."""
    step = _make_step()
    skel = SkeletonGraph(
        nodes=[SkeletonNode(pixel_x=0, pixel_y=0, degree=3)],
        edges=[],
        skeleton_pixel_count=1,
    )
    nodes, edges = step.run(skel, _ORIGIN)
    assert nodes[0].node_type == NodeType.JUNCTION


def test_endpoint_node_classification() -> None:
    """degree==1 → ENDPOINT."""
    step = _make_step()
    skel = SkeletonGraph(
        nodes=[SkeletonNode(pixel_x=0, pixel_y=0, degree=1)],
        edges=[],
        skeleton_pixel_count=1,
    )
    nodes, edges = step.run(skel, _ORIGIN)
    assert nodes[0].node_type == NodeType.ENDPOINT


# ── _polyline_length 유틸 ─────────────────────────────────────────────────────


def test_polyline_length_straight() -> None:
    """수평 직선 10m."""
    pts = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    assert math.isclose(_polyline_length(pts), 10.0, rel_tol=1e-9)


def test_polyline_length_empty() -> None:
    """1점 polyline → 0."""
    assert _polyline_length([(0.0, 0.0, 0.0)]) == 0.0
