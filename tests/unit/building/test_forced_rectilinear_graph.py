"""Forced Rectilinear Graph 단위 테스트 (Sprint 34 Cycle 8).

검증 항목:
  1. T-junction skeleton에 forced rectilinear 적용 → 모든 edge axis-aligned
  2. zigzag polyline → rectilinear 후 edge 수 감소
  3. force_rectilinear=False → 기존 방식과 동일 (회귀 방지)
  4. 단순 수평 edge → 변경 없음 (이미 axis-aligned)
  5. densify는 rectilinear 후에도 동작 (max_edge_length_m 초과 시)
"""
from __future__ import annotations

from uuid import UUID

from indoor_server.application.building.steps.node_placement import (
    NodePlacementStep,
    _apply_forced_rectilinear_to_polyline,
    _force_rectilinear_polyline,
    _rdp_simplify,
)
from indoor_server.application.building.steps.skeletonize import (
    SkeletonEdge,
    SkeletonGraph,
    SkeletonNode,
)
from indoor_server.domain.building.models import GridOrigin

_SCAN_ID = UUID("00000000-0000-0000-0000-000000000001")
_JOB_ID = UUID("00000000-0000-0000-0000-000000000002")

_ORIGIN = GridOrigin(x0=0.0, y0=0.0, z0=0.0, cell_size=1.0, w=200, h=200)


def _make_step(
    max_edge_length_m: float = 3.0,
    force_rectilinear: bool = True,
    simplify_m: float = 0.5,
    min_edge_m: float = 0.10,
) -> NodePlacementStep:
    return NodePlacementStep(
        scan_id=_SCAN_ID,
        build_job_id=_JOB_ID,
        max_edge_length_m=max_edge_length_m,
        force_rectilinear=force_rectilinear,
        rectilinear_simplify_m=simplify_m,
        rectilinear_min_edge_m=min_edge_m,
    )


def _is_axis_aligned(polyline: list[tuple[float, float, float]], tol: float = 1e-6) -> bool:
    """polyline의 모든 segment가 수평 또는 수직 (axis-aligned) 인지 확인."""
    for i in range(len(polyline) - 1):
        dx = abs(polyline[i + 1][0] - polyline[i][0])
        dy = abs(polyline[i + 1][1] - polyline[i][1])
        # 수평: dy ≈ 0, 수직: dx ≈ 0
        if dx > tol and dy > tol:
            return False
    return True


# ── 헬퍼 함수 단위 테스트 ────────────────────────────────────────────────────


class TestRdpSimplify:
    def test_straight_line_simplify(self) -> None:
        """직선 위 중간 점들은 모두 제거된다."""
        pts = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (10.0, 0.0)]
        result = _rdp_simplify(pts, tolerance=0.1)
        assert result == [(0.0, 0.0), (10.0, 0.0)], f"직선 simplify 실패: {result}"

    def test_zigzag_keeps_corners(self) -> None:
        """꺾임 포인트는 보존된다."""
        pts = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)]
        result = _rdp_simplify(pts, tolerance=0.1)
        assert len(result) == 3, f"코너 보존 실패: {result}"

    def test_two_points_unchanged(self) -> None:
        """2점 이하는 그대로 반환."""
        pts = [(0.0, 0.0), (1.0, 1.0)]
        assert _rdp_simplify(pts, 0.5) == pts


class TestForceRectilinearPolyline:
    def test_diagonal_becomes_hv(self) -> None:
        """45도 사선 → H 또는 V로 강제."""
        pts = [(0.0, 0.0), (5.0, 5.0)]
        result = _force_rectilinear_polyline(pts, min_edge_m=0.05)
        # 결과는 2점: 첫 점과 H 또는 V로 꺾인 점
        assert len(result) >= 2
        # 모든 segment가 H 또는 V
        for i in range(len(result) - 1):
            dx = abs(result[i + 1][0] - result[i][0])
            dy = abs(result[i + 1][1] - result[i][1])
            assert dx < 1e-9 or dy < 1e-9, f"segment {i} axis-aligned 아님: dx={dx}, dy={dy}"

    def test_horizontal_edge_unchanged(self) -> None:
        """이미 수평인 edge → 변경 없음."""
        pts = [(0.0, 0.0), (5.0, 0.0)]
        result = _force_rectilinear_polyline(pts, min_edge_m=0.05)
        assert len(result) == 2
        assert abs(result[0][0] - 0.0) < 1e-9
        assert abs(result[1][0] - 5.0) < 1e-9

    def test_vertical_edge_unchanged(self) -> None:
        """이미 수직인 edge → 변경 없음."""
        pts = [(0.0, 0.0), (0.0, 5.0)]
        result = _force_rectilinear_polyline(pts, min_edge_m=0.05)
        assert len(result) == 2
        assert abs(result[0][1] - 0.0) < 1e-9
        assert abs(result[1][1] - 5.0) < 1e-9

    def test_stair_step_merged(self) -> None:
        """작은 stair-step → collinear merge로 직선화."""
        # (0,0) → (5,0) → (5,0.05) → (10,0.05) — y shift 0.05m는 min_edge_m 이하
        pts = [(0.0, 0.0), (5.0, 0.0), (5.0, 0.05), (10.0, 0.05)]
        result = _force_rectilinear_polyline(pts, min_edge_m=0.1)
        # 노이즈 shift(0.05m < min_edge_m=0.1m) 때문에 segment가 줄어야 함
        assert len(result) <= 4


class TestApplyForcedRectilinear:
    def test_diagonal_world_polyline(self) -> None:
        """사선 world polyline → axis-aligned 변환."""
        poly = [(0.0, 0.0, 1.0), (3.0, 3.0, 1.0), (6.0, 3.0, 1.0)]
        result = _apply_forced_rectilinear_to_polyline(poly, simplify_m=0.5, min_edge_m=0.1)
        assert _is_axis_aligned(result), f"axis-aligned 아님: {result}"

    def test_z_preserved(self) -> None:
        """z 좌표는 첫 점의 값으로 유지."""
        poly = [(0.0, 0.0, 2.5), (5.0, 5.0, 2.5)]
        result = _apply_forced_rectilinear_to_polyline(poly, simplify_m=0.0, min_edge_m=0.05)
        for pt in result:
            assert abs(pt[2] - 2.5) < 1e-9, f"z 좌표 변경됨: {pt[2]}"

    def test_already_aligned_preserved(self) -> None:
        """이미 aligned된 polyline → 첫·마지막 점 보존."""
        poly = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
        result = _apply_forced_rectilinear_to_polyline(poly, simplify_m=0.5, min_edge_m=0.1)
        assert len(result) >= 2
        assert abs(result[0][0] - 0.0) < 1e-6
        assert abs(result[-1][0] - 10.0) < 1e-6


# ── NodePlacementStep 통합 테스트 ────────────────────────────────────────────


def _make_skel(
    nodes: list[tuple[int, int, int]],     # (px, py, degree)
    edges: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]],
) -> SkeletonGraph:
    skel_nodes = [SkeletonNode(pixel_x=px, pixel_y=py, degree=deg) for px, py, deg in nodes]
    skel_edges = [
        SkeletonEdge(u=u, v=v, polyline_px=poly) for u, v, poly in edges
    ]
    return SkeletonGraph(
        nodes=skel_nodes,
        edges=skel_edges,
        skeleton_pixel_count=sum(len(e.polyline_px) for e in skel_edges),
    )


class TestNodePlacementRectilinear:
    def test_single_horizontal_edge_axis_aligned(self) -> None:
        """수평 straight edge → axis-aligned 유지."""
        skel = _make_skel(
            nodes=[(0, 0, 1), (10, 0, 1)],
            edges=[((0, 0), (10, 0), [(0, 0), (10, 0)])],
        )
        step = _make_step(max_edge_length_m=100.0)
        nodes, edges = step.run(skel, _ORIGIN)
        assert len(edges) == 1
        assert _is_axis_aligned(edges[0].polyline), (
            f"수평 edge axis-aligned 아님: {edges[0].polyline}"
        )

    def test_diagonal_edge_becomes_axis_aligned(self) -> None:
        """대각선 zigzag polyline → axis-aligned 변환."""
        # (0,0) → (1,1) → (2,0) → (3,1) → (4,0): stair-step zigzag
        skel = _make_skel(
            nodes=[(0, 0, 1), (4, 0, 1)],
            edges=[
                ((0, 0), (4, 0), [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0)])
            ],
        )
        step = _make_step(max_edge_length_m=100.0, simplify_m=0.5)
        nodes, edges = step.run(skel, _ORIGIN)
        assert len(edges) >= 1
        for e in edges:
            assert _is_axis_aligned(e.polyline), f"axis-aligned 아님: {e.polyline}"

    def test_t_junction_all_edges_axis_aligned(self) -> None:
        """T-junction (3-way): 모든 edge axis-aligned."""
        # Junction at (5,5), endpoints at (0,5), (10,5), (5,0)
        skel = _make_skel(
            nodes=[(0, 5, 1), (10, 5, 1), (5, 0, 1), (5, 5, 3)],
            edges=[
                ((0, 5), (5, 5), [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]),
                ((5, 5), (10, 5), [(5, 5), (6, 5), (7, 5), (8, 5), (9, 5), (10, 5)]),
                ((5, 0), (5, 5), [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]),
            ],
        )
        step = _make_step(max_edge_length_m=100.0)
        nodes, edges = step.run(skel, _ORIGIN)
        assert len(edges) == 3
        for e in edges:
            assert _is_axis_aligned(e.polyline), f"T-junction edge axis-aligned 아님: {e.polyline}"

    def test_force_rectilinear_false_no_change(self) -> None:
        """force_rectilinear=False → 기존 동작 (zigzag 유지 회귀 방지)."""
        skel = _make_skel(
            nodes=[(0, 0, 1), (4, 4, 1)],
            edges=[
                ((0, 0), (4, 4), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])
            ],
        )
        step_raw = _make_step(max_edge_length_m=100.0, force_rectilinear=False)
        _, edges_raw = step_raw.run(skel, _ORIGIN)
        assert len(edges_raw) == 1
        # raw mode에서는 대각선이 유지됨 (axis-aligned 아님)
        poly = edges_raw[0].polyline
        has_diagonal = any(
            abs(poly[i + 1][0] - poly[i][0]) > 1e-6 and abs(poly[i + 1][1] - poly[i][1]) > 1e-6
            for i in range(len(poly) - 1)
        )
        assert has_diagonal, "force_rectilinear=False인데 diagonal이 없음 (예상 외)"

    def test_densify_after_rectilinear(self) -> None:
        """rectilinear 후 max_edge_length_m 초과 시 densify 동작."""
        # 10m 수평 edge → rectilinear 후에도 10m, max=3m → corridor 노드 삽입
        skel = _make_skel(
            nodes=[(0, 0, 1), (10, 0, 1)],
            edges=[((0, 0), (10, 0), [(0, 0), (10, 0)])],
        )
        step = _make_step(max_edge_length_m=3.0)
        nodes, edges = step.run(skel, _ORIGIN)
        total_len = sum(e.length_m for e in edges)
        assert abs(total_len - 10.0) < 0.01, f"densify 후 길이 합계 오류: {total_len}"
        assert len(edges) >= 3, f"densify 후 edge 수 부족: {len(edges)}"
        # 모든 edge axis-aligned
        for e in edges:
            assert _is_axis_aligned(e.polyline), f"densify 후 axis-aligned 아님: {e.polyline}"

    def test_zigzag_reduces_edge_count(self) -> None:
        """zigzag polyline → rectilinear 후 segment 수 감소 (simplify 효과)."""
        # 작은 stair-step들이 연속된 zigzag: simplify 0.5m로 제거되어야 함
        poly = [(i, i % 2) for i in range(10)]  # stair-step (pixel coords: 2-tuple)
        skel = _make_skel(
            nodes=[(0, 0, 1), (9, 1, 1)],
            edges=[((0, 0), (9, 1), poly)],
        )
        step_rect = _make_step(max_edge_length_m=100.0, simplify_m=0.5)
        step_raw = _make_step(max_edge_length_m=100.0, force_rectilinear=False)
        _, edges_rect = step_rect.run(skel, _ORIGIN)
        _, edges_raw = step_raw.run(skel, _ORIGIN)
        raw_pts = sum(len(e.polyline) for e in edges_raw)
        rect_pts = sum(len(e.polyline) for e in edges_rect)
        assert rect_pts <= raw_pts, (
            f"rectilinear 후 점 수가 줄지 않음: raw={raw_pts}, rect={rect_pts}"
        )
