"""Sprint 50 v2 — axis pair + multi-component + dynamic precision tests.

본 모듈은 v2 추가 기능만 검증한다. v1 호환은 test_rectangle_dictionary_cover.py
가 책임. v1 mode (axes_override=None, precision_threshold_dynamic=False)는
정확히 v1 결과를 재현해야 하므로, 본 파일은 axes_override 또는 dynamic 활성
시의 동작만 본다.
"""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from indoor_server.application.building.steps.rectangle_dictionary_cover import (
    RectangleDictionaryCoverParams,
    RectangleDictionaryCoverStep,
    axes_pair_from_primary,
    compute_dynamic_precision_threshold,
    estimate_dominant_axis_from_links,
    estimate_dominant_axis_from_obb,
    split_heatmap_by_components,
)
from indoor_server.domain.building.models import GridOrigin, WalkableGrid

# ── fixtures ────────────────────────────────────────────────────────────────


def _make_grid(
    heatmap: np.ndarray,
    cell_size: float = 0.10,
    x0: float = 0.0,
    y0: float = 0.0,
) -> WalkableGrid:
    h, w = heatmap.shape
    origin = GridOrigin(
        x0=x0, y0=y0, z0=0.0, cell_size=cell_size, w=w, h=h
    )
    return WalkableGrid(
        origin=origin,
        mask=(heatmap > 0),
        observation_count=heatmap.astype(np.uint16),
    )


def _corridor_heatmap(
    *, width_cells: int = 60, height_cells: int = 40,
    corridor_thickness_cells: int = 8, hits_per_cell: int = 4,
) -> np.ndarray:
    h = np.zeros((height_cells, width_cells), dtype=np.int32)
    r0 = (height_cells - corridor_thickness_cells) // 2
    h[r0 : r0 + corridor_thickness_cells, 5 : width_cells - 5] = hits_per_cell
    return h


def _two_disjoint_corridors_heatmap() -> np.ndarray:
    """가로 + 세로 두 corridor (4 cell gap) — multi-component 검증용."""
    h = np.zeros((100, 100), dtype=np.int32)
    h[10:18, 5:55] = 5  # 가로 corridor 1
    # 4 cell gap (row 18~21)
    h[22:55, 30:38] = 5  # 세로 corridor 2 (가로와 겹치지 않음)
    return h


# ── compute_dynamic_precision_threshold ──────────────────────────────────────


def test_dynamic_threshold_below_3m_returns_base() -> None:
    base = 0.85
    floor_min = 0.65
    for thickness in (0.5, 1.0, 2.0, 3.0):
        v = compute_dynamic_precision_threshold(
            thickness_m=thickness, base=base, floor_min=floor_min
        )
        assert v == pytest.approx(base, abs=1e-9)


def test_dynamic_threshold_above_3m_relaxes() -> None:
    base = 0.85
    floor_min = 0.65
    # thickness 4 → drop = (4/3 - 1) * 0.05 = 0.0167 → 0.833
    v_4 = compute_dynamic_precision_threshold(
        thickness_m=4.0, base=base, floor_min=floor_min
    )
    assert 0.82 < v_4 < base
    # thickness 6 → drop = (6/3 - 1) * 0.05 = 0.05 → 0.80
    v_6 = compute_dynamic_precision_threshold(
        thickness_m=6.0, base=base, floor_min=floor_min
    )
    assert v_6 == pytest.approx(0.80, abs=1e-9)
    # thickness 15 → drop = (15/3 - 1) * 0.05 = 0.20 → 0.65 (floor)
    v_15 = compute_dynamic_precision_threshold(
        thickness_m=15.0, base=base, floor_min=floor_min
    )
    assert v_15 == pytest.approx(floor_min, abs=1e-9)


def test_dynamic_threshold_floor_clamps() -> None:
    """floor_min 이하로 절대 안 떨어진다."""
    v = compute_dynamic_precision_threshold(
        thickness_m=100.0, base=0.85, floor_min=0.50
    )
    assert v == pytest.approx(0.50, abs=1e-9)


# ── axes_pair_from_primary ───────────────────────────────────────────────────


def test_axes_pair_orthogonal() -> None:
    p, q = axes_pair_from_primary(0.0)
    assert p == 0.0
    assert q == 90.0
    p, q = axes_pair_from_primary(45.0)
    assert p == 45.0
    assert q == 135.0
    p, q = axes_pair_from_primary(-22.0)  # negative → mod 180
    assert p == pytest.approx(158.0, abs=1e-6)
    assert q == pytest.approx(68.0, abs=1e-6)


# ── estimate_dominant_axis_from_links ────────────────────────────────────────


def test_estimate_axis_from_links_horizontal() -> None:
    segments = [((0.0, 0.0), (5.0, 0.0)), ((5.0, 0.0), (10.0, 0.0))]
    out = estimate_dominant_axis_from_links(segments, min_best_bin_ratio=0.10)
    assert out is not None
    angle, ratio = out
    assert abs(angle) < 5.0
    assert ratio >= 0.5


def test_estimate_axis_from_links_empty() -> None:
    out = estimate_dominant_axis_from_links([], min_best_bin_ratio=0.10)
    assert out is None


def test_estimate_axis_from_links_low_ratio_rejects() -> None:
    """모든 axis 에 균등 분포 → best_bin_ratio 낮아 None."""
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    # 다양한 각도 segment
    for ang_deg in (0, 10, 20, 30, 40, 50, 60, 70, 80):
        rad = np.deg2rad(ang_deg)
        segments.append(((0.0, 0.0), (np.cos(rad), np.sin(rad))))
    out = estimate_dominant_axis_from_links(segments, min_best_bin_ratio=0.50)
    assert out is None


# ── estimate_dominant_axis_from_obb ──────────────────────────────────────────


def test_estimate_axis_from_obb_elongated() -> None:
    # 10×2 rectangle along x-axis
    poly = Polygon([(0, 0), (10, 0), (10, 2), (0, 2)])
    out = estimate_dominant_axis_from_obb(poly, min_aspect_ratio=1.5)
    assert out is not None
    angle, aspect = out
    assert abs(angle) < 5.0
    assert aspect >= 4.5


def test_estimate_axis_from_obb_square_rejects() -> None:
    """square (aspect=1) → None."""
    poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    out = estimate_dominant_axis_from_obb(poly, min_aspect_ratio=1.5)
    assert out is None


# ── split_heatmap_by_components ──────────────────────────────────────────────


def test_split_components_single() -> None:
    heatmap = _corridor_heatmap()
    grid = _make_grid(heatmap)
    comps = split_heatmap_by_components(
        heatmap, origin=grid.origin, min_component_cells=10
    )
    assert len(comps) == 1


def test_split_components_two_disjoint() -> None:
    heatmap = _two_disjoint_corridors_heatmap()
    grid = _make_grid(heatmap)
    comps = split_heatmap_by_components(
        heatmap, origin=grid.origin, min_component_cells=10
    )
    assert len(comps) == 2
    # 각 component 가 sub_grid 의 origin 정확히 가짐
    for sub_grid, bbox in comps:
        r0, c0, r1, c1 = bbox
        assert sub_grid.origin.cell_size == grid.origin.cell_size
        assert sub_grid.origin.w == c1 - c0
        assert sub_grid.origin.h == r1 - r0
        assert int(sub_grid.observation_count.sum()) > 0


def test_split_components_filter_small() -> None:
    """작은 component drop."""
    h = np.zeros((40, 40), dtype=np.int32)
    h[5:7, 5:7] = 3  # 4 cells (작음)
    h[20:30, 20:35] = 5  # 150 cells (큼)
    grid = _make_grid(h)
    comps = split_heatmap_by_components(
        h, origin=grid.origin, min_component_cells=50
    )
    assert len(comps) == 1


# ── axes_override pair sweep ─────────────────────────────────────────────────


def test_axes_override_only_uses_specified_angles() -> None:
    """axes_override=(0, 90) → rotated_grid_count == 2."""
    heatmap = _corridor_heatmap()
    grid = _make_grid(heatmap)
    params = RectangleDictionaryCoverParams(
        axes_override=(0.0, 90.0),
    )
    result = RectangleDictionaryCoverStep(params).run(grid)
    # rotated_grid_count = len(axes_override)
    assert result.metadata["rotated_grid_count"] == 2


def test_axes_override_all_axes_in_pair_true() -> None:
    """corridor (axis-aligned) → axes_override=(0, 90) 으로 cover. all_axes_in_pair=True."""
    heatmap = _corridor_heatmap()
    grid = _make_grid(heatmap)
    params = RectangleDictionaryCoverParams(
        axes_override=(0.0, 90.0),
    )
    result = RectangleDictionaryCoverStep(params).run(grid)
    assert result.accepted
    assert result.metadata["all_axes_in_pair"] is True
    # axes_used 는 (0, 90) set 안에 들어있음
    axes_used = set(result.metadata["axes_used"])  # type: ignore[arg-type]
    assert axes_used.issubset({0.0, 90.0})


def test_v1_default_no_axes_override_metadata() -> None:
    """axes_override=None (v1) → axes_override_active=False, all_axes_in_pair True (trivial)."""
    heatmap = _corridor_heatmap()
    grid = _make_grid(heatmap)
    result = RectangleDictionaryCoverStep().run(grid)
    assert result.metadata["axes_override_active"] is False
    # v1 default 는 all_axes_in_pair True (자기참조)
    assert result.metadata["all_axes_in_pair"] is True


# ── dynamic precision threshold ─────────────────────────────────────────────


def test_dynamic_precision_metadata_dump() -> None:
    """dynamic mode 활성 시 metadata 에 thickness 별 임계 dump."""
    heatmap = _corridor_heatmap()
    grid = _make_grid(heatmap)
    params = RectangleDictionaryCoverParams(
        precision_threshold=0.85,
        precision_threshold_min=0.65,
        precision_threshold_dynamic=True,
    )
    result = RectangleDictionaryCoverStep(params).run(grid)
    assert result.metadata["precision_threshold_dynamic_active"] is True
    by_thickness = result.metadata["precision_threshold_by_thickness"]
    assert isinstance(by_thickness, dict)
    # thickness 0.5m → 0.85 그대로
    assert by_thickness[0.5] == pytest.approx(0.85, abs=1e-9)
    # thickness 6m → 0.80
    assert by_thickness[6.0] == pytest.approx(0.80, abs=1e-9)
    # thickness 15m → 0.65 floor
    assert by_thickness[15.0] == pytest.approx(0.65, abs=1e-9)


def test_dynamic_precision_off_uses_fixed() -> None:
    """dynamic=False → 모든 thickness 에 동일 base 임계."""
    heatmap = _corridor_heatmap()
    grid = _make_grid(heatmap)
    params = RectangleDictionaryCoverParams(
        precision_threshold=0.85,
        precision_threshold_dynamic=False,
    )
    result = RectangleDictionaryCoverStep(params).run(grid)
    by_thickness = result.metadata["precision_threshold_by_thickness"]
    assert isinstance(by_thickness, dict)
    for v in by_thickness.values():
        assert v == pytest.approx(0.85, abs=1e-9)


# ── invalid params ──────────────────────────────────────────────────────────


def test_axes_override_empty_raises() -> None:
    with pytest.raises(ValueError):
        RectangleDictionaryCoverStep(
            RectangleDictionaryCoverParams(axes_override=())
        )


def test_precision_min_above_base_raises() -> None:
    with pytest.raises(ValueError):
        RectangleDictionaryCoverStep(
            RectangleDictionaryCoverParams(
                precision_threshold=0.70,
                precision_threshold_min=0.85,
            )
        )


# ── v1 mode regression: axes_override=None + dynamic=False == v1 ────────────


def test_v1_mode_reproduces_v1_behavior_corridor() -> None:
    """axes_override=None + dynamic=False (default) → v1 corridor 결과와 동일하게 accept."""
    heatmap = _corridor_heatmap()
    grid = _make_grid(heatmap)
    params = RectangleDictionaryCoverParams(
        axes_override=None,
        precision_threshold_dynamic=False,
    )
    result = RectangleDictionaryCoverStep(params).run(grid)
    assert result.accepted is True
    assert result.metadata["axes_override_active"] is False
    assert result.metadata["precision_threshold_dynamic_active"] is False


# ── pair mode reduces selected angle diversity ──────────────────────────────


def test_axes_pair_constrains_angles() -> None:
    """L corner 환경에서 axis-aligned axes_override (0, 90) 적용 시 모든 selected
    angle 이 {0, 90}. 사선 채택 X (사용자 의도 직각 보장).
    """
    h = np.zeros((60, 60), dtype=np.int32)
    h[10:18, 5:55] = 5
    h[10:55, 5:13] = 5
    grid = _make_grid(h)
    params = RectangleDictionaryCoverParams(
        axes_override=(0.0, 90.0),
        precision_threshold_dynamic=True,
    )
    result = RectangleDictionaryCoverStep(params).run(grid)
    assert result.accepted
    selected_angles = set(
        result.metadata["selected_angles_unique"]  # type: ignore[arg-type]
    )
    assert selected_angles.issubset({0.0, 90.0})
    assert result.metadata["all_axes_in_pair"] is True
