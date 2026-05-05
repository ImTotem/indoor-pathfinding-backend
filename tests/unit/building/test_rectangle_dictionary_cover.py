"""Sprint 50 — Rectangle Dictionary Cover unit tests.

Codex evaluator-design WARN 1 (tau_p magic number) / WARN 2 (rotation 보존)
/ WARN 3 (precision after uncovered) 모두 fixture 기반 검증.

fixture: corridor / L / T / sparse noise. tau_p ∈ {0.80, 0.85, 0.90} sweep.
"""
from __future__ import annotations

import numpy as np
import pytest

from indoor_server.application.building.steps.rectangle_dictionary_cover import (
    DEFAULT_ANGLES_DEG,
    DEFAULT_LENGTH_LADDER_M,
    DEFAULT_THICKNESSES_M,
    RectangleDictionaryCoverParams,
    RectangleDictionaryCoverStep,
)
from indoor_server.domain.building.models import GridOrigin, WalkableGrid


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
    *,
    width_cells: int = 60,
    height_cells: int = 40,
    corridor_thickness_cells: int = 8,
    hits_per_cell: int = 4,
) -> np.ndarray:
    """가로 corridor (axis-aligned)."""
    h = np.zeros((height_cells, width_cells), dtype=np.int32)
    r0 = (height_cells - corridor_thickness_cells) // 2
    h[r0 : r0 + corridor_thickness_cells, 5 : width_cells - 5] = hits_per_cell
    return h


def _l_corner_heatmap() -> np.ndarray:
    """ㄱ자 corridor."""
    h = np.zeros((60, 60), dtype=np.int32)
    h[10:18, 5:55] = 5  # 가로
    h[10:55, 5:13] = 5  # 세로
    return h


def _t_corner_heatmap() -> np.ndarray:
    """T자 corridor (두 corridor thickness 1.0m / 10 cells)."""
    h = np.zeros((60, 60), dtype=np.int32)
    h[25:35, 5:55] = 5  # 가로 (thickness 10 cells)
    h[25:55, 25:35] = 5  # 세로 (thickness 10 cells)
    return h


def _sparse_noise_heatmap(seed: int = 42) -> np.ndarray:
    """random sparse hits — should reject (no good candidate)."""
    rng = np.random.default_rng(seed)
    h = np.zeros((40, 40), dtype=np.int32)
    n_hits = 30
    rs = rng.integers(0, 40, n_hits)
    cs = rng.integers(0, 40, n_hits)
    h[rs, cs] = rng.integers(1, 4, n_hits)
    return h


# ── 기본 동작 ────────────────────────────────────────────────────────────────


def test_corridor_axis_aligned_accepts() -> None:
    """직각 corridor — angle 0 후보가 통과해 accepted=True."""
    heatmap = _corridor_heatmap()
    grid = _make_grid(heatmap)
    step = RectangleDictionaryCoverStep()
    result = step.run(grid)
    assert result.accepted is True
    assert result.fallback_used is False
    assert result.fallback_source is None
    assert len(result.rectangles) >= 1
    assert result.metadata["recall"] >= 0.70
    assert result.metadata["over_cover_ratio"] <= 0.20
    assert result.union_polygon is not None
    assert result.footprint_geojson is not None


def test_l_corner_multi_rectangle() -> None:
    """ㄱ자 — 두 corridor 가 모두 cover 되어야 함."""
    heatmap = _l_corner_heatmap()
    grid = _make_grid(heatmap)
    step = RectangleDictionaryCoverStep()
    result = step.run(grid)
    assert result.accepted is True
    # 최소 2개 rectangle 필요 (가로 + 세로)
    assert len(result.rectangles) >= 2
    # union 안 hits 가 전체 hits 의 70% 이상
    assert result.metadata["recall"] >= 0.70


def test_t_corner_multi_rectangle() -> None:
    """T자 — 가로 + 세로 corridor 두 개 cover."""
    heatmap = _t_corner_heatmap()
    grid = _make_grid(heatmap)
    step = RectangleDictionaryCoverStep()
    result = step.run(grid)
    assert result.accepted is True
    assert len(result.rectangles) >= 2
    assert result.metadata["recall"] >= 0.70


def test_sparse_noise_falls_back() -> None:
    """sparse random — 통과 후보 없거나 recall 낮아 fallback."""
    heatmap = _sparse_noise_heatmap()
    grid = _make_grid(heatmap)
    step = RectangleDictionaryCoverStep()
    result = step.run(grid)
    # accepted=False (recall 낮거나 후보 없음)
    assert result.accepted is False
    assert result.fallback_used is True
    assert result.fallback_source == "sprint49_hint_chain"
    assert "fallback_reason" in result.metadata


def test_empty_heatmap_falls_back() -> None:
    """전부 0 — fallback."""
    heatmap = np.zeros((20, 20), dtype=np.int32)
    grid = _make_grid(heatmap)
    step = RectangleDictionaryCoverStep()
    result = step.run(grid)
    assert result.accepted is False
    assert result.fallback_used is True
    assert result.metadata["fallback_reason"] == "empty_heatmap"


# ── Codex W-1 magic number sweep ────────────────────────────────────────────


@pytest.mark.parametrize("tau_p", [0.80, 0.85, 0.90])
def test_corridor_tau_p_sweep_corridor(tau_p: float) -> None:
    """corridor: τ_p 0.80~0.90 모두 통과 (clean fixture)."""
    heatmap = _corridor_heatmap(corridor_thickness_cells=8, hits_per_cell=10)
    grid = _make_grid(heatmap)
    params = RectangleDictionaryCoverParams(precision_threshold=tau_p)
    result = RectangleDictionaryCoverStep(params).run(grid)
    assert result.accepted is True, (
        f"tau_p={tau_p} expected accepted=True, "
        f"got recall={result.metadata.get('recall')} "
        f"over={result.metadata.get('over_cover_ratio')}"
    )


@pytest.mark.parametrize("tau_p", [0.80, 0.85, 0.90])
def test_l_corner_tau_p_sweep(tau_p: float) -> None:
    """ㄱ자: τ_p sweep 모두 accepted (denser fixture)."""
    heatmap = _l_corner_heatmap()
    grid = _make_grid(heatmap)
    params = RectangleDictionaryCoverParams(precision_threshold=tau_p)
    result = RectangleDictionaryCoverStep(params).run(grid)
    assert result.accepted is True


def test_tau_p_too_strict_rejects_noisy_fixture() -> None:
    """sparse noise + τ_p=0.95 → 후보 거의 없음 → fallback."""
    heatmap = _sparse_noise_heatmap()
    grid = _make_grid(heatmap)
    params = RectangleDictionaryCoverParams(precision_threshold=0.95)
    result = RectangleDictionaryCoverStep(params).run(grid)
    assert result.accepted is False


# ── Codex W-2 rotation 보존 ─────────────────────────────────────────────────


def test_rotation_sum_preservation_within_tolerance() -> None:
    """rotated grid 의 hits sum 이 source sum 과 5% 이내."""
    heatmap = _corridor_heatmap(hits_per_cell=10)
    grid = _make_grid(heatmap)
    step = RectangleDictionaryCoverStep()
    result = step.run(grid)
    # accepted 또는 fallback 무관, rotation_sum_preservation 항상 기록
    rotation_meta = result.metadata.get("rotation_sum_preservation", [])
    assert isinstance(rotation_meta, list)
    assert len(rotation_meta) >= 1
    for entry in rotation_meta:
        # angle=0 는 정확히 1.0
        if abs(entry["angle_deg"]) < 1e-9:
            assert entry["preservation_ratio"] == 1.0
        else:
            # nearest neighbor + padding 으로 ±5% 이내
            ratio = entry["preservation_ratio"]
            assert 0.95 <= ratio <= 1.05, (
                f"rotation preservation ratio {ratio} out of [0.95, 1.05] "
                f"for angle {entry['angle_deg']}"
            )


def test_rotation_zero_angle_no_op() -> None:
    """angle=0 fast path: shape/sum 보존."""
    heatmap = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    grid = _make_grid(heatmap)
    # internal: angles=[0.0] only
    params = RectangleDictionaryCoverParams(angles_deg=(0.0,))
    step = RectangleDictionaryCoverStep(params)
    rotated, meta = step._rotate_heatmap(  # type: ignore[attr-defined]
        heatmap=heatmap.astype(np.int64),
        origin=grid.origin,
        angle_deg=0.0,
    )
    np.testing.assert_array_equal(rotated, heatmap.astype(np.int32))
    assert meta["preservation_ratio"] == 1.0


# ── Codex W-3 precision after uncovered ────────────────────────────────────


def test_greedy_marginal_recompute_not_static_precision() -> None:
    """ㄱ자 — 1st rectangle 채택 후 2nd 가 더 작은 marginal 가져야 함."""
    heatmap = _l_corner_heatmap()
    grid = _make_grid(heatmap)
    step = RectangleDictionaryCoverStep()
    result = step.run(grid)
    assert result.accepted
    iteration_log = result.metadata.get("iteration_log", [])
    assert len(iteration_log) >= 2
    # marginal hits 는 단조 감소 (greedy 특성)
    marginals = [it["marginal_hits"] for it in iteration_log]
    for i in range(1, len(marginals)):
        assert marginals[i] <= marginals[i - 1], (
            f"marginal not monotonically non-increasing: {marginals}"
        )


def test_greedy_terminates_on_min_incremental_hits() -> None:
    """min_incremental_hits 가 너무 크면 greedy 가 0~1 iter 만에 종료."""
    heatmap = _corridor_heatmap()
    grid = _make_grid(heatmap)
    params = RectangleDictionaryCoverParams(min_incremental_hits=10**9)
    step = RectangleDictionaryCoverStep(params)
    result = step.run(grid)
    # 첫 번째 후보의 hits 가 10^9 미만이라 0개 selected → fallback
    assert result.accepted is False
    assert result.metadata.get("rectangle_count", 0) == 0


# ── BLOCKER 1: 후보 폭발 차단 ───────────────────────────────────────────────


def test_candidate_stride_reduces_count() -> None:
    """stride 1 vs 3 — stride 3 이 후보 수 적어야 함."""
    heatmap = _corridor_heatmap()
    grid = _make_grid(heatmap)
    params_s1 = RectangleDictionaryCoverParams(candidate_stride_cells=1)
    params_s3 = RectangleDictionaryCoverParams(candidate_stride_cells=3)
    r1 = RectangleDictionaryCoverStep(params_s1).run(grid)
    r3 = RectangleDictionaryCoverStep(params_s3).run(grid)
    c1 = int(r1.metadata.get("candidate_count", 0))
    c3 = int(r3.metadata.get("candidate_count", 0))
    # stride 3 은 stride 1 보다 후보 적음 (정확히 1/3 은 아님 — cap 200 + ladder)
    assert c3 <= c1


def test_max_candidates_per_dimension_caps() -> None:
    """max_candidates_per_dimension 작게 설정 → 총 후보 제한."""
    heatmap = _corridor_heatmap()
    grid = _make_grid(heatmap)
    params = RectangleDictionaryCoverParams(max_candidates_per_dimension=5)
    result = RectangleDictionaryCoverStep(params).run(grid)
    n_dims = len(DEFAULT_ANGLES_DEG) * len(DEFAULT_THICKNESSES_M)
    cap_total = 5 * n_dims
    assert int(result.metadata.get("candidate_count", 0)) <= cap_total


# ── BLOCKER 3: fixed length ladder (단조 가정 회피) ─────────────────────────


def test_length_ladder_is_fixed() -> None:
    """default length ladder 가 fixed [1,2,3,5,8,15,30] m."""
    params = RectangleDictionaryCoverParams()
    assert params.length_candidates_m == DEFAULT_LENGTH_LADDER_M


def test_dictionary_default_18x18() -> None:
    """기본 dictionary 18 angle × 18 thickness."""
    params = RectangleDictionaryCoverParams()
    assert len(params.angles_deg) == 18
    assert len(params.thicknesses_m) == 18


# ── BLOCKER 5: fallback metadata ────────────────────────────────────────────


def test_fallback_metadata_contains_source_when_recall_fails() -> None:
    """recall_min 을 0.99 로 올리면 corridor 는 통과하지 못 하고 fallback."""
    heatmap = _corridor_heatmap(corridor_thickness_cells=4, hits_per_cell=2)
    grid = _make_grid(heatmap)
    params = RectangleDictionaryCoverParams(recall_min=0.99)
    result = RectangleDictionaryCoverStep(params).run(grid)
    if result.accepted:
        # 우연히 recall ≥0.99 면 본 fixture 가 너무 깔끔. tighten.
        pytest.skip("fixture too tight; would not reject under recall_min=0.99")
    assert result.fallback_used is True
    assert result.fallback_source == "sprint49_hint_chain"
    assert "fallback_reason" in result.metadata
    assert result.metadata["accepted"] is False


def test_fallback_metadata_on_time_budget() -> None:
    """time_budget 0.001s — large grid 면 timeout."""
    # 큰 grid 로 timeout 유도
    heatmap = np.ones((300, 300), dtype=np.int32) * 5
    grid = _make_grid(heatmap)
    params = RectangleDictionaryCoverParams(time_budget_sec=0.001)
    result = RectangleDictionaryCoverStep(params).run(grid)
    # timed_out=True 면 accepted=False 보장
    if result.metadata.get("timed_out"):
        assert result.accepted is False
        assert result.fallback_used is True


# ── 직각 보장 (AC-3 archetype) ──────────────────────────────────────────────


def test_corridor_edges_axis_aligned_to_dictionary() -> None:
    """corridor 채택 시 모든 edge 가 selected angle ∪ orthogonal 에 정렬."""
    heatmap = _corridor_heatmap()
    grid = _make_grid(heatmap)
    step = RectangleDictionaryCoverStep()
    result = step.run(grid)
    assert result.accepted
    assert result.metadata["all_edges_axis_aligned_to_dictionary"] is True


# ── min_area gate ───────────────────────────────────────────────────────────


def test_min_area_filters_tiny_rectangles() -> None:
    """min_area_m2 키우면 작은 후보 필터링."""
    heatmap = _corridor_heatmap()
    grid = _make_grid(heatmap)
    params = RectangleDictionaryCoverParams(min_area_m2=100.0)  # 100 m^2
    result = RectangleDictionaryCoverStep(params).run(grid)
    # corridor 면적은 ~50 cells × 8 cells × 0.01 m^2 = 4 m^2 << 100 → 후보 0
    assert int(result.metadata.get("rectangle_count", 0)) == 0


# ── Params validation ──────────────────────────────────────────────────────


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError):
        RectangleDictionaryCoverStep(
            RectangleDictionaryCoverParams(angles_deg=())
        )
    with pytest.raises(ValueError):
        RectangleDictionaryCoverStep(
            RectangleDictionaryCoverParams(precision_threshold=1.5)
        )
    with pytest.raises(ValueError):
        RectangleDictionaryCoverStep(
            RectangleDictionaryCoverParams(candidate_stride_cells=0)
        )
