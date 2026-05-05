"""Sprint 52 — HoughFallbackStep unit tests."""
from __future__ import annotations

import numpy as np

from indoor_server.application.building.steps.wall_polygon.density import (
    DensityRefined,
)
from indoor_server.application.building.steps.wall_polygon.hough_fallback import (
    HoughFallbackParams,
    HoughFallbackStep,
)
from indoor_server.application.building.steps.wall_polygon.obstacle_source import (
    ObstacleHeatmap,
)


def _heatmap_from_mask(mask: np.ndarray) -> ObstacleHeatmap:
    return ObstacleHeatmap(
        counts=mask.astype(np.int32),
        origin_x=0.0,
        origin_y=0.0,
        cell_size_m=0.10,
        z0=0.0,
        height_min_m=0.30,
        height_max_m=2.50,
        metadata={"world_obstacle_point_count": int(mask.sum()), "params": {}},
    )


def _refined(mask: np.ndarray) -> DensityRefined:
    return DensityRefined(
        occupancy_mask=mask.astype(bool),
        metadata={"cells_after_morph": int(mask.sum()), "params": {}},
    )


def test_straight_band_yields_at_least_one_line() -> None:
    mask = np.zeros((40, 60), dtype=bool)
    mask[18:23, 5:55] = True
    step = HoughFallbackStep()
    result = step.run(_refined(mask), heatmap=_heatmap_from_mask(mask))
    assert result.raw_hough_count >= 1
    assert len(result.segments) >= 1


def test_band_with_small_gap_recovered_via_max_line_gap() -> None:
    mask = np.zeros((40, 80), dtype=bool)
    mask[18:23, 5:35] = True
    mask[18:23, 38:75] = True  # 3-cell gap, within max_line_gap_cells default
    step = HoughFallbackStep(
        HoughFallbackParams(max_line_gap_cells=5.0, threshold_votes=20)
    )
    result = step.run(_refined(mask), heatmap=_heatmap_from_mask(mask))
    # at least one line spans across the gap (or two lines colinear).
    assert len(result.segments) >= 1


def test_short_line_rejected() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[18:23, 18:22] = True  # very short
    step = HoughFallbackStep(HoughFallbackParams(min_length_m=2.0))
    result = step.run(_refined(mask), heatmap=_heatmap_from_mask(mask))
    assert all(seg.length_m >= 2.0 for seg in result.segments) or (
        len(result.segments) == 0
    )


def test_pure_noise_dots_yields_no_lines() -> None:
    rng = np.random.default_rng(0)
    mask = (rng.uniform(0, 1, (40, 40)) < 0.005)
    step = HoughFallbackStep(HoughFallbackParams(threshold_votes=80))
    result = step.run(_refined(mask), heatmap=_heatmap_from_mask(mask))
    assert len(result.segments) == 0


def test_max_lines_cap_enforced() -> None:
    """Force many candidate lines, then cap them."""
    mask = np.zeros((60, 80), dtype=bool)
    # multiple parallel horizontal bands
    for r in range(5, 55, 5):
        mask[r:r + 2, 5:75] = True
    step = HoughFallbackStep(
        HoughFallbackParams(max_lines=2, threshold_votes=15, min_line_length_cells=8.0)
    )
    result = step.run(_refined(mask), heatmap=_heatmap_from_mask(mask))
    assert len(result.segments) <= 2
    if result.raw_hough_count > 2:
        assert result.rejected_overflow_count >= 1
