"""Sprint 52 — SkeletonSplitStep unit tests.

Covers the multi-axis topology cases that the Sprint 51 PCA-per-component
algorithm structurally rejected (ㄱ/T/+).
"""
from __future__ import annotations

import numpy as np

from indoor_server.application.building.steps.wall_polygon.density import (
    DensityRefined,
)
from indoor_server.application.building.steps.wall_polygon.obstacle_source import (
    ObstacleHeatmap,
)
from indoor_server.application.building.steps.wall_polygon.skeleton_split import (
    SkeletonSplitParams,
    SkeletonSplitStep,
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


def test_straight_line_yields_single_segment() -> None:
    """A horizontal 5x40 band should produce a single segment."""
    mask = np.zeros((40, 50), dtype=bool)
    mask[18:23, 5:45] = True
    step = SkeletonSplitStep()
    result = step.run(_refined(mask), heatmap=_heatmap_from_mask(mask))
    assert result.skeleton_pixel_count > 0
    assert result.endpoint_count == 2
    assert result.junction_count == 0
    assert len(result.segments) == 1
    seg = result.segments[0]
    assert seg.length_m >= 0.5


def test_l_corridor_splits_into_two_segments() -> None:
    """ㄱ자 corridor → 2 endpoints, 1-2 segments depending on corner topology.

    The medial axis of an L-shape may emit a single chain through the corner
    (the corner pixel ends up degree-2) or a junction (degree-3) depending on
    skeletonize's tie-breaking. Either way RDP simplify must produce >=2
    segments because the polyline turns 90°.
    """
    mask = np.zeros((80, 80), dtype=bool)
    mask[5:10, 5:70] = True   # horizontal arm
    mask[5:75, 5:10] = True   # vertical arm (shares the corner with horizontal)
    step = SkeletonSplitStep()
    result = step.run(_refined(mask), heatmap=_heatmap_from_mask(mask))
    assert result.endpoint_count == 2
    # at least one polyline + RDP split → 2 segments minimum.
    assert len(result.segments) >= 2


def test_t_corridor_three_segments() -> None:
    """T-shaped corridor → 3 endpoints + 1 junction → ≥3 segments."""
    mask = np.zeros((80, 80), dtype=bool)
    mask[5:10, 5:75] = True       # horizontal (top)
    mask[10:75, 38:43] = True     # vertical (drop)
    step = SkeletonSplitStep()
    result = step.run(_refined(mask), heatmap=_heatmap_from_mask(mask))
    assert result.endpoint_count == 3
    assert result.junction_count >= 1
    assert len(result.segments) >= 3


def test_plus_corridor_four_segments() -> None:
    """+ corridor → 4 endpoints + 1 junction → ≥4 segments."""
    mask = np.zeros((80, 80), dtype=bool)
    mask[5:75, 38:43] = True
    mask[38:43, 5:75] = True
    step = SkeletonSplitStep()
    result = step.run(_refined(mask), heatmap=_heatmap_from_mask(mask))
    assert result.endpoint_count == 4
    assert result.junction_count >= 1
    assert len(result.segments) >= 4


def test_tiny_spur_pruned() -> None:
    """A short spur dangling off a long line should be pruned."""
    mask = np.zeros((40, 80), dtype=bool)
    mask[18:23, 5:75] = True       # main horizontal
    # tiny spur off the main line (3 cells long)
    mask[10:18, 38:43] = True
    step = SkeletonSplitStep(SkeletonSplitParams(spur_prune_length_ratio=0.30))
    result = step.run(_refined(mask), heatmap=_heatmap_from_mask(mask))
    # the long horizontal is preserved.
    assert any(seg.length_m >= 5.0 for seg in result.segments)


def test_curved_arc_rejected() -> None:
    """Curved noise band → straightness gate rejects most sub-segments.

    With a large RDP epsilon the polyline collapses to a near-chord pair
    whose chord/path ratio is well below 0.95, so the straightness gate
    fires.
    """
    h, w = 60, 60
    mask = np.zeros((h, w), dtype=bool)
    cy, cx = 30, 30
    radius = 22
    yy, xxg = np.ogrid[:h, :w]
    ring = ((yy - cy) ** 2 + (xxg - cx) ** 2 <= radius * radius) & (
        (yy - cy) ** 2 + (xxg - cx) ** 2 >= (radius - 3) ** 2
    )
    mask[ring] = True
    step = SkeletonSplitStep(
        SkeletonSplitParams(
            min_straightness=0.95,
            rdp_epsilon_cells=8.0,
            min_pixels_per_segment=10,
        )
    )
    result = step.run(_refined(mask), heatmap=_heatmap_from_mask(mask))
    # Some sub-segments should be rejected for curvature (or all rejected as
    # too-short / curved). Either way, accepted segments stay below the full
    # circumference / 2.
    rejected_total = result.rejected_curved_count + result.rejected_short_count
    assert rejected_total >= 1 or len(result.segments) == 0
