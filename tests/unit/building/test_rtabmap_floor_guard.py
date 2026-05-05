from __future__ import annotations

import numpy as np

from indoor_server.application.building.steps.rtabmap_floor_guard import (
    RtabmapFloorGuardStep,
)
from indoor_server.domain.building.models import GridOrigin, WalkableGrid


def _grid(mask: np.ndarray) -> WalkableGrid:
    return WalkableGrid(
        origin=GridOrigin(
            x0=0.0,
            y0=0.0,
            z0=0.0,
            cell_size=0.1,
            w=mask.shape[1],
            h=mask.shape[0],
        ),
        mask=mask.astype(bool),
        observation_count=mask.astype(np.uint16),
    )


def test_floor_guard_intersects_with_dilated_floor_support() -> None:
    mask = np.ones((9, 9), dtype=bool)
    confidence = np.zeros((9, 9), dtype=np.float32)
    confidence[4, 4] = 1.0

    result = RtabmapFloorGuardStep(
        floor_support_dilation_m=0.2,
        min_floor_support_cells=1,
        min_retained_ratio_after_floor_support=0.0,
        keep_largest_component=False,
    ).run(_grid(mask), confidence=confidence)

    assert int(result.grid.mask.sum()) < int(mask.sum())
    assert result.grid.mask[4, 4]
    assert not result.grid.mask[0, 0]
    assert result.metadata["floor_support_used"] is True


def test_floor_guard_hard_subtracts_avoid_cells() -> None:
    mask = np.ones((7, 7), dtype=bool)
    confidence = np.ones((7, 7), dtype=np.float32)
    avoid = np.zeros((7, 7), dtype=np.float32)
    avoid[3, 3] = 1.0

    result = RtabmapFloorGuardStep(
        floor_support_dilation_m=0.0,
        avoid_dilation_m=0.0,
        keep_largest_component=False,
    ).run(_grid(mask), confidence=confidence, avoid=avoid)

    assert not result.grid.mask[3, 3]
    assert result.grid.mask[3, 2]
    assert result.metadata["avoid_subtraction_used"] is True
    assert result.metadata["avoid_cells"] == 1


def test_floor_guard_skips_floor_support_when_too_sparse() -> None:
    mask = np.ones((5, 5), dtype=bool)
    confidence = np.zeros((5, 5), dtype=np.float32)
    confidence[2, 2] = 1.0

    result = RtabmapFloorGuardStep(
        min_floor_support_cells=2,
        avoid_dilation_m=0.0,
        keep_largest_component=False,
    ).run(_grid(mask), confidence=confidence)

    assert int(result.grid.mask.sum()) == int(mask.sum())
    assert result.metadata["floor_support_used"] is False
    assert result.metadata["issues"] == ["floor_support_below_min:1"]


def test_floor_guard_skips_floor_support_when_it_would_overcrop() -> None:
    mask = np.ones((10, 10), dtype=bool)
    confidence = np.zeros((10, 10), dtype=np.float32)
    confidence[0, 0] = 1.0

    result = RtabmapFloorGuardStep(
        floor_support_dilation_m=0.1,
        min_floor_support_cells=1,
        min_retained_ratio_after_floor_support=0.5,
        keep_largest_component=False,
    ).run(_grid(mask), confidence=confidence)

    assert int(result.grid.mask.sum()) == int(mask.sum())
    assert result.metadata["floor_support_used"] is False
    assert result.metadata["issues"] == ["floor_support_retained_below_min:3/100"]
