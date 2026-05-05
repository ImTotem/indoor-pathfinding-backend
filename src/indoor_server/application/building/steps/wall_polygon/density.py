"""Sprint 51 — Step 1: density refinement.

Obstacle hit count grid → binary occupancy mask.
sequence: count threshold → median filter → opening → closing.
scipy.ndimage 만 사용 (cv2 의존성 없이 동일 결과).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from indoor_server.application.building.steps.wall_polygon.obstacle_source import (
    ObstacleHeatmap,
)


@dataclass(frozen=True)
class DensityRefineParams:
    """Density refinement tunables."""

    min_cell_hits: int = 4
    median_filter_size_cells: int = 3
    morph_open_radius_cells: int = 1
    morph_close_radius_cells: int = 1

    def to_metadata(self) -> dict[str, object]:
        return {
            "min_cell_hits": self.min_cell_hits,
            "median_filter_size_cells": self.median_filter_size_cells,
            "morph_open_radius_cells": self.morph_open_radius_cells,
            "morph_close_radius_cells": self.morph_close_radius_cells,
        }


@dataclass(frozen=True)
class DensityRefined:
    """Step 1 output."""

    occupancy_mask: np.ndarray  # (H, W) bool
    metadata: dict[str, object] = field(default_factory=dict)


def _disk_kernel(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    size = 2 * radius + 1
    yy, xx = np.ogrid[:size, :size]
    cy = cx = radius
    kernel = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (radius * radius)
    return np.asarray(kernel, dtype=bool)


class DensityRefineStep:
    """Step 1 — density refinement."""

    def __init__(self, params: DensityRefineParams | None = None) -> None:
        self._params = params if params is not None else DensityRefineParams()
        self._validate()

    def _validate(self) -> None:
        p = self._params
        if p.min_cell_hits < 1:
            raise ValueError("min_cell_hits must be >= 1")
        if p.median_filter_size_cells < 1:
            raise ValueError("median_filter_size_cells must be >= 1")
        if p.morph_open_radius_cells < 0:
            raise ValueError("morph_open_radius_cells must be >= 0")
        if p.morph_close_radius_cells < 0:
            raise ValueError("morph_close_radius_cells must be >= 0")

    def run(self, heatmap: ObstacleHeatmap) -> DensityRefined:
        params = self._params
        counts = np.asarray(heatmap.counts, dtype=np.int32)
        cells_before = int(np.count_nonzero(counts))

        # 1. count threshold
        mask = counts >= params.min_cell_hits
        cells_after_threshold = int(np.count_nonzero(mask))

        # 2. median filter (preserves walls but kills isolated noise)
        if params.median_filter_size_cells >= 2:
            mask_u8 = mask.astype(np.uint8)
            filtered = ndimage.median_filter(
                mask_u8, size=params.median_filter_size_cells
            )
            mask = filtered.astype(bool)
        cells_after_median = int(np.count_nonzero(mask))

        # 3. morphology open → close
        if params.morph_open_radius_cells > 0:
            kernel = _disk_kernel(params.morph_open_radius_cells)
            mask = ndimage.binary_opening(mask, structure=kernel)
        if params.morph_close_radius_cells > 0:
            kernel = _disk_kernel(params.morph_close_radius_cells)
            mask = ndimage.binary_closing(mask, structure=kernel)
        cells_after_morph = int(np.count_nonzero(mask))

        metadata: dict[str, object] = {
            "cells_before": cells_before,
            "cells_after_threshold": cells_after_threshold,
            "cells_after_median": cells_after_median,
            "cells_after_morph": cells_after_morph,
            "params": params.to_metadata(),
        }
        return DensityRefined(
            occupancy_mask=mask.astype(bool),
            metadata=metadata,
        )
