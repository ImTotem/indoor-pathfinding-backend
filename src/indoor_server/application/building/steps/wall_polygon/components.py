"""Sprint 51 — Step 2: component split (connected components + min area filter).

DEPRECATED in Sprint 52: facade no longer invokes this step. Replaced by
`skeleton_split.SkeletonSplitStep` (primary) + `hough_fallback.HoughFallbackStep`
(secondary). This module is kept for rollback safety and as a sentinel for
the Sprint 51 PCA path; planned for removal in Sprint 53+ cleanup. The
existing unit tests under `tests/unit/building/wall_polygon/test_components.py`
continue to pass and protect the dead path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from indoor_server.application.building.steps.wall_polygon.density import DensityRefined


@dataclass(frozen=True)
class ComponentSplitParams:
    min_area_cells: int = 8
    max_components: int = 64
    connectivity: int = 8  # 4 or 8

    def to_metadata(self) -> dict[str, object]:
        return {
            "min_area_cells": self.min_area_cells,
            "max_components": self.max_components,
            "connectivity": self.connectivity,
        }


@dataclass(frozen=True)
class WallFragment:
    component_id: int
    pixel_count: int
    bbox_rows: tuple[int, int]
    bbox_cols: tuple[int, int]
    pixels_rc: np.ndarray  # (N, 2) int32 (row, col)


@dataclass(frozen=True)
class FragmentSet:
    fragments: list[WallFragment]
    label_map: np.ndarray  # (H, W) int32, 0=background
    metadata: dict[str, object] = field(default_factory=dict)


class ComponentSplitStep:
    """Step 2 — connected component labeling + min area filter."""

    def __init__(self, params: ComponentSplitParams | None = None) -> None:
        self._params = params if params is not None else ComponentSplitParams()
        self._validate()

    def _validate(self) -> None:
        p = self._params
        if p.min_area_cells < 1:
            raise ValueError("min_area_cells must be >= 1")
        if p.max_components < 1:
            raise ValueError("max_components must be >= 1")
        if p.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")

    def run(self, refined: DensityRefined) -> FragmentSet:
        params = self._params
        mask = np.asarray(refined.occupancy_mask, dtype=bool)
        if params.connectivity == 4:
            structure = ndimage.generate_binary_structure(2, 1)
        else:
            structure = ndimage.generate_binary_structure(2, 2)

        labels_raw, num = ndimage.label(mask, structure=structure)
        labels_raw = np.asarray(labels_raw, dtype=np.int32)

        if num == 0:
            empty_label = np.zeros_like(mask, dtype=np.int32)
            return FragmentSet(
                fragments=[],
                label_map=empty_label,
                metadata={
                    "total_components_before_filter": 0,
                    "components_kept": 0,
                    "components_dropped_by_area": 0,
                    "largest_pixel_count": 0,
                    "params": params.to_metadata(),
                },
            )

        # measure component sizes — labels 1..num
        sizes = ndimage.sum(mask, labels_raw, index=range(1, num + 1))
        sizes = np.asarray(sizes, dtype=np.int64)

        # sort by size desc, keep up to max_components AND area >= min_area_cells
        order = np.argsort(sizes)[::-1]  # largest first
        kept_labels: list[int] = []
        for raw_idx in order:
            label_id = int(raw_idx) + 1  # ndimage labels are 1-indexed
            size = int(sizes[raw_idx])
            if size < params.min_area_cells:
                break
            if len(kept_labels) >= params.max_components:
                break
            kept_labels.append(label_id)

        # remap kept labels into compact ids 1..K
        new_label_map = np.zeros_like(labels_raw, dtype=np.int32)
        fragments: list[WallFragment] = []
        for new_id, old_id in enumerate(kept_labels, start=1):
            ys, xs = np.where(labels_raw == old_id)
            if ys.size == 0:
                continue
            new_label_map[ys, xs] = new_id
            pixels_rc = np.column_stack([ys, xs]).astype(np.int32)
            fragments.append(
                WallFragment(
                    component_id=new_id,
                    pixel_count=int(ys.size),
                    bbox_rows=(int(ys.min()), int(ys.max())),
                    bbox_cols=(int(xs.min()), int(xs.max())),
                    pixels_rc=pixels_rc,
                )
            )

        components_dropped = max(0, int(num) - len(kept_labels))
        largest = int(sizes.max()) if sizes.size > 0 else 0

        metadata: dict[str, object] = {
            "total_components_before_filter": int(num),
            "components_kept": len(kept_labels),
            "components_dropped_by_area": components_dropped,
            "largest_pixel_count": largest,
            "params": params.to_metadata(),
        }
        return FragmentSet(
            fragments=fragments,
            label_map=new_label_map,
            metadata=metadata,
        )
