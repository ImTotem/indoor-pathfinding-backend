"""Guard RTAB-Map trajectory road grids with semantic floor/object evidence."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from indoor_server.domain.building.models import WalkableGrid


@dataclass(frozen=True)
class RtabmapFloorGuardResult:
    grid: WalkableGrid
    metadata: dict[str, object]


class RtabmapFloorGuardStep:
    """Apply hard floor support and non-walkable subtraction to a road grid.

    The RTAB-Map trajectory road grid is a map-style corridor hypothesis. This
    step prevents sparse furniture/wall evidence from widening or surviving as
    walkable road cells by treating projected floor confidence as support and
    projected wall/object evidence as forbidden cells.
    """

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.05,
        avoid_threshold: float = 0.05,
        floor_support_dilation_m: float = 1.6,
        avoid_dilation_m: float = 0.15,
        min_floor_support_cells: int = 8,
        min_retained_ratio_after_floor_support: float = 0.35,
        keep_largest_component: bool = True,
    ) -> None:
        if confidence_threshold < 0.0:
            raise ValueError("confidence_threshold must be >= 0")
        if avoid_threshold < 0.0:
            raise ValueError("avoid_threshold must be >= 0")
        if floor_support_dilation_m < 0.0:
            raise ValueError("floor_support_dilation_m must be >= 0")
        if avoid_dilation_m < 0.0:
            raise ValueError("avoid_dilation_m must be >= 0")
        if min_floor_support_cells < 0:
            raise ValueError("min_floor_support_cells must be >= 0")
        if not (0.0 <= min_retained_ratio_after_floor_support <= 1.0):
            raise ValueError("min_retained_ratio_after_floor_support must be in [0, 1]")
        self._confidence_threshold = confidence_threshold
        self._avoid_threshold = avoid_threshold
        self._floor_support_dilation_m = floor_support_dilation_m
        self._avoid_dilation_m = avoid_dilation_m
        self._min_floor_support_cells = min_floor_support_cells
        self._min_retained_ratio_after_floor_support = (
            min_retained_ratio_after_floor_support
        )
        self._keep_largest_component = keep_largest_component

    def run(
        self,
        grid: WalkableGrid,
        *,
        confidence: np.ndarray | None = None,
        avoid: np.ndarray | None = None,
    ) -> RtabmapFloorGuardResult:
        mask = np.asarray(grid.mask, dtype=bool).copy()
        input_cells = int(mask.sum())
        if input_cells == 0:
            return RtabmapFloorGuardResult(
                grid=grid,
                metadata={
                    "source": "rtabmap_floor_guard",
                    "input_cells": 0,
                    "output_cells": 0,
                    "floor_support_used": False,
                    "avoid_subtraction_used": False,
                    "issues": ["empty_input_grid"],
                },
            )

        issues: list[str] = []
        support_used = False
        support_cells = 0
        support_dilated_cells = 0
        if confidence is not None and confidence.shape == mask.shape:
            support = np.asarray(confidence > self._confidence_threshold, dtype=bool)
            support_cells = int(support.sum())
            if support_cells >= self._min_floor_support_cells:
                support = _dilate(
                    support,
                    radius_cells=_radius_cells(
                        self._floor_support_dilation_m,
                        grid.origin.cell_size,
                    ),
                )
                support_dilated_cells = int(support.sum())
                supported_mask = mask & support
                supported_cells = int(supported_mask.sum())
                min_cells = int(round(
                    input_cells * self._min_retained_ratio_after_floor_support,
                ))
                if supported_cells >= min_cells:
                    mask = supported_mask
                    support_used = True
                else:
                    issues.append(
                        "floor_support_retained_below_min:"
                        f"{supported_cells}/{input_cells}",
                    )
            else:
                issues.append(f"floor_support_below_min:{support_cells}")
        elif confidence is not None:
            issues.append("confidence_shape_mismatch")

        avoid_used = False
        forbidden_cells = 0
        forbidden_dilated_cells = 0
        if avoid is not None and avoid.shape == mask.shape:
            forbidden = np.asarray(avoid > self._avoid_threshold, dtype=bool)
            forbidden_cells = int(forbidden.sum())
            if forbidden_cells:
                forbidden = _dilate(
                    forbidden,
                    radius_cells=_radius_cells(
                        self._avoid_dilation_m,
                        grid.origin.cell_size,
                    ),
                )
                forbidden_dilated_cells = int(forbidden.sum())
                mask &= ~forbidden
                avoid_used = True
        elif avoid is not None:
            issues.append("avoid_shape_mismatch")

        before_largest = int(mask.sum())
        component_count = 0
        largest_component_cells = 0
        if self._keep_largest_component and before_largest:
            mask, component_count, largest_component_cells = _largest_component(mask)

        observation_count = np.where(mask, grid.observation_count, 0).astype(
            grid.observation_count.dtype,
            copy=False,
        )
        guarded = WalkableGrid(
            origin=grid.origin,
            mask=mask,
            observation_count=observation_count,
        )
        output_cells = int(mask.sum())
        return RtabmapFloorGuardResult(
            grid=guarded,
            metadata={
                "source": "rtabmap_floor_guard",
                "input_cells": input_cells,
                "output_cells": output_cells,
                "retained_ratio": (
                    float(output_cells / input_cells)
                    if input_cells
                    else 0.0
                ),
                "floor_support_used": support_used,
                "floor_support_cells": support_cells,
                "floor_support_dilated_cells": support_dilated_cells,
                "floor_support_dilation_m": self._floor_support_dilation_m,
                "confidence_threshold": self._confidence_threshold,
                "min_retained_ratio_after_floor_support": (
                    self._min_retained_ratio_after_floor_support
                ),
                "avoid_subtraction_used": avoid_used,
                "avoid_cells": forbidden_cells,
                "avoid_dilated_cells": forbidden_dilated_cells,
                "avoid_dilation_m": self._avoid_dilation_m,
                "avoid_threshold": self._avoid_threshold,
                "cells_before_largest_component": before_largest,
                "component_count": component_count,
                "largest_component_cells": largest_component_cells,
                "keep_largest_component": self._keep_largest_component,
                "issues": issues,
            },
        )


def _radius_cells(distance_m: float, cell_size_m: float) -> int:
    if distance_m <= 0.0 or cell_size_m <= 0.0:
        return 0
    return int(np.ceil(distance_m / cell_size_m))


def _dilate(mask: np.ndarray, *, radius_cells: int) -> np.ndarray:
    if radius_cells <= 0 or not mask.any():
        return mask
    import cv2

    size = radius_cells * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return np.asarray(dilated > 0, dtype=bool)


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    import cv2

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    if component_count <= 1:
        return mask, 0, int(mask.sum())

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas) + 1)
    largest = labels == largest_label
    return largest, int(component_count - 1), int(areas[largest_label - 1])
