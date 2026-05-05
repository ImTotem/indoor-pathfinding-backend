"""Sprint 52 — Step 2~3 secondary: cv2.HoughLinesP fallback.

Used when the primary skeleton split returns too few segments for the facade
gate (`min_wall_lines`). Probabilistic Hough on a slightly dilated occupancy
mask avoids the multi-axis blob limitation by treating every pixel cluster
as a Hough vote source rather than a connected component.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

from indoor_server.application.building.steps.wall_polygon.density import (
    DensityRefined,
)
from indoor_server.application.building.steps.wall_polygon.line_fit import (
    LineSegment,
    _fold_angle_deg,
)
from indoor_server.application.building.steps.wall_polygon.obstacle_source import (
    ObstacleHeatmap,
)


@dataclass(frozen=True)
class HoughFallbackParams:
    """secondary fallback parameters."""

    rho_cells: float = 1.0
    theta_deg: float = 1.0
    threshold_votes: int = 30
    min_line_length_cells: float = 5.0
    max_line_gap_cells: float = 2.0
    pre_dilate_radius_cells: int = 1
    min_length_m: float = 0.5
    max_lines: int = 200

    def to_metadata(self) -> dict[str, object]:
        return {
            "rho_cells": self.rho_cells,
            "theta_deg": self.theta_deg,
            "threshold_votes": self.threshold_votes,
            "min_line_length_cells": self.min_line_length_cells,
            "max_line_gap_cells": self.max_line_gap_cells,
            "pre_dilate_radius_cells": self.pre_dilate_radius_cells,
            "min_length_m": self.min_length_m,
            "max_lines": self.max_lines,
        }


@dataclass(frozen=True)
class HoughFallbackResult:
    segments: list[LineSegment]
    raw_hough_count: int
    rejected_short_count: int
    rejected_overflow_count: int
    metadata: dict[str, object] = field(default_factory=dict)


def _disk_kernel(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    size = 2 * radius + 1
    yy, xx = np.ogrid[:size, :size]
    cy = cx = radius
    kernel = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (radius * radius)
    return np.asarray(kernel, dtype=bool)


class HoughFallbackStep:
    """Step 2~3 secondary — probabilistic Hough line fallback."""

    def __init__(self, params: HoughFallbackParams | None = None) -> None:
        self._params = params if params is not None else HoughFallbackParams()
        self._validate()

    def _validate(self) -> None:
        p = self._params
        if p.rho_cells <= 0:
            raise ValueError("rho_cells must be > 0")
        if p.theta_deg <= 0:
            raise ValueError("theta_deg must be > 0")
        if p.threshold_votes < 1:
            raise ValueError("threshold_votes must be >= 1")
        if p.min_line_length_cells <= 0:
            raise ValueError("min_line_length_cells must be > 0")
        if p.max_line_gap_cells < 0:
            raise ValueError("max_line_gap_cells must be >= 0")
        if p.pre_dilate_radius_cells < 0:
            raise ValueError("pre_dilate_radius_cells must be >= 0")
        if p.min_length_m < 0:
            raise ValueError("min_length_m must be >= 0")
        if p.max_lines < 1:
            raise ValueError("max_lines must be >= 1")

    @property
    def params(self) -> HoughFallbackParams:
        return self._params

    def run(
        self,
        refined: DensityRefined,
        *,
        heatmap: ObstacleHeatmap,
    ) -> HoughFallbackResult:
        params = self._params
        mask = np.asarray(refined.occupancy_mask, dtype=bool)
        if not mask.any():
            return HoughFallbackResult(
                segments=[],
                raw_hough_count=0,
                rejected_short_count=0,
                rejected_overflow_count=0,
                metadata={
                    "raw_hough_count": 0,
                    "accept_count": 0,
                    "rejected_short_count": 0,
                    "rejected_overflow_count": 0,
                    "params": params.to_metadata(),
                },
            )

        if params.pre_dilate_radius_cells > 0:
            kernel = _disk_kernel(params.pre_dilate_radius_cells)
            dilated = ndimage.binary_dilation(mask, structure=kernel)
        else:
            dilated = mask
        edges = (dilated.astype(np.uint8)) * 255
        theta_rad = math.radians(params.theta_deg)
        hough = cv2.HoughLinesP(
            edges,
            rho=float(params.rho_cells),
            theta=float(theta_rad),
            threshold=int(params.threshold_votes),
            minLineLength=float(params.min_line_length_cells),
            maxLineGap=float(params.max_line_gap_cells),
        )

        cell = float(heatmap.cell_size_m)
        ox = float(heatmap.origin_x)
        oy = float(heatmap.origin_y)

        segments: list[LineSegment] = []
        raw_count = 0
        rejected_short = 0
        rejected_overflow = 0
        if hough is not None:
            raw_count = int(len(hough))
            for idx, line in enumerate(hough):
                if len(segments) >= params.max_lines:
                    rejected_overflow += 1
                    continue
                x1c, y1c, x2c, y2c = (int(v) for v in line[0])
                # convert pixel (col, row) → world
                wx1 = ox + (float(x1c) + 0.5) * cell
                wy1 = oy + (float(y1c) + 0.5) * cell
                wx2 = ox + (float(x2c) + 0.5) * cell
                wy2 = oy + (float(y2c) + 0.5) * cell
                length = float(math.hypot(wx2 - wx1, wy2 - wy1))
                if length < params.min_length_m:
                    rejected_short += 1
                    continue
                pixel_count = int(
                    max(2, math.hypot(x2c - x1c, y2c - y1c)) + 1
                )
                angle_deg = _fold_angle_deg(
                    math.degrees(math.atan2(wy2 - wy1, wx2 - wx1))
                )
                segments.append(
                    LineSegment(
                        component_id=idx,
                        x1=wx1,
                        y1=wy1,
                        x2=wx2,
                        y2=wy2,
                        length_m=length,
                        angle_deg=angle_deg,
                        linearity=1.0,
                        pixel_count=pixel_count,
                    )
                )

        metadata: dict[str, object] = {
            "raw_hough_count": int(raw_count),
            "accept_count": len(segments),
            "rejected_short_count": int(rejected_short),
            "rejected_overflow_count": int(rejected_overflow),
            "params": params.to_metadata(),
        }
        return HoughFallbackResult(
            segments=segments,
            raw_hough_count=int(raw_count),
            rejected_short_count=int(rejected_short),
            rejected_overflow_count=int(rejected_overflow),
            metadata=metadata,
        )
