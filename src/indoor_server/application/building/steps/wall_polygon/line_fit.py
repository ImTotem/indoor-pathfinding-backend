"""Sprint 51 — Step 3: per-component PCA line fitting.

DEPRECATED in Sprint 52: facade no longer invokes `LineFitStep`. The
`LineSegment` / `LineSegmentSet` dataclasses and `_fold_angle_deg` helper
are still imported by Sprint 52 modules (`skeleton_split`, `hough_fallback`)
because the downstream Step 4~7 schema is unchanged. The PCA-per-component
algorithm itself is dead path — kept for rollback / Sprint 53+ cleanup.

linearity = lambda_max / (lambda_max + lambda_min). >= min_linearity 채택.
주축 vector v 위 projection 으로 (x1,y1)..(x2,y2) endpoint 추출.
blob 판정: linearity 와 bbox aspect ratio 둘 다 검사 (이중 검증).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from indoor_server.application.building.steps.wall_polygon.components import (
    FragmentSet,
)
from indoor_server.application.building.steps.wall_polygon.obstacle_source import (
    ObstacleHeatmap,
)


@dataclass(frozen=True)
class LineFitParams:
    min_pixels: int = 8
    min_linearity: float = 0.85
    min_length_m: float = 0.5
    blob_reject_aspect: float = 1.5

    def to_metadata(self) -> dict[str, object]:
        return {
            "min_pixels": self.min_pixels,
            "min_linearity": self.min_linearity,
            "min_length_m": self.min_length_m,
            "blob_reject_aspect": self.blob_reject_aspect,
        }


@dataclass(frozen=True)
class LineSegment:
    component_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    length_m: float
    angle_deg: float  # (-90, 90]
    linearity: float
    pixel_count: int


@dataclass(frozen=True)
class LineSegmentSet:
    segments: list[LineSegment]
    rejected_blob_count: int
    rejected_short_count: int
    metadata: dict[str, object] = field(default_factory=dict)


def _fold_angle_deg(angle_deg: float) -> float:
    """Fold to (-90, 90]."""
    a = float(angle_deg)
    while a > 90.0:
        a -= 180.0
    while a <= -90.0:
        a += 180.0
    return a


class LineFitStep:
    """Step 3 — PCA line fit per component."""

    def __init__(self, params: LineFitParams | None = None) -> None:
        self._params = params if params is not None else LineFitParams()
        self._validate()

    def _validate(self) -> None:
        p = self._params
        if p.min_pixels < 2:
            raise ValueError("min_pixels must be >= 2")
        if not (0.0 < p.min_linearity <= 1.0):
            raise ValueError("min_linearity must be in (0, 1]")
        if p.min_length_m < 0:
            raise ValueError("min_length_m must be >= 0")
        if p.blob_reject_aspect <= 0:
            raise ValueError("blob_reject_aspect must be > 0")

    def run(
        self, fragments: FragmentSet, *, heatmap: ObstacleHeatmap
    ) -> LineSegmentSet:
        params = self._params
        cell = float(heatmap.cell_size_m)
        ox = float(heatmap.origin_x)
        oy = float(heatmap.origin_y)

        accepted: list[LineSegment] = []
        reject_blob = 0
        reject_short = 0
        linearity_values: list[float] = []
        length_values: list[float] = []

        for frag in fragments.fragments:
            if frag.pixel_count < params.min_pixels:
                reject_blob += 1
                continue

            # convert pixels (row, col) to world (x, y) using cell centers.
            rc = frag.pixels_rc.astype(np.float64)
            ys = rc[:, 0]
            xs = rc[:, 1]
            world_x = ox + (xs + 0.5) * cell
            world_y = oy + (ys + 0.5) * cell
            pts = np.column_stack([world_x, world_y])

            # PCA via covariance matrix
            mean = pts.mean(axis=0)
            centered = pts - mean
            cov = np.cov(centered.T)
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
            except np.linalg.LinAlgError:
                reject_blob += 1
                continue
            # eigh returns ascending; max idx = -1
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            lam_max = float(max(eigvals[0], 0.0))
            lam_min = float(max(eigvals[1], 0.0))
            denom = lam_max + lam_min
            if denom <= 1e-12:
                reject_blob += 1
                continue
            linearity = lam_max / denom

            # bbox aspect (in world units)
            row_lo, row_hi = frag.bbox_rows
            col_lo, col_hi = frag.bbox_cols
            bbox_h = (row_hi - row_lo + 1) * cell
            bbox_w = (col_hi - col_lo + 1) * cell
            long_side = max(bbox_h, bbox_w)
            short_side = max(min(bbox_h, bbox_w), 1e-9)
            aspect = long_side / short_side

            if (
                linearity < params.min_linearity
                or aspect < params.blob_reject_aspect
            ):
                reject_blob += 1
                continue

            v = eigvecs[:, 0]  # principal axis unit vector
            proj = centered @ v  # 1D projection
            t_min = float(proj.min())
            t_max = float(proj.max())
            x1 = float(mean[0] + t_min * v[0])
            y1 = float(mean[1] + t_min * v[1])
            x2 = float(mean[0] + t_max * v[0])
            y2 = float(mean[1] + t_max * v[1])
            length = float(math.hypot(x2 - x1, y2 - y1))
            if length < params.min_length_m:
                reject_short += 1
                continue
            angle_deg = _fold_angle_deg(math.degrees(math.atan2(y2 - y1, x2 - x1)))

            accepted.append(
                LineSegment(
                    component_id=frag.component_id,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    length_m=length,
                    angle_deg=angle_deg,
                    linearity=float(linearity),
                    pixel_count=frag.pixel_count,
                )
            )
            linearity_values.append(float(linearity))
            length_values.append(length)

        if linearity_values:
            linearity_p50 = float(np.percentile(linearity_values, 50))
        else:
            linearity_p50 = 0.0
        if length_values:
            length_p50 = float(np.percentile(length_values, 50))
            total_len = float(sum(length_values))
            length_weighted_mean_lin = float(
                sum(
                    seg.linearity * seg.length_m for seg in accepted
                ) / total_len
            ) if total_len > 0 else 0.0
        else:
            length_p50 = 0.0
            length_weighted_mean_lin = 0.0

        metadata: dict[str, object] = {
            "accept_count": len(accepted),
            "reject_blob_count": reject_blob,
            "reject_short_count": reject_short,
            "linearity_p50": linearity_p50,
            "linearity_weighted_mean": length_weighted_mean_lin,
            "length_p50_m": length_p50,
            "params": params.to_metadata(),
        }
        return LineSegmentSet(
            segments=accepted,
            rejected_blob_count=reject_blob,
            rejected_short_count=reject_short,
            metadata=metadata,
        )
