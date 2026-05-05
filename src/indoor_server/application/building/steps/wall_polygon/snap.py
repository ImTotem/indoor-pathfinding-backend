"""Sprint 51 — Step 4: dominant angle snap.

length-weighted histogram top-1 angle ± snap_tolerance 안에 들면 axis 위로
projection 후 endpoint 보존, 안 들면 reject.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from indoor_server.application.building.steps.wall_polygon.line_fit import (
    LineSegment,
    LineSegmentSet,
)


@dataclass(frozen=True)
class AngleSnapParams:
    histogram_bin_deg: float = 5.0
    snap_tolerance_deg: float = 15.0
    min_total_length_m: float = 2.0

    def to_metadata(self) -> dict[str, object]:
        return {
            "histogram_bin_deg": self.histogram_bin_deg,
            "snap_tolerance_deg": self.snap_tolerance_deg,
            "min_total_length_m": self.min_total_length_m,
        }


@dataclass(frozen=True)
class SnappedSegment:
    base: LineSegment
    snapped_angle_deg: float  # absolute world angle after snap
    axis_id: int  # 0=primary, 1=perpendicular
    pre_snap_residual_deg: float


@dataclass(frozen=True)
class SnappedSegmentSet:
    segments: list[SnappedSegment]
    rejected: list[LineSegment]
    dominant_angle_deg: float
    dominant_confidence: float
    snap_pass_ratio: float
    metadata: dict[str, object] = field(default_factory=dict)


def _fold_to_0_180(angle_deg: float) -> float:
    """Fold angle to [0, 180)."""
    a = float(angle_deg) % 180.0
    if a < 0:
        a += 180.0
    return a


def _diff_to_axis(angle_deg: float, axis_deg: float) -> float:
    """Signed minimum diff |angle - axis| considering 180° periodicity."""
    a = _fold_to_0_180(angle_deg)
    b = _fold_to_0_180(axis_deg)
    d = a - b
    while d > 90.0:
        d -= 180.0
    while d < -90.0:
        d += 180.0
    return d


def _length_weighted_dominant(
    segments: list[LineSegment], bin_deg: float
) -> tuple[float, float, float]:
    """length-weighted histogram top-1 angle.

    Returns (dominant_angle_deg [0, 180), best_bin_ratio, total_length).
    """
    if not segments:
        return 0.0, 0.0, 0.0
    total_len = float(sum(seg.length_m for seg in segments))
    if total_len <= 0:
        return 0.0, 0.0, 0.0

    n_bins = int(round(180.0 / float(bin_deg)))
    if n_bins <= 0:
        n_bins = 36
    hist = np.zeros(n_bins, dtype=np.float64)
    for seg in segments:
        a = _fold_to_0_180(seg.angle_deg)
        idx = int(a / 180.0 * n_bins)
        if idx >= n_bins:
            idx = n_bins - 1
        hist[idx] += seg.length_m
    best_idx = int(np.argmax(hist))
    best_weight = float(hist[best_idx])
    best_bin_ratio = best_weight / total_len if total_len > 0 else 0.0
    bin_center = (best_idx + 0.5) * (180.0 / n_bins)
    return bin_center, best_bin_ratio, total_len


class AngleSnapStep:
    """Step 4 — dominant angle snap."""

    def __init__(self, params: AngleSnapParams | None = None) -> None:
        self._params = params if params is not None else AngleSnapParams()
        self._validate()

    def _validate(self) -> None:
        p = self._params
        if p.histogram_bin_deg <= 0:
            raise ValueError("histogram_bin_deg must be > 0")
        if p.snap_tolerance_deg <= 0:
            raise ValueError("snap_tolerance_deg must be > 0")
        if p.min_total_length_m < 0:
            raise ValueError("min_total_length_m must be >= 0")

    def run(
        self,
        lines: LineSegmentSet,
        *,
        external_hint_deg: float | None = None,
    ) -> SnappedSegmentSet:
        params = self._params
        segs = list(lines.segments)
        total_len = float(sum(s.length_m for s in segs))

        if not segs or total_len < params.min_total_length_m:
            return SnappedSegmentSet(
                segments=[],
                rejected=segs,
                dominant_angle_deg=0.0,
                dominant_confidence=0.0,
                snap_pass_ratio=0.0,
                metadata={
                    "snap_pass_count": 0,
                    "snap_reject_count": len(segs),
                    "dominant_angle_deg": 0.0,
                    "dominant_confidence": 0.0,
                    "external_hint_used": external_hint_deg is not None,
                    "skipped_reason": "insufficient_total_length",
                    "params": params.to_metadata(),
                },
            )

        if external_hint_deg is not None:
            dominant = _fold_to_0_180(external_hint_deg)
            confidence = 1.0
        else:
            dominant, confidence, _ = _length_weighted_dominant(
                segs, params.histogram_bin_deg
            )

        accepted: list[SnappedSegment] = []
        rejected: list[LineSegment] = []
        for seg in segs:
            d_primary = _diff_to_axis(seg.angle_deg, dominant)
            d_secondary = _diff_to_axis(seg.angle_deg, dominant + 90.0)
            if abs(d_primary) <= params.snap_tolerance_deg:
                axis_id = 0
                snapped_angle = dominant
                residual = d_primary
            elif abs(d_secondary) <= params.snap_tolerance_deg:
                axis_id = 1
                snapped_angle = _fold_to_0_180(dominant + 90.0)
                residual = d_secondary
            else:
                rejected.append(seg)
                continue

            # rotate endpoints onto axis by projection.
            mid_x = (seg.x1 + seg.x2) / 2.0
            mid_y = (seg.y1 + seg.y2) / 2.0
            length = seg.length_m
            half = length / 2.0
            rad = math.radians(snapped_angle)
            dx = math.cos(rad)
            dy = math.sin(rad)
            x1 = mid_x - half * dx
            y1 = mid_y - half * dy
            x2 = mid_x + half * dx
            y2 = mid_y + half * dy
            new_seg = LineSegment(
                component_id=seg.component_id,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                length_m=length,
                angle_deg=snapped_angle if snapped_angle <= 90.0 else snapped_angle - 180.0,
                linearity=seg.linearity,
                pixel_count=seg.pixel_count,
            )
            accepted.append(
                SnappedSegment(
                    base=new_seg,
                    snapped_angle_deg=snapped_angle,
                    axis_id=axis_id,
                    pre_snap_residual_deg=float(abs(residual)),
                )
            )

        total = len(segs)
        pass_ratio = (len(accepted) / total) if total > 0 else 0.0

        metadata: dict[str, object] = {
            "snap_pass_count": len(accepted),
            "snap_reject_count": len(rejected),
            "dominant_angle_deg": float(dominant),
            "dominant_confidence": float(confidence),
            "snap_pass_ratio": float(pass_ratio),
            "external_hint_used": external_hint_deg is not None,
            "params": params.to_metadata(),
        }
        return SnappedSegmentSet(
            segments=accepted,
            rejected=rejected,
            dominant_angle_deg=float(dominant),
            dominant_confidence=float(confidence),
            snap_pass_ratio=float(pass_ratio),
            metadata=metadata,
        )
