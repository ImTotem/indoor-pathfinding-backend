"""Sprint 51 — Step 5: collinear merge + gap fill.

rotated frame 으로 회전 → axis_id 별 group → offset clustering → along-axis
sweep merge (gap < gap_fill_max_m 이면 같은 line).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from indoor_server.application.building.steps.wall_polygon.snap import (
    SnappedSegment,
    SnappedSegmentSet,
)


@dataclass(frozen=True)
class MergeParams:
    same_offset_tolerance_m: float = 0.20
    gap_fill_max_m: float = 1.0
    min_merged_length_m: float = 0.5

    def to_metadata(self) -> dict[str, object]:
        return {
            "same_offset_tolerance_m": self.same_offset_tolerance_m,
            "gap_fill_max_m": self.gap_fill_max_m,
            "min_merged_length_m": self.min_merged_length_m,
        }


@dataclass(frozen=True)
class WallLine:
    axis_id: int
    offset_m: float
    start_along_axis_m: float
    end_along_axis_m: float
    world_x1: float
    world_y1: float
    world_x2: float
    world_y2: float
    member_segment_ids: list[int]
    confidence: float

    @property
    def length_m(self) -> float:
        return float(self.end_along_axis_m - self.start_along_axis_m)


@dataclass(frozen=True)
class WallLineSet:
    lines: list[WallLine]
    dominant_angle_deg: float
    metadata: dict[str, object] = field(default_factory=dict)


def _segment_to_axis_coords(
    seg: SnappedSegment,
    dominant_rad: float,
) -> tuple[float, float, float, int]:
    """Project segment endpoints onto rotated frame.

    Returns (along_min, along_max, offset, axis_id).
    rotated frame: u = primary axis (angle = dominant), v = perpendicular.
    For axis_id=0 (primary), segment runs along u → offset = v_mean.
    For axis_id=1 (perpendicular), runs along v → offset = u_mean (sign-keyed).
    """
    cos_a = math.cos(dominant_rad)
    sin_a = math.sin(dominant_rad)
    # rotation: (x, y) → (u = x*cos + y*sin, v = -x*sin + y*cos)
    p1u = seg.base.x1 * cos_a + seg.base.y1 * sin_a
    p1v = -seg.base.x1 * sin_a + seg.base.y1 * cos_a
    p2u = seg.base.x2 * cos_a + seg.base.y2 * sin_a
    p2v = -seg.base.x2 * sin_a + seg.base.y2 * cos_a

    if seg.axis_id == 0:
        along = (min(p1u, p2u), max(p1u, p2u))
        offset = (p1v + p2v) / 2.0
    else:
        along = (min(p1v, p2v), max(p1v, p2v))
        offset = (p1u + p2u) / 2.0
    return along[0], along[1], offset, seg.axis_id


class CollinearMergeStep:
    """Step 5 — collinear merge + gap fill."""

    def __init__(self, params: MergeParams | None = None) -> None:
        self._params = params if params is not None else MergeParams()
        self._validate()

    def _validate(self) -> None:
        p = self._params
        if p.same_offset_tolerance_m < 0:
            raise ValueError("same_offset_tolerance_m must be >= 0")
        if p.gap_fill_max_m < 0:
            raise ValueError("gap_fill_max_m must be >= 0")
        if p.min_merged_length_m < 0:
            raise ValueError("min_merged_length_m must be >= 0")

    def run(self, snapped: SnappedSegmentSet) -> WallLineSet:
        params = self._params
        if not snapped.segments:
            return WallLineSet(
                lines=[],
                dominant_angle_deg=snapped.dominant_angle_deg,
                metadata={
                    "lines_count": 0,
                    "merged_member_count": 0,
                    "gap_filled_count": 0,
                    "length_p50_m": 0.0,
                    "length_total_m": 0.0,
                    "params": params.to_metadata(),
                },
            )

        dominant_deg = snapped.dominant_angle_deg
        dominant_rad = math.radians(dominant_deg)
        cos_a = math.cos(dominant_rad)
        sin_a = math.sin(dominant_rad)

        # gather (along_min, along_max, offset, axis_id, source_idx)
        per_axis: dict[int, list[tuple[float, float, float, int]]] = defaultdict(list)
        for idx, seg in enumerate(snapped.segments):
            a_min, a_max, offset, axis_id = _segment_to_axis_coords(seg, dominant_rad)
            per_axis[axis_id].append((a_min, a_max, offset, idx))

        merged: list[WallLine] = []
        gap_filled = 0
        merged_members = 0

        for axis_id, items in per_axis.items():
            # cluster by offset using greedy (sort by offset)
            items_sorted = sorted(items, key=lambda t: t[2])
            clusters: list[list[tuple[float, float, float, int]]] = []
            current: list[tuple[float, float, float, int]] = []
            current_offset_sum = 0.0
            for it in items_sorted:
                if not current:
                    current = [it]
                    current_offset_sum = it[2]
                    continue
                mean_offset = current_offset_sum / len(current)
                if abs(it[2] - mean_offset) <= params.same_offset_tolerance_m:
                    current.append(it)
                    current_offset_sum += it[2]
                else:
                    clusters.append(current)
                    current = [it]
                    current_offset_sum = it[2]
            if current:
                clusters.append(current)

            for cluster in clusters:
                # sort by along_min, sweep merge.
                ranges = sorted(
                    [(a_min, a_max, src_idx) for (a_min, a_max, _, src_idx) in cluster]
                )
                offset_mean = sum(c[2] for c in cluster) / len(cluster)
                grouped: list[tuple[float, float, list[int]]] = []
                cur_min, cur_max, cur_src = ranges[0]
                cur_members = [cur_src]
                for a_min, a_max, src_idx in ranges[1:]:
                    gap = a_min - cur_max
                    if gap <= params.gap_fill_max_m:
                        if gap > 0:
                            gap_filled += 1
                        cur_max = max(cur_max, a_max)
                        cur_members.append(src_idx)
                    else:
                        grouped.append((cur_min, cur_max, cur_members))
                        cur_min, cur_max = a_min, a_max
                        cur_members = [src_idx]
                grouped.append((cur_min, cur_max, cur_members))

                for g_min, g_max, members in grouped:
                    length = g_max - g_min
                    if length < params.min_merged_length_m:
                        continue
                    # rotate back to world frame
                    if axis_id == 0:
                        # along=u, offset=v
                        u1, u2 = g_min, g_max
                        v = offset_mean
                    else:
                        u1, u2 = offset_mean, offset_mean
                        v = (g_min + g_max) / 2.0  # placeholder, override below
                    # general: world = (u, v) → x = u*cos - v*sin, y = u*sin + v*cos
                    if axis_id == 0:
                        x1 = u1 * cos_a - v * sin_a
                        y1 = u1 * sin_a + v * cos_a
                        x2 = u2 * cos_a - v * sin_a
                        y2 = u2 * sin_a + v * cos_a
                    else:
                        u = offset_mean
                        v1, v2 = g_min, g_max
                        x1 = u * cos_a - v1 * sin_a
                        y1 = u * sin_a + v1 * cos_a
                        x2 = u * cos_a - v2 * sin_a
                        y2 = u * sin_a + v2 * cos_a

                    member_lengths = [
                        snapped.segments[m].base.length_m for m in members
                    ]
                    member_lin = [
                        snapped.segments[m].base.linearity for m in members
                    ]
                    total_member_len = sum(member_lengths)
                    if total_member_len > 0:
                        confidence = (
                            sum(
                                lin * ml
                                for lin, ml in zip(
                                    member_lin, member_lengths, strict=True
                                )
                            )
                            / total_member_len
                        )
                    else:
                        confidence = 0.0

                    merged.append(
                        WallLine(
                            axis_id=axis_id,
                            offset_m=float(offset_mean),
                            start_along_axis_m=float(g_min),
                            end_along_axis_m=float(g_max),
                            world_x1=float(x1),
                            world_y1=float(y1),
                            world_x2=float(x2),
                            world_y2=float(y2),
                            member_segment_ids=list(members),
                            confidence=float(confidence),
                        )
                    )
                    merged_members += len(members)

        lengths = [m.length_m for m in merged]
        length_p50 = float(np.percentile(lengths, 50)) if lengths else 0.0
        length_total = float(sum(lengths))

        metadata: dict[str, object] = {
            "lines_count": len(merged),
            "merged_member_count": merged_members,
            "gap_filled_count": int(gap_filled),
            "length_p50_m": length_p50,
            "length_total_m": length_total,
            "params": params.to_metadata(),
        }
        return WallLineSet(
            lines=merged,
            dominant_angle_deg=dominant_deg,
            metadata=metadata,
        )
