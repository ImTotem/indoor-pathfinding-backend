"""Sprint 51 — CollinearMergeStep unit tests."""
from __future__ import annotations

from indoor_server.application.building.steps.wall_polygon.line_fit import LineSegment
from indoor_server.application.building.steps.wall_polygon.merge import (
    CollinearMergeStep,
    MergeParams,
)
from indoor_server.application.building.steps.wall_polygon.snap import (
    SnappedSegment,
    SnappedSegmentSet,
)


def _seg(
    x1: float, y1: float, x2: float, y2: float, axis_id: int, angle_deg: float = 0.0
) -> SnappedSegment:
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    base = LineSegment(
        component_id=1,
        x1=x1, y1=y1, x2=x2, y2=y2,
        length_m=length,
        angle_deg=angle_deg,
        linearity=0.95,
        pixel_count=int(length * 10),
    )
    return SnappedSegment(
        base=base,
        snapped_angle_deg=angle_deg if axis_id == 0 else 90.0,
        axis_id=axis_id,
        pre_snap_residual_deg=0.0,
    )


def test_merge_two_collinear_with_small_gap() -> None:
    # axis_id=0 (horizontal), same offset y=0, segments (0..2) and (2.5..5)
    segs = [
        _seg(0.0, 0.0, 2.0, 0.0, axis_id=0),
        _seg(2.5, 0.0, 5.0, 0.0, axis_id=0),
    ]
    snapped = SnappedSegmentSet(
        segments=segs,
        rejected=[],
        dominant_angle_deg=0.0,
        dominant_confidence=1.0,
        snap_pass_ratio=1.0,
        metadata={},
    )
    merged = CollinearMergeStep(MergeParams(gap_fill_max_m=1.0)).run(snapped)
    assert len(merged.lines) == 1
    line = merged.lines[0]
    assert abs(line.length_m - 5.0) < 0.01


def test_merge_separate_offsets_kept_distinct() -> None:
    segs = [
        _seg(0.0, 0.0, 2.0, 0.0, axis_id=0),
        _seg(0.0, 0.5, 2.0, 0.5, axis_id=0),  # offset 0.5m → above tolerance 0.20
    ]
    snapped = SnappedSegmentSet(
        segments=segs,
        rejected=[],
        dominant_angle_deg=0.0,
        dominant_confidence=1.0,
        snap_pass_ratio=1.0,
        metadata={},
    )
    merged = CollinearMergeStep(MergeParams(same_offset_tolerance_m=0.20)).run(snapped)
    assert len(merged.lines) == 2


def test_merge_large_gap_not_merged() -> None:
    segs = [
        _seg(0.0, 0.0, 2.0, 0.0, axis_id=0),
        _seg(4.0, 0.0, 6.0, 0.0, axis_id=0),  # gap 2m > 1.0m
    ]
    snapped = SnappedSegmentSet(
        segments=segs,
        rejected=[],
        dominant_angle_deg=0.0,
        dominant_confidence=1.0,
        snap_pass_ratio=1.0,
        metadata={},
    )
    merged = CollinearMergeStep(MergeParams(gap_fill_max_m=1.0)).run(snapped)
    assert len(merged.lines) == 2
