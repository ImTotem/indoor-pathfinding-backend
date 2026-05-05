"""Sprint 51 — AngleSnapStep unit tests."""
from __future__ import annotations

from indoor_server.application.building.steps.wall_polygon.line_fit import (
    LineSegment,
    LineSegmentSet,
)
from indoor_server.application.building.steps.wall_polygon.snap import (
    AngleSnapParams,
    AngleSnapStep,
)


def _seg(angle_deg: float, length_m: float = 3.0) -> LineSegment:
    import math

    cx, cy = 0.0, 0.0
    rad = math.radians(angle_deg)
    half = length_m / 2.0
    return LineSegment(
        component_id=1,
        x1=cx - half * math.cos(rad),
        y1=cy - half * math.sin(rad),
        x2=cx + half * math.cos(rad),
        y2=cy + half * math.sin(rad),
        length_m=length_m,
        angle_deg=angle_deg,
        linearity=0.95,
        pixel_count=30,
    )


def test_snap_aligns_close_angle_to_axis() -> None:
    # 0°, 88°, 90° → all snap (88° within 15° tolerance of 90°)
    segs = [_seg(0.0), _seg(88.0), _seg(90.0)]
    lines = LineSegmentSet(
        segments=segs, rejected_blob_count=0, rejected_short_count=0, metadata={}
    )
    snapped = AngleSnapStep(AngleSnapParams(snap_tolerance_deg=15.0)).run(lines)
    assert len(snapped.segments) == 3
    assert snapped.snap_pass_ratio == 1.0


def test_snap_rejects_45_with_15_tolerance() -> None:
    segs = [_seg(0.0), _seg(45.0), _seg(0.0)]
    lines = LineSegmentSet(
        segments=segs, rejected_blob_count=0, rejected_short_count=0, metadata={}
    )
    snapped = AngleSnapStep(AngleSnapParams(snap_tolerance_deg=15.0)).run(lines)
    # 0° and 0° pass, 45° rejected
    assert len(snapped.segments) == 2
    assert len(snapped.rejected) == 1


def test_snap_skipped_when_below_min_total_length() -> None:
    segs = [_seg(0.0, length_m=0.5)]
    lines = LineSegmentSet(
        segments=segs, rejected_blob_count=0, rejected_short_count=0, metadata={}
    )
    snapped = AngleSnapStep(AngleSnapParams(min_total_length_m=2.0)).run(lines)
    assert snapped.segments == []
    assert snapped.metadata.get("skipped_reason") == "insufficient_total_length"
