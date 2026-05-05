"""Sprint 51 — PolygonAssemblyStep unit tests."""
from __future__ import annotations

from indoor_server.application.building.steps.wall_polygon.assembly import (
    PolygonAssemblyParams,
    PolygonAssemblyStep,
)
from indoor_server.application.building.steps.wall_polygon.merge import (
    WallLine,
    WallLineSet,
)


def _wall_line(
    x1: float, y1: float, x2: float, y2: float, axis_id: int, offset: float = 0.0
) -> WallLine:
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return WallLine(
        axis_id=axis_id,
        offset_m=offset,
        start_along_axis_m=0.0,
        end_along_axis_m=length,
        world_x1=x1,
        world_y1=y1,
        world_x2=x2,
        world_y2=y2,
        member_segment_ids=[0],
        confidence=0.95,
    )


def _axis_line(
    *,
    axis_id: int,
    offset: float,
    start: float,
    end: float,
) -> WallLine:
    if axis_id == 0:
        x1, y1, x2, y2 = start, offset, end, offset
    else:
        x1, y1, x2, y2 = offset, start, offset, end
    return WallLine(
        axis_id=axis_id,
        offset_m=offset,
        start_along_axis_m=start,
        end_along_axis_m=end,
        world_x1=x1,
        world_y1=y1,
        world_x2=x2,
        world_y2=y2,
        member_segment_ids=[0],
        confidence=0.95,
    )


def test_assembly_rectangle_4_walls() -> None:
    # ㅁ자 — 4 walls forming a 4x3 rectangle.
    lines = [
        _wall_line(0.0, 0.0, 4.0, 0.0, axis_id=0),  # bottom
        _wall_line(4.0, 0.0, 4.0, 3.0, axis_id=1),  # right
        _wall_line(0.0, 3.0, 4.0, 3.0, axis_id=0),  # top
        _wall_line(0.0, 0.0, 0.0, 3.0, axis_id=1),  # left
    ]
    line_set = WallLineSet(lines=lines, dominant_angle_deg=0.0, metadata={})
    result = PolygonAssemblyStep().run(line_set)
    assert result.polygon_geojson is not None
    assert result.fallback_used is None
    assert result.corner_orthogonality_ratio >= 0.95
    assert result.vertex_count == 4


def test_assembly_empty_input() -> None:
    line_set = WallLineSet(lines=[], dominant_angle_deg=0.0, metadata={})
    result = PolygonAssemblyStep().run(line_set)
    assert result.polygon_geojson is None
    assert result.vertex_count == 0


def test_assembly_alpha_shape_fallback_for_disjoint_lines() -> None:
    # 4 lines that don't intersect cleanly to form a cycle but make a rough hull.
    lines = [
        _wall_line(0.0, 0.0, 1.0, 0.0, axis_id=0),
        _wall_line(2.0, 0.5, 3.0, 0.5, axis_id=0),
        _wall_line(0.0, 1.5, 1.0, 1.5, axis_id=0),
        _wall_line(2.0, 2.0, 3.0, 2.0, axis_id=0),
    ]
    line_set = WallLineSet(lines=lines, dominant_angle_deg=0.0, metadata={})
    result = PolygonAssemblyStep(
        PolygonAssemblyParams(use_alpha_shape_fallback=True)
    ).run(line_set)
    # primary cycle finding fails (all axis_id=0 → no intersections kept),
    # alpha shape or convex hull should produce a polygon.
    assert result.fallback_used in ("alpha_shape", "convex_hull")
    assert result.polygon_geojson is not None


def test_assembly_manhattan_pair_fallback_builds_l_corridor() -> None:
    """Parallel wall pairs should form rectilinear corridor rectangles before alpha-shape."""
    lines = [
        _axis_line(axis_id=0, offset=0.6, start=0.8, end=6.8),
        _axis_line(axis_id=0, offset=2.1, start=0.8, end=5.1),
        _axis_line(axis_id=0, offset=7.2, start=5.2, end=6.8),
        _axis_line(axis_id=1, offset=0.8, start=0.9, end=2.1),
        _axis_line(axis_id=1, offset=5.2, start=2.4, end=7.2),
        _axis_line(axis_id=1, offset=6.8, start=2.3, end=7.2),
    ]
    line_set = WallLineSet(lines=lines, dominant_angle_deg=0.0, metadata={})
    result = PolygonAssemblyStep().run(line_set)
    assert result.polygon_geojson is not None
    assert result.fallback_used == "manhattan_pair"
    assert result.corner_orthogonality_ratio >= 0.95
    assert result.vertex_count <= 10
