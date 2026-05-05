"""Display-time Manhattan rectification for route polylines."""
from __future__ import annotations

from math import hypot
from statistics import median

Point3 = tuple[float, float, float]


def manhattanize_route_polyline(
    polyline: list[Point3],
    *,
    grid_m: float = 0.25,
    min_segment_m: float = 0.15,
) -> list[Point3]:
    """Return a rectilinear display polyline without changing route topology."""

    if len(polyline) <= 1:
        return polyline

    snapped = [_snap_point(point, grid_m) for point in polyline]
    axes = _segment_axes(snapped, min_segment_m=min_segment_m)
    if not axes:
        return _dedupe(snapped)

    runs = _axis_runs(axes)
    display_xy: list[tuple[float, float]] = []
    previous_end = (snapped[0][0], snapped[0][1])
    display_xy.append(previous_end)

    for run_index, run in enumerate(runs):
        start_index, end_index, axis = run
        run_points = snapped[start_index : end_index + 1]
        start_raw = (snapped[start_index][0], snapped[start_index][1])
        end_raw = (snapped[end_index][0], snapped[end_index][1])
        if axis == "h":
            const = _run_constant(
                [point[1] for point in run_points],
                preferred=previous_end[1] if run_index == 0 else None,
            )
            start = (start_raw[0], const)
            end = (end_raw[0], const)
        else:
            const = _run_constant(
                [point[0] for point in run_points],
                preferred=previous_end[0] if run_index == 0 else None,
            )
            start = (const, start_raw[1])
            end = (const, end_raw[1])

        display_xy.extend(_orthogonal_connector(display_xy[-1], start, prefer_axis=axis))
        display_xy.append(start)
        display_xy.append(end)
        previous_end = end

    final_xy = (snapped[-1][0], snapped[-1][1])
    display_xy.extend(_orthogonal_connector(display_xy[-1], final_xy, prefer_axis=runs[-1][2]))
    display_xy.append(final_xy)

    z_values = _z_values_for_display(display_xy, snapped)
    return _dedupe([
        (x, y, z)
        for (x, y), z in zip(display_xy, z_values, strict=True)
    ])


def polyline_length(polyline: list[Point3]) -> float:
    total = 0.0
    for first, second in zip(polyline, polyline[1:], strict=False):
        total += hypot(second[0] - first[0], second[1] - first[1])
    return total


def is_rectilinear_polyline(
    polyline: list[Point3],
    *,
    tolerance_m: float = 1e-6,
) -> bool:
    for first, second in zip(polyline, polyline[1:], strict=False):
        dx = abs(second[0] - first[0])
        dy = abs(second[1] - first[1])
        if dx > tolerance_m and dy > tolerance_m:
            return False
    return True


def _segment_axes(polyline: list[Point3], *, min_segment_m: float) -> list[str]:
    axes: list[str] = []
    previous = "h"
    for first, second in zip(polyline, polyline[1:], strict=False):
        dx = abs(second[0] - first[0])
        dy = abs(second[1] - first[1])
        if dx < min_segment_m and dy < min_segment_m:
            axes.append(previous)
            continue
        if dx >= dy * 1.25:
            previous = "h"
        elif dy >= dx * 1.25:
            previous = "v"
        axes.append(previous)
    return axes


def _axis_runs(axes: list[str]) -> list[tuple[int, int, str]]:
    runs: list[tuple[int, int, str]] = []
    start = 0
    current = axes[0]
    for index, axis in enumerate(axes[1:], start=1):
        if axis == current:
            continue
        runs.append((start, index, current))
        start = index
        current = axis
    runs.append((start, len(axes), current))
    return runs


def _run_constant(values: list[float], *, preferred: float | None) -> float:
    if preferred is not None:
        return preferred
    return float(median(values))


def _orthogonal_connector(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    prefer_axis: str,
) -> list[tuple[float, float]]:
    if start == end:
        return []
    if abs(start[0] - end[0]) < 1e-9 or abs(start[1] - end[1]) < 1e-9:
        return []
    if prefer_axis == "h":
        return [(end[0], start[1])]
    return [(start[0], end[1])]


def _z_values_for_display(
    display_xy: list[tuple[float, float]],
    raw: list[Point3],
) -> list[float]:
    if not raw:
        return [0.0 for _ in display_xy]
    z_values: list[float] = []
    for x, y in display_xy:
        nearest = min(raw, key=lambda point: hypot(point[0] - x, point[1] - y))
        z_values.append(nearest[2])
    return z_values


def _snap_point(point: Point3, grid_m: float) -> Point3:
    if grid_m <= 0:
        return point
    return (
        round(point[0] / grid_m) * grid_m,
        round(point[1] / grid_m) * grid_m,
        point[2],
    )


def _dedupe(polyline: list[Point3]) -> list[Point3]:
    deduped: list[Point3] = []
    for point in polyline:
        if deduped and point == deduped[-1]:
            continue
        deduped.append(point)
    return deduped
