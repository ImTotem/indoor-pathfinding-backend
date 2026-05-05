from __future__ import annotations

import pytest

from indoor_server.application.routing.manhattan_path import (
    is_rectilinear_polyline,
    manhattanize_route_polyline,
    polyline_length,
)


def test_manhattanize_route_polyline_outputs_axis_aligned_segments() -> None:
    raw = [
        (0.05, 0.05, 0.0),
        (1.15, 0.38, 0.0),
        (2.30, 0.92, 0.0),
        (2.75, 2.60, 0.0),
        (3.20, 4.10, 0.0),
    ]

    rectified = manhattanize_route_polyline(raw, grid_m=0.25)

    assert rectified[0] == (0.0, 0.0, 0.0)
    assert rectified[-1] == (3.25, 4.0, 0.0)
    assert is_rectilinear_polyline(rectified)
    assert len(rectified) < len(raw) * 2


def test_manhattanize_route_polyline_keeps_single_diagonal_as_right_angle() -> None:
    rectified = manhattanize_route_polyline(
        [(0.0, 0.0, 0.0), (2.0, 2.0, 0.0)],
        grid_m=0.25,
    )

    assert rectified == [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0)]
    assert polyline_length(rectified) == pytest.approx(4.0)
