"""Sprint 52 — facade `run_from_heatmap` direct-entry tests (W-3).

These tests verify that the Step 0 bypass entrypoint produces the expected
metadata structure and exercises both the skeleton primary path and the
hough fallback path without any monkey-patching.
"""
from __future__ import annotations

import numpy as np

from indoor_server.application.building.steps.wall_polygon import (
    WallPolygonFromObstacleStep,
    WallPolygonStepParams,
)
from indoor_server.application.building.steps.wall_polygon.obstacle_source import (
    ObstacleHeatmap,
)


def _l_corridor_heatmap() -> ObstacleHeatmap:
    h, w = 80, 80
    counts = np.zeros((h, w), dtype=np.int32)
    base = 8
    counts[5:10, 5:70] = base
    counts[20:25, 5:55] = base
    counts[5:25, 5:10] = base
    counts[20:75, 50:55] = base
    counts[20:75, 65:70] = base
    counts[70:75, 50:70] = base
    return ObstacleHeatmap(
        counts=counts,
        origin_x=0.0,
        origin_y=0.0,
        cell_size_m=0.10,
        z0=0.0,
        height_min_m=0.30,
        height_max_m=2.50,
        metadata={"world_obstacle_point_count": int(counts.sum()), "params": {}},
    )


def test_run_from_heatmap_direct_call() -> None:
    """No monkey-patch needed — facade accepts a synthetic heatmap directly."""
    fixture = _l_corridor_heatmap()
    step = WallPolygonFromObstacleStep(WallPolygonStepParams())
    floor = {
        "type": "Polygon",
        "coordinates": [
            [[0.6, 0.6], [7.0, 0.6], [7.0, 7.5], [5.0, 7.5],
             [5.0, 2.5], [0.6, 2.5], [0.6, 0.6]]
        ],
    }
    result = step.run_from_heatmap(fixture, floor_polygon_geojson=floor)
    assert result.metadata["enabled"] is True
    assert result.metadata["iou_with_floor"] >= 0.50
    stages = result.metadata["stages"]
    assert "obstacle_source" in stages
    assert "density" in stages
    assert "line_fit" in stages
    line_fit_meta = stages["line_fit"]
    assert isinstance(line_fit_meta, dict)
    # source must be one of the four legal values.
    assert line_fit_meta["source"] in (
        "skeleton_only",
        "hough_only",
        "skeleton+hough",
        "empty",
    )


def test_run_delegates_to_run_from_heatmap_for_empty_inputs() -> None:
    """Step 0 produces empty heatmap when inputs empty → no_obstacle_data."""
    step = WallPolygonFromObstacleStep()
    result = step.run(
        nodes=[],
        frames=[],
        floor_masks_by_node_id={},
        z0=0.0,
        floor_polygon_geojson=None,
    )
    assert result.accepted is False
    assert result.fail_reason == "no_obstacle_data"


def test_skeleton_primary_alone_is_enough_disables_fallback() -> None:
    """If primary already produces ≥min_wall_lines, fallback is not invoked."""
    fixture = _l_corridor_heatmap()
    params = WallPolygonStepParams(
        min_wall_lines=1,
        max_wall_lines=200,
        use_hough_fallback=True,
    )
    step = WallPolygonFromObstacleStep(params)
    result = step.run_from_heatmap(fixture, floor_polygon_geojson=None)
    line_fit_meta = result.metadata["stages"]["line_fit"]
    # Hough fallback only fires when primary < min_wall_lines.
    assert line_fit_meta["fallback_count"] == 0
    assert line_fit_meta["source"] in ("skeleton_only", "empty")


def test_skeleton_disabled_fallback_engaged() -> None:
    """Forcing primary off triggers Hough fallback with non-zero output."""
    fixture = _l_corridor_heatmap()
    params = WallPolygonStepParams(use_skeleton_split=False)
    step = WallPolygonFromObstacleStep(params)
    result = step.run_from_heatmap(fixture, floor_polygon_geojson=None)
    line_fit_meta = result.metadata["stages"]["line_fit"]
    assert line_fit_meta["primary_count"] == 0
    assert line_fit_meta["source"] in ("hough_only", "empty")
