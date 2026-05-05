"""Sprint 51 — Wall polygon facade end-to-end tests.

Synthetic ㄱ자 corridor: build an obstacle heatmap by directly constructing
ObstacleHeatmap and bypassing Step 0 (which requires real RTABMap depth).
We monkeypatch Step 0 to return our fixture, exercising Steps 1..7.
"""
from __future__ import annotations

import numpy as np
import pytest

from indoor_server.application.building.steps.wall_polygon import (
    WallPolygonFromObstacleStep,
    WallPolygonStepParams,
)
from indoor_server.application.building.steps.wall_polygon.obstacle_source import (
    ObstacleHeatmap,
    ObstacleSourceStep,
)


def _l_shape_heatmap() -> ObstacleHeatmap:
    """Synthetic ㄱ-shape — 4 walls forming an L corridor outline."""
    h, w = 60, 60
    counts = np.zeros((h, w), dtype=np.int32)
    # Outer rectangle 5x5 to 55x55 walls (only outer perimeter for simplicity)
    counts[5:7, 5:55] = 8       # bottom wall
    counts[53:55, 5:55] = 8     # top wall
    counts[5:55, 5:7] = 8       # left wall
    counts[5:55, 53:55] = 8     # right wall
    # Add interior wall to make ㄱ shape (vertical wall at col=30, rows 5..30)
    # Left out for simplicity; rectangle is the simplest cycle for assembly.
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


@pytest.fixture
def patched_step(monkeypatch: pytest.MonkeyPatch) -> WallPolygonFromObstacleStep:
    """Replace Step 0 obstacle source with synthetic heatmap."""
    fixture = _l_shape_heatmap()

    def fake_run(self, **kwargs):  # type: ignore[no-untyped-def]
        return fixture

    monkeypatch.setattr(ObstacleSourceStep, "run", fake_run)
    return WallPolygonFromObstacleStep(WallPolygonStepParams())


def test_facade_happy_rectangle(patched_step: WallPolygonFromObstacleStep) -> None:
    floor_geojson = {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [6.0, 0.0], [6.0, 6.0], [0.0, 6.0], [0.0, 0.0]]],
    }
    result = patched_step.run(
        nodes=[],
        frames=[],
        floor_masks_by_node_id={},
        z0=0.0,
        floor_polygon_geojson=floor_geojson,
    )
    # Sprint 52: rectangle 4-walls always yield at least a line set; assembly
    # may PASS or need fallback. The Step 2~3 dispatch is no longer
    # `components`/`line_fit` PCA — it is `line_fit` metadata that captures
    # the new skeleton+hough source.
    assert result.metadata["enabled"] is True
    assert "stages" in result.metadata
    stages = result.metadata["stages"]
    assert "obstacle_source" in stages
    assert "density" in stages
    assert "line_fit" in stages
    line_fit_meta = stages["line_fit"]
    assert isinstance(line_fit_meta, dict)
    assert "source" in line_fit_meta
    assert line_fit_meta["source"] in (
        "skeleton_only",
        "hough_only",
        "skeleton+hough",
        "empty",
    )


def test_facade_empty_obstacle(monkeypatch: pytest.MonkeyPatch) -> None:
    """No obstacle data → fail_reason='no_obstacle_data'."""
    empty = ObstacleHeatmap(
        counts=np.zeros((1, 1), dtype=np.int32),
        origin_x=0.0, origin_y=0.0, cell_size_m=0.10,
        z0=0.0, height_min_m=0.0, height_max_m=2.0,
        metadata={"world_obstacle_point_count": 0, "params": {}},
    )
    monkeypatch.setattr(
        ObstacleSourceStep,
        "run",
        lambda self, **kw: empty,
    )
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


def test_facade_metadata_round_trip() -> None:
    """params dataclass round-trip via to_metadata."""
    params = WallPolygonStepParams(min_wall_lines=3, max_wall_lines=25)
    meta = params.to_metadata()
    assert meta["min_wall_lines"] == 3
    assert meta["max_wall_lines"] == 25
    assert "obstacle_source" in meta
    assert "density" in meta
