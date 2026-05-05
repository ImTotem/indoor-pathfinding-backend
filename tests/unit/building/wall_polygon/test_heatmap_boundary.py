"""HeatmapBoundaryStep tests."""
from __future__ import annotations

import numpy as np

from indoor_server.application.building.steps.wall_polygon.heatmap_boundary import (
    HeatmapBoundaryParams,
    HeatmapBoundaryStep,
)
from indoor_server.application.building.steps.wall_polygon.obstacle_source import (
    ObstacleHeatmap,
)


def _heatmap(counts: np.ndarray) -> ObstacleHeatmap:
    return ObstacleHeatmap(
        counts=counts.astype(np.int32),
        origin_x=0.0,
        origin_y=0.0,
        cell_size_m=0.10,
        z0=0.0,
        height_min_m=0.30,
        height_max_m=2.50,
        metadata={"world_obstacle_point_count": int(counts.sum())},
    )


def test_heatmap_boundary_connects_short_axis_gap() -> None:
    counts = np.zeros((40, 60), dtype=np.int32)
    counts[10:12, 5:28] = 9
    counts[10:12, 31:55] = 9  # 3-cell gap from the first segment

    result = HeatmapBoundaryStep(
        HeatmapBoundaryParams(
            min_cell_hits=8,
            bridge_gap_radius_cells=3,
            close_radius_cells=0,
            min_component_area_m2=0.01,
            simplify_tolerance_m=0.0,
        )
    ).run(_heatmap(counts))

    assert result.accepted is True
    assert result.boundary_geojson is not None
    assert result.metadata["threshold_cells"] == 94
    assert result.metadata["boundary_cells"] > result.metadata["threshold_cells"]
    assert result.metadata["polygon_vertex_count"] >= 4


def test_heatmap_boundary_returns_failure_for_empty_threshold() -> None:
    counts = np.zeros((10, 10), dtype=np.int32)

    result = HeatmapBoundaryStep(
        HeatmapBoundaryParams(min_cell_hits=8)
    ).run(_heatmap(counts))

    assert result.accepted is False
    assert result.fail_reason == "no_threshold_cells"
    assert result.boundary_geojson is None


def test_heatmap_boundary_keeps_largest_components_only() -> None:
    counts = np.zeros((30, 30), dtype=np.int32)
    counts[3:8, 3:8] = 9
    counts[12:16, 12:16] = 9
    counts[25, 25] = 9

    result = HeatmapBoundaryStep(
        HeatmapBoundaryParams(
            min_cell_hits=8,
            bridge_gap_radius_cells=0,
            close_radius_cells=0,
            keep_largest_components=2,
            min_component_area_m2=0.01,
            simplify_tolerance_m=0.0,
        )
    ).run(_heatmap(counts))

    assert result.accepted is True
    assert result.metadata["components_before_filter"] == 3
    assert result.metadata["boundary_cells"] == 41
