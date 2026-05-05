"""WallInteriorFillStep synthetic unit tests."""
from __future__ import annotations

import numpy as np

from indoor_server.application.building.steps.wall_polygon.interior_fill import (
    WallInteriorFillParams,
    WallInteriorFillStep,
)
from indoor_server.application.building.steps.wall_polygon.obstacle_source import (
    ObstacleHeatmap,
)


def _closed_room_heatmap() -> ObstacleHeatmap:
    counts = np.zeros((12, 12), dtype=np.int32)
    counts[2, 2:10] = 8
    counts[9, 2:10] = 8
    counts[2:10, 2] = 8
    counts[2:10, 9] = 8
    return ObstacleHeatmap(
        counts=counts,
        origin_x=0.0,
        origin_y=0.0,
        cell_size_m=1.0,
        z0=0.0,
        height_min_m=0.0,
        height_max_m=2.5,
        metadata={"world_obstacle_point_count": int(counts.sum())},
    )


def test_wall_interior_fill_floods_seeded_inside_region() -> None:
    result = WallInteriorFillStep(
        WallInteriorFillParams(
            wall_min_cell_hits=8,
            wall_bridge_gap_radius_cells=0,
            wall_close_radius_cells=0,
            wall_dilate_radius_cells=0,
            support_dilate_radius_cells=0,
            seed_radius_cells=0,
            interior_close_radius_cells=0,
            rectilinear_enabled=False,
            min_component_area_m2=2.0,
        )
    ).run(
        heatmap=_closed_room_heatmap(),
        seed_points_xy=[(4.5, 4.5), (7.5, 7.5)],
    )

    assert result.accepted
    assert result.interior_geojson is not None
    assert result.metadata["seed_point_inside_ratio"] == 1.0
    assert result.metadata["wall_inside_ratio"] == 0.0
    assert result.metadata["interior_cells"] > 0


def test_wall_interior_fill_requires_seed_points() -> None:
    result = WallInteriorFillStep().run(
        heatmap=_closed_room_heatmap(),
        seed_points_xy=[],
    )

    assert not result.accepted
    assert result.fail_reason == "not_enough_seed_points"
