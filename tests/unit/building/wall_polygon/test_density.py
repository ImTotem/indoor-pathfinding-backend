"""Sprint 51 — DensityRefineStep unit tests."""
from __future__ import annotations

import numpy as np

from indoor_server.application.building.steps.wall_polygon.density import (
    DensityRefined,
    DensityRefineParams,
    DensityRefineStep,
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
        metadata={},
    )


def test_density_threshold_drops_isolated_noise() -> None:
    counts = np.zeros((10, 10), dtype=np.int32)
    counts[5, 5] = 1
    counts[2, 3] = 1
    refined = DensityRefineStep(DensityRefineParams(min_cell_hits=4)).run(_heatmap(counts))
    assert refined.metadata["cells_after_threshold"] == 0


def test_density_keeps_dense_cluster() -> None:
    counts = np.zeros((10, 10), dtype=np.int32)
    counts[3:6, 3:6] = 5  # 9 cell cluster, all hit 5 times
    refined = DensityRefineStep(
        DensityRefineParams(
            min_cell_hits=4,
            morph_open_radius_cells=0,
            morph_close_radius_cells=0,
            median_filter_size_cells=1,
        )
    ).run(_heatmap(counts))
    assert refined.metadata["cells_after_threshold"] == 9
    assert refined.metadata["cells_after_morph"] == 9


def test_density_returns_metadata_with_params() -> None:
    counts = np.zeros((5, 5), dtype=np.int32)
    refined = DensityRefineStep(DensityRefineParams(min_cell_hits=2)).run(_heatmap(counts))
    assert isinstance(refined, DensityRefined)
    assert refined.metadata["params"]["min_cell_hits"] == 2
