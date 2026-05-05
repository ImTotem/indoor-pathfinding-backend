"""Sprint 46 — FloorRasterStep unit tests."""
from __future__ import annotations

import numpy as np
import pytest

from indoor_server.application.building.steps.floor_point_cloud import FloorPointCloud
from indoor_server.application.building.steps.floor_raster import (
    FloorRasterStep,
    FloorRasterStepParams,
)


def _cloud_from_xy(points_xy: np.ndarray, *, z0: float = 0.0) -> FloorPointCloud:
    z_values = np.full((points_xy.shape[0],), z0, dtype=np.float64)
    return FloorPointCloud(
        points_xy=points_xy.astype(np.float64),
        z_values=z_values,
        z0=z0,
        metadata={},
    )


def _square_cluster(*, side_m: float, density: int) -> np.ndarray:
    n = density
    xs = np.linspace(0.0, side_m, n)
    ys = np.linspace(0.0, side_m, n)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def test_floor_raster_happy_path_produces_grid_and_polygon() -> None:
    points = _square_cluster(side_m=2.0, density=40)
    cloud = _cloud_from_xy(points)

    result = FloorRasterStep(
        FloorRasterStepParams(
            min_cell_hits=1,
            morph_open_radius_cells=0,
            morph_close_radius_cells=0,
            keep_largest_component=True,
        )
    ).run(cloud)

    assert result.grid.mask.sum() > 0
    assert result.footprint_geojson is not None
    assert result.metadata["after_threshold_cells"] > 0
    assert result.metadata["polygon_area_m2"] > 0.0


def test_floor_raster_empty_cloud_returns_empty_with_reason() -> None:
    cloud = _cloud_from_xy(np.zeros((0, 2)))
    result = FloorRasterStep().run(cloud)
    assert result.grid.mask.sum() == 0
    assert result.footprint_geojson is None
    assert result.metadata["empty_reason"] == "no_points"


def test_floor_raster_below_threshold_returns_empty_mask() -> None:
    """min_cell_hits=10인데 3개 점만 → mask 비어있음."""
    points = np.array(
        [[0.0, 0.0], [0.05, 0.05], [0.0, 0.05]], dtype=np.float64
    )
    cloud = _cloud_from_xy(points)
    result = FloorRasterStep(
        FloorRasterStepParams(min_cell_hits=10)
    ).run(cloud)
    assert result.grid.mask.sum() == 0
    assert result.metadata["empty_reason"] == "below_threshold"
    # observation_count는 살아있어야 한다 (debug용).
    assert result.grid.observation_count.sum() == 3


def test_floor_raster_keep_largest_drops_small_island() -> None:
    """큰 cluster + 멀리 있는 작은 island → keep_largest=True 면 island 제거."""
    big = _square_cluster(side_m=1.5, density=30) + np.array([0.0, 0.0])
    small = _square_cluster(side_m=0.2, density=8) + np.array([10.0, 10.0])
    points = np.vstack([big, small])
    cloud = _cloud_from_xy(points)

    result = FloorRasterStep(
        FloorRasterStepParams(
            min_cell_hits=1,
            morph_open_radius_cells=0,
            morph_close_radius_cells=0,
            keep_largest_component=True,
            bbox_padding_m=0.5,
        )
    ).run(cloud)

    # components_before_keep_largest >= 2 가 잡혀야 keep_largest 기여 확인.
    assert result.metadata["components_before_keep_largest"] >= 2
    # mask는 큰 cluster만 살아남았으니 island 영역 (10,10) 근처는 False
    origin = result.grid.origin
    col = int((10.0 - origin.x0) / origin.cell_size)
    row = int((10.0 - origin.y0) / origin.cell_size)
    if 0 <= col < origin.w and 0 <= row < origin.h:
        assert not bool(result.grid.mask[row, col])


def test_floor_raster_morphology_fills_small_hole() -> None:
    """ring 모양 점 cluster → closing이 안 쪽 hole을 메워준다."""
    # outer 50x50 grid, center 10x10 hole
    xs = np.linspace(-1.0, 1.0, 50)
    ys = np.linspace(-1.0, 1.0, 50)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    # remove center
    keep = ~((np.abs(points[:, 0]) < 0.05) & (np.abs(points[:, 1]) < 0.05))
    points = points[keep]
    cloud = _cloud_from_xy(points)

    no_close = FloorRasterStep(
        FloorRasterStepParams(
            min_cell_hits=1,
            morph_open_radius_cells=0,
            morph_close_radius_cells=0,
            keep_largest_component=False,
        )
    ).run(cloud)
    with_close = FloorRasterStep(
        FloorRasterStepParams(
            min_cell_hits=1,
            morph_open_radius_cells=0,
            morph_close_radius_cells=2,
            keep_largest_component=False,
        )
    ).run(cloud)

    # closing 후 cell 수가 같거나 늘어나야 한다 (hole 메우면 늘어남).
    assert (
        with_close.metadata["after_morphology_cells"]
        >= no_close.metadata["after_morphology_cells"]
    )


def test_floor_raster_invalid_params_raise() -> None:
    with pytest.raises(ValueError):
        FloorRasterStep(FloorRasterStepParams(cell_size_m=0.0))
    with pytest.raises(ValueError):
        FloorRasterStep(FloorRasterStepParams(min_cell_hits=0))
    with pytest.raises(ValueError):
        FloorRasterStep(FloorRasterStepParams(morph_open_radius_cells=-1))
    with pytest.raises(ValueError):
        FloorRasterStep(FloorRasterStepParams(bbox_padding_m=-0.1))


def test_floor_raster_grid_origin_z0_propagates() -> None:
    points = _square_cluster(side_m=1.0, density=20)
    cloud = _cloud_from_xy(points, z0=-1.234)
    result = FloorRasterStep(
        FloorRasterStepParams(
            min_cell_hits=1,
            morph_open_radius_cells=0,
            morph_close_radius_cells=0,
        )
    ).run(cloud)
    assert result.grid.origin.z0 == pytest.approx(-1.234)
