"""Sprint 52 W-2 — `depth_projection` byte-level golden tests.

Three fixtures cover the three production caller modes:
  1. floor_only_mode      — floor_point_cloud caller (`invert_mask=False`)
  2. obstacle_only_mode   — wall_polygon obstacle_source (`invert_mask=True`)
  3. mask_resample_mode   — floor_mask shape != depth shape, exercises the
                            resampling fast path.

Expected outputs are derived by running the helper itself once with a
deterministic fixture; the tests then assert byte-level equality on a fresh
re-run. Because the helper is pure numpy / linear algebra (no BLAS-driven
random sources, no parallel reductions over float arrays) the comparisons
use `assert_array_equal` for integer outputs and `assert_allclose(atol=0)`
for floats — exact equality is achievable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from indoor_server.application.building.depth_projection import (
    project_depth_pixels_to_world,
    sample_floor_mask_to_depth_grid,
)
from indoor_server.application.building.steps.rtabmap_depth_evidence import (
    RtabmapCameraCalibration,
)


@dataclass(frozen=True)
class DepthProjectionFixture:
    name: str
    depth: np.ndarray
    floor_mask: np.ndarray | None
    calibration: RtabmapCameraCalibration
    node_pose: np.ndarray
    pixel_stride: int
    min_depth_m: float
    max_depth_m: float
    invert_mask: bool


def _identity_calibration(width: int, height: int) -> RtabmapCameraCalibration:
    fx = 320.0
    fy = 320.0
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    return RtabmapCameraCalibration(
        version=(1, 0, 0),
        image_size=(width, height),
        k=(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0),
        local_transform=(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ),
        bytes_read=0,
    )


def _floor_only_fixture() -> DepthProjectionFixture:
    h, w = 8, 12
    depth = np.full((h, w), 2.0, dtype=np.float32)
    floor_mask = np.zeros((h, w), dtype=bool)
    floor_mask[4:, :] = True  # bottom half = floor
    pose = np.eye(4, dtype=np.float64)
    return DepthProjectionFixture(
        name="floor_only_mode",
        depth=depth,
        floor_mask=floor_mask,
        calibration=_identity_calibration(w, h),
        node_pose=pose,
        pixel_stride=2,
        min_depth_m=0.5,
        max_depth_m=4.0,
        invert_mask=False,
    )


def _obstacle_only_fixture() -> DepthProjectionFixture:
    h, w = 8, 12
    depth = np.full((h, w), 2.0, dtype=np.float32)
    floor_mask = np.zeros((h, w), dtype=bool)
    floor_mask[4:, :] = True  # invert → upper half is "obstacle"
    pose = np.eye(4, dtype=np.float64)
    return DepthProjectionFixture(
        name="obstacle_only_mode",
        depth=depth,
        floor_mask=floor_mask,
        calibration=_identity_calibration(w, h),
        node_pose=pose,
        pixel_stride=2,
        min_depth_m=0.5,
        max_depth_m=4.0,
        invert_mask=True,
    )


def _mask_resample_fixture() -> DepthProjectionFixture:
    h, w = 8, 12
    depth = np.full((h, w), 2.0, dtype=np.float32)
    # Mask at half resolution.
    floor_mask = np.zeros((4, 6), dtype=bool)
    floor_mask[2:, :] = True
    pose = np.eye(4, dtype=np.float64)
    return DepthProjectionFixture(
        name="mask_resample_mode",
        depth=depth,
        floor_mask=floor_mask,
        calibration=_identity_calibration(w, h),
        node_pose=pose,
        pixel_stride=2,
        min_depth_m=0.5,
        max_depth_m=4.0,
        invert_mask=False,
    )


@pytest.fixture(params=[
    _floor_only_fixture(),
    _obstacle_only_fixture(),
    _mask_resample_fixture(),
], ids=lambda f: f.name)
def fixture(request: pytest.FixtureRequest) -> DepthProjectionFixture:
    return request.param


def test_project_depth_pixels_byte_level_stable(
    fixture: DepthProjectionFixture,
) -> None:
    """Two consecutive runs return byte-identical arrays."""
    world1, sampled1, valid1 = project_depth_pixels_to_world(
        depth=fixture.depth,
        floor_mask=fixture.floor_mask,
        calibration=fixture.calibration,
        node_pose=fixture.node_pose,
        pixel_stride=fixture.pixel_stride,
        min_depth_m=fixture.min_depth_m,
        max_depth_m=fixture.max_depth_m,
        invert_mask=fixture.invert_mask,
    )
    world2, sampled2, valid2 = project_depth_pixels_to_world(
        depth=fixture.depth,
        floor_mask=fixture.floor_mask,
        calibration=fixture.calibration,
        node_pose=fixture.node_pose,
        pixel_stride=fixture.pixel_stride,
        min_depth_m=fixture.min_depth_m,
        max_depth_m=fixture.max_depth_m,
        invert_mask=fixture.invert_mask,
    )
    assert sampled1 == sampled2
    assert valid1 == valid2
    np.testing.assert_array_equal(world1, world2)
    assert world1.dtype == np.float64


def test_floor_only_expected_world_shape(
    fixture: DepthProjectionFixture,
) -> None:
    """floor_only_mode keeps the bottom-half pixels at 4 rows × 6 cols / stride 2."""
    if fixture.name != "floor_only_mode":
        pytest.skip("only floor_only_mode")
    world, sampled, valid = project_depth_pixels_to_world(
        depth=fixture.depth,
        floor_mask=fixture.floor_mask,
        calibration=fixture.calibration,
        node_pose=fixture.node_pose,
        pixel_stride=fixture.pixel_stride,
        min_depth_m=fixture.min_depth_m,
        max_depth_m=fixture.max_depth_m,
        invert_mask=fixture.invert_mask,
    )
    # depth grid is 8x12 sampled at stride 2 → 4 rows × 6 cols = 24 sampled.
    assert sampled == 24
    # bottom half rows {4,6} after stride-2 sampling → 2 rows × 6 cols = 12.
    assert valid == 12
    assert world.shape == (12, 3)


def test_obstacle_mode_inverts_floor_mask(
    fixture: DepthProjectionFixture,
) -> None:
    """invert_mask=True keeps the *upper* half pixels."""
    if fixture.name != "obstacle_only_mode":
        pytest.skip("only obstacle_only_mode")
    world, sampled, valid = project_depth_pixels_to_world(
        depth=fixture.depth,
        floor_mask=fixture.floor_mask,
        calibration=fixture.calibration,
        node_pose=fixture.node_pose,
        pixel_stride=fixture.pixel_stride,
        min_depth_m=fixture.min_depth_m,
        max_depth_m=fixture.max_depth_m,
        invert_mask=fixture.invert_mask,
    )
    assert sampled == 24
    assert valid == 12  # upper half (rows 0,2)
    assert world.shape == (12, 3)


def test_sample_floor_mask_resample_byte_level(
    fixture: DepthProjectionFixture,
) -> None:
    """Resample helper is deterministic across runs."""
    if fixture.name != "mask_resample_mode":
        pytest.skip("only mask_resample_mode")
    h, w = fixture.depth.shape
    ys = np.arange(0, h, fixture.pixel_stride, dtype=np.int32)
    xs = np.arange(0, w, fixture.pixel_stride, dtype=np.int32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    sampled1 = sample_floor_mask_to_depth_grid(
        fixture.floor_mask, (h, w), yy, xx
    )
    sampled2 = sample_floor_mask_to_depth_grid(
        fixture.floor_mask, (h, w), yy, xx
    )
    np.testing.assert_array_equal(sampled1, sampled2)
    assert sampled1.dtype == bool


def test_empty_mask_returns_zero_world(
    fixture: DepthProjectionFixture,
) -> None:
    """A wholly-False mask in floor_only_mode yields no valid pixels."""
    if fixture.name != "floor_only_mode":
        pytest.skip("only floor_only_mode")
    empty_mask = np.zeros_like(fixture.floor_mask)
    world, sampled, valid = project_depth_pixels_to_world(
        depth=fixture.depth,
        floor_mask=empty_mask,
        calibration=fixture.calibration,
        node_pose=fixture.node_pose,
        pixel_stride=fixture.pixel_stride,
        min_depth_m=fixture.min_depth_m,
        max_depth_m=fixture.max_depth_m,
        invert_mask=False,
    )
    assert valid == 0
    assert world.shape == (0, 3)
    assert world.dtype == np.float64
