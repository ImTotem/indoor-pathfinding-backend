"""Sprint 51 — Shared RTABMap depth → world frame projection helper.

Extracted from `floor_point_cloud._project_floor_pixels_to_world` and
`rtabmap_depth_evidence._project_depth_confidence` (Sprint 46 TODO marker).

Both call sites previously inlined the same algorithm:
  1. uniform pixel grid sampling (`pixel_stride` apart)
  2. optional floor mask gating (with shape-mismatch resampling)
  3. depth value range filter (`min_depth_m <= z <= max_depth_m`)
  4. camera-frame 3D back-projection using scaled intrinsics
  5. local_transform → node_pose multiplication to get world xyz

Centralising here guarantees both callers stay byte-identical and any future
fix lands once. Inputs/outputs and contract are unchanged from the inlined
version (golden tests verify byte-level equality).
"""
from __future__ import annotations

import numpy as np

from indoor_server.application.building.steps.rtabmap_depth_evidence import (
    RtabmapCameraCalibration,
)


def sample_floor_mask_to_depth_grid(
    floor_mask: np.ndarray,
    depth_shape: tuple[int, int],
    yy: np.ndarray,
    xx: np.ndarray,
) -> np.ndarray:
    """Resample floor mask to align with depth pixel grid.

    Mirrors `floor_point_cloud._sample_floor_mask` and
    `rtabmap_depth_evidence._sample_floor_mask` (which were duplicates).
    """
    if floor_mask.shape != depth_shape:
        y_scale = floor_mask.shape[0] / float(depth_shape[0])
        x_scale = floor_mask.shape[1] / float(depth_shape[1])
        my = np.clip((yy * y_scale).astype(np.int32), 0, floor_mask.shape[0] - 1)
        mx = np.clip((xx * x_scale).astype(np.int32), 0, floor_mask.shape[1] - 1)
        sampled = floor_mask[my, mx].astype(bool)
        return np.asarray(sampled, dtype=bool)
    sampled = floor_mask[yy, xx].astype(bool)
    return np.asarray(sampled, dtype=bool)


def project_depth_pixels_to_world(
    *,
    depth: np.ndarray,
    floor_mask: np.ndarray | None,
    calibration: RtabmapCameraCalibration,
    node_pose: np.ndarray,
    pixel_stride: int,
    min_depth_m: float,
    max_depth_m: float,
    invert_mask: bool = False,
) -> tuple[np.ndarray, int, int]:
    """Project depth pixel grid into the world frame.

    Args:
        depth: (H, W) float depth in meters.
        floor_mask: (H', W') bool mask (resampled to depth shape internally).
            None means "no mask gate" (every depth pixel passes the mask
            criterion). For obstacle source we pass `floor_mask` and set
            `invert_mask=True` so non-floor pixels are kept.
        calibration: RTABMap camera calibration (used for scaled intrinsics
            and local_transform).
        node_pose: (4, 4) world ← local SE(3).
        pixel_stride: integer stride for pixel grid sampling.
        min_depth_m / max_depth_m: depth range filter (inclusive).
        invert_mask: if True and floor_mask is given, keep pixels where
            mask is False (i.e. non-floor → obstacle). Default False keeps
            existing floor_pointcloud / depth_evidence behaviour.

    Returns:
        (world_xyz (N, 3) float64, sampled_pixel_count int,
         valid_depth_pixel_count int).
        Returns empty (0, 3) array when nothing passes filters.
    """
    h, w = depth.shape
    ys = np.arange(0, h, pixel_stride, dtype=np.int32)
    xs = np.arange(0, w, pixel_stride, dtype=np.int32)
    if ys.size == 0 or xs.size == 0:
        return np.zeros((0, 3), dtype=np.float64), 0, 0
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    sampled_pixel_count = int(yy.size)

    z = depth[yy, xx].astype(np.float64)
    valid = (
        np.isfinite(z)
        & (z >= min_depth_m)
        & (z <= max_depth_m)
    )
    if floor_mask is not None:
        sampled_mask = sample_floor_mask_to_depth_grid(
            floor_mask, depth.shape, yy, xx
        )
        if invert_mask:
            valid &= ~sampled_mask
        else:
            valid &= sampled_mask

    valid_count = int(valid.sum())
    if valid_count == 0:
        return np.zeros((0, 3), dtype=np.float64), sampled_pixel_count, 0

    fx, fy, cx, cy = calibration.scaled_intrinsics(width=w, height=h)
    z_valid = z[valid]
    x_cam = (xx[valid].astype(np.float64) - cx) * z_valid / fx
    y_cam = (yy[valid].astype(np.float64) - cy) * z_valid / fy
    points_cam = np.column_stack(
        [
            x_cam,
            y_cam,
            z_valid,
            np.ones(valid_count, dtype=np.float64),
        ]
    )
    local = (calibration.local_transform_matrix() @ points_cam.T).T
    world = (node_pose @ local.T).T[:, :3]
    return world.astype(np.float64), sampled_pixel_count, valid_count
