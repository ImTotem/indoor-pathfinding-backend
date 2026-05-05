"""
raycast_to_world.py — Cycle 2 axis-fixed ray-cast.

Coordinate convention (DB frame = RTAB-Map world frame, Z-up):
  - DB Node.pose = T_world_cam: R[3x3] maps cam axes → world, t = cam_pos_world.
  - Camera frame = OpenCV (+X right, +Y down, +Z forward). No sign inversion.
  - d_cam = ((u-cx)/fx, (v-cy)/fy, 1.0)   [OpenCV pinhole, +Z forward]
  - d_world = R @ d_cam
  - world up = +Z.  floor plane z = z0.
  - λ = (z0 - tz) / d_world[2]
  - Drop: d_world[2] >= -1e-3  (ray not pointing downward)
         λ <= 0                 (floor behind camera)
         λ > λ_max              (too far, noise-dominated)
  - Output world point: (X, Y) = corridor horizontal plane.
    X = tx + λ*d_world[0],  Y = ty + λ*d_world[1].
"""

from __future__ import annotations

import struct

import numpy as np


def load_db_poses_raw(db_path: str, map_id: int) -> tuple[list[int], list[np.ndarray]]:
    """
    Load Node poses from merged_v2.db for a given map_id.
    Returns (node_ids, poses) where each pose is (3,4) float64 in DB world frame.
    No coordinate conversion applied — raw DB frame.
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, pose FROM Node WHERE map_id=? ORDER BY id", (map_id,))
    rows = c.fetchall()
    conn.close()

    node_ids = []
    poses = []
    for nid, blob in rows:
        vals = struct.unpack("<12f", blob)
        mat = np.array(vals, dtype=np.float64).reshape(3, 4)  # (3,4)
        node_ids.append(nid)
        poses.append(mat)
    return node_ids, poses


def raycast_frame(
    mask: np.ndarray,       # bool (H, W)
    K: np.ndarray,          # (3,3) intrinsic
    pose: np.ndarray,       # (3,4) or (4,4) T_world_cam in DB Z-up frame
    z0: float,
    stride: int = 4,
    lambda_max: float = 5.0,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Cast rays from camera through floor-masked pixels.
    Returns (N, 2) float32 array of world (X, Y) positions on floor plane z=z0.

    Camera frame = OpenCV (+Z forward). No ARKit -Z inversion.
    Drop condition: d_world[2] >= -eps  (ray not going downward in Z-up world).
    """
    H, W = mask.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    R = pose[:3, :3]   # (3,3)
    t = pose[:3, 3]    # (3,)
    tz = t[2]          # camera world Z (height)

    # Pixel grid with stride
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    mask_strided = mask[::stride, ::stride]
    vs = vs[mask_strided]
    us = us[mask_strided]

    if len(vs) == 0:
        return np.empty((0, 2), dtype=np.float32)

    # Pixel → camera ray direction (OpenCV: +Z forward, no sign inversion)
    dx = (us.astype(np.float32) - cx) / fx
    dy = (vs.astype(np.float32) - cy) / fy
    dz = np.ones(len(us), dtype=np.float32)  # +Z forward (OpenCV)

    d_cam = np.stack([dx, dy, dz], axis=1)  # (N, 3)

    # Camera → world direction
    d_world = d_cam @ R.T  # (N, 3)

    dwz = d_world[:, 2]  # world Z component of each ray

    # Drop rays not pointing downward (Z-up world: floor-pointing ray has dwz < -eps)
    valid_dir = dwz < -eps

    # λ = (z0 - tz) / dwz  for valid rays
    lam = np.where(valid_dir, (z0 - tz) / dwz, -1.0)

    # Drop: λ not in (0, λ_max]
    keep = valid_dir & (lam > 0) & (lam <= lambda_max)

    lam_k = lam[keep]
    d_world_k = d_world[keep]

    tx, ty = t[0], t[1]
    X = tx + lam_k * d_world_k[:, 0]
    Y = ty + lam_k * d_world_k[:, 1]

    return np.stack([X, Y], axis=1).astype(np.float32)


def raycast_scan(
    masks: list[np.ndarray],
    poses: list[np.ndarray],  # (3,4) DB Z-up frame poses
    K: np.ndarray,
    z0: float,
    stride: int = 4,
    lambda_max: float = 5.0,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Raycast all frames. Returns (world_points (N,2) XY, stats dict).
    """
    import time
    t0 = time.time()
    all_pts = []
    total_mask_px = 0
    total_valid = 0
    total_drop_dir = 0
    total_drop_lam = 0
    lam_vals_sample = []

    for i, (mask, pose) in enumerate(zip(masks, poses)):
        H, W = mask.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        R = pose[:3, :3]
        t_vec = pose[:3, 3]
        tz = t_vec[2]

        vs, us = np.mgrid[0:H:stride, 0:W:stride]
        mask_strided = mask[::stride, ::stride]
        vs_m = vs[mask_strided]
        us_m = us[mask_strided]
        n_floor = len(vs_m)
        total_mask_px += n_floor

        if n_floor == 0:
            all_pts.append(np.empty((0, 2), dtype=np.float32))
            continue

        dx = (us_m.astype(np.float32) - cx) / fx
        dy = (vs_m.astype(np.float32) - cy) / fy
        dz_cam = np.ones(n_floor, dtype=np.float32)
        d_cam = np.stack([dx, dy, dz_cam], axis=1)
        d_world = d_cam @ R.T

        dwz = d_world[:, 2]
        valid_dir = dwz < -1e-3
        n_drop_dir = int((~valid_dir).sum())
        total_drop_dir += n_drop_dir

        lam = np.where(valid_dir, (z0 - tz) / dwz, -1.0)
        keep = valid_dir & (lam > 0) & (lam <= lambda_max)
        n_drop_lam = int((valid_dir & ~keep).sum())
        total_drop_lam += n_drop_lam

        lam_k = lam[keep]
        d_world_k = d_world[keep]
        X = t_vec[0] + lam_k * d_world_k[:, 0]
        Y = t_vec[1] + lam_k * d_world_k[:, 1]
        pts = np.stack([X, Y], axis=1).astype(np.float32)
        all_pts.append(pts)
        total_valid += len(pts)

        # Collect sample λ values for histogram (first 100 valid frames)
        if i < 100 and len(lam_k) > 0:
            lam_vals_sample.extend(lam_k[:10].tolist())

        if verbose and i % 50 == 0:
            elapsed = time.time() - t0
            print(f"[raycast] frame {i}/{len(masks)}, pts={total_valid}, {elapsed:.1f}s")

    if all_pts:
        world_points = np.concatenate(all_pts, axis=0)
    else:
        world_points = np.empty((0, 2), dtype=np.float32)

    elapsed = time.time() - t0
    lam_arr = np.array(lam_vals_sample, dtype=np.float32)

    stats = {
        "total_frames": len(masks),
        "total_mask_px": total_mask_px,
        "total_world_points": len(world_points),
        "drop_dir_count": total_drop_dir,
        "drop_lam_count": total_drop_lam,
        "valid_count": total_valid,
        "lambda_gt0_ratio": round(float(total_valid / max(1, total_mask_px - total_drop_dir)), 4),
        "lambda_valid_ratio": round(float(total_valid / max(1, total_mask_px)), 4),
        "elapsed_s": round(elapsed, 2),
        "z0": round(z0, 4),
        "lambda_max": lambda_max,
        "stride": stride,
    }
    if len(lam_arr) > 0:
        stats["lambda_sample_mean"] = round(float(lam_arr.mean()), 3)
        stats["lambda_sample_min"] = round(float(lam_arr.min()), 3)
        stats["lambda_sample_max"] = round(float(lam_arr.max()), 3)
    if len(world_points) > 0:
        stats["x_range"] = [round(float(world_points[:, 0].min()), 3),
                            round(float(world_points[:, 0].max()), 3)]
        stats["y_range"] = [round(float(world_points[:, 1].min()), 3),
                            round(float(world_points[:, 1].max()), 3)]
    print(f"[raycast] done: {len(world_points)} world(X,Y) pts from {len(masks)} frames in {elapsed:.1f}s")
    return world_points, stats
