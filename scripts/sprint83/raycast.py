"""
raycast.py — ARKit ray-cast: floor mask pixels → world (X, Z) points.

Coordinate convention:
  - ARKit camera frame: +X right, +Y up (phone top in portrait), +Z toward user (camera looks -Z).
    But intrinsics are stored as standard OpenCV pinhole (u→right, v→down, +Z forward in cam frame).
  - manifest intrinsics use standard OpenCV convention (+Z forward, +Y down).
  - d_cam = ((u-cx)/fx, (v-cy)/fy, 1.0)  [OpenCV forward = ARKit's "screen space"]
  - World frame: +Y up. floor plane = { y = y0 }.
  - Node.pose (from merged_v2.db): 3×4 [R|t], T_world_cam meaning:
      p_world = R @ p_cam + t
      cam_pos_world = t (3rd column)
"""

from __future__ import annotations

import numpy as np


def estimate_floor_y(poses_ty: np.ndarray, hand_height_m: float = 1.4) -> float:
    """Per-scan floor y0 = mean(camera ty) - hand_height_m."""
    return float(np.mean(poses_ty) - hand_height_m)


def raycast_floor_pixels(
    mask: np.ndarray,          # bool (H, W)
    K: np.ndarray,             # (3, 3) intrinsic
    T_world_cam: np.ndarray,   # (4, 4) or (3, 4) world←cam
    y0: float,
    stride: int = 4,
    lambda_max: float = 5.0,
) -> np.ndarray:
    """
    Cast rays from camera through floor-masked pixels.
    Returns (N, 2) float32 array of world (X, Z) positions.
    """
    H, W = mask.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Rotation and translation from pose
    if T_world_cam.shape == (4, 4):
        R = T_world_cam[:3, :3]
        t = T_world_cam[:3, 3]
    else:  # (3, 4)
        R = T_world_cam[:3, :3]
        t = T_world_cam[:3, 3]

    cam_y = t[1]  # camera world Y position

    # Pixel grid with stride
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    # Apply mask (downsample mask to same grid)
    mask_strided = mask[::stride, ::stride]
    vs = vs[mask_strided]
    us = us[mask_strided]

    if len(vs) == 0:
        return np.empty((0, 2), dtype=np.float32)

    # Pixel → camera direction (ARKit convention, -Z forward)
    # Empirically verified: ARKit world frame +Y = up, camera looks in -Z direction.
    # d_cam = ((u-cx)/fx, (v-cy)/fy, -1.0)
    dx = (us.astype(np.float32) - cx) / fx
    dy = (vs.astype(np.float32) - cy) / fy
    dz = -np.ones(len(us), dtype=np.float32)

    d_cam = np.stack([dx, dy, dz], axis=1)  # (N, 3)

    # Camera → world direction
    # R: (3,3), d_cam: (N,3) → d_world: (N,3)
    d_world = d_cam @ R.T  # (N, 3)

    # Ray-floor intersection: y = y0
    # t (scalar) = (y0 - cam_y) / d_world_y
    dwy = d_world[:, 1]

    # Keep rays pointing toward floor
    valid = dwy != 0
    lam = np.where(valid, (y0 - cam_y) / dwy, np.inf)

    # Filter: λ > 0 (floor in front), λ ≤ λ_max
    keep = (lam > 0) & (lam <= lambda_max)

    lam_k = lam[keep]
    d_world_k = d_world[keep]

    tx, ty_cam, tz = t[0], t[1], t[2]
    X = tx + lam_k * d_world_k[:, 0]
    Z = tz + lam_k * d_world_k[:, 2]

    points = np.stack([X, Z], axis=1).astype(np.float32)
    return points


def raycast_scan(
    masks: list[np.ndarray],
    poses: list[np.ndarray],   # list of (4,4) or (3,4) T_world_cam
    K: np.ndarray,
    y0: float,
    stride: int = 4,
    lambda_max: float = 5.0,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Raycast all frames of a scan. Returns (world_points (N,2), stats).
    """
    import time
    t0 = time.time()
    all_pts = []
    total_mask_px = 0
    total_valid = 0
    lam_neg_count = 0
    lam_over_count = 0

    for i, (mask, pose) in enumerate(zip(masks, poses)):
        pts = raycast_floor_pixels(mask, K, pose, y0, stride, lambda_max)
        all_pts.append(pts)
        n_floor = int(mask[::stride, ::stride].sum())
        total_mask_px += n_floor
        total_valid += len(pts)

        if verbose and i % 50 == 0:
            elapsed = time.time() - t0
            print(f"[raycast] frame {i}/{len(masks)}, pts so far={sum(len(p) for p in all_pts)}, {elapsed:.1f}s")

    if all_pts:
        world_points = np.concatenate(all_pts, axis=0)
    else:
        world_points = np.empty((0, 2), dtype=np.float32)

    elapsed = time.time() - t0
    stats = {
        "total_frames": len(masks),
        "total_mask_px": total_mask_px,
        "total_world_points": len(world_points),
        "elapsed_s": round(elapsed, 2),
        "y0": round(y0, 4),
        "lambda_max": lambda_max,
        "stride": stride,
    }
    if len(world_points) > 0:
        stats["x_range"] = [round(float(world_points[:, 0].min()), 3), round(float(world_points[:, 0].max()), 3)]
        stats["z_range"] = [round(float(world_points[:, 1].min()), 3), round(float(world_points[:, 1].max()), 3)]
    print(
        f"[raycast] done: {len(world_points)} world points from {len(masks)} frames in {elapsed:.1f}s"
    )
    return world_points, stats


def load_poses_from_index(index: list[dict]) -> list[np.ndarray]:
    """Load 4×4 ARKit poses from index.json entries."""
    poses = []
    for entry in index:
        p = np.array(entry["pose_world_arkit"], dtype=np.float64)
        poses.append(p)
    return poses


def db_pose_to_arkit(blob: bytes) -> np.ndarray:
    """
    Convert RTAB-Map merged_v2.db 3×4 float32 blob → ARKit 4×4 world←cam pose.

    DB coordinate convention (empirically verified):
      DB_row0 = ARKit_row0  (X unchanged)
      DB_row1 = -ARKit_row2 (negated ARKit Z row)
      DB_row2 = ARKit_row1  (ARKit Y row)
    Translation:
      arkit_tx = db_t[0]
      arkit_ty = db_t[2]
      arkit_tz = -db_t[1]
    """
    import struct
    vals = struct.unpack("<12f", blob)
    R3x4 = np.array(vals, dtype=np.float64).reshape(3, 4)
    R_db = R3x4[:3, :3]
    t_db = R3x4[:3, 3]

    R_arkit = np.array([R_db[0], R_db[2], -R_db[1]], dtype=np.float64)
    t_arkit = np.array([t_db[0], t_db[2], -t_db[1]], dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_arkit
    T[:3, 3] = t_arkit
    return T


def load_poses_from_db(db_path: str, map_id: int) -> tuple[list[int], list[np.ndarray]]:
    """
    Load Node poses from merged_v2.db for a given map_id.
    Returns (node_ids, poses list of (4,4) ARKit-space float64).
    DB poses are converted via db_pose_to_arkit().
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
        T = db_pose_to_arkit(blob)
        node_ids.append(nid)
        poses.append(T)
    return node_ids, poses
