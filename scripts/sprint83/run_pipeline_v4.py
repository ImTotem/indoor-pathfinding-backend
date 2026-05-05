#!/usr/bin/env python3
"""
run_pipeline_v4.py — Sprint 83 cycle_4 entrypoint (Option B: portrait mask back-transform).

Changes vs cycle_3:
  - K' REMOVED. Portrait mask pixel (u', v') → landscape pixel (u, v) via back-transform.
  - Back-transform formula (CCW 90° unrotation inverse):
      u = H' - 1 - v'   # H'=1920 (portrait height), u is landscape col
      v = u'             # u' is portrait col, v is landscape row
  - Landscape K (per-scan original): A=(fx=1360.562, cx=967.84, cy=538.95), B=(fx=1276.55, ...)
  - R matrix unchanged (original DB landscape R).
  - cycle_2 axis fix: Z-up, z0_A=-1.4601, z0_B=-1.3238.
  - λ drop: λ <= 0 OR λ > 5m OR d_world.z >= -1e-3.

Reuses: cycle_3 portrait masks_A.npz, masks_B.npz (NO SegFormer re-inference).

Output: _workspace/sprint83-floor-segmentation-heatmap/evidence/cycle4/

NOTE (pre-run analysis):
  cycle_3 portrait masks have floor pixels concentrated at portrait BOTTOM (v' mean ~1682).
  Back-transform: landscape_col = H'-1-v' = 1919-v' → col range [0, 663] (LEFT side).
  For these frames, R[2,0] ~ -0.996 and dx = (col-968)/1360 < 0 → d_world.z = R[2,0]*dx + ... > 0.
  Expected PASS rate: ~0% (same as cycle_3). AC3 target ≥50% is NOT achievable with these masks.
  This is a design blocker: cycle_3 portrait SegFormer detections are in wrong quadrant.
  Correct floor in portrait should be at v' ~ [349, 1582] (cycle_1 landscape floor mapped to portrait).
  Actual cycle_3 detections: v' ~ [1256, 1919] — shifted by ~300-400 rows too far bottom.
  Valid floor pixels (v' < 919 → landscape_col > 1000 → dx > 0) count: ~2/314 frames only.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = "/Users/leehyeonsu/home/koreatech/graduate_project/Indoor-pathfinding-v2"

EVIDENCE_DIR_C1 = os.path.join(
    PROJECT_ROOT, "_workspace/sprint83-floor-segmentation-heatmap/evidence/cycle1"
)
EVIDENCE_DIR_C2 = os.path.join(
    PROJECT_ROOT, "_workspace/sprint83-floor-segmentation-heatmap/evidence/cycle2"
)
EVIDENCE_DIR_C3 = os.path.join(
    PROJECT_ROOT, "_workspace/sprint83-floor-segmentation-heatmap/evidence/cycle3"
)
EVIDENCE_DIR = os.path.join(
    PROJECT_ROOT, "_workspace/sprint83-floor-segmentation-heatmap/evidence/cycle4"
)
MERGED_DB = os.path.join(
    PROJECT_ROOT,
    "_workspace/sprint81-frame-pose-rtabmap-merge/evidence/cycle2/merged_v2.db",
)
SPRINT82_POLYGON = os.path.join(
    PROJECT_ROOT,
    "_workspace/sprint82-floor-polygon-navgraph/evidence/cycle1/polygon/floor_polygon_refined.geojson",
)
CYCLE1_POLYGON = os.path.join(
    EVIDENCE_DIR_C1, "polygon/floor_polygon_seg.geojson"
)
CYCLE2_POLYGON = os.path.join(
    EVIDENCE_DIR_C2, "polygon/floor_polygon_seg_v2.geojson"
)
CYCLE3_POLYGON = os.path.join(
    EVIDENCE_DIR_C3, "polygon/floor_polygon_seg_v3.geojson"
)
CALIB_A = os.path.join(
    PROJECT_ROOT,
    "_workspace/sprint81-frame-pose-rtabmap-merge/evidence/cycle1/calib_A.json",
)
CALIB_B = os.path.join(
    PROJECT_ROOT,
    "_workspace/sprint81-frame-pose-rtabmap-merge/evidence/cycle1/calib_B.json",
)

# Portrait mask paths (cycle_3 cached — NO re-inference)
MASKS_A_PATH = os.path.join(EVIDENCE_DIR_C3, "masks_A.npz")
MASKS_B_PATH = os.path.join(EVIDENCE_DIR_C3, "masks_B.npz")

# ---------------------------------------------------------------------------
# Settings (same as cycle_3 except K' removed → back-transform)
# ---------------------------------------------------------------------------
LAMBDA_MAX = 5.0
BIN_SIZE = 0.2
FLOOR_CLASS_IDS = [3]
STRIDE = 4
MIN_COUNT = 3

# Z-up axis fix values (from cycle_2/3 preflight)
Z0_A = -1.4601   # tz_min_A - 1.4 (cycle_3 preflight z0_A)
Z0_B = -1.3238   # tz_min_B - 1.4 (cycle_3 preflight z0_B)

# Landscape K values (per-scan original, from calib JSON)
# Scan A: fx=fy=1360.562, cx=967.84314, cy=538.9509, W=1920, H=1080
# Scan B: fx=fy=1276.5535, cx=967.3513, cy=539.65106, W=1920, H=1080
K_A_LANDSCAPE = {"fx": 1360.562, "fy": 1360.562, "cx": 967.84314, "cy": 538.9509}
K_B_LANDSCAPE = {"fx": 1276.5535, "fy": 1276.5535, "cx": 967.3513, "cy": 539.65106}

# Portrait mask dimensions (cycle_3 confirmed)
H_PRIME = 1920   # portrait height (= landscape width)
W_PRIME = 1080   # portrait width  (= landscape height)

# cycle_2 sample frame IDs (for PASS rate comparison)
CYCLE2_SAMPLE_FRAMES_A = [0, 34, 69, 104, 139, 173, 208, 243, 278, 313]
CYCLE2_SAMPLE_FRAMES_B = [0, 32, 65, 97, 130, 162, 195, 227, 260, 293]

# Trajectory bbox for near-check
TRAJ_XMIN, TRAJ_XMAX = -9.0, 45.0
TRAJ_YMIN, TRAJ_YMAX = -70.0, 10.0


# ---------------------------------------------------------------------------
# Back-transform: portrait (u', v') → landscape (u, v)
# ---------------------------------------------------------------------------
def back_transform_portrait_to_landscape(
    portrait_us: np.ndarray,   # portrait col coordinates (u')
    portrait_vs: np.ndarray,   # portrait row coordinates (v')
    h_prime: int = H_PRIME,
) -> tuple[np.ndarray, np.ndarray]:
    """
    CCW 90° unrotation inverse:
        landscape_col (u) = H' - 1 - v'   (H'=1920)
        landscape_row (v) = u'

    Derivation:
      cv2.ROTATE_90_COUNTERCLOCKWISE: portrait(u', v') came from landscape(1919-v', u')
      Inverse: landscape_col = H'-1-v', landscape_row = u'

    Hand-check (cycle_4 build report):
      Portrait floor pixel (u'=540, v'=119) → landscape (1919-119, 540) = (1800, 540)
      d_cam = ((1800-968)/1360, (540-539)/1360, 1) = (0.612, 0.001, 1) — positive dx → floor pointing
      portrait floor pixel (u'=988, v'=1028) → landscape (1919-1028, 988) = (891, 988)
      d_cam = ((891-968)/1360, (988-539)/1360, 1) = (-0.057, 0.330, 1) — near-center → valid
    """
    u_land = (h_prime - 1 - portrait_vs).astype(np.float32)   # landscape col
    v_land = portrait_us.astype(np.float32)                     # landscape row
    return u_land, v_land


def load_poses(db_path: str, map_id: int) -> list[np.ndarray]:
    import struct
    import sqlite3
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, pose FROM Node WHERE map_id=? ORDER BY id", (map_id,))
    rows = c.fetchall()
    conn.close()
    poses = []
    for nid, blob in rows:
        vals = struct.unpack("<12f", blob)
        poses.append(np.array(vals, dtype=np.float64).reshape(3, 4))
    return poses


# ---------------------------------------------------------------------------
# Step 0: Preflight — confirm mask shape, K, z0
# ---------------------------------------------------------------------------
def step0_preflight() -> dict:
    print("\n=== Step 0: Preflight (cycle_4 back-transform) ===")

    # Load masks to confirm shape
    arr_A = np.load(MASKS_A_PATH)["masks"]
    arr_B = np.load(MASKS_B_PATH)["masks"]
    print(f"  masks_A shape: {arr_A.shape} (portrait: H'={arr_A.shape[1]}, W'={arr_A.shape[2]})")
    print(f"  masks_B shape: {arr_B.shape}")

    assert arr_A.shape[1] == H_PRIME, f"Expected H'={H_PRIME}, got {arr_A.shape[1]}"
    assert arr_A.shape[2] == W_PRIME, f"Expected W'={W_PRIME}, got {arr_A.shape[2]}"

    # Hand-check back-transform with one example pixel
    # Portrait (u'=540, v'=119) → landscape (1919-119, 540) = (1800, 540)
    u_ex, v_ex = back_transform_portrait_to_landscape(
        np.array([540], dtype=float), np.array([119], dtype=float)
    )
    print(f"  Back-transform check: portrait (u'=540, v'=119) → landscape (u={u_ex[0]:.0f}, v={v_ex[0]:.0f})")
    print(f"    Expected: (1800, 540). Match: {abs(u_ex[0]-1800) < 0.5 and abs(v_ex[0]-540) < 0.5}")

    # Verify z0 values
    with open(CALIB_A) as f:
        calib_A = json.load(f)
    with open(CALIB_B) as f:
        calib_B = json.load(f)
    print(f"  Landscape K_A: fx={calib_A['fx']}, cx={calib_A['cx']}, cy={calib_A['cy']}")
    print(f"  Landscape K_B: fx={calib_B['fx']}, cx={calib_B['cx']}, cy={calib_B['cy']}")
    print(f"  z0_A={Z0_A}m, z0_B={Z0_B}m (from cycle_3 preflight)")

    # Portrait floor location analysis — key for blocker understanding
    frame278 = arr_A[278]
    ys278, xs278 = np.where(frame278)
    if len(ys278) > 0:
        u_land278, v_land278 = back_transform_portrait_to_landscape(xs278, ys278)
        print(f"\n  Frame 278 portrait floor: v'=[{ys278.min()},{ys278.max()}] mean={ys278.mean():.1f}")
        print(f"  After back-transform: landscape_col=[{u_land278.min():.0f},{u_land278.max():.0f}] mean={u_land278.mean():.1f}")
        print(f"  dx = (col - cx) / fx: mean={(u_land278.mean()-calib_A['cx'])/calib_A['fx']:.3f}")
        print(f"  Note: dx<0 → R[2,0]*dx > 0 → d_world.z > 0 → rays invalid")

    # Global portrait floor mean-v distribution
    mean_vs = []
    for i in range(len(arr_A)):
        m = arr_A[i]
        ys, _ = np.where(m)
        if len(ys) > 0:
            mean_vs.append(float(ys.mean()))
    arr_mv = np.array(mean_vs)
    below_919 = int((arr_mv < 919).sum())
    print(f"\n  Portrait floor mean-v < 919 (→ landscape col>1000, valid ray): {below_919}/{len(arr_mv)} frames")
    print(f"  Portrait floor mean-v overall: mean={arr_mv.mean():.1f}, min={arr_mv.min():.1f}")
    print(f"  Key insight: {below_919}/{len(arr_mv)} frames can produce valid rays with Option B")

    result = {
        "cycle": "cycle_4",
        "method": "portrait_mask_back_transform",
        "back_transform_formula": {
            "landscape_col_u": "H_prime - 1 - portrait_row_v_prime",
            "landscape_row_v": "portrait_col_u_prime",
            "H_prime": H_PRIME,
        },
        "landscape_K_A": K_A_LANDSCAPE,
        "landscape_K_B": K_B_LANDSCAPE,
        "z0_A": Z0_A,
        "z0_B": Z0_B,
        "up_axis": "+Z",
        "lambda_max": LAMBDA_MAX,
        "stride": STRIDE,
        "portrait_mask_source": "cycle_3 (no re-inference)",
        "masks_A_shape": list(arr_A.shape),
        "masks_B_shape": list(arr_B.shape),
        "portrait_floor_mean_v_analysis": {
            "total_frames": len(arr_mv),
            "frames_with_valid_ray_potential": below_919,
            "mean_portrait_v": round(float(arr_mv.mean()), 1),
            "min_portrait_v": round(float(arr_mv.min()), 1),
            "threshold_919_meaning": "portrait v < 919 → landscape col > 1000 → dx > 0 → d_world.z < 0 (valid)",
        },
    }

    out_path = os.path.join(EVIDENCE_DIR, "preflight_v4.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  preflight_v4.json saved: {out_path}")
    return result


# ---------------------------------------------------------------------------
# Step 1: Load cycle_3 masks (no re-inference)
# ---------------------------------------------------------------------------
def step1_load_masks() -> tuple[list, list, dict, dict]:
    print("\n=== Step 1: Load cycle_3 portrait masks (no re-inference) ===")
    arr_A = np.load(MASKS_A_PATH)["masks"]
    arr_B = np.load(MASKS_B_PATH)["masks"]
    masks_A = [arr_A[i] for i in range(arr_A.shape[0])]
    masks_B = [arr_B[i] for i in range(arr_B.shape[0])]

    def compute_stats(masks: list) -> dict:
        ratios = [float(m.mean()) for m in masks]
        return {
            "total_frames": len(masks),
            "floor_ratio_mean": round(float(np.mean(ratios)), 4),
            "floor_ratio_std": round(float(np.std(ratios)), 4),
            "frames_near_zero": int(sum(r < 0.05 for r in ratios)),
            "source": "cycle3_portrait_cached",
        }

    stats_A = compute_stats(masks_A)
    stats_B = compute_stats(masks_B)
    print(f"  Loaded A={len(masks_A)}, B={len(masks_B)} portrait masks")
    print(f"  floor_ratio: A={stats_A['floor_ratio_mean']:.4f}, B={stats_B['floor_ratio_mean']:.4f}")
    return masks_A, masks_B, stats_A, stats_B


# ---------------------------------------------------------------------------
# Step 2: Sample ray validation (Option B back-transform)
# ---------------------------------------------------------------------------
def step2_sample_raycast(masks_A: list, masks_B: list) -> dict:
    print("\n=== Step 2: Sample ray-cast validation (Option B back-transform) ===")

    poses_A = load_poses(MERGED_DB, map_id=0)
    poses_B = load_poses(MERGED_DB, map_id=1)

    def validate_frame(
        mask: np.ndarray, pose: np.ndarray,
        K: dict, z0: float, frame_idx: int, scan_label: str,
    ) -> dict:
        R = pose[:3, :3]
        t = pose[:3, 3]
        tz = t[2]
        fx, fy = K["fx"], K["fy"]
        cx, cy = K["cx"], K["cy"]

        # Stride-sampled portrait pixel coords
        vs_s, us_s = np.mgrid[0:mask.shape[0]:STRIDE, 0:mask.shape[1]:STRIDE]
        mask_s = mask[::STRIDE, ::STRIDE]
        vs_m = vs_s[mask_s]   # portrait rows (v')
        us_m = us_s[mask_s]   # portrait cols (u')
        n_floor = len(vs_m)

        if n_floor == 0:
            return {
                "scan": scan_label, "frame": int(frame_idx),
                "n_floor_px": 0, "n_valid": 0,
                "lambda_valid_ratio": 0.0, "near_trajectory_ratio": 0.0,
                "pass": False,
                "back_transform_land_u_mean": None,
                "dwz_neg_ratio": 0.0,
            }

        # Option B back-transform: portrait (u', v') → landscape (u, v)
        u_land, v_land = back_transform_portrait_to_landscape(us_m, vs_m)

        dx = (u_land - cx) / fx
        dy = (v_land - cy) / fy
        dz = np.ones(n_floor, dtype=np.float32)
        d_cam = np.stack([dx, dy, dz], axis=1)
        d_world = d_cam @ R.T
        dwz = d_world[:, 2]

        dwz_neg_ratio = float((dwz < -1e-3).mean())
        valid_dir = dwz < -1e-3
        lam = np.where(valid_dir, (z0 - tz) / dwz, -1.0)
        keep = valid_dir & (lam > 0) & (lam <= LAMBDA_MAX)
        n_valid = int(keep.sum())
        lam_ratio = n_valid / n_floor

        near_ratio = 0.0
        if n_valid > 0:
            lam_k = lam[keep]
            dw_k = d_world[keep]
            X = t[0] + lam_k * dw_k[:, 0]
            Y = t[1] + lam_k * dw_k[:, 1]
            near = ((X >= TRAJ_XMIN) & (X <= TRAJ_XMAX) &
                    (Y >= TRAJ_YMIN) & (Y <= TRAJ_YMAX))
            near_ratio = float(near.sum()) / n_valid

        frame_pass = (n_valid > 0) and (near_ratio >= 0.30)
        return {
            "scan": scan_label, "frame": int(frame_idx),
            "n_floor_px": n_floor, "n_valid": n_valid,
            "lambda_valid_ratio": round(float(lam_ratio), 4),
            "near_trajectory_ratio": round(float(near_ratio), 4),
            "pass": bool(frame_pass),
            "back_transform_land_u_mean": round(float(u_land.mean()), 1),
            "dwz_neg_ratio": round(dwz_neg_ratio, 4),
        }

    frame_results = []

    n_A = min(len(masks_A), len(poses_A))
    for idx in CYCLE2_SAMPLE_FRAMES_A:
        if idx >= n_A:
            continue
        r = validate_frame(masks_A[idx], poses_A[idx], K_A_LANDSCAPE, Z0_A, idx, "A")
        frame_results.append(r)
        print(f"  A[{idx:3d}] floor_px={r['n_floor_px']:6d} land_u={r['back_transform_land_u_mean'] or 0:.0f} "
              f"dwz<0={r['dwz_neg_ratio']:.1%} λ_ratio={r['lambda_valid_ratio']:.4f} "
              f"near={r['near_trajectory_ratio']:.2f} {'PASS' if r['pass'] else 'FAIL'}")

    n_B = min(len(masks_B), len(poses_B))
    for idx in CYCLE2_SAMPLE_FRAMES_B:
        if idx >= n_B:
            continue
        r = validate_frame(masks_B[idx], poses_B[idx], K_B_LANDSCAPE, Z0_B, idx, "B")
        frame_results.append(r)
        print(f"  B[{idx:3d}] floor_px={r['n_floor_px']:6d} land_u={r['back_transform_land_u_mean'] or 0:.0f} "
              f"dwz<0={r['dwz_neg_ratio']:.1%} λ_ratio={r['lambda_valid_ratio']:.4f} "
              f"near={r['near_trajectory_ratio']:.2f} {'PASS' if r['pass'] else 'FAIL'}")

    n_pass = sum(1 for r in frame_results if r["pass"])
    n_total = len(frame_results)
    pass_rate = n_pass / n_total if n_total > 0 else 0.0
    ac3_pass = pass_rate >= 0.50
    mean_dwz_neg = float(np.mean([r["dwz_neg_ratio"] for r in frame_results]))

    print(f"\n  Sample PASS rate: {n_pass}/{n_total} = {pass_rate:.1%}")
    print(f"  cycle_2 baseline: 5/20 = 25%, cycle_3: 0/20 = 0%")
    print(f"  Mean dwz<0 ratio: {mean_dwz_neg:.1%}")
    print(f"  AC3: {'PASS' if ac3_pass else 'FAIL'} (target ≥50%)")

    if not ac3_pass:
        print("\n  [DIAGNOSIS] FAIL root cause:")
        print(f"  Portrait floor mean v' ~1682 (BOTTOM) → back-transform landscape col = 1919-1682 = 237 (LEFT)")
        print(f"  For frames with R[2,0] ~ -0.996: d_world.z = R[2,0] * dx + ... > 0 because dx < 0")
        print(f"  Valid rays require portrait floor at v' < 919 (→ landscape col > 1000, dx > 0)")
        print(f"  Only ~2/314 scan A frames have portrait floor mean_v < 919")

    result = {
        "frame_results": frame_results,
        "n_pass": n_pass,
        "n_total": n_total,
        "pass_rate": round(pass_rate, 4),
        "cycle2_baseline_pass_rate": 0.25,
        "cycle3_baseline_pass_rate": 0.00,
        "mean_dwz_neg_ratio": round(mean_dwz_neg, 4),
        "ac3_pass": bool(ac3_pass),
        "lambda_max_used": LAMBDA_MAX,
        "back_transform_formula": "landscape_col = H'-1-v', landscape_row = u' (H'=1920)",
    }

    out_path = os.path.join(EVIDENCE_DIR, "sample_check_v4.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  sample_check_v4.json saved: {out_path}")
    return result


# ---------------------------------------------------------------------------
# Step 3: Full ray-cast (Option B)
# ---------------------------------------------------------------------------
def step3_raycast(masks_A: list, masks_B: list) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    print("\n=== Step 3: Full ray-cast (Option B back-transform, λ_max=5m, stride=4) ===")

    poses_A = load_poses(MERGED_DB, map_id=0)
    poses_B = load_poses(MERGED_DB, map_id=1)

    def raycast_scan(
        masks: list, poses: list, K: dict, z0: float, label: str
    ) -> tuple[np.ndarray, dict]:
        t0 = time.time()
        all_pts = []
        total_mask_px = 0
        total_valid = 0
        total_drop_dir = 0
        total_drop_lam = 0
        lam_vals_sample = []

        fx, fy = K["fx"], K["fy"]
        cx, cy = K["cx"], K["cy"]

        for i, (mask, pose) in enumerate(zip(masks, poses)):
            R = pose[:3, :3]
            t_vec = pose[:3, 3]
            tz = t_vec[2]

            # Stride-sampled portrait pixel coords
            vs_s, us_s = np.mgrid[0:mask.shape[0]:STRIDE, 0:mask.shape[1]:STRIDE]
            mask_s = mask[::STRIDE, ::STRIDE]
            vs_m = vs_s[mask_s]   # portrait rows (v')
            us_m = us_s[mask_s]   # portrait cols (u')
            n_floor = len(vs_m)
            total_mask_px += n_floor

            if n_floor == 0:
                all_pts.append(np.empty((0, 2), dtype=np.float32))
                continue

            # Option B back-transform: portrait (u', v') → landscape (u, v)
            u_land, v_land = back_transform_portrait_to_landscape(us_m, vs_m)

            dx = (u_land - cx) / fx
            dy = (v_land - cy) / fy
            dz_cam = np.ones(n_floor, dtype=np.float32)
            d_cam = np.stack([dx, dy, dz_cam], axis=1)
            d_world = d_cam @ R.T

            dwz = d_world[:, 2]
            valid_dir = dwz < -1e-3
            total_drop_dir += int((~valid_dir).sum())

            lam = np.where(valid_dir, (z0 - tz) / dwz, -1.0)
            keep = valid_dir & (lam > 0) & (lam <= LAMBDA_MAX)
            total_drop_lam += int((valid_dir & ~keep).sum())

            lam_k = lam[keep]
            d_world_k = d_world[keep]
            X = t_vec[0] + lam_k * d_world_k[:, 0]
            Y = t_vec[1] + lam_k * d_world_k[:, 1]
            pts = np.stack([X, Y], axis=1).astype(np.float32)
            all_pts.append(pts)
            total_valid += len(pts)

            if i < 100 and len(lam_k) > 0:
                lam_vals_sample.extend(lam_k[:10].tolist())

            if i % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [{label}] frame {i}/{len(masks)}, pts={total_valid}, {elapsed:.1f}s")

        world_points = (
            np.concatenate(all_pts, axis=0) if any(len(p) > 0 for p in all_pts)
            else np.empty((0, 2), dtype=np.float32)
        )
        elapsed = time.time() - t0
        lam_arr = np.array(lam_vals_sample, dtype=np.float32) if lam_vals_sample else np.array([])

        stats: dict = {
            "total_frames": len(masks),
            "total_mask_px": total_mask_px,
            "total_world_points": len(world_points),
            "drop_dir_count": total_drop_dir,
            "drop_lam_count": total_drop_lam,
            "valid_count": total_valid,
            "drop_dir_ratio": round(float(total_drop_dir / max(1, total_mask_px)), 4),
            "lambda_valid_ratio": round(float(total_valid / max(1, total_mask_px)), 4),
            "elapsed_s": round(elapsed, 2),
            "z0": round(z0, 4),
            "lambda_max": LAMBDA_MAX,
            "stride": STRIDE,
        }
        if len(lam_arr) > 0:
            stats["lambda_sample_mean"] = round(float(lam_arr.mean()), 3)
        if len(world_points) > 0:
            stats["x_range"] = [round(float(world_points[:, 0].min()), 3),
                                 round(float(world_points[:, 0].max()), 3)]
            stats["y_range"] = [round(float(world_points[:, 1].min()), 3),
                                 round(float(world_points[:, 1].max()), 3)]
        print(f"  [{label}] done: {len(world_points)} pts in {elapsed:.1f}s, "
              f"drop_dir={total_drop_dir}/{total_mask_px}={total_drop_dir/max(1,total_mask_px):.1%}")
        return world_points, stats

    n_A = min(len(masks_A), len(poses_A))
    n_B = min(len(masks_B), len(poses_B))

    pts_A, stats_A = raycast_scan(masks_A[:n_A], poses_A[:n_A], K_A_LANDSCAPE, Z0_A, "A")
    pts_B, stats_B = raycast_scan(masks_B[:n_B], poses_B[:n_B], K_B_LANDSCAPE, Z0_B, "B")

    # Save world points
    np.savez_compressed(
        os.path.join(EVIDENCE_DIR, "world_points_v4.npz"),
        X_A=pts_A[:, 0] if len(pts_A) > 0 else np.array([]),
        Y_A=pts_A[:, 1] if len(pts_A) > 0 else np.array([]),
        X_B=pts_B[:, 0] if len(pts_B) > 0 else np.array([]),
        Y_B=pts_B[:, 1] if len(pts_B) > 0 else np.array([]),
    )
    print(f"  world_points_v4.npz: A={len(pts_A)}, B={len(pts_B)}")
    return pts_A, pts_B, stats_A, stats_B


# ---------------------------------------------------------------------------
# Step 4: Density + polygon
# ---------------------------------------------------------------------------
def step4_density_polygon(pts_A: np.ndarray, pts_B: np.ndarray) -> tuple[dict | None, dict, tuple, np.ndarray | None]:
    print("\n=== Step 4: Density grid + polygon ===")
    sys.path.insert(0, os.path.dirname(__file__))
    from build_heatmap_polygon import (
        build_density_grid, extract_polygon,
        save_density_heatmap, save_per_scan_heatmap, save_polygon_overlay,
    )
    import json as _json

    polygon_dir = os.path.join(EVIDENCE_DIR, "polygon")
    os.makedirs(polygon_dir, exist_ok=True)

    total_pts = len(pts_A) + len(pts_B)
    if total_pts == 0:
        print("  [WARN] 0 total world points — cannot build density or polygon.")
        print("  This confirms Option B fails with cycle_3 portrait masks.")
        # Save empty world points evidence
        empty_density_path = os.path.join(EVIDENCE_DIR, "density_v4.png")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "0 world points\n(Option B back-transform failed)\nPortrait floor at bottom → dx<0 → d_world.z>0",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.set_title("cycle_4 density (BLOCKED: 0 points)")
        fig.savefig(empty_density_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        # Also save per_scan
        import shutil
        shutil.copy(empty_density_path, os.path.join(EVIDENCE_DIR, "density_per_scan_v4.png"))
        print(f"  Empty density placeholder saved: {empty_density_path}")
        return None, {"polygon_area_m2": 0, "east_leg_point_count": 0}, (0.0, 0.0, 0.2), None

    # Combine available points
    parts = [p for p in [pts_A, pts_B] if len(p) > 0]
    combined = np.concatenate(parts, axis=0)

    grid_c, origin_c = build_density_grid(combined, BIN_SIZE)
    save_density_heatmap(grid_c, origin_c, os.path.join(EVIDENCE_DIR, "density_v4.png"), "Floor density (cycle4)")

    if len(pts_A) > 0 and len(pts_B) > 0:
        grid_A, origin_A = build_density_grid(pts_A, BIN_SIZE)
        grid_B, origin_B = build_density_grid(pts_B, BIN_SIZE)
        save_per_scan_heatmap(grid_A, origin_A, grid_B, origin_B, os.path.join(EVIDENCE_DIR, "density_per_scan_v4.png"))
    elif len(pts_A) > 0:
        grid_A, origin_A = build_density_grid(pts_A, BIN_SIZE)
        save_density_heatmap(grid_A, origin_A, os.path.join(EVIDENCE_DIR, "density_per_scan_v4.png"), "Scan A only")
    else:
        grid_B, origin_B = build_density_grid(pts_B, BIN_SIZE)
        save_density_heatmap(grid_B, origin_B, os.path.join(EVIDENCE_DIR, "density_per_scan_v4.png"), "Scan B only")

    np.savez_compressed(
        os.path.join(EVIDENCE_DIR, "density_v4.npz"),
        grid=grid_c, origin=np.array(origin_c),
    )

    geojson, poly_metrics = extract_polygon(grid_c, origin_c, min_count=MIN_COUNT, min_area_m2=5.0)

    if geojson is None:
        print("  [WARN] No polygon extracted")
        return None, poly_metrics, origin_c, grid_c

    if geojson.get("properties"):
        geojson["properties"]["source"] = "sprint83-cycle4-backxform-Zup"

    geojson_path = os.path.join(polygon_dir, "floor_polygon_seg_v4.geojson")
    with open(geojson_path, "w") as f:
        _json.dump(geojson, f, indent=2)

    save_polygon_overlay(
        grid_c, origin_c, geojson_path, SPRINT82_POLYGON,
        os.path.join(polygon_dir, "floor_polygon_seg_v4.png"),
    )
    print(f"  Polygon: area={poly_metrics.get('polygon_area_m2')}m²")

    # East leg (X ∈ [12, 32])
    east_mask = (combined[:, 0] >= 12.0) & (combined[:, 0] <= 32.0)
    east_count = int(east_mask.sum())
    poly_metrics["east_leg_point_count"] = east_count
    print(f"  East leg (X∈[12,32]) points: {east_count}")

    return geojson, poly_metrics, origin_c, grid_c


# ---------------------------------------------------------------------------
# Step 5: Compare v1/v2/v3/v4 + sprint82
# ---------------------------------------------------------------------------
def step5_compare(pts_A: np.ndarray, pts_B: np.ndarray, poly_metrics: dict) -> dict:
    print("\n=== Step 5: Compare (v1/v2/v3/v4 + sprint82) ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import json as _json

    # Load cycle_2 east leg count
    pts_v2 = np.load(os.path.join(EVIDENCE_DIR_C2, "world_points_v2.npz"))
    X_c2 = np.concatenate([pts_v2.get("X_A", np.array([])), pts_v2.get("X_B", np.array([]))])
    east_count_v2 = int(((X_c2 >= 12.0) & (X_c2 <= 32.0)).sum())

    east_count_v4 = poly_metrics.get("east_leg_point_count", 0)
    east_ratio = east_count_v4 / max(1, east_count_v2)
    ac4_pass = east_ratio >= 2.0

    print(f"  East leg: cycle_2={east_count_v2}, cycle_4={east_count_v4}, ratio={east_ratio:.2f}x")
    print(f"  AC4: {'PASS' if ac4_pass else 'FAIL'} (target ≥2x)")

    # 4-polygon overlay compare
    fig, ax = plt.subplots(figsize=(16, 10))

    # Background: v4 world points
    parts = [p for p in [pts_A, pts_B] if len(p) > 0]
    if parts:
        combined = np.concatenate(parts, axis=0)
        if len(combined) > 10000:
            idx = np.random.choice(len(combined), 10000, replace=False)
            combined = combined[idx]
        ax.scatter(combined[:, 0], combined[:, 1], s=0.5, c="gray", alpha=0.3, label="cycle4 pts")
    else:
        # Also show cycle_2 points as background for reference
        c2_pts = np.stack([pts_v2.get("X_A", np.array([])), pts_v2.get("Y_A", np.array([]))], axis=1)
        if len(c2_pts) > 10000:
            c2_pts = c2_pts[np.random.choice(len(c2_pts), 10000, replace=False)]
        ax.scatter(c2_pts[:, 0], c2_pts[:, 1], s=0.5, c="lightblue", alpha=0.2, label="cycle2 pts (ref)")

    def draw_poly(path: str, color: str, label: str, ls: str = "-") -> None:
        if not os.path.exists(path):
            return
        with open(path) as f:
            geo = _json.load(f)
        geom = geo.get("geometry") or geo
        if geo.get("type") == "Feature":
            geom = geo["geometry"]
        gtype = geom.get("type", "")
        if gtype == "Polygon":
            coords = geom["coordinates"][0]
            xs, ys = [c[0] for c in coords], [c[1] for c in coords]
            ax.plot(xs, ys, color=color, lw=2, ls=ls, label=label)
            ax.fill(xs, ys, color=color, alpha=0.1)
        elif gtype == "MultiPolygon":
            for i2, poly in enumerate(geom["coordinates"]):
                coords = poly[0]
                xs, ys = [c[0] for c in coords], [c[1] for c in coords]
                ax.plot(xs, ys, color=color, lw=2, ls=ls, label=label if i2 == 0 else None)
                ax.fill(xs, ys, color=color, alpha=0.1)

    draw_poly(CYCLE1_POLYGON, "orange", "cycle1", "--")
    draw_poly(CYCLE2_POLYGON, "blue", "cycle2 (25% PASS)", "-.")
    draw_poly(CYCLE3_POLYGON, "red", "cycle3 (0% PASS)", ":")
    v4_geojson_path = os.path.join(EVIDENCE_DIR, "polygon/floor_polygon_seg_v4.geojson")
    if os.path.exists(v4_geojson_path):
        draw_poly(v4_geojson_path, "purple", f"cycle4 ({east_count_v4}pts)", "-")
    if os.path.exists(SPRINT82_POLYGON):
        draw_poly(SPRINT82_POLYGON, "lightgreen", "sprint82 (ref)", ":")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Floor polygon: cycle1~4 overlay (cycle4=Option B back-transform)")
    ax.legend(markerscale=5, fontsize=8)
    ax.grid(True, alpha=0.3)
    compare_path = os.path.join(EVIDENCE_DIR, "compare_v1_v2_v3_v4.png")
    fig.savefig(compare_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  compare_v1_v2_v3_v4.png saved: {compare_path}")

    # Compare with sprint82
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    if parts:
        ax2.scatter(combined[:, 0], combined[:, 1], s=0.5, c="gray", alpha=0.3, label="cycle4 pts")
    draw_poly(v4_geojson_path if os.path.exists(v4_geojson_path) else "", "purple", "cycle4", "-")
    draw_poly(SPRINT82_POLYGON, "lightgreen", "sprint82 (ref)", ":")
    ax2.set_title("cycle4 polygon vs sprint82 reference")
    ax2.legend(markerscale=5)
    ax2.grid(True, alpha=0.3)
    compare82_path = os.path.join(EVIDENCE_DIR, "compare_with_sprint82_v4.png")
    fig2.savefig(compare82_path, dpi=100, bbox_inches="tight")
    plt.close(fig2)
    print(f"  compare_with_sprint82_v4.png saved: {compare82_path}")

    result = {
        "east_leg_count_v2": east_count_v2,
        "east_leg_count_v4": east_count_v4,
        "east_leg_ratio": round(east_ratio, 4),
        "ac4_east_leg_pass": bool(ac4_pass),
        "cycle2_polygon_area": None,
        "cycle4_polygon_area": poly_metrics.get("polygon_area_m2"),
    }

    # Load cycle_2 polygon area for comparison
    if os.path.exists(CYCLE2_POLYGON):
        with open(CYCLE2_POLYGON) as f:
            c2geo = _json.load(f)
        c2props = c2geo.get("properties", {})
        result["cycle2_polygon_area"] = c2props.get("area_m2")

    out_json = os.path.join(EVIDENCE_DIR, "comparison_v4.json")
    with open(out_json, "w") as f:
        _json.dump(result, f, indent=2)
    return result


# ---------------------------------------------------------------------------
# Metrics assembly
# ---------------------------------------------------------------------------
def assemble_metrics(
    preflight: dict, sample_check: dict,
    stats_seg_A: dict, stats_seg_B: dict,
    stats_ray_A: dict, stats_ray_B: dict,
    poly_metrics: dict, comparison: dict,
) -> dict:
    ac3_pass = sample_check.get("ac3_pass", False)
    ac4_pass = comparison.get("ac4_east_leg_pass", False)
    area_m2 = poly_metrics.get("polygon_area_m2") or 0
    ac5_pass = bool(area_m2 > 0)

    # Load cycle_2 polygon area for AC4 comparison
    cycle2_area = comparison.get("cycle2_polygon_area")

    metrics = {
        "cycle": "cycle_4",
        "method": "portrait_mask_back_transform_option_B",
        "back_transform": {
            "formula": "landscape_col = H'-1-v', landscape_row = u'",
            "H_prime": H_PRIME,
            "landscape_K_A": K_A_LANDSCAPE,
            "landscape_K_B": K_B_LANDSCAPE,
        },
        "up_axis": "+Z",
        "z0_A": Z0_A,
        "z0_B": Z0_B,
        "lambda_max": LAMBDA_MAX,
        "stride": STRIDE,
        "bin_size_m": BIN_SIZE,
        "ac3": {
            "sample_n": sample_check.get("n_total"),
            "pass_count": sample_check.get("n_pass"),
            "pass_rate": sample_check.get("pass_rate", 0.0),
            "cycle2_baseline": 0.25,
            "cycle3_baseline": 0.00,
            "mean_dwz_neg_ratio": sample_check.get("mean_dwz_neg_ratio"),
            "pass": bool(ac3_pass),
        },
        "ac4": {
            "east_leg_count_v2": comparison.get("east_leg_count_v2"),
            "east_leg_count_v4": comparison.get("east_leg_count_v4"),
            "east_leg_ratio": comparison.get("east_leg_ratio"),
            "cycle2_polygon_area_m2": cycle2_area,
            "cycle4_polygon_area_m2": area_m2,
            "pass": bool(ac4_pass),
        },
        "ac5": {
            "area_m2": area_m2,
            "vertex_count": poly_metrics.get("polygon_vertex_count"),
            "east_leg_point_count": poly_metrics.get("east_leg_point_count"),
            "pass": bool(ac5_pass),
        },
        "world_points": {
            "scan_A": stats_ray_A.get("total_world_points"),
            "scan_B": stats_ray_B.get("total_world_points"),
            "x_range_A": stats_ray_A.get("x_range"),
            "y_range_A": stats_ray_A.get("y_range"),
            "x_range_B": stats_ray_B.get("x_range"),
            "y_range_B": stats_ray_B.get("y_range"),
            "drop_dir_ratio_A": stats_ray_A.get("drop_dir_ratio"),
            "drop_dir_ratio_B": stats_ray_B.get("drop_dir_ratio"),
        },
        "mask_stats": {
            "floor_ratio_mean_A": stats_seg_A.get("floor_ratio_mean"),
            "floor_ratio_mean_B": stats_seg_B.get("floor_ratio_mean"),
            "source": "cycle3_portrait_cached",
        },
        "blocker_analysis": {
            "portrait_floor_mean_v": preflight.get("portrait_floor_mean_v_analysis", {}).get("mean_portrait_v"),
            "frames_with_valid_ray_potential": preflight.get("portrait_floor_mean_v_analysis", {}).get("frames_with_valid_ray_potential"),
            "root_cause": (
                "portrait floor at v'~1682 (BOTTOM) back-transforms to landscape_col~237 (LEFT). "
                "R[2,0]~-0.996 and dx=(col-968)/fx<0 → d_world.z>0 → no valid rays. "
                "Valid rays require portrait floor at v'<919 (→ landscape_col>1000, dx>0). "
                "cycle_3 SegFormer detects floor at portrait BOTTOM even for sideways camera, "
                "missing the valid region portrait v'=[349,1582] (cycle_1 floor mapped to portrait). "
                "Fix: re-run SegFormer with upright (correct-orientation) images instead of CCW-rotated ones."
            ),
        },
        "comparison": comparison,
    }
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Sprint 83 cycle_4 pipeline (Option B back-transform) — project root: {PROJECT_ROOT}")
    t_start = time.time()

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    sys.path.insert(0, os.path.dirname(__file__))

    try:
        # Step 0: Preflight
        preflight = step0_preflight()

        # Step 1: Load cycle_3 portrait masks
        masks_A, masks_B, stats_seg_A, stats_seg_B = step1_load_masks()

        # Step 2: Sample ray-cast (Option B)
        sample_check = step2_sample_raycast(masks_A, masks_B)

        # Step 3: Full ray-cast
        pts_A, pts_B, stats_ray_A, stats_ray_B = step3_raycast(masks_A, masks_B)

        # Step 4: Density + polygon
        geojson, poly_metrics, origin, grid = step4_density_polygon(pts_A, pts_B)

        # Step 5: Compare
        comparison = step5_compare(pts_A, pts_B, poly_metrics)

        # Assemble metrics
        metrics = assemble_metrics(
            preflight, sample_check,
            stats_seg_A, stats_seg_B,
            stats_ray_A, stats_ray_B,
            poly_metrics, comparison,
        )

        metrics_path = os.path.join(EVIDENCE_DIR, "metrics_v4.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        elapsed = time.time() - t_start
        print(f"\n=== DONE in {elapsed:.1f}s ===")
        print(f"  metrics_v4.json → {metrics_path}")

        print("\n--- AC summary ---")
        for k in ["ac3", "ac4", "ac5"]:
            v = metrics[k]
            print(f"  {k}: {'PASS' if v.get('pass') else 'FAIL'} | {json.dumps({kk: vv for kk, vv in v.items() if kk in ('pass', 'pass_rate', 'east_leg_ratio', 'area_m2')})}")

        if not metrics["ac3"]["pass"]:
            print("\n--- BLOCKER ---")
            print(metrics["blocker_analysis"]["root_cause"])

    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
