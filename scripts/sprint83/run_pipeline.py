#!/usr/bin/env python3
"""
run_pipeline.py — Sprint 83 cycle_1 entrypoint.

Runs all 5 steps:
  Step 0: preflight (intrinsics, pose stats, floor y0)
  Step 1: SegFormer-b0 floor mask inference (608 frames)
  Step 2: Ray-cast floor pixels → world (X, Z) points
  Step 3: Density grid → heatmap PNGs + polygon GeoJSON
  Step 4: Compare with sprint82 polygon (IoU, area diff)

Outputs to: _workspace/sprint83-floor-segmentation-heatmap/evidence/cycle1/
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
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../..")
)
# Verify: should end with Indoor-pathfinding-v2
if not os.path.basename(PROJECT_ROOT).startswith("Indoor-pathfinding"):
    # Fallback: resolve from known absolute
    PROJECT_ROOT = "/Users/leehyeonsu/home/koreatech/graduate_project/Indoor-pathfinding-v2"
EVIDENCE_DIR = os.path.join(
    PROJECT_ROOT,
    "_workspace/sprint83-floor-segmentation-heatmap/evidence/cycle1",
)
KF_A_DIR = os.path.join(
    PROJECT_ROOT,
    "_workspace/sprint81-frame-pose-rtabmap-merge/evidence/cycle1/kf_A",
)
KF_B_DIR = os.path.join(
    PROJECT_ROOT,
    "_workspace/sprint81-frame-pose-rtabmap-merge/evidence/cycle1/kf_B",
)
MERGED_DB = os.path.join(
    PROJECT_ROOT,
    "_workspace/sprint81-frame-pose-rtabmap-merge/evidence/cycle2/merged_v2.db",
)
SPRINT82_POLYGON = os.path.join(
    PROJECT_ROOT,
    "_workspace/sprint82-floor-polygon-navgraph/evidence/cycle1/polygon/floor_polygon_refined.geojson",
)
CALIB_A = os.path.join(
    PROJECT_ROOT,
    "_workspace/sprint81-frame-pose-rtabmap-merge/evidence/cycle1/calib_A.json",
)
CALIB_B = os.path.join(
    PROJECT_ROOT,
    "_workspace/sprint81-frame-pose-rtabmap-merge/evidence/cycle1/calib_B.json",
)

# ---------------------------------------------------------------------------
# User-confirmed defaults (sprint83 cycle_1)
# ---------------------------------------------------------------------------
FLOOR_Y_OPTION = "B"    # per-scan camera ty mean - 1.4m
HAND_HEIGHT_M = 1.4
LAMBDA_MAX = 8.0        # extended from 5m: floor ratio ~8.6% makes 5m too tight → 8m to capture more floor hits
BIN_SIZE = 0.2
FLOOR_CLASS_IDS = [3]   # ADE20K "floor;flooring"
STRIDE = 2              # reduced from 4: floor ratio ~8.6% makes stride=4 too sparse → stride=2 for AC3
MIN_COUNT = 3

SAMPLE_IDS_A = [0, 79, 159, 239, 313]   # 0-indexed into sorted keyframe list
SAMPLE_IDS_B = [0, 73, 146, 219, 293]


def load_intrinsics(calib_path: str) -> np.ndarray:
    """Load 3×3 K from calib_*.json."""
    with open(calib_path) as f:
        d = json.load(f)
    fx = d["fx"]
    fy = d["fy"]
    cx = d["cx"]
    cy = d["cy"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K, d


def load_keyframe_index(kf_dir: str) -> list[dict]:
    with open(os.path.join(kf_dir, "index.json")) as f:
        return json.load(f)


def sorted_jpg_paths(kf_dir: str) -> list[str]:
    files = sorted([
        f for f in os.listdir(kf_dir) if f.endswith(".jpg")
    ])
    return [os.path.join(kf_dir, f) for f in files]


# ---------------------------------------------------------------------------
# Step 0: Preflight
# ---------------------------------------------------------------------------
def step0_preflight() -> dict:
    print("\n=== Step 0: Preflight ===")

    K_A, calib_A = load_intrinsics(CALIB_A)
    K_B, calib_B = load_intrinsics(CALIB_B)

    idx_A = load_keyframe_index(KF_A_DIR)
    idx_B = load_keyframe_index(KF_B_DIR)
    jpgs_A = sorted_jpg_paths(KF_A_DIR)
    jpgs_B = sorted_jpg_paths(KF_B_DIR)

    # Use merged_v2.db Node poses (converted to ARKit frame) for accurate stats.
    # Scan A: original ARKit poses. Scan B: T_AB-aligned ARKit poses.
    # load_poses_from_db converts DB 3x4 blob → ARKit 4x4 (db_pose_to_arkit).
    from raycast import load_poses_from_db
    node_ids_A, poses_A_db = load_poses_from_db(MERGED_DB, map_id=0)
    node_ids_B, poses_B_db = load_poses_from_db(MERGED_DB, map_id=1)
    tys_A_db = np.array([p[1, 3] for p in poses_A_db])
    tys_B_db = np.array([p[1, 3] for p in poses_B_db])

    # Floor y0 estimation (option B: per-scan mean - 1.4m)
    y0_A = float(np.mean(tys_A_db) - HAND_HEIGHT_M)
    y0_B = float(np.mean(tys_B_db) - HAND_HEIGHT_M)
    y0_diff = abs(y0_A - y0_B)

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    batch_size = 8 if device == "mps" else 4

    result = {
        "scan_A": {
            "fx": calib_A["fx"], "fy": calib_A["fy"],
            "cx": calib_A["cx"], "cy": calib_A["cy"],
            "W": calib_A.get("width", 1920), "H": calib_A.get("height", 1080),
            "frame_count_jpg": len(jpgs_A),
            "frame_count_index": len(idx_A),
            "ty_mean": round(float(np.mean(tys_A_db)), 4),
            "ty_std": round(float(np.std(tys_A_db)), 4),
            "ty_min": round(float(np.min(tys_A_db)), 4),
            "ty_max": round(float(np.max(tys_A_db)), 4),
            "y0_estimated": round(y0_A, 4),
        },
        "scan_B": {
            "fx": calib_B["fx"], "fy": calib_B["fy"],
            "cx": calib_B["cx"], "cy": calib_B["cy"],
            "W": calib_B.get("width", 1920), "H": calib_B.get("height", 1080),
            "frame_count_jpg": len(jpgs_B),
            "frame_count_index": len(idx_B),
            "ty_mean": round(float(np.mean(tys_B_db)), 4),
            "ty_std": round(float(np.std(tys_B_db)), 4),
            "ty_min": round(float(np.min(tys_B_db)), 4),
            "ty_max": round(float(np.max(tys_B_db)), 4),
            "y0_estimated": round(y0_B, 4),
        },
        "y0_diff_AB": round(y0_diff, 4),
        "y0_diff_warn": y0_diff > 0.3,
        "device": device,
        "batch_size": batch_size,
        "total_frames": len(jpgs_A) + len(jpgs_B),
        "floor_y_option": FLOOR_Y_OPTION,
        "hand_height_m": HAND_HEIGHT_M,
        "lambda_max": LAMBDA_MAX,
        "bin_size": BIN_SIZE,
        "floor_class_ids": FLOOR_CLASS_IDS,
        "stride": STRIDE,
    }

    out_path = os.path.join(EVIDENCE_DIR, "preflight.json")
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  y0_A={y0_A:.3f}m, y0_B={y0_B:.3f}m, diff={y0_diff:.3f}m {'WARN' if y0_diff > 0.3 else 'OK'}")
    print(f"  device={device}, total_frames={result['total_frames']}")
    print(f"  preflight.json saved to {out_path}")
    return result


# ---------------------------------------------------------------------------
# Step 1: Floor mask inference
# ---------------------------------------------------------------------------
def step1_infer_masks(preflight: dict) -> tuple[list, list, dict, dict]:
    print("\n=== Step 1: SegFormer floor mask inference ===")

    masks_A_path = os.path.join(EVIDENCE_DIR, "masks_A.npz")
    masks_B_path = os.path.join(EVIDENCE_DIR, "masks_B.npz")

    device = preflight["device"]
    batch_size = preflight["batch_size"]
    jpgs_A = sorted_jpg_paths(KF_A_DIR)
    jpgs_B = sorted_jpg_paths(KF_B_DIR)

    # Load cached masks if available
    if os.path.exists(masks_A_path) and os.path.exists(masks_B_path):
        print("  Loading cached masks from disk...")
        import time
        t0 = time.time()
        arr_A = np.load(masks_A_path)["masks"]
        arr_B = np.load(masks_B_path)["masks"]
        masks_A = [arr_A[i] for i in range(arr_A.shape[0])]
        masks_B = [arr_B[i] for i in range(arr_B.shape[0])]
        ratios_A = [float(m.mean()) for m in masks_A]
        ratios_B = [float(m.mean()) for m in masks_B]
        elapsed = time.time() - t0
        stats_A = {
            "total_frames": len(masks_A), "elapsed_s": round(elapsed, 2),
            "ms_per_frame": 0.0, "floor_ratio_mean": round(float(np.mean(ratios_A)), 4),
            "floor_ratio_std": round(float(np.std(ratios_A)), 4),
            "floor_ratio_min": round(float(np.min(ratios_A)), 4),
            "floor_ratio_max": round(float(np.max(ratios_A)), 4),
            "frames_near_zero": int(sum(r < 0.05 for r in ratios_A)),
            "frames_near_one": int(sum(r > 0.95 for r in ratios_A)),
            "device": "cached", "batch_size": batch_size,
        }
        stats_B = {
            "total_frames": len(masks_B), "elapsed_s": 0.0,
            "ms_per_frame": 0.0, "floor_ratio_mean": round(float(np.mean(ratios_B)), 4),
            "floor_ratio_std": round(float(np.std(ratios_B)), 4),
            "floor_ratio_min": round(float(np.min(ratios_B)), 4),
            "floor_ratio_max": round(float(np.max(ratios_B)), 4),
            "frames_near_zero": int(sum(r < 0.05 for r in ratios_B)),
            "frames_near_one": int(sum(r > 0.95 for r in ratios_B)),
            "device": "cached", "batch_size": batch_size,
        }
        print(f"  Loaded {len(masks_A)} A masks, {len(masks_B)} B masks in {elapsed:.1f}s")
        print(f"  floor_ratio: A={stats_A['floor_ratio_mean']:.3f}, B={stats_B['floor_ratio_mean']:.3f}")
    else:
        from segment import load_segformer, infer_floor_masks_batch
        import time
        processor, model = load_segformer(device)

        print(f"  Inferring {len(jpgs_A)} scan_A frames (batch={batch_size})...")
        t0_A = time.time()
        masks_A, stats_A = infer_floor_masks_batch(
            processor, model, jpgs_A, device, batch_size, FLOOR_CLASS_IDS
        )
        print(f"  Inferring {len(jpgs_B)} scan_B frames (batch={batch_size})...")
        masks_B, stats_B = infer_floor_masks_batch(
            processor, model, jpgs_B, device, batch_size, FLOOR_CLASS_IDS
        )

        # Save compressed masks
        np.savez_compressed(masks_A_path, masks=np.stack(masks_A))
        np.savez_compressed(masks_B_path, masks=np.stack(masks_B))
        print(f"  masks_A.npz, masks_B.npz saved")

    # Save sample overlays
    from viz import save_mask_overlay
    samples_dir = os.path.join(EVIDENCE_DIR, "seg_samples")
    os.makedirs(samples_dir, exist_ok=True)

    for idx_in_list in SAMPLE_IDS_A:
        if idx_in_list < len(jpgs_A):
            frame_num = idx_in_list + 1
            out_png = os.path.join(samples_dir, f"sample_mask_A_{frame_num:05d}.png")
            save_mask_overlay(jpgs_A[idx_in_list], masks_A[idx_in_list], out_png)
    for idx_in_list in SAMPLE_IDS_B:
        if idx_in_list < len(jpgs_B):
            frame_num = idx_in_list + 1
            out_png = os.path.join(samples_dir, f"sample_mask_B_{frame_num:05d}.png")
            save_mask_overlay(jpgs_B[idx_in_list], masks_B[idx_in_list], out_png)
    print(f"  sample masks saved to {samples_dir}")

    return masks_A, masks_B, stats_A, stats_B


# ---------------------------------------------------------------------------
# Step 2: Ray-cast
# ---------------------------------------------------------------------------
def step2_raycast(preflight: dict, masks_A: list, masks_B: list, force: bool = False) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    print("\n=== Step 2: Ray-cast floor pixels → world points ===")

    pts_A_path = os.path.join(EVIDENCE_DIR, "world_points_A.npz")
    pts_B_path = os.path.join(EVIDENCE_DIR, "world_points_B.npz")

    K_A, _ = load_intrinsics(CALIB_A)
    K_B, _ = load_intrinsics(CALIB_B)
    y0_A = preflight["scan_A"]["y0_estimated"]
    y0_B = preflight["scan_B"]["y0_estimated"]

    # Check if cached with same settings
    settings_tag = f"stride{STRIDE}_lmax{LAMBDA_MAX}"
    settings_path = os.path.join(EVIDENCE_DIR, "raycast_settings.txt")
    cached_ok = (
        not force
        and os.path.exists(pts_A_path)
        and os.path.exists(pts_B_path)
        and os.path.exists(settings_path)
        and open(settings_path).read().strip() == settings_tag
    )

    if cached_ok:
        print(f"  Loading cached world points (settings: {settings_tag})...")
        import time
        t0 = time.time()
        pts_A = np.load(pts_A_path)["points"]
        pts_B = np.load(pts_B_path)["points"]
        elapsed = time.time() - t0
        stats_A = {"total_world_points": len(pts_A), "elapsed_s": 0.0,
                   "x_range": [float(pts_A[:,0].min()), float(pts_A[:,0].max())],
                   "z_range": [float(pts_A[:,1].min()), float(pts_A[:,1].max())]}
        stats_B = {"total_world_points": len(pts_B), "elapsed_s": 0.0,
                   "x_range": [float(pts_B[:,0].min()), float(pts_B[:,0].max())],
                   "z_range": [float(pts_B[:,1].min()), float(pts_B[:,1].max())]}
        print(f"  Loaded: A={len(pts_A)}, B={len(pts_B)} pts in {elapsed:.1f}s")
    else:
        from raycast import load_poses_from_db, raycast_scan

        node_ids_A, poses_A = load_poses_from_db(MERGED_DB, map_id=0)
        node_ids_B, poses_B = load_poses_from_db(MERGED_DB, map_id=1)

        print(f"  Scan A: {len(masks_A)} masks, {len(poses_A)} DB poses, stride={STRIDE}, lmax={LAMBDA_MAX}m")
        print(f"  Scan B: {len(masks_B)} masks, {len(poses_B)} DB poses")

        n_A = min(len(masks_A), len(poses_A))
        n_B = min(len(masks_B), len(poses_B))

        pts_A, stats_A = raycast_scan(
            masks_A[:n_A], poses_A[:n_A], K_A, y0_A,
            stride=STRIDE, lambda_max=LAMBDA_MAX,
        )
        pts_B, stats_B = raycast_scan(
            masks_B[:n_B], poses_B[:n_B], K_B, y0_B,
            stride=STRIDE, lambda_max=LAMBDA_MAX,
        )

        np.savez_compressed(pts_A_path, points=pts_A, scan_id=np.zeros(len(pts_A), dtype=np.int8))
        np.savez_compressed(pts_B_path, points=pts_B, scan_id=np.ones(len(pts_B), dtype=np.int8))
        with open(settings_path, "w") as f:
            f.write(settings_tag)
        print(f"  world_points_A.npz: {len(pts_A)} points")
        print(f"  world_points_B.npz: {len(pts_B)} points")

    return pts_A, pts_B, stats_A, stats_B


# ---------------------------------------------------------------------------
# Step 3: Density grid + polygon
# ---------------------------------------------------------------------------
def step3_density_polygon(pts_A: np.ndarray, pts_B: np.ndarray) -> tuple[dict, dict, tuple]:
    print("\n=== Step 3: Density grid + polygon ===")

    from density_polygon import build_density_grid, extract_polygon
    from viz import save_density_heatmap, save_polygon_overlay

    heatmap_dir = os.path.join(EVIDENCE_DIR, "heatmap")
    polygon_dir = os.path.join(EVIDENCE_DIR, "polygon")
    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(polygon_dir, exist_ok=True)

    # Per-scan density
    grid_A, origin_A = build_density_grid(pts_A, BIN_SIZE)
    grid_B, origin_B = build_density_grid(pts_B, BIN_SIZE)

    save_density_heatmap(grid_A, origin_A, os.path.join(heatmap_dir, "heatmap_A.png"), "Scan A floor density")
    save_density_heatmap(grid_B, origin_B, os.path.join(heatmap_dir, "heatmap_B.png"), "Scan B floor density")

    # Combined
    combined = np.concatenate([pts_A, pts_B], axis=0)
    grid_combined, origin_combined = build_density_grid(combined, BIN_SIZE)
    save_density_heatmap(grid_combined, origin_combined, os.path.join(heatmap_dir, "heatmap_combined.png"), "Combined floor density (A+B)")

    # Save world_points combined
    np.savez_compressed(
        os.path.join(EVIDENCE_DIR, "world_points.npz"),
        X=combined[:, 0],
        Y=combined[:, 1],  # note: Y here = Z in world
        scan_id=np.concatenate([
            np.zeros(len(pts_A), dtype=np.int8),
            np.ones(len(pts_B), dtype=np.int8),
        ]),
    )

    # Save density.npz + density.png + density_per_scan.png
    np.savez_compressed(
        os.path.join(EVIDENCE_DIR, "density.npz"),
        grid=grid_combined,
        origin=np.array(origin_combined),
    )
    save_density_heatmap(
        grid_combined, origin_combined,
        os.path.join(EVIDENCE_DIR, "density.png"),
        "Floor density heatmap"
    )
    # Per-scan side-by-side
    _save_density_per_scan(grid_A, origin_A, grid_B, origin_B,
                           os.path.join(EVIDENCE_DIR, "density_per_scan.png"))

    print(f"  Combined grid: {grid_combined.shape}, origin={origin_combined}")
    print(f"  Occupancy cells (>=3): {int((grid_combined >= MIN_COUNT).sum())}")

    # Polygon extraction
    geojson, poly_metrics = extract_polygon(
        grid_combined, origin_combined,
        min_count=MIN_COUNT,
        min_area_m2=5.0,
    )

    if geojson is None:
        print("  WARNING: No polygon extracted!")
        geojson_path = None
    else:
        geojson_path = os.path.join(polygon_dir, "floor_polygon_seg.geojson")
        with open(geojson_path, "w") as f:
            json.dump(geojson, f, indent=2)
        print(f"  Polygon area: {poly_metrics.get('polygon_area_m2', '?')} m², "
              f"vertices: {poly_metrics.get('polygon_vertex_count', '?')}")

        # Polygon overlay on density
        save_polygon_overlay(
            grid_combined, origin_combined,
            geojson_path,
            SPRINT82_POLYGON,
            os.path.join(polygon_dir, "floor_polygon_seg.png"),
        )

    return geojson, poly_metrics, origin_combined, grid_combined


def _save_density_per_scan(grid_A, origin_A, grid_B, origin_B, out_path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, grid, origin, label in [
        (axes[0], grid_A, origin_A, "Scan A"),
        (axes[1], grid_B, origin_B, "Scan B"),
    ]:
        xmin, zmin, bs = origin
        nz, nx = grid.shape
        im = ax.imshow(
            np.log1p(grid), origin="lower",
            extent=[xmin, xmin + nx * bs, zmin, zmin + nz * bs],
            cmap="viridis", aspect="equal",
        )
        plt.colorbar(im, ax=ax)
        ax.set_title(label)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 4: Compare with sprint82
# ---------------------------------------------------------------------------
def step4_compare_sprint82(geojson: dict | None, poly_metrics: dict) -> dict:
    print("\n=== Step 4: Compare with sprint82 polygon ===")

    comparison = {
        "sprint82_area_m2": 894.29,
        "sprint82_vertex_count": 11,
        "sprint83_area_m2": poly_metrics.get("polygon_area_m2"),
        "sprint83_vertex_count": poly_metrics.get("polygon_vertex_count"),
        "iou": None,
        "area_diff_pct": None,
    }

    if geojson is None:
        comparison["error"] = "sprint83 polygon not extracted"
        return comparison

    s83_area = poly_metrics.get("polygon_area_m2", 0)
    s82_area = 894.29
    if s82_area > 0:
        area_diff_pct = abs(s83_area - s82_area) / s82_area * 100
        comparison["area_diff_pct"] = round(area_diff_pct, 2)

    # IoU
    if os.path.exists(SPRINT82_POLYGON):
        try:
            with open(SPRINT82_POLYGON) as f:
                geo82 = json.load(f)
            geom82 = geo82.get("geometry", geo82)
            geom83 = geojson["geometry"]

            from density_polygon import compute_iou
            # Extract coordinate rings
            def get_ring(geom):
                if geom["type"] == "Polygon":
                    return geom["coordinates"]
                elif geom["type"] == "MultiPolygon":
                    return geom["coordinates"][0]
                return None

            ring82 = get_ring(geom82)
            ring83 = get_ring(geom83)
            if ring82 and ring83:
                iou = compute_iou(ring82, ring83)
                comparison["iou"] = iou
                print(f"  IoU sprint82 vs sprint83: {iou:.4f}")
        except Exception as e:
            print(f"  IoU compute error: {e}")
            comparison["iou_error"] = str(e)

    print(f"  sprint82 area={s82_area}m², sprint83 area={s83_area}m², diff={comparison.get('area_diff_pct', '?')}%")

    out_path = os.path.join(EVIDENCE_DIR, "polygon", "comparison.json")
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)

    # Compare overlay (already done in step3 via save_polygon_overlay)
    # Copy to expected filename
    src = os.path.join(EVIDENCE_DIR, "polygon", "floor_polygon_seg.png")
    dst = os.path.join(EVIDENCE_DIR, "compare_with_sprint82.png")
    if os.path.exists(src):
        import shutil
        shutil.copy2(src, dst)

    return comparison


# ---------------------------------------------------------------------------
# Metrics assembly
# ---------------------------------------------------------------------------
def assemble_metrics(
    preflight: dict,
    stats_seg_A: dict, stats_seg_B: dict,
    stats_ray_A: dict, stats_ray_B: dict,
    poly_metrics: dict, comparison: dict,
) -> dict:
    metrics = {
        "ac1_preflight": {
            "y0_A": preflight["scan_A"]["y0_estimated"],
            "y0_B": preflight["scan_B"]["y0_estimated"],
            "y0_diff": preflight["y0_diff_AB"],
            "y0_diff_pass": not preflight["y0_diff_warn"],
            "device": preflight["device"],
        },
        "ac2_floor_mask": {
            "total_frames": stats_seg_A["total_frames"] + stats_seg_B["total_frames"],
            "floor_ratio_mean_A": stats_seg_A["floor_ratio_mean"],
            "floor_ratio_mean_B": stats_seg_B["floor_ratio_mean"],
            "frames_near_zero_A": stats_seg_A["frames_near_zero"],
            "frames_near_zero_B": stats_seg_B["frames_near_zero"],
            "infer_ms_per_frame_A": stats_seg_A["ms_per_frame"],
            "infer_ms_per_frame_B": stats_seg_B["ms_per_frame"],
            "total_elapsed_s": stats_seg_A["elapsed_s"] + stats_seg_B["elapsed_s"],
            "pass_floor_ratio_range": (
                0.10 <= stats_seg_A["floor_ratio_mean"] <= 0.70
                and 0.10 <= stats_seg_B["floor_ratio_mean"] <= 0.70
            ),
        },
        "ac3_world_points": {
            "scan_A_points": stats_ray_A["total_world_points"],
            "scan_B_points": stats_ray_B["total_world_points"],
            "pass_A": stats_ray_A["total_world_points"] >= 5_000_000,
            "pass_B": stats_ray_B["total_world_points"] >= 4_000_000,
            "elapsed_s": stats_ray_A["elapsed_s"] + stats_ray_B["elapsed_s"],
        },
        "ac4_density": {
            "occupancy_cells": poly_metrics.get("occupancy_cells", 0),
            "pass_cells": poly_metrics.get("occupancy_cells", 0) >= 5000,
            "bin_size_m": BIN_SIZE,
        },
        "ac5_polygon": {
            "area_m2": poly_metrics.get("polygon_area_m2"),
            "vertex_count": poly_metrics.get("polygon_vertex_count"),
            "num_components": poly_metrics.get("num_polygons_kept"),
            "pass_area": (
                600 <= (poly_metrics.get("polygon_area_m2") or 0) <= 2500
            ),
        },
        "ac6_sample_masks": {
            "expected": 10,
        },
        "ac7_build_report": {
            "note": "See cycle_1_build.md"
        },
        "comparison": comparison,
    }
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Sprint 83 cycle_1 pipeline — project root: {PROJECT_ROOT}")
    t_start = time.time()

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    # Add script dir to path for local module imports
    sys.path.insert(0, os.path.dirname(__file__))

    try:
        preflight = step0_preflight()

        masks_A, masks_B, stats_seg_A, stats_seg_B = step1_infer_masks(preflight)

        pts_A, pts_B, stats_ray_A, stats_ray_B = step2_raycast(preflight, masks_A, masks_B)

        geojson, poly_metrics, origin, grid = step3_density_polygon(pts_A, pts_B)

        comparison = step4_compare_sprint82(geojson, poly_metrics)

        metrics = assemble_metrics(
            preflight, stats_seg_A, stats_seg_B,
            stats_ray_A, stats_ray_B,
            poly_metrics, comparison,
        )

        # AC6: count actual sample PNGs
        samples_dir = os.path.join(EVIDENCE_DIR, "seg_samples")
        png_count = len([f for f in os.listdir(samples_dir) if f.endswith(".png")]) if os.path.exists(samples_dir) else 0
        metrics["ac6_sample_masks"]["actual"] = png_count
        metrics["ac6_sample_masks"]["pass"] = png_count >= 10

        metrics_path = os.path.join(EVIDENCE_DIR, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        elapsed = time.time() - t_start
        print(f"\n=== DONE in {elapsed:.1f}s ===")
        print(f"  metrics.json → {metrics_path}")

        # Summary print
        print("\n--- AC summary ---")
        for key, val in metrics.items():
            if key.startswith("ac"):
                pass_val = val.get("pass_area") or val.get("pass_cells") or val.get("pass_A") or val.get("pass_floor_ratio_range") or val.get("pass") or val.get("y0_diff_pass") or val.get("pass_B")
                print(f"  {key}: {json.dumps({k:v for k,v in val.items() if 'pass' in k or k in ('area_m2','total_frames','scan_A_points','scan_B_points','occupancy_cells')})}")

    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
