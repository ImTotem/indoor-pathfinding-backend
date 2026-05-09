"""최근 4건 localize 요청에 대한 통합 디버그 분석.

1) 4층 SuperPoint coverage (keyframe 분포 + 3D coverage)
2) 각 query 사진의 top-k 매칭 keyframe (DINOv2 retrieval)
3) confidence 추적 — query 별 PnP inlier ratio + 거리 편차
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import sqlite3
import struct
import torch
from PIL import Image

DEBUG_DIR = Path("/app/var/storage/debug/localize/e30f31ea-5bbe-42df-9031-fa371bb7a7b3")
RTAB_DB = "/app/var/storage/scans/bc5e28b9-5075-4a58-b948-31ab1e277a02/rtabmap.db"
MAP_ID = "125b153d-3475-46f9-9690-e703951fb83d"

# 4건 그룹 (timestamp prefix → recorded conf/inliers/pose)
QUERIES = [
    ("20260508_125358_467012", 0.643, 9, ( 0.527, -21.743, 0.230)),
    ("20260508_131737_173495", 0.205, 9, ( 0.030, -18.300, 1.032)),
    ("20260508_131801_428982", 0.323, 10, ( 0.131,  -6.501, -0.044)),
    ("20260508_132936_089777", 0.611, 11, ( 0.680, -21.669, 0.194)),
]


def load_rtab_pose_calib():
    con = sqlite3.connect(RTAB_DB)
    calib_blob = con.execute(
        "SELECT calibration FROM Data WHERE calibration IS NOT NULL LIMIT 1"
    ).fetchone()[0]
    from be.slam_engines.superpoint.map_manager import _parse_calibration_K_and_local
    K, _ = _parse_calibration_K_and_local(bytes(calib_blob))
    width = struct.unpack("<i", bytes(calib_blob)[16:20])[0]
    height = struct.unpack("<i", bytes(calib_blob)[20:24])[0]
    poses = {}
    for nid, blob in con.execute("SELECT id, pose FROM Node WHERE pose IS NOT NULL"):
        if blob and len(blob) == 48:
            vals = struct.unpack("<12f", blob)
            T = np.eye(4)
            T[:3, :] = np.array(vals).reshape(3, 4)
            poses[int(nid)] = T
    con.close()
    return K, width, height, poses


def main():
    print("=" * 78)
    print("Phase 1 — 4층 SuperPoint coverage 통계")
    print("=" * 78)

    K, width, height, poses = load_rtab_pose_calib()
    pose_arr = np.array([p[:3, 3] for p in poses.values()])
    print(f"Calibration: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f} {width}x{height}")
    print(f"Keyframes (Node.pose 있음): {len(poses)}")
    print(f"  x range: [{pose_arr[:,0].min():.2f}, {pose_arr[:,0].max():.2f}] (extent {pose_arr[:,0].max()-pose_arr[:,0].min():.2f}m)")
    print(f"  y range: [{pose_arr[:,1].min():.2f}, {pose_arr[:,1].max():.2f}] (extent {pose_arr[:,1].max()-pose_arr[:,1].min():.2f}m)")
    print(f"  z range: [{pose_arr[:,2].min():.2f}, {pose_arr[:,2].max():.2f}] (extent {pose_arr[:,2].max()-pose_arr[:,2].min():.2f}m)")

    print("\nLoading SuperPointMapManager (cache hit if warm)…")
    from be.slam_engines.superpoint.map_manager import SuperPointMapManager
    mgr = SuperPointMapManager()
    loaded = mgr.get_or_load(MAP_ID, RTAB_DB)
    print(f"Cached frames: {len(loaded.node_ids)}")

    n_with_3d = 0
    nan_per_frame = []
    total_kp = total_3d = 0
    for nid, w3d in loaded.keyframe_world3d.items():
        n = len(w3d)
        valid = ~np.isnan(w3d).any(axis=1)
        nv = int(valid.sum())
        total_kp += n
        total_3d += nv
        nan_per_frame.append(1 - nv / n if n else 1)
        if nv > 0:
            n_with_3d += 1
    print(f"Frames with 3D coverage  : {n_with_3d}/{len(loaded.node_ids)} ({100*n_with_3d/len(loaded.node_ids):.1f}%)")
    print(f"Total keypoints          : {total_kp}")
    print(f"Total 3D-mapped          : {total_3d} ({100*total_3d/total_kp:.2f}%)")
    print(f"Per-frame NaN ratio      : mean={np.mean(nan_per_frame):.2%}  median={np.median(nan_per_frame):.2%}")
    if loaded.global_descs is not None:
        print(f"Global desc tensor       : {tuple(loaded.global_descs.shape)} dtype={loaded.global_descs.dtype}")

    print()
    print("=" * 78)
    print("Phase 2 — 4건 query 의 top-k matching keyframe (DINOv2 retrieval)")
    print("=" * 78)

    from be.slam_engines.superpoint.global_descriptor import GlobalDescExtractor
    device = mgr.device
    gext = GlobalDescExtractor(device)

    all_query_results = []

    for (ts, conf, inliers, recorded_pose) in QUERIES:
        pic_path = DEBUG_DIR / f"{ts}_00.jpg"
        if not pic_path.exists():
            print(f"\n[{ts}] image missing")
            continue
        img = Image.open(pic_path).convert("L")
        gray_uint8 = np.array(img)
        # global desc
        q_global = gext.extract(gray_uint8)            # torch.Tensor (384,)
        # cosine sim
        gd = loaded.global_descs                       # (K, 384)
        sims = (gd / gd.norm(dim=1, keepdim=True)) @ (q_global / q_global.norm())
        sims = sims.cpu().numpy()
        topk_idx = np.argsort(-sims)[:5]
        print(f"\n[{ts}]  conf={conf:.3f}  inliers={inliers}  pose={recorded_pose}")
        print(f"  image {gray_uint8.shape}  query global desc dim {q_global.shape}")
        print(f"  top-5 retrieval:")
        kf_locs = []
        for rank, ki in enumerate(topk_idx):
            nid = loaded.node_ids[int(ki)]
            sim = float(sims[ki])
            if nid in poses:
                p = poses[nid][:3, 3]
                kf_locs.append(p)
                print(f"    #{rank+1}  node={nid:5d}  sim={sim:.3f}  pose=({p[0]:6.2f}, {p[1]:7.2f}, {p[2]:5.2f})")
            else:
                print(f"    #{rank+1}  node={nid:5d}  sim={sim:.3f}  (no pose)")
        if kf_locs:
            kf_arr = np.array(kf_locs)
            d_to_recorded = np.linalg.norm(kf_arr - np.array(recorded_pose), axis=1)
            spread = float(np.linalg.norm(kf_arr.std(0)))
            print(f"  top-5 → recorded pose distances: {[f'{d:.2f}' for d in d_to_recorded]}")
            print(f"  top-5 spread (3D std-norm): {spread:.2f}m")
            all_query_results.append({
                "ts": ts, "conf": conf, "inliers": inliers,
                "recorded_pose": recorded_pose,
                "top1_distance_m": float(d_to_recorded[0]),
                "top5_spread_m": spread,
            })

    print()
    print("=" * 78)
    print("Phase 3 — confidence 분석")
    print("=" * 78)
    print("confidence = inliers / pts_3d_used (PnP RANSAC inlier ratio)")
    print()
    print(f"{'time':25}{'conf':>7}{'inliers':>9}{'pts3d_est':>11}{'top1_d_m':>11}{'top5_spread_m':>15}")
    for r in all_query_results:
        pts3d = round(r["inliers"] / r["conf"]) if r["conf"] > 0 else "?"
        print(f"{r['ts']:25}{r['conf']:>7.3f}{r['inliers']:>9}{pts3d:>11}{r['top1_distance_m']:>11.2f}{r['top5_spread_m']:>15.2f}")
    print()
    print("해석:")
    print("- inliers ≥ 12 (코드 기본 min_matches) 미만이면 marginal — 모두 9~11.")
    print("- top1 retrieval 이 recorded pose 와 가까우면 매칭 자체는 정확.")
    print("- top-5 spread 가 크면 retrieval 후보가 floor 전체에 흩어진 것 — 잘못 retrieve.")
    print("- conf < 0.3: pts_3d 후보는 많은데 RANSAC 일관성 낮음 → noise/wrong-keyframe match.")


if __name__ == "__main__":
    main()
