"""triangulation 개선 후 4건 query 재측위 + coverage 비교."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

DEBUG_DIR = Path("/app/var/storage/debug/localize/e30f31ea-5bbe-42df-9031-fa371bb7a7b3")
RTAB_DB = "/app/var/storage/scans/bc5e28b9-5075-4a58-b948-31ab1e277a02/rtabmap.db"
MAP_ID = "125b153d-3475-46f9-9690-e703951fb83d"

QUERIES = [
    ("20260508_125358_467012", 0.643, 9, ( 0.527, -21.743, 0.230)),
    ("20260508_131737_173495", 0.205, 9, ( 0.030, -18.300, 1.032)),
    ("20260508_131801_428982", 0.323, 10, ( 0.131,  -6.501, -0.044)),
    ("20260508_132936_089777", 0.611, 11, ( 0.680, -21.669, 0.194)),
]


def main():
    print("=" * 78)
    print("Step 1 — Cache invalidate + re-index + re-triangulate")
    print("=" * 78)

    from be.slam_engines.superpoint.map_manager import SuperPointMapManager
    mgr = SuperPointMapManager()
    if MAP_ID in mgr._maps:
        del mgr._maps[MAP_ID]
        print(f"  cache evicted: {MAP_ID}")
    loaded = mgr.get_or_load(MAP_ID, RTAB_DB)
    print(f"\nFrames cached: {len(loaded.node_ids)}")

    n_with_3d = 0
    nan_per_frame = []
    total_kp = total_3d = 0
    for nid, w3d in loaded.keyframe_world3d.items():
        n = len(w3d)
        valid = ~np.isnan(w3d).any(axis=1)
        nv = int(valid.sum())
        total_kp += n; total_3d += nv
        nan_per_frame.append(1 - nv / n if n else 1)
        if nv > 0: n_with_3d += 1
    print(f"3D coverage frames    : {n_with_3d}/{len(loaded.node_ids)} ({100*n_with_3d/len(loaded.node_ids):.1f}%)")
    print(f"3D-mapped keypoints   : {total_3d}/{total_kp} ({100*total_3d/total_kp:.2f}%)")
    print(f"Per-frame NaN ratio   : mean={np.mean(nan_per_frame):.2%} median={np.median(nan_per_frame):.2%}")

    print()
    print("=" * 78)
    print("Step 2 — 4건 query 재측위 (engine.py 통해 PnP)")
    print("=" * 78)

    # intrinsics from rtab_db calibration
    import sqlite3, struct
    from be.slam_engines.superpoint.map_manager import _parse_calibration_K_and_local
    con = sqlite3.connect(RTAB_DB)
    blob = bytes(con.execute("SELECT calibration FROM Data WHERE calibration IS NOT NULL LIMIT 1").fetchone()[0])
    con.close()
    K_rt, _ = _parse_calibration_K_and_local(blob)
    intrinsics = {
        "fx": float(K_rt[0,0]), "fy": float(K_rt[1,1]),
        "cx": float(K_rt[0,2]), "cy": float(K_rt[1,2]),
    }

    from be.slam_engines.superpoint.engine import SuperPointEngine
    loc = SuperPointEngine()

    print(f"\n{'time':25}{'recorded':>20}{'new conf':>10}{'inliers':>9}{'pose dx,dy,dz':>22}")
    for (ts, old_conf, old_inliers, _) in QUERIES:
        imgs = []
        for k in range(5):
            p = DEBUG_DIR / f"{ts}_{k:02d}.jpg"
            if p.exists():
                imgs.append(p.read_bytes())
        if not imgs:
            print(f"  {ts}: images missing"); continue

        try:
            res = loc._localize_sync(MAP_ID, imgs, intrinsics, RTAB_DB)
        except Exception as e:
            print(f"  {ts}: FAIL {type(e).__name__}: {e}")
            continue

        new_conf = res.get('confidence', 0)
        new_in   = res.get('num_matches', 0)
        new_pose = res.get('pose', {})
        nx, ny, nz = new_pose.get('x', 0), new_pose.get('y', 0), new_pose.get('z', 0)
        recorded = f"c={old_conf:.2f} n={old_inliers}"
        print(f"{ts:25}{recorded:>20}{new_conf:>10.3f}{new_in:>9}"
              f"{f'({nx:.2f}, {ny:.2f}, {nz:.2f})':>22}")


if __name__ == "__main__":
    main()
