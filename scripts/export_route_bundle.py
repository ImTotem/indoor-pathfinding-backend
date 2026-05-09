"""경로 위 keyframe 의 SuperPoint feature 패키지 export.

Usage (in server container):
  python /app/scripts/export_route_bundle.py \
    --scan-id e3752897-3cd0-497b-a42f-542994a93886 \
    --route-nodes 6e79c9e0,...,36993cb8 \
    --localize-pose '{"x":-0.101,"y":-0.218,"z":-0.021,"qx":...,"qy":...,"qz":...,"qw":...}' \
    --radius-m 3.0 \
    --out /tmp/route_bundle.npz
"""
import argparse
import json
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/be')

import psycopg
import torch  # noqa: F401  (SuperPointMapManager 가 torch 사용)
from be.slam_engines.superpoint.map_manager import (
    SuperPointMapManager,
    _parse_calibration_K_and_local,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-id", required=True)
    ap.add_argument("--floor-id", required=True)
    ap.add_argument("--route-nodes", required=True, help="comma-separated UUIDs")
    ap.add_argument("--start-pose-json", required=True, help='{"x":..,"y":..,"z":..,"qx":..,...}')
    ap.add_argument("--start-floor-level", type=int, required=True)
    ap.add_argument("--destination-label", default="")
    ap.add_argument("--radius-m", type=float, default=3.0)
    ap.add_argument("--max-per-node", type=int, default=3,
                    help="route node 별 가장 가까운 N keyframe 만 선택")
    ap.add_argument("--quantize", choices=["fp32", "fp16"], default="fp16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    route_node_ids = [s.strip() for s in args.route_nodes.split(",") if s.strip()]
    start_pose = json.loads(args.start_pose_json)

    # 1. PG 에서 route node 좌표 + label
    pg_conn = psycopg.connect(
        "host=db port=5432 user=indoor password=indoor dbname=indoor"
    )
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT node_id::text, node_type, label, "
            "ST_X(geom::geometry), ST_Y(geom::geometry), ST_Z(geom::geometry) "
            "FROM map_node WHERE scan_id = %s AND node_id::text = ANY(%s) ",
            (args.scan_id, route_node_ids),
        )
        rows = cur.fetchall()
        node_info = {r[0]: {"node_id": r[0], "node_type": r[1], "label": r[2],
                            "x": r[3], "y": r[4], "z": r[5]} for r in rows}
        # 순서 유지
        route_nodes = [node_info[nid] for nid in route_node_ids if nid in node_info]

        # edges (route 위 인접 pair)
        cur.execute(
            "SELECT from_node_id::text, to_node_id::text, length_m "
            "FROM map_edge WHERE scan_id = %s",
            (args.scan_id,),
        )
        edges_all = cur.fetchall()
        edge_lookup: dict[tuple, float] = {}
        for f, t, l in edges_all:
            edge_lookup[(f, t)] = l
            edge_lookup[(t, f)] = l
        route_edges = []
        for i in range(len(route_nodes) - 1):
            a = route_nodes[i]["node_id"]
            b = route_nodes[i + 1]["node_id"]
            length = edge_lookup.get((a, b), float('nan'))
            route_edges.append({"from": a, "to": b, "length_m": length})

    pg_conn.close()

    # 2. RTAB-Map db 에서 Node.pose + Data.calibration
    rtab_db = Path(f"/app/var/storage/scans/{args.scan_id}/rtabmap.db")
    con = sqlite3.connect(str(rtab_db))

    # K + image_size + local_transform
    calib_blob = con.execute(
        "SELECT calibration FROM Data WHERE calibration IS NOT NULL LIMIT 1"
    ).fetchone()[0]
    K, local_transform = _parse_calibration_K_and_local(bytes(calib_blob))
    # blob layout: width @ offset 16, height @ 20 (int32)
    w = struct.unpack('<i', bytes(calib_blob)[16:20])[0]
    h = struct.unpack('<i', bytes(calib_blob)[20:24])[0]

    # Node poses
    node_poses: dict[int, np.ndarray] = {}
    for nid, blob in con.execute(
        "SELECT id, pose FROM Node WHERE pose IS NOT NULL"
    ):
        if blob and len(blob) == 48:
            vals = struct.unpack('<12f', blob)
            T = np.eye(4, dtype=np.float64)
            T[:3, :] = np.array(vals, dtype=np.float64).reshape(3, 4)
            node_poses[int(nid)] = T
    con.close()

    # 3. SuperPoint 인덱스 로드
    mgr = SuperPointMapManager()
    loaded = mgr.get_or_load(args.floor_id, str(rtab_db))
    print(f"[bundle] SuperPoint cache: {len(loaded.node_ids)} keyframes loaded")

    # 4. route node 별 가까운 keyframe 선택 — 각 route node 의 nearest N개
    selected_node_ids: set[int] = set()
    radius = args.radius_m
    for rn in route_nodes:
        rx, ry, rz = rn["x"], rn["y"], rn["z"]
        candidates: list[tuple[float, int]] = []
        for nid in loaded.node_ids:
            if nid not in node_poses:
                continue
            tx, ty, tz = node_poses[nid][:3, 3]
            d = float(np.sqrt((tx - rx) ** 2 + (ty - ry) ** 2 + (tz - rz) ** 2))
            if d <= radius:
                candidates.append((d, nid))
        candidates.sort(key=lambda x: x[0])
        for _, nid in candidates[: args.max_per_node]:
            selected_node_ids.add(nid)
    selected = sorted(selected_node_ids)
    print(f"[bundle] selected {len(selected)} keyframes "
          f"(radius={radius}m, max_per_node={args.max_per_node})")

    # 5. 각 keyframe 데이터 수집
    bundle = {
        "manifest": {
            "version": "1.0",
            "scan_id": args.scan_id,
            "floor_id": args.floor_id,
            "floor_level": args.start_floor_level,
            "destination": args.destination_label,
            "start_pose": start_pose,
            "intrinsics": {
                "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                "width": int(w), "height": int(h),
            },
            "local_transform_3x4": local_transform.tolist(),
            "route": {
                "nodes": route_nodes,
                "edges": route_edges,
                "total_nodes": len(route_nodes),
            },
            "keyframe_radius_m": radius,
            "keyframe_count": len(selected),
            "keyframe_node_ids": [int(n) for n in selected],
        },
    }

    # per-keyframe arrays
    kp_list, desc_list, world3d_list = [], [], []
    pose_list, gdesc_list, kf_meta_list = [], [], []

    desc_dtype = np.float16 if args.quantize == "fp16" else np.float32
    for nid in selected:
        feats = loaded.keyframe_feats[nid]
        kp = feats['keypoints'][0].cpu().numpy().astype(np.float32)        # (N, 2) — 좌표는 fp32 유지
        desc = feats['descriptors'][0].cpu().numpy().astype(desc_dtype)    # (N, 256) — quantize
        w3d = loaded.keyframe_world3d[nid].astype(np.float32)              # (N, 3)
        pose = node_poses[nid].astype(np.float32)                          # (4, 4)

        # global desc — order 가 loaded.node_ids 와 같음
        if loaded.global_descs is not None:
            idx_in_loaded = loaded.node_ids.index(nid)
            gdesc = loaded.global_descs[idx_in_loaded].cpu().numpy().astype(desc_dtype)
        else:
            gdesc = np.zeros(384, dtype=desc_dtype)

        kp_list.append(kp)
        desc_list.append(desc)
        world3d_list.append(w3d)
        pose_list.append(pose)
        gdesc_list.append(gdesc)
        kf_meta_list.append({
            "node_id": int(nid),
            "kp_count": int(len(kp)),
            "tx": float(pose[0, 3]),
            "ty": float(pose[1, 3]),
            "tz": float(pose[2, 3]),
        })

    bundle["manifest"]["keyframes_meta"] = kf_meta_list
    bundle["manifest"]["descriptor_dtype"] = str(np.dtype(desc_dtype))

    # 6. npz 저장
    np.savez_compressed(
        args.out,
        manifest=json.dumps(bundle["manifest"], ensure_ascii=False, indent=2),
        keypoints=np.array(kp_list, dtype=object),
        descriptors=np.array(desc_list, dtype=object),
        world_3d=np.array(world3d_list, dtype=object),
        poses=np.array(pose_list, dtype=object),
        global_descriptors=np.stack(gdesc_list) if gdesc_list else np.zeros((0, 384), dtype=desc_dtype),
    )
    out_size = Path(args.out).stat().st_size
    print(f"[bundle] saved → {args.out} ({out_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
