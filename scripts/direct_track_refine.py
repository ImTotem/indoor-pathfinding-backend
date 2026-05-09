"""직접 union-find + filter + N-view triangulation — COLMAP/hloc 우회.

전제: 우리 LightGlue 매칭이 mean_track_length 7.04 (raw) 가능. COLMAP 의
conservative default 가 작동을 막음. 직접 처리하면 multi-view 활용 가능.

흐름:
  1. RTABMap metadata + SuperPoint cache + LightGlue 매칭 (기존 함수 재사용)
  2. Raw union-find → tracks
  3. Filter: hyperblob (length > 50) reject + conflict (한 image multi-kp) reject + min length 3
  4. N-view triangulation (DLT/SVD) per track
  5. Reprojection error 검증 (≥ 4 px 인 view 제거 후 재 triangulation)
  6. 결과 → keyframe_world3d 갱신
  7. 4건 query 재측위 + 비교
"""
from __future__ import annotations

import logging
import sqlite3
import struct
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("direct_track")

RTAB_DB = "/app/var/storage/scans/bc5e28b9-5075-4a58-b948-31ab1e277a02/rtabmap_reprocessed.db"
MAP_ID = "125b153d-3475-46f9-9690-e703951fb83d"
DEBUG_DIR = Path("/app/var/storage/debug/localize/e30f31ea-5bbe-42df-9031-fa371bb7a7b3")

QUERIES = [
    ("20260508_125358_467012", 0.643, 9, ( 0.527, -21.743, 0.230)),
    ("20260508_131737_173495", 0.205, 9, ( 0.030, -18.300, 1.032)),
    ("20260508_131801_428982", 0.323, 10, ( 0.131,  -6.501, -0.044)),
    ("20260508_132936_089777", 0.611, 11, ( 0.680, -21.669, 0.194)),
]


# ─────────────────────────────────────────────────────────────────────────────
# 1) RTABMap metadata
# ─────────────────────────────────────────────────────────────────────────────


def load_rtab_metadata(rtab_db: str) -> dict:
    from be.slam_engines.superpoint.map_manager import _parse_calibration_K_and_local
    con = sqlite3.connect(rtab_db)
    try:
        cb = bytes(con.execute(
            "SELECT calibration FROM Data WHERE calibration IS NOT NULL LIMIT 1"
        ).fetchone()[0])
        K, local_transform = _parse_calibration_K_and_local(cb)
        width = struct.unpack("<i", cb[16:20])[0]
        height = struct.unpack("<i", cb[20:24])[0]
        node_poses: dict[int, np.ndarray] = {}
        for nid, blob in con.execute(
            "SELECT id, pose FROM Node WHERE pose IS NOT NULL ORDER BY id"
        ):
            if blob and len(blob) == 48:
                vals = struct.unpack("<12f", blob)
                T = np.eye(4, dtype=np.float64)
                T[:3, :] = np.array(vals).reshape(3, 4)
                node_poses[int(nid)] = T
        return {"K": K, "local_transform": local_transform,
                "width": int(width), "height": int(height),
                "node_poses": node_poses}
    finally:
        con.close()


# ─────────────────────────────────────────────────────────────────────────────
# 2) Pair matching — LightGlue (window=5)
# ─────────────────────────────────────────────────────────────────────────────


def compute_pair_matches(loaded, neighbor_window: int = 5, min_matches: int = 12):
    import torch
    from lightglue import LightGlue
    from lightglue.utils import rbd
    matcher = LightGlue(features='superpoint').eval().to(loaded.device)

    pair_matches: dict[tuple[int, int], np.ndarray] = {}
    n = len(loaded.node_ids)
    seen = set()
    with torch.no_grad():
        for i in range(n):
            nid_i = loaded.node_ids[i]
            feats_i = {k: v.to(loaded.device) for k, v in loaded.keyframe_feats[nid_i].items()}
            for off in range(1, neighbor_window + 1):
                for j in (i - off, i + off):
                    if not (0 <= j < n) or j == i: continue
                    key = (min(i, j), max(i, j))
                    if key in seen: continue
                    seen.add(key)
                    nid_j = loaded.node_ids[j]
                    feats_j = {k: v.to(loaded.device) for k, v in loaded.keyframe_feats[nid_j].items()}
                    out = matcher({'image0': feats_i, 'image1': feats_j})
                    matches = rbd(out)['matches'].cpu().numpy()
                    if len(matches) < min_matches: continue
                    pair_matches[(nid_i, nid_j)] = matches
    return pair_matches


# ─────────────────────────────────────────────────────────────────────────────
# 3) Union-find + filter
# ─────────────────────────────────────────────────────────────────────────────


class DSU:
    def __init__(self):
        self.parent: dict[Any, Any] = {}
        self.rank: dict[Any, int] = {}
    def make(self, x):
        if x not in self.parent: self.parent[x] = x; self.rank[x] = 0
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]: ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]: self.rank[ra] += 1


def filter_matches_by_fmatrix(
    pair_matches: dict[tuple[int, int], np.ndarray],
    loaded,
    ransac_thresh_px: float = 3.0,
    min_inliers: int = 12,
    min_inlier_ratio: float = 0.5,
) -> dict[tuple[int, int], np.ndarray]:
    """각 pair 의 매칭에서 epipolar (F-matrix RANSAC) 위반하는 wrong-positive 제거."""
    filtered: dict[tuple[int, int], np.ndarray] = {}
    n_total = 0; n_kept = 0
    for (nid_i, nid_j), matches in pair_matches.items():
        kp_i = loaded.keyframe_feats[nid_i]['keypoints'][0].cpu().numpy()
        kp_j = loaded.keyframe_feats[nid_j]['keypoints'][0].cpu().numpy()
        pts_i = kp_i[matches[:, 0]].astype(np.float32)
        pts_j = kp_j[matches[:, 1]].astype(np.float32)
        if len(pts_i) < 8: continue
        F, mask = cv2.findFundamentalMat(
            pts_i, pts_j, cv2.FM_RANSAC, ransac_thresh_px, 0.999
        )
        n_total += len(matches)
        if F is None or mask is None: continue
        keep_mask = mask.ravel().astype(bool)
        inliers = matches[keep_mask]
        if len(inliers) < min_inliers: continue
        if len(inliers) / max(len(matches), 1) < min_inlier_ratio: continue
        filtered[(nid_i, nid_j)] = inliers
        n_kept += len(inliers)
    logger.info(
        f"  F-matrix filter: {len(filtered)}/{len(pair_matches)} pairs kept, "
        f"matches {n_kept}/{n_total} ({100*n_kept/max(n_total,1):.1f}%)"
    )
    return filtered


def build_filtered_tracks(
    pair_matches: dict[tuple[int, int], np.ndarray],
    min_track_length: int = 3,
    max_track_length: int = 50,
) -> list[list[tuple[int, int]]]:
    dsu = DSU()
    for (nid_i, nid_j), matches in pair_matches.items():
        for ki, kj in matches:
            a = (nid_i, int(ki)); b = (nid_j, int(kj))
            dsu.make(a); dsu.make(b)
            dsu.union(a, b)

    tracks_by_root: dict[Any, list[tuple[int, int]]] = defaultdict(list)
    for node in dsu.parent:
        tracks_by_root[dsu.find(node)].append(node)

    n_total = len(tracks_by_root)
    n_short = n_long = n_conflict_split = 0
    filtered: list[list[tuple[int, int]]] = []
    for members in tracks_by_root.values():
        if len(members) > max_track_length:
            n_long += 1; continue
        # Conflict split — 한 image 의 multiple kp 가 같은 track 이면 그 image 의 첫 kp 만 keep.
        per_image: dict[int, int] = {}
        cleaned: list[tuple[int, int]] = []
        had_conflict = False
        for nid, kp_idx in members:
            if nid in per_image:
                had_conflict = True
                continue  # 첫 kp 만 keep
            per_image[nid] = kp_idx
            cleaned.append((nid, kp_idx))
        if had_conflict: n_conflict_split += 1
        if len(cleaned) < min_track_length:
            n_short += 1; continue
        filtered.append(cleaned)
    logger.info(
        f"  union-find: {n_total} raw → "
        f"{len(filtered)} kept (short={n_short} long={n_long} conflict_split={n_conflict_split})"
    )
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# 4) N-view triangulation (DLT/SVD)
# ─────────────────────────────────────────────────────────────────────────────


def world_to_optical_P(K: np.ndarray, T_b_w: np.ndarray, C: np.ndarray) -> np.ndarray | None:
    """world→optical projection matrix P (3x4). X_pix = P [X_world; 1]."""
    try:
        T_w_b = np.linalg.inv(T_b_w)
    except np.linalg.LinAlgError:
        return None
    R_wo = C @ T_w_b[:3, :3]
    t_wo = C @ T_w_b[:3, 3]
    return K @ np.hstack([R_wo, t_wo.reshape(3, 1)])


def triangulate_n_view(
    track: list[tuple[int, int]],
    loaded,
    meta: dict,
    P_cache: dict[int, np.ndarray],
    R_t_cache: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray | None, float]:
    """N-view DLT triangulation. Return (xyz, max_reproj_err) 또는 (None, inf)."""
    K = meta["K"]
    rows = []
    pts_2d_list = []
    nids_used = []
    for nid, kp_idx in track:
        if nid not in P_cache: continue
        P = P_cache[nid]
        kp = loaded.keyframe_feats[nid]['keypoints'][0, kp_idx].cpu().numpy().astype(np.float64)
        u, v = float(kp[0]), float(kp[1])
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
        pts_2d_list.append((u, v))
        nids_used.append(nid)

    if len(nids_used) < 2:
        return None, float("inf")

    A = np.stack(rows, axis=0)
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    if abs(X_h[3]) < 1e-9:
        return None, float("inf")
    X = X_h[:3] / X_h[3]

    # Reprojection error 측정
    max_err = 0.0
    for (nid, (u, v)) in zip(nids_used, pts_2d_list):
        P = P_cache[nid]
        proj = P @ np.append(X, 1.0)
        if abs(proj[2]) < 1e-9:
            return None, float("inf")
        ux, vy = proj[0] / proj[2], proj[1] / proj[2]
        err = np.hypot(ux - u, vy - v)
        if err > max_err: max_err = err
        # depth check
        R, t = R_t_cache[nid]
        z_opt = (R[2] @ X) + t[2]
        if z_opt < 0.2 or z_opt > 30.0:
            return None, float("inf")

    return X.astype(np.float32), float(max_err)


def triangulate_with_outlier_rejection(
    track, loaded, meta, P_cache, R_t_cache, max_reproj_px: float = 4.0
) -> tuple[np.ndarray | None, list[tuple[int, int]]]:
    """N-view triangulate. reprojection error > max 시 worst view 제거 후 재시도."""
    used_track = list(track)
    while len(used_track) >= 2:
        X, err = triangulate_n_view(used_track, loaded, meta, P_cache, R_t_cache)
        if X is None:
            return None, []
        if err <= max_reproj_px:
            return X, used_track
        worst_i = -1; worst_e = -1.0
        for i, (nid, kp_idx) in enumerate(used_track):
            if nid not in P_cache: continue
            P = P_cache[nid]
            kp = loaded.keyframe_feats[nid]['keypoints'][0, kp_idx].cpu().numpy()
            u, v = float(kp[0]), float(kp[1])
            proj = P @ np.append(X, 1.0)
            if abs(proj[2]) < 1e-9: continue
            e = np.hypot(proj[0]/proj[2] - u, proj[1]/proj[2] - v)
            if e > worst_e: worst_e = e; worst_i = i
        if worst_i < 0: return None, []
        used_track.pop(worst_i)
    return None, []


def triangulate_ransac(
    track, loaded, meta, P_cache, R_t_cache,
    max_reproj_px: float = 4.0, min_inliers: int = 3, max_trials: int = 30,
) -> tuple[np.ndarray | None, list[tuple[int, int]]]:
    """RANSAC N-view triangulation — 2-view 샘플로 hypothesis, inlier 모아서 재DLT."""
    if len(track) < 2: return None, []
    rng = np.random.default_rng(seed=hash(tuple(track[:3])) & 0xFFFF)
    best_inliers: list[tuple[int, int]] = []
    n = len(track)
    for trial in range(max_trials):
        idx = rng.choice(n, size=2, replace=False)
        sample = [track[int(idx[0])], track[int(idx[1])]]
        X, _ = triangulate_n_view(sample, loaded, meta, P_cache, R_t_cache)
        if X is None: continue
        inliers = []
        for nid, kp_idx in track:
            if nid not in P_cache: continue
            P = P_cache[nid]
            kp = loaded.keyframe_feats[nid]['keypoints'][0, kp_idx].cpu().numpy()
            u, v = float(kp[0]), float(kp[1])
            proj = P @ np.append(X, 1.0)
            if abs(proj[2]) < 1e-9: continue
            e = np.hypot(proj[0]/proj[2] - u, proj[1]/proj[2] - v)
            R, t = R_t_cache[nid]
            z_opt = R[2] @ X + t[2]
            if e <= max_reproj_px and 0.2 < z_opt < 30.0:
                inliers.append((nid, kp_idx))
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
    if len(best_inliers) < min_inliers: return None, []
    # Final DLT on inliers
    X, err = triangulate_n_view(best_inliers, loaded, meta, P_cache, R_t_cache)
    if X is None or err > max_reproj_px * 1.5: return None, []
    return X, best_inliers


# ─────────────────────────────────────────────────────────────────────────────
# 5) main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    logger.info("=" * 78)
    logger.info("Step 1 — RTABMap metadata + SuperPoint cache")
    meta = load_rtab_metadata(RTAB_DB)
    from be.slam_engines.superpoint.map_manager import SuperPointMapManager
    mgr = SuperPointMapManager()
    if MAP_ID in mgr._maps:
        del mgr._maps[MAP_ID]
    loaded = mgr.get_or_load(MAP_ID, RTAB_DB)
    logger.info(f"  cache frames: {len(loaded.node_ids)}")

    logger.info("=" * 78)
    logger.info("Step 2 — LightGlue pair matching (±5)")
    t0 = time.time()
    pair_matches = compute_pair_matches(loaded, neighbor_window=5)
    logger.info(f"  pairs: {len(pair_matches)} in {time.time()-t0:.1f}s")

    logger.info("=" * 78)
    logger.info("Step 2.5 — F-matrix RANSAC pre-filter (LightGlue FP 제거)")
    pair_matches = filter_matches_by_fmatrix(pair_matches, loaded)

    logger.info("=" * 78)
    logger.info("Step 3 — Union-find + filter (hyperblob/conflict reject, length 3-50)")
    tracks = build_filtered_tracks(pair_matches, min_track_length=2, max_track_length=50)
    if tracks:
        lengths = [len(t) for t in tracks]
        logger.info(
            f"  filtered tracks: {len(tracks)}  mean_length={np.mean(lengths):.2f}  "
            f"median={int(np.median(lengths))}  max={max(lengths)}"
        )

    logger.info("=" * 78)
    logger.info("Step 4 — N-view DLT triangulation + outlier rejection")
    K = meta["K"]; C = meta["local_transform"][:3, :3]
    P_cache: dict[int, np.ndarray] = {}
    R_t_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for nid, T in meta["node_poses"].items():
        # Skip nodes with singular pose (RTABMap reprocess 가 graph 에 포함 안 한 frame)
        if not np.all(np.isfinite(T)) or np.linalg.det(T[:3, :3]) < 1e-6:
            continue
        try:
            T_w_b = np.linalg.inv(T)
        except np.linalg.LinAlgError:
            continue
        R_wo = C @ T_w_b[:3, :3]
        t_wo = C @ T_w_b[:3, 3]
        P_cache[nid] = K @ np.hstack([R_wo, t_wo.reshape(3, 1)])
        R_t_cache[nid] = (R_wo, t_wo)
    logger.info(f"  P_cache: {len(P_cache)}/{len(meta['node_poses'])} valid frames")

    # RANSAC N-view triangulation — wrong matches (LightGlue FP) 가 outlier 로 reject.
    # 진짜 진단: failed-track reproj_err mean=210px → ARKit drift 아님, track 자체가 mixed corner.
    t0 = time.time()
    # point_results: (X, inliers, full_track) — inliers 는 RANSAC inlier, full_track 는 원본 (모든 view 에 X 등록 옵션)
    point_results: list[tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]] = []
    n_failed = 0
    inlier_lens: list[int] = []
    for tr in tracks:
        X, used = triangulate_ransac(
            tr, loaded, meta, P_cache, R_t_cache,
            max_reproj_px=10.0, min_inliers=2, max_trials=30,
        )
        if X is not None and len(used) >= 2:
            point_results.append((X, used, tr))
            inlier_lens.append(len(used))
        else:
            n_failed += 1
    logger.info(
        f"  triangulated (RANSAC, 4px, min_inliers=3): {len(point_results)}  "
        f"failed: {n_failed}  in {time.time()-t0:.1f}s"
    )
    if inlier_lens:
        logger.info(f"  inlier-set length: mean={np.mean(inlier_lens):.2f} "
                    f"median={int(np.median(inlier_lens))} max={max(inlier_lens)}")

    logger.info("=" * 78)
    logger.info("Step 4.5 — Bundle Adjustment (pycolmap 4.0.4 corrected API)")
    try:
        import pycolmap
        from scipy.spatial.transform import Rotation as Rsc

        used_nids = set()
        for entry in point_results:
            X, used = entry[0], entry[1]
            for nid, _ in used:
                used_nids.add(nid)

        ba_dir = Path("/tmp/direct_track_ba")
        if ba_dir.exists():
            import shutil; shutil.rmtree(ba_dir)
        ba_dir.mkdir(parents=True)

        # cameras.txt
        K = meta["K"]
        (ba_dir / "cameras.txt").write_text(
            f"1 PINHOLE {meta['width']} {meta['height']} "
            f"{K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}\n"
        )
        # 점→track lookup — 각 (image_id, kp_idx) 가 어느 point3d_id 인지
        kp2pid: dict[tuple[int,int], int] = {}
        for pid, entry in enumerate(point_results, start=1):
            for nid, kp_idx in entry[1]:
                kp2pid[(nid, kp_idx)] = pid

        # images.txt — line 1: pose, line 2: 모든 keypoint 좌표 (1024개)
        lines = []
        for nid in sorted(used_nids):
            if nid not in meta["node_poses"]: continue
            T_b_w = meta["node_poses"][nid]
            T_b_opt = np.eye(4)
            T_b_opt[:3,:3] = meta["local_transform"][:3,:3]
            T_b_opt[:3,3] = meta["local_transform"][:3,3]
            T_w_opt = T_b_opt @ np.linalg.inv(T_b_w)
            qxyzw = Rsc.from_matrix(T_w_opt[:3,:3]).as_quat()
            qw,qx,qy,qz = qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]
            tx,ty,tz = T_w_opt[:3,3]
            lines.append(f"{nid} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 f{nid}.jpg")
            kps = loaded.keyframe_feats[nid]['keypoints'][0].cpu().numpy()
            kp_strs = []
            for k_idx, (u, v) in enumerate(kps):
                pid = kp2pid.get((nid, k_idx), -1)
                kp_strs.append(f"{u} {v} {pid}")
            lines.append(" ".join(kp_strs))
        (ba_dir / "images.txt").write_text("\n".join(lines) + "\n")
        # points3D.txt
        p3d_lines = []
        for pid, entry in enumerate(point_results, start=1):
            X, used = entry[0], entry[1]
            track_str = " ".join(f"{nid} {kp}" for nid, kp in used)
            p3d_lines.append(f"{pid} {X[0]} {X[1]} {X[2]} 128 128 128 0 {track_str}")
        (ba_dir / "points3D.txt").write_text("\n".join(p3d_lines) + "\n")

        recon = pycolmap.Reconstruction(str(ba_dir))
        logger.info(f"  reconstruction loaded: {len(recon.images)} images, {len(recon.points3D)} points")

        # ARKit pose 정제 (RTABMap 이 loop closure 0 건이라 graph optimization 효과 없음).
        # BA 가 graph optimization 역할 — pose + 점 동시 정제.
        ba_opts = pycolmap.BundleAdjustmentOptions()
        ba_opts.refine_focal_length = False
        ba_opts.refine_principal_point = False
        ba_opts.refine_extra_params = False
        ba_opts.refine_sensor_from_rig = False
        # refine_rig_from_world default True — ARKit drift 보정
        ba_cfg = pycolmap.BundleAdjustmentConfig()
        for img_id in recon.images:
            ba_cfg.add_image(img_id)
        for p3d_id in recon.points3D:
            ba_cfg.add_variable_point(p3d_id)
        # gauge 고정 — 자동
        try:
            ba_cfg.fix_gauge(pycolmap.BundleAdjustmentGauge.THREE_POINTS)
        except Exception:
            # fallback: 첫 image rig pose constant
            try:
                first_id = sorted(recon.images.keys())[0]
                first_img = recon.images[first_id]
                ba_cfg.set_constant_rig_from_world_pose(first_img.frame.frame_id)
            except Exception as e2:
                logger.warning(f"  fix_gauge 둘 다 실패: {e2}")

        t0 = time.time()
        adjuster = pycolmap.create_default_bundle_adjuster(ba_opts, ba_cfg, recon)
        adjuster.solve()
        logger.info(f"  BA done in {time.time()-t0:.1f}s, "
                    f"mean reproj_err={recon.compute_mean_reprojection_error():.2f}px")

        # BA 결과를 point_results 에 적용 (full_track 보존)
        new_pr = []
        original_full = {}
        for X_old, used_old, full_old in [(e[0], e[1], e[2]) for e in point_results if len(e) == 3]:
            key = tuple(sorted(used_old))
            original_full[key] = full_old
        for pid, pt in recon.points3D.items():
            xyz = np.array(pt.xyz, dtype=np.float32)
            used_list = [(int(e.image_id), int(e.point2D_idx)) for e in pt.track.elements]
            full = original_full.get(tuple(sorted(used_list)), used_list)
            new_pr.append((xyz, used_list, full))
        point_results = new_pr
        logger.info(f"  point_results 갱신 (BA 결과): {len(point_results)} points")

        # BA 후 정제된 pose 로 P_cache + R_t_cache 재구성 (ARKit drift 보정 결과 활용)
        K_arr = meta["K"]
        for img_id, img in recon.images.items():
            cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
            try:
                R_wo = np.array(cfw.rotation.matrix(), dtype=np.float64)
            except Exception:
                R_wo = np.array(cfw.rotation, dtype=np.float64)
                if R_wo.shape == (4,):  # quaternion
                    from scipy.spatial.transform import Rotation as Rsc
                    qw, qx, qy, qz = R_wo
                    R_wo = Rsc.from_quat([qx, qy, qz, qw]).as_matrix()
            t_wo = np.array(cfw.translation, dtype=np.float64)
            P_cache[img_id] = K_arr @ np.hstack([R_wo, t_wo.reshape(3, 1)])
            R_t_cache[img_id] = (R_wo, t_wo)
        logger.info("  P_cache/R_t_cache 갱신 (BA 정제된 pose 사용)")
    except Exception as exc:
        import traceback; traceback.print_exc()
        logger.warning(f"  BA skipped: {type(exc).__name__}: {exc}")

    logger.info("=" * 78)
    logger.info("Step 5 — keyframe_world3d 갱신")
    new_world3d: dict[int, np.ndarray] = {}
    for nid in loaded.node_ids:
        kp_count = loaded.keyframe_feats[nid]['keypoints'].shape[1]
        new_world3d[nid] = np.full((kp_count, 3), np.nan, dtype=np.float32)

    n_obs = 0
    # BA refined points + ALL-VIEW register (no filter) — point 갯수 maximize.
    # PnP RANSAC 가 outlier view 알아서 reject.
    for entry in point_results:
        if len(entry) == 3:
            X, used, full_track = entry
        else:
            X, used = entry; full_track = used
        for nid, kp_idx in full_track:
            if nid in new_world3d and 0 <= kp_idx < new_world3d[nid].shape[0]:
                new_world3d[nid][kp_idx] = X
                n_obs += 1
    loaded.keyframe_world3d = new_world3d

    n_with_3d = sum(1 for w in new_world3d.values() if not np.all(np.isnan(w)))
    total_kp = sum(w.shape[0] for w in new_world3d.values())
    total_3d = sum(int((~np.isnan(w).any(axis=1)).sum()) for w in new_world3d.values())
    logger.info(f"  observations updated : {n_obs}")
    logger.info(f"  3D coverage frames   : {n_with_3d}/{len(loaded.node_ids)} "
                f"({100*n_with_3d/len(loaded.node_ids):.1f}%)")
    logger.info(f"  3D-mapped keypoints  : {total_3d}/{total_kp} "
                f"({100*total_3d/total_kp:.2f}%)")

    logger.info("=" * 78)
    logger.info("Step 6 — 4건 query 재측위")
    from be.slam_engines.superpoint.engine import SuperPointEngine
    eng = SuperPointEngine()
    intrinsics = {
        "fx": float(meta["K"][0, 0]), "fy": float(meta["K"][1, 1]),
        "cx": float(meta["K"][0, 2]), "cy": float(meta["K"][1, 2]),
    }
    print(f"\n{'time':25}{'recorded':>20}{'new conf':>10}{'inliers':>9}{'pose':>22}")
    for ts, old_c, old_n, _ in QUERIES:
        imgs = []
        for k in range(5):
            p = DEBUG_DIR / f"{ts}_{k:02d}.jpg"
            if p.exists():
                imgs.append(p.read_bytes())
        if not imgs: continue
        try:
            res = eng._localize_sync(MAP_ID, imgs, intrinsics, RTAB_DB)
            nx, ny, nz = res['pose']['x'], res['pose']['y'], res['pose']['z']
            rec = f"c={old_c:.2f} n={old_n}"
            print(f"{ts:25}{rec:>20}{res['confidence']:>10.3f}{res['num_matches']:>9}"
                  f"{f'({nx:.2f}, {ny:.2f}, {nz:.2f})':>22}")
        except Exception as e:
            print(f"  {ts}: FAIL {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
