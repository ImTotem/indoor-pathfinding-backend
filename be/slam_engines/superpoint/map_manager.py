"""SuperPoint feature index built from RTABMap .db keyframe images.

Loads every keyframe image stored in the RTABMap SQLite database,
extracts SuperPoint features, and associates them with world-frame
3D positions taken from the RTABMap Feature table.

No service-layer files are modified — this is a standalone index.
"""
import sqlite3
import struct
import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from utils import logger

MAX_CACHED_MAPS = 5
TOP_K = 5


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _parse_node_transforms(conn: sqlite3.Connection) -> Dict[int, np.ndarray]:
    """Return {node_id: 3x4 world-transform matrix} from the Node table."""
    result: Dict[int, np.ndarray] = {}
    for node_id, blob in conn.execute(
        "SELECT id, pose FROM Node WHERE pose IS NOT NULL"
    ):
        if not blob or len(blob) != 48:
            continue
        vals = struct.unpack('<12f', blob)
        if all(v == 0.0 for v in vals):
            continue
        result[node_id] = np.array(vals, dtype=np.float64).reshape(3, 4)
    return result


def _load_world_features(
    conn: sqlite3.Connection,
    transforms: Dict[int, np.ndarray],
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Load RTABMap 2D keypoint positions and their world-frame 3D coordinates.

    Returns {node_id: (pos_2d (M,2) float32, world_3d (M,3) float32 with NaN rows
    where depth is unavailable)}.
    """
    rows = conn.execute(
        "SELECT node_id, pos_x, pos_y, depth_x, depth_y, depth_z FROM Feature "
        "WHERE pos_x IS NOT NULL AND pos_y IS NOT NULL"
    ).fetchall()

    buf: Dict[int, Tuple[List, List]] = {}
    for node_id, px, py, dx, dy, dz in rows:
        if node_id not in buf:
            buf[node_id] = ([], [])
        buf[node_id][0].append([float(px), float(py)])
        T = transforms.get(node_id)
        if T is not None and dx is not None and dy is not None and dz is not None:
            local = np.array([dx, dy, dz], dtype=np.float64)
            world = T[:, :3] @ local + T[:, 3]
            buf[node_id][1].append(world.tolist())
        else:
            buf[node_id][1].append([float('nan')] * 3)

    return {
        nid: (
            np.array(p2d, dtype=np.float32),
            np.array(w3d, dtype=np.float32),
        )
        for nid, (p2d, w3d) in buf.items()
    }


def _assign_world_3d(
    sp_kps: np.ndarray,     # (N, 2) SuperPoint keypoints in image coords
    rtab_2d: np.ndarray,    # (M, 2) RTABMap 2D feature positions
    rtab_w3d: np.ndarray,   # (M, 3) corresponding world 3D (may have NaN)
    max_px: float = 8.0,
) -> np.ndarray:
    """Vectorised nearest-neighbour assignment of world 3D to SuperPoint kps."""
    out = np.full((len(sp_kps), 3), float('nan'), dtype=np.float32)
    if len(rtab_2d) == 0:
        return out
    # (N, M) pairwise L2 distances in image space
    diffs = sp_kps[:, None, :] - rtab_2d[None, :, :]   # (N, M, 2)
    dists = np.linalg.norm(diffs, axis=2)               # (N, M)
    j_min = np.argmin(dists, axis=1)                    # (N,)
    min_dists = dists[np.arange(len(sp_kps)), j_min]
    mask = min_dists <= max_px
    out[mask] = rtab_w3d[j_min[mask]]
    return out


def _load_gray_float(conn: sqlite3.Connection, node_id: int) -> Optional[np.ndarray]:
    """Load grayscale float [0,1] image for a node from the Data table."""
    row = conn.execute(
        "SELECT image FROM Data WHERE id = ?", (node_id,)
    ).fetchone()
    if not row or not row[0]:
        return None
    arr = np.frombuffer(bytes(row[0]), dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def _parse_calibration_K_and_local(blob: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """RTABMap 0.23.x calibration BLOB (164B) → (K 3x3, local_transform 3x4).

    Layout (write_rtabmap_db 와 동일):
      [44..115]  9x float64 = K matrix (row-major)
      [116..163] 12x float32 = local_transform (row-major)
    """
    import struct
    if len(blob) < 116:
        raise ValueError(f"calibration blob too short: {len(blob)} bytes")
    K_vals = struct.unpack('<9d', blob[44:44 + 72])
    K = np.array(K_vals, dtype=np.float64).reshape(3, 3)
    if len(blob) >= 164:
        lt_vals = struct.unpack('<12f', blob[116:116 + 48])
        local_transform = np.array(lt_vals, dtype=np.float64).reshape(3, 4)
    else:
        local_transform = np.zeros((3, 4))
        local_transform[:3, :3] = np.eye(3)
    return K, local_transform


def _smart_select_keyframes(
    all_ids: List[int],
    transforms: Dict[int, np.ndarray],
    min_translation_m: float = 0.30,
    min_rotation_deg: float = 15.0,
    force_keep_rotation_deg: float = 45.0,
) -> List[int]:
    """Translation/rotation 누적 기반 keyframe subsample.

    인접 frame 간 baseline 이 너무 작으면 triangulation 부정확. 같은 view 를
    오래 본 frame 도 redundant. 마지막 keep 된 frame 으로부터:
      - translation ≥ min_translation_m  → keep
      - rotation ≥ min_rotation_deg     → keep
      - rotation ≥ force_keep_rotation_deg → keep (큰 방향 전환은 무조건)
    셋 다 미달이면 redundant.
    """
    import math

    def _yaw(R: np.ndarray) -> float:
        return math.atan2(float(R[1, 0]), float(R[0, 0]))

    def _rot_diff(R_a: np.ndarray, R_b: np.ndarray) -> float:
        # axis-angle 거리 (안정적). trace((R_a^T R_b)) → angle
        R = R_b @ R_a.T
        cos_t = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
        return math.degrees(math.acos(cos_t))

    keep: List[int] = []
    last_T: np.ndarray | None = None
    for nid in all_ids:
        T34 = transforms.get(nid)
        if T34 is None:
            continue
        if last_T is None:
            keep.append(nid)
            last_T = T34
            continue
        d = float(np.linalg.norm(T34[:3, 3] - last_T[:3, 3]))
        rot = _rot_diff(last_T[:3, :3], T34[:3, :3])
        if (
            d >= min_translation_m
            or rot >= min_rotation_deg
            or rot >= force_keep_rotation_deg
        ):
            keep.append(nid)
            last_T = T34
    return keep


# ---------------------------------------------------------------------------
# Loaded map
# ---------------------------------------------------------------------------

class SuperPointLoadedMap:
    """SuperPoint feature index for all keyframes in one RTABMap DB."""

    def __init__(self, map_id: str, db_path: str, device: torch.device):
        self.map_id = map_id
        self.db_path = db_path
        self.device = device

        self.node_ids: List[int] = []
        # CPU tensors: {'keypoints': (1,N,2), 'descriptors': (1,N,256), 'image_size': (1,2)}
        self.keyframe_feats: Dict[int, dict] = {}
        # (N, 3) world 3D per keyframe keypoint; NaN where unavailable
        self.keyframe_world3d: Dict[int, np.ndarray] = {}
        # (K, 256) mean descriptors for global retrieval
        self.global_descs: Optional[torch.Tensor] = None

        self._build_index()

    def _build_index(self):
        from lightglue import SuperPoint

        extractor = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
        t0 = time.time()

        conn = sqlite3.connect(self.db_path)
        try:
            all_ids = [r[0] for r in conn.execute(
                "SELECT id FROM Node WHERE pose IS NOT NULL ORDER BY id"
            ).fetchall()]

            transforms = _parse_node_transforms(conn)
            world_feats = _load_world_features(conn, transforms)

            global_descs: List[torch.Tensor] = []
            from .global_descriptor import GlobalDescExtractor
            global_desc_ext = GlobalDescExtractor(self.device)

            for node_id in all_ids:
                img = _load_gray_float(conn, node_id)
                if img is None:
                    continue

                tensor = torch.from_numpy(img)[None, None].to(self.device)
                with torch.no_grad():
                    feats = extractor.extract(tensor)

                cpu = {k: v.cpu() for k, v in feats.items()}
                self.keyframe_feats[node_id] = cpu
                self.node_ids.append(node_id)

                # DINOv2 global descriptor (384-dim) instead of mean SuperPoint (256-dim)
                img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
                global_descs.append(global_desc_ext.extract(img_uint8))  # (384,)

                sp_kps = cpu['keypoints'][0].numpy()   # (N, 2)
                if node_id in world_feats:
                    r2d, w3d = world_feats[node_id]
                    self.keyframe_world3d[node_id] = _assign_world_3d(sp_kps, r2d, w3d)
                else:
                    self.keyframe_world3d[node_id] = np.full(
                        (len(sp_kps), 3), float('nan'), dtype=np.float32
                    )

                n = len(self.node_ids)
                if n % 100 == 0:
                    logger.info(
                        f"[SuperPoint] '{self.map_id}': indexed {n}/{len(all_ids)} frames"
                    )
        finally:
            conn.close()

        if global_descs:
            self.global_descs = torch.stack(global_descs)   # (K, 384)

        n_with_3d = sum(
            1 for v in self.keyframe_world3d.values()
            if not np.all(np.isnan(v))
        )
        logger.info(
            f"[SuperPoint] Map '{self.map_id}' indexed in {time.time()-t0:.1f}s: "
            f"{len(self.node_ids)} frames, {n_with_3d} with 3D coverage"
        )

        # Fallback: RTAB-Map Feature 에 depth 가 없어 3D coverage 가 0 인 경우
        # 이웃 keyframe pair multi-view triangulation 으로 world 3D 추정
        if n_with_3d == 0 and len(self.node_ids) >= 2:
            logger.info(
                "[SuperPoint] No depth coverage from RTAB-Map — running multi-view triangulation"
            )
            t1 = time.time()
            self._triangulate_via_multi_view()
            n_with_3d = sum(
                1 for v in self.keyframe_world3d.values()
                if not np.all(np.isnan(v))
            )
            logger.info(
                f"[SuperPoint] Triangulation done in {time.time()-t1:.1f}s: "
                f"{n_with_3d}/{len(self.node_ids)} frames with 3D coverage"
            )

            # ML depth fill — 실험적. NaN 은 줄지만 PnP inliers 안 늘어 측위 정확도 역효과.
            # opt-in: env INDOOR_ENABLE_ML_DEPTH_FILL=1 일 때만 활성.
            import os as _os
            if _os.environ.get("INDOOR_ENABLE_ML_DEPTH_FILL") == "1":
                try:
                    t2 = time.time()
                    n_filled, n_attempted = self._fill_nan_with_ml_depth()
                    logger.info(
                        f"[SuperPoint] ML depth fill done in {time.time()-t2:.1f}s: "
                        f"filled {n_filled}/{n_attempted} NaN keypoints"
                    )
                except Exception as exc:
                    logger.warning(
                        f"[SuperPoint] ML depth fill skipped: {type(exc).__name__}: {exc}"
                    )

    def _triangulate_via_multi_view(
        self,
        neighbor_offsets: tuple[int, ...] = (-2, -1, 1, 2),
        min_matches: int = 12,
        min_depth_m: float = 0.2,
        max_depth_m: float = 30.0,
    ) -> None:
        """이웃 keyframe pair LightGlue 매칭 + cv2.triangulatePoints → world 3D.

        Node.pose 는 base_link → world (3x4). Calibration BLOB 의 local_transform
        은 base_link → camera_optical 변환. P = K @ T_world_to_optical.
        """
        from lightglue import LightGlue
        from lightglue.utils import rbd

        conn = sqlite3.connect(self.db_path)
        try:
            transforms = _parse_node_transforms(conn)  # base→world (3x4)
            # K + local_transform 추출 (모든 keyframe 동일 가정 — 첫 keyframe 사용)
            calib_row = conn.execute(
                "SELECT calibration FROM Data WHERE id = ? LIMIT 1",
                (self.node_ids[0],),
            ).fetchone()
            if not calib_row or not calib_row[0]:
                logger.warning("[SuperPoint] No calibration in Data — skipping triangulation")
                return
            K, local_transform = _parse_calibration_K_and_local(bytes(calib_row[0]))
            C = local_transform[:3, :3]  # base → optical (3x3)
        finally:
            conn.close()

        matcher = LightGlue(features='superpoint').eval().to(self.device)

        # 누적 buffer: keyframe id → kp index → list of triangulated 3D points
        accum: Dict[int, list[list[np.ndarray]]] = {}
        for nid in self.node_ids:
            kp_count = self.keyframe_feats[nid]['keypoints'].shape[1]
            accum[nid] = [[] for _ in range(kp_count)]

        n = len(self.node_ids)
        pair_processed = 0
        with torch.no_grad():
            for i in range(n):
                nid_i = self.node_ids[i]
                if nid_i not in transforms:
                    continue
                T_bi_w = np.eye(4, dtype=np.float64)
                T_bi_w[:3, :] = transforms[nid_i]
                T_w_bi = np.linalg.inv(T_bi_w)
                # world → optical_i
                R_wo_i = C @ T_w_bi[:3, :3]
                t_wo_i = C @ T_w_bi[:3, 3]
                P_i = K @ np.hstack([R_wo_i, t_wo_i.reshape(3, 1)])

                feats_i = {k: v.to(self.device) for k, v in self.keyframe_feats[nid_i].items()}
                kps_i_np = self.keyframe_feats[nid_i]['keypoints'][0].cpu().numpy()

                for offset in neighbor_offsets:
                    j = i + offset
                    if j < 0 or j >= n:
                        continue
                    nid_j = self.node_ids[j]
                    if nid_j not in transforms:
                        continue
                    T_bj_w = np.eye(4, dtype=np.float64)
                    T_bj_w[:3, :] = transforms[nid_j]
                    T_w_bj = np.linalg.inv(T_bj_w)
                    R_wo_j = C @ T_w_bj[:3, :3]
                    t_wo_j = C @ T_w_bj[:3, 3]
                    P_j = K @ np.hstack([R_wo_j, t_wo_j.reshape(3, 1)])

                    feats_j = {k: v.to(self.device) for k, v in self.keyframe_feats[nid_j].items()}
                    kps_j_np = self.keyframe_feats[nid_j]['keypoints'][0].cpu().numpy()

                    result = matcher({'image0': feats_i, 'image1': feats_j})
                    matches = rbd(result)['matches'].cpu().numpy()
                    if len(matches) < min_matches:
                        continue

                    pts_i = kps_i_np[matches[:, 0]].T.astype(np.float64)  # (2, M)
                    pts_j = kps_j_np[matches[:, 1]].T.astype(np.float64)

                    X_h = cv2.triangulatePoints(P_i, P_j, pts_i, pts_j)  # (4, M)
                    valid_w = np.abs(X_h[3]) > 1e-9
                    X = (X_h[:3, valid_w] / X_h[3, valid_w]).T  # (M', 3) world

                    if len(X) == 0:
                        continue

                    # 두 카메라 모두에서 positive depth & in-range
                    Xi_opt = (R_wo_i @ X.T).T + t_wo_i  # optical_i frame
                    Xj_opt = (R_wo_j @ X.T).T + t_wo_j
                    di = Xi_opt[:, 2]
                    dj = Xj_opt[:, 2]
                    in_range = (
                        (di > min_depth_m) & (di < max_depth_m) &
                        (dj > min_depth_m) & (dj < max_depth_m)
                    )

                    matches_valid_idx = np.flatnonzero(valid_w)[in_range]
                    X_kept = X[in_range]
                    for k_idx, mi in zip(matches[matches_valid_idx, 0], X_kept):
                        accum[nid_i][int(k_idx)].append(mi.astype(np.float32))
                    pair_processed += 1

        # 각 keyframe 의 kp 별 median (또는 평균) 으로 final world 3D
        for nid in self.node_ids:
            kp_count = len(accum[nid])
            arr = np.full((kp_count, 3), np.nan, dtype=np.float32)
            for kp_idx, points in enumerate(accum[nid]):
                if points:
                    arr[kp_idx] = np.median(np.stack(points, axis=0), axis=0)
            self.keyframe_world3d[nid] = arr

        logger.info(
            f"[SuperPoint] triangulation: pairs={pair_processed} (neighbor offsets={neighbor_offsets})"
        )

    def _fill_nan_with_ml_depth(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Base-hf",
        min_anchors: int = 20,
        ransac_iters: int = 400,
        ransac_inlier_thresh_rel: float = 0.08,
        depth_min_m: float = 0.2,
        depth_max_m: float = 30.0,
    ) -> tuple[int, int]:
        """ML monocular depth 로 NaN keypoint 의 world 3D 채움.

        흐름:
          1. Depth Anything V2 로 keyframe 별 relative depth map 추론.
          2. multi-view triangulated 3D 가 있는 keypoint (anchor) 의 absolute depth 와
             ML depth 의 (a, b) 를 RANSAC linear fit: abs = a * rel + b.
          3. NaN keypoint 의 (u, v) 에서 rel depth → absolute → unproject → world 3D.

        Returns: (filled_count, attempted_count)
        """
        from transformers import pipeline
        from PIL import Image

        # 모델 로드 (CUDA 시 device=0)
        device_arg = 0 if str(self.device).startswith("cuda") else -1
        depth_pipe = pipeline("depth-estimation", model=model_id, device=device_arg)

        # calibration + transforms
        conn = sqlite3.connect(self.db_path)
        try:
            transforms = _parse_node_transforms(conn)
            calib_row = conn.execute(
                "SELECT calibration FROM Data WHERE id = ? LIMIT 1",
                (self.node_ids[0],),
            ).fetchone()
            if not calib_row or not calib_row[0]:
                raise RuntimeError("calibration BLOB missing")
            K, local_transform = _parse_calibration_K_and_local(bytes(calib_row[0]))
            C = local_transform[:3, :3]
        finally:
            conn.close()

        n_filled = 0
        n_attempted = 0
        n_frames_aligned = 0
        n_frames_no_anchor = 0

        for nid in self.node_ids:
            if nid not in transforms:
                continue
            # 이미지 로드 (다른 connection — 동시성 안전)
            with sqlite3.connect(self.db_path) as conn:
                img = _load_gray_float(conn, nid)
            if img is None:
                continue

            # ML depth 추론
            img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(img_uint8).convert("RGB")
            try:
                out = depth_pipe(pil)
                rel_depth = np.array(out["depth"], dtype=np.float32)
            except Exception as exc:
                logger.warning(f"[ML depth] inference failed for node={nid}: {exc}")
                continue
            if rel_depth.shape != img.shape:
                rel_depth = cv2.resize(
                    rel_depth, (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            # camera pose (world → optical)
            T_b_w = np.eye(4, dtype=np.float64)
            T_b_w[:3, :] = transforms[nid]
            T_w_b = np.linalg.inv(T_b_w)
            R_wo = C @ T_w_b[:3, :3]
            t_wo = C @ T_w_b[:3, 3]

            kps = self.keyframe_feats[nid]['keypoints'][0].cpu().numpy().astype(np.float32)
            world3d = self.keyframe_world3d[nid]
            valid = ~np.isnan(world3d).any(axis=1)
            n_anchor = int(valid.sum())
            if n_anchor < min_anchors:
                n_frames_no_anchor += 1
                continue

            # anchor 의 absolute depth (optical z)
            X_opt_anchor = (R_wo @ world3d[valid].T).T + t_wo
            abs_d = X_opt_anchor[:, 2]

            # rel depth at anchor (u, v)
            uv_anchor = kps[valid]
            uv_int = np.round(uv_anchor).astype(int)
            uv_int[:, 0] = np.clip(uv_int[:, 0], 0, rel_depth.shape[1] - 1)
            uv_int[:, 1] = np.clip(uv_int[:, 1], 0, rel_depth.shape[0] - 1)
            rel_at = rel_depth[uv_int[:, 1], uv_int[:, 0]]

            # 이상값 제외 (anchor depth 가 양수 + 범위 내)
            ok_anchor = (abs_d > depth_min_m) & (abs_d < depth_max_m) & np.isfinite(rel_at)
            if int(ok_anchor.sum()) < min_anchors:
                n_frames_no_anchor += 1
                continue
            rel_at = rel_at[ok_anchor]
            abs_d = abs_d[ok_anchor]

            # RANSAC linear fit (a, b)
            best_inliers = 0
            best_a = best_b = None
            rng = np.random.default_rng(seed=int(nid))
            for _ in range(ransac_iters):
                idx = rng.choice(len(rel_at), size=2, replace=False)
                r1, r2 = rel_at[idx]
                a1, a2 = abs_d[idx]
                if abs(r1 - r2) < 1e-6:
                    continue
                a = (a1 - a2) / (r1 - r2)
                b = a1 - a * r1
                pred = a * rel_at + b
                err = np.abs(pred - abs_d)
                inlier = err < (np.abs(abs_d) * ransac_inlier_thresh_rel + 0.05)
                if int(inlier.sum()) > best_inliers:
                    best_inliers = int(inlier.sum())
                    best_a, best_b = a, b
            if best_a is None or best_inliers < min_anchors // 2:
                n_frames_no_anchor += 1
                continue
            # least squares re-fit on inliers
            pred = best_a * rel_at + best_b
            err = np.abs(pred - abs_d)
            inlier_mask = err < (np.abs(abs_d) * ransac_inlier_thresh_rel + 0.05)
            if int(inlier_mask.sum()) >= 2:
                A = np.stack([rel_at[inlier_mask], np.ones(inlier_mask.sum())], axis=1)
                sol, *_ = np.linalg.lstsq(A, abs_d[inlier_mask], rcond=None)
                best_a, best_b = float(sol[0]), float(sol[1])
            n_frames_aligned += 1

            # NaN keypoint 채우기
            nan_idx = np.flatnonzero(~valid)
            if len(nan_idx) == 0:
                continue
            n_attempted += len(nan_idx)
            uv_nan = kps[nan_idx]
            uv_nan_int = np.round(uv_nan).astype(int)
            uv_nan_int[:, 0] = np.clip(uv_nan_int[:, 0], 0, rel_depth.shape[1] - 1)
            uv_nan_int[:, 1] = np.clip(uv_nan_int[:, 1], 0, rel_depth.shape[0] - 1)
            rel_nan = rel_depth[uv_nan_int[:, 1], uv_nan_int[:, 0]]
            abs_nan = best_a * rel_nan + best_b

            ok_d = (abs_nan > depth_min_m) & (abs_nan < depth_max_m) & np.isfinite(abs_nan)
            if not ok_d.any():
                continue

            u = uv_nan[:, 0]
            v = uv_nan[:, 1]
            x_opt = (u - K[0, 2]) * abs_nan / K[0, 0]
            y_opt = (v - K[1, 2]) * abs_nan / K[1, 1]
            z_opt = abs_nan
            pts_opt = np.stack([x_opt, y_opt, z_opt], axis=1)

            # optical → world: X_w = R_wo^T (X_opt - t_wo)
            R_ow = R_wo.T
            pts_world = (R_ow @ (pts_opt - t_wo).T).T

            for k, w_pt, ok in zip(nan_idx, pts_world, ok_d):
                if ok and np.all(np.isfinite(w_pt)):
                    self.keyframe_world3d[nid][k] = w_pt.astype(np.float32)
                    n_filled += 1

        logger.info(
            f"[ML depth] frames_aligned={n_frames_aligned} "
            f"frames_skipped(no_anchor)={n_frames_no_anchor}"
        )
        return n_filled, n_attempted

    def top_k_candidates(
        self, q_desc_mean: torch.Tensor, k: int = TOP_K
    ) -> List[int]:
        """Return top-K node IDs by cosine similarity of mean descriptors."""
        if self.global_descs is None or not self.node_ids:
            return self.node_ids[:k]
        sims = torch.cosine_similarity(
            q_desc_mean.unsqueeze(0), self.global_descs
        )
        k = min(k, len(self.node_ids))
        indices = sims.topk(k).indices.tolist()
        return [self.node_ids[i] for i in indices]


# ---------------------------------------------------------------------------
# Singleton manager
# ---------------------------------------------------------------------------

class SuperPointMapManager:
    """Singleton LRU cache for SuperPointLoadedMap instances."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._maps: OrderedDict[str, SuperPointLoadedMap] = OrderedDict()
                    inst._device = torch.device(
                        'cuda' if torch.cuda.is_available() else 'cpu'
                    )
                    # map_id 별 indexing 동시 진행 방지 lock
                    inst._build_locks: Dict[str, threading.Lock] = {}
                    inst._build_locks_guard = threading.Lock()
                    cls._instance = inst
        return cls._instance

    def _get_build_lock(self, map_id: str) -> threading.Lock:
        with self._build_locks_guard:
            if map_id not in self._build_locks:
                self._build_locks[map_id] = threading.Lock()
            return self._build_locks[map_id]

    @property
    def device(self) -> torch.device:
        return self._device

    def get_or_load(
        self, map_id: str, db_path: Optional[str] = None
    ) -> SuperPointLoadedMap:
        # active scan 이 바뀌었거나 db 파일 mtime 이 바뀐 경우 캐시 invalidate
        if map_id in self._maps:
            cached = self._maps[map_id]
            try:
                cached_mtime = getattr(cached, "_db_mtime", None)
                if db_path is not None and str(cached.db_path) != str(db_path):
                    logger.info(
                        f"[SuperPoint] map '{map_id}' db_path changed "
                        f"({cached.db_path} → {db_path}) — invalidating cache"
                    )
                    del self._maps[map_id]
                else:
                    import os
                    current_mtime = os.path.getmtime(cached.db_path) if cached.db_path else None
                    if cached_mtime is not None and current_mtime != cached_mtime:
                        logger.info(
                            f"[SuperPoint] map '{map_id}' db file modified "
                            f"(mtime {cached_mtime} → {current_mtime}) — invalidating cache"
                        )
                        del self._maps[map_id]
                    else:
                        self._maps.move_to_end(map_id)
                        return cached
            except Exception as exc:
                logger.warning(f"[SuperPoint] cache validation failed: {exc} — invalidating")
                self._maps.pop(map_id, None)

        if db_path is None:
            raise ValueError(f"Map '{map_id}' not cached and no db_path provided")

        # map_id 별 lock — 동시 요청 시 첫 thread 만 indexing, 나머지는 대기 후 cache hit
        build_lock = self._get_build_lock(map_id)
        with build_lock:
            # double-check: lock 대기 중 다른 thread 가 이미 indexing 했을 수 있음
            if map_id in self._maps:
                cached = self._maps[map_id]
                if str(cached.db_path) == str(db_path):
                    self._maps.move_to_end(map_id)
                    return cached

            m = SuperPointLoadedMap(map_id, db_path, self._device)
            try:
                import os
                m._db_mtime = os.path.getmtime(db_path)
            except Exception:
                m._db_mtime = None
            self._maps[map_id] = m
            while len(self._maps) > MAX_CACHED_MAPS:
                evicted, _ = self._maps.popitem(last=False)
                logger.info(f"[SuperPoint] Evicted map '{evicted}' from cache")
            return m
