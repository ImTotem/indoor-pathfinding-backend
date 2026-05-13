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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from utils import logger

MAX_CACHED_MAPS = 5
TOP_K = 30


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


def _to_homogeneous_pose(transform_3x4: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :] = transform_3x4
    return pose


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

    def __init__(
        self,
        map_id: str,
        db_path: str,
        device: torch.device,
        *,
        skip_build: bool = False,
    ):
        self.map_id = map_id
        self.db_path = db_path
        self.device = device

        self.node_ids: List[int] = []
        # CPU tensors: {'keypoints': (1,N,2), 'descriptors': (1,N,256), 'image_size': (1,2)}
        self.keyframe_feats: Dict[int, dict] = {}
        # (N, 3) world 3D per keyframe keypoint; NaN where unavailable
        self.keyframe_world3d: Dict[int, np.ndarray] = {}
        # RTAB-Map Node.pose, base_link -> world, homogeneous 4x4.
        self.node_poses: Dict[int, np.ndarray] = {}
        # (K, 256) mean descriptors for global retrieval
        self.global_descs: Optional[torch.Tensor] = None
        # DB calibration — base→optical (3x3) frame 변환 매트릭스. localize 시 응답 R 변환에 사용.
        self.optical_to_base: Optional[np.ndarray] = None
        self.base_to_optical: Optional[np.ndarray] = None

        if not skip_build:
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
            self.node_poses = {
                nid: _to_homogeneous_pose(T34) for nid, T34 in transforms.items()
            }
            world_feats = _load_world_features(conn, transforms)

            # DB calibration 의 base→optical 매트릭스 캐싱 (localize 시 응답 R 변환에 사용).
            # device 별로 sensor mounting / orientation 다를 수 있어 하드코딩 금지.
            if all_ids:
                calib_row = conn.execute(
                    "SELECT calibration FROM Data WHERE id=? LIMIT 1", (all_ids[0],)
                ).fetchone()
                if calib_row and calib_row[0]:
                    try:
                        _K, lt = _parse_calibration_K_and_local(bytes(calib_row[0]))
                        self.optical_to_base = lt[:3, :3].astype(np.float64)
                        self.base_to_optical = self.optical_to_base
                    except Exception as exc:
                        logger.warning(f"[SuperPoint] optical_to_base parse failed: {exc}")

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
        valid_ratios = [
            float(np.count_nonzero(~np.isnan(v).any(axis=1)) / max(len(v), 1))
            for v in self.keyframe_world3d.values()
        ]
        median_valid_ratio = float(np.median(valid_ratios)) if valid_ratios else 0.0
        logger.info(
            f"[SuperPoint] Map '{self.map_id}' indexed in {time.time()-t0:.1f}s: "
            f"{len(self.node_ids)} frames, {n_with_3d} with 3D coverage "
            f"(median valid kp ratio={median_valid_ratio:.3f})"
        )

        # Fallback: RTAB-Map Feature 에 depth 가 없어 3D coverage 가 0 인 경우
        # 1) Data 테이블의 depth blob (iOS LiDAR) 우선 사용 — sensor-level 정확도, 즉시.
        # 2) depth blob 없으면 multi-view triangulation 으로 fallback.
        if (n_with_3d == 0 or median_valid_ratio < 0.45) and len(self.node_ids) >= 1:
            t1 = time.time()
            n_filled = self._fill_3d_from_depth_blob()
            if n_filled > 0:
                n_with_3d = sum(
                    1 for v in self.keyframe_world3d.values()
                    if not np.all(np.isnan(v))
                )
                valid_ratios = [
                    float(np.count_nonzero(~np.isnan(v).any(axis=1)) / max(len(v), 1))
                    for v in self.keyframe_world3d.values()
                ]
                median_valid_ratio = float(np.median(valid_ratios)) if valid_ratios else 0.0
                logger.info(
                    f"[SuperPoint] RGBD depth lookup done in {time.time()-t1:.1f}s: "
                    f"{n_with_3d}/{len(self.node_ids)} frames with 3D coverage "
                    f"(median valid kp ratio={median_valid_ratio:.3f})"
                )
            elif len(self.node_ids) >= 2:
                logger.info(
                    "[SuperPoint] No depth blob — running multi-view triangulation"
                )
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

    def _fill_3d_from_depth_blob(
        self,
        min_depth_m: float = 0.2,
        max_depth_m: float = 15.0,
    ) -> int:
        """Data.depth blob (iOS LiDAR) 사용해서 SuperPoint kp 3D world 좌표 직접 lookup.

        iOS RTABMap 의 depth 는 PNG 으로 압축된 (H, W, 4) uint8 = float32 단일 채널 (m 단위).
        SuperPoint kp 좌표 (RGB image frame) → depth image frame 으로 scale → depth lookup.
        (u, v, d) → camera optical 3D → world (Node.pose @ base→optical inverse).

        Returns: 3D 채워진 keyframe 개수.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            transforms = _parse_node_transforms(conn)
            calib_row = conn.execute(
                "SELECT calibration FROM Data WHERE id=? LIMIT 1",
                (self.node_ids[0],),
            ).fetchone()
            if not calib_row or not calib_row[0]:
                return 0
            K, local_transform = _parse_calibration_K_and_local(bytes(calib_row[0]))
            M_oc_to_base = local_transform[:3, :3]  # optical → base/local
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            n_filled = 0
            for nid in self.node_ids:
                if nid not in transforms:
                    continue
                row = conn.execute(
                    "SELECT depth FROM Data WHERE id=?", (nid,)
                ).fetchone()
                if not row or not row[0]:
                    continue

                # decode iOS depth: PNG → (H, W, 4) uint8 → reinterpret float32 (H, W)
                arr = np.frombuffer(bytes(row[0]), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                if img is None or img.ndim != 3 or img.shape[2] != 4:
                    continue
                depth = img.view(np.float32).reshape(img.shape[0], img.shape[1])
                depth_h, depth_w = depth.shape

                # SuperPoint kp 의 RGB image dim (extract 시 image_size 저장됨)
                kp = self.keyframe_feats[nid]['keypoints'][0].cpu().numpy()  # (N, 2) in RGB frame
                rgb_size = self.keyframe_feats[nid]['image_size'][0].cpu().numpy()  # (W, H)
                rgb_w, rgb_h = float(rgb_size[0]), float(rgb_size[1])

                # depth scale (RGB → depth coord)
                sx = depth_w / rgb_w
                sy = depth_h / rgb_h

                # Node.pose: base→world (3x4 row-major)
                T_b_w = transforms[nid]
                R_b_w = T_b_w[:, :3]
                t_b_w = T_b_w[:, 3]

                w3d = np.full((len(kp), 3), float('nan'), dtype=np.float32)
                for i, (u_rgb, v_rgb) in enumerate(kp):
                    u_d = int(round(u_rgb * sx))
                    v_d = int(round(v_rgb * sy))
                    if not (0 <= u_d < depth_w and 0 <= v_d < depth_h):
                        continue
                    d = float(depth[v_d, u_d])
                    if d < min_depth_m or d > max_depth_m or not np.isfinite(d):
                        continue
                    # (u, v, d) → 3D in optical camera frame
                    x_oc = (u_rgb - cx) / fx * d
                    y_oc = (v_rgb - cy) / fy * d
                    z_oc = d
                    p_oc = np.array([x_oc, y_oc, z_oc], dtype=np.float64)
                    # optical → base → world
                    p_base = M_oc_to_base @ p_oc
                    p_world = R_b_w @ p_base + t_b_w
                    w3d[i] = p_world.astype(np.float32)

                self.keyframe_world3d[nid] = w3d
                if not np.all(np.isnan(w3d)):
                    n_filled += 1
            return n_filled
        finally:
            conn.close()

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
            C = local_transform[:3, :3]  # optical → base/local (3x3)
            base_to_optical = C.T
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
                R_wo_i = base_to_optical @ T_w_bi[:3, :3]
                t_wo_i = base_to_optical @ T_w_bi[:3, 3]
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
                    R_wo_j = base_to_optical @ T_w_bj[:3, :3]
                    t_wo_j = base_to_optical @ T_w_bj[:3, 3]
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
            base_to_optical = C.T
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
            R_wo = base_to_optical @ T_w_b[:3, :3]
            t_wo = base_to_optical @ T_w_b[:3, 3]

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

            # Disk cache hit before falling back to a 26s rebuild from
            # rtabmap.db. Survives uvicorn --reload and process restart.
            disk_loaded = _try_load_disk_cache(map_id, db_path, self._device)
            if disk_loaded is not None:
                self._maps[map_id] = disk_loaded
                while len(self._maps) > MAX_CACHED_MAPS:
                    evicted, _ = self._maps.popitem(last=False)
                    logger.info(f"[SuperPoint] Evicted map '{evicted}' from cache")
                return disk_loaded

            m = SuperPointLoadedMap(map_id, db_path, self._device)
            try:
                import os
                m._db_mtime = os.path.getmtime(db_path)
            except Exception:
                m._db_mtime = None
            # Persist to disk so the next process / reload can mmap it.
            try:
                _save_disk_cache(m)
            except Exception as exc:
                logger.warning(
                    f"[SuperPoint] disk cache save failed map='{map_id}' err={exc}"
                )
            self._maps[map_id] = m
            while len(self._maps) > MAX_CACHED_MAPS:
                evicted, _ = self._maps.popitem(last=False)
                logger.info(f"[SuperPoint] Evicted map '{evicted}' from cache")
            return m


# ─── persistent disk cache (safetensors + sidecar JSON) ──────────────────────
# Layout per floor under config.settings.SUPERPOINT_CACHE_DIR:
#   <map_id>.safetensors  — tensor bundle (stacked across all keyframes)
#   <map_id>.json         — schema_version, source db mtime, node_poses,
#                            node_offsets, calibration matrices, etc.
# Invalidation: sidecar.source_db_mtime != current rtabmap.db mtime → rebuild.

_CACHE_SCHEMA_VERSION = 1


def _floor_cache_paths(map_id: str) -> tuple[Path, Path]:
    from config.settings import settings as _slam_settings
    cache_dir = Path(_slam_settings.SUPERPOINT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (
        cache_dir / f"{map_id}.safetensors",
        cache_dir / f"{map_id}.json",
    )


def _try_load_disk_cache(
    map_id: str, db_path: str, device: torch.device,
) -> Optional["SuperPointLoadedMap"]:
    """Return a fully-populated SuperPointLoadedMap from disk cache if the
    sidecar's recorded source_db_mtime matches the current rtabmap.db, else
    None (caller falls back to rebuild)."""
    import json
    import os
    import time

    bundle_path, sidecar_path = _floor_cache_paths(map_id)
    if not bundle_path.exists() or not sidecar_path.exists():
        return None
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"[SuperPoint] sidecar parse failed map='{map_id}': {exc}")
        return None
    if sidecar.get("schema_version") != _CACHE_SCHEMA_VERSION:
        logger.info(
            f"[SuperPoint] disk cache schema mismatch map='{map_id}' — rebuild"
        )
        return None
    if sidecar.get("source_db_path") != str(db_path):
        return None
    try:
        current_mtime = os.path.getmtime(db_path)
    except OSError:
        return None
    if abs(sidecar.get("source_db_mtime", -1.0) - current_mtime) > 1e-3:
        logger.info(
            f"[SuperPoint] disk cache mtime mismatch map='{map_id}' — rebuild"
        )
        return None

    t0 = time.time()
    from safetensors.torch import load_file as _load_safetensors

    tensors = _load_safetensors(str(bundle_path), device="cpu")

    m = SuperPointLoadedMap(map_id, str(db_path), device, skip_build=True)
    try:
        m._db_mtime = current_mtime  # type: ignore[attr-defined]
    except Exception:
        pass

    node_ids = [int(x) for x in sidecar["node_ids"]]
    offsets = [int(x) for x in sidecar["node_offsets"]]
    image_sizes = tensors["image_sizes"]  # (K, 2)
    kpts_all = tensors["keypoints"]       # (total_N, 2)
    desc_all = tensors["descriptors"].to(torch.float32)  # fp16 → fp32
    world3d_all = tensors["keypoints_world3d"].numpy()    # (total_N, 3)

    m.node_ids = list(node_ids)
    for idx, nid in enumerate(node_ids):
        a, b = offsets[idx], offsets[idx + 1]
        m.keyframe_feats[nid] = {
            "keypoints": kpts_all[a:b].unsqueeze(0),       # (1, N, 2)
            "descriptors": desc_all[a:b].unsqueeze(0),      # (1, N, 256)
            "image_size": image_sizes[idx : idx + 1],       # (1, 2)
        }
        m.keyframe_world3d[nid] = world3d_all[a:b].copy()

    if "global_descs" in tensors:
        m.global_descs = tensors["global_descs"]

    np_node_poses = sidecar.get("node_poses", {})
    m.node_poses = {
        int(k): np.array(v, dtype=np.float64) for k, v in np_node_poses.items()
    }
    if "optical_to_base" in tensors:
        m.optical_to_base = tensors["optical_to_base"].numpy().astype(np.float64)
    if "base_to_optical" in tensors:
        m.base_to_optical = tensors["base_to_optical"].numpy().astype(np.float64)

    logger.info(
        f"[SuperPoint] disk cache loaded map='{map_id}' frames={len(node_ids)} "
        f"in {time.time() - t0:.2f}s"
    )
    return m


def _save_disk_cache(m: "SuperPointLoadedMap") -> None:
    """Dump SuperPointLoadedMap to safetensors + sidecar JSON. Atomic via
    tmp + rename so concurrent loaders never see a partial file.
    """
    import json
    import os
    import tempfile
    import time

    if not m.node_ids:
        return
    bundle_path, sidecar_path = _floor_cache_paths(m.map_id)

    t0 = time.time()
    # Stack per-keyframe tensors with offset metadata.
    kpts_list, desc_list, world3d_list, image_size_list, offsets = [], [], [], [], [0]
    for nid in m.node_ids:
        feats = m.keyframe_feats.get(nid)
        if feats is None:
            continue
        kp = feats["keypoints"][0].detach().cpu()       # (N, 2)
        de = feats["descriptors"][0].detach().cpu().to(torch.float16)  # (N, 256) fp16
        sz = feats["image_size"][0].detach().cpu()      # (2,)
        w3 = m.keyframe_world3d.get(nid)
        if w3 is None:
            w3 = np.full((kp.shape[0], 3), float("nan"), dtype=np.float32)
        kpts_list.append(kp)
        desc_list.append(de)
        world3d_list.append(torch.from_numpy(w3.astype(np.float32)))
        image_size_list.append(sz)
        offsets.append(offsets[-1] + kp.shape[0])

    tensors = {
        "keypoints": torch.cat(kpts_list, dim=0),                # (total_N, 2) float32
        "descriptors": torch.cat(desc_list, dim=0),               # (total_N, 256) fp16
        "keypoints_world3d": torch.cat(world3d_list, dim=0),      # (total_N, 3) float32
        "image_sizes": torch.stack(image_size_list, dim=0),       # (K, 2) float32
    }
    if m.global_descs is not None:
        tensors["global_descs"] = m.global_descs.detach().cpu()
    if m.optical_to_base is not None:
        tensors["optical_to_base"] = torch.from_numpy(
            np.asarray(m.optical_to_base, dtype=np.float32)
        )
    if m.base_to_optical is not None:
        tensors["base_to_optical"] = torch.from_numpy(
            np.asarray(m.base_to_optical, dtype=np.float32)
        )

    sidecar = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "map_id": m.map_id,
        "source_db_path": str(m.db_path),
        "source_db_mtime": (
            os.path.getmtime(m.db_path) if Path(m.db_path).exists() else None
        ),
        "node_ids": list(m.node_ids),
        "node_offsets": offsets,
        "node_poses": {
            str(k): v.tolist() for k, v in m.node_poses.items()
        },
    }

    from safetensors.torch import save_file as _save_safetensors

    bundle_tmp = tempfile.NamedTemporaryFile(
        prefix=f".{m.map_id}.", suffix=".safetensors.tmp",
        dir=str(bundle_path.parent), delete=False,
    )
    bundle_tmp.close()
    try:
        _save_safetensors(tensors, bundle_tmp.name)
        os.replace(bundle_tmp.name, bundle_path)
    except Exception:
        Path(bundle_tmp.name).unlink(missing_ok=True)
        raise

    sidecar_tmp = sidecar_path.with_suffix(".json.tmp")
    try:
        sidecar_tmp.write_text(json.dumps(sidecar), encoding="utf-8")
        os.replace(sidecar_tmp, sidecar_path)
    except Exception:
        sidecar_tmp.unlink(missing_ok=True)
        raise

    sz_mb = bundle_path.stat().st_size / (1 << 20)
    logger.info(
        f"[SuperPoint] disk cache saved map='{m.map_id}' frames={len(m.node_ids)} "
        f"size={sz_mb:.1f}MB in {time.time() - t0:.2f}s"
    )
