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
