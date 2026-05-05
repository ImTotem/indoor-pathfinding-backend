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
                    cls._instance = inst
        return cls._instance

    @property
    def device(self) -> torch.device:
        return self._device

    def get_or_load(
        self, map_id: str, db_path: Optional[str] = None
    ) -> SuperPointLoadedMap:
        if map_id in self._maps:
            self._maps.move_to_end(map_id)
            return self._maps[map_id]
        if db_path is None:
            raise ValueError(f"Map '{map_id}' not cached and no db_path provided")
        m = SuperPointLoadedMap(map_id, db_path, self._device)
        self._maps[map_id] = m
        while len(self._maps) > MAX_CACHED_MAPS:
            evicted, _ = self._maps.popitem(last=False)
            logger.info(f"[SuperPoint] Evicted map '{evicted}' from cache")
        return m
