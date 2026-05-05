"""SuperPoint + LightGlue localization engine.

Map building (process / save_map / load_map) is delegated to RTABMapEngine
unchanged. Only localize() is re-implemented here.
"""
import asyncio
import functools
import io
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from slam_interface.base import SLAMEngineBase
from utils import logger


def _to_gray_float(img_bytes: bytes) -> Optional[np.ndarray]:
    """Decode image bytes → grayscale float [0,1], applying EXIF orientation."""
    try:
        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes)))
        if pil.mode != 'L':
            pil = pil.convert('L')
        return np.array(pil, dtype=np.float32) / 255.0
    except Exception:
        return None


def _rotation_to_quat(R: np.ndarray):
    """Convert 3x3 rotation matrix to (qx, qy, qz, qw), qw >= 0."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / (trace + 1.0) ** 0.5
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * (1 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * (1 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * (1 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    if qw < 0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    return float(qx), float(qy), float(qz), float(qw)


class SuperPointEngine(SLAMEngineBase):
    """Localization engine using SuperPoint + LightGlue."""

    def __init__(self):
        from lightglue import LightGlue, SuperPoint

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._extractor = SuperPoint(max_num_keypoints=1024).eval().to(self._device)
        self._matcher = LightGlue(features='superpoint').eval().to(self._device)
        logger.info(f"[SuperPoint] Engine ready on {self._device}")

    # --- map building: delegate unchanged to RTABMapEngine ---

    async def process(self, *args, **kwargs):
        from slam_engines.rtabmap.engine import RTABMapEngine
        return await RTABMapEngine().process(*args, **kwargs)

    def save_map(self, *args, **kwargs):
        from slam_engines.rtabmap.engine import RTABMapEngine
        return RTABMapEngine().save_map(*args, **kwargs)

    def load_map(self, *args, **kwargs):
        from slam_engines.rtabmap.engine import RTABMapEngine
        return RTABMapEngine().load_map(*args, **kwargs)

    def scale_intrinsics(self, original: dict, new_width: int, new_height: int) -> dict:
        from slam_engines.rtabmap.engine import RTABMapEngine
        return RTABMapEngine().scale_intrinsics(original, new_width, new_height)

    def extract_intrinsics_from_db(self, db_path: str) -> dict:
        from slam_engines.rtabmap.engine import RTABMapEngine
        return RTABMapEngine().extract_intrinsics_from_db(db_path)

    # --- feature extraction & matching ---

    def _extract(self, gray: np.ndarray) -> dict:
        tensor = torch.from_numpy(gray)[None, None].to(self._device)
        with torch.no_grad():
            return self._extractor.extract(tensor)

    def _match(self, feats0: dict, feats1: dict) -> np.ndarray:
        from lightglue.utils import rbd
        f0 = {k: v.to(self._device) for k, v in feats0.items()}
        f1 = {k: v.to(self._device) for k, v in feats1.items()}
        with torch.no_grad():
            result = self._matcher({'image0': f0, 'image1': f1})
        return rbd(result)['matches'].cpu().numpy()  # (M, 2)

    # --- localization core (runs in thread executor) ---

    def _localize_sync(
        self,
        map_id: str,
        images: List[bytes],
        intrinsics: Optional[Dict],
        db_path: Optional[str],
    ) -> dict:
        from .map_manager import SuperPointMapManager

        if intrinsics is None:
            raise ValueError("intrinsics required for SuperPoint localization")

        K = np.array([
            [intrinsics['fx'], 0,              intrinsics['cx']],
            [0,               intrinsics['fy'], intrinsics['cy']],
            [0,               0,               1              ],
        ], dtype=np.float64)

        mgr = SuperPointMapManager()
        loaded = mgr.get_or_load(map_id, db_path)

        if not loaded.node_ids:
            raise ValueError("No keyframes with stored images in this map")

        # RTABMap camera-frame → OpenCV optical-frame conversion matrix
        C = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=np.float64)

        best: Optional[dict] = None

        for img_idx, img_bytes in enumerate(images):
            gray = _to_gray_float(img_bytes)
            if gray is None:
                continue

            q_feats = self._extract(gray)
            q_kps = q_feats['keypoints'][0].cpu().numpy()  # (N, 2)

            from .global_descriptor import GlobalDescExtractor
            gray_uint8 = (gray * 255).clip(0, 255).astype(np.uint8)
            q_global = GlobalDescExtractor(self._device).extract(gray_uint8)  # (384,)

            candidates = loaded.top_k_candidates(q_global)

            for node_id in candidates:
                db_feats = loaded.keyframe_feats[node_id]
                world3d = loaded.keyframe_world3d[node_id]          # (M, 3)

                matches = self._match(
                    {k: v.cpu() for k, v in q_feats.items()},
                    db_feats,
                )  # (P, 2)

                if len(matches) < 4:
                    continue

                pts_2d, pts_3d = [], []
                for qi, di in matches:
                    w = world3d[di]
                    if np.any(np.isnan(w)):
                        continue
                    pts_2d.append(q_kps[qi])
                    pts_3d.append(w)

                if len(pts_3d) < 4:
                    continue

                pts_2d = np.array(pts_2d, dtype=np.float64)
                pts_3d = np.array(pts_3d, dtype=np.float64)

                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    pts_3d, pts_2d, K, None,
                    flags=cv2.SOLVEPNP_EPNP,
                    reprojectionError=8.0,
                    confidence=0.99,
                    iterationsCount=1000,
                )

                if not ok or inliers is None or len(inliers) < 4:
                    continue

                n_in = len(inliers)

                # Convert PnP result to RTABMap world pose
                # (same convention as RTABMapEngine / map_manager)
                R_w2c, _ = cv2.Rodrigues(rvec)
                R_cw = R_w2c.T @ C
                t_cw = (-R_w2c.T @ tvec).flatten()

                qx, qy, qz, qw = _rotation_to_quat(R_cw)
                confidence = min(0.99, max(0.01, n_in / len(pts_3d)))

                candidate = {
                    'num_matches': n_in,
                    'confidence': confidence,
                    'matched_image_index': img_idx,
                    'pose': {
                        'x': float(t_cw[0]), 'y': float(t_cw[1]), 'z': float(t_cw[2]),
                        'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw,
                    },
                }
                if best is None or n_in > best['num_matches']:
                    best = candidate

        if best is None:
            raise ValueError("SuperPoint+LightGlue: insufficient matches")

        logger.info(
            f"[SuperPoint] map={map_id} inliers={best['num_matches']} "
            f"confidence={best['confidence']:.3f}"
        )
        return {**best, 'map_id': map_id, 'method': 'SuperPoint+LightGlue'}

    async def localize(
        self,
        map_id: str,
        images: List[bytes],
        intrinsics: Optional[Dict] = None,
        initial_pose: Optional[Dict] = None,
        db_path: Optional[str] = None,
        **kwargs,
    ) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(
                self._localize_sync, map_id, images, intrinsics, db_path
            ),
        )
