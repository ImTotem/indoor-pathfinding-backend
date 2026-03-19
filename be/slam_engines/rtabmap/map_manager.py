# slam_engines/rtabmap/map_manager.py
"""In-memory map manager for fast relocalization.

Eliminates cold-start overhead by keeping map descriptors and poses
loaded in memory. Supports GFTT/BRIEF, SIFT, ORB with easy switching.
"""

import sqlite3
import struct
import threading
import time
import logging
from collections import OrderedDict
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
import io

from config.settings import settings

logger = logging.getLogger(__name__)


def _decode_image_with_exif(img_bytes: bytes) -> Optional[np.ndarray]:
    """Decode image bytes to grayscale numpy array, applying EXIF orientation.

    cv2.imdecode ignores EXIF rotation tags, so portrait photos taken on phones
    appear rotated. This function uses PIL to apply EXIF orientation first,
    then converts to OpenCV grayscale format.

    Pipeline: RGB → Zero-DCE enhance (if dark) → grayscale → CLAHE
    """
    try:
        pil_img = Image.open(io.BytesIO(img_bytes))
        from PIL import ImageOps
        pil_img = ImageOps.exif_transpose(pil_img)
        rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    except Exception:
        bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Zero-DCE low-light enhancement (skipped if image is bright enough)
    from .low_light_enhancer import LowLightEnhancer
    rgb = LowLightEnhancer().enhance(rgb)

    # Convert to grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # CLAHE: normalize contrast for robust feature detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# RTABMap detector strategies that use binary descriptors
_FLOAT_DESCRIPTOR_STRATEGIES = {0, 1, 9}  # SURF, SIFT, KAZE


def _parse_pose_blob(blob: bytes) -> Optional[List[float]]:
    """Parse 48-byte pose BLOB (3x4 transform matrix) to [x,y,z,qx,qy,qz,qw]."""
    if not blob or len(blob) != 48:
        return None

    values = struct.unpack('<12f', blob)
    if all(v == 0.0 for v in values):
        return None

    r00, r01, r02, tx = values[0:4]
    r10, r11, r12, ty = values[4:8]
    r20, r21, r22, tz = values[8:12]

    trace = r00 + r11 + r22
    if trace > 0:
        s = 0.5 / (trace + 1.0) ** 0.5
        qw = 0.25 / s
        qx = (r21 - r12) * s
        qy = (r02 - r20) * s
        qz = (r10 - r01) * s
    elif r00 > r11 and r00 > r22:
        s = 2.0 * (1.0 + r00 - r11 - r22) ** 0.5
        qw = (r21 - r12) / s
        qx = 0.25 * s
        qy = (r01 + r10) / s
        qz = (r02 + r20) / s
    elif r11 > r22:
        s = 2.0 * (1.0 + r11 - r00 - r22) ** 0.5
        qw = (r02 - r20) / s
        qx = (r01 + r10) / s
        qy = 0.25 * s
        qz = (r12 + r21) / s
    else:
        s = 2.0 * (1.0 + r22 - r00 - r11) ** 0.5
        qw = (r10 - r01) / s
        qx = (r02 + r20) / s
        qy = (r12 + r21) / s
        qz = 0.25 * s

    return [tx, ty, tz, qx, qy, qz, qw]


class LoadedMap:
    """A map loaded into memory for fast relocalization."""

    def __init__(self, map_id: str, db_path: str):
        self.map_id = map_id
        self.db_path = db_path
        self.node_poses: Dict[int, List[float]] = {}
        # Per-node pose as 3x4 matrix for transforming local 3D points to world
        self.node_transforms: Dict[int, np.ndarray] = {}
        self.all_descriptors: Optional[np.ndarray] = None
        self.descriptor_node_ids: Optional[np.ndarray] = None
        # 2D keypoint positions (pos_x, pos_y) per descriptor
        self.descriptor_pos_2d: Optional[np.ndarray] = None
        # 3D local coordinates (depth_x, depth_y, depth_z) per descriptor; None if unavailable
        self.descriptor_pos_3d: Optional[np.ndarray] = None
        self.detector_strategy: int = 6
        self.brief_bytes: int = 64
        self.max_features: int = 1000
        self.min_inliers: int = 3
        self.is_binary: bool = True

        self._load()

    def _load(self):
        t0 = time.time()
        conn = sqlite3.connect(self.db_path)
        try:
            self._load_params(conn)
            self._load_poses(conn)
            self._load_descriptors(conn)
        finally:
            conn.close()

        n_descs = len(self.all_descriptors) if self.all_descriptors is not None else 0
        n_3d = int(np.sum(~np.isnan(self.descriptor_pos_3d[:, 0]))) if self.descriptor_pos_3d is not None else 0
        logger.info(
            f"Map '{self.map_id}' loaded in {time.time() - t0:.2f}s: "
            f"{len(self.node_poses)} nodes, {n_descs} descriptors ({n_3d} with 3D), "
            f"strategy={self.detector_strategy}"
        )

    def _load_params(self, conn: sqlite3.Connection):
        params: Dict[str, str] = {}
        for table in ("Info", "Statistics"):
            try:
                rows = conn.execute(
                    f"SELECT key, value FROM {table} "
                    f"WHERE key LIKE 'Kp/%' OR key LIKE 'Vis/%' OR key LIKE 'BRIEF/%'"
                ).fetchall()
                for k, v in rows:
                    params[k] = v
            except sqlite3.OperationalError:
                continue

        self.detector_strategy = int(params.get("Kp/DetectorStrategy", "6"))
        self.brief_bytes = int(params.get("BRIEF/Bytes", "64"))
        self.max_features = int(params.get("Kp/MaxFeatures", "1000"))
        self.min_inliers = int(params.get("Vis/MinInliers", "3"))
        self.is_binary = self.detector_strategy not in _FLOAT_DESCRIPTOR_STRATEGIES

    def _load_poses(self, conn: sqlite3.Connection):
        rows = conn.execute(
            "SELECT id, pose FROM Node WHERE pose IS NOT NULL"
        ).fetchall()
        for node_id, pose_blob in rows:
            pose = _parse_pose_blob(pose_blob)
            if pose:
                self.node_poses[node_id] = pose
                # Store raw 3x4 transform matrix for local→world conversion
                values = struct.unpack('<12f', pose_blob)
                self.node_transforms[node_id] = np.array(values, dtype=np.float64).reshape(3, 4)

    def _load_descriptors(self, conn: sqlite3.Connection):
        rows = conn.execute(
            "SELECT node_id, descriptor_size, descriptor, pos_x, pos_y, "
            "depth_x, depth_y, depth_z FROM Feature "
            "WHERE descriptor IS NOT NULL AND descriptor_size > 0"
        ).fetchall()

        if not rows:
            logger.warning(f"Map '{self.map_id}': Feature table is empty")
            return

        dtype = np.uint8 if self.is_binary else np.float32
        desc_size = None
        descs = []
        node_ids = []
        pos_2d_list = []
        pos_3d_list = []

        for node_id, d_size, d_blob, pos_x, pos_y, depth_x, depth_y, depth_z in rows:
            if node_id not in self.node_poses:
                continue
            if not d_blob:
                continue

            arr = np.frombuffer(d_blob, dtype=dtype)
            if len(arr) == 0:
                continue

            # Use first valid descriptor to determine size
            if desc_size is None:
                desc_size = d_size

            if len(arr) >= desc_size:
                descs.append(arr[:desc_size])
                node_ids.append(node_id)
                pos_2d_list.append([pos_x, pos_y])
                # depth_x/y/z are local 3D coords; may be NULL
                if depth_x is not None and depth_y is not None and depth_z is not None:
                    pos_3d_list.append([depth_x, depth_y, depth_z])
                else:
                    pos_3d_list.append([float('nan'), float('nan'), float('nan')])

        if descs:
            self.all_descriptors = np.vstack(
                [d.reshape(1, -1) for d in descs]
            ).astype(dtype)
            self.descriptor_node_ids = np.array(node_ids, dtype=np.int32)
            self.descriptor_pos_2d = np.array(pos_2d_list, dtype=np.float64)
            self.descriptor_pos_3d = np.array(pos_3d_list, dtype=np.float64)

    @property
    def norm_type(self) -> int:
        return cv2.NORM_HAMMING if self.is_binary else cv2.NORM_L2


def _create_detector(strategy: int, max_features: int = 1000, brief_bytes: int = 64):
    """Create (detector, descriptor_extractor) pair for given strategy.

    Strategies:
        0: SURF  |  1: SIFT  |  2: ORB  |  6: GFTT/BRIEF  |  8: GFTT/ORB
    """
    if strategy == 1:  # SIFT
        det = cv2.SIFT_create(nfeatures=max_features)
        return det, det
    elif strategy == 2:  # ORB
        det = cv2.ORB_create(nfeatures=max_features)
        return det, det
    elif strategy == 6:  # GFTT/BRIEF
        det = cv2.GFTTDetector_create(maxCorners=max_features)
        desc = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=brief_bytes)
        return det, desc
    elif strategy == 8:  # GFTT/ORB
        det = cv2.GFTTDetector_create(maxCorners=max_features)
        desc = cv2.ORB_create()
        return det, desc
    else:  # fallback
        det = cv2.ORB_create(nfeatures=max_features)
        return det, det


MAX_CACHED_MAPS = 3  # LRU limit — oldest-accessed map evicted when exceeded


class MapManager:
    """Manages loaded maps in memory for fast relocalization. Singleton.

    Uses LRU eviction: when more than MAX_CACHED_MAPS are loaded,
    the least-recently-used map is automatically unloaded.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._maps: OrderedDict[str, LoadedMap] = OrderedDict()
                    cls._instance = inst
        return cls._instance

    def get_or_load(self, map_id: str) -> LoadedMap:
        """Return cached map or load from disk."""
        if map_id in self._maps:
            self._maps.move_to_end(map_id)
            return self._maps[map_id]

        db_path = settings.MAPS_DIR / f"{map_id}.db"
        if not db_path.exists():
            raise FileNotFoundError(f"Map not found: {map_id}")

        self._maps[map_id] = LoadedMap(map_id, str(db_path))
        while len(self._maps) > MAX_CACHED_MAPS:
            evicted_id, _ = self._maps.popitem(last=False)
            logger.info(f"Map '{evicted_id}' evicted (LRU, max={MAX_CACHED_MAPS})")

        return self._maps[map_id]

    def unload(self, map_id: str):
        self._maps.pop(map_id, None)
        logger.info(f"Map '{map_id}' unloaded")

    def unload_all(self):
        self._maps.clear()
        logger.info("All maps unloaded")

    def localize(
        self, map_id: str, images: List[bytes], intrinsics: Optional[dict] = None
    ) -> dict:
        """Localize query images against a loaded map using PnP.

        1. Extract features from query images (same detector as map)
        2. Match against all map descriptors (BFMatcher + ratio test)
        3. Build 2D-3D correspondences from good matches
        4. Solve PnP with RANSAC to compute precise 6DoF camera pose
        5. Return the result with the most inliers across all query images
        """
        loaded = self.get_or_load(map_id)

        if loaded.all_descriptors is None:
            raise ValueError(
                "Map has no feature descriptors — "
                "ensure the map was processed with rtabmap-reprocess"
            )

        has_3d = loaded.descriptor_pos_3d is not None
        use_pnp = has_3d and intrinsics is not None

        if use_pnp:
            camera_matrix = np.array([
                [intrinsics['fx'], 0, intrinsics['cx']],
                [0, intrinsics['fy'], intrinsics['cy']],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros(4, dtype=np.float64)

        detector, desc_extractor = _create_detector(
            loaded.detector_strategy, loaded.max_features, loaded.brief_bytes
        )
        matcher = cv2.BFMatcher(loaded.norm_type, crossCheck=False)

        # Collect per-query-image results
        all_results = []

        for query_idx, img_bytes in enumerate(images):
            img = _decode_image_with_exif(img_bytes)
            if img is None:
                continue

            kps = detector.detect(img)
            kps, q_descs = desc_extractor.compute(img, kps)
            if q_descs is None or len(q_descs) == 0:
                continue

            matches = matcher.knnMatch(q_descs, loaded.all_descriptors, k=2)

            # Lowe's ratio test
            good = []
            for pair in matches:
                if len(pair) >= 2 and pair[0].distance < 0.75 * pair[1].distance:
                    good.append(pair[0])

            if not good:
                continue

            if use_pnp:
                # Build 2D-3D correspondences for PnP
                pts_2d = []
                pts_3d = []
                for m in good:
                    train_idx = m.trainIdx
                    local_3d = loaded.descriptor_pos_3d[train_idx]
                    # Skip if no 3D coordinate
                    if np.isnan(local_3d[0]):
                        continue
                    # Transform local 3D to world coordinates
                    # RTAB-Map depth coords: (depth_x=Z, depth_y=-Y, depth_z=X)
                    # in OpenCV optical frame (X-right, Y-down, Z-forward)
                    nid = int(loaded.descriptor_node_ids[train_idx])
                    transform = loaded.node_transforms.get(nid)
                    if transform is None:
                        continue
                    world_3d = transform[:, :3] @ local_3d + transform[:, 3]
                    pts_3d.append(world_3d)
                    # Query image 2D keypoint
                    pts_2d.append(kps[m.queryIdx].pt)

                if len(pts_2d) < 4:
                    continue

                obj_points = np.array(pts_3d, dtype=np.float64)
                img_points = np.array(pts_2d, dtype=np.float64)

                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_points, img_points, camera_matrix, dist_coeffs,
                    iterationsCount=300,
                    reprojectionError=5.0,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                if not success or inliers is None or len(inliers) < loaded.min_inliers:
                    continue

                num_inliers = len(inliers)

                # Convert PnP result to RTAB-Map pose convention.
                #
                # solvePnP returns R_w2c_opt, t_w2c_opt in OpenCV optical frame
                # (X-right, Y-down, Z-forward).
                #
                # RTAB-Map uses a different camera frame where:
                #   depth_x = Z_optical (forward)
                #   depth_y = -Y_optical (up, not down)
                #   depth_z = X_optical (right)
                #
                # Conversion: C = [[0,0,1],[0,-1,0],[1,0,0]]
                # maps RTAB-Map camera frame → OpenCV optical frame.
                #
                # Recovery: R_c2w_rtab = R_w2c_opt^T @ C
                #           t_c2w = -R_w2c_opt^T @ t_w2c_opt
                C = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=np.float64)
                R_w2c, _ = cv2.Rodrigues(rvec)
                R_cw = R_w2c.T @ C
                t_cw = (-R_w2c.T @ tvec).flatten()

                # Build 3x4 matrix in RTAB-Map convention (row-major)
                # [r00 r01 r02 tx]
                # [r10 r11 r12 ty]
                # [r20 r21 r22 tz]
                # Then use _parse_pose_blob logic for quaternion
                r00, r01, r02 = R_cw[0, 0], R_cw[0, 1], R_cw[0, 2]
                r10, r11, r12 = R_cw[1, 0], R_cw[1, 1], R_cw[1, 2]
                r20, r21, r22 = R_cw[2, 0], R_cw[2, 1], R_cw[2, 2]
                tx, ty, tz = t_cw[0], t_cw[1], t_cw[2]

                # Quaternion from rotation matrix (same as _parse_pose_blob)
                trace = r00 + r11 + r22
                if trace > 0:
                    s = 0.5 / (trace + 1.0) ** 0.5
                    qw = 0.25 / s
                    qx = (r21 - r12) * s
                    qy = (r02 - r20) * s
                    qz = (r10 - r01) * s
                elif r00 > r11 and r00 > r22:
                    s = 2.0 * (1.0 + r00 - r11 - r22) ** 0.5
                    qw = (r21 - r12) / s
                    qx = 0.25 * s
                    qy = (r01 + r10) / s
                    qz = (r02 + r20) / s
                elif r11 > r22:
                    s = 2.0 * (1.0 + r11 - r00 - r22) ** 0.5
                    qw = (r02 - r20) / s
                    qx = (r01 + r10) / s
                    qy = 0.25 * s
                    qz = (r12 + r21) / s
                else:
                    s = 2.0 * (1.0 + r22 - r00 - r11) ** 0.5
                    qw = (r10 - r01) / s
                    qx = (r02 + r20) / s
                    qy = (r12 + r21) / s
                    qz = 0.25 * s

                # Ensure consistent sign (qw positive convention)
                if qw < 0:
                    qx, qy, qz, qw = -qx, -qy, -qz, -qw

                confidence = min(0.95, max(0.1, num_inliers / len(obj_points)))

                all_results.append({
                    "query_index": query_idx,
                    "num_inliers": num_inliers,
                    "num_correspondences": len(obj_points),
                    "confidence": confidence,
                    "pose": {
                        "x": float(tx), "y": float(ty), "z": float(tz),
                        "qx": float(qx), "qy": float(qy),
                        "qz": float(qz), "qw": float(qw),
                    },
                })
            else:
                # Fallback: vote-based matching (no 3D data or no intrinsics)
                votes: Dict[int, int] = {}
                for m in good:
                    nid = int(loaded.descriptor_node_ids[m.trainIdx])
                    votes[nid] = votes.get(nid, 0) + 1

                best_nid = max(votes, key=votes.get)
                best_count = votes[best_nid]
                if best_count < loaded.min_inliers:
                    continue
                pose = loaded.node_poses.get(best_nid)
                if pose is None:
                    continue

                all_results.append({
                    "query_index": query_idx,
                    "num_inliers": best_count,
                    "num_correspondences": len(good),
                    "confidence": min(0.9, max(0.1, best_count / loaded.max_features)),
                    "pose": {
                        "x": pose[0], "y": pose[1], "z": pose[2],
                        "qx": pose[3], "qy": pose[4], "qz": pose[5], "qw": pose[6],
                    },
                })

        if not all_results:
            raise ValueError(
                "insufficient feature matches for reliable relocalization"
            )

        # Best result = highest inlier count
        best = max(all_results, key=lambda r: r["num_inliers"])

        method = "PnP+RANSAC" if use_pnp else "vote-based (fallback)"
        logger.info(
            f"Relocalized on map '{map_id}' ({method}): "
            f"inliers={best['num_inliers']}/{best['num_correspondences']}, "
            f"confidence={best['confidence']:.2f}"
        )

        return {
            "pose": best["pose"],
            "confidence": best["confidence"],
            "map_id": map_id,
            "num_matches": best["num_inliers"],
            "matched_image_index": best["query_index"],
            "method": method,
        }
