"""Feature-match visualization for debug purposes.

Reads from MapManager's in-memory cache without modifying any service logic.
"""
import sqlite3
from typing import Dict, List, Optional

import cv2
import numpy as np

from utils import logger


def _load_node_image(db_path: str, node_id: int) -> Optional[np.ndarray]:
    """Load the stored JPEG image for a node from the RTABMap SQLite DB."""
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT image FROM Data WHERE id = ?", (node_id,)).fetchone()
        conn.close()
        if row and row[0]:
            arr = np.frombuffer(bytes(row[0]), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.warning(f"[MatchDebugger] Cannot load node {node_id} image: {e}")
    return None


def visualize_matches(
    db_path: str,
    map_id: str,
    img_bytes: bytes,
    max_draw: int = 50,
) -> dict:
    """
    Extract features from a query image, match against the loaded map,
    and return annotated images.

    Returns:
        dict with keys:
            vis_bgr      – side-by-side matches image (numpy BGR)
            query_bgr    – query image with detected keypoints
            db_bgr       – matched DB keyframe image (or None)
            best_node_id – node ID of the best-matched keyframe
            num_good_matches  – total good matches (ratio test)
            num_node_matches  – matches belonging to best node
            has_db_image – whether the DB had a stored image for the node
    """
    from .map_manager import (
        MapManager,
        LoadedMap,
        _create_detector,
        _decode_image_with_exif,
    )

    mgr = MapManager()
    loaded: LoadedMap = mgr.get_or_load(map_id, db_path)

    if loaded.all_descriptors is None:
        raise ValueError("Map has no descriptors")

    # --- query: decode + extract features (same detector as DB used) ---
    gray = _decode_image_with_exif(img_bytes)
    if gray is None:
        raise ValueError("Cannot decode query image")

    det, desc_ext = _create_detector(
        loaded.detector_strategy, loaded.max_features, loaded.brief_bytes
    )
    q_kps = det.detect(gray, None)
    q_kps, q_descs = desc_ext.compute(gray, q_kps)

    if q_descs is None or len(q_kps) == 0:
        raise ValueError("No features detected in query image")

    # --- BF match with Lowe ratio test (mirrors map_manager) ---
    matcher = cv2.BFMatcher(loaded.norm_type)
    raw = matcher.knnMatch(q_descs, loaded.all_descriptors, k=2)

    good: List[cv2.DMatch] = []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.7 * n.distance:
                good.append(m)

    if not good:
        raise ValueError("No good matches found")

    # --- find node with most matches ---
    node_counts: Dict[int, int] = {}
    for m in good:
        nid = int(loaded.descriptor_node_ids[m.trainIdx])
        node_counts[nid] = node_counts.get(nid, 0) + 1

    best_node_id = max(node_counts, key=node_counts.get)
    node_matches = [
        m for m in good
        if int(loaded.descriptor_node_ids[m.trainIdx]) == best_node_id
    ]

    # Build local KeyPoint + DMatch lists for cv2.drawMatches
    db_kps_local: List[cv2.KeyPoint] = []
    local_matches: List[cv2.DMatch] = []
    for i, m in enumerate(node_matches):
        px, py = loaded.descriptor_pos_2d[m.trainIdx]
        db_kps_local.append(cv2.KeyPoint(float(px), float(py), 5.0))
        local_matches.append(cv2.DMatch(m.queryIdx, i, m.distance))

    # --- load DB keyframe image ---
    db_bgr = _load_node_image(db_path, best_node_id)
    query_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if db_bgr is not None:
        vis = cv2.drawMatches(
            query_bgr, list(q_kps),
            db_bgr, db_kps_local,
            local_matches[:max_draw],
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(180, 180, 180),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
    else:
        # No stored image in DB — draw query keypoints only
        vis = cv2.drawKeypoints(
            query_bgr, list(q_kps), None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

    return {
        "vis_bgr": vis,
        "query_bgr": query_bgr,
        "db_bgr": db_bgr,
        "best_node_id": best_node_id,
        "num_good_matches": len(good),
        "num_node_matches": len(node_matches),
        "has_db_image": db_bgr is not None,
    }
