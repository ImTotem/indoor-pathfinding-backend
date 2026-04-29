"""SuperPoint + LightGlue match visualization for debug purposes."""
import sqlite3
from typing import Optional

import cv2
import numpy as np

from utils import logger


def _load_node_image_bgr(db_path: str, node_id: int) -> Optional[np.ndarray]:
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT image FROM Data WHERE id = ?", (node_id,)).fetchone()
        conn.close()
        if row and row[0]:
            arr = np.frombuffer(bytes(row[0]), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.warning(f"[SPMatchDebugger] Cannot load node {node_id} image: {e}")
    return None


def visualize_matches_sp(
    db_path: str,
    map_id: str,
    img_bytes: bytes,
    engine,
    max_draw: int = 50,
) -> dict:
    """Visualize SuperPoint+LightGlue matches between query image and best DB keyframe.

    Returns the same dict structure as the RTABMap match_debugger.visualize_matches().
    """
    from .map_manager import SuperPointMapManager
    from .engine import _to_gray_float

    mgr = SuperPointMapManager()
    loaded = mgr.get_or_load(map_id, db_path)

    if not loaded.node_ids:
        raise ValueError("No keyframes with stored images in this map")

    gray = _to_gray_float(img_bytes)
    if gray is None:
        raise ValueError("Cannot decode query image")

    q_feats = engine._extract(gray)
    q_kps = q_feats['keypoints'][0].cpu().numpy()   # (N, 2)
    q_desc_mean = q_feats['descriptors'][0].cpu().mean(0)  # (256,)

    candidates = loaded.top_k_candidates(q_desc_mean)

    best_node_id = None
    best_matches = None
    best_count = 0

    for node_id in candidates:
        db_feats = loaded.keyframe_feats[node_id]
        matches = engine._match(
            {k: v.cpu() for k, v in q_feats.items()},
            db_feats,
        )  # (P, 2)
        if len(matches) > best_count:
            best_count = len(matches)
            best_node_id = node_id
            best_matches = matches

    if best_node_id is None or best_matches is None or best_count == 0:
        raise ValueError("No SuperPoint+LightGlue matches found")

    db_feats = loaded.keyframe_feats[best_node_id]
    db_kps = db_feats['keypoints'][0].cpu().numpy()  # (M, 2)

    db_bgr = _load_node_image_bgr(db_path, best_node_id)
    query_bgr = cv2.cvtColor(
        (gray * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
    )

    q_cv_kps = [cv2.KeyPoint(float(x), float(y), 5.0) for x, y in q_kps]
    draw_matches = best_matches[:max_draw]

    if db_bgr is not None:
        db_cv_kps = [cv2.KeyPoint(float(x), float(y), 5.0) for x, y in db_kps]
        cv_dmatches = [
            cv2.DMatch(int(qi), int(di), 0.0)
            for qi, di in draw_matches
        ]
        vis = cv2.drawMatches(
            query_bgr, q_cv_kps,
            db_bgr, db_cv_kps,
            cv_dmatches,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(180, 180, 180),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
    else:
        vis = cv2.drawKeypoints(
            query_bgr, q_cv_kps, None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

    return {
        "vis_bgr": vis,
        "query_bgr": query_bgr,
        "db_bgr": db_bgr,
        "best_node_id": best_node_id,
        "num_good_matches": best_count,
        "num_node_matches": best_count,
        "has_db_image": db_bgr is not None,
    }
