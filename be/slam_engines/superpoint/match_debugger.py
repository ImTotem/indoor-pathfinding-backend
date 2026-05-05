from __future__ import annotations

import numpy as np


def visualize_matches_sp(db_path: str, map_id: str, image_bytes: bytes, sp_engine: object) -> dict:
    _ = (db_path, map_id, image_bytes, sp_engine)
    blank = np.zeros((32, 32, 3), dtype=np.uint8)
    return {
        "query_bgr": blank,
        "vis_bgr": blank,
        "db_bgr": blank,
        "best_node_id": 0,
        "num_good_matches": 0,
        "num_node_matches": 0,
        "has_db_image": False,
    }
