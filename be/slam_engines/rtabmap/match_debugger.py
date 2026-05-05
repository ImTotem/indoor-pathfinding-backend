from __future__ import annotations

import numpy as np


def visualize_matches(
    db_path: str,
    map_id: str,
    image_bytes: bytes,
    max_matches: int = 50,
    mask_persons: bool = False,
) -> dict:
    _ = (db_path, map_id, image_bytes, max_matches, mask_persons)
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
