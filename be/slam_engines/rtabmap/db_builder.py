"""Build RTAB-Map database with proper calibration, depth, and timestamps.

rtabmap-console cannot accept depth images or calibration via CLI.
This module builds the SQLite database directly so that rtabmap-reprocess
can compute odometry and optimization with full RGB-D data.
"""

import sqlite3
import struct
import os
import json
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np


RTABMAP_DB_VERSION = "0.22.0"

# Schema copied verbatim from introlab/rtabmap DatabaseSchema_0_22_0.sql
# https://github.com/introlab/rtabmap/blob/master/corelib/src/resources/backward_compatibility/DatabaseSchema_0_22_0.sql
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS Node (
    id INTEGER NOT NULL,
    map_id INTEGER NOT NULL,
    weight INTEGER,
    stamp FLOAT,
    pose BLOB,
    ground_truth_pose BLOB,
    velocity BLOB,
    label TEXT,
    gps BLOB,
    env_sensors BLOB,
    time_enter DATE,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS Data (
    id INTEGER NOT NULL,
    image BLOB,
    depth BLOB,
    depth_confidence BLOB,
    calibration BLOB,
    scan BLOB,
    scan_info BLOB,
    ground_cells BLOB,
    obstacle_cells BLOB,
    empty_cells BLOB,
    cell_size FLOAT,
    view_point_x FLOAT,
    view_point_y FLOAT,
    view_point_z FLOAT,
    user_data BLOB,
    time_enter DATE,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS Link (
    from_id INTEGER NOT NULL,
    to_id INTEGER NOT NULL,
    type INTEGER NOT NULL,
    information_matrix BLOB NOT NULL,
    transform BLOB,
    user_data BLOB,
    FOREIGN KEY (from_id) REFERENCES Node(id),
    FOREIGN KEY (to_id) REFERENCES Node(id)
);

CREATE TABLE IF NOT EXISTS Word (
    id INTEGER NOT NULL,
    descriptor_size INTEGER NOT NULL,
    descriptor BLOB NOT NULL,
    time_enter DATE,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS Feature (
    node_id INTEGER NOT NULL,
    word_id INTEGER NOT NULL,
    pos_x FLOAT NOT NULL,
    pos_y FLOAT NOT NULL,
    size INTEGER NOT NULL,
    dir FLOAT NOT NULL,
    response FLOAT NOT NULL,
    octave INTEGER NOT NULL,
    depth_x FLOAT,
    depth_y FLOAT,
    depth_z FLOAT,
    descriptor_size INTEGER,
    descriptor BLOB,
    FOREIGN KEY (node_id) REFERENCES Node(id)
);

CREATE TABLE IF NOT EXISTS GlobalDescriptor (
    node_id INTEGER NOT NULL,
    type INTEGER NOT NULL,
    info BLOB,
    data BLOB NOT NULL,
    FOREIGN KEY (node_id) REFERENCES Node(id)
);

CREATE TABLE IF NOT EXISTS Info (
    STM_size INTEGER,
    last_sign_added INTEGER,
    process_mem_used INTEGER,
    database_mem_used INTEGER,
    dictionary_size INTEGER,
    parameters TEXT,
    time_enter DATE
);

CREATE TABLE IF NOT EXISTS Statistics (
    id INTEGER NOT NULL,
    stamp FLOAT,
    data BLOB,
    wm_state BLOB,
    FOREIGN KEY (id) REFERENCES Node(id)
);

CREATE TABLE IF NOT EXISTS Admin (
    version TEXT,
    preview_image BLOB,
    opt_cloud BLOB,
    opt_ids BLOB,
    opt_poses BLOB,
    opt_last_localization BLOB,
    opt_polygons_size INTEGER,
    opt_polygons BLOB,
    opt_tex_coords BLOB,
    opt_tex_materials BLOB,
    opt_map BLOB,
    opt_map_x_min FLOAT,
    opt_map_y_min FLOAT,
    opt_map_resolution FLOAT,
    time_enter DATE
);

CREATE TRIGGER IF NOT EXISTS insert_Node_timeEnter AFTER INSERT ON Node
BEGIN
 UPDATE Node SET time_enter = DATETIME('NOW') WHERE rowid = new.rowid;
END;

CREATE TRIGGER IF NOT EXISTS insert_Data_timeEnter AFTER INSERT ON Data
BEGIN
 UPDATE Data SET time_enter = DATETIME('NOW') WHERE rowid = new.rowid;
END;

CREATE TRIGGER IF NOT EXISTS insert_Word_timeEnter AFTER INSERT ON Word
BEGIN
 UPDATE Word SET time_enter = DATETIME('NOW') WHERE rowid = new.rowid;
END;

CREATE TRIGGER IF NOT EXISTS insert_Info_timeEnter AFTER INSERT ON Info
BEGIN
 UPDATE Info SET time_enter = DATETIME('NOW') WHERE rowid = new.rowid;
END;

CREATE INDEX IF NOT EXISTS IDX_Feature_node_id ON Feature (node_id);
CREATE INDEX IF NOT EXISTS IDX_GlobalDescriptor_node_id ON GlobalDescriptor (node_id);
CREATE INDEX IF NOT EXISTS IDX_Link_from_id ON Link (from_id);
"""


def build_calibration_blob(
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int
) -> bytes:
    """Build a 164-byte RTAB-Map CameraModel calibration blob.

    Format verified from existing RTAB-Map databases:
      '<3i i 2i 4i i 9d 12f' = 164 bytes
    """
    K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

    optical_rotation = [
        0.0, 0.0, 1.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
    ]

    blob = struct.pack(
        '<3i i 2i 4i i 9d 12f',
        0, 22, 0,           # version
        0,                   # type
        width, height,       # image size
        9, 0, 0, 0,         # K_size, D_size, R_size, P_size
        12,                  # localTransformSize
        *K,                  # K matrix (9 doubles)
        *optical_rotation,   # local transform (12 floats)
    )
    return blob


def build_identity_pose() -> bytes:
    """3x4 identity transform as 12 floats (48 bytes)."""
    return struct.pack('<12f',
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
    )


def load_and_resize_depth(depth_path: str, target_w: int, target_h: int) -> Optional[bytes]:
    """Load 16-bit depth PNG and resize to target resolution.

    Returns compressed PNG bytes at target resolution, or None if file
    is missing / empty.
    """
    if not os.path.exists(depth_path):
        return None

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None or depth.size == 0 or depth.max() == 0:
        return None

    if depth.shape[1] != target_w or depth.shape[0] != target_h:
        depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    _, buf = cv2.imencode('.png', depth)
    return buf.tobytes()


def build_database(
    session_path: str,
    intrinsics: Dict,
    output_db: Optional[str] = None,
    slam_params: Optional[Dict[str, str]] = None,
    monocular: bool = False,
) -> str:
    """Build an RTAB-Map database from session data.

    Reads images/, depth/, chunks/ from session_path and creates a
    properly structured rtabmap.db with calibration, depth, and timestamps.

    Args:
        session_path: Directory containing images/, depth/, chunks/
        intrinsics: Dict with fx, fy, cx, cy, width, height
        output_db: Output path (default: session_path/rtabmap_input.db)
        slam_params: RTAB-Map parameters dict (will be serialized to Info table)
        monocular: If True, skip depth data (monocular SLAM mode)

    Returns:
        Path to created database file
    """
    session = Path(session_path)
    images_dir = session / "images"
    depth_dir = session / "depth"
    chunks_dir = session / "chunks"

    if output_db is None:
        output_db = str(session / "rtabmap_input.db")

    if os.path.exists(output_db):
        os.remove(output_db)

    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']
    img_w = intrinsics.get('width', 1920)
    img_h = intrinsics.get('height', 1080)

    calib_blob = build_calibration_blob(fx, fy, cx, cy, img_w, img_h)
    identity_pose = build_identity_pose()

    frame_meta = _load_frame_metadata(chunks_dir)

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    print(f"[DB Builder] Building database: {len(image_files)} images found, "
          f"intrinsics {img_w}x{img_h} fx={fx:.1f} fy={fy:.1f}")

    conn = sqlite3.connect(output_db)
    conn.executescript(SCHEMA_SQL)

    conn.execute(
        "INSERT INTO Admin (version) VALUES (?)",
        (RTABMAP_DB_VERSION,)
    )

    skipped = 0
    node_id = 0
    for img_file in image_files:
        stem = Path(img_file).stem

        depth_data = None
        if not monocular:
            depth_file = stem + ".png"
            depth_path = str(depth_dir / depth_file)
            depth_data = load_and_resize_depth(depth_path, img_w, img_h)

            if depth_data is None:
                skipped += 1
                continue

        node_id += 1

        img_path = str(images_dir / img_file)
        with open(img_path, 'rb') as f:
            image_data = f.read()

        meta = frame_meta.get(stem, {})
        stamp = meta.get('timestamp', 0.0)
        if stamp > 0:
            stamp = stamp / 1000.0

        conn.execute(
            "INSERT INTO Node (id, map_id, weight, stamp, pose) VALUES (?, 0, 0, ?, ?)",
            (node_id, stamp, identity_pose)
        )

        conn.execute(
            "INSERT INTO Data (id, image, depth, calibration) VALUES (?, ?, ?, ?)",
            (node_id, image_data, depth_data, calib_blob)
        )

    print(f"[DB Builder] Inserted {node_id} frames, skipped {skipped} ({'monocular mode' if monocular else 'no valid depth'})")

    params_str = ""
    if slam_params:
        params_str = ";".join(f"{k}:{v}" for k, v in slam_params.items()) + ";"
        print(f"[DB Builder] Embedding {len(slam_params)} SLAM parameters into database")

    conn.execute(
        "INSERT INTO Info (STM_size, last_sign_added, process_mem_used, "
        "database_mem_used, dictionary_size, parameters) VALUES (0, ?, 0, 0, 0, ?)",
        (node_id, params_str)
    )

    conn.commit()
    conn.close()

    db_size = os.path.getsize(output_db) / (1024 * 1024)
    print(f"[DB Builder] Database created: {output_db} ({db_size:.1f} MB)")
    return output_db


def _load_frame_metadata(chunks_dir: Path) -> Dict[str, Dict]:
    """Load per-frame metadata from chunk JSON files.

    Returns dict keyed by image stem (e.g. '000000') with timestamp etc.
    """
    result = {}
    if not chunks_dir.exists():
        return result

    for chunk_file in sorted(chunks_dir.iterdir()):
        if not chunk_file.suffix == '.json':
            continue
        try:
            with open(chunk_file) as f:
                frames = json.load(f)
            for frame in frames:
                img_path = frame.get('image_path', '')
                stem = Path(img_path).stem
                if stem:
                    result[stem] = {
                        'timestamp': frame.get('timestamp', 0.0),
                        'position': frame.get('position'),
                        'orientation': frame.get('orientation'),
                    }
        except (json.JSONDecodeError, KeyError):
            continue

    return result
