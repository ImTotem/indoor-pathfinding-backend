"""Incremental RTAB-Map sqlite db writer for streaming scan ingest.

Client streams frames (Node + Data tuples + Link entries) and the server
appends rows directly to an open scan's rtabmap.db. Schema mirrors RTAB-Map
0.23.x as observed in production iOS uploads.

This module is sync (sqlite3 stdlib). Callers wrap in `asyncio.to_thread`.
"""
from __future__ import annotations

import logging
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# Information matrix default for neighbor links (288B, 6×6 float64 identity).
_DEFAULT_INFO_MATRIX = struct.pack("<36d", *[
    1.0 if i == j else 0.0 for i in range(6) for j in range(6)
])


class FrameAppendError(Exception):
    """rtabmap.db incremental append failed."""


@dataclass(frozen=True)
class FrameRecord:
    """One Node + Data row to append.

    `pose` is 48B (3×4 float32 row-major, RTAB-Map world frame).
    `image` is JPEG bytes. `depth` is RVL compressed bytes (optional).
    `calibration` is 164B (RTAB-Map intrinsics + local_transform blob).
    """

    node_id: int
    map_id: int
    weight: int
    stamp: float
    pose: bytes
    image: bytes
    calibration: bytes
    depth: bytes | None = None
    depth_confidence: bytes | None = None
    ground_truth_pose: bytes | None = None
    velocity: bytes | None = None
    label: str | None = None
    gps: bytes | None = None
    env_sensors: bytes | None = None
    user_data: bytes | None = None
    # Optional 2D scan blobs (rarely populated by iOS client).
    scan: bytes | None = None
    scan_info: bytes | None = None


@dataclass(frozen=True)
class LinkRecord:
    """Edge between two Nodes.

    `transform` is 48B (3×4 float32 row-major, relative pose from_id → to_id).
    """

    from_id: int
    to_id: int
    type: int
    transform: bytes
    information_matrix: bytes | None = None
    user_data: bytes | None = None


def create_empty_db(db_path: Path) -> None:
    """Create a new rtabmap.db with the schema RTAB-Map 0.23.x expects.

    Existing file is left untouched (idempotent for restart). Callers must
    delete first if they want a clean re-init.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    try:
        con.executescript(
            """
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
                user_data BLOB
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
                descriptor BLOB
            );
            CREATE TABLE IF NOT EXISTS GlobalDescriptor (
                node_id INTEGER NOT NULL,
                type INTEGER,
                info BLOB,
                data BLOB
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
                id INTEGER PRIMARY KEY,
                stamp FLOAT,
                data BLOB,
                wm_state BLOB,
                time_enter DATE
            );
            CREATE TABLE IF NOT EXISTS Admin (
                version TEXT,
                preview_image BLOB,
                opt_ids BLOB,
                opt_poses BLOB,
                opt_last_localization BLOB,
                opt_graph BLOB,
                opt_map BLOB,
                opt_poses_constraints BLOB,
                opt_last_link_added BLOB,
                opt_map_correction BLOB,
                time_enter DATE
            );
            """
        )
        # Seed Admin with the RTAB-Map version the iOS client uses so reprocess
        # binary version checks pass. Idempotent — only insert if empty.
        cur = con.execute("SELECT COUNT(*) FROM Admin").fetchone()
        if cur and int(cur[0]) == 0:
            con.execute(
                "INSERT INTO Admin (version, time_enter) "
                "VALUES (?, datetime('now'))",
                ("0.23.5",),
            )
        con.commit()
    finally:
        con.close()


def last_node_id(db_path: Path) -> int:
    """Largest Node.id present. Returns 0 if Node table empty."""
    if not db_path.exists():
        return 0
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute("SELECT COALESCE(MAX(id), 0) FROM Node").fetchone()
        return int(row[0]) if row else 0
    finally:
        con.close()


def node_count(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute("SELECT COUNT(*) FROM Node").fetchone()
        return int(row[0]) if row else 0
    finally:
        con.close()


def append_batch(
    db_path: Path,
    frames: list[FrameRecord],
    links: list[LinkRecord],
) -> tuple[int, int]:
    """Append frames + links to rtabmap.db within one transaction.

    Drops frames whose node_id has already been ingested (idempotent retry).
    Drops links whose from_id/to_id is missing from Node table after append.

    Returns: (frames_applied, links_applied).
    """
    if not db_path.exists():
        raise FrameAppendError(f"rtabmap.db not initialized: {db_path}")

    con = sqlite3.connect(str(db_path))
    try:
        con.execute("BEGIN")
        existing_ids = {
            int(r[0])
            for r in con.execute("SELECT id FROM Node").fetchall()
        }

        frames_applied = 0
        for f in frames:
            if f.node_id in existing_ids:
                logger.debug(
                    "frame node_id=%d already ingested — skipping", f.node_id
                )
                continue
            if len(f.pose) != 48:
                raise FrameAppendError(
                    f"frame node_id={f.node_id}: pose must be 48 bytes, got "
                    f"{len(f.pose)}"
                )
            if len(f.calibration) != 164:
                raise FrameAppendError(
                    f"frame node_id={f.node_id}: calibration must be 164 bytes, "
                    f"got {len(f.calibration)}"
                )

            con.execute(
                "INSERT INTO Node "
                "(id, map_id, weight, stamp, pose, ground_truth_pose, velocity, "
                " label, gps, env_sensors, time_enter) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                (
                    f.node_id,
                    f.map_id,
                    f.weight,
                    f.stamp,
                    f.pose,
                    f.ground_truth_pose,
                    f.velocity,
                    f.label,
                    f.gps,
                    f.env_sensors,
                ),
            )
            con.execute(
                "INSERT INTO Data "
                "(id, image, depth, depth_confidence, calibration, scan, "
                " scan_info, cell_size, view_point_x, view_point_y, "
                " view_point_z, user_data, time_enter) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 0.0, 0.0, 0.0, 0.0, ?, "
                "        datetime('now'))",
                (
                    f.node_id,
                    f.image,
                    f.depth,
                    f.depth_confidence,
                    f.calibration,
                    f.scan,
                    f.scan_info,
                    f.user_data,
                ),
            )
            existing_ids.add(f.node_id)
            frames_applied += 1

        links_applied = 0
        for lk in links:
            if lk.from_id not in existing_ids or lk.to_id not in existing_ids:
                logger.debug(
                    "link from=%d to=%d skipped — node not yet ingested",
                    lk.from_id,
                    lk.to_id,
                )
                continue
            if len(lk.transform) != 48:
                raise FrameAppendError(
                    f"link {lk.from_id}->{lk.to_id}: transform must be 48 bytes,"
                    f" got {len(lk.transform)}"
                )
            info = lk.information_matrix or _DEFAULT_INFO_MATRIX
            if len(info) != 288:
                raise FrameAppendError(
                    f"link {lk.from_id}->{lk.to_id}: information_matrix must be "
                    f"288 bytes, got {len(info)}"
                )
            con.execute(
                "INSERT INTO Link "
                "(from_id, to_id, type, information_matrix, transform, "
                " user_data) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    lk.from_id,
                    lk.to_id,
                    lk.type,
                    info,
                    lk.transform,
                    lk.user_data,
                ),
            )
            links_applied += 1

        con.commit()
        return frames_applied, links_applied
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def sha256_of_db(db_path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(db_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
