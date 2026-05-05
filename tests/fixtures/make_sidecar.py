"""v4 scan_metadata.db를 프로그래밍 방식으로 생성하는 헬퍼."""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

_FK = "FOREIGN KEY (scan_id, keyframe_seq) REFERENCES keyframe_meta(scan_id, seq) ON DELETE CASCADE"  # noqa: E501

_DDL = f"""
    CREATE TABLE scan_session (
        id TEXT PRIMARY KEY,
        started_at INTEGER NOT NULL,
        ended_at INTEGER,
        device_model TEXT NOT NULL,
        app_version TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('recording','saved','discarded')),
        keyframe_count INTEGER NOT NULL DEFAULT 0,
        notes TEXT
    );

    CREATE TABLE keyframe_meta (
        scan_id TEXT NOT NULL REFERENCES scan_session(id) ON DELETE CASCADE,
        seq INTEGER NOT NULL,
        captured_at INTEGER NOT NULL,
        image_path TEXT NOT NULL,
        pose_matrix BLOB NOT NULL,
        tx REAL NOT NULL, ty REAL NOT NULL, tz REAL NOT NULL,
        tracking_state TEXT NOT NULL,
        rtabmap_node_id INTEGER,
        PRIMARY KEY (scan_id, seq)
    );

    CREATE TABLE poi_mark (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id TEXT NOT NULL,
        keyframe_seq INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        pose_matrix BLOB NOT NULL,
        tx REAL NOT NULL, ty REAL NOT NULL, tz REAL NOT NULL,
        track_id INTEGER,
        label TEXT,
        source TEXT NOT NULL DEFAULT 'track_lock'
            CHECK(source IN ('track_lock','manual')),
        {_FK}
    );

    CREATE UNIQUE INDEX idx_poi_mark_unique_track
        ON poi_mark(scan_id, track_id) WHERE track_id IS NOT NULL;

    CREATE TABLE poi_photo (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        poi_mark_id INTEGER NOT NULL REFERENCES poi_mark(id) ON DELETE CASCADE,
        scan_id TEXT NOT NULL REFERENCES scan_session(id) ON DELETE CASCADE,
        keyframe_seq INTEGER NOT NULL,
        captured_at INTEGER NOT NULL,
        bbox_x REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL,
        class_name TEXT NOT NULL,
        confidence REAL NOT NULL,
        {_FK}
    );

    CREATE TABLE branch_mark (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id TEXT NOT NULL,
        keyframe_seq INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        pose_matrix BLOB NOT NULL,
        tx REAL NOT NULL, ty REAL NOT NULL, tz REAL NOT NULL,
        {_FK}
    );

    CREATE TABLE yolo_detection (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id TEXT NOT NULL,
        keyframe_seq INTEGER NOT NULL,
        class_name TEXT NOT NULL,
        confidence REAL NOT NULL,
        bbox_x REAL NOT NULL, bbox_y REAL NOT NULL,
        bbox_w REAL NOT NULL, bbox_h REAL NOT NULL,
        mask_rle BLOB,
        source TEXT NOT NULL DEFAULT 'on_device'
            CHECK(source IN ('on_device','server')),
        track_id INTEGER,
        {_FK}
    );
"""


def make_v4_sidecar(path: Path, scan_id: str, *, keyframes: int = 3) -> None:
    """테스트용 v4 sidecar DB를 path에 생성한다."""
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_DDL)
    _insert_data(conn, scan_id, keyframes)
    conn.execute("PRAGMA user_version = 4")
    conn.commit()
    conn.close()


_DDL_V6 = f"""
    CREATE TABLE scan_session (
        id TEXT PRIMARY KEY,
        started_at INTEGER NOT NULL,
        ended_at INTEGER,
        device_model TEXT NOT NULL,
        app_version TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('recording','saved','discarded')),
        keyframe_count INTEGER NOT NULL DEFAULT 0,
        notes TEXT
    );

    CREATE TABLE keyframe_meta (
        scan_id TEXT NOT NULL REFERENCES scan_session(id) ON DELETE CASCADE,
        seq INTEGER NOT NULL,
        captured_at INTEGER NOT NULL,
        image_path TEXT NOT NULL,
        pose_matrix BLOB NOT NULL,
        tx REAL NOT NULL, ty REAL NOT NULL, tz REAL NOT NULL,
        tracking_state TEXT NOT NULL,
        rtabmap_node_id INTEGER,
        PRIMARY KEY (scan_id, seq)
    );

    -- Sprint 65 v6: track_id, bbox 컬럼 제거.
    CREATE TABLE poi_mark (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id TEXT NOT NULL,
        keyframe_seq INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        pose_matrix BLOB NOT NULL,
        tx REAL NOT NULL, ty REAL NOT NULL, tz REAL NOT NULL,
        label TEXT,
        source TEXT NOT NULL DEFAULT 'manual'
            CHECK(source IN ('track_lock','manual')),
        {_FK}
    );

    CREATE TABLE poi_photo (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        poi_mark_id INTEGER NOT NULL REFERENCES poi_mark(id) ON DELETE CASCADE,
        scan_id TEXT NOT NULL REFERENCES scan_session(id) ON DELETE CASCADE,
        keyframe_seq INTEGER NOT NULL,
        captured_at INTEGER NOT NULL,
        class_name TEXT NOT NULL,
        confidence REAL NOT NULL,
        image_blob BLOB,
        {_FK}
    );

    CREATE TABLE branch_mark (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id TEXT NOT NULL,
        keyframe_seq INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        pose_matrix BLOB NOT NULL,
        tx REAL NOT NULL, ty REAL NOT NULL, tz REAL NOT NULL,
        {_FK}
    );

    -- Sprint 65 v6: yolo_detection 통째 제거.

    -- Sprint 65 v6: interfloor_mark 신설.
    CREATE TABLE interfloor_mark (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id TEXT NOT NULL REFERENCES scan_session(id) ON DELETE CASCADE,
        keyframe_seq INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        connector_type TEXT NOT NULL
            CHECK(connector_type IN ('elevator','escalator','stairs')),
        prefix TEXT NOT NULL,
        pose_matrix BLOB NOT NULL,
        tx REAL NOT NULL, ty REAL NOT NULL, tz REAL NOT NULL,
        {_FK}
    );
"""


def make_v6_sidecar(
    path: Path,
    scan_id: str,
    *,
    keyframes: int = 3,
    interfloor: list[tuple[int, str, str]] | None = None,
) -> None:
    """Sprint 65 v6 sidecar DB.

    interfloor: [(keyframe_seq, connector_type, prefix), ...]
    """
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_DDL_V6)

    conn.execute(
        "INSERT INTO scan_session VALUES (?,?,?,?,?,?,?,?)",
        (scan_id, 1000000, 1001000, "iPhone 15 Pro", "0.20.0", "saved", keyframes, None),
    )
    for i in range(1, keyframes + 1):
        conn.execute(
            "INSERT INTO keyframe_meta VALUES (?,?,?,?,?,?,?,?,?,?)",
            (scan_id, i, 1000000 + i * 100, f"keyframes/{i:06d}.jpg",
             _fake_pose(), float(i), 0.0, 0.0, "normal", i),
        )

    # 수동 POI 1개 (manual single-path). keyframes 가 1 이면 POI 생성 SKIP — FK 보호.
    if keyframes >= 2:
        poi_seq = 2
        conn.execute(
            "INSERT INTO poi_mark "
            "(scan_id, keyframe_seq, created_at, pose_matrix, tx, ty, tz, label, source) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (scan_id, poi_seq, 1000200, _fake_pose(), 2.0, 0.0, 0.0, "exit", "manual"),
        )
        poi_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        conn.execute(
            "INSERT INTO poi_photo "
            "(poi_mark_id, scan_id, keyframe_seq, captured_at, class_name, confidence, image_blob) "
            "VALUES (?,?,?,?,?,?,?)",
            (poi_id, scan_id, poi_seq, 1000210, "manual", 0.0, None),
        )

    # branch_mark
    conn.execute(
        "INSERT INTO branch_mark (scan_id, keyframe_seq, created_at, pose_matrix, tx, ty, tz) "
        "VALUES (?,?,?,?,?,?,?)",
        (scan_id, 1, 1000100, _fake_pose(), 0.5, 0.0, 0.0),
    )

    for seq, connector_type, prefix in (interfloor or []):
        conn.execute(
            "INSERT INTO interfloor_mark "
            "(scan_id, keyframe_seq, created_at, connector_type, prefix, "
            "pose_matrix, tx, ty, tz) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (scan_id, seq, 1000300 + seq, connector_type, prefix,
             _fake_pose(), float(seq), 0.0, 0.0),
        )

    conn.execute("PRAGMA user_version = 6")
    conn.commit()
    conn.close()


def make_v6_manifest(scan_id: str, *, keyframes: int = 3) -> dict[str, object]:
    """Sprint 65 v6 manifest.json 사전 형식."""
    return {
        "metadata_version": 6,
        "scan_id": scan_id,
        "mode": "raw_arkit_recording",
        "keyframes_included": False,
        "keyframe_image_source": "rtabmap_db_data_table",
        "poi_image_source": "poi_photo_image_blob",
        "rtabmap_accepted_frame_count": keyframes,
        "sidecar_keyframe_meta_count": keyframes,
        "dropped_reject_frame_image_count": 0,
        "rtabmap_reprocessed": False,
        "client_app_version": "0.20.0",
    }


def _fake_pose() -> bytes:
    return struct.pack("16f", *([0.0] * 16))


_POI_COLS = (
    "scan_id, keyframe_seq, created_at, pose_matrix, tx, ty, tz, track_id, label, source"
)
_PHOTO_COLS = (
    "poi_mark_id, scan_id, keyframe_seq, captured_at, "
    "bbox_x, bbox_y, bbox_w, bbox_h, class_name, confidence"
)
_YOLO_COLS = (
    "scan_id, keyframe_seq, class_name, confidence, "
    "bbox_x, bbox_y, bbox_w, bbox_h, source, track_id"
)


def _insert_data(conn: sqlite3.Connection, scan_id: str, keyframes: int) -> None:
    conn.execute(
        "INSERT INTO scan_session VALUES (?,?,?,?,?,?,?,?)",
        (scan_id, 1000000, 1001000, "iPhone 15 Pro", "0.15.0", "saved", keyframes, None),
    )
    for i in range(1, keyframes + 1):
        conn.execute(
            "INSERT INTO keyframe_meta VALUES (?,?,?,?,?,?,?,?,?,?)",
            (scan_id, i, 1000000 + i * 100, f"keyframes/{i:06d}.jpg",
             _fake_pose(), float(i), 0.0, 0.0, "normal", i),
        )

    # track_lock POI
    conn.execute(
        f"INSERT INTO poi_mark ({_POI_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (scan_id, 1, 1000100, _fake_pose(), 1.0, 0.0, 0.0, 42, None, "track_lock"),
    )
    poi_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # poi_photo (track_lock — bbox 있음)
    conn.execute(
        f"INSERT INTO poi_photo ({_PHOTO_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (poi_id, scan_id, 1, 1000110, 0.1, 0.2, 0.3, 0.4, "door", 0.92),
    )

    # manual POI (bbox NULL)
    conn.execute(
        f"INSERT INTO poi_mark ({_POI_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (scan_id, 2, 1000200, _fake_pose(), 2.0, 0.0, 0.0, None, "exit", "manual"),
    )
    manual_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        f"INSERT INTO poi_photo ({_PHOTO_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (manual_id, scan_id, 2, 1000210, None, None, None, None, "manual", 1.0),
    )

    # branch_mark
    conn.execute(
        "INSERT INTO branch_mark (scan_id, keyframe_seq, created_at, pose_matrix, tx, ty, tz) "
        "VALUES (?,?,?,?,?,?,?)",
        (scan_id, 1, 1000100, _fake_pose(), 0.5, 0.0, 0.0),
    )

    # yolo_detection
    conn.execute(
        f"INSERT INTO yolo_detection ({_YOLO_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (scan_id, 1, "door", 0.88, 0.1, 0.2, 0.3, 0.4, "on_device", 42),
    )
