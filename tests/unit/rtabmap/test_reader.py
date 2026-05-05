from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import pytest

from indoor_server.application.rtabmap.reader import RtabmapReader, decode_pose_3x4_blob


def _pose_blob(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> bytes:
    return struct.pack(
        "<12f",
        1.0, 0.0, 0.0, tx,
        0.0, 1.0, 0.0, ty,
        0.0, 0.0, 1.0, tz,
    )


def _create_rtabmap_db(path: Path, *, nodes: int = 3, features_per_node: int = 120) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE Node (
                id INTEGER PRIMARY KEY,
                map_id INTEGER NOT NULL,
                weight INTEGER,
                stamp FLOAT,
                pose BLOB,
                ground_truth_pose BLOB,
                velocity BLOB,
                label TEXT,
                gps BLOB,
                env_sensors BLOB,
                time_enter DATE
            );
            CREATE TABLE Data (
                id INTEGER PRIMARY KEY,
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
                view_point_z FLOAT
            );
            CREATE TABLE Link (
                from_id INTEGER NOT NULL,
                to_id INTEGER NOT NULL,
                type INTEGER NOT NULL,
                information_matrix BLOB NOT NULL,
                transform BLOB,
                user_data BLOB
            );
            CREATE TABLE Feature (
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
            CREATE TABLE Word (id INTEGER PRIMARY KEY, word BLOB);
            CREATE TABLE GlobalDescriptor (node_id INTEGER, descriptor BLOB);
            CREATE TABLE Statistics (id INTEGER, stamp FLOAT, data BLOB, wm_state BLOB);
            """
        )
        for node_id in range(1, nodes + 1):
            conn.execute(
                "INSERT INTO Node (id, map_id, weight, stamp, pose, ground_truth_pose, label) "
                "VALUES (?, 0, 0, ?, ?, ?, ?)",
                (
                    node_id,
                    1000.0 + node_id,
                    _pose_blob(tx=float(node_id), ty=0.0, tz=0.0),
                    _pose_blob(),
                    f"node-{node_id}",
                ),
            )
            conn.execute(
                "INSERT INTO Data (id, image, depth, calibration, scan, cell_size) "
                "VALUES (?, ?, ?, ?, ?, 0.0)",
                (node_id, b"\xff\xd8jpg", b"\x89PNGdepth", b"calibration", b"scan"),
            )
            conn.execute(
                "INSERT INTO Statistics (id, stamp, data, wm_state) VALUES (?, ?, ?, ?)",
                (node_id, 1000.0 + node_id, b"stats", b"wm"),
            )
            if node_id < nodes:
                conn.execute(
                    "INSERT INTO Link (from_id, to_id, type, information_matrix, transform) "
                    "VALUES (?, ?, 0, ?, ?)",
                    (node_id, node_id + 1, b"i" * 288, _pose_blob(tx=1.0)),
                )
            for idx in range(features_per_node):
                word_id = node_id * 1000 + idx
                conn.execute("INSERT INTO Word (id, word) VALUES (?, ?)", (word_id, b"w"))
                conn.execute(
                    """
                    INSERT INTO Feature (
                        node_id, word_id, pos_x, pos_y, size, dir, response, octave,
                        depth_x, depth_y, depth_z, descriptor_size, descriptor
                    ) VALUES (?, ?, ?, ?, 3, 0, 1, 0, ?, ?, ?, 64, ?)
                    """,
                    (
                        node_id,
                        word_id,
                        float(idx),
                        float(idx + 1),
                        float(idx) / 100.0,
                        0.0,
                        1.0,
                        b"d" * 64,
                    ),
                )
        conn.commit()
    finally:
        conn.close()


def test_decode_pose_3x4_blob_returns_homogeneous_matrix() -> None:
    matrix = decode_pose_3x4_blob(_pose_blob(tx=1.5, ty=-2.0, tz=0.25))

    assert matrix[0] == (1.0, 0.0, 0.0, 1.5)
    assert matrix[1] == (0.0, 1.0, 0.0, -2.0)
    assert matrix[2] == (0.0, 0.0, 1.0, 0.25)
    assert matrix[3] == (0.0, 0.0, 0.0, 1.0)


def test_inspect_ready_database(tmp_path: Path) -> None:
    db_path = tmp_path / "rtabmap.db"
    _create_rtabmap_db(db_path, nodes=3, features_per_node=120)

    diagnostics = RtabmapReader().inspect(
        db_path,
        keyframe_node_ids=[1, 2, 3, None],
    )

    assert diagnostics.ready
    assert diagnostics.issues == []
    assert diagnostics.node_count == 3
    assert diagnostics.data_count == 3
    assert diagnostics.feature_count == 360
    assert diagnostics.feature_3d_count == 360
    assert diagnostics.keyframe_count == 4
    assert diagnostics.keyframe_node_count == 3
    assert diagnostics.keyframe_node_coverage == 0.75


def test_inspect_accepts_partial_sidecar_to_node_coverage(tmp_path: Path) -> None:
    """RTAB-Map 자체 evidence가 충분하면 일부 keyframe reject는 허용한다."""
    db_path = tmp_path / "rtabmap.db"
    _create_rtabmap_db(db_path, nodes=3, features_per_node=120)

    diagnostics = RtabmapReader().inspect(
        db_path,
        keyframe_node_ids=[1, 2, None, None, 3, None, None],
    )

    assert diagnostics.ready
    assert diagnostics.keyframe_node_coverage == pytest.approx(3 / 7)


def test_inspect_rejects_mock_file(tmp_path: Path) -> None:
    db_path = tmp_path / "rtabmap.db"
    db_path.write_bytes(b"RTABMap mock")

    diagnostics = RtabmapReader().inspect(db_path, keyframe_node_ids=[None])

    assert not diagnostics.ready
    assert diagnostics.issues
    assert diagnostics.issues[0].startswith("sqlite_open_failed:")


def test_inspect_missing_database(tmp_path: Path) -> None:
    diagnostics = RtabmapReader().inspect(tmp_path / "missing.db")

    assert not diagnostics.ready
    assert diagnostics.issues == ["rtabmap_db_missing"]


def test_inspect_rejects_low_keyframe_node_coverage(tmp_path: Path) -> None:
    db_path = tmp_path / "rtabmap.db"
    _create_rtabmap_db(db_path, nodes=3, features_per_node=120)

    diagnostics = RtabmapReader().inspect(
        db_path,
        keyframe_node_ids=[1, None, None, None],
    )

    assert not diagnostics.ready
    assert any(issue.startswith("keyframe_node_coverage_below_min") for issue in diagnostics.issues)


def test_load_nodes_links_and_data_summaries(tmp_path: Path) -> None:
    db_path = tmp_path / "rtabmap.db"
    _create_rtabmap_db(db_path, nodes=3, features_per_node=120)

    reader = RtabmapReader()
    nodes = reader.load_nodes(db_path)
    links = reader.load_links(db_path)
    frames = reader.load_data_summaries(db_path)

    assert [node.node_id for node in nodes] == [1, 2, 3]
    assert nodes[0].pose[0][3] == 1.0
    assert [(link.from_id, link.to_id, link.link_type) for link in links] == [
        (1, 2, 0),
        (2, 3, 0),
    ]
    assert [frame.node_id for frame in frames] == [1, 2, 3]
    assert all(frame.image_bytes is not None for frame in frames)


def test_load_data_frames_reads_compressed_blobs(tmp_path: Path) -> None:
    db_path = tmp_path / "rtabmap.db"
    _create_rtabmap_db(db_path, nodes=3, features_per_node=120)

    frames = RtabmapReader().load_data_frames(db_path, limit=2)

    assert [frame.node_id for frame in frames] == [1, 2]
    assert frames[0].image_bytes == b"\xff\xd8jpg"
    assert frames[0].depth_bytes == b"\x89PNGdepth"
    assert frames[0].calibration_bytes == b"calibration"


def test_load_feature_points_uses_depth_xyz_without_descriptors(tmp_path: Path) -> None:
    db_path = tmp_path / "rtabmap.db"
    _create_rtabmap_db(db_path, nodes=3, features_per_node=120)

    features = RtabmapReader().load_feature_points(db_path, limit=3)

    assert [(feature.node_id, feature.word_id) for feature in features] == [
        (1, 1000),
        (1, 1001),
        (1, 1002),
    ]
    assert features[2].pixel_x == 2.0
    assert features[2].pixel_y == 3.0
    assert features[2].local_xyz == (0.02, 0.0, 1.0)
    assert features[2].descriptor_size == 64
    assert features[2].descriptor is None
