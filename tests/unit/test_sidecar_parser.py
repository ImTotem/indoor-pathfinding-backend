"""SidecarParser 단위 테스트."""
from __future__ import annotations

import sqlite3

import pytest

from indoor_server.application.sidecar_parser import SidecarParser
from indoor_server.domain.poi.enums import POISource
from indoor_server.domain.scan.errors import ScanIdMismatch, SidecarSchemaMismatch
from tests.fixtures.make_sidecar import make_v4_sidecar

SCAN_ID = "22222222-2222-2222-2222-222222222222"


def test_parse_v4_success(tmp_path):
    db_path = tmp_path / "scan_metadata.db"
    make_v4_sidecar(db_path, SCAN_ID, keyframes=3)
    contents = SidecarParser().parse(db_path, SCAN_ID)

    assert contents.scan_session.id == SCAN_ID
    assert len(contents.keyframes) == 3
    assert len(contents.poi_marks) == 2
    assert len(contents.poi_photos) == 2

    track_lock = next(pm for pm in contents.poi_marks if pm.source == POISource.TRACK_LOCK)
    assert track_lock.track_id == 42

    manual = next(pm for pm in contents.poi_marks if pm.source == POISource.MANUAL)
    assert manual.track_id is None
    assert manual.label == "exit"

    # manual poi_photo — bbox NULL, class_name='manual'
    manual_photo = next(
        pp for pp in contents.poi_photos if pp.poi_mark_id == manual.id
    )
    assert manual_photo.bbox_x is None
    assert manual_photo.class_name == "manual"


def test_parse_v4_uuid_case_insensitive_and_normalized(tmp_path):
    db_path = tmp_path / "scan_metadata.db"
    make_v4_sidecar(db_path, SCAN_ID.upper(), keyframes=2)
    contents = SidecarParser().parse(db_path, SCAN_ID.lower())

    assert contents.scan_session.id == SCAN_ID.lower()
    assert len(contents.keyframes) == 2
    assert all(kf.scan_id == SCAN_ID.lower() for kf in contents.keyframes)
    assert all(pm.scan_id == SCAN_ID.lower() for pm in contents.poi_marks)


def test_wrong_user_version(tmp_path):
    db_path = tmp_path / "scan_metadata.db"
    make_v4_sidecar(db_path, SCAN_ID)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA user_version = 3")
    conn.commit()
    conn.close()
    with pytest.raises(SidecarSchemaMismatch) as exc_info:
        SidecarParser().parse(db_path, SCAN_ID)
    assert exc_info.value.detail["got_user_version"] == 3


def test_missing_column(tmp_path):
    db_path = tmp_path / "scan_metadata.db"
    make_v4_sidecar(db_path, SCAN_ID)
    conn = sqlite3.connect(str(db_path))
    # source 컬럼을 제거한 테이블로 교체
    conn.executescript("""
        CREATE TABLE poi_mark_bad AS SELECT id, scan_id, keyframe_seq, created_at,
            pose_matrix, tx, ty, tz, track_id, label FROM poi_mark;
        DROP TABLE poi_mark;
        ALTER TABLE poi_mark_bad RENAME TO poi_mark;
        PRAGMA user_version = 4;
    """)
    conn.commit()
    conn.close()
    with pytest.raises(SidecarSchemaMismatch) as exc_info:
        SidecarParser().parse(db_path, SCAN_ID)
    assert "poi_mark.source" in exc_info.value.detail.get("missing_columns", [])


def test_scan_id_mismatch(tmp_path):
    db_path = tmp_path / "scan_metadata.db"
    make_v4_sidecar(db_path, SCAN_ID)
    with pytest.raises(ScanIdMismatch):
        SidecarParser().parse(db_path, "wrong-scan-id")


# ── Sprint 49: v5 schema (image_blob 컬럼 추가) ────────────────────────────


def _make_v5_sidecar(db_path, scan_id: str) -> None:
    """v4 fixture 위에 image_blob 컬럼 추가 + user_version=5."""
    make_v4_sidecar(db_path, scan_id)
    conn = sqlite3.connect(str(db_path))
    conn.execute("ALTER TABLE poi_photo ADD COLUMN image_blob BLOB")
    conn.execute(
        "UPDATE poi_photo SET image_blob = ? WHERE id = 1",
        (b"fake-image-bytes",),
    )
    conn.execute("PRAGMA user_version = 5")
    conn.commit()
    conn.close()


def test_parse_v5_success(tmp_path):
    """Sprint 49: PRAGMA user_version=5 + poi_photo.image_blob 컬럼 → 정상 파싱."""
    db_path = tmp_path / "scan_metadata.db"
    _make_v5_sidecar(db_path, SCAN_ID)
    contents = SidecarParser().parse(db_path, SCAN_ID)
    assert contents.scan_session.id == SCAN_ID
    assert contents.poi_photos[0].image_blob == b"fake-image-bytes"


def test_get_user_version(tmp_path):
    """SidecarParser.get_user_version 은 sidecar 만 열고 PRAGMA 반환."""
    db_path = tmp_path / "scan_metadata.db"
    make_v4_sidecar(db_path, SCAN_ID)
    assert SidecarParser().get_user_version(db_path) == 4

    db_path_v5 = tmp_path / "scan_metadata_v5.db"
    _make_v5_sidecar(db_path_v5, SCAN_ID)
    assert SidecarParser().get_user_version(db_path_v5) == 5


def test_v5_missing_image_blob_column_rejected(tmp_path):
    """v5 user_version 인데 image_blob 컬럼이 없으면 SidecarSchemaMismatch."""
    db_path = tmp_path / "scan_metadata.db"
    make_v4_sidecar(db_path, SCAN_ID)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA user_version = 5")
    conn.commit()
    conn.close()
    with pytest.raises(SidecarSchemaMismatch) as exc_info:
        SidecarParser().parse(db_path, SCAN_ID)
    assert "poi_photo.image_blob" in exc_info.value.detail.get("missing_columns", [])
