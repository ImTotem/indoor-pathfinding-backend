"""Sprint 65 Phase 4a — v6 sidecar 파싱 단위 테스트."""
from __future__ import annotations

from pathlib import Path

from tests.fixtures.make_sidecar import make_v6_sidecar

from indoor_server.application.sidecar_parser import SidecarParser


def test_v6_sidecar_parses_with_interfloor_marks(tmp_path: Path) -> None:
    db = tmp_path / "v6.db"
    scan_id = "12345678-1234-1234-1234-123456789abc"
    make_v6_sidecar(
        db,
        scan_id,
        keyframes=3,
        interfloor=[
            (1, "stairs", "ST-A"),
            (2, "elevator", "EV-1"),
        ],
    )
    parser = SidecarParser()
    contents = parser.parse(db, scan_id)

    assert contents.scan_session.id == scan_id
    assert len(contents.keyframes) == 3
    # Sprint 65 v6: yolo_detection 빈 리스트
    assert contents.yolo_detections == []
    assert len(contents.interfloor_marks) == 2

    stairs = next(im for im in contents.interfloor_marks if im.connector_type == "stairs")
    assert stairs.prefix == "ST-A"
    elevator = next(im for im in contents.interfloor_marks if im.connector_type == "elevator")
    assert elevator.prefix == "EV-1"

    # POI: 수동 단일 경로. track_id None, bbox None.
    assert len(contents.poi_marks) == 1
    assert contents.poi_marks[0].track_id is None
    assert contents.poi_marks[0].source.value == "manual"
    assert len(contents.poi_photos) == 1
    photo = contents.poi_photos[0]
    assert photo.bbox_x is None and photo.bbox_y is None


def test_v6_sidecar_user_version_is_6(tmp_path: Path) -> None:
    db = tmp_path / "v6.db"
    make_v6_sidecar(db, "12345678-1234-1234-1234-123456789abc", keyframes=1)
    assert SidecarParser().get_user_version(db) == 6


def test_v6_sidecar_without_interfloor(tmp_path: Path) -> None:
    """interfloor_mark 없는 scan 도 정상 파싱."""
    db = tmp_path / "v6.db"
    scan_id = "12345678-1234-1234-1234-123456789abc"
    make_v6_sidecar(db, scan_id, keyframes=2)
    contents = SidecarParser().parse(db, scan_id)
    assert contents.interfloor_marks == []
