"""ZipUnpacker 단위 테스트."""
from __future__ import annotations

import zipfile

import pytest

from indoor_server.application.zip_unpacker import ZipUnpacker
from indoor_server.domain.scan.errors import MissingRequiredFile, ZipStructureInvalid

SCAN_ID = "11111111-1111-1111-1111-111111111111"


def _make_minimal_zip(tmp_path, scan_id: str, *, missing: str | None = None) -> object:
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        files = ["rtabmap.db", "scan_metadata.db", "keyframes/000001.jpg"]
        if missing:
            files = [f for f in files if missing not in f]
        for name in files:
            zf.writestr(f"{scan_id}/{name}", b"mock")
    return zip_path


def test_unpack_success(tmp_path):
    zip_path = _make_minimal_zip(tmp_path, SCAN_ID)
    dest = tmp_path / "dest"
    result = ZipUnpacker().unpack(zip_path, SCAN_ID, dest)
    assert result.scan_id == SCAN_ID
    assert result.rtabmap_db_path.exists()
    assert result.sidecar_db_path.exists()
    assert len(result.keyframe_paths) == 1
    assert len(result.payload_sha256) == 64


def test_wrong_root_dir(tmp_path):
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("other-id/rtabmap.db", b"mock")
    with pytest.raises(ZipStructureInvalid):
        ZipUnpacker().unpack(zip_path, SCAN_ID, tmp_path / "dest")


def test_missing_required_file(tmp_path):
    zip_path = _make_minimal_zip(tmp_path, SCAN_ID, missing="scan_metadata.db")
    with pytest.raises(MissingRequiredFile):
        ZipUnpacker().unpack(zip_path, SCAN_ID, tmp_path / "dest")


def test_zip_slip_rejected(tmp_path):
    zip_path = tmp_path / "slip.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{SCAN_ID}/../../evil.txt", b"evil")
        zf.writestr(f"{SCAN_ID}/rtabmap.db", b"mock")
        zf.writestr(f"{SCAN_ID}/scan_metadata.db", b"mock")
    with pytest.raises(ZipStructureInvalid):
        ZipUnpacker().unpack(zip_path, SCAN_ID, tmp_path / "dest")
