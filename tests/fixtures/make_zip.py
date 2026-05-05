"""테스트용 zip 파일 생성 헬퍼."""
from __future__ import annotations

import zipfile
from pathlib import Path

from tests.fixtures.make_sidecar import make_v4_sidecar


def make_scan_zip(dest: Path, scan_id: str, *, keyframes: int = 3) -> Path:
    """
    {scan_id}/rtabmap.db + scan_metadata.db + keyframes/{i}.jpg 구조의 zip을 dest에 생성.
    반환값: zip 파일 경로.
    """
    zip_path = dest / f"{scan_id}.zip"
    dest.mkdir(parents=True, exist_ok=True)

    sidecar_tmp = dest / "scan_metadata.db"
    make_v4_sidecar(sidecar_tmp, scan_id, keyframes=keyframes)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(sidecar_tmp, f"{scan_id}/scan_metadata.db")
        # rtabmap.db — 빈 파일로 대체
        zf.writestr(f"{scan_id}/rtabmap.db", b"RTABMap mock")
        # keyframe 이미지
        for i in range(1, keyframes + 1):
            zf.writestr(f"{scan_id}/keyframes/{i:06d}.jpg", b"\xff\xd8\xff\xe0mock")

    sidecar_tmp.unlink(missing_ok=True)
    return zip_path
