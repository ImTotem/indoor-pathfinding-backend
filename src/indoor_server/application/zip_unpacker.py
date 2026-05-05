"""ZIP 검증 + 전개. Zip Slip / symlink / DoS 방어 포함."""
from __future__ import annotations

import hashlib
import logging
import shutil
import zipfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from indoor_server.domain.scan.errors import MissingRequiredFile, ZipStructureInvalid

logger = logging.getLogger(__name__)

# Sprint 67/74: v7 raw_video_recording 은 scan.mp4 + poses.bin 을 추가로 보내지만
# main build 는 여전히 rtabmap.db 기반이다. video/poses 는 dense evidence 및 quality report
# 보조 입력으로만 사용한다.
REQUIRED_FILES = ("scan_metadata.db", "rtabmap.db")
MAX_ENTRY_SIZE = 4 * 1024 * 1024 * 1024  # 개별 엔트리 상한 4 GiB


class UnpackedScan(BaseModel):
    model_config = ConfigDict(frozen=True)

    scan_id: str
    rtabmap_db_path: Path
    sidecar_db_path: Path
    keyframe_paths: list[Path]
    payload_sha256: str
    # Sprint 49 (Codex BLOCKER 4/5): manifest.json 경로 — 존재하면 v5+,
    # 없으면 legacy v4. 호출자(IngestService) 가 SidecarParser.get_user_version
    # 과 함께 enforce_version_contract 로 4 분기 검증.
    manifest_path: Path | None = None
    # Sprint 67 v7: HEVC video + binary poses. 둘 다 있으면 dense evidence 입력으로 사용.
    video_path: Path | None = None
    poses_path: Path | None = None


class ZipUnpacker:
    def unpack(self, zip_path: Path, expected_scan_id: str, dest_dir: Path) -> UnpackedScan:
        """
        zip_path를 검증하고 dest_dir에 전개한다.

        1. 단일 루트 디렉터리가 {expected_scan_id}/ 인지 확인
        2. rtabmap.db / scan_metadata.db 존재 확인
        3. Zip Slip / symlink 방어
        4. sha256(zip payload) 계산
        5. dest_dir 에 전개
        """
        payload_sha256 = _sha256_of_file(zip_path)

        if not zipfile.is_zipfile(zip_path):
            raise ZipStructureInvalid("업로드 파일이 유효한 ZIP 형식이 아닙니다.")

        with zipfile.ZipFile(zip_path, "r") as zf:
            entries = zf.infolist()
            _validate_root(entries, expected_scan_id)
            # ZIP 내 실제 폴더명(대소문자 원본)을 구한다
            zip_prefix = next(
                e.filename.split("/")[0] for e in entries if e.filename
            )
            _validate_required_files(entries, zip_prefix)
            _validate_entries(entries, zip_prefix)
            dest_dir.mkdir(parents=True, exist_ok=True)
            _extract_safe(zf, entries, zip_prefix, dest_dir, expected_scan_id)

        scan_root = dest_dir / expected_scan_id
        sidecar_db = scan_root / "scan_metadata.db"

        rtabmap_db = scan_root / "rtabmap.db"

        # Sprint 67 v7: scan.mp4 + poses.bin 검출.
        video_candidate = scan_root / "scan.mp4"
        poses_candidate = scan_root / "poses.bin"
        video_path: Path | None = (
            video_candidate if video_candidate.exists() else None
        )
        poses_path: Path | None = (
            poses_candidate if poses_candidate.exists() else None
        )

        keyframes_dir = scan_root / "keyframes"
        keyframe_paths = sorted(keyframes_dir.glob("*")) if keyframes_dir.exists() else []
        manifest_path: Path | None = scan_root / "manifest.json"
        if manifest_path is not None and not manifest_path.exists():
            manifest_path = None

        return UnpackedScan(
            scan_id=expected_scan_id,
            rtabmap_db_path=rtabmap_db,
            sidecar_db_path=sidecar_db,
            keyframe_paths=keyframe_paths,
            payload_sha256=payload_sha256,
            manifest_path=manifest_path,
            video_path=video_path,
            poses_path=poses_path,
        )

    def cleanup(self, dest_dir: Path) -> None:
        if dest_dir.exists():
            shutil.rmtree(dest_dir, ignore_errors=True)


# ── helpers ──────────────────────────────────────────────────────────────────


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 18), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_root(entries: list[zipfile.ZipInfo], scan_id: str) -> None:
    roots = {e.filename.split("/")[0] for e in entries if e.filename}
    roots_lower = {r.lower() for r in roots}
    if roots_lower != {scan_id.lower()}:
        raise ZipStructureInvalid(
            f"ZIP 루트 디렉터리가 '{scan_id}' 단일이어야 합니다. 발견: {roots}"
        )


def _validate_required_files(entries: list[zipfile.ZipInfo], scan_id: str) -> None:
    names = {e.filename for e in entries}
    for req in REQUIRED_FILES:
        expected = f"{scan_id}/{req}"
        if expected not in names:
            raise MissingRequiredFile(f"필수 파일 누락: {req}")


def _validate_entries(entries: list[zipfile.ZipInfo], scan_id: str) -> None:
    """Zip Slip / symlink 검사."""
    for entry in entries:
        # symlink 거부: external_attr 상위 16비트가 Unix permission. 0xA000 = symlink.
        if (entry.external_attr >> 16) & 0xFFFF == 0xA000:
            raise ZipStructureInvalid(f"심볼릭 링크 엔트리 거부: {entry.filename}")

        # 정규화 후 scan_id/ 범위 밖으로 탈출하는지 확인
        normalized = Path(entry.filename).as_posix()
        if ".." in normalized.split("/"):
            raise ZipStructureInvalid(f"경로 탈출 엔트리 거부: {entry.filename}")

        if entry.file_size > MAX_ENTRY_SIZE:
            raise ZipStructureInvalid(f"단일 엔트리 크기 초과: {entry.filename}")


def _extract_safe(
    zf: zipfile.ZipFile,
    entries: list[zipfile.ZipInfo],
    zip_prefix: str,
    dest_dir: Path,
    canonical_id: str | None = None,
) -> None:
    """ZIP 엔트리를 안전하게 추출한다.

    zip_prefix: ZIP 내 실제 루트 폴더명 (대소문자 원본).
    canonical_id: 추출 후 폴더 이름 (소문자 정규화). None 이면 zip_prefix 그대로 사용.
    """
    canonical = canonical_id.lower() if canonical_id else zip_prefix.lower()
    for entry in entries:
        # ZIP 내 경로를 canonical_id 기준으로 재작성
        rel = entry.filename
        if zip_prefix and rel.startswith(zip_prefix + "/"):
            rel = canonical + "/" + rel[len(zip_prefix) + 1:]
        elif rel == zip_prefix:
            rel = canonical

        target = (dest_dir / rel).resolve()
        safe_base = (dest_dir / canonical).resolve()
        # dest_dir 직하 canonical 폴더 자체는 허용
        if target != dest_dir.resolve() / canonical and not str(target).startswith(
            str(safe_base) + "/"
        ):
            raise ZipStructureInvalid(f"경로 탈출 감지: {entry.filename}")

        if entry.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(entry) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
