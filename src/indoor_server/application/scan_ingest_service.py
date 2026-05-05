"""ScanIngestService — 핵심 유스케이스."""
from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.application.scan_manifest import (
    ManifestStructureInvalid,
    ScanManifest,
    enforce_version_contract,
    parse_manifest_file,
)
from indoor_server.application.sidecar_parser import SidecarParser
from indoor_server.application.zip_unpacker import UnpackedScan, ZipUnpacker
from indoor_server.config import settings
from indoor_server.domain.building.enums import BuildState
from indoor_server.domain.scan.errors import MissingRequiredFile, ScanConflict
from indoor_server.domain.scan.models import ScanIngestResult
from indoor_server.infrastructure.db.repositories.build_job_repo import BuildJobRepository
from indoor_server.infrastructure.db.repositories.scan_ingest_repo import (
    ScanIngestRepository,
)
from indoor_server.infrastructure.jobs.build_enqueuer import BuildEnqueuer
from indoor_server.infrastructure.storage.protocol import ScanFileStore  # F-2 수정: Protocol 사용

logger = logging.getLogger(__name__)


class ScanIngestService:
    def __init__(
        self,
        unpacker: ZipUnpacker,
        parser: SidecarParser,
        store: ScanFileStore,  # F-2 수정: Protocol 타입
        session: AsyncSession,
        build_enqueuer: BuildEnqueuer | None = None,  # W-1: auto_enqueue 연동
    ) -> None:
        self._unpacker = unpacker
        self._parser = parser
        self._store = store
        self._repo = ScanIngestRepository(session)
        self._session = session
        self._build_enqueuer = build_enqueuer

    async def ingest(
        self,
        *,
        zip_path: Path,
        expected_scan_id: str,
        device_info: str | None,
        force: bool,
        tmp_dir: Path,
    ) -> ScanIngestResult:
        """
        검증 순서:
          1. zip 전개 + sha256 계산
          2. sidecar 파싱 + 스키마 검증  (W-4: asyncio.to_thread로 블로킹 IO 분리)
          3. 기존 레코드 충돌 판단
          4. DB 트랜잭션 커밋 → 파일 커밋 (F-1 수정: DB 먼저, 파일 이동 나중)
        """
        request_id = uuid.uuid4().hex[:8]
        logger.info(
            "ingest start request_id=%s scan_id=%s force=%s",
            request_id, expected_scan_id, force,
        )

        unpack_dest = tmp_dir / request_id
        try:
            unpacked = self._unpacker.unpack(zip_path, expected_scan_id, unpack_dest)
        except Exception:
            self._unpacker.cleanup(unpack_dest)
            raise

        try:
            # Sprint 49 (Codex BLOCKER 4): manifest + sidecar version 4-way 검증.
            sidecar_user_version = await asyncio.to_thread(
                self._parser.get_user_version, unpacked.sidecar_db_path
            )
            manifest = None
            if unpacked.manifest_path is not None:
                try:
                    manifest = await asyncio.to_thread(
                        parse_manifest_file,
                        unpacked.manifest_path,
                        expected_scan_id=expected_scan_id,
                    )
                except ManifestStructureInvalid:
                    self._unpacker.cleanup(unpack_dest)
                    raise
            enforce_version_contract(
                manifest=manifest,
                sidecar_user_version=sidecar_user_version,
            )

            # Sprint 67 — mode 별 필수 파일 검증.
            _enforce_mode_specific_files(manifest=manifest, unpacked=unpacked)

            # W-4 수정: sync sqlite3 호출을 asyncio.to_thread로 분리
            contents = await asyncio.to_thread(
                self._parser.parse, unpacked.sidecar_db_path, expected_scan_id
            )
        except Exception:
            self._unpacker.cleanup(unpack_dest)
            raise

        existing = await self._repo.find_existing(expected_scan_id)
        replace = self._resolve_replace(existing, unpacked.payload_sha256, force, expected_scan_id)

        if existing and existing.payload_sha256 == unpacked.payload_sha256 and not force:
            # 동일 payload — idempotent 200
            self._unpacker.cleanup(unpack_dest)
            sp = f"scans/{expected_scan_id}"
            # W-1: idempotent 경로에서도 auto_enqueue 적용 (이미 완료된 build는 재enqueue 안 함)
            # find_existing()이 암시적 트랜잭션을 시작했으므로 rollback 후 새 트랜잭션에서 enqueue
            await self._session.rollback()
            existing_build_job_id = await self._maybe_enqueue(expected_scan_id)
            result = _build_result(contents, existing.payload_sha256, sp, "ingested")
            return result.model_copy(update={"build_job_id": existing_build_job_id})

        # F-1 수정: DB 트랜잭션 먼저 커밋, 성공 후에 파일 이동
        source_dir = unpack_dest / expected_scan_id
        storage_path = f"scans/{expected_scan_id}"

        # find_existing이 이미 SELECT로 트랜잭션을 implicit-begin시킨 상태이므로
        # session.begin() context manager를 다시 열면 InvalidRequestError가 발생한다.
        # 명시적 commit/rollback 패턴으로 처리.
        try:
            await self._repo.ingest(
                contents=contents,
                storage_path=storage_path,
                payload_sha256=unpacked.payload_sha256,
                device_info=device_info,
                replace=replace,
            )
            await self._session.commit()
        except Exception:
            # DB 실패 — tmp 정리 후 재raise
            await self._session.rollback()
            self._unpacker.cleanup(unpack_dest)
            raise

        # DB 커밋 성공 → 파일 이동
        try:
            storage_path = await self._store.commit(
                scan_id=expected_scan_id,
                source_dir=source_dir,
                replace=replace,
            )
        except Exception:
            # 파일 이동 실패 — 보상: DB에서 ingest row 삭제 후 500
            logger.error(
                "file commit failed after DB commit; compensating DB rollback scan_id=%s",
                expected_scan_id,
            )
            try:
                async with self._session.begin():
                    await self._repo.delete(expected_scan_id)
            except Exception:
                logger.error(
                    "compensating DB delete also failed scan_id=%s — manual cleanup required",
                    expected_scan_id,
                )
            self._unpacker.cleanup(unpack_dest)
            raise

        state = "replaced" if replace and existing else "ingested"
        logger.info(
            "ingest done request_id=%s scan_id=%s state=%s sha256=%s",
            request_id, expected_scan_id, state, unpacked.payload_sha256,
        )

        # W-1: DB 커밋 성공 + 파일 이동 완료 후 auto_enqueue
        build_job_id = await self._maybe_enqueue(expected_scan_id)
        result = _build_result(contents, unpacked.payload_sha256, storage_path, state)
        return result.model_copy(update={"build_job_id": build_job_id})

    # ── helpers ───────────────────────────────────────────────────────────────

    async def _maybe_enqueue(self, scan_id: str) -> str | None:
        """
        W-1: build_auto_enqueue=True면 BuildEnqueuer.enqueue()를 호출한다.
        - enqueuer가 없거나 build_auto_enqueue=False면 None 반환.
        - idempotent 경로: 이미 succeeded/failed/cancelled 상태면 재enqueue 안 하고
          기존 job_id를 반환. pending/running 상태면 기존 job_id 반환.
        설계 §3.4 참조.
        """
        if not settings.build_auto_enqueue or self._build_enqueuer is None:
            return None

        async with self._session.begin():
            build_repo = BuildJobRepository(self._session)
            latest = await build_repo.get_latest(scan_id)

            # 이미 완료/진행 중인 job이 있으면 재enqueue 안 함
            if latest is not None and latest.state in (
                BuildState.SUCCEEDED,
                BuildState.PENDING,
                BuildState.RUNNING,
            ):
                logger.info(
                    "auto_enqueue skipped scan_id=%s existing_job=%s state=%s",
                    scan_id, latest.build_job_id, latest.state,
                )
                return str(latest.build_job_id)

            job = await self._build_enqueuer.enqueue(scan_id)
            return str(job.build_job_id)

    def _resolve_replace(
        self,
        existing: object | None,
        incoming_sha256: str,
        force: bool,
        scan_id: str,
    ) -> bool:
        from indoor_server.domain.scan.models import ExistingScan

        if existing is None:
            return False
        assert isinstance(existing, ExistingScan)
        if force:
            return True
        if existing.payload_sha256 == incoming_sha256:
            return False
        raise ScanConflict(
            "scan_id already exists with different payload. use ?force=true to overwrite.",
            detail={
                "existing_sha256": existing.payload_sha256,
                "incoming_sha256": incoming_sha256,
            },
        )


def _enforce_mode_specific_files(
    *,
    manifest: ScanManifest | None,
    unpacked: UnpackedScan,
) -> None:
    """Sprint 67 — manifest.mode 에 따라 zip 안의 필수 파일 검증.

    - raw_video_recording (v7): rtabmap.db + scan.mp4 + poses.bin 모두 필요.
      rtabmap.db 는 main build 입력이고 video/poses 는 dense evidence / quality report 입력.
    - raw_arkit_recording (v6) 또는 None (legacy v4/v5): rtabmap.db 필요.
    """
    is_video_mode = manifest is not None and manifest.is_video_mode
    if is_video_mode:
        missing: list[str] = []
        if unpacked.video_path is None:
            missing.append("scan.mp4")
        if unpacked.poses_path is None:
            missing.append("poses.bin")
        if missing:
            raise MissingRequiredFile(
                f"raw_video_recording 모드 필수 파일 누락: {', '.join(missing)}"
            )


def _build_result(
    contents: object, sha256: str, storage_path: str, state: str
) -> ScanIngestResult:
    from indoor_server.domain.poi.enums import POISource
    from indoor_server.domain.scan.models import SidecarContents

    assert isinstance(contents, SidecarContents)
    track_lock_count = sum(1 for pm in contents.poi_marks if pm.source == POISource.TRACK_LOCK)
    manual_count = sum(1 for pm in contents.poi_marks if pm.source == POISource.MANUAL)

    return ScanIngestResult(
        scan_id=contents.scan_session.id,
        state=state,
        keyframe_count=len(contents.keyframes),
        poi_marks_track_lock=track_lock_count,
        poi_marks_manual=manual_count,
        poi_photo_count=len(contents.poi_photos),
        branch_mark_count=len(contents.branch_marks),
        yolo_detection_count=len(contents.yolo_detections),
        storage_path=storage_path,
        payload_sha256=sha256,
    )
