"""V1 scan chunk/merge/process compatibility wrapper.

The old V1 names are retained, but the implementation accepts only the new zip
scan archive and routes it through `ScanIngestService` with auto enqueue.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import sqlalchemy as sa
from fastapi import UploadFile
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.application.api_v1.building_floor_service import BuildingFloorService
from indoor_server.application.api_v1.errors import V1ServiceError
from indoor_server.application.building.multiscan_rtabmap_merge import (
    MultiScanReprocessParams,
    MultiScanRtabmapMergeError,
    MultiScanRtabmapReprocessRunner,
    SourceRtabmapScan,
)
from indoor_server.application.scan_ingest_service import ScanIngestService
from indoor_server.application.sidecar_parser import SidecarParser
from indoor_server.application.zip_unpacker import ZipUnpacker
from indoor_server.config import settings
from indoor_server.domain.building.enums import BuildState
from indoor_server.infrastructure.db import tables as t
from indoor_server.infrastructure.db.repositories.build_job_repo import BuildJobRepository
from indoor_server.infrastructure.jobs.build_enqueuer import BuildEnqueuer
from indoor_server.infrastructure.storage.local_fs import LocalFileStorage
from indoor_server.interfaces.api.v1_schemas import (
    MergedScanResponse,
    ProcessingStatusResponse,
    ScanChunkResponse,
)

logger = logging.getLogger(__name__)


class ScanCompatService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upload_archive(
        self,
        *,
        floor_id: UUID,
        upload: UploadFile,
        scan_id: str | None,
        device_info: str | None,
        force: bool,
    ) -> ScanChunkResponse:
        if not _looks_like_zip(upload.filename):
            raise V1ServiceError(
                status_code=400,
                code="ZIP_ARCHIVE_REQUIRED",
                message="V1 chunk wrapper accepts only zip scan archives.",
                detail={"filename": upload.filename},
            )
        await BuildingFloorService(self._session).get_floor(floor_id)

        effective_scan_id = scan_id or str(uuid4())
        tmp_dir = settings.tmp_root / uuid4().hex
        tmp_dir.mkdir(parents=True, exist_ok=True)
        zip_path = tmp_dir / f"{effective_scan_id}.zip"

        try:
            file_size = await _save_upload(upload, zip_path)
            enqueuer = BuildEnqueuer(self._session) if settings.build_auto_enqueue else None
            result = await ScanIngestService(
                unpacker=ZipUnpacker(),
                parser=SidecarParser(),
                store=LocalFileStorage(settings.storage_root),
                session=self._session,
                build_enqueuer=enqueuer,
            ).ingest(
                zip_path=zip_path,
                expected_scan_id=effective_scan_id,
                device_info=device_info,
                force=force,
                tmp_dir=tmp_dir,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        row = await self._activate_floor_scan(
            floor_id=floor_id,
            scan_id=UUID(result.scan_id),
            file_name=upload.filename,
            file_size=file_size,
            status_value="UPLOADED",
        )
        await self._session.commit()
        return _chunk_response(row)

    async def list_chunks(self, floor_id: UUID) -> list[ScanChunkResponse]:
        await BuildingFloorService(self._session).get_floor(floor_id)
        rows = (
            await self._session.execute(
                sa.select(t.floor_scan)
                .where(t.floor_scan.c.floor_id == str(floor_id))
                .order_by(t.floor_scan.c.upload_order.asc(), t.floor_scan.c.created_at.asc())
            )
        ).fetchall()
        return [_chunk_response(row) for row in rows]

    async def delete_chunk(self, floor_id: UUID, chunk_id: UUID) -> None:
        result = await self._session.execute(
            sa.delete(t.floor_scan).where(
                t.floor_scan.c.floor_id == str(floor_id),
                t.floor_scan.c.floor_scan_id == str(chunk_id),
            )
        )
        if int(getattr(result, "rowcount", 0)) == 0:
            raise V1ServiceError(404, "CHUNK_NOT_FOUND", "scan chunk not found")
        await self._session.commit()

    async def merge(self, floor_id: UUID, chunk_ids: list[UUID]) -> MergedScanResponse:
        sources = await self._collect_merge_sources(floor_id=floor_id, chunk_ids=chunk_ids)
        if not sources:
            raise V1ServiceError(404, "CHUNK_NOT_FOUND", "scan chunk not found")

        # 단일 청크: 기존 active flag flip 만 (실 merge 불필요).
        if len(sources) == 1:
            row = sources[0]
            await self._session.execute(
                sa.update(t.floor_scan)
                .where(t.floor_scan.c.floor_id == str(floor_id))
                .values(active=False)
            )
            await self._session.execute(
                sa.update(t.floor_scan)
                .where(t.floor_scan.c.floor_scan_id == row.floor_scan_id)
                .values(active=True, status="MERGED")
            )
            await self._session.commit()
            return MergedScanResponse(
                floor_id=floor_id,
                active_scan_id=UUID(str(row.scan_id)),
                status=await self._merge_status_for_scan(str(row.scan_id)),
            )

        # 다중 청크: rtabmap-reprocess 로 실 RTABMap multi-scan 통합.
        return await self._real_multiscan_merge(floor_id=floor_id, sources=sources)

    async def _collect_merge_sources(
        self, *, floor_id: UUID, chunk_ids: list[UUID]
    ) -> list[Any]:
        filters = [t.floor_scan.c.floor_id == str(floor_id)]
        if chunk_ids:
            chunk_id_strs = [str(c) for c in chunk_ids]
            filters.append(
                sa.or_(
                    t.floor_scan.c.floor_scan_id.in_(chunk_id_strs),
                    t.floor_scan.c.scan_id.in_(chunk_id_strs),
                )
            )
        stmt = (
            sa.select(
                t.floor_scan.c.floor_scan_id,
                t.floor_scan.c.scan_id,
                t.floor_scan.c.upload_order,
                t.floor_scan.c.created_at,
                t.scan_ingest.c.storage_path,
            )
            .join(t.scan_ingest, t.scan_ingest.c.scan_id == t.floor_scan.c.scan_id)
            .where(*filters)
            .order_by(
                t.floor_scan.c.upload_order.asc(), t.floor_scan.c.created_at.asc()
            )
        )
        return list((await self._session.execute(stmt)).fetchall())

    async def _real_multiscan_merge(
        self, *, floor_id: UUID, sources: list[Any]
    ) -> MergedScanResponse:
        runner = MultiScanRtabmapReprocessRunner()
        if not runner.is_available():
            raise V1ServiceError(
                status_code=503,
                code="RTABMAP_REPROCESS_UNAVAILABLE",
                message="rtabmap-reprocess binary not available.",
            )

        merged_scan_id = uuid4()
        storage_root = Path(settings.storage_root)
        rel_storage_path = f"scans/{merged_scan_id}"
        merged_dir = storage_root / rel_storage_path
        merged_db_path = merged_dir / "rtabmap.db"
        work_dir = merged_dir / "_merge_work"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: 각 source 단독 reprocess.
        # rtabmap-reprocess `-a` (append) 모드는 첫 .db 를 init 으로 두고 둘째부터만
        # reprocess → 첫 source 노드에 SIFT feature/depth back-projection 안 됨.
        # 따라서 각 source 를 단독으로 먼저 돌려 Feature/Word 를 채운 뒤 -a merge 한다.
        single_reprocessed: list[SourceRtabmapScan] = []
        for index, row in enumerate(sources):
            source_db = storage_root / row.storage_path / "rtabmap.db"
            if not source_db.exists():
                raise V1ServiceError(
                    status_code=500,
                    code="SOURCE_DB_MISSING",
                    message=f"source rtabmap.db not found at {source_db}",
                )
            stage1_out = work_dir / f"stage1_{index:02d}_{row.scan_id}.db"
            if stage1_out.exists():
                stage1_out.unlink()
            logger.info(
                "rtabmap single reprocess start (stage1) scan_id=%s input=%s output=%s",
                row.scan_id,
                source_db,
                stage1_out,
            )
            try:
                await _run_rtabmap_reprocess_single(
                    binary_path=runner.binary_path or "rtabmap-reprocess",
                    input_db=source_db,
                    output_db=stage1_out,
                    timeout_s=600.0,
                )
            except _ReprocessFailed as e:
                raise V1ServiceError(
                    status_code=503,
                    code="RTABMAP_REPROCESS_FAILED",
                    message=f"single reprocess failed for scan {row.scan_id}: {e}",
                ) from e
            single_reprocessed.append(
                SourceRtabmapScan(scan_id=str(row.scan_id), db_path=stage1_out)
            )

        # Stage 2: 단독 reprocess 결과들을 -a 로 multi-scan merge.
        logger.info(
            "rtabmap multi-scan merge start (stage2) floor_id=%s sources=%d "
            "merged_scan_id=%s",
            floor_id,
            len(single_reprocessed),
            merged_scan_id,
        )
        try:
            result = await runner.run(
                sources=single_reprocessed,
                output_db=merged_db_path,
                work_dir=work_dir / "stage2",
                params=MultiScanReprocessParams(),
                timeout_s=900.0,
            )
        except MultiScanRtabmapMergeError as e:
            raise V1ServiceError(
                status_code=503,
                code="RTABMAP_REPROCESS_FAILED",
                message=str(e),
            ) from e

        payload_sha256 = await asyncio.get_running_loop().run_in_executor(
            None, _sha256_file, merged_db_path
        )
        file_size = merged_db_path.stat().st_size
        shutil.rmtree(work_dir, ignore_errors=True)

        await self._session.execute(
            sa.insert(t.scan_ingest).values(
                scan_id=str(merged_scan_id),
                payload_sha256=payload_sha256,
                storage_path=rel_storage_path,
                device_info=None,
            )
        )
        await self._session.execute(
            sa.update(t.floor_scan)
            .where(t.floor_scan.c.floor_id == str(floor_id))
            .values(active=False)
        )
        max_order = (
            await self._session.execute(
                sa.select(sa.func.coalesce(sa.func.max(t.floor_scan.c.upload_order), 0))
                .where(t.floor_scan.c.floor_id == str(floor_id))
            )
        ).scalar_one()
        await self._session.execute(
            sa.insert(t.floor_scan).values(
                floor_id=str(floor_id),
                scan_id=str(merged_scan_id),
                file_name=f"merged_{merged_scan_id}.db",
                file_size=file_size,
                status="MERGED",
                active=True,
                upload_order=int(max_order) + 1,
            )
        )
        await self._session.commit()
        logger.info(
            "rtabmap multi-scan merge complete floor_id=%s merged_scan_id=%s "
            "duration=%.1fs nodes=%d loop_closures=%d",
            floor_id,
            merged_scan_id,
            result.duration_s,
            result.merged_node_count,
            result.loop_closure_count,
        )
        return MergedScanResponse(
            floor_id=floor_id,
            active_scan_id=merged_scan_id,
            status=await self._merge_status_for_scan(str(merged_scan_id)),
        )

    async def merge_status(self, floor_id: UUID) -> MergedScanResponse:
        active = await BuildingFloorService(self._session).get_active_scan_for_floor(floor_id)
        return MergedScanResponse(
            floor_id=floor_id,
            active_scan_id=UUID(active.scan_id) if active is not None else None,
            status=await self._merge_status_for_scan(active.scan_id) if active else "NOT_STARTED",
        )

    async def process(self, floor_id: UUID) -> ProcessingStatusResponse:
        active = await BuildingFloorService(self._session).get_active_scan_for_floor(floor_id)
        if active is None:
            raise V1ServiceError(404, "ACTIVE_SCAN_NOT_FOUND", "active scan not found")

        repo = BuildJobRepository(self._session)
        latest = await repo.get_latest(active.scan_id)
        if latest is not None and latest.state in (
            BuildState.PENDING,
            BuildState.RUNNING,
            BuildState.SUCCEEDED,
        ):
            return _processing_response(floor_id, active.scan_id, latest)

        job = await BuildEnqueuer(self._session).enqueue(active.scan_id)
        await self._session.commit()
        return _processing_response(floor_id, active.scan_id, job)

    async def process_status(self, floor_id: UUID) -> ProcessingStatusResponse:
        active = await BuildingFloorService(self._session).get_active_scan_for_floor(floor_id)
        if active is None:
            return ProcessingStatusResponse(floor_id=floor_id, scan_id=None, status="NOT_STARTED")
        latest = await BuildJobRepository(self._session).get_latest(active.scan_id)
        if latest is None:
            return ProcessingStatusResponse(
                floor_id=floor_id,
                scan_id=UUID(active.scan_id),
                status="NOT_STARTED",
            )
        return _processing_response(floor_id, active.scan_id, latest)

    async def _activate_floor_scan(
        self,
        *,
        floor_id: UUID,
        scan_id: UUID,
        file_name: str | None,
        file_size: int,
        status_value: str,
    ) -> Any:
        max_order = (
            await self._session.execute(
                sa.select(sa.func.coalesce(sa.func.max(t.floor_scan.c.upload_order), 0)).where(
                    t.floor_scan.c.floor_id == str(floor_id)
                )
            )
        ).scalar_one()
        await self._session.execute(
            sa.update(t.floor_scan)
            .where(t.floor_scan.c.floor_id == str(floor_id))
            .values(active=False)
        )
        insert_stmt = pg_insert(t.floor_scan).values(
            floor_id=str(floor_id),
            scan_id=str(scan_id),
            file_name=file_name,
            file_size=file_size,
            status=status_value,
            active=True,
            upload_order=int(max_order) + 1,
        )
        stmt = insert_stmt.on_conflict_do_update(
            constraint="uq_floor_scan_scan",
            set_={
                "file_name": file_name,
                "file_size": file_size,
                "status": status_value,
                "active": True,
            },
        ).returning(t.floor_scan)
        row = (await self._session.execute(stmt)).first()
        assert row is not None
        return row

    async def _select_chunk_for_merge(self, *, floor_id: UUID, chunk_ids: list[UUID]) -> Any:
        filters = [t.floor_scan.c.floor_id == str(floor_id)]
        if chunk_ids:
            chunk_id_strings = [str(value) for value in chunk_ids]
            filters.append(
                sa.or_(
                    t.floor_scan.c.floor_scan_id.in_(chunk_id_strings),
                    t.floor_scan.c.scan_id.in_(chunk_id_strings),
                )
            )
        row = (
            await self._session.execute(
                sa.select(t.floor_scan)
                .where(*filters)
                .order_by(t.floor_scan.c.active.desc(), t.floor_scan.c.created_at.asc())
                .limit(1)
            )
        ).first()
        if row is None:
            raise V1ServiceError(404, "CHUNK_NOT_FOUND", "scan chunk not found")
        return row

    async def _merge_status_for_scan(self, scan_id: str) -> str:
        latest = await BuildJobRepository(self._session).get_latest(scan_id)
        if latest is None:
            return "MERGED"
        if latest.state == BuildState.SUCCEEDED:
            return "COMPLETED"
        if latest.state == BuildState.FAILED:
            return "FAILED"
        if latest.state in (BuildState.PENDING, BuildState.RUNNING):
            return "PROCESSING"
        return "MERGED"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _ReprocessFailed(Exception):
    """single rtabmap-reprocess invocation failed."""


async def _run_rtabmap_reprocess_single(
    *,
    binary_path: str,
    input_db: Path,
    output_db: Path,
    timeout_s: float,
) -> None:
    """단독 reprocess: input.db 의 모든 노드에 SIFT feature + 3D back-projection 강제.

    `-default` 로 input db 의 params 를 무시하고 default 를 사용해야 클라이언트가
    `Mem/IncrementalMemory=false` 등으로 박아 보낸 db 도 정상 reprocess 된다.
    `--Vis/FeatureType 1` + `--Kp/DetectorStrategy 1` 로 SIFT 강제.
    """
    output_db.parent.mkdir(parents=True, exist_ok=True)
    if output_db.exists():
        output_db.unlink()
    cmd = [
        binary_path,
        "-default",
        "--Mem/IncrementalMemory", "true",
        "--Vis/FeatureType", "1",
        "--Kp/DetectorStrategy", "1",
        "--uwarn",
        str(input_db),
        str(output_db),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError as e:
        proc.kill()
        await proc.wait()
        raise _ReprocessFailed(f"timeout > {timeout_s}s") from e
    if proc.returncode != 0:
        stderr_text = stderr_b.decode(errors="replace") if stderr_b else ""
        raise _ReprocessFailed(
            f"exit {proc.returncode}; stderr tail: {stderr_text[-1200:]}"
        )


async def _save_upload(upload: UploadFile, dest: Path) -> int:
    total = 0
    with open(dest, "wb") as f:
        while chunk := await upload.read(1 << 18):
            total += len(chunk)
            if total > settings.max_upload_bytes:
                raise V1ServiceError(413, "PAYLOAD_TOO_LARGE", "upload too large")
            f.write(chunk)
    return total


def _looks_like_zip(filename: str | None) -> bool:
    return bool(filename and filename.lower().endswith(".zip"))


def _chunk_response(row: Any) -> ScanChunkResponse:
    return ScanChunkResponse(
        chunk_id=UUID(str(row.floor_scan_id)),
        floor_id=UUID(str(row.floor_id)),
        scan_id=UUID(str(row.scan_id)),
        file_name=str(row.file_name) if row.file_name is not None else None,
        file_size=int(row.file_size) if row.file_size is not None else None,
        status=str(row.status),
        active=bool(row.active),
        upload_order=int(row.upload_order),
        created_at=row.created_at,
    )


def _processing_response(
    floor_id: UUID,
    scan_id: str,
    job: Any,
) -> ProcessingStatusResponse:
    state = str(job.state.value if hasattr(job.state, "value") else job.state)
    mapped = {
        "pending": "PROCESSING",
        "running": "PROCESSING",
        "succeeded": "COMPLETED",
        "failed": "FAILED",
        "cancelled": "FAILED",
    }.get(state, "NOT_STARTED")
    failure = getattr(job, "failure_detail", None)
    return ProcessingStatusResponse(
        floor_id=floor_id,
        scan_id=UUID(scan_id),
        build_job_id=job.build_job_id,
        status=mapped,
        progress=getattr(job, "progress", None),
        error=str(failure) if failure is not None else None,
    )
