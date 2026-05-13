"""V1 scan chunk/process compatibility wrapper.

Backs the legacy `/floors/{id}/scans/chunks` (zip ingest), the multi-scan
merge endpoints, and the build/process endpoints.

The streaming endpoints (`/floors/{id}/scans/start` →
`/scans/{id}/frames` → `/scans/{id}/finalize`) produce READY scans. The
zip chunk endpoint produces UPLOADED scans. `merge` accepts either as a
source and produces a new merged scan (status=READY, active=true).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
import sqlite3
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


# Source scans accepted as merge inputs. OPEN scans are still receiving
# frames; MERGED scans are themselves merge results and shouldn't be
# re-merged transitively (use the underlying sources instead).
_MERGE_SOURCE_STATUSES = ("READY", "UPLOADED")

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
        """List source scans only — merge results (status='MERGED') are
        internal artifacts that drive the build/graph but are never shown to
        the user. Each source row carries `includedInMergedScan` if it was
        a source of the floor's currently active merged scan.
        """
        await BuildingFloorService(self._session).get_floor(floor_id)
        # Source scans on this floor (everything except merge-result rows).
        rows = (
            await self._session.execute(
                sa.select(t.floor_scan)
                .where(
                    t.floor_scan.c.floor_id == str(floor_id),
                    t.floor_scan.c.status != "MERGED",
                )
                .order_by(
                    t.floor_scan.c.upload_order.asc(),
                    t.floor_scan.c.created_at.asc(),
                )
            )
        ).fetchall()

        # Find the floor's currently active merged scan (if any) and the set of
        # source scan_ids it was built from. Stored in scan_ingest.device_info
        # (`{"merged_from": [...]}`) so no schema change required.
        merged_active_id: str | None = None
        included_ids: set[str] = set()
        active_merged = (
            await self._session.execute(
                sa.select(t.floor_scan.c.scan_id, t.scan_ingest.c.device_info)
                .join(
                    t.scan_ingest,
                    t.scan_ingest.c.scan_id == t.floor_scan.c.scan_id,
                )
                .where(
                    t.floor_scan.c.floor_id == str(floor_id),
                    t.floor_scan.c.status == "MERGED",
                    t.floor_scan.c.active.is_(True),
                )
                .order_by(t.floor_scan.c.created_at.desc())
                .limit(1)
            )
        ).first()
        if active_merged is not None:
            merged_active_id = str(active_merged.scan_id)
            di = active_merged.device_info or {}
            for sid in (di.get("merged_from") or []):
                included_ids.add(str(sid).lower())

        return [
            _chunk_response(
                row,
                included_in_merged_scan=(
                    UUID(merged_active_id)
                    if merged_active_id and str(row.scan_id).lower() in included_ids
                    else None
                ),
            )
            for row in rows
        ]

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

    async def merge(
        self, floor_id: UUID, chunk_ids: list[UUID]
    ) -> MergedScanResponse:
        """Merge two or more READY/UPLOADED scans on a floor into one active scan.

        - Empty `chunk_ids` → all READY/UPLOADED scans on the floor are merged.
        - Single source → promote it to active (status=READY, active=true).
        - Multi source → rtabmap-reprocess multi-scan → new merged scan_id,
          union-copy per-source keyframe_meta + poi_mark + poi_photo into
          merged scan_id with rtabmap_node_id remapped via provenance labels.
        """
        sources = await self._collect_merge_sources(
            floor_id=floor_id, chunk_ids=chunk_ids
        )
        if not sources:
            raise V1ServiceError(
                404,
                "MERGE_SOURCES_NOT_FOUND",
                "no READY or UPLOADED scans found for merge",
            )
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
                .values(active=True, status="READY")
            )
            await self._session.commit()
            await _kickoff_floor_warmup(
                floor_id=str(floor_id), scan_id=str(row.scan_id),
            )
            return MergedScanResponse(
                floor_id=floor_id,
                active_scan_id=UUID(str(row.scan_id)),
                status=await self._merge_status_for_scan(str(row.scan_id)),
            )
        return await self._real_multiscan_merge(floor_id=floor_id, sources=sources)

    async def merge_status(self, floor_id: UUID) -> MergedScanResponse:
        active = await BuildingFloorService(
            self._session
        ).get_active_scan_for_floor(floor_id)
        return MergedScanResponse(
            floor_id=floor_id,
            active_scan_id=UUID(active.scan_id) if active is not None else None,
            status=await self._merge_status_for_scan(active.scan_id)
            if active else "NOT_STARTED",
        )

    async def _collect_merge_sources(
        self, *, floor_id: UUID, chunk_ids: list[UUID]
    ) -> list[Any]:
        filters: list[Any] = [t.floor_scan.c.floor_id == str(floor_id)]
        if chunk_ids:
            ids = [str(c) for c in chunk_ids]
            filters.append(
                sa.or_(
                    t.floor_scan.c.floor_scan_id.in_(ids),
                    t.floor_scan.c.scan_id.in_(ids),
                )
            )
        else:
            filters.append(t.floor_scan.c.status.in_(_MERGE_SOURCE_STATUSES))
        stmt = (
            sa.select(
                t.floor_scan.c.floor_scan_id,
                t.floor_scan.c.scan_id,
                t.floor_scan.c.upload_order,
                t.floor_scan.c.created_at,
                t.floor_scan.c.status,
                t.scan_ingest.c.storage_path,
            )
            .join(t.scan_ingest, t.scan_ingest.c.scan_id == t.floor_scan.c.scan_id)
            .where(*filters)
            .order_by(
                t.floor_scan.c.upload_order.asc(),
                t.floor_scan.c.created_at.asc(),
            )
        )
        rows = list((await self._session.execute(stmt)).fetchall())
        # When chunk_ids are explicitly provided, reject anything other than
        # READY/UPLOADED so the caller doesn't accidentally re-merge a merged
        # output or an OPEN streaming scan still receiving frames.
        if chunk_ids:
            bad = [r for r in rows if r.status not in _MERGE_SOURCE_STATUSES]
            if bad:
                raise V1ServiceError(
                    409,
                    "MERGE_SOURCE_NOT_READY",
                    "some sources are not in READY/UPLOADED state",
                    detail={"offending": [
                        {"scan_id": str(b.scan_id), "status": str(b.status)}
                        for b in bad
                    ]},
                )
        return rows

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

        # Stage 1: per-source single reprocess to populate Word/Feature.
        # Streaming-uploaded rtabmap.db ships with Word=0/Feature=0 — only raw
        # image+depth+pose+calibration — so this stage is required before
        # multi-scan -a can match cross-session loop closures.
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
                row.scan_id, source_db, stage1_out,
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

        # Stage 2: rtabmap-reprocess -a "db1;db2;..." → merged db with
        # provenance labels preserved on each Node.label.
        stage2_db = work_dir / "stage2_unaligned.db"
        logger.info(
            "rtabmap multi-scan merge start (stage2) floor_id=%s sources=%d "
            "merged_scan_id=%s",
            floor_id, len(single_reprocessed), merged_scan_id,
        )
        try:
            result = await runner.run(
                sources=single_reprocessed,
                output_db=stage2_db,
                work_dir=work_dir / "stage2",
                params=MultiScanReprocessParams(
                    optimize_max_error=settings.multiscan_merge_optimize_max_error_m,
                ),
                timeout_s=900.0,
            )
        except MultiScanRtabmapMergeError as e:
            raise V1ServiceError(
                status_code=503,
                code="RTABMAP_REPROCESS_FAILED",
                message=str(e),
            ) from e

        # Stage 3: 2-source RANSAC SE(3) align (yaw + xy translation only) when
        # the rtabmap-reprocess internal optimizer leaves the two ARKit world
        # origins union'd. 3+ sources skip alignment (TODO: generalize).
        merged_db_path.parent.mkdir(parents=True, exist_ok=True)
        if len(single_reprocessed) == 2:
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None,
                    _align_and_patch_merged,
                    stage2_db,
                    merged_db_path,
                    single_reprocessed[0].db_path,
                    single_reprocessed[1].db_path,
                )
                logger.info(
                    "rtabmap multi-scan merge stage3 align done floor_id=%s "
                    "merged_scan_id=%s",
                    floor_id, merged_scan_id,
                )
            except _AlignFailed as e:
                logger.warning(
                    "stage3 align failed (%s) — falling back to unaligned stage2 db",
                    e,
                )
                shutil.move(str(stage2_db), str(merged_db_path))
        else:
            shutil.move(str(stage2_db), str(merged_db_path))

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
                # Provenance: list_chunks reads `merged_from` to mark which
                # source scans are included in the active merged scan.
                device_info={"merged_from": [str(r.scan_id) for r in sources]},
            )
        )

        # Stage 4 (migrated): UNION-copy keyframe_meta / poi_mark / poi_photo
        # from ALL source scans into merged_scan_id, with rtabmap_node_id
        # remapped via merged db provenance labels and keyframe_seq renumbered
        # to be globally unique within merged_scan_id.
        union_counts = await self._union_metadata_into_merged(
            merged_scan_id=str(merged_scan_id),
            source_scan_ids=[str(r.scan_id) for r in sources],
            merged_db_path=merged_db_path,
        )

        await self._session.execute(
            sa.update(t.floor_scan)
            .where(t.floor_scan.c.floor_id == str(floor_id))
            .values(active=False)
        )
        max_order = (
            await self._session.execute(
                sa.select(
                    sa.func.coalesce(sa.func.max(t.floor_scan.c.upload_order), 0)
                ).where(t.floor_scan.c.floor_id == str(floor_id))
            )
        ).scalar_one()
        await self._session.execute(
            sa.insert(t.floor_scan).values(
                floor_id=str(floor_id),
                scan_id=str(merged_scan_id),
                file_name=f"merged_{merged_scan_id}.db",
                file_size=file_size,
                status="READY",
                active=True,
                upload_order=int(max_order) + 1,
            )
        )
        await self._session.commit()
        await _kickoff_floor_warmup(
            floor_id=str(floor_id), scan_id=str(merged_scan_id),
        )
        logger.info(
            "rtabmap multi-scan merge complete floor_id=%s merged_scan_id=%s "
            "duration=%.1fs nodes=%d loop_closures=%d union_counts=%s",
            floor_id, merged_scan_id, result.duration_s,
            result.merged_node_count, result.loop_closure_count, union_counts,
        )
        return MergedScanResponse(
            floor_id=floor_id,
            active_scan_id=merged_scan_id,
            status=await self._merge_status_for_scan(str(merged_scan_id)),
        )

    async def _union_metadata_into_merged(
        self,
        *,
        merged_scan_id: str,
        source_scan_ids: list[str],
        merged_db_path: Path,
    ) -> dict[str, int]:
        """Copy keyframe_meta + poi_mark + poi_photo from every source scan
        into merged_scan_id.

        - rtabmap_node_id is remapped using provenance labels embedded in
          merged_db_path Node.label (set by `inject_provenance_labels` during
          stage1 prep). Unmappable rows fall back to NULL rtabmap_node_id and
          the build-time pose_backfill will pick the nearest stamp.
        - keyframe_seq is renumbered 1..N across sources in upload order, so
          (scan_id, seq) stays unique. A per-source seq map drives poi_mark
          and poi_photo remap.
        """
        from indoor_server.application.building.multiscan_pose_mapping import (
            parse_provenance_label,
        )

        # 1. (source_scan_id, source_node_id) → merged_node_id via labels.
        node_id_map: dict[tuple[str, int], int] = {}
        con = sqlite3.connect(str(merged_db_path))
        try:
            for merged_id, label in con.execute(
                "SELECT id, label FROM Node WHERE label IS NOT NULL"
            ).fetchall():
                ref = parse_provenance_label(label)
                if ref is not None:
                    node_id_map[(ref.scan_id, ref.node_id)] = int(merged_id)
        finally:
            con.close()

        # 2. scan_session row (FK target for keyframe_meta). Use the first
        # source's session fields as device metadata seed.
        seed = (
            await self._session.execute(
                sa.select(t.scan_session).where(
                    t.scan_session.c.scan_id == source_scan_ids[0]
                )
            )
        ).first()
        await self._session.execute(
            sa.insert(t.scan_session).values(
                scan_id=merged_scan_id,
                started_at=int(seed.started_at) if seed else 0,
                ended_at=int(seed.ended_at) if seed and seed.ended_at else None,
                device_model=str(seed.device_model) if seed else "merged",
                app_version=str(seed.app_version) if seed else "merged",
                state="saved",
                keyframe_count=0,
                notes=f"multi-scan merge of {len(source_scan_ids)} sources",
            )
        )

        # 3. Renumber keyframes across sources.
        next_seq = 1
        seq_map: dict[tuple[str, int], int] = {}
        for src_id in source_scan_ids:
            rows = (
                await self._session.execute(
                    sa.select(t.keyframe_meta)
                    .where(t.keyframe_meta.c.scan_id == src_id)
                    .order_by(t.keyframe_meta.c.seq)
                )
            ).fetchall()
            for kf in rows:
                merged_node_id: int | None = None
                if kf.rtabmap_node_id is not None:
                    merged_node_id = node_id_map.get(
                        (src_id, int(kf.rtabmap_node_id))
                    )
                seq_map[(src_id, int(kf.seq))] = next_seq
                await self._session.execute(
                    sa.insert(t.keyframe_meta).values(
                        scan_id=merged_scan_id,
                        seq=next_seq,
                        captured_at=kf.captured_at,
                        image_path=kf.image_path,
                        pose_matrix=kf.pose_matrix,
                        tx=kf.tx, ty=kf.ty, tz=kf.tz,
                        tracking_state=kf.tracking_state,
                        rtabmap_node_id=merged_node_id,
                    )
                )
                next_seq += 1
        keyframe_count = next_seq - 1
        await self._session.execute(
            sa.update(t.scan_session)
            .where(t.scan_session.c.scan_id == merged_scan_id)
            .values(keyframe_count=keyframe_count)
        )

        # 4. poi_mark + poi_photo. Track old → new id to remap poi_photo FK.
        poi_count = 0
        photo_count = 0
        poi_id_map: dict[int, int] = {}
        for src_id in source_scan_ids:
            pms = (
                await self._session.execute(
                    sa.select(t.poi_mark)
                    .where(t.poi_mark.c.scan_id == src_id)
                    .order_by(t.poi_mark.c.id)
                )
            ).fetchall()
            for pm in pms:
                merged_seq = seq_map.get((src_id, int(pm.keyframe_seq)))
                if merged_seq is None:
                    continue  # orphan POI — keyframe didn't survive
                new_id = (
                    await self._session.execute(
                        sa.insert(t.poi_mark)
                        .values(
                            scan_id=merged_scan_id,
                            keyframe_seq=merged_seq,
                            created_at=pm.created_at,
                            pose_matrix=pm.pose_matrix,
                            tx=pm.tx, ty=pm.ty, tz=pm.tz,
                            track_id=getattr(pm, "track_id", None),
                            label=pm.label,
                            source=pm.source,
                            canonical_id=pm.canonical_id,
                        )
                        .returning(t.poi_mark.c.id)
                    )
                ).scalar_one()
                poi_id_map[int(pm.id)] = int(new_id)
                poi_count += 1

            photos = (
                await self._session.execute(
                    sa.select(t.poi_photo).where(t.poi_photo.c.scan_id == src_id)
                )
            ).fetchall()
            for p in photos:
                new_poi_id = poi_id_map.get(int(p.poi_mark_id))
                merged_seq = seq_map.get((src_id, int(p.keyframe_seq)))
                if new_poi_id is None or merged_seq is None:
                    continue
                await self._session.execute(
                    sa.insert(t.poi_photo).values(
                        scan_id=merged_scan_id,
                        poi_mark_id=new_poi_id,
                        keyframe_seq=merged_seq,
                        captured_at=p.captured_at,
                        bbox_x=getattr(p, "bbox_x", None),
                        bbox_y=getattr(p, "bbox_y", None),
                        bbox_w=getattr(p, "bbox_w", None),
                        bbox_h=getattr(p, "bbox_h", None),
                        class_name=p.class_name,
                        confidence=p.confidence,
                        image_path=getattr(p, "image_path", None),
                        image_blob=p.image_blob,
                    )
                )
                photo_count += 1

        # 5. branch_mark: UNION across sources, with local_id namespaced per
        # source to avoid collision (each sidecar restarts local_id at 1).
        # `_LOCAL_ID_NAMESPACE_STEP` keeps the math obvious for inspection.
        _LOCAL_ID_STEP = 1_000_000
        branch_count = 0
        for src_idx, src_id in enumerate(source_scan_ids):
            bms = (
                await self._session.execute(
                    sa.select(t.branch_mark)
                    .where(t.branch_mark.c.scan_id == src_id)
                    .order_by(t.branch_mark.c.id)
                )
            ).fetchall()
            for bm in bms:
                merged_seq = seq_map.get((src_id, int(bm.keyframe_seq)))
                if merged_seq is None:
                    continue  # orphan branch — keyframe didn't survive
                old_local = (
                    int(bm.local_id) if bm.local_id is not None else None
                )
                new_local = (
                    src_idx * _LOCAL_ID_STEP + old_local
                    if old_local is not None
                    else None
                )
                await self._session.execute(
                    sa.insert(t.branch_mark).values(
                        scan_id=merged_scan_id,
                        keyframe_seq=merged_seq,
                        created_at=bm.created_at,
                        pose_matrix=bm.pose_matrix,
                        tx=bm.tx, ty=bm.ty, tz=bm.tz,
                        node_type=bm.node_type,
                        width_m=getattr(bm, "width_m", None),
                        connect_hint=getattr(bm, "connect_hint", None),
                        connect_node_id=getattr(bm, "connect_node_id", None),
                        mark_session_id=getattr(bm, "mark_session_id", None),
                        dx_local=getattr(bm, "dx_local", None),
                        dy_local=getattr(bm, "dy_local", None),
                        dz_local=getattr(bm, "dz_local", None),
                        local_id=new_local,
                    )
                )
                branch_count += 1

        # 6. branch_edge: copy with from/to remapped via the same per-source
        # local_id namespace so cross-scan edges stay well-formed.
        edge_count = 0
        for src_idx, src_id in enumerate(source_scan_ids):
            bes = (
                await self._session.execute(
                    sa.select(t.branch_edge)
                    .where(t.branch_edge.c.scan_id == src_id)
                    .order_by(t.branch_edge.c.id)
                )
            ).fetchall()
            for be in bes:
                # branch_edge.from_node_id/to_node_id is TEXT (sidecar
                # branch_mark.local_id stringified). Apply the same offset.
                try:
                    from_local = int(be.from_node_id)
                    to_local = int(be.to_node_id)
                except (TypeError, ValueError):
                    continue
                new_from = str(src_idx * _LOCAL_ID_STEP + from_local)
                new_to = str(src_idx * _LOCAL_ID_STEP + to_local)
                await self._session.execute(
                    sa.insert(t.branch_edge).values(
                        scan_id=merged_scan_id,
                        from_node_id=new_from,
                        to_node_id=new_to,
                        kind=be.kind,
                        length_m=be.length_m,
                        mark_session_id=getattr(be, "mark_session_id", None),
                        polygon_closed=getattr(be, "polygon_closed", None),
                        created_at=be.created_at,
                    )
                )
                edge_count += 1

        # 7. interfloor_mark: copy with keyframe_seq remapped. No local_id, so
        # no namespace concern.
        interfloor_count = 0
        for src_id in source_scan_ids:
            ims = (
                await self._session.execute(
                    sa.select(t.interfloor_mark)
                    .where(t.interfloor_mark.c.scan_id == src_id)
                    .order_by(t.interfloor_mark.c.id)
                )
            ).fetchall()
            for im in ims:
                merged_seq = seq_map.get((src_id, int(im.keyframe_seq)))
                if merged_seq is None:
                    continue
                await self._session.execute(
                    sa.insert(t.interfloor_mark).values(
                        scan_id=merged_scan_id,
                        keyframe_seq=merged_seq,
                        created_at=im.created_at,
                        connector_type=im.connector_type,
                        prefix=im.prefix,
                        pose_matrix=im.pose_matrix,
                        tx=im.tx, ty=im.ty, tz=im.tz,
                        dx_local=getattr(im, "dx_local", None),
                        dy_local=getattr(im, "dy_local", None),
                        dz_local=getattr(im, "dz_local", None),
                    )
                )
                interfloor_count += 1

        # 8. poi_canonical: re-point each source's rows at the merged scan_id.
        for src_id in source_scan_ids:
            await self._session.execute(
                sa.text(
                    "UPDATE poi_canonical SET scan_id = :merged_id "
                    "WHERE scan_id = :src_id"
                ),
                {"merged_id": merged_scan_id, "src_id": src_id},
            )

        return {
            "keyframes": keyframe_count,
            "pois": poi_count,
            "photos": photo_count,
            "branches": branch_count,
            "branch_edges": edge_count,
            "interfloor": interfloor_count,
            "mapped_nodes": len(node_id_map),
        }

    async def _merge_status_for_scan(self, scan_id: str) -> str:
        latest = await BuildJobRepository(self._session).get_latest(scan_id)
        if latest is None:
            return "READY"
        if latest.state == BuildState.SUCCEEDED:
            return "COMPLETED"
        if latest.state == BuildState.FAILED:
            return "FAILED"
        if latest.state in (BuildState.PENDING, BuildState.RUNNING):
            return "PROCESSING"
        return "READY"

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


def _chunk_response(
    row: Any, *, included_in_merged_scan: UUID | None = None,
) -> ScanChunkResponse:
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
        included_in_merged_scan=included_in_merged_scan,
    )


async def _kickoff_floor_warmup(*, floor_id: str, scan_id: str) -> None:
    """Fire-and-forget POST to `/admin/superpoint/warmup` so a freshly
    activated scan's SuperPoint cache is preloaded before the next /localize.

    Mirrors `BuildService._warmup_superpoint`. Failure here never fails the
    merge — warmup is opportunistic.
    """
    try:
        reproc = settings.storage_root / "scans" / scan_id / "rtabmap_reprocessed.db"
        raw = settings.storage_root / "scans" / scan_id / "rtabmap.db"
        if reproc.exists() and reproc.stat().st_size > 0:
            db_path = reproc
        elif raw.exists():
            db_path = raw
        else:
            logger.warning(
                "merge warmup skipped — no rtabmap db floor=%s scan=%s",
                floor_id, scan_id,
            )
            return
        import httpx as _httpx
        url = "http://server:8000/admin/superpoint/warmup"
        payload = {"map_id": floor_id, "db_path": str(db_path)}
        async with _httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(url, json=payload)
        logger.info(
            "merge warmup queued floor=%s status=%d",
            floor_id, resp.status_code,
        )
    except Exception as exc:
        logger.warning(
            "merge warmup trigger failed floor=%s scan=%s err=%s",
            floor_id, scan_id, exc,
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _ReprocessFailed(Exception):
    """Single rtabmap-reprocess invocation failed."""


class _AlignFailed(Exception):
    """RANSAC SE(3) alignment failed."""


def _align_and_patch_merged(
    stage2_db: Path,
    out_db: Path,
    a_source_db: Path,
    b_source_db: Path,
) -> None:
    """Yaw-only RANSAC SE(3) alignment for the 2-source merge case.

    rtabmap-reprocess `-a` leaves the two ARKit-world origins union'd when the
    cross-session loop closures imply translations the graph optimizer rejects
    as outliers. We recover the rigid transform from A's frame to B's frame and
    apply yaw-only constraint (indoor single-floor assumption) so the floor
    plane stays flat.

    Reuses sprint81/align_and_rebuild_merged helpers (validated).
    """
    from scripts.sprint81.align_and_rebuild_merged import (
        estimate_T_AB_ransac,
        load_cross_session_links,
        load_poses,
        patch_merged_db,
    )

    a_poses = load_poses(a_source_db)
    b_poses = load_poses(b_source_db)
    if not a_poses or not b_poses:
        raise _AlignFailed(
            f"empty pose set (a={len(a_poses)}, b={len(b_poses)})"
        )
    a_count = len(a_poses)
    cross_pairs = load_cross_session_links(stage2_db, a_count)
    if len(cross_pairs) < 3:
        raise _AlignFailed(
            f"insufficient cross-session loop closures: {len(cross_pairs)}"
        )
    try:
        T_AB, rmse, inliers = estimate_T_AB_ransac(
            a_poses, b_poses, cross_pairs, a_count
        )
    except ValueError as e:
        raise _AlignFailed(f"RANSAC failed: {e}") from e

    import numpy as np

    yaw = float(np.arctan2(T_AB[1, 0], T_AB[0, 0]))
    R_yaw = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    T_AB_planar = np.eye(4, dtype=np.float64)
    T_AB_planar[:3, :3] = R_yaw
    T_AB_planar[0, 3] = T_AB[0, 3]
    T_AB_planar[1, 3] = T_AB[1, 3]
    T_AB_planar[2, 3] = 0.0
    logger.info(
        "stage3 yaw-only constraint applied: yaw=%.2f deg, t=(%.2f, %.2f, 0.0)",
        np.degrees(yaw), T_AB_planar[0, 3], T_AB_planar[1, 3],
    )
    patch_merged_db(stage2_db, out_db, a_poses, b_poses, T_AB_planar, a_count)
    logger.info(
        "stage3 align: cross_pairs=%d inliers=%d rmse=%.3f T_AB.t=%s",
        len(cross_pairs), inliers, rmse, T_AB_planar[:3, 3].round(3).tolist(),
    )


async def _run_rtabmap_reprocess_single(
    *,
    binary_path: str,
    input_db: Path,
    output_db: Path,
    timeout_s: float,
) -> None:
    """Single-source reprocess to populate SIFT Word/Feature.

    `-default` discards the input db's params so client-side overrides
    (e.g. Mem/IncrementalMemory=false) don't leak through. We force SIFT via
    `Vis/FeatureType=1` + `Kp/DetectorStrategy=1`.
    """
    output_db.parent.mkdir(parents=True, exist_ok=True)
    if output_db.exists():
        output_db.unlink()
    cmd = [
        binary_path, "-default",
        "--Mem/IncrementalMemory", "true",
        "--Vis/FeatureType", "1",
        "--Kp/DetectorStrategy", "1",
        "--uwarn",
        str(input_db), str(output_db),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError as e:
        proc.kill()
        await proc.wait()
        raise _ReprocessFailed(f"timeout > {timeout_s}s") from e
    if proc.returncode != 0:
        stderr_text = stderr_b.decode(errors="replace") if stderr_b else ""
        raise _ReprocessFailed(
            f"exit {proc.returncode}; stderr tail: {stderr_text[-1200:]}"
        )
