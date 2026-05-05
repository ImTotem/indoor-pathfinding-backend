"""ScanIngestRepository — ingest record + 도메인 테이블 upsert 통합."""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.domain.scan.models import ExistingScan, SidecarContents
from indoor_server.infrastructure.db import tables as t

logger = logging.getLogger(__name__)


class ScanIngestRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def find_existing(self, scan_id: str) -> ExistingScan | None:
        row = await self._session.execute(
            sa.select(
                t.scan_ingest.c.scan_id,
                t.scan_ingest.c.payload_sha256,
                t.scan_ingest.c.ingested_at,
            ).where(t.scan_ingest.c.scan_id == scan_id)
        )
        record = row.first()
        if record is None:
            return None
        return ExistingScan(
            scan_id=record.scan_id,
            payload_sha256=record.payload_sha256,
            ingested_at=record.ingested_at,  # W-2: ingested_at 포함
        )

    async def ingest(
        self,
        *,
        contents: SidecarContents,
        storage_path: str,
        payload_sha256: str,
        device_info: str | None,
        replace: bool,
    ) -> None:
        """
        트랜잭션 1개로:
          - scan_ingest upsert
          - replace=True 시 도메인 테이블 DELETE → INSERT
          - replace=False 시 INSERT (중복 없음 가정)
        """
        scan_id = contents.scan_session.id
        device_info_json: dict[str, object] | None = (
            json.loads(device_info) if device_info else None
        )
        now = datetime.now(tz=UTC)

        await self._upsert_scan_ingest(
            scan_id=scan_id,
            payload_sha256=payload_sha256,
            storage_path=storage_path,
            device_info=device_info_json,
            replace=replace,
            replaced_at=now if replace else None,
        )

        if replace:
            await self._delete_domain_rows(scan_id)

        await self._insert_scan_session(contents, scan_id)
        await self._insert_keyframes(contents, scan_id)
        poi_id_map = await self._insert_poi_marks(contents, scan_id)
        await self._insert_poi_photos(contents, scan_id, poi_id_map)
        await self._insert_branch_marks(contents, scan_id)
        await self._insert_yolo_detections(contents, scan_id)
        await self._insert_interfloor_marks(contents, scan_id)

        logger.info(
            "ingest complete",
            extra={
                "scan_id": scan_id,
                "keyframes": len(contents.keyframes),
                "poi_marks": len(contents.poi_marks),
                "interfloor_marks": len(contents.interfloor_marks),
                "replace": replace,
            },
        )

    async def delete(self, scan_id: str) -> None:
        """보상 트랜잭션용: scan_ingest + 연관 도메인 테이블 삭제 (CASCADE)."""
        await self._session.execute(
            sa.delete(t.scan_ingest).where(t.scan_ingest.c.scan_id == scan_id)
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    async def _upsert_scan_ingest(
        self,
        *,
        scan_id: str,
        payload_sha256: str,
        storage_path: str,
        device_info: dict[str, object] | None,
        replace: bool,
        replaced_at: datetime | None,
    ) -> None:
        stmt = pg_insert(t.scan_ingest).values(
            scan_id=scan_id,
            payload_sha256=payload_sha256,
            storage_path=storage_path,
            device_info=device_info,
            replaced_at=replaced_at,
        )
        update_dict: dict[str, object] = {
            "payload_sha256": payload_sha256,
            "storage_path": storage_path,
            "device_info": device_info,
        }
        if replace:
            update_dict["replaced_at"] = replaced_at
        stmt = stmt.on_conflict_do_update(
            index_elements=["scan_id"],
            set_=update_dict,
        )
        await self._session.execute(stmt)

    async def _delete_domain_rows(self, scan_id: str) -> None:
        # cascade로 연관 테이블 자동 삭제
        await self._session.execute(
            sa.delete(t.scan_session).where(t.scan_session.c.scan_id == scan_id)
        )

    async def _insert_scan_session(self, contents: SidecarContents, scan_id: str) -> None:
        s = contents.scan_session
        await self._session.execute(
            sa.insert(t.scan_session).values(
                scan_id=scan_id,
                started_at=s.started_at,
                ended_at=s.ended_at,
                device_model=s.device_model,
                app_version=s.app_version,
                state=s.state,
                keyframe_count=s.keyframe_count,
                notes=s.notes,
            )
        )

    async def _insert_keyframes(self, contents: SidecarContents, scan_id: str) -> None:
        if not contents.keyframes:
            return
        rows = [
            {
                "scan_id": scan_id,
                "seq": kf.seq,
                "captured_at": kf.captured_at,
                "image_path": kf.image_path,
                "pose_matrix": kf.pose_matrix,
                "tx": kf.tx,
                "ty": kf.ty,
                "tz": kf.tz,
                "tracking_state": kf.tracking_state,
                "rtabmap_node_id": kf.rtabmap_node_id,
            }
            for kf in contents.keyframes
        ]
        await self._session.execute(sa.insert(t.keyframe_meta), rows)

    async def _insert_poi_marks(
        self, contents: SidecarContents, scan_id: str
    ) -> dict[int, int]:
        if not contents.poi_marks:
            return {}
        rows = [
            {
                "scan_id": scan_id,
                "keyframe_seq": pm.keyframe_seq,
                "created_at": pm.created_at,
                "pose_matrix": pm.pose_matrix,
                "tx": pm.tx,
                "ty": pm.ty,
                "tz": pm.tz,
                "track_id": pm.track_id,
                "label": pm.label,
                "source": pm.source.value,
                "world_pose": None,   # R-2 placeholder
                "canonical_id": None,  # R-4 placeholder
            }
            for pm in contents.poi_marks
        ]
        result = await self._session.execute(
            sa.insert(t.poi_mark).returning(t.poi_mark.c.id),
            rows,
        )
        generated_ids = [int(row.id) for row in result]
        return {
            int(local.id): generated
            for local, generated in zip(contents.poi_marks, generated_ids, strict=True)
        }

    async def _insert_poi_photos(
        self, contents: SidecarContents, scan_id: str, poi_id_map: dict[int, int]
    ) -> None:
        if not contents.poi_photos:
            return
        rows = [
            {
                "poi_mark_id": poi_id_map[pp.poi_mark_id],
                "scan_id": scan_id,
                "keyframe_seq": pp.keyframe_seq,
                "captured_at": pp.captured_at,
                "bbox_x": pp.bbox_x,
                "bbox_y": pp.bbox_y,
                "bbox_w": pp.bbox_w,
                "bbox_h": pp.bbox_h,
                "class_name": pp.class_name,
                "confidence": pp.confidence,
                "image_path": None,  # 편의 캐시 — 본 스프린트 미설정
                "image_blob": pp.image_blob,
            }
            for pp in contents.poi_photos
        ]
        await self._session.execute(sa.insert(t.poi_photo), rows)

    async def _insert_branch_marks(self, contents: SidecarContents, scan_id: str) -> None:
        if not contents.branch_marks:
            return
        rows = [
            {
                "scan_id": scan_id,
                "keyframe_seq": bm.keyframe_seq,
                "created_at": bm.created_at,
                "pose_matrix": bm.pose_matrix,
                "tx": bm.tx,
                "ty": bm.ty,
                "tz": bm.tz,
            }
            for bm in contents.branch_marks
        ]
        await self._session.execute(sa.insert(t.branch_mark), rows)

    async def _insert_yolo_detections(self, contents: SidecarContents, scan_id: str) -> None:
        if not contents.yolo_detections:
            return
        rows = [
            {
                "scan_id": scan_id,
                "keyframe_seq": yd.keyframe_seq,
                "class_name": yd.class_name,
                "confidence": yd.confidence,
                "bbox_x": yd.bbox_x,
                "bbox_y": yd.bbox_y,
                "bbox_w": yd.bbox_w,
                "bbox_h": yd.bbox_h,
                "mask_rle": yd.mask_rle,
                "source": "on_device",  # 본 스프린트는 on_device 고정
                "track_id": yd.track_id,
            }
            for yd in contents.yolo_detections
        ]
        await self._session.execute(sa.insert(t.yolo_detection), rows)

    async def _insert_interfloor_marks(
        self, contents: SidecarContents, scan_id: str
    ) -> None:
        """Sprint 65 v6: interfloor_mark 도메인 테이블 영속화."""
        if not contents.interfloor_marks:
            return
        rows = [
            {
                "scan_id": scan_id,
                "keyframe_seq": im.keyframe_seq,
                "created_at": im.created_at,
                "connector_type": im.connector_type,
                "prefix": im.prefix,
                "pose_matrix": im.pose_matrix,
                "tx": im.tx,
                "ty": im.ty,
                "tz": im.tz,
            }
            for im in contents.interfloor_marks
        ]
        await self._session.execute(sa.insert(t.interfloor_mark), rows)
