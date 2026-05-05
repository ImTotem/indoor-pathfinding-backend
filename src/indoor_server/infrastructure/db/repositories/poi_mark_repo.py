"""POIMarkRepository — poi_mark 테이블 읽기/갱신."""
from __future__ import annotations

import logging

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.domain.scan.models import POIMarkRow
from indoor_server.infrastructure.db import tables as t

logger = logging.getLogger(__name__)


class POIMarkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_by_scan(self, scan_id: str) -> list[POIMarkRow]:
        """scan_id에 속한 poi_mark 전체 조회."""
        from indoor_server.domain.poi.enums import POISource

        rows = (
            await self._session.execute(
                sa.select(t.poi_mark).where(t.poi_mark.c.scan_id == scan_id)
            )
        ).fetchall()

        return [
            POIMarkRow(
                id=row.id,
                scan_id=row.scan_id,
                keyframe_seq=row.keyframe_seq,
                created_at=row.created_at,
                pose_matrix=row.pose_matrix,
                tx=row.tx,
                ty=row.ty,
                tz=row.tz,
                track_id=row.track_id,
                label=row.label,
                source=POISource(row.source),
            )
            for row in rows
        ]

    async def set_world_pose(
        self, *, updates: list[tuple[int, float, float, float]]
    ) -> None:
        """poi_mark_id 단위 bulk UPDATE world_pose (PointZ)."""
        if not updates:
            return
        for poi_mark_id, x, y, z in updates:
            await self._session.execute(
                sa.update(t.poi_mark)
                .where(t.poi_mark.c.id == poi_mark_id)
                .values(
                    world_pose=sa.func.ST_MakePoint(x, y, z)
                )
            )
        logger.info("poi_mark world_pose updated count=%d", len(updates))
