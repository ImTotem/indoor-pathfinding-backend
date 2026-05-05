"""InterfloorMarkRepository — interfloor_mark 테이블 read.

Sprint 65 Phase 4d (실제 closure: Sprint 67 후속): build_service 가 interfloor_mark
도메인 row 를 읽어 display_navigation_grid 에 합성 POI 로 흘려준다. 기존 vertical
connector inference 가 explicit prefix 기반으로 멀티-층 edge 를 만든다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.infrastructure.db import tables as t

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InterfloorMarkDbRow:
    """server-side interfloor_mark row (sidecar pre-ingest 와 별도)."""

    id: int
    scan_id: str
    keyframe_seq: int
    created_at: int
    connector_type: str  # "elevator" | "escalator" | "stairs"
    prefix: str          # ex: "EV-A", "ST-1"
    pose_matrix: bytes
    tx: float
    ty: float
    tz: float


class InterfloorMarkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_by_scan(self, scan_id: str) -> list[InterfloorMarkDbRow]:
        rows = (
            await self._session.execute(
                sa.select(t.interfloor_mark).where(
                    t.interfloor_mark.c.scan_id == scan_id
                )
            )
        ).fetchall()
        return [
            InterfloorMarkDbRow(
                id=int(row.id),
                scan_id=str(row.scan_id),
                keyframe_seq=int(row.keyframe_seq),
                created_at=int(row.created_at),
                connector_type=str(row.connector_type),
                prefix=str(row.prefix),
                pose_matrix=bytes(row.pose_matrix),
                tx=float(row.tx),
                ty=float(row.ty),
                tz=float(row.tz),
            )
            for row in rows
        ]
