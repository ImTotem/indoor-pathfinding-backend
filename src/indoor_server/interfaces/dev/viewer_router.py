"""Dev viewer 라우터 — INDOOR_DEV_VIEWER_ENABLED=true 일 때만 등록.

/dev/viewer/scans — 빌드 완료된 scan 목록 반환 (개발 편의용).
"""
from __future__ import annotations

import logging
from typing import Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.domain.building.enums import BuildState
from indoor_server.infrastructure.db import tables as t
from indoor_server.infrastructure.db.engine import get_session

logger = logging.getLogger(__name__)

dev_router = APIRouter(prefix="/dev/api", tags=["dev-viewer"])


@dev_router.get(
    "/scans",
)
async def list_scans(
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, Any]]:
    """빌드 완료된 scan 목록 반환.

    응답 예시:
    [{"scan_id": "9c481325...", "build_job_id": "...", "walkable_cells": 56832,
      "map_nodes": 19, "map_edges": 19}]
    """
    rows = (
        await session.execute(
            sa.select(
                t.build_job.c.scan_id,
                t.build_job.c.build_job_id,
                t.build_job.c.counts,
                t.build_job.c.finished_at,
            )
            .where(t.build_job.c.state == BuildState.SUCCEEDED.value)
            .order_by(t.build_job.c.finished_at.desc())
        )
    ).fetchall()

    result: list[dict[str, Any]] = []
    for row in rows:
        counts = row.counts or {}
        result.append(
            {
                "scan_id": row.scan_id,
                "build_job_id": row.build_job_id,
                "walkable_cells": counts.get("walkable_cells", 0),
                "map_nodes": counts.get("map_nodes", 0),
                "map_edges": counts.get("map_edges", 0),
                "finished_at": row.finished_at.isoformat() if row.finished_at else None,
            }
        )
    return result
