"""StaleJobWatchdog — running 상태인데 locked_at이 오래된 job을 pending으로 복구."""
from __future__ import annotations

import logging

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class StaleJobWatchdog:
    def __init__(self, session: AsyncSession, stale_after_sec: int = 600) -> None:
        self._session = session
        self._stale_after_sec = stale_after_sec

    async def recover_stale(self) -> int:
        """stale running job → pending 복구. 복구된 건수 반환."""
        result = await self._session.execute(
            sa.text("""
                UPDATE build_job
                SET state     = 'pending',
                    locked_at = NULL,
                    worker_id = NULL
                WHERE state = 'running'
                  AND locked_at < now() - make_interval(secs => :stale_sec)
            """),
            {"stale_sec": self._stale_after_sec},
        )
        count: int = getattr(result, "rowcount", 0)
        if count > 0:
            logger.warning("stale jobs recovered count=%d", count)
        return count
