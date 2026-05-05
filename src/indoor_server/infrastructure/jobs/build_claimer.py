"""BuildJobClaimer — SKIP LOCKED worker claim SQL."""
from __future__ import annotations

import logging

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.domain.building.models import BuildJob
from indoor_server.infrastructure.db.repositories.build_job_repo import BuildJobRepository

logger = logging.getLogger(__name__)

_CLAIM_SQL = sa.text("""
UPDATE build_job
SET state        = 'running',
    locked_at    = now(),
    worker_id    = :worker_id,
    attempt_count = attempt_count + 1,
    started_at   = COALESCE(started_at, now())
WHERE build_job_id = (
    SELECT build_job_id FROM build_job
    WHERE state = 'pending' AND attempt_count < max_attempts
    ORDER BY enqueued_at
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
RETURNING build_job_id, scan_id, state, current_step, progress,
          enqueued_at, started_at, finished_at, locked_at,
          worker_id, attempt_count, max_attempts,
          failure_reason, failure_detail, counts
""")


class BuildJobClaimer:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repo = BuildJobRepository(session)

    async def claim_one(self, worker_id: str) -> BuildJob | None:
        """pending job을 원자적으로 running으로 전환 후 반환. 없으면 None."""
        result = await self._session.execute(
            _CLAIM_SQL, {"worker_id": worker_id}
        )
        row = result.first()
        if row is None:
            return None
        # claim 직후 전체 job 데이터를 re-fetch (RETURNING은 제한적)
        job = await self._repo.get_by_id(str(row.build_job_id))
        logger.info(
            "job claimed build_job_id=%s scan_id=%s worker_id=%s attempt=%s",
            row.build_job_id,
            row.scan_id,
            worker_id,
            row.attempt_count,
        )
        return job
