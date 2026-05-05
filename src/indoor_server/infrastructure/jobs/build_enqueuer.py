"""BuildEnqueuer — build_job INSERT 래퍼. API/ingest 공용."""
from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.domain.building.models import BuildJob
from indoor_server.infrastructure.db.repositories.build_job_repo import BuildJobRepository

logger = logging.getLogger(__name__)


class BuildEnqueuer:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repo = BuildJobRepository(session)

    async def enqueue(self, scan_id: str) -> BuildJob:
        """pending job INSERT. 호출자가 트랜잭션을 관리해야 한다."""
        job = await self._repo.create(scan_id=scan_id)
        logger.info(
            "build job enqueued scan_id=%s build_job_id=%s",
            scan_id,
            job.build_job_id,
        )
        return job
