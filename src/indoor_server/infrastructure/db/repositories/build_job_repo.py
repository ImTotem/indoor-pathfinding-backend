"""BuildJobRepository — build_job 테이블 CRUD."""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.domain.building.enums import BuildFailureReason, BuildState, BuildStep
from indoor_server.domain.building.models import BuildCounts, BuildJob
from indoor_server.infrastructure.db import tables as t

logger = logging.getLogger(__name__)


def _row_to_job(row: sa.engine.Row) -> BuildJob:  # type: ignore[type-arg]
    counts = None
    if row.counts is not None:
        counts = BuildCounts(**row.counts)
    return BuildJob(
        build_job_id=UUID(row.build_job_id),
        scan_id=UUID(row.scan_id),
        state=BuildState(row.state),
        current_step=BuildStep(row.current_step) if row.current_step else None,
        progress=row.progress,
        enqueued_at=row.enqueued_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
        locked_at=row.locked_at,
        worker_id=row.worker_id,
        attempt_count=row.attempt_count,
        max_attempts=row.max_attempts,
        failure_reason=BuildFailureReason(row.failure_reason) if row.failure_reason else None,
        failure_detail=row.failure_detail,
        counts=counts,
    )


class BuildJobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, *, scan_id: str) -> BuildJob:
        """pending 상태 job INSERT."""
        job_id = str(uuid4())
        now = datetime.now(tz=UTC)
        await self._session.execute(
            sa.insert(t.build_job).values(
                build_job_id=job_id,
                scan_id=scan_id,
                state="pending",
                enqueued_at=now,
                attempt_count=0,
                max_attempts=3,
            )
        )
        row = (
            await self._session.execute(
                sa.select(t.build_job).where(t.build_job.c.build_job_id == job_id)
            )
        ).first()
        assert row is not None
        return _row_to_job(row)

    async def get_latest(self, scan_id: str) -> BuildJob | None:
        """scan_id의 가장 최근 job 조회."""
        row = (
            await self._session.execute(
                sa.select(t.build_job)
                .where(t.build_job.c.scan_id == scan_id)
                .order_by(t.build_job.c.enqueued_at.desc())
                .limit(1)
            )
        ).first()
        if row is None:
            return None
        return _row_to_job(row)

    async def get_by_id(self, build_job_id: str) -> BuildJob | None:
        row = (
            await self._session.execute(
                sa.select(t.build_job).where(t.build_job.c.build_job_id == build_job_id)
            )
        ).first()
        if row is None:
            return None
        return _row_to_job(row)

    async def update_state(
        self,
        *,
        build_job_id: str,
        state: BuildState,
        step: BuildStep | None = None,
        progress: float | None = None,
        failure_reason: BuildFailureReason | None = None,
        failure_detail: str | None = None,
        counts: BuildCounts | None = None,
        finished_at: datetime | None = None,
    ) -> None:
        values: dict[str, object] = {"state": state.value}
        if step is not None:
            values["current_step"] = step.value
        if progress is not None:
            values["progress"] = progress
        if failure_reason is not None:
            values["failure_reason"] = failure_reason.value
        if failure_detail is not None:
            values["failure_detail"] = failure_detail
        if counts is not None:
            values["counts"] = json.loads(counts.model_dump_json())
        if finished_at is not None:
            values["finished_at"] = finished_at
        await self._session.execute(
            sa.update(t.build_job)
            .where(t.build_job.c.build_job_id == build_job_id)
            .values(**values)
        )

    async def update_progress(
        self,
        *,
        build_job_id: str,
        step: BuildStep,
        progress: float,
    ) -> None:
        """단계 경계에서 빠른 진행률 업데이트 (state 유지)."""
        await self._session.execute(
            sa.update(t.build_job)
            .where(
                t.build_job.c.build_job_id == build_job_id,
                t.build_job.c.state == "running",
            )
            .values(current_step=step.value, progress=progress)
        )

    async def cancel_active_jobs(self, scan_id: str) -> int:
        """scan_id의 pending/running job을 모두 cancelled로 표시. 취소된 건수 반환."""
        result = await self._session.execute(
            sa.update(t.build_job)
            .where(
                t.build_job.c.scan_id == scan_id,
                t.build_job.c.state.in_(["pending", "running"]),
            )
            .values(
                state="cancelled",
                finished_at=datetime.now(tz=UTC),
                failure_detail="cancelled by force requeue",
            )
        )
        return int(getattr(result, "rowcount", 0))

    async def get_current_state(self, build_job_id: str) -> BuildState | None:
        """worker 취소 체크용 단일 컬럼 조회."""
        row = (
            await self._session.execute(
                sa.select(t.build_job.c.state).where(
                    t.build_job.c.build_job_id == build_job_id
                )
            )
        ).first()
        if row is None:
            return None
        return BuildState(row.state)
