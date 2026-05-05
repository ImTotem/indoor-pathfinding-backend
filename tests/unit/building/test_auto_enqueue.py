"""W-1 Critical: auto_enqueue 회귀 테스트."""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from indoor_server.application.scan_ingest_service import ScanIngestService
from indoor_server.domain.building.enums import BuildState
from indoor_server.domain.building.models import BuildJob
from indoor_server.domain.scan.models import ScanIngestResult

# _maybe_enqueue가 사용하는 BuildJobRepository 패치 경로 (모듈 레벨 import)
_BUILD_JOB_REPO_PATCH = "indoor_server.application.scan_ingest_service.BuildJobRepository"


def _fake_result(state: str = "ingested") -> ScanIngestResult:
    return ScanIngestResult(
        scan_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        state=state,  # type: ignore[arg-type]
        keyframe_count=0,
        poi_marks_track_lock=0,
        poi_marks_manual=0,
        poi_photo_count=0,
        branch_mark_count=0,
        yolo_detection_count=0,
        storage_path="scans/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        payload_sha256="a" * 64,
    )


def _fake_job(state: BuildState = BuildState.PENDING) -> BuildJob:
    return BuildJob(
        build_job_id=uuid4(),
        scan_id=uuid4(),
        state=state,
        enqueued_at=datetime.now(tz=UTC),
    )


def _make_mock_session() -> AsyncMock:
    """async with session.begin(): 패턴을 지원하는 mock session."""

    session = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    # begin()은 동기 호출 → ctx 반환 (AsyncMock으로 하면 coroutine 반환으로 실패)
    session.begin = MagicMock(return_value=ctx)
    return session


@pytest.mark.asyncio
async def test_auto_enqueue_true_calls_enqueuer() -> None:
    """build_auto_enqueue=True → BuildEnqueuer.enqueue() 호출, build_job_id 반환."""
    scan_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    job = _fake_job(BuildState.PENDING)

    session = _make_mock_session()
    build_enqueuer = AsyncMock()
    build_enqueuer.enqueue.return_value = job

    svc = ScanIngestService.__new__(ScanIngestService)
    svc._session = session
    svc._build_enqueuer = build_enqueuer

    with (
        patch("indoor_server.application.scan_ingest_service.settings") as mock_settings,
        patch(
            _BUILD_JOB_REPO_PATCH
        ) as mock_repo_cls,
    ):
        mock_settings.build_auto_enqueue = True
        repo_instance = AsyncMock()
        repo_instance.get_latest.return_value = None  # 기존 job 없음
        mock_repo_cls.return_value = repo_instance

        result_job_id = await svc._maybe_enqueue(scan_id)

    assert result_job_id == str(job.build_job_id)
    build_enqueuer.enqueue.assert_awaited_once_with(scan_id)


@pytest.mark.asyncio
async def test_auto_enqueue_false_skips_enqueuer() -> None:
    """build_auto_enqueue=False → enqueue 미호출, None 반환."""
    scan_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    session = _make_mock_session()
    build_enqueuer = AsyncMock()

    svc = ScanIngestService.__new__(ScanIngestService)
    svc._session = session
    svc._build_enqueuer = build_enqueuer

    with patch("indoor_server.application.scan_ingest_service.settings") as mock_settings:
        mock_settings.build_auto_enqueue = False
        result = await svc._maybe_enqueue(scan_id)

    assert result is None
    build_enqueuer.enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_enqueue_skips_if_succeeded() -> None:
    """이미 succeeded build → 재enqueue 안 하고 기존 build_job_id 반환."""
    scan_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    existing_job = _fake_job(BuildState.SUCCEEDED)

    session = _make_mock_session()
    build_enqueuer = AsyncMock()

    svc = ScanIngestService.__new__(ScanIngestService)
    svc._session = session
    svc._build_enqueuer = build_enqueuer

    with (
        patch("indoor_server.application.scan_ingest_service.settings") as mock_settings,
        patch(
            _BUILD_JOB_REPO_PATCH
        ) as mock_repo_cls,
    ):
        mock_settings.build_auto_enqueue = True
        repo_instance = AsyncMock()
        repo_instance.get_latest.return_value = existing_job
        mock_repo_cls.return_value = repo_instance

        result = await svc._maybe_enqueue(scan_id)

    # 기존 succeeded job의 id 반환
    assert result == str(existing_job.build_job_id)
    # enqueue는 호출되지 않아야 함
    build_enqueuer.enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_enqueue_skips_if_pending() -> None:
    """이미 pending build → 재enqueue 안 하고 기존 build_job_id 반환."""
    scan_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    existing_job = _fake_job(BuildState.PENDING)

    session = _make_mock_session()
    build_enqueuer = AsyncMock()

    svc = ScanIngestService.__new__(ScanIngestService)
    svc._session = session
    svc._build_enqueuer = build_enqueuer

    with (
        patch("indoor_server.application.scan_ingest_service.settings") as mock_settings,
        patch(
            _BUILD_JOB_REPO_PATCH
        ) as mock_repo_cls,
    ):
        mock_settings.build_auto_enqueue = True
        repo_instance = AsyncMock()
        repo_instance.get_latest.return_value = existing_job
        mock_repo_cls.return_value = repo_instance

        result = await svc._maybe_enqueue(scan_id)

    assert result == str(existing_job.build_job_id)
    build_enqueuer.enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_enqueue_reenqueues_if_failed() -> None:
    """이전 build가 failed → 새로 enqueue."""
    scan_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    existing_job = _fake_job(BuildState.FAILED)
    new_job = _fake_job(BuildState.PENDING)

    session = _make_mock_session()
    build_enqueuer = AsyncMock()
    build_enqueuer.enqueue.return_value = new_job

    svc = ScanIngestService.__new__(ScanIngestService)
    svc._session = session
    svc._build_enqueuer = build_enqueuer

    with (
        patch("indoor_server.application.scan_ingest_service.settings") as mock_settings,
        patch(
            _BUILD_JOB_REPO_PATCH
        ) as mock_repo_cls,
    ):
        mock_settings.build_auto_enqueue = True
        repo_instance = AsyncMock()
        repo_instance.get_latest.return_value = existing_job
        mock_repo_cls.return_value = repo_instance

        result = await svc._maybe_enqueue(scan_id)

    assert result == str(new_job.build_job_id)
    build_enqueuer.enqueue.assert_awaited_once_with(scan_id)


@pytest.mark.asyncio
async def test_auto_enqueue_none_enqueuer_returns_none() -> None:
    """build_enqueuer=None (build_auto_enqueue=True이지만 enqueuer 미전달) → None 반환."""
    scan_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    session = _make_mock_session()

    svc = ScanIngestService.__new__(ScanIngestService)
    svc._session = session
    svc._build_enqueuer = None  # enqueuer 없음

    with patch("indoor_server.application.scan_ingest_service.settings") as mock_settings:
        mock_settings.build_auto_enqueue = True
        result = await svc._maybe_enqueue(scan_id)

    assert result is None
