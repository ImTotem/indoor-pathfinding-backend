"""Build API 통합 테스트 — DB mock 방식."""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from indoor_server.domain.building.enums import BuildState
from indoor_server.domain.building.models import BuildCounts, BuildJob
from indoor_server.domain.scan.models import ExistingScan
from indoor_server.main import app

SCAN_ID = "44444444-4444-4444-4444-444444444444"
JOB_ID = "55555555-5555-5555-5555-555555555555"
TOKEN = "dev-token"

_REPO_PATH = "indoor_server.interfaces.api.router.ScanIngestRepository"
_BUILD_REPO_PATH = "indoor_server.interfaces.api.router.BuildJobRepository"
_ENQUEUER_PATH = "indoor_server.interfaces.api.router.BuildEnqueuer"


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def _existing_scan() -> ExistingScan:
    return ExistingScan(
        scan_id=SCAN_ID,
        payload_sha256="abc" * 21 + "ab",
        ingested_at=datetime.now(tz=UTC),
    )


def _make_job(state: BuildState = BuildState.PENDING) -> BuildJob:
    return BuildJob(
        build_job_id=uuid4(),
        scan_id=uuid4(),
        state=state,
        enqueued_at=datetime.now(tz=UTC),
    )


def _make_session_mock() -> AsyncMock:
    session = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    session.begin.return_value = ctx
    return session


@pytest.mark.asyncio
async def test_trigger_build_success():
    """POST /scan/{id}/build → 202 pending."""
    mock_session = _make_session_mock()
    repo_mock = AsyncMock()
    repo_mock.find_existing.return_value = _existing_scan()

    build_repo_mock = AsyncMock()
    build_repo_mock.get_latest.return_value = None

    enqueuer_mock = AsyncMock()
    job = _make_job(BuildState.PENDING)
    enqueuer_mock.enqueue.return_value = job

    async def _fake_get_session():
        yield mock_session

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_REPO_PATH, return_value=repo_mock),
        patch(_BUILD_REPO_PATH, return_value=build_repo_mock),
        patch(_ENQUEUER_PATH, return_value=enqueuer_mock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post(f"/scan/{SCAN_ID}/build", headers=_auth_headers())

    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["state"] == "pending"
    assert "build_job_id" in body


@pytest.mark.asyncio
async def test_trigger_build_already_running_409():
    """이미 running인 job이 있으면 force=false → 409."""
    mock_session = _make_session_mock()
    repo_mock = AsyncMock()
    repo_mock.find_existing.return_value = _existing_scan()

    build_repo_mock = AsyncMock()
    build_repo_mock.get_latest.return_value = _make_job(BuildState.RUNNING)

    async def _fake_get_session():
        yield mock_session

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_REPO_PATH, return_value=repo_mock),
        patch(_BUILD_REPO_PATH, return_value=build_repo_mock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post(f"/scan/{SCAN_ID}/build", headers=_auth_headers())

    assert resp.status_code == 409, resp.text


@pytest.mark.asyncio
async def test_trigger_build_force_cancel_and_requeue():
    """force=true → 기존 job 취소 후 새 job 생성."""
    mock_session = _make_session_mock()
    repo_mock = AsyncMock()
    repo_mock.find_existing.return_value = _existing_scan()

    build_repo_mock = AsyncMock()
    build_repo_mock.get_latest.return_value = _make_job(BuildState.RUNNING)
    build_repo_mock.cancel_active_jobs = AsyncMock(return_value=1)

    enqueuer_mock = AsyncMock()
    enqueuer_mock.enqueue.return_value = _make_job(BuildState.PENDING)

    async def _fake_get_session():
        yield mock_session

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_REPO_PATH, return_value=repo_mock),
        patch(_BUILD_REPO_PATH, return_value=build_repo_mock),
        patch(_ENQUEUER_PATH, return_value=enqueuer_mock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post(f"/scan/{SCAN_ID}/build?force=true", headers=_auth_headers())

    assert resp.status_code == 202, resp.text
    build_repo_mock.cancel_active_jobs.assert_called_once_with(SCAN_ID)


@pytest.mark.asyncio
async def test_get_build_status():
    """GET /scan/{id}/build → 200 with state."""
    mock_session = _make_session_mock()
    repo_mock = AsyncMock()
    repo_mock.find_existing.return_value = _existing_scan()

    job = _make_job(BuildState.RUNNING)
    build_repo_mock = AsyncMock()
    build_repo_mock.get_latest.return_value = job

    async def _fake_get_session():
        yield mock_session

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_REPO_PATH, return_value=repo_mock),
        patch(_BUILD_REPO_PATH, return_value=build_repo_mock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get(f"/scan/{SCAN_ID}/build", headers=_auth_headers())

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["state"] == "running"


@pytest.mark.asyncio
async def test_get_graph_not_ready_422():
    """GET /scan/{id}/graph — succeeded 아닌 상태 → 422."""
    mock_session = _make_session_mock()
    repo_mock = AsyncMock()
    repo_mock.find_existing.return_value = _existing_scan()

    build_repo_mock = AsyncMock()
    build_repo_mock.get_latest.return_value = _make_job(BuildState.RUNNING)

    async def _fake_get_session():
        yield mock_session

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_REPO_PATH, return_value=repo_mock),
        patch(_BUILD_REPO_PATH, return_value=build_repo_mock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get(f"/scan/{SCAN_ID}/graph", headers=_auth_headers())

    assert resp.status_code == 422, resp.text


@pytest.mark.asyncio
async def test_get_graph_succeeded():
    """GET /scan/{id}/graph — succeeded → 200 GeoJSON."""
    mock_session = _make_session_mock()
    repo_mock = AsyncMock()
    repo_mock.find_existing.return_value = _existing_scan()

    job = _make_job(BuildState.SUCCEEDED)
    build_repo_mock = AsyncMock()
    build_repo_mock.get_latest.return_value = job

    graph_repo_mock = AsyncMock()
    graph_repo_mock.load_graph_geojson.return_value = []

    async def _fake_get_session():
        yield mock_session

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_REPO_PATH, return_value=repo_mock),
        patch(_BUILD_REPO_PATH, return_value=build_repo_mock),
        patch(
            "indoor_server.interfaces.api.router.MapGraphRepository",
            return_value=graph_repo_mock,
        ),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get(f"/scan/{SCAN_ID}/graph", headers=_auth_headers())

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["type"] == "FeatureCollection"
    assert "features" in body


@pytest.mark.asyncio
async def test_build_scan_not_found():
    """scan_id 없음 → 404."""
    mock_session = _make_session_mock()
    repo_mock = AsyncMock()
    repo_mock.find_existing.return_value = None

    async def _fake_get_session():
        yield mock_session

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_REPO_PATH, return_value=repo_mock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post(f"/scan/{SCAN_ID}/build", headers=_auth_headers())

    assert resp.status_code == 404, resp.text


# ── W-6: floor_z0 실제 값 반환 회귀 테스트 ───────────────────────────────────────

@pytest.mark.asyncio
async def test_get_graph_floor_z0_returned_from_counts() -> None:
    """W-6: GET /scan/{id}/graph 응답의 floor_z0가 BuildCounts.floor_z0에서 채워진다."""
    mock_session = _make_session_mock()
    repo_mock = AsyncMock()
    repo_mock.find_existing.return_value = _existing_scan()

    expected_z0 = 0.042
    counts = BuildCounts(floor_z0=expected_z0)
    job = BuildJob(
        build_job_id=uuid4(),
        scan_id=uuid4(),
        state=BuildState.SUCCEEDED,
        enqueued_at=datetime.now(tz=UTC),
        counts=counts,
    )
    build_repo_mock = AsyncMock()
    build_repo_mock.get_latest.return_value = job

    graph_repo_mock = AsyncMock()
    graph_repo_mock.load_graph_geojson.return_value = []

    async def _fake_get_session():
        yield mock_session

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_REPO_PATH, return_value=repo_mock),
        patch(_BUILD_REPO_PATH, return_value=build_repo_mock),
        patch(
            "indoor_server.interfaces.api.router.MapGraphRepository",
            return_value=graph_repo_mock,
        ),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get(f"/scan/{SCAN_ID}/graph", headers=_auth_headers())

    assert resp.status_code == 200, resp.text
    body = resp.json()
    props = body.get("properties", {})
    assert props.get("floor_z0") == pytest.approx(expected_z0, abs=1e-9), (
        f"floor_z0={props.get('floor_z0')!r} should equal {expected_z0}"
    )


@pytest.mark.asyncio
async def test_get_graph_floor_z0_null_when_counts_none() -> None:
    """W-6: counts가 None이면 floor_z0는 null을 반환한다."""
    mock_session = _make_session_mock()
    repo_mock = AsyncMock()
    repo_mock.find_existing.return_value = _existing_scan()

    job = BuildJob(
        build_job_id=uuid4(),
        scan_id=uuid4(),
        state=BuildState.SUCCEEDED,
        enqueued_at=datetime.now(tz=UTC),
        counts=None,  # counts 없음
    )
    build_repo_mock = AsyncMock()
    build_repo_mock.get_latest.return_value = job

    graph_repo_mock = AsyncMock()
    graph_repo_mock.load_graph_geojson.return_value = []

    async def _fake_get_session():
        yield mock_session

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_REPO_PATH, return_value=repo_mock),
        patch(_BUILD_REPO_PATH, return_value=build_repo_mock),
        patch(
            "indoor_server.interfaces.api.router.MapGraphRepository",
            return_value=graph_repo_mock,
        ),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get(f"/scan/{SCAN_ID}/graph", headers=_auth_headers())

    assert resp.status_code == 200, resp.text
    body = resp.json()
    props = body.get("properties", {})
    assert props.get("floor_z0") is None
