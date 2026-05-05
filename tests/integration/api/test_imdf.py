"""GET /scan/{scan_id}/imdf 통합 테스트 — I1~I8."""
from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

from indoor_server.domain.building.enums import BuildState
from indoor_server.domain.building.models import BuildCounts, BuildJob
from indoor_server.domain.scan.models import ExistingScan
from indoor_server.main import app

# ── 상수 ──────────────────────────────────────────────────────────────────────

SCAN_ID = "cccccccc-cccc-cccc-cccc-cccccccccccc"
TOKEN = "dev-token"
SCAN_ID_UUID = UUID(SCAN_ID)
BUILD_JOB_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")

_REPO_PATH = "indoor_server.interfaces.api.router.ScanIngestRepository"
_BUILD_REPO_PATH = "indoor_server.interfaces.api.router.BuildJobRepository"
_GRAPH_REPO_PATH = "indoor_server.interfaces.api.router.MapGraphRepository"
_EXPORT_SERVICE_PATH = "indoor_server.interfaces.api.router.ImdfExportService"

_FOOTPRINT_GEOJSON: dict = {
    "type": "MultiPolygon",
    "coordinates": [
        [[[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0], [0.0, 0.0]]]
    ],
}


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def _existing_scan() -> ExistingScan:
    return ExistingScan(
        scan_id=SCAN_ID,
        payload_sha256="abc" * 21 + "ab",
        ingested_at=datetime.now(tz=UTC),
    )


def _make_succeeded_job(with_footprint: bool = True) -> BuildJob:
    counts = BuildCounts(
        walkable_cells=56832,
        floor_z0=1.34,
        footprint_geojson=_FOOTPRINT_GEOJSON if with_footprint else None,
    )
    return BuildJob(
        build_job_id=BUILD_JOB_ID,
        scan_id=SCAN_ID_UUID,
        state=BuildState.SUCCEEDED,
        enqueued_at=datetime.now(tz=UTC),
        counts=counts,
    )


def _make_running_job() -> BuildJob:
    return BuildJob(
        build_job_id=BUILD_JOB_ID,
        scan_id=SCAN_ID_UUID,
        state=BuildState.RUNNING,
        enqueued_at=datetime.now(tz=UTC),
    )


def _make_session_mock() -> AsyncMock:
    session = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    session.begin.return_value = ctx
    return session


async def _fake_get_session_factory():
    """get_session DI 대체 헬퍼."""
    yield _make_session_mock()


# ── I1: happy — 200 + zip 파싱 ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_i1_happy_200_zip() -> None:
    """I1: build succeeded + footprint → 200 application/zip, zip 6개 파일."""
    from indoor_server.application.imdf.builder import ImdfBuilder
    from indoor_server.application.imdf.zip_archiver import build_imdf_zip
    from indoor_server.domain.imdf.models import ImdfArchive

    # 실제 zip 생성
    builder = ImdfBuilder()
    features = builder.build(
        scan_id=SCAN_ID_UUID,
        build_job_id=BUILD_JOB_ID,
        footprint_geojson=_FOOTPRINT_GEOJSON,
        poi_nodes=[],
        poi_labels={},
        floor_z0=1.34,
        generated_at=datetime.now(tz=UTC),
    )
    zip_bytes = build_imdf_zip(features)
    archive = ImdfArchive(
        scan_id=SCAN_ID_UUID,
        build_job_id=BUILD_JOB_ID,
        payload=zip_bytes,
        filename=f"{SCAN_ID}.imdf.zip",
    )

    export_mock = AsyncMock()
    export_mock.build_archive = AsyncMock(return_value=archive)

    with (
        patch(
            "indoor_server.interfaces.api.router.get_session",
            return_value=_fake_get_session_factory(),
        ),
        patch(_EXPORT_SERVICE_PATH, return_value=export_mock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get(f"/scan/{SCAN_ID}/imdf", headers=_auth_headers())

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/zip"
    assert f"{SCAN_ID}.imdf.zip" in resp.headers["content-disposition"]

    # zip 내부 검증
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))

    assert len(names) == 6
    assert "manifest.json" in names
    assert manifest["scan_id"] == SCAN_ID


# ── I2: 404 SCAN_NOT_FOUND ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_i2_scan_not_found() -> None:
    """I2: scan_id 없음 → ImdfScanNotFoundError → 404 SCAN_NOT_FOUND."""
    from indoor_server.domain.imdf.errors import ImdfScanNotFoundError

    export_mock = AsyncMock()
    export_mock.build_archive = AsyncMock(side_effect=ImdfScanNotFoundError(SCAN_ID))

    with (
        patch(
            "indoor_server.interfaces.api.router.get_session",
            return_value=_fake_get_session_factory(),
        ),
        patch(_EXPORT_SERVICE_PATH, return_value=export_mock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get(f"/scan/{SCAN_ID}/imdf", headers=_auth_headers())

    assert resp.status_code == 404, resp.text
    body = resp.json()
    assert body["detail"]["code"] == "SCAN_NOT_FOUND"


# ── I3: 422 GRAPH_NOT_READY ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_i3_graph_not_ready() -> None:
    """I3: build 미완료 → ImdfGraphNotReadyError → 422 GRAPH_NOT_READY."""
    from indoor_server.domain.imdf.errors import ImdfGraphNotReadyError

    export_mock = AsyncMock()
    export_mock.build_archive = AsyncMock(
        side_effect=ImdfGraphNotReadyError(SCAN_ID, "running")
    )

    with (
        patch(
            "indoor_server.interfaces.api.router.get_session",
            return_value=_fake_get_session_factory(),
        ),
        patch(_EXPORT_SERVICE_PATH, return_value=export_mock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get(f"/scan/{SCAN_ID}/imdf", headers=_auth_headers())

    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["detail"]["code"] == "GRAPH_NOT_READY"


# ── I4: 422 FOOTPRINT_NOT_AVAILABLE ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_i4_footprint_not_available() -> None:
    """I4: build succeeded but footprint NULL → FootprintNotAvailableError → 422."""
    from indoor_server.domain.imdf.errors import FootprintNotAvailableError

    export_mock = AsyncMock()
    export_mock.build_archive = AsyncMock(
        side_effect=FootprintNotAvailableError(SCAN_ID, str(BUILD_JOB_ID))
    )

    with (
        patch(
            "indoor_server.interfaces.api.router.get_session",
            return_value=_fake_get_session_factory(),
        ),
        patch(_EXPORT_SERVICE_PATH, return_value=export_mock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get(f"/scan/{SCAN_ID}/imdf", headers=_auth_headers())

    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["detail"]["code"] == "FOOTPRINT_NOT_AVAILABLE"


# ── I5: POI 0개 → amenity.features == [] ────────────────────────────────────


@pytest.mark.asyncio
async def test_i5_no_poi_amenity_empty() -> None:
    """I5: POI 0개 → 200 OK, zip 내부 amenity.geojson.features == []."""
    from indoor_server.application.imdf.builder import ImdfBuilder
    from indoor_server.application.imdf.zip_archiver import build_imdf_zip
    from indoor_server.domain.imdf.models import ImdfArchive

    builder = ImdfBuilder()
    features = builder.build(
        scan_id=SCAN_ID_UUID,
        build_job_id=BUILD_JOB_ID,
        footprint_geojson=_FOOTPRINT_GEOJSON,
        poi_nodes=[],
        poi_labels={},
        floor_z0=1.34,
        generated_at=datetime.now(tz=UTC),
    )
    zip_bytes = build_imdf_zip(features)
    archive = ImdfArchive(
        scan_id=SCAN_ID_UUID,
        build_job_id=BUILD_JOB_ID,
        payload=zip_bytes,
        filename=f"{SCAN_ID}.imdf.zip",
    )

    export_mock = AsyncMock()
    export_mock.build_archive = AsyncMock(return_value=archive)

    with (
        patch(
            "indoor_server.interfaces.api.router.get_session",
            return_value=_fake_get_session_factory(),
        ),
        patch(_EXPORT_SERVICE_PATH, return_value=export_mock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get(f"/scan/{SCAN_ID}/imdf", headers=_auth_headers())

    assert resp.status_code == 200, resp.text

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        amenity = json.loads(zf.read("amenity.geojson").decode("utf-8"))

    assert amenity["features"] == []


# ── I6: 401 인증 실패 ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_i6_unauthorized() -> None:
    """I6: Bearer 없음 → 401."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get(f"/scan/{SCAN_ID}/imdf")

    assert resp.status_code == 401, resp.text


# ── I7: 400 scan_id 형식 위반 ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_i7_invalid_scan_id_format() -> None:
    """I7: scan_id가 UUID 형식 아님 → 400/422."""
    with patch(
        "indoor_server.interfaces.api.router.get_session",
        return_value=_fake_get_session_factory(),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/scan/not-a-uuid/imdf", headers=_auth_headers())

    # ScanIdInvalidFormat → ScanDomainError handler → 422 (INVALID_SCAN_ID 매핑)
    assert resp.status_code in (400, 422), resp.text


# ── I8: Content-Disposition filename 확인 ───────────────────────────────────


@pytest.mark.asyncio
async def test_i8_content_disposition_filename() -> None:
    """I8: Content-Disposition header에 {scan_id}.imdf.zip 파일명 포함."""
    from indoor_server.application.imdf.builder import ImdfBuilder
    from indoor_server.application.imdf.zip_archiver import build_imdf_zip
    from indoor_server.domain.imdf.models import ImdfArchive

    builder = ImdfBuilder()
    features = builder.build(
        scan_id=SCAN_ID_UUID,
        build_job_id=BUILD_JOB_ID,
        footprint_geojson=_FOOTPRINT_GEOJSON,
        poi_nodes=[],
        poi_labels={},
        floor_z0=1.34,
        generated_at=datetime.now(tz=UTC),
    )
    zip_bytes = build_imdf_zip(features)
    expected_filename = f"{SCAN_ID}.imdf.zip"
    archive = ImdfArchive(
        scan_id=SCAN_ID_UUID,
        build_job_id=BUILD_JOB_ID,
        payload=zip_bytes,
        filename=expected_filename,
    )

    export_mock = AsyncMock()
    export_mock.build_archive = AsyncMock(return_value=archive)

    with (
        patch(
            "indoor_server.interfaces.api.router.get_session",
            return_value=_fake_get_session_factory(),
        ),
        patch(_EXPORT_SERVICE_PATH, return_value=export_mock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get(f"/scan/{SCAN_ID}/imdf", headers=_auth_headers())

    assert resp.status_code == 200, resp.text
    cd_header = resp.headers.get("content-disposition", "")
    assert expected_filename in cd_header


# ── I9: ETag 헤더 확인 ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_i9_etag_header() -> None:
    """I9: ETag 헤더가 build_job_id를 포함."""
    from indoor_server.application.imdf.builder import ImdfBuilder
    from indoor_server.application.imdf.zip_archiver import build_imdf_zip
    from indoor_server.domain.imdf.models import ImdfArchive

    builder = ImdfBuilder()
    features = builder.build(
        scan_id=SCAN_ID_UUID,
        build_job_id=BUILD_JOB_ID,
        footprint_geojson=_FOOTPRINT_GEOJSON,
        poi_nodes=[],
        poi_labels={},
        floor_z0=1.34,
        generated_at=datetime.now(tz=UTC),
    )
    zip_bytes = build_imdf_zip(features)
    archive = ImdfArchive(
        scan_id=SCAN_ID_UUID,
        build_job_id=BUILD_JOB_ID,
        payload=zip_bytes,
        filename=f"{SCAN_ID}.imdf.zip",
    )

    export_mock = AsyncMock()
    export_mock.build_archive = AsyncMock(return_value=archive)

    with (
        patch(
            "indoor_server.interfaces.api.router.get_session",
            return_value=_fake_get_session_factory(),
        ),
        patch(_EXPORT_SERVICE_PATH, return_value=export_mock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get(f"/scan/{SCAN_ID}/imdf", headers=_auth_headers())

    assert resp.status_code == 200, resp.text
    etag = resp.headers.get("etag", "")
    assert str(BUILD_JOB_ID) in etag
