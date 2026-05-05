"""POST /scan/upload 통합 테스트 (DB mock 사용)."""
from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from indoor_server.application.sidecar_parser import SidecarParser
from indoor_server.application.zip_unpacker import UnpackedScan
from indoor_server.domain.scan.models import ExistingScan, ScanIngestResult
from indoor_server.main import app
from tests.fixtures.make_sidecar import make_v4_sidecar
from tests.fixtures.make_zip import make_scan_zip

SCAN_ID = "33333333-3333-3333-3333-333333333333"
TOKEN = "dev-token"

_REPO_PATH = "indoor_server.application.scan_ingest_service.ScanIngestRepository"
_STORAGE_COMMIT = "indoor_server.infrastructure.storage.local_fs.LocalFileStorage.commit"
_UNPACK_PATH = "indoor_server.application.zip_unpacker.ZipUnpacker.unpack"
_PARSE_PATH = "indoor_server.application.scan_ingest_service.SidecarParser.parse"
# Sprint 49: SidecarParser.get_user_version 도 mock (manifest contract 검증용)
_VERSION_PATH = "indoor_server.application.scan_ingest_service.SidecarParser.get_user_version"
# W-1: _maybe_enqueue가 BuildJobRepository를 참조하므로 mock 경로 추가
_BUILD_JOB_REPO_PATH = "indoor_server.application.scan_ingest_service.BuildJobRepository"


def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


def _minimal_result(scan_id: str = SCAN_ID, state: str = "ingested") -> ScanIngestResult:
    return ScanIngestResult(
        scan_id=scan_id,
        state=state,
        keyframe_count=3,
        poi_marks_track_lock=1,
        poi_marks_manual=1,
        poi_photo_count=2,
        branch_mark_count=1,
        yolo_detection_count=1,
        storage_path=f"scans/{scan_id}",
        payload_sha256="abc123" * 10 + "abcd",
    )


def _existing_scan(sha256: str = "abc123" * 10 + "abcd") -> ExistingScan:
    return ExistingScan(
        scan_id=SCAN_ID,
        payload_sha256=sha256,
        ingested_at=datetime.now(tz=UTC),
    )


def _fake_unpacked(sha256: str = "abc123" * 10 + "abcd") -> UnpackedScan:
    return UnpackedScan(
        scan_id=SCAN_ID,
        rtabmap_db_path=Path("/tmp/rtabmap.db"),
        sidecar_db_path=Path("/tmp/scan_metadata.db"),
        keyframe_paths=[],
        payload_sha256=sha256,
    )


def _make_mock_session() -> AsyncMock:
    session = AsyncMock()
    # begin()을 여러 번 호출해도 동작하도록 context manager mock 설정
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    session.begin.return_value = ctx
    return session


def _make_repo_mock(*, find_existing_return: ExistingScan | None = None) -> AsyncMock:
    repo = AsyncMock()
    repo.find_existing.return_value = find_existing_return
    repo.ingest = AsyncMock()
    repo.delete = AsyncMock()
    return repo


def _make_build_job_repo_mock(*, latest_return: object = None) -> AsyncMock:
    """W-1: _maybe_enqueue 내부에서 쓰는 BuildJobRepository mock."""
    from datetime import UTC, datetime
    from uuid import uuid4

    from indoor_server.domain.building.enums import BuildState
    from indoor_server.domain.building.models import BuildJob

    repo = AsyncMock()
    repo.get_latest.return_value = latest_return
    # enqueue 시 반환할 가짜 job
    fake_job = BuildJob(
        build_job_id=uuid4(),
        scan_id=uuid4(),
        state=BuildState.PENDING,
        enqueued_at=datetime.now(tz=UTC),
    )
    repo.create.return_value = fake_job
    return repo


def _make_real_contents():
    """실제 v4 sidecar에서 SidecarContents를 파싱해 반환하는 helper."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        make_v4_sidecar(Path(db_path), SCAN_ID)
        return SidecarParser().parse(Path(db_path), SCAN_ID)
    finally:
        os.unlink(db_path)


@pytest.fixture
def zip_bytes(tmp_path) -> bytes:
    zip_path = make_scan_zip(tmp_path, SCAN_ID)
    return zip_path.read_bytes()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upload_success(zip_bytes):
    """정상 업로드 → 200 ingested."""
    mock_session = _make_mock_session()
    repo_instance = _make_repo_mock(find_existing_return=None)
    contents = _make_real_contents()

    async def _fake_get_session():
        yield mock_session

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_REPO_PATH, return_value=repo_instance),
        patch(_UNPACK_PATH, return_value=_fake_unpacked()),
        patch(_PARSE_PATH, return_value=contents),
        patch(_VERSION_PATH, return_value=4),
        patch(_STORAGE_COMMIT, new_callable=AsyncMock) as mock_commit,
    ):
        mock_commit.return_value = f"scans/{SCAN_ID}"

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/scan/upload",
                headers=_auth_headers(),
                files={"payload": (f"{SCAN_ID}.zip", zip_bytes, "application/zip")},
                data={"scan_id": SCAN_ID},
            )

    assert resp.status_code == 200, f"unexpected: {resp.status_code} {resp.text}"
    body = resp.json()
    assert body["scan_id"] == SCAN_ID
    assert body["state"] == "ingested"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upload_idempotent_same_sha256(zip_bytes):
    """DoD #4: 동일 scan_id + 동일 sha256 → 200 ingested (idempotent)."""
    fixed_sha = "abc123" * 10 + "abcd"
    mock_session = _make_mock_session()
    existing = _existing_scan(sha256=fixed_sha)
    repo_instance = _make_repo_mock(find_existing_return=existing)
    contents = _make_real_contents()

    async def _fake_get_session():
        yield mock_session

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_REPO_PATH, return_value=repo_instance),
        patch(_UNPACK_PATH, return_value=_fake_unpacked(sha256=fixed_sha)),
        patch(_PARSE_PATH, return_value=contents),
        patch(_VERSION_PATH, return_value=4),
        patch(_STORAGE_COMMIT, new_callable=AsyncMock) as mock_commit,
    ):
        mock_commit.return_value = f"scans/{SCAN_ID}"

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/scan/upload",
                headers=_auth_headers(),
                files={"payload": (f"{SCAN_ID}.zip", zip_bytes, "application/zip")},
                data={"scan_id": SCAN_ID},
            )

    assert resp.status_code == 200, f"unexpected: {resp.status_code} {resp.text}"
    body = resp.json()
    assert body["state"] == "ingested"
    # idempotent 경로에서는 repo.ingest가 호출되지 않아야 함
    repo_instance.ingest.assert_not_called()


@pytest.mark.asyncio
async def test_upload_conflict_different_sha256(zip_bytes):
    """DoD #5: 동일 scan_id + 다른 sha256 + force=false → 409 Conflict."""
    mock_session = _make_mock_session()
    existing = _existing_scan(sha256="different_sha" + "x" * 51)
    repo_instance = _make_repo_mock(find_existing_return=existing)
    contents = _make_real_contents()

    async def _fake_get_session():
        yield mock_session

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_REPO_PATH, return_value=repo_instance),
        patch(_UNPACK_PATH, return_value=_fake_unpacked(sha256="incoming_sha" + "y" * 52)),
        patch(_PARSE_PATH, return_value=contents),
        patch(_VERSION_PATH, return_value=4),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/scan/upload",
                headers=_auth_headers(),
                files={"payload": (f"{SCAN_ID}.zip", zip_bytes, "application/zip")},
                data={"scan_id": SCAN_ID},
            )

    assert resp.status_code == 409, f"unexpected: {resp.status_code} {resp.text}"
    body = resp.json()
    assert body["code"] == "SCAN_CONFLICT"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upload_force_true_replaces(zip_bytes):
    """DoD #6: force=true → 200 replaced."""
    incoming_sha = "incoming_sha256" + "y" * 49
    mock_session = _make_mock_session()
    existing = _existing_scan(sha256="different_sha256" + "x" * 48)
    repo_instance = _make_repo_mock(find_existing_return=existing)
    contents = _make_real_contents()

    async def _fake_get_session():
        yield mock_session

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_REPO_PATH, return_value=repo_instance),
        patch(_UNPACK_PATH, return_value=_fake_unpacked(sha256=incoming_sha)),
        patch(_PARSE_PATH, return_value=contents),
        patch(_VERSION_PATH, return_value=4),
        patch(_STORAGE_COMMIT, new_callable=AsyncMock) as mock_commit,
    ):
        mock_commit.return_value = f"scans/{SCAN_ID}"

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/scan/upload?force=true",
                headers=_auth_headers(),
                files={"payload": (f"{SCAN_ID}.zip", zip_bytes, "application/zip")},
                data={"scan_id": SCAN_ID},
            )

    assert resp.status_code == 200, f"unexpected: {resp.status_code} {resp.text}"
    body = resp.json()
    assert body["state"] == "replaced"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upload_force_true_replaces_even_same_sha256(zip_bytes):
    """force=true이면 같은 payload라도 storage까지 다시 commit한다."""
    fixed_sha = "abc123" * 10 + "abcd"
    mock_session = _make_mock_session()
    existing = _existing_scan(sha256=fixed_sha)
    repo_instance = _make_repo_mock(find_existing_return=existing)
    contents = _make_real_contents()

    async def _fake_get_session():
        yield mock_session

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_REPO_PATH, return_value=repo_instance),
        patch(_UNPACK_PATH, return_value=_fake_unpacked(sha256=fixed_sha)),
        patch(_PARSE_PATH, return_value=contents),
        patch(_VERSION_PATH, return_value=4),
        patch(_STORAGE_COMMIT, new_callable=AsyncMock) as mock_commit,
    ):
        mock_commit.return_value = f"scans/{SCAN_ID}"

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/scan/upload?force=true",
                headers=_auth_headers(),
                files={"payload": (f"{SCAN_ID}.zip", zip_bytes, "application/zip")},
                data={"scan_id": SCAN_ID},
            )

    assert resp.status_code == 200, f"unexpected: {resp.status_code} {resp.text}"
    body = resp.json()
    assert body["state"] == "replaced"
    repo_instance.ingest.assert_awaited_once()
    assert repo_instance.ingest.await_args.kwargs["replace"] is True
    mock_commit.assert_awaited_once()
    assert mock_commit.await_args.kwargs["replace"] is True


@pytest.mark.asyncio
async def test_file_rename_failure_triggers_db_rollback(zip_bytes):
    """F-1 회귀: 파일 commit 실패 시 DB 보상 삭제(delete)가 호출되는지 검증."""
    mock_session = _make_mock_session()
    repo_instance = _make_repo_mock(find_existing_return=None)
    contents = _make_real_contents()

    async def _fake_get_session():
        yield mock_session

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_REPO_PATH, return_value=repo_instance),
        patch(_UNPACK_PATH, return_value=_fake_unpacked()),
        patch(_PARSE_PATH, return_value=contents),
        patch(_VERSION_PATH, return_value=4),
        patch(_STORAGE_COMMIT, new_callable=AsyncMock, side_effect=OSError("disk full")),
    ):
        # raise_app_exceptions=False: 서버 내부 예외를 ASGITransport가 500으로 처리하도록
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/scan/upload",
                headers=_auth_headers(),
                files={"payload": (f"{SCAN_ID}.zip", zip_bytes, "application/zip")},
                data={"scan_id": SCAN_ID},
            )

    # 파일 commit 실패 → 500 반환
    assert resp.status_code == 500, f"expected 500 got {resp.status_code}"
    # 보상 delete가 호출되어야 함
    repo_instance.delete.assert_called_once_with(SCAN_ID)


@pytest.mark.asyncio
async def test_upload_413_is_api_error_format(zip_bytes, tmp_path):
    """W-7 회귀: 413 응답이 ApiError 포맷(code, message)인지 검증."""
    # max_upload_bytes를 1바이트로 제한해 작은 파일도 413 발생
    with patch("indoor_server.interfaces.api.router.settings") as mock_settings:
        mock_settings.tmp_root = tmp_path
        mock_settings.max_upload_bytes = 1
        mock_settings.storage_root = tmp_path

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/scan/upload",
                headers=_auth_headers(),
                files={"payload": (f"{SCAN_ID}.zip", zip_bytes, "application/zip")},
                data={"scan_id": SCAN_ID},
            )

    assert resp.status_code == 413
    body = resp.json()
    # ApiError 포맷: top-level에 code, message 필드가 있어야 함
    assert "code" in body, f"ApiError format violation: {body}"
    assert body["code"] == "PAYLOAD_TOO_LARGE"
    assert "message" in body


@pytest.mark.asyncio
async def test_upload_422_invalid_scan_id_is_api_error_format(zip_bytes):
    """W-7 회귀: 422 (invalid scan_id) 응답이 ApiError 포맷인지 검증."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/scan/upload",
            headers=_auth_headers(),
            files={"payload": ("test.zip", zip_bytes, "application/zip")},
            data={"scan_id": "not-a-uuid"},
        )

    assert resp.status_code == 422
    body = resp.json()
    # ApiError 포맷: top-level에 code, message 필드가 있어야 함
    assert "code" in body, f"ApiError format violation: {body}"
    assert body["code"] == "INVALID_SCAN_ID"
    assert "message" in body


@pytest.mark.asyncio
async def test_upload_missing_token(zip_bytes):
    """토큰 없음 → 401."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/scan/upload",
            files={"payload": (f"{SCAN_ID}.zip", zip_bytes, "application/zip")},
            data={"scan_id": SCAN_ID},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_upload_wrong_token(zip_bytes):
    """잘못된 토큰 → 401."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/scan/upload",
            headers={"Authorization": "Bearer wrong-token"},
            files={"payload": (f"{SCAN_ID}.zip", zip_bytes, "application/zip")},
            data={"scan_id": SCAN_ID},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_upload_invalid_scan_id(zip_bytes):
    """scan_id가 UUID 형식 아님 → 422."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/scan/upload",
            headers=_auth_headers(),
            files={"payload": ("test.zip", zip_bytes, "application/zip")},
            data={"scan_id": "not-a-uuid"},
        )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_upload_invalid_zip():
    """유효하지 않은 zip → 400."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/scan/upload",
            headers=_auth_headers(),
            files={"payload": (f"{SCAN_ID}.zip", b"not a zip", "application/zip")},
            data={"scan_id": SCAN_ID},
        )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_healthz():
    """GET /healthz → 200."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
