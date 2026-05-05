"""Sprint 84 V1 scan compatibility wrapper tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

from indoor_server.infrastructure.db.engine import get_session
from indoor_server.interfaces.api.v1_schemas import ScanChunkResponse
from indoor_server.main import app

FLOOR_ID = UUID("bbbbbbbb-1111-2222-3333-bbbbbbbbbbbb")
SCAN_ID = UUID("cccccccc-1111-2222-3333-cccccccccccc")
CHUNK_ID = UUID("dddddddd-1111-2222-3333-dddddddddddd")


async def _fake_get_session():
    yield AsyncMock()


@pytest.mark.asyncio
async def test_v1_scan_chunk_rejects_raw_db_upload() -> None:
    app.dependency_overrides[get_session] = _fake_get_session
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                f"/api/v1/floors/{FLOOR_ID}/scans/chunks",
                files={"file": ("rtabmap.db", b"sqlite", "application/octet-stream")},
            )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 400, resp.text
    assert resp.json() == {
        "code": "ZIP_ARCHIVE_REQUIRED",
        "message": "V1 chunk wrapper accepts only zip scan archives.",
        "detail": {"filename": "rtabmap.db"},
    }


@pytest.mark.asyncio
async def test_v1_scan_chunk_accepts_zip_field_file() -> None:
    service = AsyncMock()
    service.upload_archive.return_value = ScanChunkResponse(
        chunk_id=CHUNK_ID,
        floor_id=FLOOR_ID,
        scan_id=SCAN_ID,
        file_name="scan.zip",
        file_size=12,
        status="UPLOADED",
        active=True,
        upload_order=1,
    )
    app.dependency_overrides[get_session] = _fake_get_session
    try:
        with patch(
            "indoor_server.interfaces.api.v1_router.ScanCompatService",
            return_value=service,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.post(
                    f"/api/v1/floors/{FLOOR_ID}/scans/chunks",
                    files={"file": ("scan.zip", b"zip", "application/zip")},
                    data={"scan_id": str(SCAN_ID)},
                )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["chunkId"] == str(CHUNK_ID)
    assert body["scanId"] == str(SCAN_ID)
    service.upload_archive.assert_awaited_once()
