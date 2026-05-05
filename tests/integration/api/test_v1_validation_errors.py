"""Sprint 84 V1 request validation error envelope tests."""
from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient, Response

from indoor_server.infrastructure.db.engine import get_session
from indoor_server.main import app

FLOOR_ID = UUID("bbbbbbbb-1111-2222-3333-bbbbbbbbbbbb")


async def _fake_get_session():
    yield AsyncMock()


def _assert_v1_validation_error(resp: Response, *, loc: list[str]) -> None:
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["code"] == "VALIDATION_ERROR"
    assert body["message"] == "request validation failed"
    assert "errors" in body["detail"]
    assert body["detail"]["errors"][0]["loc"] == loc


@pytest.mark.asyncio
async def test_v1_body_validation_uses_top_level_error_envelope() -> None:
    app.dependency_overrides[get_session] = _fake_get_session
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/buildings", json={})
    finally:
        app.dependency_overrides.clear()

    _assert_v1_validation_error(resp, loc=["body", "name"])


@pytest.mark.asyncio
async def test_v1_path_validation_uses_top_level_error_envelope() -> None:
    app.dependency_overrides[get_session] = _fake_get_session
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/v1/floors/not-a-uuid/path")
    finally:
        app.dependency_overrides.clear()

    _assert_v1_validation_error(resp, loc=["path", "floorId"])


@pytest.mark.asyncio
async def test_v1_query_validation_uses_top_level_error_envelope() -> None:
    app.dependency_overrides[get_session] = _fake_get_session
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/api/v1/floors/{FLOOR_ID}/scans/chunks?force=not-a-bool"
            )
    finally:
        app.dependency_overrides.clear()

    _assert_v1_validation_error(resp, loc=["query", "force"])
