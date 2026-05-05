"""Sprint 84 V1 localization adapter API tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

from indoor_server.infrastructure.db.engine import get_session
from indoor_server.interfaces.api.v1_schemas import LocalizeResponse
from indoor_server.main import app

BUILDING_ID = UUID("44444444-aaaa-bbbb-cccc-444444444444")
SCAN_ID = "55555555-aaaa-bbbb-cccc-555555555555"


async def _fake_get_session():
    yield AsyncMock()


@pytest.mark.asyncio
async def test_v1_localize_accepts_multipart_images() -> None:
    adapter = AsyncMock()
    adapter.localize.return_value = LocalizeResponse(
        building_id=BUILDING_ID,
        map_id=SCAN_ID,
        pose={"x": 0.0, "y": 0.0, "z": 0.0, "floorLevel": 1},
        confidence=0.5,
        candidates=[{"mapId": SCAN_ID, "score": 0.5}],
    )
    app.dependency_overrides[get_session] = _fake_get_session
    try:
        with patch(
            "indoor_server.interfaces.api.v1_router.LocalizationAdapter",
            return_value=adapter,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.post(
                    f"/api/v1/buildings/{BUILDING_ID}/localize",
                    files=[
                        ("images", ("frame-1.jpg", b"jpg", "image/jpeg")),
                        ("images", ("frame-2.jpg", b"jpg", "image/jpeg")),
                    ],
                )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["mapId"] == SCAN_ID
    assert body["pose"]["floorLevel"] == 1
    adapter.localize.assert_awaited_once()
