"""Sprint 84 V1 building/floor compatibility API tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

from indoor_server.infrastructure.db.engine import get_session
from indoor_server.interfaces.api.v1_schemas import BuildingResponse
from indoor_server.main import app

BUILDING_ID = UUID("aaaaaaaa-1111-2222-3333-aaaaaaaaaaaa")


async def _fake_get_session():
    yield AsyncMock()


@pytest.mark.asyncio
async def test_v1_openapi_exposes_required_paths() -> None:
    data = app.openapi()
    for path in [
        "/api/v1/buildings",
        "/api/v1/floors/{floorId}/path",
        "/api/v1/buildings/{buildingId}/pathfinding",
        "/api/v1/buildings/{buildingId}/localize",
        "/api/v1/buildings/{buildingId}/pois/search",
    ]:
        assert path in data["paths"]


@pytest.mark.asyncio
async def test_list_buildings_uses_camel_case_response() -> None:
    service = AsyncMock()
    service.list_buildings.return_value = [
        BuildingResponse(
            building_id=BUILDING_ID,
            name="Engineering Hall",
            status="ACTIVE",
        )
    ]
    app.dependency_overrides[get_session] = _fake_get_session
    try:
        with patch(
            "indoor_server.interfaces.api.v1_router.BuildingFloorService",
            return_value=service,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.get("/api/v1/buildings?status=ACTIVE")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body[0]["buildingId"] == str(BUILDING_ID)
    assert body[0]["status"] == "ACTIVE"
    service.list_buildings.assert_awaited_once_with(status_filter="ACTIVE")
