"""Sprint 84 V1 POI API tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

from indoor_server.infrastructure.db.engine import get_session
from indoor_server.interfaces.api.v1_schemas import POIResponse
from indoor_server.main import app

BUILDING_ID = UUID("11111111-aaaa-bbbb-cccc-111111111111")
POI_ID = UUID("22222222-aaaa-bbbb-cccc-222222222222")
NODE_ID = UUID("33333333-aaaa-bbbb-cccc-333333333333")


async def _fake_get_session():
    yield AsyncMock()


@pytest.mark.asyncio
async def test_v1_poi_search_returns_camel_case_route_node() -> None:
    service = AsyncMock()
    service.search_pois.return_value = [
        POIResponse(
            poi_id=POI_ID,
            building_id=BUILDING_ID,
            name="A101",
            label="A101",
            category="room",
            route_node_id=NODE_ID,
        )
    ]
    app.dependency_overrides[get_session] = _fake_get_session
    try:
        with patch(
            "indoor_server.interfaces.api.v1_router.POICatalogService",
            return_value=service,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.get(
                    f"/api/v1/buildings/{BUILDING_ID}/pois/search?query=a101"
                )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body[0]["poiId"] == str(POI_ID)
    assert body[0]["routeNodeId"] == str(NODE_ID)
    service.search_pois.assert_awaited_once_with(BUILDING_ID, "a101")
