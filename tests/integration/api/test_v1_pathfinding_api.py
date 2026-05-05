"""Sprint 84 V1 pathfinding adapter API tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

from indoor_server.infrastructure.db.engine import get_session
from indoor_server.interfaces.api.v1_schemas import (
    FloorCoordinateRouteResponse,
    PathfindingResponse,
    PathStepResponse,
    V1Position,
    V1RouteSnapInfo,
)
from indoor_server.main import app

BUILDING_ID = UUID("eeeeeeee-1111-2222-3333-eeeeeeeeeeee")
FLOOR_ID = UUID("dddddddd-1111-2222-3333-dddddddddddd")
SCAN_ID = UUID("cccccccc-1111-2222-3333-cccccccccccc")
NODE_ID = UUID("ffffffff-1111-2222-3333-ffffffffffff")


async def _fake_get_session():
    yield AsyncMock()


@pytest.mark.asyncio
async def test_v1_pathfinding_returns_v1_shape() -> None:
    adapter = AsyncMock()
    adapter.compute.return_value = PathfindingResponse(
        building_id=BUILDING_ID,
        total_distance=12.4,
        estimated_time_seconds=11,
        steps=[
            PathStepResponse(
                step_number=1,
                floor_level=1,
                position=V1Position(x=0.0, y=0.0, z=0.0, floor_level=1),
                instruction="Proceed to corridor.",
                node_id=NODE_ID,
            )
        ],
        route_metadata={"preference_ignored": True},
    )
    app.dependency_overrides[get_session] = _fake_get_session
    try:
        with patch(
            "indoor_server.interfaces.api.v1_router.PathfindingAdapter",
            return_value=adapter,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.post(
                    f"/api/v1/buildings/{BUILDING_ID}/pathfinding",
                    json={
                        "startFloorLevel": 1,
                        "startX": 0.0,
                        "startY": 0.0,
                        "startZ": 0.0,
                        "destinationName": "A101",
                        "preference": "ELEVATOR_FIRST",
                    },
                )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["totalDistance"] == 12.4
    assert body["estimatedTimeSeconds"] == 11
    assert body["steps"][0]["floorLevel"] == 1
    adapter.compute.assert_awaited_once()


@pytest.mark.asyncio
async def test_v1_floor_coordinate_route_returns_v1_shape_without_scan_id() -> None:
    adapter = AsyncMock()
    adapter.compute_floor_coordinate_route.return_value = FloorCoordinateRouteResponse(
        building_id=BUILDING_ID,
        floor_id=FLOOR_ID,
        scan_id=SCAN_ID,
        path_geometry={"type": "LineString", "coordinates": [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]},
        length_m=4.0,
        node_count=2,
        snap_info=V1RouteSnapInfo(start_snap_distance_m=0.1, goal_snap_distance_m=0.2),
        route_metadata={"route_scope": "single_scan"},
    )
    app.dependency_overrides[get_session] = _fake_get_session
    try:
        with patch(
            "indoor_server.interfaces.api.v1_router.PathfindingAdapter",
            return_value=adapter,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.post(
                    f"/api/v1/buildings/{BUILDING_ID}/floors/{FLOOR_ID}/routes/coordinates",
                    json={
                        "start": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "goal": {"x": 4.0, "y": 0.0, "z": 0.0},
                    },
                )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["buildingId"] == str(BUILDING_ID)
    assert body["floorId"] == str(FLOOR_ID)
    assert body["scanId"] == str(SCAN_ID)
    assert body["pathGeometry"]["type"] == "LineString"
    assert body["lengthM"] == pytest.approx(4.0)
    assert body["nodeCount"] == 2
    assert body["snapInfo"]["startSnapDistanceM"] == pytest.approx(0.1)
    adapter.compute_floor_coordinate_route.assert_awaited_once()


@pytest.mark.asyncio
async def test_v1_openapi_exposes_floor_coordinate_route_path() -> None:
    data = app.openapi()
    assert (
        "/api/v1/buildings/{buildingId}/floors/{floorId}/routes/coordinates"
        in data["paths"]
    )
