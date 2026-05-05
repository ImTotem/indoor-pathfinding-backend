"""Swagger/OpenAPI compatibility endpoint tests."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from indoor_server.main import app


@pytest.mark.asyncio
async def test_swagger_ui_index_uses_v3_api_docs() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get("/swagger-ui/index.html")

    assert resp.status_code == 200
    assert "Swagger UI" in resp.text
    assert "/v3/api-docs" in resp.text


@pytest.mark.asyncio
async def test_swagger_ui_short_paths_redirect_to_index() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        follow_redirects=False,
    ) as client:
        for path in ["/swagger-ui", "/swagger-ui/"]:
            resp = await client.get(path)
            assert resp.status_code in {307, 308}
            assert resp.headers["location"] == "/swagger-ui/index.html"


@pytest.mark.asyncio
async def test_v3_api_docs_alias_exposes_v1_paths_without_breaking_default_openapi() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        v3_resp = await client.get("/v3/api-docs")
        default_resp = await client.get("/openapi.json")

    assert v3_resp.status_code == 200
    assert default_resp.status_code == 200

    v3 = v3_resp.json()
    default = default_resp.json()
    for path in [
        "/api/v1/buildings",
        "/api/v1/floors/{floorId}/path",
        "/api/v1/buildings/{buildingId}/pathfinding",
        "/api/v1/buildings/{buildingId}/floors/{floorId}/routes/coordinates",
        "/api/v1/buildings/{buildingId}/localize",
        "/api/v1/buildings/{buildingId}/pois/search",
    ]:
        assert path in v3["paths"]
        assert path in default["paths"]


@pytest.mark.asyncio
async def test_v1_swagger_tags_are_domain_grouped() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get("/v3/api-docs")

    assert resp.status_code == 200
    schema = resp.json()
    declared_tags = {tag["name"] for tag in schema["tags"]}
    assert "v1-compat" not in declared_tags
    assert {
        "사용자 앱 API",
        "V1 - 건물",
        "V1 - 층",
        "V1 - 지도 데이터",
        "V1 - 스캔/처리",
        "V1 - 길찾기",
        "V1 - 위치추정",
        "V1 - SLAM",
        "V1 - POI",
        "V1 - 통로",
    } <= declared_tags

    path_tags = {
        path: next(iter(methods.values()))["tags"]
        for path, methods in schema["paths"].items()
        if path.startswith("/api/v1/")
    }
    assert all(tags != ["v1-compat"] for tags in path_tags.values())
    assert path_tags["/api/v1/buildings"] == ["사용자 앱 API", "V1 - 건물"]
    assert path_tags["/api/v1/buildings/{buildingId}/floors"] == ["V1 - 층"]
    assert path_tags["/api/v1/floors/{floorId}/path"] == ["V1 - 지도 데이터"]
    assert path_tags["/api/v1/floors/{floorId}/scans/chunks"] == ["V1 - 스캔/처리"]
    assert path_tags["/api/v1/buildings/{buildingId}/pathfinding"] == ["V1 - 길찾기"]
    assert (
        path_tags["/api/v1/buildings/{buildingId}/floors/{floorId}/routes/coordinates"]
        == ["사용자 앱 API", "V1 - 길찾기"]
    )
    assert path_tags["/api/v1/buildings/{buildingId}/localize"] == ["V1 - 위치추정"]
    assert path_tags["/api/v1/buildings/{buildingId}/slam/status"] == ["V1 - SLAM"]
    assert path_tags["/api/v1/buildings/{buildingId}/pois/search"] == ["사용자 앱 API", "V1 - POI"]
    assert path_tags["/api/v1/buildings/{buildingId}/passages"] == ["V1 - 통로"]


@pytest.mark.asyncio
async def test_swagger_user_app_api_group_contains_user_called_paths_only() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get("/v3/api-docs")

    assert resp.status_code == 200
    schema = resp.json()
    user_paths = {
        path
        for path, methods in schema["paths"].items()
        for op in methods.values()
        if "사용자 앱 API" in op.get("tags", [])
    }
    assert user_paths == {
        "/api/v1/buildings",
        "/api/v1/buildings/{buildingId}",
        "/api/slam/v3/localize",
        "/api/v1/buildings/{buildingId}/pois",
        "/api/v1/buildings/{buildingId}/pois/search",
        "/api/v1/buildings/{buildingId}/floors/{floorId}/routes/coordinates",
    }
