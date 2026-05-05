"""POST /route 통합 테스트 — DB mock 방식 (T1~T9, Sprint 23 트랙 B).

설계 §트랙 B 테스트 케이스:
  T1 happy: coord → coord, 같은 component → 200
  T2 node_id → node_id → 200
  T3 poi_mark_id 목적지 → 200
  T4 scan_id 없음 → 404 SCAN_NOT_FOUND
  T5 build 미완료 → 422 GRAPH_NOT_READY
  T6 도달 불가능 (인위 분리 graph) → 422 PATH_NOT_FOUND
  T7 snap 거리 초과 → 422 SNAP_DISTANCE_EXCEEDED
  T8 RouteEndpoint exactly-one 위반 → 400/422
  T9 poi_mark_id 있으나 map_node 없음 → 422 POI_NODE_NOT_FOUND
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from indoor_server.domain.building.enums import BuildState
from indoor_server.domain.building.models import BuildCounts, BuildJob
from indoor_server.domain.routing.models import MapEdgeRow, MapNodeRow
from indoor_server.domain.scan.models import ExistingScan
from indoor_server.main import app

# ── 상수 ──────────────────────────────────────────────────────────────────────

SCAN_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
TOKEN = "dev-token"
SCAN_ID_UUID = UUID(SCAN_ID)

NODE_A_ID = UUID("11111111-1111-1111-1111-111111111111")
NODE_B_ID = UUID("22222222-2222-2222-2222-222222222222")
NODE_C_ID = UUID("33333333-3333-3333-3333-333333333333")  # 단절 노드
BUILD_JOB_ID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

_REPO_PATH = "indoor_server.interfaces.api.router.ScanIngestRepository"
_BUILD_REPO_PATH = "indoor_server.interfaces.api.router.BuildJobRepository"
_ROUTE_SERVICE_PATH = "indoor_server.interfaces.api.router.RouteService"


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def _existing_scan() -> ExistingScan:
    return ExistingScan(
        scan_id=SCAN_ID,
        payload_sha256="abc" * 21 + "ab",
        ingested_at=datetime.now(tz=UTC),
    )


def _make_succeeded_job() -> BuildJob:
    return BuildJob(
        build_job_id=BUILD_JOB_ID,
        scan_id=SCAN_ID_UUID,
        state=BuildState.SUCCEEDED,
        enqueued_at=datetime.now(tz=UTC),
        counts=BuildCounts(walkable_cells=56832),
    )


def _make_node(
    node_id: UUID,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 1.5,
    node_type: str = "corridor",
    poi_mark_id: int | None = None,
) -> MapNodeRow:
    return MapNodeRow(
        node_id=node_id,
        x=x,
        y=y,
        z=z,
        node_type=node_type,
        label=None,
        poi_mark_id=poi_mark_id,
    )


def _make_edge(from_id: UUID, to_id: UUID, length_m: float = 5.0) -> MapEdgeRow:
    return MapEdgeRow(
        edge_id=uuid4(),
        from_node_id=from_id,
        to_node_id=to_id,
        length_m=length_m,
    )


def _make_session_mock() -> AsyncMock:
    session = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    session.begin.return_value = ctx
    return session


def _make_route_service_mock(route_result: object) -> AsyncMock:
    """RouteService.compute() 를 route_result로 고정하는 mock."""
    svc = AsyncMock()
    svc.compute = AsyncMock(return_value=route_result)
    return svc


# ── RouteService compute 결과 픽스처 ──────────────────────────────────────────

def _make_route_path(
    nodes: list[MapNodeRow] | None = None,
    length_m: float = 10.0,
    start_snap: float | None = 0.5,
    goal_snap: float | None = None,
) -> object:
    """RoutePath 유사 객체 (MagicMock)."""
    from indoor_server.domain.routing.models import RoutePath, SnapInfo

    if nodes is None:
        nodes = [_make_node(NODE_A_ID, x=0.0), _make_node(NODE_B_ID, x=10.0)]
    polyline = [(n.x, n.y, n.z) for n in nodes]
    snap = SnapInfo(
        start_snap_distance_m=start_snap,
        goal_snap_distance_m=goal_snap,
    )
    return RoutePath(
        nodes_in_order=nodes,
        polyline=polyline,
        length_m=length_m,
        snap=snap,
        build_job_id=BUILD_JOB_ID,
    )


# ── T1: coord → coord happy path ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_t1_happy_coord_to_coord() -> None:
    """T1: coordinate → coordinate, 200 path_nodes ≥ 2, length_m > 0."""
    route = _make_route_path()
    svc_mock = _make_route_service_mock(route)

    async def _fake_get_session():
        yield _make_session_mock()

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_ROUTE_SERVICE_PATH, return_value=svc_mock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post(
                "/route",
                json={
                    "scan_id": SCAN_ID,
                    "start": {"coordinate": [0.0, 0.0, 1.5]},
                    "goal": {"coordinate": [10.0, 0.0, 1.5]},
                },
                headers=_auth_headers(),
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["path_nodes"]) >= 2
    assert body["length_m"] > 0
    assert body["snap_info"]["start_snap_distance_m"] == pytest.approx(0.5)
    assert body["snap_info"]["goal_snap_distance_m"] is None


# ── T2: node_id → node_id ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_t2_node_id_to_node_id() -> None:
    """T2: node_id → node_id, 200, 시작/끝 node_id 일치."""
    nodes = [_make_node(NODE_A_ID, x=0.0), _make_node(NODE_B_ID, x=10.0)]
    route = _make_route_path(nodes=nodes, start_snap=None, goal_snap=None)
    svc_mock = _make_route_service_mock(route)

    async def _fake_get_session():
        yield _make_session_mock()

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_ROUTE_SERVICE_PATH, return_value=svc_mock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post(
                "/route",
                json={
                    "scan_id": SCAN_ID,
                    "start": {"node_id": str(NODE_A_ID)},
                    "goal": {"node_id": str(NODE_B_ID)},
                },
                headers=_auth_headers(),
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["path_nodes"][0]["node_id"] == str(NODE_A_ID)
    assert body["path_nodes"][-1]["node_id"] == str(NODE_B_ID)
    assert body["snap_info"]["start_snap_distance_m"] is None
    assert body["snap_info"]["goal_snap_distance_m"] is None


# ── T3: poi_mark_id 목적지 ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_t3_poi_mark_id_goal() -> None:
    """T3: poi_mark_id 목적지 → 200, 마지막 RouteNode.poi_mark_id == 입력값."""
    poi_node = _make_node(NODE_B_ID, x=10.0, node_type="poi", poi_mark_id=42)
    nodes = [_make_node(NODE_A_ID, x=0.0), poi_node]
    route = _make_route_path(nodes=nodes, start_snap=0.3, goal_snap=None)
    svc_mock = _make_route_service_mock(route)

    async def _fake_get_session():
        yield _make_session_mock()

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_ROUTE_SERVICE_PATH, return_value=svc_mock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post(
                "/route",
                json={
                    "scan_id": SCAN_ID,
                    "start": {"coordinate": [0.0, 0.0, 1.5]},
                    "goal": {"poi_mark_id": 42},
                },
                headers=_auth_headers(),
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    last_node = body["path_nodes"][-1]
    assert last_node["poi_mark_id"] == 42


@pytest.mark.asyncio
async def test_multiscan_request_passes_scan_ids_and_merge_flag() -> None:
    """Sprint 62: API는 scan_ids/merge_overlaps를 RouteService로 전달한다."""
    route = _make_route_path()
    svc_mock = _make_route_service_mock(route)
    second_scan_id = "cccccccc-cccc-cccc-cccc-cccccccccccc"

    async def _fake_get_session():
        yield _make_session_mock()

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_ROUTE_SERVICE_PATH, return_value=svc_mock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post(
                "/route",
                json={
                    "scan_id": SCAN_ID,
                    "scan_ids": [SCAN_ID, second_scan_id],
                    "merge_overlaps": True,
                    "start": {"coordinate": [0.0, 0.0, 1.5]},
                    "goal": {"coordinate": [10.0, 0.0, 1.5]},
                },
                headers=_auth_headers(),
            )

    assert resp.status_code == 200, resp.text
    svc_mock.compute.assert_awaited_once()
    assert svc_mock.compute.await_args.kwargs["scan_ids"] == [SCAN_ID, second_scan_id]
    assert svc_mock.compute.await_args.kwargs["merge_overlaps"] is True


@pytest.mark.asyncio
async def test_mobile_coordinate_route_wraps_route_service() -> None:
    """사용자앱 좌표 API: start/goal coordinate만 받아 기존 RouteService로 위임."""
    route = _make_route_path(start_snap=0.25, goal_snap=0.75)
    svc_mock = _make_route_service_mock(route)
    second_scan_id = "cccccccc-cccc-cccc-cccc-cccccccccccc"

    async def _fake_get_session():
        yield _make_session_mock()

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_ROUTE_SERVICE_PATH, return_value=svc_mock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post(
                "/mobile/v1/routes/coordinates",
                json={
                    "scan_id": SCAN_ID,
                    "scan_ids": [SCAN_ID, second_scan_id],
                    "merge_overlaps": True,
                    "start_coordinate": [0.0, 0.0, 1.5],
                    "goal_coordinate": [10.0, 0.0, 1.5],
                },
                headers=_auth_headers(),
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["length_m"] == pytest.approx(10.0)
    assert body["snap_info"]["start_snap_distance_m"] == pytest.approx(0.25)
    assert body["snap_info"]["goal_snap_distance_m"] == pytest.approx(0.75)

    svc_mock.compute.assert_awaited_once()
    kwargs = svc_mock.compute.await_args.kwargs
    assert kwargs["scan_id"] == SCAN_ID
    assert kwargs["scan_ids"] == [SCAN_ID, second_scan_id]
    assert kwargs["merge_overlaps"] is True
    assert kwargs["start"].coordinate == (0.0, 0.0, 1.5)
    assert kwargs["goal"].coordinate == (10.0, 0.0, 1.5)


# ── T4: scan_id 없음 ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_t4_scan_not_found() -> None:
    """T4: scan_id 없음 → ScanNotFoundError → 404 SCAN_NOT_FOUND."""
    from indoor_server.domain.routing.errors import ScanNotFoundError

    svc_mock = AsyncMock()
    svc_mock.compute = AsyncMock(side_effect=ScanNotFoundError(SCAN_ID))

    async def _fake_get_session():
        yield _make_session_mock()

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_ROUTE_SERVICE_PATH, return_value=svc_mock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post(
                "/route",
                json={
                    "scan_id": SCAN_ID,
                    "start": {"coordinate": [0.0, 0.0, 1.5]},
                    "goal": {"coordinate": [10.0, 0.0, 1.5]},
                },
                headers=_auth_headers(),
            )

    assert resp.status_code == 404, resp.text
    body = resp.json()
    assert body["detail"]["code"] == "SCAN_NOT_FOUND"


# ── T5: build 미완료 ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_t5_graph_not_ready() -> None:
    """T5: build 미완료 → 422 GRAPH_NOT_READY."""
    from indoor_server.domain.routing.errors import GraphNotReadyError

    svc_mock = AsyncMock()
    svc_mock.compute = AsyncMock(
        side_effect=GraphNotReadyError(SCAN_ID, "running", str(BUILD_JOB_ID))
    )

    async def _fake_get_session():
        yield _make_session_mock()

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_ROUTE_SERVICE_PATH, return_value=svc_mock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post(
                "/route",
                json={
                    "scan_id": SCAN_ID,
                    "start": {"coordinate": [0.0, 0.0, 1.5]},
                    "goal": {"coordinate": [10.0, 0.0, 1.5]},
                },
                headers=_auth_headers(),
            )

    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["detail"]["code"] == "GRAPH_NOT_READY"
    assert body["detail"]["detail"]["state"] == "running"


# ── T6: 도달 불가능 ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_t6_path_not_found() -> None:
    """T6: 인위적으로 분리된 graph → 422 PATH_NOT_FOUND."""
    from indoor_server.domain.routing.errors import PathNotFoundError

    svc_mock = AsyncMock()
    svc_mock.compute = AsyncMock(
        side_effect=PathNotFoundError(str(NODE_A_ID), str(NODE_C_ID))
    )

    async def _fake_get_session():
        yield _make_session_mock()

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_ROUTE_SERVICE_PATH, return_value=svc_mock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post(
                "/route",
                json={
                    "scan_id": SCAN_ID,
                    "start": {"node_id": str(NODE_A_ID)},
                    "goal": {"node_id": str(NODE_C_ID)},
                },
                headers=_auth_headers(),
            )

    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["detail"]["code"] == "PATH_NOT_FOUND"


# ── T7: snap 거리 초과 ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_t7_snap_distance_exceeded() -> None:
    """T7: 좌표가 너무 멀리 있을 때 → 422 SNAP_DISTANCE_EXCEEDED."""
    from indoor_server.domain.routing.errors import SnapDistanceExceededError

    svc_mock = AsyncMock()
    svc_mock.compute = AsyncMock(
        side_effect=SnapDistanceExceededError(7.31, 5.0, "start")
    )

    async def _fake_get_session():
        yield _make_session_mock()

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_ROUTE_SERVICE_PATH, return_value=svc_mock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post(
                "/route",
                json={
                    "scan_id": SCAN_ID,
                    "start": {"coordinate": [999.0, 999.0, 1.5]},
                    "goal": {"coordinate": [10.0, 0.0, 1.5]},
                },
                headers=_auth_headers(),
            )

    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["detail"]["code"] == "SNAP_DISTANCE_EXCEEDED"
    assert body["detail"]["detail"]["distance_m"] == pytest.approx(7.31)
    assert body["detail"]["detail"]["threshold_m"] == pytest.approx(5.0)


# ── T8: RouteEndpoint exactly-one 위반 ───────────────────────────────────────


@pytest.mark.asyncio
async def test_t8_endpoint_exactly_one_violation() -> None:
    """T8: coordinate + node_id 동시 지정 → 422 (Pydantic validator)."""
    async def _fake_get_session():
        yield _make_session_mock()

    with patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post(
                "/route",
                json={
                    "scan_id": SCAN_ID,
                    "start": {
                        "coordinate": [0.0, 0.0, 1.5],
                        "node_id": str(NODE_A_ID),
                    },
                    "goal": {"coordinate": [10.0, 0.0, 1.5]},
                },
                headers=_auth_headers(),
            )

    # Pydantic validation error → FastAPI가 422로 변환
    assert resp.status_code == 422, resp.text


@pytest.mark.asyncio
async def test_t8_endpoint_none_provided() -> None:
    """T8: 아무것도 지정하지 않음 → 422."""
    async def _fake_get_session():
        yield _make_session_mock()

    with patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post(
                "/route",
                json={
                    "scan_id": SCAN_ID,
                    "start": {},
                    "goal": {"coordinate": [10.0, 0.0, 1.5]},
                },
                headers=_auth_headers(),
            )

    assert resp.status_code == 422, resp.text


# ── T9: poi_mark_id 입력, map_node 없음 ─────────────────────────────────────


@pytest.mark.asyncio
async def test_t9_poi_node_not_found() -> None:
    """T9: poi_mark_id 입력, map_node에 없음 → 422 POI_NODE_NOT_FOUND."""
    from indoor_server.domain.routing.errors import POINodeNotFoundError

    svc_mock = AsyncMock()
    svc_mock.compute = AsyncMock(
        side_effect=POINodeNotFoundError(999, "goal")
    )

    async def _fake_get_session():
        yield _make_session_mock()

    with (
        patch("indoor_server.interfaces.api.router.get_session", return_value=_fake_get_session()),
        patch(_ROUTE_SERVICE_PATH, return_value=svc_mock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post(
                "/route",
                json={
                    "scan_id": SCAN_ID,
                    "start": {"coordinate": [0.0, 0.0, 1.5]},
                    "goal": {"poi_mark_id": 999},
                },
                headers=_auth_headers(),
            )

    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["detail"]["code"] == "POI_NODE_NOT_FOUND"
    assert body["detail"]["detail"]["poi_mark_id"] == 999


# ── 인증 실패 ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_route_unauthorized() -> None:
    """인증 토큰 없음 → 401."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/route",
            json={
                "scan_id": SCAN_ID,
                "start": {"coordinate": [0.0, 0.0, 1.5]},
                "goal": {"coordinate": [10.0, 0.0, 1.5]},
            },
        )

    assert resp.status_code == 401, resp.text
