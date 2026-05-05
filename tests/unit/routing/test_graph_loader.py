"""graph_loader.py 단위 테스트 — GraphNotReadyError / ScanNotFoundError 분기."""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from indoor_server.application.routing.graph_loader import GraphLoader
from indoor_server.domain.building.enums import BuildState
from indoor_server.domain.building.models import BuildCounts, BuildJob
from indoor_server.domain.routing.errors import GraphNotReadyError, ScanNotFoundError
from indoor_server.domain.routing.models import MapEdgeRow, MapNodeRow
from indoor_server.domain.scan.models import ExistingScan

_SCAN_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
_JOB_ID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
_NODE_A = UUID("11111111-1111-1111-1111-111111111111")
_NODE_B = UUID("22222222-2222-2222-2222-222222222222")
_SCAN_B_ID = "cccccccc-cccc-cccc-cccc-cccccccccccc"
_NODE_C = UUID("33333333-3333-3333-3333-333333333333")
_NODE_D = UUID("44444444-4444-4444-4444-444444444444")


def _existing_scan() -> ExistingScan:
    return ExistingScan(
        scan_id=_SCAN_ID,
        payload_sha256="x" * 64,
        ingested_at=datetime.now(tz=UTC),
    )


def _make_job(state: BuildState) -> BuildJob:
    return BuildJob(
        build_job_id=_JOB_ID,
        scan_id=UUID(_SCAN_ID),
        state=state,
        enqueued_at=datetime.now(tz=UTC),
        counts=BuildCounts(walkable_cells=1000),
    )


def _make_session_mock() -> AsyncMock:
    return AsyncMock()


# ── scan 미존재 → ScanNotFoundError ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_scan_not_found_raises() -> None:
    """scan_id 없음 → ScanNotFoundError (404 SCAN_NOT_FOUND 계약)."""
    session_mock = _make_session_mock()

    repo_mock = AsyncMock()
    repo_mock.find_existing = AsyncMock(return_value=None)

    with patch(
        "indoor_server.application.routing.graph_loader.ScanIngestRepository",
        return_value=repo_mock,
    ):
        loader = GraphLoader(session_mock)
        with pytest.raises(ScanNotFoundError) as exc_info:
            await loader.load(scan_id=_SCAN_ID)

    assert exc_info.value.scan_id == _SCAN_ID


# ── build 미완료 → GraphNotReadyError ────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_build_not_succeeded_raises() -> None:
    """build state=running → GraphNotReadyError."""
    session_mock = _make_session_mock()

    scan_repo_mock = AsyncMock()
    scan_repo_mock.find_existing = AsyncMock(return_value=_existing_scan())

    build_repo_mock = AsyncMock()
    build_repo_mock.get_latest = AsyncMock(return_value=_make_job(BuildState.RUNNING))

    with (
        patch(
            "indoor_server.application.routing.graph_loader.ScanIngestRepository",
            return_value=scan_repo_mock,
        ),
        patch(
            "indoor_server.application.routing.graph_loader.BuildJobRepository",
            return_value=build_repo_mock,
        ),
    ):
        loader = GraphLoader(session_mock)
        with pytest.raises(GraphNotReadyError) as exc_info:
            await loader.load(scan_id=_SCAN_ID)

    assert exc_info.value.state == "running"
    assert exc_info.value.build_job_id == str(_JOB_ID)


# ── 행복 경로 — graph 반환 ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_happy_returns_graph() -> None:
    """build=succeeded → networkx Graph + node_lookup + build_job_id 반환."""
    session_mock = _make_session_mock()

    scan_repo_mock = AsyncMock()
    scan_repo_mock.find_existing = AsyncMock(return_value=_existing_scan())

    build_repo_mock = AsyncMock()
    build_repo_mock.get_latest = AsyncMock(return_value=_make_job(BuildState.SUCCEEDED))

    nodes = [
        MapNodeRow(
            node_id=_NODE_A, x=0.0, y=0.0, z=1.5,
            node_type="corridor", label=None, poi_mark_id=None,
        ),
        MapNodeRow(
            node_id=_NODE_B, x=5.0, y=0.0, z=1.5,
            node_type="endpoint", label=None, poi_mark_id=None,
        ),
    ]
    edges = [
        MapEdgeRow(edge_id=uuid4(), from_node_id=_NODE_A, to_node_id=_NODE_B, length_m=5.0),
    ]

    graph_repo_mock = AsyncMock()
    graph_repo_mock.load_graph_for_routing = AsyncMock(return_value=(nodes, edges, _JOB_ID))

    with (
        patch(
            "indoor_server.application.routing.graph_loader.ScanIngestRepository",
            return_value=scan_repo_mock,
        ),
        patch(
            "indoor_server.application.routing.graph_loader.BuildJobRepository",
            return_value=build_repo_mock,
        ),
        patch(
            "indoor_server.application.routing.graph_loader.MapGraphRepository",
            return_value=graph_repo_mock,
        ),
    ):
        loader = GraphLoader(session_mock)
        g, node_lookup, build_job_id = await loader.load(scan_id=_SCAN_ID)

    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1
    assert _NODE_A in node_lookup
    assert _NODE_B in node_lookup
    assert build_job_id == _JOB_ID

    # edge weight 확인
    edge_data = g.get_edge_data(_NODE_A, _NODE_B)
    assert edge_data is not None
    assert edge_data["length_m"] == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_load_many_infers_vertical_connector_edges() -> None:
    """같은 connector_key의 elevator node가 서로 다른 scan에 있으면 transition edge 추가."""
    session_mock = _make_session_mock()

    scan_repo_mock = AsyncMock()
    scan_repo_mock.find_existing = AsyncMock(return_value=_existing_scan())

    build_repo_mock = AsyncMock()
    build_repo_mock.get_latest = AsyncMock(return_value=_make_job(BuildState.SUCCEEDED))

    nodes_a = [
        MapNodeRow(
            node_id=_NODE_A,
            x=0.0,
            y=0.0,
            z=0.0,
            node_type="corridor",
            label=None,
            poi_mark_id=None,
            scan_id=UUID(_SCAN_ID),
        ),
        MapNodeRow(
            node_id=_NODE_B,
            x=1.0,
            y=0.0,
            z=0.0,
            node_type="poi",
            label="ELEV_CENTER",
            poi_mark_id=10,
            source_ref={
                "facility_type": "elevator",
                "connector_key": "elevator:center",
            },
            scan_id=UUID(_SCAN_ID),
        ),
    ]
    nodes_b = [
        MapNodeRow(
            node_id=_NODE_C,
            x=1.0,
            y=0.0,
            z=3.0,
            node_type="poi",
            label="ELEV_CENTER",
            poi_mark_id=20,
            source_ref={
                "facility_type": "elevator",
                "connector_key": "elevator:center",
            },
            scan_id=UUID(_SCAN_B_ID),
        ),
        MapNodeRow(
            node_id=_NODE_D,
            x=2.0,
            y=0.0,
            z=3.0,
            node_type="corridor",
            label=None,
            poi_mark_id=None,
            scan_id=UUID(_SCAN_B_ID),
        ),
    ]
    edges_a = [MapEdgeRow(edge_id=uuid4(), from_node_id=_NODE_A, to_node_id=_NODE_B, length_m=1.0)]
    edges_b = [MapEdgeRow(edge_id=uuid4(), from_node_id=_NODE_C, to_node_id=_NODE_D, length_m=1.0)]

    graph_repo_mock = AsyncMock()
    graph_repo_mock.load_graph_for_routing = AsyncMock(
        side_effect=[
            (nodes_a, edges_a, _JOB_ID),
            (nodes_b, edges_b, _JOB_ID),
        ]
    )
    graph_repo_mock.load_explicit_vertical_edges = AsyncMock(return_value=[])

    with (
        patch(
            "indoor_server.application.routing.graph_loader.ScanIngestRepository",
            return_value=scan_repo_mock,
        ),
        patch(
            "indoor_server.application.routing.graph_loader.BuildJobRepository",
            return_value=build_repo_mock,
        ),
        patch(
            "indoor_server.application.routing.graph_loader.MapGraphRepository",
            return_value=graph_repo_mock,
        ),
    ):
        loader = GraphLoader(session_mock)
        g, node_lookup, build_job_ids, report = await loader.load_many(
            scan_ids=[_SCAN_ID, _SCAN_B_ID],
            merge_overlaps=False,
        )

    assert _NODE_B in node_lookup
    assert _NODE_C in node_lookup
    assert build_job_ids == [_JOB_ID, _JOB_ID]
    assert report is not None
    vertical_edge = g.get_edge_data(_NODE_B, _NODE_C)
    assert vertical_edge is not None
    assert vertical_edge["length_m"] == pytest.approx(8.0)
