"""Regression coverage for IMDF export — unit polygon 제거 후."""
from __future__ import annotations

import io
import json
import zipfile
from asyncio import run
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

import pytest
from shapely.geometry import Polygon, mapping

from indoor_server.application.imdf.export_service import ImdfExportService
from indoor_server.domain.building.enums import BuildState
from indoor_server.domain.building.models import BuildCounts, BuildJob
from indoor_server.domain.routing.models import MapEdgeRow, MapNodeRow
from indoor_server.domain.scan.models import ExistingScan

_SCAN_ID = "11111111-1111-1111-1111-111111111111"
_BUILD_JOB_ID = UUID("22222222-2222-2222-2222-222222222222")


async def _build_fixture_archive() -> Any:
    scan_repo = _FakeScanRepo()
    build_repo = _FakeBuildRepo()
    graph_repo = _FakeGraphRepo()
    service = ImdfExportService(
        scan_repo=cast(Any, scan_repo),
        build_repo=cast(Any, build_repo),
        graph_repo=cast(Any, graph_repo),
    )
    return await service.build_archive(scan_id=_SCAN_ID)


def _build_fixture_archive_sync() -> Any:
    scan_repo = _FakeScanRepo()
    build_repo = _FakeBuildRepo()
    graph_repo = _FakeGraphRepo()
    service = ImdfExportService(
        scan_repo=cast(Any, scan_repo),
        build_repo=cast(Any, build_repo),
        graph_repo=cast(Any, graph_repo),
    )
    return run(service.build_archive(scan_id=_SCAN_ID))


class _FakeScanRepo:
    async def find_existing(self, scan_id: str) -> ExistingScan | None:
        assert scan_id == _SCAN_ID
        return ExistingScan(
            scan_id=scan_id,
            payload_sha256="a" * 64,
            ingested_at=datetime(2026, 4, 25, tzinfo=UTC),
        )


class _FakeBuildRepo:
    async def get_latest(self, scan_id: str) -> BuildJob | None:
        assert scan_id == _SCAN_ID
        return BuildJob(
            build_job_id=_BUILD_JOB_ID,
            scan_id=UUID(scan_id),
            state=BuildState.SUCCEEDED,
            enqueued_at=datetime(2026, 4, 25, tzinfo=UTC),
            counts=BuildCounts(
                floor_z0=0.0,
                footprint_geojson=_fixture_footprint(),
            ),
        )


class _FakeGraphRepo:
    async def load_graph_for_routing(
        self,
        scan_id: str,
    ) -> tuple[list[MapNodeRow], list[MapEdgeRow], UUID]:
        assert scan_id == _SCAN_ID
        return _fixture_nodes(), _fixture_edges(), _BUILD_JOB_ID


def _fixture_footprint() -> dict[str, object]:
    polygon = Polygon([
        (0.0, -2.0),
        (30.0, -2.0),
        (30.0, 7.0),
        (0.0, 7.0),
        (0.0, -2.0),
    ])
    return cast(dict[str, object], mapping(polygon))


def _fixture_nodes() -> list[MapNodeRow]:
    corridor_points = [
        ("00000000-0000-0000-0000-000000000001", 2.0, 0.0),
        ("00000000-0000-0000-0000-000000000002", 12.0, 0.0),
        ("00000000-0000-0000-0000-000000000003", 22.0, 0.0),
        ("00000000-0000-0000-0000-000000000004", 28.0, 0.0),
    ]
    poi_points = [
        ("00000000-0000-0000-0000-000000000101", 6.0, 0.0, 1, "301호"),
        ("00000000-0000-0000-0000-000000000102", 10.0, 0.0, 2, "302호"),
        ("00000000-0000-0000-0000-000000000103", 15.0, 0.0, 3, "연구실"),
        ("00000000-0000-0000-0000-000000000104", 20.0, 0.0, 4, "계단"),
        ("00000000-0000-0000-0000-000000000105", 24.0, 0.0, 5, "엘리베이터"),
    ]
    nodes = [
        MapNodeRow(
            node_id=UUID(node_id),
            x=x,
            y=y,
            z=0.0,
            node_type="corridor",
            label=None,
            poi_mark_id=None,
        )
        for node_id, x, y in corridor_points
    ]
    nodes.extend(
        MapNodeRow(
            node_id=UUID(node_id),
            x=x,
            y=y,
            z=0.0,
            node_type="poi",
            label=label,
            poi_mark_id=poi_mark_id,
        )
        for node_id, x, y, poi_mark_id, label in poi_points
    )
    return nodes


def _fixture_edges() -> list[MapEdgeRow]:
    edge_specs = [
        (
            "00000000-0000-0000-0000-000000000201",
            "00000000-0000-0000-0000-000000000001",
            "00000000-0000-0000-0000-000000000002",
            10.0,
        ),
        (
            "00000000-0000-0000-0000-000000000202",
            "00000000-0000-0000-0000-000000000002",
            "00000000-0000-0000-0000-000000000003",
            10.0,
        ),
        (
            "00000000-0000-0000-0000-000000000203",
            "00000000-0000-0000-0000-000000000003",
            "00000000-0000-0000-0000-000000000004",
            6.0,
        ),
    ]
    return [
        MapEdgeRow(
            edge_id=UUID(edge_id),
            from_node_id=UUID(from_node_id),
            to_node_id=UUID(to_node_id),
            length_m=length_m,
        )
        for edge_id, from_node_id, to_node_id, length_m in edge_specs
    ]


def _parse_imdf_archive(payload: bytes) -> dict[str, dict[str, object]]:
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        return {
            "manifest": json.loads(zf.read("manifest.json")),
            "unit_geojson": json.loads(zf.read("unit.geojson")),
            "footprint_geojson": json.loads(zf.read("footprint.geojson")),
            "amenity_geojson": json.loads(zf.read("amenity.geojson")),
        }


# ── 회귀 테스트 ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_imdf_export_unit_geojson_empty() -> None:
    """unit polygon 완전 제거 — unit.geojson.features 항상 빈 배열."""
    archive = await _build_fixture_archive()
    parsed = _parse_imdf_archive(archive.payload)

    assert parsed["unit_geojson"]["features"] == []


@pytest.mark.asyncio
async def test_imdf_export_footprint_present() -> None:
    """footprint.geojson은 1개 feature를 포함해야 한다."""
    archive = await _build_fixture_archive()
    parsed = _parse_imdf_archive(archive.payload)

    assert len(parsed["footprint_geojson"]["features"]) == 1
    assert parsed["footprint_geojson"]["features"][0]["properties"]["feature_type"] == "footprint"


@pytest.mark.asyncio
async def test_imdf_export_amenity_count() -> None:
    """amenity.geojson은 POI 노드 수만큼 feature를 포함해야 한다."""
    archive = await _build_fixture_archive()
    parsed = _parse_imdf_archive(archive.payload)

    # fixture에 POI 5개
    assert len(parsed["amenity_geojson"]["features"]) == 5


@pytest.mark.asyncio
async def test_imdf_export_is_deterministic() -> None:
    """동일 build_job에 대해 두 번 export해도 unit.geojson은 동일 (빈 배열)."""
    first = _parse_imdf_archive((await _build_fixture_archive()).payload)
    second = _parse_imdf_archive((await _build_fixture_archive()).payload)

    assert first["unit_geojson"] == second["unit_geojson"]
    assert first["footprint_geojson"]["features"][0]["geometry"] == \
        second["footprint_geojson"]["features"][0]["geometry"]


@pytest.mark.asyncio
async def test_imdf_export_manifest_no_unit_split() -> None:
    """unit_split 제거 — manifest에 unit_split 키가 없어야 한다."""
    archive = await _build_fixture_archive()
    parsed = _parse_imdf_archive(archive.payload)

    assert "unit_split" not in parsed["manifest"]
    assert parsed["manifest"]["format"] == "indoor-pathfinding-imdf-lite"
