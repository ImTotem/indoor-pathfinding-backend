"""Sprint 84 V1 service/adapter compatibility tests."""
from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi import UploadFile

from indoor_server.application.api_v1 import (
    localization_adapter,
    pathfinding_adapter,
    scan_compat_service,
)
from indoor_server.application.api_v1.building_floor_service import ActiveScan
from indoor_server.application.api_v1.errors import V1ServiceError
from indoor_server.application.api_v1.pathfinding_adapter import PathfindingAdapter
from indoor_server.application.api_v1.poi_catalog_service import POICatalogService
from indoor_server.application.api_v1.scan_compat_service import ScanCompatService
from indoor_server.domain.routing.models import SnapInfo
from indoor_server.domain.semantic.models import SemanticAnalysis
from indoor_server.interfaces.api.schemas import RouteEndpoint
from indoor_server.interfaces.api.v1_schemas import (
    FloorCoordinateRouteRequest,
    FloorResponse,
    PathfindingRequest,
    POICreateRequest,
)

BUILDING_ID = UUID("aaaaaaaa-1111-2222-3333-aaaaaaaaaaaa")
FLOOR_ID = UUID("bbbbbbbb-1111-2222-3333-bbbbbbbbbbbb")
SCAN_ID = UUID("cccccccc-1111-2222-3333-cccccccccccc")
OTHER_SCAN_ID = UUID("dddddddd-1111-2222-3333-dddddddddddd")
NODE_ID = UUID("eeeeeeee-1111-2222-3333-eeeeeeeeeeee")
GOAL_NODE_ID = UUID("ffffffff-1111-2222-3333-ffffffffffff")


class _Result:
    def __init__(self, *, rows: list[object] | None = None, scalar: object | None = None) -> None:
        self._rows = rows or []
        self._scalar = scalar

    def fetchall(self) -> list[object]:
        return self._rows

    def first(self) -> object | None:
        return self._rows[0] if self._rows else None

    def scalar_one(self) -> object:
        return self._scalar


class _QueuedSession:
    def __init__(self, results: list[_Result]) -> None:
        self._results = results
        self.statements: list[object] = []

    async def execute(self, statement: object) -> _Result:
        self.statements.append(statement)
        if not self._results:
            return _Result()
        return self._results.pop(0)

    async def commit(self) -> None:
        return None


def _active_scan(scan_id: UUID = SCAN_ID, floor_level: int = 1) -> ActiveScan:
    return ActiveScan(
        scan_id=str(scan_id),
        floor_id=str(FLOOR_ID),
        building_id=str(BUILDING_ID),
        floor_level=floor_level,
        floor_name=f"{floor_level}F",
    )


@pytest.mark.asyncio
async def test_pathfinding_adapter_delegates_to_route_service_with_active_scan_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    active_scans = [_active_scan(SCAN_ID, 1), _active_scan(OTHER_SCAN_ID, 2)]

    class _BuildingService:
        async def get_active_scans_for_building(self, building_id: UUID) -> list[ActiveScan]:
            assert building_id == BUILDING_ID
            return active_scans

    class _POIService:
        async def destination_endpoint(
            self,
            *,
            building_id: UUID,
            destination_name: str,
        ) -> RouteEndpoint:
            assert building_id == BUILDING_ID
            assert destination_name == "A101"
            return RouteEndpoint(node_id=GOAL_NODE_ID)

    class _RouteService:
        def __init__(self, graph_loader: object) -> None:
            captured["graph_loader"] = graph_loader

        async def compute(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return SimpleNamespace(
                length_m=7.2,
                route_metadata={},
                nodes_in_order=[
                    SimpleNamespace(
                        node_id=NODE_ID,
                        scan_id=str(SCAN_ID),
                        level_id=f"floor:{FLOOR_ID}",
                        node_type="poi",
                        label="A101",
                        x=1.0,
                        y=2.0,
                        z=0.0,
                        source_ref={},
                    )
                ],
            )

    monkeypatch.setattr(
        pathfinding_adapter,
        "BuildingFloorService",
        lambda session: _BuildingService(),
    )
    monkeypatch.setattr(pathfinding_adapter, "POICatalogService", lambda session: _POIService())
    monkeypatch.setattr(pathfinding_adapter, "GraphLoader", lambda session: "loader")
    monkeypatch.setattr(pathfinding_adapter, "RouteService", _RouteService)

    response = await PathfindingAdapter(AsyncMock()).compute(
        building_id=BUILDING_ID,
        request=PathfindingRequest(
            startFloorLevel=1,
            startX=1.0,
            startY=2.0,
            startZ=0.0,
            destinationName="A101",
            preference="ELEVATOR_FIRST",
        ),
    )

    assert captured["scan_id"] == str(SCAN_ID)
    assert captured["scan_ids"] == [str(SCAN_ID), str(OTHER_SCAN_ID)]
    assert captured["merge_overlaps"] is False
    assert captured["start"] == RouteEndpoint(coordinate=(1.0, 2.0, 0.0))
    assert captured["goal"] == RouteEndpoint(node_id=GOAL_NODE_ID)
    assert response.route_metadata["preference_ignored"] is True


@pytest.mark.asyncio
async def test_floor_coordinate_route_resolves_active_scan_and_delegates_to_route_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    active = _active_scan(SCAN_ID, 1)

    class _BuildingService:
        async def get_floor(self, floor_id: UUID) -> FloorResponse:
            assert floor_id == FLOOR_ID
            return FloorResponse(
                floor_id=FLOOR_ID,
                building_id=BUILDING_ID,
                name="1F",
                level=1,
                active_scan_id=SCAN_ID,
            )

        async def get_active_scan_for_floor(self, floor_id: UUID) -> ActiveScan | None:
            assert floor_id == FLOOR_ID
            return active

    class _RouteService:
        def __init__(self, graph_loader: object) -> None:
            captured["graph_loader"] = graph_loader

        async def compute(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return SimpleNamespace(
                polyline=[(0.0, 0.0, 0.0), (5.0, 1.0, 0.0)],
                length_m=5.4,
                nodes_in_order=[
                    SimpleNamespace(node_id=NODE_ID),
                    SimpleNamespace(node_id=GOAL_NODE_ID),
                ],
                snap=SnapInfo(start_snap_distance_m=0.2, goal_snap_distance_m=0.4),
                route_metadata={"route_scope": "single_scan"},
            )

    monkeypatch.setattr(
        pathfinding_adapter,
        "BuildingFloorService",
        lambda session: _BuildingService(),
    )
    monkeypatch.setattr(pathfinding_adapter, "GraphLoader", lambda session: "loader")
    monkeypatch.setattr(pathfinding_adapter, "RouteService", _RouteService)

    response = await PathfindingAdapter(AsyncMock()).compute_floor_coordinate_route(
        building_id=BUILDING_ID,
        floor_id=FLOOR_ID,
        request=FloorCoordinateRouteRequest(
            start={"x": 0.0, "y": 0.0},
            goal={"x": 5.0, "y": 1.0, "z": 0.0},
        ),
    )

    assert captured["scan_id"] == str(SCAN_ID)
    assert captured["scan_ids"] is None
    assert captured["merge_overlaps"] is False
    assert captured["start"] == RouteEndpoint(coordinate=(0.0, 0.0, 0.0))
    assert captured["goal"] == RouteEndpoint(coordinate=(5.0, 1.0, 0.0))
    assert response.building_id == BUILDING_ID
    assert response.floor_id == FLOOR_ID
    assert response.scan_id == SCAN_ID
    assert response.path_geometry["coordinates"] == [[0.0, 0.0, 0.0], [5.0, 1.0, 0.0]]
    assert response.snap_info.start_snap_distance_m == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_floor_coordinate_route_rejects_floor_building_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other_building_id = UUID("abababab-1111-2222-3333-abababababab")

    class _BuildingService:
        async def get_floor(self, floor_id: UUID) -> FloorResponse:
            assert floor_id == FLOOR_ID
            return FloorResponse(
                floor_id=FLOOR_ID,
                building_id=other_building_id,
                name="1F",
                level=1,
            )

        async def get_active_scan_for_floor(self, floor_id: UUID) -> ActiveScan | None:
            raise AssertionError("active scan should not be resolved for mismatched floor")

    monkeypatch.setattr(
        pathfinding_adapter,
        "BuildingFloorService",
        lambda session: _BuildingService(),
    )

    with pytest.raises(V1ServiceError) as exc:
        await PathfindingAdapter(AsyncMock()).compute_floor_coordinate_route(
            building_id=BUILDING_ID,
            floor_id=FLOOR_ID,
            request=FloorCoordinateRouteRequest(
                start={"x": 0.0, "y": 0.0},
                goal={"x": 1.0, "y": 1.0},
            ),
        )

    assert exc.value.status_code == 422
    assert exc.value.code == "FLOOR_BUILDING_MISMATCH"


@pytest.mark.asyncio
async def test_floor_coordinate_route_requires_active_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BuildingService:
        async def get_floor(self, floor_id: UUID) -> FloorResponse:
            assert floor_id == FLOOR_ID
            return FloorResponse(
                floor_id=FLOOR_ID,
                building_id=BUILDING_ID,
                name="1F",
                level=1,
            )

        async def get_active_scan_for_floor(self, floor_id: UUID) -> ActiveScan | None:
            assert floor_id == FLOOR_ID
            return None

    monkeypatch.setattr(
        pathfinding_adapter,
        "BuildingFloorService",
        lambda session: _BuildingService(),
    )

    with pytest.raises(V1ServiceError) as exc:
        await PathfindingAdapter(AsyncMock()).compute_floor_coordinate_route(
            building_id=BUILDING_ID,
            floor_id=FLOOR_ID,
            request=FloorCoordinateRouteRequest(
                start={"x": 0.0, "y": 0.0},
                goal={"x": 1.0, "y": 1.0},
            ),
        )

    assert exc.value.status_code == 404
    assert exc.value.code == "ACTIVE_SCAN_NOT_FOUND"


@pytest.mark.asyncio
async def test_scan_compat_upload_invokes_ingest_and_activates_floor_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingest_calls: dict[str, object] = {}
    session = AsyncMock()
    service = ScanCompatService(session)
    activation_row = SimpleNamespace(
        floor_scan_id=UUID("12121212-1111-2222-3333-121212121212"),
        floor_id=FLOOR_ID,
        scan_id=SCAN_ID,
        file_name="scan.zip",
        file_size=128,
        status="UPLOADED",
        active=True,
        upload_order=1,
        created_at=None,
    )
    activate = AsyncMock(return_value=activation_row)
    monkeypatch.setattr(service, "_activate_floor_scan", activate)

    class _BuildingService:
        async def get_floor(self, floor_id: UUID) -> object:
            assert floor_id == FLOOR_ID
            return object()

    class _IngestService:
        def __init__(self, **kwargs: object) -> None:
            ingest_calls["init"] = kwargs

        async def ingest(self, **kwargs: object) -> object:
            ingest_calls["ingest"] = kwargs
            return SimpleNamespace(scan_id=str(SCAN_ID))

    async def _fake_save_upload(upload: UploadFile, dest: object) -> int:
        assert upload.filename == "scan.zip"
        ingest_calls["zip_path"] = dest
        return 128

    monkeypatch.setattr(
        scan_compat_service,
        "BuildingFloorService",
        lambda session: _BuildingService(),
    )
    monkeypatch.setattr(scan_compat_service, "ScanIngestService", _IngestService)
    monkeypatch.setattr(scan_compat_service, "_save_upload", _fake_save_upload)
    monkeypatch.setattr(scan_compat_service.settings, "build_auto_enqueue", False)

    response = await service.upload_archive(
        floor_id=FLOOR_ID,
        upload=UploadFile(BytesIO(b"zip"), filename="scan.zip"),
        scan_id=str(SCAN_ID),
        device_info="ios",
        force=True,
    )

    activate.assert_awaited_once_with(
        floor_id=FLOOR_ID,
        scan_id=SCAN_ID,
        file_name="scan.zip",
        file_size=128,
        status_value="UPLOADED",
    )
    assert ingest_calls["ingest"]["expected_scan_id"] == str(SCAN_ID)
    assert ingest_calls["ingest"]["device_info"] == "ios"
    assert ingest_calls["ingest"]["force"] is True
    assert response.scan_id == SCAN_ID
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_poi_sync_passes_image_blob_to_analyzer_and_writes_route_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Analyzer:
        async def analyze(
            self,
            *,
            label: str | None,
            class_name: str | None,
            image_bytes: bytes | None,
        ) -> SemanticAnalysis:
            captured["label"] = label
            captured["class_name"] = class_name
            captured["image_bytes"] = image_bytes
            return SemanticAnalysis(
                category="room",
                name="A101",
                confidence=0.9,
                analyzer="mock-llm",
            )

    session = _QueuedSession(
        [
            _Result(),
            _Result(
                rows=[
                    SimpleNamespace(
                        node_id=NODE_ID,
                        poi_mark_id=42,
                        label="node-A101",
                        source_ref={},
                        x=1.0,
                        y=2.0,
                        z=0.0,
                    )
                ]
            ),
            _Result(scalar=UUID("13131313-1111-2222-3333-131313131313")),
            _Result(),
            _Result(),
            _Result(),
        ]
    )
    service = POICatalogService(session, analyzer=_Analyzer())
    monkeypatch.setattr(
        service,
        "_floor_for_scan",
        AsyncMock(return_value=SimpleNamespace(building_id=BUILDING_ID, floor_id=FLOOR_ID)),
    )
    monkeypatch.setattr(service, "_poi_mark", AsyncMock(return_value=SimpleNamespace(label="A101")))
    monkeypatch.setattr(
        service,
        "_latest_photo",
        AsyncMock(return_value=SimpleNamespace(class_name="doorplate", image_blob=b"jpeg")),
    )

    synced = await service.sync_scan_pois(scan_id=str(SCAN_ID), build_job_id="build-1")

    assert synced == 1
    assert captured == {"label": "A101", "class_name": "doorplate", "image_bytes": b"jpeg"}
    compiled_insert = str(session.statements[2])
    assert "route_node_id" in compiled_insert


@pytest.mark.asyncio
async def test_poi_search_query_is_scoped_to_active_scans_and_non_stale_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _QueuedSession([_Result(rows=[])])
    service = POICatalogService(session)
    monkeypatch.setattr(service, "_active_scan_ids", AsyncMock(return_value=[str(SCAN_ID)]))

    rows = await service.search_pois(BUILDING_ID, "a101")

    assert rows == []
    compiled = str(session.statements[0].compile(compile_kwargs={"literal_binds": False}))
    assert "poi_canonical.scan_id IS NULL" in compiled
    assert "poi_canonical.scan_id IN" in compiled
    assert "map_node.is_stale = false" in compiled.lower()


@pytest.mark.asyncio
async def test_manual_poi_rejects_floor_outside_building() -> None:
    service = POICatalogService(_QueuedSession([_Result(rows=[])]))

    with pytest.raises(V1ServiceError) as exc:
        await service.create_poi(
            BUILDING_ID,
            POICreateRequest(name="A101", floorId=FLOOR_ID, x=0.0, y=0.0),
        )

    assert exc.value.code == "FLOOR_NOT_FOUND"
    assert exc.value.detail == {"floorId": str(FLOOR_ID), "buildingId": str(BUILDING_ID)}


def test_non_mock_vps_mode_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(localization_adapter.settings, "vps_localizer_mode", "http")

    with pytest.raises(V1ServiceError) as exc:
        localization_adapter.LocalizationAdapter(AsyncMock())

    assert exc.value.status_code == 503
    assert exc.value.code == "VPS_SERVICE_ERROR"
