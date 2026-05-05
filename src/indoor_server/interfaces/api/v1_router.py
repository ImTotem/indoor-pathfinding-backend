"""V1 compatibility router.

These endpoints keep the original `/api/v1/*` surface as thin adapters over the
current zip ingest, auto build, semantic POI, and route APIs.
"""
from __future__ import annotations

from typing import Annotated, Any, NoReturn
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.application.api_v1.building_floor_service import BuildingFloorService
from indoor_server.application.api_v1.errors import V1ServiceError
from indoor_server.application.api_v1.localization_adapter import LocalizationAdapter
from indoor_server.application.api_v1.pathfinding_adapter import PathfindingAdapter
from indoor_server.application.api_v1.poi_catalog_service import POICatalogService
from indoor_server.application.api_v1.scan_compat_service import ScanCompatService
from indoor_server.infrastructure.db.engine import get_session
from indoor_server.interfaces.api.v1_schemas import (
    BuildingCreateRequest,
    BuildingDetailResponse,
    BuildingResponse,
    BuildingStatusRequest,
    BuildingUpdateRequest,
    FloorCoordinateRouteRequest,
    FloorCoordinateRouteResponse,
    FloorCreateRequest,
    FloorPathResponse,
    FloorResponse,
    FloorUpdateRequest,
    LocalizeResponse,
    MergedScanResponse,
    MergeScansRequest,
    NodeImagesRequest,
    NodeImagesResponse,
    PathfindingRequest,
    PathfindingResponse,
    POICreateRequest,
    POIResponse,
    ProcessingStatusResponse,
    ScanChunkResponse,
    SlamMetadataResponse,
    SlamStatusResponse,
    V1ErrorResponse,
    VerticalPassageResponse,
)

v1_router = APIRouter(prefix="/api/v1")

USER_APP_TAG = "사용자 앱 API"
V1_TAG_BUILDINGS = "V1 - 건물"
V1_TAG_FLOORS = "V1 - 층"
V1_TAG_MAP_DATA = "V1 - 지도 데이터"
V1_TAG_SCAN_PROCESSING = "V1 - 스캔/처리"
V1_TAG_ROUTE = "V1 - 길찾기"
V1_TAG_LOCALIZATION = "V1 - 위치추정"
V1_TAG_SLAM = "V1 - SLAM"
V1_TAG_POI = "V1 - POI"
V1_TAG_PASSAGES = "V1 - 통로"

_V1_ERRORS: dict[int | str, dict[str, Any]] = {
    400: {"model": V1ErrorResponse},
    404: {"model": V1ErrorResponse},
    422: {"model": V1ErrorResponse},
    503: {"model": V1ErrorResponse},
}


@v1_router.get(
    "/buildings",
    response_model=list[BuildingResponse],
    responses=_V1_ERRORS,
    tags=[USER_APP_TAG, V1_TAG_BUILDINGS],
)
async def list_buildings(
    status_filter: str | None = Query(None, alias="status"),
    session: AsyncSession = Depends(get_session),
) -> list[BuildingResponse]:
    try:
        return await BuildingFloorService(session).list_buildings(status_filter=status_filter)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.post(
    "/buildings",
    response_model=BuildingResponse,
    status_code=status.HTTP_201_CREATED,
    responses=_V1_ERRORS,
    tags=[V1_TAG_BUILDINGS],
)
async def create_building(
    request: BuildingCreateRequest,
    session: AsyncSession = Depends(get_session),
) -> BuildingResponse:
    try:
        return await BuildingFloorService(session).create_building(request)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.get(
    "/buildings/{buildingId}",
    response_model=BuildingDetailResponse,
    responses=_V1_ERRORS,
    tags=[USER_APP_TAG, V1_TAG_BUILDINGS],
)
async def get_building(
    building_id: Annotated[UUID, Path(alias="buildingId")],
    session: AsyncSession = Depends(get_session),
) -> BuildingDetailResponse:
    try:
        return await BuildingFloorService(session).get_building(building_id)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.put(
    "/buildings/{buildingId}",
    response_model=BuildingResponse,
    responses=_V1_ERRORS,
    tags=[V1_TAG_BUILDINGS],
)
async def update_building(
    building_id: Annotated[UUID, Path(alias="buildingId")],
    request: BuildingUpdateRequest,
    session: AsyncSession = Depends(get_session),
) -> BuildingResponse:
    try:
        return await BuildingFloorService(session).update_building(building_id, request)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.delete(
    "/buildings/{buildingId}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=_V1_ERRORS,
    tags=[V1_TAG_BUILDINGS],
)
async def delete_building(
    building_id: Annotated[UUID, Path(alias="buildingId")],
    session: AsyncSession = Depends(get_session),
) -> None:
    try:
        await BuildingFloorService(session).delete_building(building_id)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.patch(
    "/buildings/{buildingId}/status",
    response_model=BuildingResponse,
    responses=_V1_ERRORS,
    tags=[V1_TAG_BUILDINGS],
)
async def patch_building_status(
    building_id: Annotated[UUID, Path(alias="buildingId")],
    request: BuildingStatusRequest,
    session: AsyncSession = Depends(get_session),
) -> BuildingResponse:
    try:
        return await BuildingFloorService(session).patch_status(building_id, request)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.get(
    "/buildings/{buildingId}/floors",
    response_model=list[FloorResponse],
    responses=_V1_ERRORS,
    tags=[V1_TAG_FLOORS],
)
async def list_floors(
    building_id: Annotated[UUID, Path(alias="buildingId")],
    session: AsyncSession = Depends(get_session),
) -> list[FloorResponse]:
    try:
        return await BuildingFloorService(session).list_floors(building_id)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.post(
    "/buildings/{buildingId}/floors",
    response_model=FloorResponse,
    status_code=status.HTTP_201_CREATED,
    responses=_V1_ERRORS,
    tags=[V1_TAG_FLOORS],
)
async def create_floor(
    building_id: Annotated[UUID, Path(alias="buildingId")],
    request: FloorCreateRequest,
    session: AsyncSession = Depends(get_session),
) -> FloorResponse:
    try:
        return await BuildingFloorService(session).create_floor(building_id, request)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.get(
    "/floors/{floorId}",
    response_model=FloorResponse,
    responses=_V1_ERRORS,
    tags=[V1_TAG_FLOORS],
)
async def get_floor(
    floor_id: Annotated[UUID, Path(alias="floorId")],
    session: AsyncSession = Depends(get_session),
) -> FloorResponse:
    try:
        return await BuildingFloorService(session).get_floor(floor_id)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.put(
    "/floors/{floorId}",
    response_model=FloorResponse,
    responses=_V1_ERRORS,
    tags=[V1_TAG_FLOORS],
)
async def update_floor(
    floor_id: Annotated[UUID, Path(alias="floorId")],
    request: FloorUpdateRequest,
    session: AsyncSession = Depends(get_session),
) -> FloorResponse:
    try:
        return await BuildingFloorService(session).update_floor(floor_id, request)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.delete(
    "/floors/{floorId}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=_V1_ERRORS,
    tags=[V1_TAG_FLOORS],
)
async def delete_floor(
    floor_id: Annotated[UUID, Path(alias="floorId")],
    session: AsyncSession = Depends(get_session),
) -> None:
    try:
        await BuildingFloorService(session).delete_floor(floor_id)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.get(
    "/floors/{floorId}/path",
    response_model=FloorPathResponse,
    responses=_V1_ERRORS,
    tags=[V1_TAG_MAP_DATA],
)
async def get_floor_path(
    floor_id: Annotated[UUID, Path(alias="floorId")],
    session: AsyncSession = Depends(get_session),
) -> FloorPathResponse:
    try:
        return await BuildingFloorService(session).get_floor_path(floor_id)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.get(
    "/floors/{floorId}/route",
    responses=_V1_ERRORS,
    tags=[V1_TAG_ROUTE],
)
async def get_floor_route(
    floor_id: Annotated[UUID, Path(alias="floorId")],
    from_node: Annotated[UUID, Query(alias="from")],
    to_node: Annotated[UUID, Query(alias="to")],
    session: AsyncSession = Depends(get_session),
) -> Any:
    """Sprint 78 A-7: corridor-only Dijkstra + leaf endpoint 허용 라우팅."""
    from indoor_server.application.routing.floor_route_service import (
        FloorRouteError,
        FloorRouteService,
    )

    try:
        return await FloorRouteService(session).find_route(
            floor_id=floor_id,
            from_node_id=from_node,
            to_node_id=to_node,
        )
    except FloorRouteError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={"code": e.code, "message": e.message},
        ) from e


@v1_router.get(
    "/floors/{floorId}/pointcloud",
    responses={**_V1_ERRORS, 200: {"content": {"application/octet-stream": {}}}},
    tags=[V1_TAG_MAP_DATA],
)
async def get_floor_pointcloud(
    floor_id: Annotated[UUID, Path(alias="floorId")],
    session: AsyncSession = Depends(get_session),
) -> FileResponse:
    try:
        path = await BuildingFloorService(session).pointcloud_path(floor_id)
        return FileResponse(path)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.post(
    "/floors/{floorId}/scans/chunks",
    response_model=ScanChunkResponse,
    status_code=status.HTTP_201_CREATED,
    responses=_V1_ERRORS,
    tags=[V1_TAG_SCAN_PROCESSING],
)
async def upload_scan_chunk(
    floor_id: Annotated[UUID, Path(alias="floorId")],
    file: UploadFile | None = File(None),
    payload: UploadFile | None = File(None),
    scan_id: str | None = Form(None),
    device_info: str | None = Form(None),
    force: bool = False,
    session: AsyncSession = Depends(get_session),
) -> ScanChunkResponse:
    upload = file or payload
    if upload is None:
        _raise_v1(V1ServiceError(400, "FILE_REQUIRED", "file or payload is required"))
    try:
        return await ScanCompatService(session).upload_archive(
            floor_id=floor_id,
            upload=upload,
            scan_id=scan_id,
            device_info=device_info,
            force=force,
        )
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.get(
    "/floors/{floorId}/scans/chunks",
    response_model=list[ScanChunkResponse],
    responses=_V1_ERRORS,
    tags=[V1_TAG_SCAN_PROCESSING],
)
async def list_scan_chunks(
    floor_id: Annotated[UUID, Path(alias="floorId")],
    session: AsyncSession = Depends(get_session),
) -> list[ScanChunkResponse]:
    try:
        return await ScanCompatService(session).list_chunks(floor_id)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.delete(
    "/floors/{floorId}/scans/chunks/{chunkId}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=_V1_ERRORS,
    tags=[V1_TAG_SCAN_PROCESSING],
)
async def delete_scan_chunk(
    floor_id: Annotated[UUID, Path(alias="floorId")],
    chunk_id: Annotated[UUID, Path(alias="chunkId")],
    session: AsyncSession = Depends(get_session),
) -> None:
    try:
        await ScanCompatService(session).delete_chunk(floor_id, chunk_id)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.post(
    "/floors/{floorId}/scans/merge",
    response_model=MergedScanResponse,
    responses=_V1_ERRORS,
    tags=[V1_TAG_SCAN_PROCESSING],
)
async def merge_scans(
    floor_id: Annotated[UUID, Path(alias="floorId")],
    request: MergeScansRequest,
    session: AsyncSession = Depends(get_session),
) -> MergedScanResponse:
    try:
        return await ScanCompatService(session).merge(floor_id, request.chunk_ids)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.get(
    "/floors/{floorId}/scans/merge/status",
    response_model=MergedScanResponse,
    responses=_V1_ERRORS,
    tags=[V1_TAG_SCAN_PROCESSING],
)
async def get_merge_status(
    floor_id: Annotated[UUID, Path(alias="floorId")],
    session: AsyncSession = Depends(get_session),
) -> MergedScanResponse:
    try:
        return await ScanCompatService(session).merge_status(floor_id)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.post(
    "/floors/{floorId}/process",
    response_model=ProcessingStatusResponse,
    responses=_V1_ERRORS,
    tags=[V1_TAG_SCAN_PROCESSING],
)
async def process_floor(
    floor_id: Annotated[UUID, Path(alias="floorId")],
    session: AsyncSession = Depends(get_session),
) -> ProcessingStatusResponse:
    try:
        return await ScanCompatService(session).process(floor_id)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.get(
    "/floors/{floorId}/process/status",
    response_model=ProcessingStatusResponse,
    responses=_V1_ERRORS,
    tags=[V1_TAG_SCAN_PROCESSING],
)
async def get_process_status(
    floor_id: Annotated[UUID, Path(alias="floorId")],
    session: AsyncSession = Depends(get_session),
) -> ProcessingStatusResponse:
    try:
        return await ScanCompatService(session).process_status(floor_id)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.post(
    "/buildings/{buildingId}/pathfinding",
    response_model=PathfindingResponse,
    responses=_V1_ERRORS,
    tags=[V1_TAG_ROUTE],
)
async def post_pathfinding(
    building_id: Annotated[UUID, Path(alias="buildingId")],
    request: PathfindingRequest,
    session: AsyncSession = Depends(get_session),
) -> PathfindingResponse:
    try:
        return await PathfindingAdapter(session).compute(
            building_id=building_id,
            request=request,
        )
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.post(
    "/buildings/{buildingId}/floors/{floorId}/routes/coordinates",
    response_model=FloorCoordinateRouteResponse,
    responses=_V1_ERRORS,
    tags=[USER_APP_TAG, V1_TAG_ROUTE],
)
async def post_floor_coordinate_route(
    building_id: Annotated[UUID, Path(alias="buildingId")],
    floor_id: Annotated[UUID, Path(alias="floorId")],
    request: FloorCoordinateRouteRequest,
    session: AsyncSession = Depends(get_session),
) -> FloorCoordinateRouteResponse:
    try:
        return await PathfindingAdapter(session).compute_floor_coordinate_route(
            building_id=building_id,
            floor_id=floor_id,
            request=request,
        )
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.post(
    "/buildings/{buildingId}/localize",
    response_model=LocalizeResponse,
    responses=_V1_ERRORS,
    tags=[V1_TAG_LOCALIZATION],
)
async def post_localize(
    building_id: Annotated[UUID, Path(alias="buildingId")],
    images: list[UploadFile] = File(...),
    session: AsyncSession = Depends(get_session),
) -> LocalizeResponse:
    try:
        image_bytes = [await image.read() for image in images]
        return await LocalizationAdapter(session).localize(
            building_id=building_id,
            images=image_bytes,
        )
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.get(
    "/buildings/{buildingId}/slam/status",
    response_model=SlamStatusResponse,
    responses=_V1_ERRORS,
    tags=[V1_TAG_SLAM],
)
async def get_slam_status(
    building_id: Annotated[UUID, Path(alias="buildingId")],
    session: AsyncSession = Depends(get_session),
) -> SlamStatusResponse:
    try:
        return await LocalizationAdapter(session).slam_status(building_id)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.get(
    "/buildings/{buildingId}/slam/metadata",
    response_model=SlamMetadataResponse,
    responses=_V1_ERRORS,
    tags=[V1_TAG_SLAM],
)
async def get_slam_metadata(
    building_id: Annotated[UUID, Path(alias="buildingId")],
    session: AsyncSession = Depends(get_session),
) -> SlamMetadataResponse:
    try:
        return await LocalizationAdapter(session).slam_metadata(building_id)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.post(
    "/buildings/{buildingId}/node-images",
    response_model=NodeImagesResponse,
    responses=_V1_ERRORS,
    tags=[V1_TAG_LOCALIZATION],
)
async def post_node_images(
    building_id: Annotated[UUID, Path(alias="buildingId")],
    request: NodeImagesRequest,
    session: AsyncSession = Depends(get_session),
) -> NodeImagesResponse:
    try:
        return await LocalizationAdapter(session).node_images(
            building_id=building_id,
            request=request,
        )
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.get(
    "/buildings/{buildingId}/pois",
    response_model=list[POIResponse],
    responses=_V1_ERRORS,
    tags=[USER_APP_TAG, V1_TAG_POI],
)
async def list_pois(
    building_id: Annotated[UUID, Path(alias="buildingId")],
    session: AsyncSession = Depends(get_session),
) -> list[POIResponse]:
    try:
        return await POICatalogService(session).list_pois(building_id)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.post(
    "/buildings/{buildingId}/pois",
    response_model=POIResponse,
    status_code=status.HTTP_201_CREATED,
    responses=_V1_ERRORS,
    tags=[V1_TAG_POI],
)
async def create_poi(
    building_id: Annotated[UUID, Path(alias="buildingId")],
    request: POICreateRequest,
    session: AsyncSession = Depends(get_session),
) -> POIResponse:
    try:
        return await POICatalogService(session).create_poi(building_id, request)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.get(
    "/buildings/{buildingId}/pois/search",
    response_model=list[POIResponse],
    responses=_V1_ERRORS,
    tags=[USER_APP_TAG, V1_TAG_POI],
)
async def search_pois(
    building_id: Annotated[UUID, Path(alias="buildingId")],
    query: str | None = None,
    session: AsyncSession = Depends(get_session),
) -> list[POIResponse]:
    try:
        return await POICatalogService(session).search_pois(building_id, query)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.get(
    "/buildings/{buildingId}/passages",
    response_model=list[VerticalPassageResponse],
    responses=_V1_ERRORS,
    tags=[V1_TAG_PASSAGES],
)
async def list_passages(
    building_id: Annotated[UUID, Path(alias="buildingId")],
    session: AsyncSession = Depends(get_session),
) -> list[VerticalPassageResponse]:
    try:
        return await BuildingFloorService(session).list_passages(building_id)
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.get(
    "/passages/{passageId}",
    response_model=VerticalPassageResponse,
    responses=_V1_ERRORS,
    tags=[V1_TAG_PASSAGES],
)
async def get_passage(
    passage_id: Annotated[UUID, Path(alias="passageId")],
    session: AsyncSession = Depends(get_session),
) -> VerticalPassageResponse:
    try:
        return await BuildingFloorService(session).get_passage(passage_id)
    except V1ServiceError as e:
        _raise_v1(e)


def _raise_v1(error: V1ServiceError) -> NoReturn:
    raise HTTPException(
        status_code=error.status_code,
        detail={"code": error.code, "message": error.message, "detail": error.detail},
    )
