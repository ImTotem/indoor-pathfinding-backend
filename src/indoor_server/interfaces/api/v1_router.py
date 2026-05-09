"""V1 compatibility router.

These endpoints keep the original `/api/v1/*` surface as thin adapters over the
current zip ingest, auto build, semantic POI, and route APIs.
"""
from __future__ import annotations

import logging
from typing import Annotated, Any, NoReturn
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Path,
    Query,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.application.api_v1.building_floor_service import BuildingFloorService
from indoor_server.application.api_v1.errors import V1ServiceError
from indoor_server.application.api_v1.feature_lookup_service import FeaturePointLookupService
from indoor_server.application.api_v1.floor_map_image_service import FloorMapImageService
from indoor_server.application.api_v1.floor_map_service import FloorMapService
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
    FeatureLookupRequest,
    FeatureLookupResponse,
    FloorMapResponse,
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

logger = logging.getLogger(__name__)

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
    "/floors/{floorId}/map.png",
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Floor polygon 의 PNG 이미지. 좌표/스케일 정보는 응답 헤더로.",
        },
        304: {"description": "If-None-Match 일치 → 변경 없음."},
        **_V1_ERRORS,
    },
    response_class=Response,
    tags=[USER_APP_TAG, V1_TAG_MAP_DATA],
    summary="floor polygon PNG 이미지 (raster)",
)
async def get_floor_map_image(
    floor_id: Annotated[UUID, Path(alias="floorId")],
    width_px: Annotated[
        int,
        Query(
            alias="widthPx",
            ge=64,
            le=4096,
            description=(
                "출력 이미지의 longest side (pixel). polygon 의 가로/세로 비율에 따라 "
                "다른 축은 자동 계산. 64~4096 범위."
            ),
        ),
    ] = 1024,
    padding_px: Annotated[
        int,
        Query(
            alias="paddingPx",
            ge=0,
            le=256,
            description="이미지 가장자리 여백 (pixel). 0~256.",
        ),
    ] = 16,
    fill: Annotated[
        str,
        Query(
            description=(
                "polygon 채움 색상. `#RRGGBB` 또는 `#RRGGBBAA` (alpha 포함). "
                "`#3399FF80` 처럼 alpha 50% 권장 — 지도 위에 overlay 시 자연스러움."
            )
        ),
    ] = "#3399FF80",
    stroke: Annotated[
        str,
        Query(
            description="polygon 외곽선 색상 (`#RRGGBB`/`#RRGGBBAA`). 빈 문자열은 외곽선 없음."
        ),
    ] = "#1A66CC",
    stroke_width: Annotated[
        int,
        Query(
            alias="strokeWidth",
            ge=0,
            le=16,
            description="외곽선 두께 (pixel). 0 = 외곽선 없음.",
        ),
    ] = 2,
    background: Annotated[
        str,
        Query(
            description=(
                "배경. `transparent` (기본) | `white` | `#RRGGBB[AA]` 직접 지정. "
                "AR overlay 면 transparent 권장, 별도 카드로 표시면 white."
            )
        ),
    ] = "transparent",
    if_none_match: Annotated[str | None, Header(alias="If-None-Match")] = None,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Floor polygon 을 PNG 라스터로 렌더링.

    `floor_polygon.geojson` 의 `floor_union` (없으면 모든 Polygon) 만 그림. 그래프
    노드/엣지는 포함하지 않음 — 그건 `/floors/{id}/map` (vector) 에서 별도로 받음.

    ## 언제 사용하나
    - 빌딩 카드 / floor 선택 화면 등 **썸네일 미리보기**.
    - AR 카메라 overlay 위에 **반투명 floor footprint** 표시 (alpha < 1).
    - vector renderer 가 없는 환경 (Email, OG image 등) 에서의 빠른 미리보기.

    ## 어떻게 사용하나
    ```
    GET /api/v1/floors/{floorId}/map.png?widthPx=512&fill=%233399FF80&background=transparent
    ```
    응답을 `UIImage(data:)` 로 즉시 표시. 좌표 매핑이 필요하면 응답 헤더 사용:

    | 헤더 | 의미 |
    |---|---|
    | `X-Map-Min-X`, `X-Map-Min-Y`, `X-Map-Max-X`, `X-Map-Max-Y` | polygon bbox (world meter) |
    | `X-Map-Width-Px`, `X-Map-Height-Px` | 실제 출력 이미지 크기 |
    | `X-Map-Scale-Px-Per-M` | 픽셀/미터 변환 상수 |
    | `X-Map-Padding-Px` | 가장자리 여백 (pixel) |
    | `ETag` | `"<buildJobId>"` — 다음 요청 `If-None-Match` |

    클라 측 좌표 매핑:
    ```swift
    // world (x, y meter) → pixel
    let scale = Double(headers["X-Map-Scale-Px-Per-M"]!)!
    let minX = Double(headers["X-Map-Min-X"]!)!
    let maxY = Double(headers["X-Map-Max-Y"]!)!
    let pad = Int(headers["X-Map-Padding-Px"]!)!
    let px = (worldX - minX) * scale + Double(pad)
    let py = (maxY - worldY) * scale + Double(pad)  // y 반전
    ```

    ## 왜 사용하나
    - **즉시 렌더**: SwiftUI `Image(uiImage:)` 한 줄. SVG/GeoJSON parser 불필요.
    - **고정 사이즈 썸네일**: vector 보다 작고 캐시 친화적.
    - **AR overlay**: 알파 채널 PNG 가 합성에 가장 단순.

    ## 쿼리 파라미터 활용 팁
    - **썸네일 (목록)**: `widthPx=256, background=white, fill=#E3F2FD, stroke=#1976D2`
    - **AR overlay**: `widthPx=1024, background=transparent, fill=#3399FF40`
    - **PDF/캡처**: `widthPx=2048, background=white`

    ## 캐싱
    - 응답 헤더 `ETag: "<buildJobId>"` + `Cache-Control: private, max-age=300`.
    - `If-None-Match` 일치 시 304. **단, 동일 build_job 이라도 쿼리 파라미터가 다르면
      이미지가 달라지므로 주의** — ETag 는 build_job 만 반영. 클라가 같은 파라미터
      세트로만 비교해야 안전.

    ## 에러
    - `404 ACTIVE_SCAN_NOT_FOUND`: 이 floor 에 active scan 없음.
    - `422 GRAPH_NOT_READY`: build 미완료.
    - `422 POLYGON_NOT_AVAILABLE`: 옛날 빌드 — `floor_polygon.geojson` 부재.
      vector endpoint `/floors/{id}/map` 는 정상 동작하므로 그걸 사용하거나 v9 재빌드.
    - `422 POLYGON_EMPTY` / `POLYGON_DEGENERATE`: polygon 자체 문제.
    - `422 INVALID_WIDTH` / `INVALID_PADDING` / `INVALID_COLOR` / `PADDING_TOO_LARGE`.
    """
    try:
        result = await FloorMapImageService(session).render(
            floor_id,
            width_px=width_px,
            padding_px=padding_px,
            fill=fill,
            stroke=stroke,
            stroke_width=stroke_width,
            background=background,
        )
    except V1ServiceError as e:
        _raise_v1(e)

    etag = f'"{result.build_job_id}"'
    headers = {
        "ETag": etag,
        "Cache-Control": "private, max-age=300",
        "X-Map-Min-X": f"{result.min_x:.6f}",
        "X-Map-Min-Y": f"{result.min_y:.6f}",
        "X-Map-Max-X": f"{result.max_x:.6f}",
        "X-Map-Max-Y": f"{result.max_y:.6f}",
        "X-Map-Width-Px": str(result.width_px),
        "X-Map-Height-Px": str(result.height_px),
        "X-Map-Scale-Px-Per-M": f"{result.scale_px_per_m:.6f}",
        "X-Map-Padding-Px": str(padding_px),
        "X-Map-Build-Job-Id": result.build_job_id,
    }
    if if_none_match is not None and if_none_match.strip() == etag:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=headers)
    return Response(content=result.png_bytes, media_type="image/png", headers=headers)


@v1_router.get(
    "/floors/{floorId}/map",
    response_model=FloorMapResponse,
    responses=_V1_ERRORS,
    tags=[USER_APP_TAG, V1_TAG_MAP_DATA],
    summary="floor 2D 지도 (polygon + 그래프, world meter)",
)
async def get_floor_map(
    floor_id: Annotated[UUID, Path(alias="floorId")],
    response: Response,
    if_none_match: Annotated[str | None, Header(alias="If-None-Match")] = None,
    session: AsyncSession = Depends(get_session),
) -> FloorMapResponse | Response:
    """Floor 2D 지도 — polygon (GeoJSON FeatureCollection) + 그래프 (nodes/edges).
    모두 **server world frame, meter** 좌표계. 측위 결과/route polyline 과 동일.

    ## 언제 사용하나
    - 사용자가 빌딩에 진입했을 때 floor 별로 1번씩 다운로드 후 캐시.
    - active scan / build 가 갱신되면 ETag 가 바뀌므로 다음 요청 시 재전송.
    - 그 외에는 `If-None-Match` 헤더로 304 받고 캐시 재사용.

    ## 어떻게 사용하나
    1. `GET /buildings/{id}` 로 floor 목록 확인.
    2. floor 별 `GET /floors/{floorId}/map` 호출 (선택: `If-None-Match: "<etag>"`).
    3. 응답:
       - `polygon`: GeoJSON FeatureCollection 그대로 SwiftUI Canvas / SceneKit 에 그림.
       - `nodes` / `edges`: 그래프 위에 마커/선 그리기.
       - `bounds` 로 viewport scale 계산:
         ```swift
         let scale = canvasW / bounds.widthM
         let pixel = CGPoint(
             x: (worldX - bounds.minX) * scale,
             y: (bounds.maxY - worldY) * scale  // y 반전
         )
         ```
    4. 측위 좌표 (`pose.tx/ty/tz`) 를 같은 변환으로 점 찍으면 끝. 별도 transform 불필요.
    5. routing 응답의 `steps[].position` 도 같은 변환으로 polyline 그림.

    ## 왜 사용하나
    - **좌표계 통일**: polygon, 그래프, 측위, route 모두 같은 world meter.
      클라는 viewport 변환만 하면 됨 (회전/투영 없음).
    - **vector 형식**: PNG 가 아니라 GeoJSON 이라 줌/스케일 자유. 디스플레이 해상도 무관.
    - **graph 포함**: passage 노드의 `connector` 필드로 엘베/계단 아이콘 자동 표시.
    - **ETag 캐싱**: build 변경 없으면 304, 트래픽 절약.

    ## 응답 해석
    - `polygon.features[].properties.kind`:
      - `floor_union`: 전체 footprint. **렌더할 때 이거 하나면 충분**.
      - `room`: 닫힌 방 (corner cycle). 디버그용.
      - `corridor`: 복도 buffer. 디버그용.
    - `nodes[].type`:
      - `corridor` / `junction`: backbone (사용자 명시 / route 전용).
      - `poi` (label 있음, 마커 표시) / `poi_attach` (foot, 화면에서 hidden 권장).
      - `passage`: 층간 연결 — `connector.type` 으로 아이콘 매핑.
    - `edges[].type`: `corridor` (굵게) | `spur` (얇게 또는 hidden).
    - `nodes[].id` 는 길찾기 요청의 nodeId 로 그대로 사용 가능 (POI 클릭 → routing).

    ## 캐싱
    - 응답 헤더: `ETag: "<buildJobId>"`, `Cache-Control: private, max-age=60`.
    - 다음 요청에 `If-None-Match: "<buildJobId>"` 보내면 변경 없을 때 **304 Not Modified**.

    ## 에러
    - `404 ACTIVE_SCAN_NOT_FOUND`: 이 floor 에 active scan 없음.
    - `422 GRAPH_NOT_READY`: build 가 아직 완료 안 됨.

    ## 빈 polygon 케이스
    옛날 빌드 (sprint83 이전) 는 `floor_polygon.geojson` 이 없어서 `polygon.features` 가
    빈 배열. 이 경우 `nodes`/`edges` 만으로 그래프 wireframe 그리기 가능.
    v9 데이터로 재빌드 시 자동 채워짐.
    """
    try:
        result = await FloorMapService(session).load(floor_id)
    except V1ServiceError as e:
        _raise_v1(e)
    etag = f'"{result.etag}"'
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "private, max-age=60"
    if if_none_match is not None and if_none_match.strip() == etag:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers={"ETag": etag})
    return result


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
    tags=[USER_APP_TAG, V1_TAG_ROUTE],
    summary="길찾기 (멀티층 지원, 엘베/계단 선택)",
)
async def post_pathfinding(
    building_id: Annotated[UUID, Path(alias="buildingId")],
    request: PathfindingRequest,
    session: AsyncSession = Depends(get_session),
) -> PathfindingResponse:
    """현재 위치에서 도착 POI 까지 최단 경로 계산. **멀티층 자동 처리**.

    ## 언제 사용하나
    사용자가 도착지를 선택했을 때 한 번 호출. 응답을 받으면 화면에 polyline 을
    그려주고, 이후 사용자 이동에 따라 클라가 alongTrack 진행률만 갱신하면 된다.
    경로 자체가 바뀌지 않으면 재호출 불필요.

    ## 어떻게 사용하나
    1. `/buildings/{id}/localize` 로 현재 위치 측위 (`pose.tx/ty/tz`, `mapId`).
    2. 이 endpoint 호출:
       - `startScanId` = localize 응답의 `mapId` (시작 floor 자동 결정)
       - `startX/Y/Z` = localize 응답의 `pose.tx/ty/tz`
       - `destinationName` = 사용자가 선택한 POI 이름 (예: '301호')
       - `verticalPreference` = `ELEVATOR` 또는 `STAIRS` (기본 ELEVATOR)
    3. 응답의 `steps[]` 좌표를 polyline 으로 그림. 좌표계는 지도와 동일 (world meter).
    4. `floorTransitions[]` 가 있으면 층 전환 안내 표시 (예: "엘리베이터 EV-A 타고 3층").

    ## 왜 사용하나
    - 멀티층 자동 라우팅: 같은 connector key 끼리 자동 cross-floor edge 생성.
    - `verticalPreference` 로 사용자 선호 반영 (엘베만 / 계단만).
    - `startScanId` 만 보내면 floor 자동 — 클라가 floor_level 직접 관리할 필요 없음.

    ## 응답 해석
    - `steps[]`: 각 노드의 (x, y, z, floorLevel) 순서대로. polyline 으로 연결.
    - `floorTransitions[]`: 층 변경 지점. `connectorType`/`connectorKey` 표시.
    - `routeMetadata.verticalPreference`: 적용된 preference echo.
    - `routeMetadata.startScanId` / `startFloorLevel`: 실제 사용된 시작 정보.

    ## 에러
    - `404 ACTIVE_SCAN_NOT_FOUND`: 빌딩에 active scan 이 하나도 없음.
    - `404 START_SCAN_NOT_FOUND`: `startScanId` 가 이 빌딩의 active scan 이 아님.
    - `404 START_FLOOR_NOT_FOUND`: `startFloorLevel` 에 active scan 없음.
    - `422 START_NOT_SPECIFIED`: `startScanId` / `startFloorLevel` 둘 다 없음.
    - `422 SNAP_DISTANCE_EXCEEDED`: 시작 좌표가 그래프에서 너무 멀음 (5m 초과).
    - `422 PATH_NOT_FOUND`: 경로 없음. `verticalPreference` 가 `STAIRS` 인데
      빌딩에 계단이 없으면 발생 — 클라는 `ELEVATOR` 로 fallback 권장.
    """
    try:
        return await PathfindingAdapter(session).compute(
            building_id=building_id,
            request=request,
        )
    except V1ServiceError as e:
        _raise_v1(e)


@v1_router.post(
    "/buildings/{buildingId}/feature-points/lookup",
    response_model=FeatureLookupResponse,
    responses=_V1_ERRORS,
    tags=[USER_APP_TAG, V1_TAG_LOCALIZATION],
    summary="좌표 배열 → 근처 keyframe SuperPoint feature pack",
)
async def post_feature_points_lookup(
    building_id: Annotated[UUID, Path(alias="buildingId")],
    request: FeatureLookupRequest,
    session: AsyncSession = Depends(get_session),
) -> FeatureLookupResponse:
    """좌표 배열 → 각 좌표 주변 keyframe 의 SuperPoint feature pack 반환.

    클라가 디바이스에서 직접 LightGlue 매칭 + PnP 로 정밀 위치를 구할 때 필요한
    raw feature 데이터를 한 번에 다운로드한다.

    ## 언제 사용하나
    - **길찾기 시작 직후 (route bundle 효과)**: 경로 위 좌표를 모두 보내면 그 영역의
      keyframe 만 다운로드. AR 안내 중 끊김 없이 매칭 가능.
    - **이동 중 정밀 보정**: 현재 위치 1개만 보내면 1~3 keyframe 짜리 작은 응답.
      매 N초 호출해서 디바이스 측 위치 보정.
    - **POI 미리보기 / 360 뷰**: 특정 POI 좌표 1개로 주변 keyframe 받아서 표시.

    ## 어떻게 사용하나
    ### 패턴 A — 경로 전체 한 번에
    ```json
    {
      "queries": [
        {"floorLevel": 1, "x": 0.0, "y": 0.0, "z": -1.4},
        {"floorLevel": 1, "x": 1.5, "y": 3.0, "z": -1.4},
        ... (route nodes 좌표 모두)
      ],
      "options": {"radiusM": 2.5, "maxKeyframesPerQuery": 5}
    }
    ```

    ### 패턴 B — 현재 위치만 (실시간)
    ```json
    {
      "queries": [
        {"floorLevel": 1, "x": 1.05, "y": 4.51, "z": -1.40,
         "viewDirection": [0.0, 0.0, -1.0]}
      ],
      "options": {"radiusM": 1.5, "maxKeyframesPerQuery": 3, "viewConeDeg": 60}
    }
    ```

    ## 왜 사용하나
    - 디바이스 측 LightGlue 매칭으로 서버 round-trip 없이 정밀 위치 갱신 (60Hz 가능).
    - dedup: 여러 query 가 같은 keyframe 시야에 들면 한 번만 응답 → 데이터 절약.
    - 같은 endpoint 로 'route bundle (전체)' 와 '실시간 보정 (작음)' 둘 다 처리.

    ## 응답 해석
    - `keyframes[]` 의 각 항목은 한 keyframe 의 모든 feature.
    - `keypoints` (N,2) f32 / `descriptors` (N,256) **f16** / `world3d` (N,3) f32 — base64.
      디코드 시 dtype 정확히 맞춰야 함 (`model.descriptorDtype` 참조).
    - `world3d` 는 **NaN 행 다수 포함** (3D 추정 실패한 keypoint). PnP 전 NaN 필터.
    - `pose` 의 마지막 열 = 카메라 위치 (world meter), 3번째 열 = forward direction.
    - `matchedQueryIndices` / `distancesM` — 어느 query 와 매칭됐는지 + 거리.
    - `globalDescriptor` (DINOv2 384 f16) 는 retrieval 단계 (top-k 후보 선정)에 사용.

    ## 가드레일
    - `queries` 최대 64개.
    - `maxKeyframesPerQuery` 최대 16.
    - dedup 후 keyframe 총합 128 초과 시 422 — 클라는 `radiusM` 줄이거나 query 분할.

    ## 에러
    - `422 EMPTY_QUERIES` / `TOO_MANY_QUERIES` / `TOO_MANY_KEYFRAMES`
    - `404 ACTIVE_SCAN_NOT_FOUND` / `FLOOR_NOT_FOUND` / `RTABMAP_DB_NOT_FOUND`

    ## 성능 주의
    - 첫 호출 시 floor 별 SuperPoint cache build 30s~1분 (300+ keyframes 기준).
      이후 cache hit 으로 즉시 응답.
    - `/admin/superpoint/warmup` 으로 사전 warm-up 가능 (build 직후 worker 가 자동 호출).
    """
    try:
        return await FeaturePointLookupService(session).lookup(
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
        sizes = [len(b) for b in image_bytes]
        filenames = [img.filename for img in images]
        content_types = [img.content_type for img in images]
        logger.info(
            "localize request: building_id=%s images=%d filenames=%s "
            "content_types=%s sizes_bytes=%s total_bytes=%d",
            building_id,
            len(image_bytes),
            filenames,
            content_types,
            sizes,
            sum(sizes),
        )
        # Query image debug dump
        try:
            import datetime as _dt
            from indoor_server.config import settings as _settings
            dump_dir = _settings.storage_root / "debug" / "localize" / str(building_id)
            dump_dir.mkdir(parents=True, exist_ok=True)
            ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d_%H%M%S_%f")
            for i, b in enumerate(image_bytes):
                (dump_dir / f"{ts}_{i:02d}.jpg").write_bytes(b)
            logger.info("localize query dump: %s/ %d images", dump_dir, len(image_bytes))
        except Exception as _dump_exc:
            logger.warning("localize query dump 실패: %s", _dump_exc)
        response = await LocalizationAdapter(session).localize(
            building_id=building_id,
            images=image_bytes,
        )
        logger.info(
            "localize response: building_id=%s map_id=%s confidence=%.4f "
            "pose=%s candidates=%d",
            building_id,
            response.map_id,
            response.confidence,
            response.pose,
            len(response.candidates),
        )
        return response
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
