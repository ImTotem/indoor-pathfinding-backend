"""V1 compatibility request/response schemas.

V1 uses camelCase JSON because the original Spring API did. The main server API
keeps snake_case; these models are scoped to `/api/v1/*`.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class V1Model(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class V1ErrorResponse(V1Model):
    code: str
    message: str
    detail: dict[str, Any] | None = None


class BuildingCreateRequest(V1Model):
    name: str
    description: str | None = None
    latitude: float | None = None
    longitude: float | None = None


class BuildingUpdateRequest(V1Model):
    name: str | None = None
    description: str | None = None
    latitude: float | None = None
    longitude: float | None = None


class BuildingStatusRequest(V1Model):
    status: Literal["DRAFT", "ACTIVE"]


class FloorCreateRequest(V1Model):
    name: str
    level: int
    height: float | None = None


class FloorUpdateRequest(V1Model):
    name: str | None = None
    height: float | None = None


class FloorResponse(V1Model):
    floor_id: UUID = Field(alias="floorId")
    building_id: UUID = Field(alias="buildingId")
    name: str
    level: int
    height: float | None = None
    has_path: bool = Field(False, alias="hasPath")
    has_ply: bool = Field(False, alias="hasPly")
    active_scan_id: UUID | None = Field(None, alias="activeScanId")
    created_at: datetime | None = Field(None, alias="createdAt")
    updated_at: datetime | None = Field(None, alias="updatedAt")


class BuildingResponse(V1Model):
    building_id: UUID = Field(alias="buildingId")
    name: str
    description: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    status: Literal["DRAFT", "ACTIVE"]
    created_at: datetime | None = Field(None, alias="createdAt")
    updated_at: datetime | None = Field(None, alias="updatedAt")


class BuildingDetailResponse(BuildingResponse):
    floors: list[FloorResponse] = Field(default_factory=list)
    vertical_passages: list[VerticalPassageResponse] = Field(
        default_factory=list,
        alias="verticalPassages",
    )


class FloorPathResponse(V1Model):
    floor_id: UUID = Field(alias="floorId")
    scan_id: UUID | None = Field(None, alias="scanId")
    build_job_id: UUID | None = Field(None, alias="buildJobId")
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    bounds: dict[str, float] | None = None


class ScanChunkResponse(V1Model):
    chunk_id: UUID = Field(alias="chunkId")
    floor_id: UUID = Field(alias="floorId")
    scan_id: UUID = Field(alias="scanId")
    file_name: str | None = Field(None, alias="fileName")
    file_size: int | None = Field(None, alias="fileSize")
    status: str
    active: bool
    upload_order: int = Field(alias="uploadOrder")
    created_at: datetime | None = Field(None, alias="createdAt")
    included_in_merged_scan: UUID | None = Field(
        None,
        alias="includedInMergedScan",
        description=(
            "If this source scan is part of the floor's currently active "
            "merged scan, the merged scan_id. NULL otherwise."
        ),
    )


class MergeScansRequest(V1Model):
    chunk_ids: list[UUID] = Field(
        default_factory=list,
        alias="chunkIds",
        description=(
            "Specific floor_scan UUIDs (or scan UUIDs) to include. Empty list = "
            "merge every READY/UPLOADED scan currently on the floor."
        ),
    )


class MergedScanResponse(V1Model):
    floor_id: UUID = Field(alias="floorId")
    active_scan_id: UUID | None = Field(None, alias="activeScanId")
    status: str = Field(
        description="READY / PROCESSING / COMPLETED / FAILED / NOT_STARTED"
    )


class ProcessingStatusResponse(V1Model):
    floor_id: UUID = Field(alias="floorId")
    scan_id: UUID | None = Field(None, alias="scanId")
    build_job_id: UUID | None = Field(None, alias="buildJobId")
    status: str
    progress: float | None = None
    error: str | None = None


# ── Streaming scan ingest (start → frames → finalize → build) ────────────────


class ScanStartRequest(V1Model):
    """Optional scan_id (client-generated UUID) + device info JSON string."""

    scan_id: str | None = Field(
        None,
        alias="scanId",
        description="UUID for this scan. Server generates one if omitted.",
    )
    device_info: str | None = Field(
        None,
        alias="deviceInfo",
        description="Optional JSON string with device metadata "
        "({model, app_version, ...}).",
    )


class ScanStartResponse(V1Model):
    scan_id: UUID = Field(alias="scanId")
    floor_id: UUID = Field(alias="floorId")
    storage_path: str = Field(alias="storagePath")
    state: Literal["OPEN"] = "OPEN"


class FrameLinkPayload(V1Model):
    """Edge between two Nodes within the same scan.

    `transform` is the 48-byte (3x4 float32 row-major) relative pose blob.
    `informationMatrix` is the 288-byte 6x6 float64 covariance blob.
    Both are base64-encoded in JSON.
    """

    from_id: int = Field(alias="fromId")
    to_id: int = Field(alias="toId")
    type: int = Field(
        0,
        description="RTAB-Map link type. 0=NEIGHBOR, 1=LOOP_CLOSURE, etc.",
    )
    transform_b64: str = Field(
        alias="transform",
        description="base64 of 48-byte 3x4 float32 row-major transform.",
    )
    information_matrix_b64: str | None = Field(
        None,
        alias="informationMatrix",
        description="base64 of 288-byte 6x6 float64. Identity if omitted.",
    )
    user_data_b64: str | None = Field(None, alias="userData")


class FramePayload(V1Model):
    """One RTAB-Map Node + Data tuple to append.

    Blobs are base64-encoded in JSON. iOS clients should send the exact
    rtabmap-formatted blobs they would have written to the local rtabmap.db:
    image as JPEG, depth as RVL, pose as 48-byte 3x4 float32, calibration as
    164-byte intrinsics+local_transform blob.
    """

    node_id: int = Field(alias="nodeId", ge=1)
    map_id: int = Field(0, alias="mapId")
    weight: int = Field(0, alias="weight")
    stamp: float = Field(
        description="Capture timestamp (seconds since RTAB-Map epoch).",
    )
    pose_b64: str = Field(
        alias="pose",
        description="base64 of 48-byte 3x4 float32 row-major pose.",
    )
    image_b64: str = Field(
        alias="image",
        description="base64 of JPEG-encoded RGB keyframe.",
    )
    calibration_b64: str = Field(
        alias="calibration",
        description="base64 of 164-byte calibration blob.",
    )
    depth_b64: str | None = Field(
        None,
        alias="depth",
        description="base64 of RVL-compressed depth map (optional).",
    )
    depth_confidence_b64: str | None = Field(
        None, alias="depthConfidence",
    )
    ground_truth_pose_b64: str | None = Field(
        None, alias="groundTruthPose",
    )
    velocity_b64: str | None = Field(None, alias="velocity")
    gps_b64: str | None = Field(None, alias="gps")
    env_sensors_b64: str | None = Field(None, alias="envSensors")
    label: str | None = None
    user_data_b64: str | None = Field(None, alias="userData")
    scan_b64: str | None = Field(None, alias="scan")
    scan_info_b64: str | None = Field(None, alias="scanInfo")


class ScanFramesRequest(V1Model):
    """Batch of frames + links to append.

    K is unbounded — server consumes whatever the client sends. Frames whose
    `nodeId` has already been ingested are skipped (idempotent retry).
    """

    frames: list[FramePayload] = Field(default_factory=list)
    links: list[FrameLinkPayload] = Field(default_factory=list)


class ScanFramesResponse(V1Model):
    scan_id: UUID = Field(alias="scanId")
    frames_applied: int = Field(alias="framesApplied")
    frames_skipped: int = Field(alias="framesSkipped")
    links_applied: int = Field(alias="linksApplied")
    links_skipped: int = Field(alias="linksSkipped")
    last_node_id: int = Field(alias="lastNodeId")
    node_count: int = Field(alias="nodeCount")


class ScanFinalizeResponse(V1Model):
    scan_id: UUID = Field(alias="scanId")
    floor_id: UUID = Field(alias="floorId")
    state: Literal["READY"] = "READY"
    node_count: int = Field(alias="nodeCount")
    keyframe_count: int = Field(alias="keyframeCount")
    poi_mark_count: int = Field(alias="poiMarkCount")
    payload_sha256: str = Field(alias="payloadSha256")


class PathfindingRequest(V1Model):
    """길찾기 요청. 사용자의 현재 위치(world meter) + 도착 POI 이름.

    시작 floor 는 좌표 z 기반으로 서버가 자동 결정한다. 명시 override 가 필요할
    때만 `startFloorLevel` 을 보낸다.
    """

    start_floor_level: int | None = Field(
        None,
        alias="startFloorLevel",
        description=(
            "Optional override: 시작 floor 의 층수를 강제로 지정. 보통 보낼 필요 없다. "
            "생략하면 `startZ` 좌표로 서버가 가장 가까운 floor 의 active scan 을 자동 선택."
        ),
    )
    start_x: float = Field(
        alias="startX",
        description="사용자 현재 위치 X (server world frame, meter). 측위 응답의 pose.tx 그대로.",
    )
    start_y: float = Field(
        alias="startY",
        description="사용자 현재 위치 Y (server world frame, meter). 측위 응답의 pose.ty 그대로.",
    )
    start_z: float = Field(
        0.0,
        alias="startZ",
        description="사용자 현재 위치 Z (높이, meter). 보통 측위 응답의 pose.tz 그대로.",
    )
    destination_name: str = Field(
        alias="destinationName",
        description=(
            "도착 POI 의 이름 (예: '301호', 'ELEVATOR EV-A'). "
            "`/buildings/{id}/pois` 의 `name` 또는 `label` 과 정확히 일치해야 한다."
        ),
    )
    preference: Literal["SHORTEST", "ELEVATOR_FIRST", "STAIRCASE_FIRST"] = Field(
        "SHORTEST",
        description="Legacy 필드. 현재 무시되며 metadata 에 echo. 신규 클라는 `verticalPreference` 사용.",
    )
    vertical_preference: Literal["ELEVATOR", "STAIRS"] = Field(
        "ELEVATOR",
        alias="verticalPreference",
        description=(
            "층간 이동 수단 선택. **기본값 ELEVATOR**.\n"
            "- `ELEVATOR`: 엘리베이터만 cross-floor edge 로 사용.\n"
            "- `STAIRS`: 계단/에스컬레이터만 사용.\n"
            "선택한 수단이 빌딩에 없으면 `PATH_NOT_FOUND` 가 반환되므로, 클라는 "
            "fallback 으로 다른 값으로 재요청 권장."
        ),
    )


class CoordinatePoint(V1Model):
    x: float
    y: float
    z: float = 0.0


class FloorCoordinateRouteRequest(V1Model):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    start: CoordinatePoint
    goal: CoordinatePoint


class V1RouteSnapInfo(V1Model):
    start_snap_distance_m: float | None = Field(None, alias="startSnapDistanceM")
    goal_snap_distance_m: float | None = Field(None, alias="goalSnapDistanceM")


class FloorCoordinateRouteResponse(V1Model):
    building_id: UUID = Field(alias="buildingId")
    floor_id: UUID = Field(alias="floorId")
    scan_id: UUID = Field(alias="scanId")
    path_geometry: dict[str, Any] = Field(alias="pathGeometry")
    length_m: float = Field(alias="lengthM")
    node_count: int = Field(alias="nodeCount")
    snap_info: V1RouteSnapInfo = Field(alias="snapInfo")
    route_metadata: dict[str, Any] = Field(default_factory=dict, alias="routeMetadata")


class V1Position(V1Model):
    x: float
    y: float
    z: float
    floor_level: int | None = Field(None, alias="floorLevel")


class PathStepResponse(V1Model):
    step_number: int = Field(alias="stepNumber")
    floor_level: int | None = Field(None, alias="floorLevel")
    position: V1Position
    instruction: str
    node_id: UUID | None = Field(None, alias="nodeId")


class FloorTransitionResponse(V1Model):
    from_floor_level: int | None = Field(None, alias="fromFloorLevel")
    to_floor_level: int | None = Field(None, alias="toFloorLevel")
    connector_type: str | None = Field(None, alias="connectorType")
    connector_key: str | None = Field(None, alias="connectorKey")


class PathfindingResponse(V1Model):
    building_id: UUID = Field(alias="buildingId")
    total_distance: float = Field(alias="totalDistance")
    estimated_time_seconds: int = Field(alias="estimatedTimeSeconds")
    steps: list[PathStepResponse]
    floor_transitions: list[FloorTransitionResponse] = Field(
        default_factory=list,
        alias="floorTransitions",
    )
    route_metadata: dict[str, Any] = Field(default_factory=dict, alias="routeMetadata")


class FloorMapCoordinateSystem(V1Model):
    """좌표계 메타. 측위/route/지도 모두 같은 frame 이라 변환 0회."""

    frame: str = Field(
        "world_xy_meters",
        description="frame 이름. 'world_xy_meters' 고정.",
    )
    description: str = Field(
        "Server world frame. (x, y) is the floor plane in meters; z is height. "
        "Same frame as map_node, route polyline, and localize pose.",
        description="frame 설명 (사람용).",
    )


class FloorMapBounds(V1Model):
    """floor 의 axis-aligned bounding box. 클라 viewport scale 계산용."""

    min_x: float = Field(alias="minX", description="minX (meter)")
    min_y: float = Field(alias="minY", description="minY (meter)")
    max_x: float = Field(alias="maxX", description="maxX (meter)")
    max_y: float = Field(alias="maxY", description="maxY (meter)")
    width_m: float = Field(alias="widthM", description="maxX - minX (meter)")
    height_m: float = Field(alias="heightM", description="maxY - minY (meter)")


class FloorMapConnector(V1Model):
    """passage 노드의 층간 이동 수단 정보. passage 가 아닌 노드는 null."""

    type: str = Field(
        description="`elevator` | `stair` | `stairs` | `escalator`. 아이콘 매핑에 사용."
    )
    key: str | None = Field(
        None,
        description=(
            "같은 connector 의 cross-floor 식별자 (예: `ev_a`). "
            "다른 floor 의 같은 key 는 같은 엘리베이터/계단."
        ),
    )


class FloorMapNode(V1Model):
    """그래프 노드. 측위 좌표계와 동일 (world meter)."""

    id: UUID = Field(description="map_node UUID. routing 요청의 nodeId 로 그대로 사용 가능.")
    type: str = Field(
        description=(
            "노드 종류. "
            "`corridor` (사용자 명시 backbone), "
            "`junction` (route-only 분기 노드, width 없음), "
            "`poi` (POI 자체 노드, 라벨 표시), "
            "`poi_attach` (POI/connector 가 corridor 에 붙는 foot, 화면에서 보통 hidden), "
            "`passage_stairs` / `passage_elevator` / `passage_escalator` "
            "(interfloor 연결 노드, connector 필드도 함께 채움). "
            "`endpoint` 는 legacy."
        )
    )
    x: float = Field(description="X 좌표 (world meter).")
    y: float = Field(description="Y 좌표 (world meter).")
    z: float = Field(description="Z 좌표 (height, meter).")
    label: str | None = Field(
        None,
        description="표시용 라벨 (POI 이름, connector 이름 등). 없으면 null.",
    )
    connector: FloorMapConnector | None = Field(
        None,
        description="passage 노드일 때만 채움. 그 외 null.",
    )


class FloorMapEdge(V1Model):
    """그래프 엣지. polyline 그릴 때 from→to 순서 따라 연결."""

    id: UUID = Field(description="map_edge UUID.")
    from_id: UUID = Field(alias="fromId", description="시작 노드 UUID.")
    to_id: UUID = Field(alias="toId", description="끝 노드 UUID.")
    length_m: float = Field(alias="lengthM", description="엣지 길이 (meter).")
    type: str = Field(
        description=(
            "`corridor` (backbone 사이) | `spur` (POI/connector 가 corridor 로 붙는 짧은 엣지). "
            "스타일 분리 시 사용 — corridor 는 굵게, spur 는 얇거나 hidden 권장."
        )
    )


class FloorMapResponse(V1Model):
    """floor 2D 지도 응답. polygon (GeoJSON) + 그래프 + bounds. 모두 world meter.

    클라는 `bounds` 로 viewport scale 만 계산하면 polygon, nodes, edges, 측위 좌표,
    route polyline 까지 같은 좌표계 위에 겹쳐 그릴 수 있다.
    """

    floor_id: UUID = Field(alias="floorId", description="floor UUID.")
    building_id: UUID = Field(alias="buildingId", description="속한 building UUID.")
    scan_id: UUID = Field(
        alias="scanId",
        description="이 floor 의 active scan UUID. 디버깅/캐시 키 용도.",
    )
    floor_level: int = Field(alias="floorLevel", description="층수 (지하면 음수).")
    floor_name: str | None = Field(
        None, alias="floorName", description="floor 표시명 (예: '1F')."
    )
    build_job_id: UUID = Field(
        alias="buildJobId",
        description="이 응답을 만든 build job UUID. ETag 와 동일 값.",
    )
    coordinate_system: FloorMapCoordinateSystem = Field(
        default_factory=FloorMapCoordinateSystem, alias="coordinateSystem"
    )
    bounds: FloorMapBounds = Field(description="polygon + nodes 합산한 axis-aligned bbox.")
    polygon: dict[str, Any] = Field(
        description=(
            "GeoJSON FeatureCollection. 각 Feature 의 `properties.kind`:\n"
            "- `room`: 방 (corner cycle 으로 닫힌 polygon)\n"
            "- `corridor`: 복도 (corridor backbone 의 width buffer)\n"
            "- `floor_union`: 위 둘의 union (전체 footprint).\n"
            "클라는 보통 `floor_union` 만 채워서 그리고, room/corridor 는 디버그용."
        )
    )
    nodes: list[FloorMapNode] = Field(description="floor 의 모든 그래프 노드.")
    edges: list[FloorMapEdge] = Field(description="floor 의 모든 그래프 엣지.")
    etag: str = Field(
        description=(
            "응답의 ETag (build_job_id). 다음 요청 시 헤더 `If-None-Match: \"<etag>\"` "
            "보내면 변경 없을 때 304 반환."
        ),
    )


class FeatureLookupQuery(V1Model):
    """단일 조회 좌표. 클라가 어느 floor 의 어느 위치 근처 keyframe 을 받고 싶은지 지정."""

    floor_level: int = Field(
        alias="floorLevel",
        description="조회 대상 floor 의 층수. 같은 빌딩의 active scan 이 있어야 한다.",
    )
    x: float = Field(description="질의 좌표 X (world meter, 측위/그래프와 동일 frame).")
    y: float = Field(description="질의 좌표 Y (world meter).")
    z: float = Field(0.0, description="질의 좌표 Z (height, meter). 보통 측위 결과 그대로.")
    view_direction: list[float] | None = Field(
        None,
        alias="viewDirection",
        description=(
            "선택: 카메라 forward 방향 벡터 [dx, dy, dz] (world frame). "
            "정규화돼 있지 않아도 됨. `viewConeDeg` 와 함께 보내면, "
            "방향이 cone 안에 있는 keyframe 만 후보가 된다 (정밀 매칭용). "
            "생략 시 모든 방향 허용."
        ),
    )


class FeatureLookupOptions(V1Model):
    radius_m: float = Field(
        2.5,
        alias="radiusM",
        description=(
            "각 query 좌표 ± 반경 (meter). 후보 keyframe 은 query 와의 3D 거리가 "
            "이 값 이하인 것만. 너무 크면 응답 사이즈 폭증."
        ),
    )
    max_keyframes_per_query: int = Field(
        5,
        alias="maxKeyframesPerQuery",
        description="query 1개당 가장 가까운 keyframe N개만 선택. 서버 한계 16.",
    )
    view_cone_deg: float | None = Field(
        None,
        alias="viewConeDeg",
        description=(
            "선택: viewDirection 과 keyframe 의 forward 가 이루는 각 ≤ value/2 인 "
            "keyframe 만 허용 (단위: 도). `viewDirection` 이 없는 query 에는 미적용."
        ),
    )
    format: Literal["json_b64"] = Field(
        "json_b64",
        description=(
            "응답 포맷. 현재는 `json_b64` (base64 inline) 만 지원. "
            "추후 `msgpack`/`npz` 추가 예정."
        ),
    )


class FeatureLookupRequest(V1Model):
    """좌표 배열 → keyframe SuperPoint feature pack 조회.

    클라가 경로 전체 좌표를 보내면 'route bundle' 효과를, 현재 위치 1개만 보내면
    실시간 보정용 minimal 응답을 받는다.
    """

    queries: list[FeatureLookupQuery] = Field(
        description="조회 좌표 배열. 1~64개. 같은 빌딩 안에서 floor 를 섞어도 됨."
    )
    options: FeatureLookupOptions = Field(default_factory=FeatureLookupOptions)


class FeatureLookupIntrinsics(V1Model):
    fx: float = Field(description="focal length x (pixel).")
    fy: float = Field(description="focal length y (pixel).")
    cx: float = Field(description="principal point x (pixel).")
    cy: float = Field(description="principal point y (pixel).")
    width: int = Field(description="이미지 너비 (pixel).")
    height: int = Field(description="이미지 높이 (pixel).")


class FeatureLookupKeyframe(V1Model):
    """단일 keyframe 의 SuperPoint feature pack. 클라 LightGlue 매칭에 그대로 사용."""

    kf_id: str = Field(
        alias="kfId",
        description="keyframe 고유 ID. `<scanId>:<rtabmapNodeId>` 형식.",
    )
    scan_id: UUID = Field(alias="scanId", description="소속 scan UUID.")
    floor_level: int = Field(alias="floorLevel", description="소속 floor 의 층수.")
    rtabmap_node_id: int = Field(
        alias="rtabmapNodeId",
        description="RTABMap Node 테이블의 id. global_descriptor 와 같은 keyframe 식별.",
    )
    pose: list[list[float]] = Field(
        description=(
            "4x4 row-major SE(3) 행렬 (world ← camera). "
            "마지막 열 [tx, ty, tz] 는 카메라 위치 (world meter). "
            "rotation 의 3번째 열 [r13, r23, r33] 이 카메라 forward (world frame)."
        )
    )
    intrinsics: FeatureLookupIntrinsics = Field(description="카메라 내부 파라미터.")
    matched_query_indices: list[int] = Field(
        alias="matchedQueryIndices",
        description=(
            "이 keyframe 이 매칭된 request.queries 의 인덱스 배열. "
            "여러 query 가 같은 keyframe 의 시야에 들면 dedup 되어 한 번만 응답."
        ),
    )
    distances_m: list[float] = Field(
        alias="distancesM",
        description="`matchedQueryIndices` 와 같은 순서. query → keyframe 3D 거리 (meter).",
    )
    keypoint_count: int = Field(
        alias="keypointCount",
        description="keypoints/descriptors/world3d 행 개수 N. 보통 1024 (모델 max).",
    )
    keypoints: str = Field(
        description=(
            "(N, 2) float32 의 base64. 이미지 픽셀 좌표 [u, v]. "
            "디코드: `Data(base64Encoded:).withUnsafeBytes`."
        )
    )
    descriptors: str = Field(
        description=(
            "(N, 256) **float16** 의 base64. SuperPoint descriptor. "
            "LightGlue 매칭 시 dtype 주의 (디바이스에서 float32 로 cast 필요)."
        )
    )
    world3d: str = Field(
        description=(
            "(N, 3) float32 의 base64. 각 keypoint 의 world 좌표 (meter). "
            "**일부 행은 NaN** (3D 추정 실패) — PnP 전 NaN 필터링 필요."
        )
    )
    global_descriptor: str | None = Field(
        None,
        alias="globalDescriptor",
        description=(
            "(384,) float16 의 base64 (DINOv2). retrieval 단계용. "
            "indexing 실패 시 null."
        ),
    )


class FeatureLookupModel(V1Model):
    """응답에 포함된 feature 의 모델 정보. 클라 측 디코드 dtype 결정에 사용."""

    extractor: str = Field("superpoint_v1", description="local feature extractor 이름.")
    matcher: str = Field("superpoint_lightglue", description="권장 matcher.")
    descriptor_dim: int = Field(256, alias="descriptorDim", description="descriptor 차원.")
    max_keypoints: int = Field(1024, alias="maxKeypoints", description="keyframe 당 최대 N.")
    descriptor_dtype: str = Field(
        "float16", alias="descriptorDtype", description="`descriptors` 의 dtype."
    )
    global_descriptor_dim: int = Field(
        384, alias="globalDescriptorDim", description="global_descriptor 차원."
    )
    global_descriptor_extractor: str = Field(
        "dinov2",
        alias="globalDescriptorExtractor",
        description="global descriptor 모델 이름.",
    )


class FeatureLookupStats(V1Model):
    query_count: int = Field(alias="queryCount", description="요청 query 개수.")
    keyframe_count: int = Field(
        alias="keyframeCount", description="dedup 후 응답 keyframe 개수."
    )
    total_keypoints: int = Field(
        alias="totalKeypoints", description="모든 keyframe keypoint 합."
    )
    byte_size: int = Field(
        alias="byteSize",
        description="base64 페이로드 합산 (디버그용). 실제 응답은 더 큼.",
    )


class FeatureLookupResponse(V1Model):
    building_id: UUID = Field(alias="buildingId")
    keyframes: list[FeatureLookupKeyframe] = Field(
        description="dedup 된 keyframe 배열. floor 가 섞여 있을 수 있음 (`floorLevel` 로 구분)."
    )
    model: FeatureLookupModel = Field(default_factory=FeatureLookupModel)
    stats: FeatureLookupStats


class LocalizeResponse(V1Model):
    building_id: UUID = Field(alias="buildingId")
    map_id: str | None = Field(None, alias="mapId")
    pose: dict[str, Any]
    confidence: float
    candidates: list[dict[str, Any]] = Field(default_factory=list)


class SlamStatusResponse(V1Model):
    building_id: UUID = Field(alias="buildingId")
    active_scan_count: int = Field(alias="activeScanCount")
    latest_status: str = Field(alias="latestStatus")
    scans: list[dict[str, Any]]


class SlamMetadataResponse(V1Model):
    building_id: UUID = Field(alias="buildingId")
    active_scan_id: UUID | None = Field(None, alias="activeScanId")
    keyframe_count: int = Field(0, alias="keyframeCount")
    created_at: datetime | None = Field(None, alias="createdAt")


class NodeImagesRequest(V1Model):
    x: float | None = None
    y: float | None = None
    z: float | None = None
    floor_level: int | None = Field(None, alias="floorLevel")


class NodeImagesResponse(V1Model):
    building_id: UUID = Field(alias="buildingId")
    images: list[dict[str, Any]]


class POICreateRequest(V1Model):
    name: str
    category: str = "destination"
    floor_id: UUID | None = Field(None, alias="floorId")
    floor_level: int | None = Field(None, alias="floorLevel")
    x: float | None = None
    y: float | None = None
    z: float | None = None


class POIResponse(V1Model):
    poi_id: UUID = Field(alias="poiId")
    building_id: UUID | None = Field(None, alias="buildingId")
    floor_id: UUID | None = Field(None, alias="floorId")
    name: str | None = None
    label: str | None = None
    category: str
    route_node_id: UUID | None = Field(None, alias="routeNodeId")
    display_point: dict[str, float] | None = Field(None, alias="displayPoint")
    needs_review: bool = Field(False, alias="needsReview")
    llm_confidence: float | None = Field(None, alias="llmConfidence")


class PassageSegment(V1Model):
    """층간 연결 passage 의 한 층 좌표 정보."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    stop_id: str | None = Field(None, alias="stopId")
    level_id: str | None = Field(None, alias="levelId")
    route_node_id: str | None = Field(None, alias="routeNodeId")
    # 좌표 (interfloor_mark 에서 가져옴, 없으면 None)
    x: float | None = None
    y: float | None = None
    # floor_id: UUID (stops 가 floor 단위로 존재하는 경우)
    floor_id: str | None = Field(None, alias="floorId")
    # connector kind (STAIRCASE / ELEVATOR / ESCALATOR 등)
    kind: str | None = None


class VerticalPassageResponse(V1Model):
    passage_id: UUID = Field(alias="passageId")
    building_id: UUID | None = Field(None, alias="buildingId")
    connector_type: str = Field(alias="connectorType")
    connector_key: str = Field(alias="connectorKey")
    name: str | None = None
    mock: bool = False  # Sprint 78: is_mock=true인 seed 데이터 식별
    segments: list[PassageSegment] = Field(default_factory=list)


BuildingDetailResponse.model_rebuild()
