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


class MergeScansRequest(V1Model):
    chunk_ids: list[UUID] = Field(default_factory=list, alias="chunkIds")


class MergedScanResponse(V1Model):
    floor_id: UUID = Field(alias="floorId")
    active_scan_id: UUID | None = Field(None, alias="activeScanId")
    status: str


class ProcessingStatusResponse(V1Model):
    floor_id: UUID = Field(alias="floorId")
    scan_id: UUID | None = Field(None, alias="scanId")
    build_job_id: UUID | None = Field(None, alias="buildJobId")
    status: str
    progress: float | None = None
    error: str | None = None


class PathfindingRequest(V1Model):
    start_floor_level: int = Field(alias="startFloorLevel")
    start_x: float = Field(alias="startX")
    start_y: float = Field(alias="startY")
    start_z: float = Field(0.0, alias="startZ")
    destination_name: str = Field(alias="destinationName")
    preference: Literal["SHORTEST", "ELEVATOR_FIRST", "STAIRCASE_FIRST"] = "SHORTEST"


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
