"""Pydantic request/response 스키마."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, model_validator


class ScanUploadCounts(BaseModel):
    keyframes: int
    poi_marks_track_lock: int
    poi_marks_manual: int
    poi_photos: int
    branch_marks: int
    yolo_detections: int


class ScanUploadResponse(BaseModel):
    scan_id: str
    state: Literal["ingested", "replaced"]
    counts: ScanUploadCounts
    build_job_id: str | None = None
    storage_path: str
    payload_sha256: str


class LatestBuildInfo(BaseModel):
    build_job_id: str
    state: str
    finished_at: datetime | None = None


class ScanStatusResponse(BaseModel):
    scan_id: str
    ingested_at: str
    payload_sha256: str
    counts: ScanUploadCounts
    build_job_id: str | None
    build_state: str
    latest_build: LatestBuildInfo | None = None


class ApiError(BaseModel):
    code: str
    message: str
    detail: dict[str, object] | None = None


# ── Build API 스키마 ──────────────────────────────────────────────────────────

class BuildCounts(BaseModel):
    keyframes_processed: int = 0
    walkable_cells: int = 0
    skeleton_pixels: int = 0
    map_nodes: int = 0
    map_edges: int = 0
    pois_projected: int = 0
    walkable_coverage: float = 0.0
    connected_components: int = 0
    floor_z0: float | None = None  # W-6: floor plane z0 (5%ile of trajectory tz)
    footprint_geojson: dict[str, Any] | None = None  # Sprint 24: IMDF export용
    # Sprint 25: Manhattan rectification 메타
    rectified_footprint_geojson: dict[str, Any] | None = None
    rectification: dict[str, Any] | None = None
    # Sprint 37: RTAB-Map uploaded artifact readiness/source diagnostics
    build_source: str | None = None
    rtabmap: dict[str, Any] | None = None
    # Sprint 38: RTAB-Map Node/Link trajectory road polygon metadata
    rtabmap_trajectory: dict[str, Any] | None = None


class BuildEnqueueResponse(BaseModel):
    scan_id: str
    build_job_id: str
    state: Literal["pending"]
    enqueued_at: datetime


class BuildStatusResponse(BaseModel):
    scan_id: str
    build_job_id: str | None
    state: str  # BuildState value
    current_step: str | None = None
    progress: float | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    failure_reason: str | None = None
    counts: BuildCounts | None = None


class GraphCollectionProperties(BaseModel):
    scan_id: str
    build_job_id: str | None = None
    srid: int = 0
    floor_z0: float | None = None
    cell_size_m: float = 0.10


class GraphFeatureCollection(BaseModel):
    """GeoJSON FeatureCollection."""

    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[dict[str, Any]]
    properties: GraphCollectionProperties


# ── Route API 스키마 ─────────────────────────────────────────────────────────


class RouteEndpoint(BaseModel):
    """경로 시작/끝 지점 — coordinate XOR node_id XOR poi_mark_id 중 정확히 1개."""

    model_config = ConfigDict(extra="forbid")

    coordinate: tuple[float, float, float] | None = None
    node_id: UUID | None = None
    poi_mark_id: int | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> RouteEndpoint:
        """coordinate / node_id / poi_mark_id 중 정확히 1개만 허용."""
        provided = sum(
            v is not None for v in [self.coordinate, self.node_id, self.poi_mark_id]
        )
        if provided != 1:
            raise ValueError(
                "exactly one of coordinate, node_id, poi_mark_id must be provided "
                f"(got {provided})"
            )
        return self


class RouteRequest(BaseModel):
    """POST /route 요청 스키마."""

    scan_id: UUID
    scan_ids: list[UUID] | None = None
    merge_overlaps: bool = False
    start: RouteEndpoint
    goal: RouteEndpoint


class CoordinateRouteRequest(BaseModel):
    """사용자앱용 좌표 기반 경로 요청.

    VPS/localization이 이미 map local metric 좌표를 반환한다고 가정한다.
    """

    scan_id: UUID
    scan_ids: list[UUID] | None = None
    merge_overlaps: bool = False
    start_coordinate: tuple[float, float, float]
    goal_coordinate: tuple[float, float, float]


class SnapInfo(BaseModel):
    """좌표 snap 결과 정보. node_id/poi_mark_id 직접 지정 시 None."""

    start_snap_distance_m: float | None
    goal_snap_distance_m: float | None


class RouteNode(BaseModel):
    """경로 노드 하나."""

    node_id: UUID
    x: float
    y: float
    z: float
    node_type: str
    label: str | None
    poi_mark_id: int | None
    scan_id: UUID | None = None
    level_id: str | None = None


class RouteResponse(BaseModel):
    """POST /route 응답 스키마."""

    scan_id: UUID
    scan_ids: list[UUID] | None = None
    build_job_id: UUID
    build_job_ids: list[UUID] | None = None
    path_nodes: list[RouteNode]
    path_geometry: dict[str, Any]  # GeoJSON LineString Z (raw geometry)
    length_m: float
    node_count: int
    snap_info: SnapInfo
    route_metadata: dict[str, Any] | None = None


# ── IMDF export 스키마 ────────────────────────────────────────────────────────


class ImdfErrorCode(StrEnum):
    """IMDF export 에러 코드."""

    SCAN_NOT_FOUND = "SCAN_NOT_FOUND"
    GRAPH_NOT_READY = "GRAPH_NOT_READY"
    FOOTPRINT_NOT_AVAILABLE = "FOOTPRINT_NOT_AVAILABLE"


class ImdfManifest(BaseModel):
    """zip 내부 manifest.json 스키마 — 외부 검증/OpenAPI 명시용."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["indoor-pathfinding-imdf-lite"]
    version: str
    generated_at: str  # ISO8601
    scan_id: str
    build_job_id: str
    coordinate_system: Literal["local_metric", "epsg:4326"]
    language: str
    feature_files: list[str]
    z_floor_m: float
    extents_m: dict[str, float]
    rectification: dict[str, Any] | None = None
    unit_split: dict[str, Any] | None = None
