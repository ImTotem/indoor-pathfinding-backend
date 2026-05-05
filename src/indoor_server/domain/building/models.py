"""Build Phase 도메인 VO. numpy 의존은 허용 (geometry 연산 필수)."""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

import numpy as np
from pydantic import BaseModel, ConfigDict

from indoor_server.domain.building.enums import (
    BuildFailureReason,
    BuildState,
    BuildStep,
    EdgeType,
    NodeType,
)


class GridOrigin(BaseModel):
    """world frame → grid 좌표계 변환 원점."""

    model_config = ConfigDict(frozen=True)

    x0: float  # world meters (grid[0,0] 대응 코너)
    y0: float
    z0: float  # floor plane height (fixed, A-1)
    cell_size: float  # meters
    w: int  # grid width (x 방향)
    h: int  # grid height (y 방향)


class WalkableGrid(BaseModel):
    """2D walkable grid. mask/observation_count는 numpy — 직렬화 제외."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    origin: GridOrigin
    mask: np.ndarray          # (h, w) bool
    observation_count: np.ndarray  # (h, w) uint16


class MapNodeVO(BaseModel):
    """map_node 도메인 VO."""

    model_config = ConfigDict(frozen=True)

    node_id: UUID
    scan_id: UUID
    build_job_id: UUID
    node_type: NodeType
    x: float
    y: float
    z: float
    label: str | None = None
    poi_mark_id: int | None = None
    source_ref: dict[str, object] | None = None


class MapEdgeVO(BaseModel):
    """map_edge 도메인 VO."""

    model_config = ConfigDict(frozen=True)

    edge_id: UUID
    scan_id: UUID
    build_job_id: UUID
    from_node_id: UUID
    to_node_id: UUID
    edge_type: EdgeType
    polyline: list[tuple[float, float, float]]  # world frame
    length_m: float


class BuildCounts(BaseModel):
    """빌드 단계별 수량 통계."""

    model_config = ConfigDict(frozen=True)

    keyframes_processed: int = 0
    walkable_cells: int = 0
    skeleton_pixels: int = 0
    map_nodes: int = 0
    map_edges: int = 0
    pois_projected: int = 0
    walkable_coverage: float = 0.0
    connected_components: int = 0
    floor_z0: float | None = None  # W-6: 설계 §3.4 floor plane z (5%ile of trajectory tz)
    # Sprint 24: IMDF export용 footprint MultiPolygon GeoJSON
    footprint_geojson: dict[str, object] | None = None
    # Sprint 25: CAD-style post-processed footprint + metadata
    rectified_footprint_geojson: dict[str, object] | None = None
    rectification: dict[str, object] | None = None
    unit_split: dict[str, object] | None = None
    # Sprint 34 Cycle 8: Forced Rectilinear Graph 메타
    graph_dominant_angle_deg: float | None = None
    graph_forced_rectilinear: bool = False
    graph_original_node_count: int | None = None
    graph_rectified_node_count: int | None = None
    # Sprint 37: RTAB-Map uploaded artifact readiness/source diagnostics
    build_source: str | None = None
    rtabmap: dict[str, object] | None = None
    # Sprint 38: RTAB-Map Node/Link trajectory road polygon metadata
    rtabmap_trajectory: dict[str, object] | None = None
    # Sprint 46: Floor segmentation point cloud + raster metadata (source-of-truth)
    floor_pointcloud: dict[str, object] | None = None
    floor_raster: dict[str, object] | None = None
    # Sprint 47: post-rectification CAD cleanup (floor_pointcloud 모드 전용)
    polygon_cad_cleanup: dict[str, object] | None = None
    # Sprint 48: dominant_angle hint chain candidates + acceptance flags
    dominant_angle_hint: dict[str, object] | None = None
    cad_effect_pass: bool | None = None
    cad_effect_small_polygon_pass: bool | None = None
    # Sprint 49 (Codex BLOCKER 1/2/6): ARKit↔RTABMap rigid SE(3) transform
    # metadata + per-candidate hint chain dump.
    poi_frame_transform: dict[str, object] | None = None
    # Sprint 49 (Codex BLOCKER 5): R-1 evidence — keyframes 미포함 + image source.
    scan_manifest: dict[str, object] | None = None
    # Sprint 50 (Codex BLOCKER 1~5): rectangle dictionary cover metadata.
    # accepted=True 면 hint chain skip, footprint_geojson 자체가 rectangle union.
    # accepted=False 면 fallback metadata 만 채워지고 hint chain 결과 사용.
    rectangle_cover: dict[str, object] | None = None
    # Sprint 51: wall-fitting polygon (default OFF, observer only). production
    # polygon 자체는 변경하지 않고 obstacle heatmap → 7-step pipeline 결과만
    # metadata 로 노출. pipeline `use_wall_polygon=False` 면 None.
    wall_polygon: dict[str, object] | None = None
    # Sprint 61: 2D display-routing dense tile graph + POI/facility attachment.
    display_navigation_grid: dict[str, object] | None = None


class BuildJob(BaseModel):
    """build_job 테이블 VO."""

    model_config = ConfigDict(frozen=True)

    build_job_id: UUID
    scan_id: UUID
    state: BuildState
    current_step: BuildStep | None = None
    progress: float | None = None
    enqueued_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    locked_at: datetime | None = None
    worker_id: str | None = None
    attempt_count: int = 0
    max_attempts: int = 3
    failure_reason: BuildFailureReason | None = None
    failure_detail: str | None = None
    counts: BuildCounts | None = None


class BuildOutcome(BaseModel):
    """파이프라인 실행 결과."""

    model_config = ConfigDict(frozen=True)

    nodes: list[MapNodeVO]
    edges: list[MapEdgeVO]
    poi_world_poses: dict[int, tuple[float, float, float]]  # poi_mark_id → (x,y,z)
    counts: BuildCounts
    passed_quality_gate: bool
    failure_reason: BuildFailureReason | None = None


class FloorPolygon(BaseModel):
    """per-frame 단순화된 floor polygon (image pixel coords)."""

    model_config = ConfigDict(frozen=True)

    seq: int                              # keyframe seq
    polygon_index: int                    # 한 frame 안에서 polygon 순번 (0..)
    vertices_px: list[tuple[int, int]]    # contour vertex (x, y) image px
    image_w: int
    image_h: int
    area_px: float                        # 디버그용
    pose_matrix: bytes                    # back-project 시 필요 (KeyframeMasks에서 복사)
    tx: float
    ty: float
    tz: float


class FloorLayout(BaseModel):
    """multi-frame union된 world floor layout."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    # shapely MultiPolygon — Pydantic이 모르므로 arbitrary_types_allowed
    multipolygon: object                  # shapely.geometry.MultiPolygon
    z0: float                             # floor plane height (world)
    sub_polygon_count: int                # MultiPolygon.geoms 수
    total_area_m2: float                  # 합 면적


class DepthMap(BaseModel):
    """per-keyframe depth NN 출력 VO (시각화/디버그 전달용)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    seq: int                        # keyframe seq
    depth_relative: np.ndarray      # (H, W) float32 relative inverse depth
    scale: float | None             # metric scale factor (None = calibration 실패)
    valid_pixel_count: int          # z-tolerance 통과 floor 픽셀 수


class DepthAwareLayout(BaseModel):
    """Depth-aware back-projection 결과로 만든 walkable layout."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    multipolygon: object            # shapely.geometry.MultiPolygon
    z0: float
    sub_polygon_count: int
    total_area_m2: float
