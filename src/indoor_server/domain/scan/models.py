"""scan_metadata.db v4 파싱 결과 VO. read-only immutable."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from indoor_server.domain.poi.enums import DetectionSource, POISource


class ScanSessionRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    started_at: int
    ended_at: int | None
    device_model: str
    app_version: str
    state: str
    keyframe_count: int
    notes: str | None


class KeyframeRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    scan_id: str
    seq: int
    captured_at: int
    image_path: str
    pose_matrix: bytes
    tx: float
    ty: float
    tz: float
    tracking_state: str
    rtabmap_node_id: int | None


class POIMarkRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    scan_id: str
    keyframe_seq: int
    created_at: int
    pose_matrix: bytes
    tx: float
    ty: float
    tz: float
    track_id: int | None
    label: str | None
    source: POISource


class POIPhotoRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    poi_mark_id: int
    scan_id: str
    keyframe_seq: int
    captured_at: int
    bbox_x: float | None
    bbox_y: float | None
    bbox_w: float | None
    bbox_h: float | None
    class_name: str  # 'manual' 리터럴 포함
    confidence: float
    image_blob: bytes | None = None


class BranchMarkRow(BaseModel):
    """Sidecar v8 부터 node_type/width_m/mark_session_id/connect_*/dx-dy-dz_local 추가.

    v6/v7 호환: 새 컬럼 모두 Optional (default None). v8 sidecar 만 채움.
    """

    model_config = ConfigDict(frozen=True)

    id: int
    scan_id: str
    keyframe_seq: int
    created_at: int
    pose_matrix: bytes
    tx: float
    ty: float
    tz: float
    # v8 추가
    node_type: str = "corridor"             # 'corridor' | 'corner'
    width_m: float | None = None
    connect_hint: str | None = None
    connect_node_id: str | None = None
    mark_session_id: str | None = None
    dx_local: float | None = None
    dy_local: float | None = None
    dz_local: float | None = None


class YOLODetectionRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    scan_id: str
    keyframe_seq: int
    class_name: str
    confidence: float
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    mask_rle: bytes | None
    source: DetectionSource
    track_id: int | None


class InterfloorMarkRow(BaseModel):
    """Sprint 65 v6: 층간 연결 노드 (계단/엘리베이터/에스컬레이터).

    iOS sidecar v6 `interfloor_mark` 테이블 row.
    prefix 는 사용자 입력 (예: "EV-A"). 서버는 이 값을
    Sprint 62 VerticalConnectorResolver 의 connector_key 매칭에 사용한다.
    """

    model_config = ConfigDict(frozen=True)

    id: int
    scan_id: str
    keyframe_seq: int
    created_at: int
    connector_type: str  # 'elevator' | 'escalator' | 'stairs'
    prefix: str
    pose_matrix: bytes
    tx: float
    ty: float
    tz: float
    # v8 추가
    dx_local: float | None = None
    dy_local: float | None = None
    dz_local: float | None = None


class BranchEdgeRow(BaseModel):
    """v9: 사용자가 직접 찍은 branch_mark 사이의 명시적 edge.

    kind: 'sequential' (corridor↔corridor 직접) | 'cornerPolygon' (corner cycle 의 한 변).
    polygon_closed: 1 = closed cycle, NULL = open.
    from/to_node_id: branch_mark.id 의 string 표현.
    """

    model_config = ConfigDict(frozen=True)

    id: int
    scan_id: str
    from_node_id: str
    to_node_id: str
    kind: str  # 'sequential' | 'cornerPolygon'
    length_m: float
    mark_session_id: str | None = None
    polygon_closed: int | None = None
    created_at: int | None = None


class SidecarContents(BaseModel):
    """scan_metadata.db v4/v5/v6 파싱 결과 전체. DB write 전 검증용 VO."""

    model_config = ConfigDict(frozen=True)

    scan_session: ScanSessionRow
    keyframes: list[KeyframeRow]
    poi_marks: list[POIMarkRow]
    poi_photos: list[POIPhotoRow]
    branch_marks: list[BranchMarkRow]
    yolo_detections: list[YOLODetectionRow]
    # Sprint 65 v6 추가. v4/v5 sidecar 에서는 빈 리스트.
    interfloor_marks: list[InterfloorMarkRow] = []
    # v9 추가 — 사용자가 명시한 branch_mark 사이의 edge. 이전 버전은 빈 리스트.
    branch_edges: list[BranchEdgeRow] = []


class ScanIngestResult(BaseModel):
    """IngestService 반환값."""

    model_config = ConfigDict(frozen=True)

    scan_id: str
    state: Literal["ingested", "replaced"]
    keyframe_count: int
    poi_marks_track_lock: int
    poi_marks_manual: int
    poi_photo_count: int
    branch_mark_count: int
    yolo_detection_count: int
    storage_path: str
    payload_sha256: str
    build_job_id: str | None = None  # W-1: auto_enqueue 결과


class ExistingScan(BaseModel):
    model_config = ConfigDict(frozen=True)

    scan_id: str
    payload_sha256: str
    ingested_at: datetime  # W-2 수정: 설계 §3.4 명세 반영
