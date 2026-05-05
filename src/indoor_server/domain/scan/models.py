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
    model_config = ConfigDict(frozen=True)

    id: int
    scan_id: str
    keyframe_seq: int
    created_at: int
    pose_matrix: bytes
    tx: float
    ty: float
    tz: float


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
