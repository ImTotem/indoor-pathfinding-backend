"""Build Phase 도메인 열거형."""
from __future__ import annotations

from enum import StrEnum


class BuildStep(StrEnum):
    INIT = "init"
    FLOOR_SEG = "floor_seg"
    BACK_PROJECT = "back_project"
    WALKABLE_GRID = "walkable_grid"
    SKELETON = "skeleton"
    NODE_PLACEMENT = "node_placement"
    POI_PROJECTION = "poi_projection"
    QUALITY_GATE = "quality_gate"
    PERSIST = "persist"
    DONE = "done"


class BuildState(StrEnum):
    NOT_STARTED = "not_started"
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeType(StrEnum):
    JUNCTION = "junction"       # degree >= 3
    ENDPOINT = "endpoint"       # degree = 1
    CORRIDOR = "corridor"       # degree = 2, 등간격 샘플
    POI = "poi"                 # 의미 객체
    POI_ATTACH = "poi_attach"   # POI <-> skeleton 연결 virtual 노드
    PASSAGE_STAIRS = "passage_stairs"
    PASSAGE_ELEVATOR = "passage_elevator"
    PASSAGE_ESCALATOR = "passage_escalator"


class EdgeType(StrEnum):
    SKELETON = "skeleton"    # medial axis 실제 경로
    POI_SPUR = "poi_spur"    # POI <-> POI_ATTACH


class BuildFailureReason(StrEnum):
    WALKABLE_COVERAGE_LOW = "walkable_coverage_low"
    GRAPH_DISCONNECTED = "graph_disconnected"
    MODEL_LOAD_FAILED = "model_load_failed"
    INTRINSICS_MISSING = "intrinsics_missing"
    RTABMAP_DATA_NOT_READY = "rtabmap_data_not_ready"
    INTERNAL = "internal"
