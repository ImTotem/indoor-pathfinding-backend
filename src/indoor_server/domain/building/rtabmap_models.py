"""RTAB-Map artifact domain models."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

Matrix4x4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


class RtabmapNode(BaseModel):
    """RTAB-Map Node row decoded into a homogeneous 4x4 pose."""

    model_config = ConfigDict(frozen=True)

    node_id: int
    map_id: int
    stamp: float
    pose: Matrix4x4
    label: str | None = None


class RtabmapLink(BaseModel):
    """RTAB-Map Link row summary."""

    model_config = ConfigDict(frozen=True)

    from_id: int
    to_id: int
    link_type: int
    transform: Matrix4x4 | None = None


class RtabmapDataSummary(BaseModel):
    """RTAB-Map Data row byte-size summary."""

    model_config = ConfigDict(frozen=True)

    node_id: int
    image_bytes: int | None
    depth_bytes: int | None
    calibration_bytes: int | None
    scan_bytes: int | None


class RtabmapDataFrame(BaseModel):
    """RTAB-Map Data row with compressed image/depth/calibration blobs."""

    model_config = ConfigDict(frozen=True)

    node_id: int
    image_bytes: bytes | None
    depth_bytes: bytes | None
    calibration_bytes: bytes | None


class RtabmapFeaturePoint(BaseModel):
    """RTAB-Map Feature row with local 3D coordinate."""

    model_config = ConfigDict(frozen=True)

    node_id: int
    word_id: int
    pixel_x: float
    pixel_y: float
    local_xyz: tuple[float, float, float]
    descriptor_size: int | None = None
    descriptor: bytes | None = None


class RtabmapDiagnostics(BaseModel):
    """Readiness summary for using an uploaded rtabmap.db as map source."""

    model_config = ConfigDict(frozen=True)

    db_path: str
    ready: bool
    issues: list[str] = Field(default_factory=list)
    node_count: int = 0
    data_count: int = 0
    link_count: int = 0
    neighbor_link_count: int = 0
    loop_closure_link_count: int = 0
    feature_count: int = 0
    feature_3d_count: int = 0
    word_count: int = 0
    global_descriptor_count: int = 0
    statistics_count: int = 0
    data_image_count: int = 0
    data_depth_count: int = 0
    data_calibration_count: int = 0
    keyframe_count: int = 0
    keyframe_node_count: int = 0
    keyframe_node_coverage: float = 0.0
    min_node_id: int | None = None
    max_node_id: int | None = None
    min_stamp: float | None = None
    max_stamp: float | None = None
    duration_s: float | None = None

    @classmethod
    def missing(cls, db_path: Path, issue: str) -> RtabmapDiagnostics:
        return cls(db_path=str(db_path), ready=False, issues=[issue])

    def to_counts_dict(self) -> dict[str, Any]:
        """JSON-safe shape stored under BuildCounts.rtabmap."""
        return {
            "db_path": self.db_path,
            "ready": self.ready,
            "issues": self.issues,
            "node_count": self.node_count,
            "data_count": self.data_count,
            "link_count": self.link_count,
            "neighbor_link_count": self.neighbor_link_count,
            "loop_closure_link_count": self.loop_closure_link_count,
            "feature_count": self.feature_count,
            "feature_3d_count": self.feature_3d_count,
            "word_count": self.word_count,
            "global_descriptor_count": self.global_descriptor_count,
            "statistics_count": self.statistics_count,
            "data_image_count": self.data_image_count,
            "data_depth_count": self.data_depth_count,
            "data_calibration_count": self.data_calibration_count,
            "keyframe_count": self.keyframe_count,
            "keyframe_node_count": self.keyframe_node_count,
            "keyframe_node_coverage": self.keyframe_node_coverage,
            "min_node_id": self.min_node_id,
            "max_node_id": self.max_node_id,
            "min_stamp": self.min_stamp,
            "max_stamp": self.max_stamp,
            "duration_s": self.duration_s,
        }
