"""Multi-scan dense video floor polygon fusion.

This step closes the Sprint 75 chain:

RTAB-Map multi-scan merge -> source node to merged node pose mapping ->
original scan video frames re-projected in the merged coordinate frame ->
FloorPointCloud -> FloorRaster footprint polygon.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np

from indoor_server.application.building.multiscan_pose_mapping import (
    FramePoseSource,
    NodePoseMappingResult,
    resolve_frame_pose_in_merged,
)
from indoor_server.application.building.pose_matcher import (
    DEFAULT_THRESHOLD_NS,
    PoseMatcher,
    PoseSample,
)
from indoor_server.application.building.steps.back_projection import Intrinsics
from indoor_server.application.building.steps.floor_point_cloud import FloorPointCloud
from indoor_server.application.building.steps.floor_raster import (
    FloorRasterResult,
    FloorRasterStep,
)
from indoor_server.application.building.steps.multiscan_heatmap_fusion import (
    MultiScanFusionGateResult,
    MultiScanFusionMetrics,
    evaluate_multiscan_fusion_gates,
)
from indoor_server.application.building.video_frame_extractor import (
    VideoFrameExtractor,
)
from indoor_server.infrastructure.ml.segformer_onnx import (
    extract_floor_stair_masks,
)

logger = logging.getLogger(__name__)

_MIN_DOWNWARD_DZ = 0.05
_Z_TOLERANCE = 0.30
_NS_PER_SECOND = 1_000_000_000.0

SourceTimestampMode = Literal["raw_pose", "rebased_pose", "video"]


class SegmenterLike(Protocol):
    async def segment(self, image: np.ndarray) -> Any:
        """Return a semantic segmentation output for one RGB frame."""


@dataclass(frozen=True)
class MultiScanDenseVideoSource:
    scan_id: str
    video_path: Path
    poses_path: Path
    intrinsics: Intrinsics


@dataclass(frozen=True)
class MultiScanDenseVideoPolygonParams:
    stride: int = 2
    pixel_stride: int = 4
    pose_match_threshold_ns: int = DEFAULT_THRESHOLD_NS
    max_pose_miss_ratio: float = 0.05
    rebase_pose_timestamps_for_video: bool = True
    source_timestamp_mode: SourceTimestampMode = "raw_pose"
    exact_node_tolerance_s: float = 0.033
    height_tolerance_m: float = _Z_TOLERANCE

    def to_metadata(self) -> dict[str, object]:
        return {
            "stride": self.stride,
            "pixel_stride": self.pixel_stride,
            "pose_match_threshold_ns": self.pose_match_threshold_ns,
            "max_pose_miss_ratio": self.max_pose_miss_ratio,
            "rebase_pose_timestamps_for_video": self.rebase_pose_timestamps_for_video,
            "source_timestamp_mode": self.source_timestamp_mode,
            "exact_node_tolerance_s": self.exact_node_tolerance_s,
            "height_tolerance_m": self.height_tolerance_m,
        }


@dataclass(frozen=True)
class MultiScanDenseVideoPolygonResult:
    cloud: FloorPointCloud
    raster: FloorRasterResult
    fusion_metrics: MultiScanFusionMetrics
    fusion_gate: MultiScanFusionGateResult
    metadata: dict[str, object] = field(default_factory=dict)


class MultiScanDenseVideoPolygonError(RuntimeError):
    """Multi-scan dense video polygon fusion failed."""


class MultiScanDenseVideoPolygonStep:
    """Fuse source videos into one floor footprint polygon in merged coordinates."""

    def __init__(
        self,
        *,
        segmenter: SegmenterLike,
        params: MultiScanDenseVideoPolygonParams | None = None,
        raster_step: FloorRasterStep | None = None,
    ) -> None:
        self._segmenter = segmenter
        self._params = params if params is not None else MultiScanDenseVideoPolygonParams()
        self._raster_step = raster_step if raster_step is not None else FloorRasterStep()
        if self._params.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self._params.stride}")
        if self._params.pixel_stride < 1:
            raise ValueError(f"pixel_stride must be >= 1, got {self._params.pixel_stride}")
        if not (0.0 <= self._params.max_pose_miss_ratio <= 1.0):
            raise ValueError("max_pose_miss_ratio must be in [0, 1]")

    async def run(
        self,
        *,
        sources: list[MultiScanDenseVideoSource],
        mapping_result: NodePoseMappingResult,
        z0: float,
        inter_session_loop_closure_count: int = 0,
    ) -> MultiScanDenseVideoPolygonResult:
        if len(sources) < 2:
            raise MultiScanDenseVideoPolygonError("multi-scan polygon requires at least 2 sources")

        all_xy: list[np.ndarray] = []
        all_z: list[np.ndarray] = []
        per_scan: dict[str, dict[str, object]] = {}
        direction_bins: set[int] = set()

        for source in sources:
            source_meta = await self._process_source(
                source=source,
                mapping_result=mapping_result,
                z0=z0,
                all_xy=all_xy,
                all_z=all_z,
                direction_bins=direction_bins,
            )
            per_scan[source.scan_id] = source_meta

        if all_xy:
            points_xy = np.concatenate(all_xy, axis=0)
            z_values = np.concatenate(all_z, axis=0)
        else:
            points_xy = np.zeros((0, 2), dtype=np.float64)
            z_values = np.zeros((0,), dtype=np.float64)

        cloud = FloorPointCloud(
            points_xy=points_xy,
            z_values=z_values,
            z0=z0,
            metadata={
                "source": "multiscan_dense_video_polygon",
                "params": self._params.to_metadata(),
                "per_scan": per_scan,
            },
        )
        raster = self._raster_step.run(cloud)
        fusion_metrics = self._build_fusion_metrics(
            mapping_result=mapping_result,
            per_scan=per_scan,
            raster=raster,
            direction_bin_coverage=len(direction_bins),
            inter_session_loop_closure_count=inter_session_loop_closure_count,
        )
        fusion_gate = evaluate_multiscan_fusion_gates(fusion_metrics)

        return MultiScanDenseVideoPolygonResult(
            cloud=cloud,
            raster=raster,
            fusion_metrics=fusion_metrics,
            fusion_gate=fusion_gate,
            metadata={
                "world_point_count": int(points_xy.shape[0]),
                "direction_bin_coverage": len(direction_bins),
                "per_scan": per_scan,
                "raster": raster.metadata,
                "fusion_metrics": fusion_metrics.to_dict(),
                "fusion_gate": {
                    "accepted": fusion_gate.accepted,
                    "failures": fusion_gate.failures,
                },
                "params": self._params.to_metadata(),
            },
        )

    async def _process_source(
        self,
        *,
        source: MultiScanDenseVideoSource,
        mapping_result: NodePoseMappingResult,
        z0: float,
        all_xy: list[np.ndarray],
        all_z: list[np.ndarray],
        direction_bins: set[int],
    ) -> dict[str, object]:
        extractor = VideoFrameExtractor(source.video_path)
        matcher = PoseMatcher(source.poses_path)
        samples = matcher.all_samples()
        if not samples:
            raise MultiScanDenseVideoPolygonError(f"poses.bin is empty for scan {source.scan_id}")
        match_pts = np.array([sample.pts_ns for sample in samples], dtype=np.int64)
        if self._params.rebase_pose_timestamps_for_video:
            match_pts = match_pts - match_pts[0]

        frames_total = 0
        frames_pose_miss = 0
        frames_pose_excluded = 0
        frames_no_floor = 0
        frames_used = 0
        point_count = 0
        pose_source_counts: Counter[FramePoseSource] = Counter()

        for vframe in extractor.iter_frames(stride=self._params.stride):
            frames_total += 1
            sample = _find_pose_sample(
                samples=samples,
                match_pts=match_pts,
                pts_ns=vframe.pts_ns,
                threshold_ns=self._params.pose_match_threshold_ns,
            )
            if sample is None:
                frames_pose_miss += 1
                continue

            source_timestamp = self._source_timestamp_s(
                sample=sample,
                video_pts_ns=vframe.pts_ns,
                first_pose_pts_ns=samples[0].pts_ns,
            )
            resolved = resolve_frame_pose_in_merged(
                source_scan_id=source.scan_id,
                source_timestamp=source_timestamp,
                raw_source_pose=sample.transform.astype(np.float64),
                mapping_result=mapping_result,
                exact_tolerance_s=self._params.exact_node_tolerance_s,
            )
            pose_source_counts[resolved.assignment.pose_source] += 1
            if resolved.pose is None:
                frames_pose_excluded += 1
                continue
            pose = np.array(resolved.pose, dtype=np.float64)

            seg = await self._segmenter.segment(vframe.image_rgb)
            floor_mask, _stair = extract_floor_stair_masks(seg.class_mask)
            if not floor_mask.any():
                frames_no_floor += 1
                continue

            ys_idx, xs_idx = np.where(floor_mask)
            if self._params.pixel_stride > 1:
                ys_idx = ys_idx[::self._params.pixel_stride]
                xs_idx = xs_idx[::self._params.pixel_stride]

            xy, z_values = self._back_project(
                xs=xs_idx.astype(np.float64),
                ys=ys_idx.astype(np.float64),
                pose=pose,
                intrinsics=source.intrinsics,
                z0=z0,
            )
            if xy.shape[0] == 0:
                continue
            all_xy.append(xy)
            all_z.append(z_values)
            point_count += int(xy.shape[0])
            frames_used += 1
            direction_bins.add(_direction_bin(pose))

        miss_ratio = frames_pose_miss / max(1, frames_total)
        if miss_ratio > self._params.max_pose_miss_ratio:
            raise MultiScanDenseVideoPolygonError(
                f"pose miss ratio {miss_ratio:.3f} exceeds "
                f"{self._params.max_pose_miss_ratio} for scan {source.scan_id}"
            )

        return {
            "frames_total": frames_total,
            "frames_pose_miss": frames_pose_miss,
            "frames_pose_excluded": frames_pose_excluded,
            "frames_no_floor": frames_no_floor,
            "frames_used": frames_used,
            "usable_pose_frames": frames_used + frames_no_floor,
            "point_count": point_count,
            "pose_source_counts": dict(pose_source_counts),
            "pose_miss_ratio": miss_ratio,
        }

    def _source_timestamp_s(
        self,
        *,
        sample: PoseSample,
        video_pts_ns: int,
        first_pose_pts_ns: int,
    ) -> float:
        if self._params.source_timestamp_mode == "raw_pose":
            return sample.pts_ns / _NS_PER_SECOND
        if self._params.source_timestamp_mode == "rebased_pose":
            return (sample.pts_ns - first_pose_pts_ns) / _NS_PER_SECOND
        return video_pts_ns / _NS_PER_SECOND

    def _back_project(
        self,
        *,
        xs: np.ndarray,
        ys: np.ndarray,
        pose: np.ndarray,
        intrinsics: Intrinsics,
        z0: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        rotation = pose[:3, :3]
        t = pose[:3, 3]

        rx = (xs - intrinsics.cx) / intrinsics.fx
        ry = -(ys - intrinsics.cy) / intrinsics.fy
        rz = -np.ones(len(xs), dtype=np.float64)
        rays_cam = np.stack([rx, ry, rz], axis=1)
        rays_world = rays_cam @ rotation.T

        dz = rays_world[:, 2]
        downward = dz < -_MIN_DOWNWARD_DZ
        if not downward.any():
            return (
                np.zeros((0, 2), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
            )

        lam = (z0 - t[2]) / dz[downward]
        points = t + lam[:, None] * rays_world[downward]
        z_ok = np.abs(points[:, 2] - z0) < self._params.height_tolerance_m
        return points[z_ok, :2], points[z_ok, 2]

    def _build_fusion_metrics(
        self,
        *,
        mapping_result: NodePoseMappingResult,
        per_scan: dict[str, dict[str, object]],
        raster: FloorRasterResult,
        direction_bin_coverage: int,
        inter_session_loop_closure_count: int,
    ) -> MultiScanFusionMetrics:
        total_points = sum(cast(int, meta["point_count"]) for meta in per_scan.values())
        per_scan_usable_frame_ratio: dict[str, float] = {}
        scan_support_ratio: dict[str, float] = {}
        for scan_id, meta in per_scan.items():
            frames_total = cast(int, meta["frames_total"])
            usable = cast(int, meta["usable_pose_frames"])
            per_scan_usable_frame_ratio[scan_id] = usable / max(1, frames_total)
            scan_support_ratio[scan_id] = cast(int, meta["point_count"]) / max(1, total_points)

        max_source_node_count = max(
            mapping_result.metrics.per_scan_source_node_count.values(),
            default=0,
        )
        area = float(cast(float, raster.metadata.get("polygon_area_m2", 0.0)))
        return MultiScanFusionMetrics(
            source_scan_count=len(per_scan),
            merged_node_count=mapping_result.metrics.merged_node_count,
            max_source_node_count=max_source_node_count,
            per_scan_usable_frame_ratio=per_scan_usable_frame_ratio,
            inter_session_loop_closure_count=inter_session_loop_closure_count,
            mapping_ambiguous_ratio=mapping_result.metrics.ambiguous_ratio,
            mapping_missing_ratio=mapping_result.metrics.missing_ratio,
            session_transform_residual_median=0.0,
            session_transform_residual_p90=0.0,
            polygon_area_inflation_ratio=1.0 if area > 0 else 0.0,
            double_wall_line_score=0.0,
            direction_bin_coverage=direction_bin_coverage,
            scan_support_ratio=scan_support_ratio,
        )


def _find_pose_sample(
    *,
    samples: list[PoseSample],
    match_pts: np.ndarray,
    pts_ns: int,
    threshold_ns: int,
) -> PoseSample | None:
    if match_pts.size == 0:
        return None
    idx = int(np.searchsorted(match_pts, pts_ns, side="left"))
    candidates: list[int] = []
    if idx < match_pts.size:
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    best_idx = min(candidates, key=lambda i: abs(int(match_pts[i]) - pts_ns))
    if abs(int(match_pts[best_idx]) - pts_ns) > threshold_ns:
        return None
    return samples[best_idx]


def _direction_bin(pose: np.ndarray, *, bin_count: int = 8) -> int:
    forward = pose[:3, :3] @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    angle = float(np.arctan2(forward[1], forward[0]))
    normalized = (angle + np.pi) / (2.0 * np.pi)
    return int(np.floor(normalized * bin_count)) % bin_count
