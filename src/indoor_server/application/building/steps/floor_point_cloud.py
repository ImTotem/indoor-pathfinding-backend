"""Sprint 46 — Floor segmentation point cloud.

floor mask × per-pixel `Data.depth` × `Data.calibration` × `Node.pose` →
world frame xy point cloud (height filtered).

이 step은 2D 지도 polygon의 source-of-truth를 RTABMap trajectory road에서 floor
segmentation point cloud로 전환하기 위한 구성 요소다. Trajectory가 침대 위를
지나가도 polygon에 포함되지 않도록, 실제 floor mask가 yes 한 픽셀의 metric
depth back-projection만 사용한다.

재사용 부품: `decode_rtabmap_depth`, `decode_rtabmap_calibration`,
`RtabmapCameraCalibration`, `_FloorFrame`. (rtabmap_depth_evidence.py /
rtabmap_trajectory.py 에서 그대로 import — 변경 금지.)

Sprint 51: 픽셀→world 변환은 `building.depth_projection.project_depth_pixels_to_world`
공유 helper 로 추출. 본 step (floor) 와 wall_polygon obstacle source 모두
동일 함수를 호출 — 한 군데 fix 가 양쪽에 즉시 반영된다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from indoor_server.application.building.depth_projection import (
    project_depth_pixels_to_world,
    sample_floor_mask_to_depth_grid,
)
from indoor_server.application.building.steps.rtabmap_depth_evidence import (
    RtabmapCameraCalibration,
    decode_rtabmap_calibration,
    decode_rtabmap_depth,
)
from indoor_server.application.building.steps.rtabmap_trajectory import _FloorFrame
from indoor_server.domain.building.rtabmap_models import RtabmapDataFrame, RtabmapNode

logger = logging.getLogger(__name__)


Z0Mode = Literal["node_pose_median", "depth_percentile", "blended"]


@dataclass(frozen=True)
class FloorPointCloudStepParams:
    """FloorPointCloudStep tunable parameters."""

    pixel_stride: int = 4
    min_depth_m: float = 0.20
    max_depth_m: float = 6.0
    height_tolerance_m: float = 0.30
    z0_mode: Z0Mode = "blended"
    z0_camera_height_offset_m: float = 1.5
    z0_warn_disagreement_m: float = 0.50
    z0_depth_percentile: float = 5.0

    def to_metadata(self) -> dict[str, object]:
        return {
            "pixel_stride": self.pixel_stride,
            "min_depth_m": self.min_depth_m,
            "max_depth_m": self.max_depth_m,
            "height_tolerance_m": self.height_tolerance_m,
            "z0_mode": self.z0_mode,
            "z0_camera_height_offset_m": self.z0_camera_height_offset_m,
            "z0_warn_disagreement_m": self.z0_warn_disagreement_m,
            "z0_depth_percentile": self.z0_depth_percentile,
        }


@dataclass(frozen=True)
class FloorPointCloud:
    """world frame floor point cloud + estimated floor plane."""

    points_xy: np.ndarray  # (N, 2) float64 — world horizontal axes
    z_values: np.ndarray   # (N,) float64 — world vertical axis (post-filter)
    z0: float
    metadata: dict[str, object] = field(default_factory=dict)


class FloorPointCloudStep:
    """Floor mask × Data.depth × calibration × Node.pose → world floor cloud."""

    def __init__(
        self,
        params: FloorPointCloudStepParams | None = None,
    ) -> None:
        self._params = params if params is not None else FloorPointCloudStepParams()
        self._validate()

    def _validate(self) -> None:
        p = self._params
        if p.pixel_stride <= 0:
            raise ValueError(f"pixel_stride must be > 0, got {p.pixel_stride}")
        if p.min_depth_m <= 0:
            raise ValueError(f"min_depth_m must be > 0, got {p.min_depth_m}")
        if p.max_depth_m <= p.min_depth_m:
            raise ValueError(
                f"max_depth_m must be > min_depth_m, got {p.max_depth_m} <= {p.min_depth_m}"
            )
        if p.height_tolerance_m <= 0:
            raise ValueError(
                f"height_tolerance_m must be > 0, got {p.height_tolerance_m}"
            )
        if not (0.0 < p.z0_depth_percentile < 100.0):
            raise ValueError(
                f"z0_depth_percentile must be in (0, 100), got {p.z0_depth_percentile}"
            )

    def run(
        self,
        *,
        nodes: list[RtabmapNode],
        frames: list[RtabmapDataFrame],
        floor_masks_by_node_id: dict[int, np.ndarray],
    ) -> FloorPointCloud:
        params = self._params
        empty_xy = np.zeros((0, 2), dtype=np.float64)
        empty_z = np.zeros((0,), dtype=np.float64)

        if not nodes:
            return FloorPointCloud(
                points_xy=empty_xy,
                z_values=empty_z,
                z0=0.0,
                metadata=self._build_metadata(
                    node_count=0,
                    frames_with_depth=0,
                    frames_with_floor_mask=0,
                    raw_pixel_count=0,
                    valid_depth_pixel_count=0,
                    pre_height_filter_count=0,
                    world_point_count=0,
                    decode_failures=0,
                    z0_node_pose_median=0.0,
                    z0_depth_percentile=0.0,
                    z0_used=0.0,
                    empty_reason="no_nodes",
                ),
            )

        floor_frame = _FloorFrame.from_nodes(nodes)
        node_pose_by_id = {
            node.node_id: np.asarray(node.pose, dtype=np.float64)
            for node in nodes
        }
        frames_by_node_id = {frame.node_id: frame for frame in frames}

        node_z_world = np.array(
            [floor_frame.project_node_z(node) for node in nodes],
            dtype=np.float64,
        )
        z0_node_pose_median = float(np.median(node_z_world))

        # Pass 1 — collect raw world points (no height filter) so we can compute
        # z0_depth_percentile on the actual floor distribution.
        raw_world_points: list[np.ndarray] = []
        raw_pixel_count = 0
        valid_depth_pixel_count = 0
        frames_with_depth = 0
        frames_with_floor_mask = 0
        decode_failures = 0
        for node in nodes:
            mask = floor_masks_by_node_id.get(node.node_id)
            if mask is None:
                continue
            frame = frames_by_node_id.get(node.node_id)
            if frame is None or frame.depth_bytes is None or frame.calibration_bytes is None:
                continue
            frames_with_depth += 1
            frames_with_floor_mask += 1

            depth = decode_rtabmap_depth(frame.depth_bytes)
            if depth is None:
                decode_failures += 1
                continue
            try:
                calibration = decode_rtabmap_calibration(frame.calibration_bytes)
            except ValueError as e:
                logger.warning(
                    "floor_point_cloud: calibration decode failed node=%d err=%s",
                    node.node_id,
                    e,
                )
                decode_failures += 1
                continue

            pose = node_pose_by_id.get(node.node_id)
            if pose is None:
                continue

            world, sampled, valid_count = _project_floor_pixels_to_world(
                depth=depth,
                floor_mask=mask,
                calibration=calibration,
                node_pose=pose,
                pixel_stride=params.pixel_stride,
                min_depth_m=params.min_depth_m,
                max_depth_m=params.max_depth_m,
            )
            raw_pixel_count += sampled
            valid_depth_pixel_count += valid_count
            if world.size:
                raw_world_points.append(world)

        if not raw_world_points:
            return FloorPointCloud(
                points_xy=empty_xy,
                z_values=empty_z,
                z0=z0_node_pose_median,
                metadata=self._build_metadata(
                    node_count=len(nodes),
                    frames_with_depth=frames_with_depth,
                    frames_with_floor_mask=frames_with_floor_mask,
                    raw_pixel_count=raw_pixel_count,
                    valid_depth_pixel_count=valid_depth_pixel_count,
                    pre_height_filter_count=0,
                    world_point_count=0,
                    decode_failures=decode_failures,
                    z0_node_pose_median=z0_node_pose_median,
                    z0_depth_percentile=0.0,
                    z0_used=z0_node_pose_median,
                    empty_reason="no_valid_world_points",
                ),
            )

        all_world = np.vstack(raw_world_points)
        pre_height_filter_count = int(all_world.shape[0])
        v_axis = floor_frame.vertical_axis
        h_axes = floor_frame.horizontal_axes
        world_z = all_world[:, v_axis]
        z0_depth_percentile = float(
            np.percentile(world_z, params.z0_depth_percentile)
        )

        z0_used, disagreement, warn = _resolve_z0(
            z0_node_pose_median=z0_node_pose_median,
            z0_depth_percentile=z0_depth_percentile,
            params=params,
        )

        keep = np.abs(world_z - z0_used) <= params.height_tolerance_m
        kept = all_world[keep]
        kept_xy = np.column_stack(
            [kept[:, h_axes[0]], kept[:, h_axes[1]]]
        ) if kept.size else empty_xy
        kept_z = kept[:, v_axis] if kept.size else empty_z

        accepted_ratio = (
            float(kept.shape[0]) / float(pre_height_filter_count)
            if pre_height_filter_count > 0
            else 0.0
        )

        empty_reason: str | None = None
        if int(kept.shape[0]) == 0:
            empty_reason = "all_points_filtered_by_height"

        return FloorPointCloud(
            points_xy=np.asarray(kept_xy, dtype=np.float64),
            z_values=np.asarray(kept_z, dtype=np.float64),
            z0=z0_used,
            metadata=self._build_metadata(
                node_count=len(nodes),
                frames_with_depth=frames_with_depth,
                frames_with_floor_mask=frames_with_floor_mask,
                raw_pixel_count=raw_pixel_count,
                valid_depth_pixel_count=valid_depth_pixel_count,
                pre_height_filter_count=pre_height_filter_count,
                world_point_count=int(kept.shape[0]),
                decode_failures=decode_failures,
                z0_node_pose_median=z0_node_pose_median,
                z0_depth_percentile=z0_depth_percentile,
                z0_used=z0_used,
                z0_disagreement_m=disagreement,
                z0_disagreement_warn=warn,
                accepted_ratio=accepted_ratio,
                floor_frame=floor_frame.metadata(),
                empty_reason=empty_reason,
            ),
        )

    def _build_metadata(
        self,
        *,
        node_count: int,
        frames_with_depth: int,
        frames_with_floor_mask: int,
        raw_pixel_count: int,
        valid_depth_pixel_count: int,
        pre_height_filter_count: int,
        world_point_count: int,
        decode_failures: int,
        z0_node_pose_median: float,
        z0_depth_percentile: float,
        z0_used: float,
        z0_disagreement_m: float = 0.0,
        z0_disagreement_warn: bool = False,
        accepted_ratio: float = 0.0,
        floor_frame: dict[str, object] | None = None,
        empty_reason: str | None = None,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "node_count": node_count,
            "frames_with_depth": frames_with_depth,
            "frames_with_floor_mask": frames_with_floor_mask,
            "raw_pixel_count": raw_pixel_count,
            "valid_depth_pixel_count": valid_depth_pixel_count,
            "pre_height_filter_count": pre_height_filter_count,
            "world_point_count": world_point_count,
            "decode_failures": decode_failures,
            "accepted_ratio": accepted_ratio,
            "z0_node_pose_median": z0_node_pose_median,
            "z0_depth_percentile": z0_depth_percentile,
            "z0_used": z0_used,
            "z0_disagreement_m": z0_disagreement_m,
            "z0_disagreement_warn": z0_disagreement_warn,
            "params": self._params.to_metadata(),
        }
        if floor_frame is not None:
            metadata["floor_frame"] = floor_frame
        if empty_reason is not None:
            metadata["empty_reason"] = empty_reason
        return metadata


def _resolve_z0(
    *,
    z0_node_pose_median: float,
    z0_depth_percentile: float,
    params: FloorPointCloudStepParams,
) -> tuple[float, float, bool]:
    """z0 추정 — design §3.1 / §7.2."""
    expected_floor_from_pose = z0_node_pose_median - params.z0_camera_height_offset_m
    disagreement = float(abs(expected_floor_from_pose - z0_depth_percentile))
    warn = disagreement > params.z0_warn_disagreement_m

    if params.z0_mode == "node_pose_median":
        return z0_node_pose_median, disagreement, warn
    if params.z0_mode == "depth_percentile":
        return z0_depth_percentile, disagreement, warn
    # blended: 실측 우선
    return z0_depth_percentile, disagreement, warn


def _project_floor_pixels_to_world(
    *,
    depth: np.ndarray,
    floor_mask: np.ndarray,
    calibration: RtabmapCameraCalibration,
    node_pose: np.ndarray,
    pixel_stride: int,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, int, int]:
    """floor mask 픽셀 grid → world frame xyz.

    반환: (world_xyz (N,3), sampled_pixel_count, valid_depth_pixel_count).

    Sprint 51: shared `project_depth_pixels_to_world` helper 위임. 동작/dtype
    /순서 모두 inline 시점과 동일하게 유지된다 (golden test 보호).
    """
    return project_depth_pixels_to_world(
        depth=depth,
        floor_mask=floor_mask,
        calibration=calibration,
        node_pose=node_pose,
        pixel_stride=pixel_stride,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        invert_mask=False,
    )


def _sample_floor_mask(
    floor_mask: np.ndarray,
    depth_shape: tuple[int, int],
    yy: np.ndarray,
    xx: np.ndarray,
) -> np.ndarray:
    """floor mask가 depth shape과 다르면 nearest-neighbor resample.

    Sprint 51: backward-compat shim — `building.depth_projection.sample_floor_mask_to_depth_grid`
    호출.
    """
    return sample_floor_mask_to_depth_grid(floor_mask, depth_shape, yy, xx)
