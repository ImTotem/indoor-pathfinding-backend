"""Sprint 51 — Step 0: RTABMap depth → world obstacle heatmap.

floor mask 의 inverse pixel × `Data.depth` × `Node.pose` 를 world frame xy
plane 에 누적해 obstacle hit count grid 를 만든다. floor pointcloud 와 동일
helper (`building.depth_projection.project_depth_pixels_to_world`) 를 사용
하므로 변환 path 가 한 군데로 통일된다.

기여:
- height filter: `z0 + height_above_floor_min_m..max_m` 만 obstacle 후보
  (천장/바닥 cluster 제거).
- grid origin / shape 자동 결정 — 점 분포 bbox + padding.
- LiDAR 의존 0 (RTABMap RGB stereo / mono SfM 만 사용).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from indoor_server.application.building.depth_projection import (
    project_depth_pixels_to_world,
)
from indoor_server.application.building.steps.rtabmap_depth_evidence import (
    decode_rtabmap_calibration,
    decode_rtabmap_depth,
)
from indoor_server.application.building.steps.rtabmap_trajectory import _FloorFrame
from indoor_server.domain.building.rtabmap_models import RtabmapDataFrame, RtabmapNode

logger = logging.getLogger(__name__)

ObstacleMaskMode = Literal["inverse_floor", "direct_mask"]


@dataclass(frozen=True)
class ObstacleSourceStepParams:
    """ObstacleSourceStep tunable parameters."""

    pixel_stride: int = 4
    min_depth_m: float = 0.20
    max_depth_m: float = 6.0
    height_above_floor_min_m: float = 0.30
    height_above_floor_max_m: float = 2.50
    cell_size_m: float = 0.10
    grid_padding_cells: int = 8
    mask_mode: ObstacleMaskMode = "inverse_floor"

    def to_metadata(self) -> dict[str, object]:
        return {
            "pixel_stride": self.pixel_stride,
            "min_depth_m": self.min_depth_m,
            "max_depth_m": self.max_depth_m,
            "height_above_floor_min_m": self.height_above_floor_min_m,
            "height_above_floor_max_m": self.height_above_floor_max_m,
            "cell_size_m": self.cell_size_m,
            "grid_padding_cells": self.grid_padding_cells,
            "mask_mode": self.mask_mode,
        }


@dataclass(frozen=True)
class ObstacleHeatmap:
    """world frame x,y plane 에 누적된 obstacle hit count grid."""

    counts: np.ndarray  # (H, W) int32
    origin_x: float
    origin_y: float
    cell_size_m: float
    z0: float
    height_min_m: float
    height_max_m: float
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.counts.shape[0]), int(self.counts.shape[1]))

    def cell_to_world(self, row: float, col: float) -> tuple[float, float]:
        """grid cell index → world (x, y) (cell center)."""
        x = self.origin_x + (float(col) + 0.5) * self.cell_size_m
        y = self.origin_y + (float(row) + 0.5) * self.cell_size_m
        return (x, y)


class ObstacleSourceStep:
    """RTABMap depth → world obstacle heatmap (Step 0)."""

    def __init__(self, params: ObstacleSourceStepParams | None = None) -> None:
        self._params = params if params is not None else ObstacleSourceStepParams()
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
        if p.height_above_floor_min_m < 0:
            raise ValueError("height_above_floor_min_m must be >= 0")
        if p.height_above_floor_max_m <= p.height_above_floor_min_m:
            raise ValueError(
                "height_above_floor_max_m must be > height_above_floor_min_m"
            )
        if p.cell_size_m <= 0:
            raise ValueError("cell_size_m must be > 0")
        if p.grid_padding_cells < 0:
            raise ValueError("grid_padding_cells must be >= 0")
        if p.mask_mode not in ("inverse_floor", "direct_mask"):
            raise ValueError(f"unsupported mask_mode: {p.mask_mode}")

    def run(
        self,
        *,
        nodes: list[RtabmapNode],
        frames: list[RtabmapDataFrame],
        floor_masks_by_node_id: dict[int, np.ndarray],
        obstacle_masks_by_node_id: dict[int, np.ndarray] | None = None,
        z0: float,
    ) -> ObstacleHeatmap:
        params = self._params
        uses_direct_mask = params.mask_mode == "direct_mask"
        obstacle_masks = obstacle_masks_by_node_id or {}
        mask_source = "direct_obstacle_mask" if uses_direct_mask else "inverse_floor_mask"
        empty_counts = np.zeros((1, 1), dtype=np.int32)
        empty_meta_base: dict[str, object] = {
            "world_obstacle_point_count": 0,
            "frames_with_depth": 0,
            "frames_with_floor_mask": 0,
            "frames_with_obstacle_mask": 0,
            "mask_coverage_ratio": 0.0,
            "mask_mode": params.mask_mode,
            "mask_source": mask_source,
            "decode_failures": 0,
            "height_filtered_count": 0,
            "height_min_m": z0 + params.height_above_floor_min_m,
            "height_max_m": z0 + params.height_above_floor_max_m,
            "cell_size_m": params.cell_size_m,
            "grid_shape": [1, 1],
            "params": params.to_metadata(),
        }

        if not nodes or not frames:
            return ObstacleHeatmap(
                counts=empty_counts,
                origin_x=0.0,
                origin_y=0.0,
                cell_size_m=params.cell_size_m,
                z0=z0,
                height_min_m=z0 + params.height_above_floor_min_m,
                height_max_m=z0 + params.height_above_floor_max_m,
                metadata={
                    **empty_meta_base,
                    "empty_reason": "no_nodes_or_frames",
                },
            )

        floor_frame = _FloorFrame.from_nodes(nodes)
        node_pose_by_id = {
            node.node_id: np.asarray(node.pose, dtype=np.float64)
            for node in nodes
        }
        frames_by_node_id = {frame.node_id: frame for frame in frames}

        h_axes = floor_frame.horizontal_axes
        v_axis = floor_frame.vertical_axis
        height_min = z0 + params.height_above_floor_min_m
        height_max = z0 + params.height_above_floor_max_m

        all_world_xy: list[np.ndarray] = []
        frames_with_depth = 0
        frames_with_floor_mask = 0
        frames_with_obstacle_mask = 0
        decode_failures = 0
        pre_height_count = 0

        for node in nodes:
            frame = frames_by_node_id.get(node.node_id)
            if frame is None:
                continue
            if frame.depth_bytes is None or frame.calibration_bytes is None:
                continue
            frames_with_depth += 1

            depth = decode_rtabmap_depth(frame.depth_bytes)
            if depth is None:
                decode_failures += 1
                continue
            try:
                calibration = decode_rtabmap_calibration(frame.calibration_bytes)
            except ValueError as e:
                logger.warning(
                    "obstacle_source: calibration decode failed node=%d err=%s",
                    node.node_id,
                    e,
                )
                decode_failures += 1
                continue

            pose = node_pose_by_id.get(node.node_id)
            if pose is None:
                continue

            if uses_direct_mask:
                mask = obstacle_masks.get(node.node_id)
                if mask is not None:
                    frames_with_obstacle_mask += 1
                else:
                    continue
            else:
                mask = floor_masks_by_node_id.get(node.node_id)
                if mask is not None:
                    frames_with_floor_mask += 1
            # inverse_floor: floor 가 아닌 픽셀(천장/벽/가구)을 통과시키는 legacy path.
            # direct_mask: wall/nonwalkable 같은 직접 obstacle mask 픽셀만 통과.
            # direct_mask mode 에서는 mask 없는 frame 을 건너뛰어 wall/nonwalkable
            # segmentation 이 없는 depth frame 이 heatmap 을 오염시키지 않게 한다.
            world, _sampled, valid = project_depth_pixels_to_world(
                depth=depth,
                floor_mask=mask,
                calibration=calibration,
                node_pose=pose,
                pixel_stride=params.pixel_stride,
                min_depth_m=params.min_depth_m,
                max_depth_m=params.max_depth_m,
                invert_mask=not uses_direct_mask,
            )
            if valid == 0 or world.shape[0] == 0:
                continue
            pre_height_count += int(world.shape[0])

            world_v = world[:, v_axis]
            keep = (world_v >= height_min) & (world_v <= height_max)
            if not np.any(keep):
                continue
            world_xy = np.column_stack(
                [world[keep, h_axes[0]], world[keep, h_axes[1]]]
            )
            all_world_xy.append(world_xy)

        if not all_world_xy:
            mask_ratio_empty = (
                float(
                    frames_with_obstacle_mask
                    if uses_direct_mask
                    else frames_with_floor_mask
                )
                / float(max(1, frames_with_depth))
                if frames_with_depth > 0
                else 0.0
            )
            return ObstacleHeatmap(
                counts=empty_counts,
                origin_x=0.0,
                origin_y=0.0,
                cell_size_m=params.cell_size_m,
                z0=z0,
                height_min_m=height_min,
                height_max_m=height_max,
                metadata={
                    **empty_meta_base,
                    "frames_with_depth": frames_with_depth,
                    "frames_with_floor_mask": frames_with_floor_mask,
                    "frames_with_obstacle_mask": frames_with_obstacle_mask,
                    "mask_coverage_ratio": mask_ratio_empty,
                    "decode_failures": decode_failures,
                    "height_filtered_count": pre_height_count,
                    "empty_reason": "no_obstacle_points",
                },
            )

        xy = np.vstack(all_world_xy)
        x_min = float(xy[:, 0].min())
        x_max = float(xy[:, 0].max())
        y_min = float(xy[:, 1].min())
        y_max = float(xy[:, 1].max())

        cell = params.cell_size_m
        pad = params.grid_padding_cells
        origin_x = x_min - pad * cell
        origin_y = y_min - pad * cell
        # +1 to include max cell, +pad on far side
        w_cells = max(
            1,
            int(np.ceil((x_max - origin_x) / cell)) + pad + 1,
        )
        h_cells = max(
            1,
            int(np.ceil((y_max - origin_y) / cell)) + pad + 1,
        )

        counts = np.zeros((h_cells, w_cells), dtype=np.int32)
        cols = np.floor((xy[:, 0] - origin_x) / cell).astype(np.int64)
        rows = np.floor((xy[:, 1] - origin_y) / cell).astype(np.int64)
        in_bounds = (
            (rows >= 0) & (rows < h_cells) & (cols >= 0) & (cols < w_cells)
        )
        rows = rows[in_bounds]
        cols = cols[in_bounds]
        np.add.at(counts, (rows, cols), 1)

        world_obstacle_point_count = int(xy.shape[0])
        height_filtered_count = pre_height_count - world_obstacle_point_count

        mask_coverage_ratio = (
            float(
                frames_with_obstacle_mask
                if uses_direct_mask
                else frames_with_floor_mask
            )
            / float(max(1, frames_with_depth))
            if frames_with_depth > 0
            else 0.0
        )
        metadata: dict[str, object] = {
            "world_obstacle_point_count": world_obstacle_point_count,
            "frames_with_depth": frames_with_depth,
            "frames_with_floor_mask": frames_with_floor_mask,
            "frames_with_obstacle_mask": frames_with_obstacle_mask,
            "mask_coverage_ratio": mask_coverage_ratio,
            "mask_mode": params.mask_mode,
            "mask_source": mask_source,
            "decode_failures": decode_failures,
            "height_filtered_count": int(max(0, height_filtered_count)),
            "height_min_m": float(height_min),
            "height_max_m": float(height_max),
            "cell_size_m": float(cell),
            "grid_shape": [int(h_cells), int(w_cells)],
            "origin_x": float(origin_x),
            "origin_y": float(origin_y),
            "params": params.to_metadata(),
        }

        return ObstacleHeatmap(
            counts=counts,
            origin_x=origin_x,
            origin_y=origin_y,
            cell_size_m=cell,
            z0=z0,
            height_min_m=height_min,
            height_max_m=height_max,
            metadata=metadata,
        )
