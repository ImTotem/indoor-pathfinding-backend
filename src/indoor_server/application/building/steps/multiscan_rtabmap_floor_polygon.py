"""Multi-scan RTAB-Map image/depth fallback polygon fusion.

Video mode is preferred when every source scan has `scan.mp4` and `poses.bin`.
This fallback supports older scans by using each original RTAB-Map `Data`
image/depth/calibration row as evidence, while replacing source node poses with
the optimized poses from the merged multi-scan RTAB-Map database.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from indoor_server.application.building.multiscan_pose_mapping import (
    NodePoseMappingResult,
)
from indoor_server.application.building.steps.floor_point_cloud import (
    FloorPointCloud,
    FloorPointCloudStep,
    FloorPointCloudStepParams,
)
from indoor_server.application.building.steps.floor_raster import (
    FloorRasterResult,
    FloorRasterStep,
)
from indoor_server.application.building.steps.multiscan_heatmap_fusion import (
    MultiScanFusionGateResult,
    MultiScanFusionMetrics,
    evaluate_multiscan_fusion_gates,
)
from indoor_server.application.building.steps.rtabmap_image_evidence import (
    OrientationMode,
    RtabmapImageEvidenceStep,
)
from indoor_server.application.building.steps.wall_polygon.facade import (
    WallPolygonFromObstacleStep,
    WallPolygonResult,
)
from indoor_server.application.building.steps.wall_polygon.obstacle_source import (
    ObstacleHeatmap,
    ObstacleSourceStep,
    ObstacleSourceStepParams,
)
from indoor_server.application.rtabmap.reader import RtabmapReader
from indoor_server.domain.building.rtabmap_models import RtabmapNode
from indoor_server.infrastructure.ml.protocol import SegmentationOutput

logger = logging.getLogger(__name__)


class SegmenterLike(Protocol):
    async def segment(self, image: np.ndarray) -> SegmentationOutput:
        """Return semantic segmentation for one RGB image."""


@dataclass(frozen=True)
class MultiScanRtabmapFloorSource:
    scan_id: str
    db_path: Path


@dataclass(frozen=True)
class MultiScanRtabmapFloorPolygonParams:
    floor_pixel_stride: int = 4
    floor_height_tolerance_m: float = 0.30
    image_orientation_mode: OrientationMode = "sensor"
    image_floor_mask_min_ratio: float = 0.0
    image_wall_mask_max_ratio: float = 1.0
    include_wall_polygon: bool = True
    wall_pixel_stride: int = 4

    def to_metadata(self) -> dict[str, object]:
        return {
            "floor_pixel_stride": self.floor_pixel_stride,
            "floor_height_tolerance_m": self.floor_height_tolerance_m,
            "image_orientation_mode": self.image_orientation_mode,
            "image_floor_mask_min_ratio": self.image_floor_mask_min_ratio,
            "image_wall_mask_max_ratio": self.image_wall_mask_max_ratio,
            "include_wall_polygon": self.include_wall_polygon,
            "wall_pixel_stride": self.wall_pixel_stride,
        }


@dataclass(frozen=True)
class MultiScanRtabmapFloorPolygonResult:
    cloud: FloorPointCloud
    raster: FloorRasterResult
    fusion_metrics: MultiScanFusionMetrics
    fusion_gate: MultiScanFusionGateResult
    wall_polygon: WallPolygonResult | None = None
    wall_heatmap: ObstacleHeatmap | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class MultiScanRtabmapFloorPolygonError(RuntimeError):
    """Multi-scan RTAB-Map fallback polygon fusion failed."""


class MultiScanRtabmapFloorPolygonStep:
    """Fuse RTAB-Map image/depth evidence into a merged-frame floor polygon."""

    def __init__(
        self,
        *,
        segmenter: SegmenterLike,
        params: MultiScanRtabmapFloorPolygonParams | None = None,
        raster_step: FloorRasterStep | None = None,
        reader: RtabmapReader | None = None,
    ) -> None:
        self._segmenter = segmenter
        self._params = (
            params if params is not None else MultiScanRtabmapFloorPolygonParams()
        )
        self._raster_step = raster_step if raster_step is not None else FloorRasterStep()
        self._reader = reader if reader is not None else RtabmapReader()

    async def run(
        self,
        *,
        sources: list[MultiScanRtabmapFloorSource],
        mapping_result: NodePoseMappingResult,
        inter_session_loop_closure_count: int = 0,
    ) -> MultiScanRtabmapFloorPolygonResult:
        if len(sources) < 2:
            raise MultiScanRtabmapFloorPolygonError(
                "multi-scan RTABMap polygon requires at least 2 sources"
            )

        all_xy: list[np.ndarray] = []
        all_z: list[np.ndarray] = []
        per_scan: dict[str, dict[str, object]] = {}
        wall_heatmaps: list[ObstacleHeatmap] = []

        for source in sources:
            optimized_nodes = optimized_nodes_for_scan(
                mapping_result=mapping_result,
                scan_id=source.scan_id,
            )
            node_ids = {node.node_id for node in optimized_nodes}
            frames = [
                frame
                for frame in self._reader.load_data_frames(source.db_path)
                if frame.node_id in node_ids
            ]
            image_evidence = await RtabmapImageEvidenceStep().run(
                frames,
                segmenter=self._segmenter,
                node_pose_ids=node_ids,
                orientation_mode=self._params.image_orientation_mode,
                floor_mask_min_ratio=self._params.image_floor_mask_min_ratio,
                wall_mask_max_ratio=self._params.image_wall_mask_max_ratio,
            )
            cloud = FloorPointCloudStep(
                FloorPointCloudStepParams(
                    pixel_stride=self._params.floor_pixel_stride,
                    height_tolerance_m=self._params.floor_height_tolerance_m,
                )
            ).run(
                nodes=optimized_nodes,
                frames=frames,
                floor_masks_by_node_id=image_evidence.floor_masks_by_node_id,
            )
            if cloud.points_xy.shape[0] > 0:
                all_xy.append(cloud.points_xy)
                all_z.append(cloud.z_values)

            if self._params.include_wall_polygon:
                heatmap = ObstacleSourceStep(
                    ObstacleSourceStepParams(
                        pixel_stride=self._params.wall_pixel_stride,
                        mask_mode="direct_mask",
                    )
                ).run(
                    nodes=optimized_nodes,
                    frames=frames,
                    floor_masks_by_node_id=image_evidence.floor_masks_by_node_id,
                    obstacle_masks_by_node_id=image_evidence.wall_masks_by_node_id,
                    z0=cloud.z0,
                )
                if heatmap.metadata.get("world_obstacle_point_count", 0):
                    wall_heatmaps.append(heatmap)

            per_scan[source.scan_id] = {
                "node_count": len(optimized_nodes),
                "frame_count": len(frames),
                "segmented_count": image_evidence.segmented_count,
                "floor_mask_node_count": len(image_evidence.floor_masks_by_node_id),
                "wall_mask_node_count": len(image_evidence.wall_masks_by_node_id),
                "point_count": int(cloud.points_xy.shape[0]),
                "z0": cloud.z0,
                "image_evidence": image_evidence.metadata(),
                "floor_cloud": cloud.metadata,
            }

        if all_xy:
            points_xy = np.concatenate(all_xy, axis=0)
            z_values = np.concatenate(all_z, axis=0)
            z0 = float(np.median(z_values)) if z_values.size else 0.0
        else:
            points_xy = np.zeros((0, 2), dtype=np.float64)
            z_values = np.zeros((0,), dtype=np.float64)
            z0 = 0.0

        cloud = FloorPointCloud(
            points_xy=points_xy,
            z_values=z_values,
            z0=z0,
            metadata={
                "source": "multiscan_rtabmap_floor_polygon",
                "params": self._params.to_metadata(),
                "per_scan": per_scan,
            },
        )
        raster = self._raster_step.run(cloud)
        wall_heatmap = merge_obstacle_heatmaps(wall_heatmaps) if wall_heatmaps else None
        wall_polygon = (
            WallPolygonFromObstacleStep().run_from_heatmap(
                wall_heatmap,
                floor_polygon_geojson=raster.footprint_geojson,
            )
            if wall_heatmap is not None
            else None
        )
        fusion_metrics = self._build_fusion_metrics(
            mapping_result=mapping_result,
            per_scan=per_scan,
            raster=raster,
            inter_session_loop_closure_count=inter_session_loop_closure_count,
        )
        fusion_gate = evaluate_multiscan_fusion_gates(fusion_metrics)
        return MultiScanRtabmapFloorPolygonResult(
            cloud=cloud,
            raster=raster,
            fusion_metrics=fusion_metrics,
            fusion_gate=fusion_gate,
            wall_polygon=wall_polygon,
            wall_heatmap=wall_heatmap,
            metadata={
                "world_point_count": int(points_xy.shape[0]),
                "per_scan": per_scan,
                "raster": raster.metadata,
                "wall_polygon": (
                    {
                        "accepted": wall_polygon.accepted,
                        "fail_reason": wall_polygon.fail_reason,
                        "metadata": wall_polygon.metadata,
                    }
                    if wall_polygon is not None
                    else None
                ),
                "fusion_metrics": fusion_metrics.to_dict(),
                "fusion_gate": {
                    "accepted": fusion_gate.accepted,
                    "failures": fusion_gate.failures,
                },
                "params": self._params.to_metadata(),
            },
        )

    def _build_fusion_metrics(
        self,
        *,
        mapping_result: NodePoseMappingResult,
        per_scan: dict[str, dict[str, object]],
        raster: FloorRasterResult,
        inter_session_loop_closure_count: int,
    ) -> MultiScanFusionMetrics:
        total_points = sum(cast(int, meta["point_count"]) for meta in per_scan.values())
        per_scan_usable_frame_ratio: dict[str, float] = {}
        scan_support_ratio: dict[str, float] = {}
        for scan_id, meta in per_scan.items():
            frames = cast(int, meta["frame_count"])
            masks = cast(int, meta["floor_mask_node_count"])
            per_scan_usable_frame_ratio[scan_id] = masks / max(1, frames)
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
            direction_bin_coverage=2 if len(per_scan) >= 2 else 1,
            scan_support_ratio=scan_support_ratio,
        )


def optimized_nodes_for_scan(
    *,
    mapping_result: NodePoseMappingResult,
    scan_id: str,
) -> list[RtabmapNode]:
    nodes: list[RtabmapNode] = []
    for mapping in mapping_result.mappings:
        if mapping.source_scan_id != scan_id or not mapping.is_usable:
            continue
        if mapping.optimized_pose is None:
            continue
        nodes.append(
            RtabmapNode(
                node_id=mapping.source_node_id,
                map_id=0,
                stamp=mapping.source_stamp,
                pose=mapping.optimized_pose,
                label=None,
            )
        )
    nodes.sort(key=lambda node: node.node_id)
    return nodes


def merge_obstacle_heatmaps(heatmaps: list[ObstacleHeatmap]) -> ObstacleHeatmap | None:
    if not heatmaps:
        return None
    cell = heatmaps[0].cell_size_m
    z0 = float(np.median([heatmap.z0 for heatmap in heatmaps]))
    x_min = min(heatmap.origin_x for heatmap in heatmaps)
    y_min = min(heatmap.origin_y for heatmap in heatmaps)
    x_max = max(
        heatmap.origin_x + heatmap.counts.shape[1] * heatmap.cell_size_m
        for heatmap in heatmaps
    )
    y_max = max(
        heatmap.origin_y + heatmap.counts.shape[0] * heatmap.cell_size_m
        for heatmap in heatmaps
    )
    width = max(1, int(np.ceil((x_max - x_min) / cell)))
    height = max(1, int(np.ceil((y_max - y_min) / cell)))
    counts = np.zeros((height, width), dtype=np.int32)
    for heatmap in heatmaps:
        if abs(heatmap.cell_size_m - cell) > 1e-9:
            raise ValueError("cannot merge obstacle heatmaps with different cell sizes")
        row0 = int(round((heatmap.origin_y - y_min) / cell))
        col0 = int(round((heatmap.origin_x - x_min) / cell))
        h, w = heatmap.counts.shape
        counts[row0:row0 + h, col0:col0 + w] += heatmap.counts

    return ObstacleHeatmap(
        counts=counts,
        origin_x=x_min,
        origin_y=y_min,
        cell_size_m=cell,
        z0=z0,
        height_min_m=min(heatmap.height_min_m for heatmap in heatmaps),
        height_max_m=max(heatmap.height_max_m for heatmap in heatmaps),
        metadata={
            "world_obstacle_point_count": int(counts.sum()),
            "source_heatmap_count": len(heatmaps),
            "grid_shape": [height, width],
            "cell_size_m": cell,
            "mask_mode": "direct_mask",
            "mask_source": "merged_wall_masks",
            "source_metadata": [heatmap.metadata for heatmap in heatmaps],
        },
    )
