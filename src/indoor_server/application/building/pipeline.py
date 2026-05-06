"""BuildPipeline — 단계 순차 실행 + 취소 체크."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine, Generator  # W-9: 최상단 import
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from indoor_server.infrastructure.ml.depth_anything import DepthAnythingV2Runner
    from indoor_server.infrastructure.ml.superpoint_lightglue import SuperPointLightGlueRunner

import numpy as np

from indoor_server.application.building.debug.sink import BuildDebugSink, NoopDebugSink
from indoor_server.application.building.steps.back_projection import BackProjectionStep
from indoor_server.application.building.steps.display_navigation_grid import (
    DisplayNavigationGridParams,
    DisplayNavigationGridStep,
)
from indoor_server.application.building.steps.floor_layout import (
    FloorLayoutFromGridStep,
    FloorLayoutStep,
)
from indoor_server.application.building.steps.floor_polygon import FloorPolygonSimplifyStep
from indoor_server.application.building.steps.floor_segmentation import (
    FloorSegmentationStep,
    KeyframeMasks,
    KeyframeRef,
)
from indoor_server.application.building.steps.node_placement import NodePlacementStep
from indoor_server.application.building.steps.poi_projection import POIProjectionStep
from indoor_server.application.building.steps.quality_gate import QualityGateStep
from indoor_server.application.building.steps.skeletonize import SkeletonGraph, SkeletonizeStep
from indoor_server.application.building.steps.walkable_grid import WalkableGridStep
from indoor_server.domain.building.enums import BuildFailureReason, BuildStep
from indoor_server.domain.building.models import BuildCounts, BuildOutcome, DepthMap, WalkableGrid
from indoor_server.domain.building.rtabmap_models import (
    RtabmapDataFrame,
    RtabmapFeaturePoint,
    RtabmapLink,
    RtabmapNode,
)
from indoor_server.domain.scan.models import POIMarkRow
from indoor_server.infrastructure.ml.protocol import SemanticSegmenter

logger = logging.getLogger(__name__)

ProgressSink = Callable[[BuildStep, float], Coroutine[object, object, None]]
CancelCheck = Callable[[], Coroutine[object, object, bool]]


class BuildPipeline:
    def __init__(
        self,
        segmenter: SemanticSegmenter,
        storage_root: Path,
        depth_runner: DepthAnythingV2Runner | None = None,
        use_depth_nn: bool = False,
        sp_lg_runner: SuperPointLightGlueRunner | None = None,
        use_multiview_scale: bool = False,
        multiview_window: int = 5,
        use_triangulation: bool = False,
        triangulation_window: int = 5,
        triangulation_floor_gate: bool = True,
        triangulation_min_score: float = 0.8,
        triangulation_max_matches: int = 64,
        use_trajectory_buffer: bool = False,
        trajectory_buffer_m: float = 0.8,
        use_adaptive_buffer: bool = False,
        adaptive_buffer_max_m: float = 5.0,
        adaptive_buffer_min_m: float = 0.3,
        adaptive_buffer_strip_fraction: float = 0.5,
        use_rtabmap_trajectory: bool = False,
        rtabmap_trajectory_half_width_m: float = 0.75,
        rtabmap_feature_evidence_enabled: bool = False,
        rtabmap_rectilinear_cover_enabled: bool = True,
        rtabmap_rectilinear_cover_rotated_grid_enabled: bool = True,
        rtabmap_depth_evidence_enabled: bool = True,
        rtabmap_depth_vertical_tolerance_m: float | None = 0.35,
        rtabmap_image_segmentation_enabled: bool = False,
        rtabmap_image_orientation_mode: str = "sensor",
        rtabmap_image_floor_mask_min_ratio: float = 0.0,
        rtabmap_image_wall_mask_max_ratio: float = 1.0,
        # Sprint 46: Floor segmentation point cloud source-of-truth
        use_floor_pointcloud: bool = False,
        floor_pointcloud_pixel_stride: int = 4,
        floor_pointcloud_height_tolerance_m: float = 0.30,
        floor_pointcloud_min_cell_hits: int = 2,
        # Sprint 47: CAD-style post-rectification cleanup (floor_pointcloud 전용)
        # Codex F-3: default OFF 유지 (>=2 real scan PASS 시까지).
        polygon_cad_cleanup_enabled: bool = False,
        floor_raster_cad_morph_close_cells: int = 3,
        manhattan_floor_pointcloud_max_area_change: float = 0.55,
        polygon_cad_collinear_angle_tol_deg: float = 5.0,
        polygon_cad_short_edge_min_length_m: float = 0.20,
        polygon_cad_near_vertex_merge_distance_m: float = 0.15,
        polygon_cad_orthogonality_angle_tol_deg: float = 5.0,
        # Sprint 48: dominant_angle hint chain
        dominant_angle_hint_enabled: bool = True,
        dominant_angle_hint_rtabmap_min_segments: int = 4,
        dominant_angle_hint_rtabmap_min_total_length_m: float = 3.0,
        dominant_angle_hint_rtabmap_min_best_bin_ratio: float = 0.15,
        dominant_angle_hint_obb_min_aspect_ratio: float = 1.3,
        dominant_angle_hint_cross_check_max_diff_deg: float = 15.0,
        # Sprint 49: hint chain 4-way trigger orthogonality threshold (Codex W-2)
        polygon_cad_hint_retry_orthogonality_threshold: float = 0.50,
        # Sprint 50: rectangle dictionary cover (Codex BLOCKER 4 — pipeline default OFF).
        # production config flip 은 evidence PASS 후 별도 sync 단계에서 수행.
        use_rectangle_dictionary_cover: bool = False,
        rectangle_cover_precision_threshold: float = 0.85,
        rectangle_cover_recall_min: float = 0.70,
        rectangle_cover_over_cover_max: float = 0.20,
        rectangle_cover_time_budget_sec: float = 30.0,
        rectangle_cover_candidate_stride_cells: int = 3,
        rectangle_cover_max_candidates_per_dimension: int = 200,
        # Sprint 50 v2: axes mode + dynamic precision threshold.
        # axes_mode "pair" — RTABMap link / OBB 에서 dominant axis 1개 추출 후
        #   (primary, primary+90°) pair 로만 sweep. multi-component split.
        # axes_mode "full18" — v1 그대로 18 angle full sweep.
        rectangle_cover_axes_mode: str = "pair",
        rectangle_cover_precision_threshold_dynamic: bool = True,
        rectangle_cover_precision_threshold_min: float = 0.65,
        rectangle_cover_min_component_cells: int = 50,
        rectangle_cover_axis_link_min_best_bin_ratio: float = 0.10,
        rectangle_cover_axis_obb_min_aspect_ratio: float = 1.2,
        # Sprint 51 — Wall-fitting polygon (default OFF, observer only).
        # 7-step pipeline (obstacle heatmap → density → components → line fit →
        # snap → merge → assembly → validate). production polygon 미변경, metadata
        # 만 dump. require: use_floor_pointcloud=True (floor mask + z0 의존).
        use_wall_polygon: bool = False,
        wall_polygon_density_min_cell_hits: int = 4,
        wall_polygon_density_morph_close_radius_cells: int = 1,
        wall_polygon_components_min_area_cells: int = 8,
        wall_polygon_line_min_linearity: float = 0.85,
        wall_polygon_line_min_length_m: float = 0.5,
        wall_polygon_snap_tolerance_deg: float = 15.0,
        wall_polygon_merge_offset_tolerance_m: float = 0.20,
        wall_polygon_merge_gap_fill_m: float = 1.0,
        wall_polygon_assembly_intersection_tolerance_m: float = 0.30,
        wall_polygon_assembly_use_alpha_shape: bool = True,
        wall_polygon_validate_floor_iou_min: float = 0.50,
        wall_polygon_validate_area_change_max_ratio: float = 0.40,
        wall_polygon_min_lines: int = 4,
        wall_polygon_max_lines: int = 20,
        wall_polygon_obstacle_height_min_m: float = 0.30,
        wall_polygon_obstacle_height_max_m: float = 2.50,
        # Sprint 61 — user-facing 2D display routing graph.
        # Worker config turns this on for production builds once a footprint is
        # available. Direct unit construction keeps legacy fallback unless
        # explicitly enabled.
        display_navigation_grid_enabled: bool = False,
        display_navigation_grid_cell_m: float = 0.45,
        display_navigation_grid_clearance_m: float = 0.30,
        display_navigation_grid_connectivity: int = 8,
        display_navigation_grid_poi_attach_k: int = 4,
    ) -> None:
        # geometry source mutex: depth_nn / multiview_scale / triangulation /
        # adaptive_buffer / rtabmap_trajectory / floor_pointcloud (Sprint 46)
        mode_count = sum(
            [
                use_depth_nn,
                use_multiview_scale,
                use_triangulation,
                use_adaptive_buffer,
                use_rtabmap_trajectory,
                use_floor_pointcloud,
            ]
        )
        if mode_count > 1:
            raise RuntimeError(
                "use_depth_nn / use_multiview_scale / use_triangulation / "
                "use_adaptive_buffer / use_rtabmap_trajectory / use_floor_pointcloud "
                f"are mutually exclusive (got {mode_count} active)"
            )

        # Sprint 46: floor_pointcloud 모드는 image segmentation을 강제로 켠다.
        # floor mask 없이는 source-of-truth가 사라지므로 자동 보정.
        floor_pointcloud_auto_enabled_image_seg = False
        if use_floor_pointcloud and not rtabmap_image_segmentation_enabled:
            rtabmap_image_segmentation_enabled = True
            floor_pointcloud_auto_enabled_image_seg = True
            logger.warning(
                "use_floor_pointcloud=True forced rtabmap_image_segmentation_enabled=True "
                "(floor mask is required for the point cloud source-of-truth)"
            )

        if use_trajectory_buffer and not use_triangulation:
            raise RuntimeError(
                "use_trajectory_buffer=True requires use_triangulation=True — "
                "trajectory buffer is only meaningful for the triangulation path"
            )
        # NOTE: use_adaptive_buffer + use_trajectory_buffer 조합은 위 4-way mutex (line 71)
        # 또는 traj-requires-triangulation (line 77) 이 먼저 발화하므로 dead branch.
        # Sprint 23 W-2: 제거됨. test_pipeline_mutex.py 참조.
        if use_trajectory_buffer and trajectory_buffer_m <= 0:
            raise RuntimeError(
                f"trajectory_buffer_m must be > 0, got {trajectory_buffer_m}"
            )
        if use_triangulation and sp_lg_runner is None:
            raise RuntimeError(
                "use_triangulation=True requires sp_lg_runner — "
                "pass a SuperPointLightGlueRunner instance"
            )
        if use_depth_nn and depth_runner is None:
            raise RuntimeError(
                "use_depth_nn=True requires depth_runner — "
                "pass a DepthAnythingV2Runner instance or set use_depth_nn=False"
            )
        if use_multiview_scale and (not use_depth_nn or sp_lg_runner is None):
            raise RuntimeError(
                "use_multiview_scale=True requires use_depth_nn=True + sp_lg_runner"
            )
        if use_adaptive_buffer:
            if adaptive_buffer_max_m <= 0 or adaptive_buffer_min_m < 0:
                raise RuntimeError("adaptive_buffer_min/max_m must be positive")
            if adaptive_buffer_min_m >= adaptive_buffer_max_m:
                raise RuntimeError("adaptive_buffer_min_m must be < adaptive_buffer_max_m")
            if not (0.0 < adaptive_buffer_strip_fraction <= 1.0):
                raise RuntimeError("adaptive_buffer_strip_fraction must be in (0, 1]")
        if use_rtabmap_trajectory and rtabmap_trajectory_half_width_m <= 0:
            raise RuntimeError("rtabmap_trajectory_half_width_m must be > 0")

        self._segmenter = segmenter
        self._storage_root = storage_root
        self._depth_runner = depth_runner
        self._use_depth_nn = use_depth_nn
        self._sp_lg_runner = sp_lg_runner
        self._use_multiview_scale = use_multiview_scale
        self._multiview_window = multiview_window
        self._use_triangulation = use_triangulation
        self._triangulation_window = triangulation_window
        self._triangulation_floor_gate = triangulation_floor_gate
        self._triangulation_min_score = triangulation_min_score
        self._triangulation_max_matches = triangulation_max_matches
        self._use_trajectory_buffer = use_trajectory_buffer
        self._trajectory_buffer_m = trajectory_buffer_m
        self._use_adaptive_buffer = use_adaptive_buffer
        self._adaptive_buffer_max_m = adaptive_buffer_max_m
        self._adaptive_buffer_min_m = adaptive_buffer_min_m
        self._adaptive_buffer_strip_fraction = adaptive_buffer_strip_fraction
        self._use_rtabmap_trajectory = use_rtabmap_trajectory
        self._rtabmap_trajectory_half_width_m = rtabmap_trajectory_half_width_m
        self._rtabmap_feature_evidence_enabled = rtabmap_feature_evidence_enabled
        self._rtabmap_rectilinear_cover_enabled = rtabmap_rectilinear_cover_enabled
        self._rtabmap_rectilinear_cover_rotated_grid_enabled = (
            rtabmap_rectilinear_cover_rotated_grid_enabled
        )
        self._rtabmap_depth_evidence_enabled = rtabmap_depth_evidence_enabled
        self._rtabmap_depth_vertical_tolerance_m = rtabmap_depth_vertical_tolerance_m
        self._rtabmap_image_segmentation_enabled = rtabmap_image_segmentation_enabled
        self._rtabmap_image_orientation_mode = rtabmap_image_orientation_mode
        self._rtabmap_image_floor_mask_min_ratio = rtabmap_image_floor_mask_min_ratio
        self._rtabmap_image_wall_mask_max_ratio = rtabmap_image_wall_mask_max_ratio
        # Sprint 46: floor pointcloud
        self._use_floor_pointcloud = use_floor_pointcloud
        self._floor_pointcloud_pixel_stride = floor_pointcloud_pixel_stride
        self._floor_pointcloud_height_tolerance_m = (
            floor_pointcloud_height_tolerance_m
        )
        self._floor_pointcloud_min_cell_hits = floor_pointcloud_min_cell_hits
        self._floor_pointcloud_auto_enabled_image_seg = (
            floor_pointcloud_auto_enabled_image_seg
        )
        # Sprint 47: CAD cleanup config
        self._polygon_cad_cleanup_enabled = polygon_cad_cleanup_enabled
        self._floor_raster_cad_morph_close_cells = floor_raster_cad_morph_close_cells
        self._manhattan_floor_pointcloud_max_area_change = (
            manhattan_floor_pointcloud_max_area_change
        )
        self._polygon_cad_collinear_angle_tol_deg = polygon_cad_collinear_angle_tol_deg
        self._polygon_cad_short_edge_min_length_m = polygon_cad_short_edge_min_length_m
        self._polygon_cad_near_vertex_merge_distance_m = (
            polygon_cad_near_vertex_merge_distance_m
        )
        self._polygon_cad_orthogonality_angle_tol_deg = (
            polygon_cad_orthogonality_angle_tol_deg
        )
        # Sprint 48
        self._dominant_angle_hint_enabled = dominant_angle_hint_enabled
        self._dominant_angle_hint_rtabmap_min_segments = (
            dominant_angle_hint_rtabmap_min_segments
        )
        self._dominant_angle_hint_rtabmap_min_total_length_m = (
            dominant_angle_hint_rtabmap_min_total_length_m
        )
        self._dominant_angle_hint_rtabmap_min_best_bin_ratio = (
            dominant_angle_hint_rtabmap_min_best_bin_ratio
        )
        self._dominant_angle_hint_obb_min_aspect_ratio = (
            dominant_angle_hint_obb_min_aspect_ratio
        )
        self._dominant_angle_hint_cross_check_max_diff_deg = (
            dominant_angle_hint_cross_check_max_diff_deg
        )
        # Sprint 49 (Codex W-2)
        self._polygon_cad_hint_retry_orthogonality_threshold = (
            polygon_cad_hint_retry_orthogonality_threshold
        )
        # Sprint 50 — rectangle dictionary cover. Codex BLOCKER 4: production
        # default OFF. CLI/eval path 에서만 ON (run_real_scan --rectangle-cover).
        self._use_rectangle_dictionary_cover = use_rectangle_dictionary_cover
        self._rectangle_cover_precision_threshold = (
            rectangle_cover_precision_threshold
        )
        self._rectangle_cover_recall_min = rectangle_cover_recall_min
        self._rectangle_cover_over_cover_max = rectangle_cover_over_cover_max
        self._rectangle_cover_time_budget_sec = rectangle_cover_time_budget_sec
        self._rectangle_cover_candidate_stride_cells = (
            rectangle_cover_candidate_stride_cells
        )
        self._rectangle_cover_max_candidates_per_dimension = (
            rectangle_cover_max_candidates_per_dimension
        )
        # Sprint 50 v2
        if rectangle_cover_axes_mode not in ("pair", "full18"):
            raise RuntimeError(
                "rectangle_cover_axes_mode must be 'pair' or 'full18', "
                f"got {rectangle_cover_axes_mode!r}"
            )
        self._rectangle_cover_axes_mode = rectangle_cover_axes_mode
        self._rectangle_cover_precision_threshold_dynamic = (
            rectangle_cover_precision_threshold_dynamic
        )
        self._rectangle_cover_precision_threshold_min = (
            rectangle_cover_precision_threshold_min
        )
        self._rectangle_cover_min_component_cells = (
            rectangle_cover_min_component_cells
        )
        self._rectangle_cover_axis_link_min_best_bin_ratio = (
            rectangle_cover_axis_link_min_best_bin_ratio
        )
        self._rectangle_cover_axis_obb_min_aspect_ratio = (
            rectangle_cover_axis_obb_min_aspect_ratio
        )
        # Sprint 51 — Wall-fitting polygon (observer, default OFF).
        if use_wall_polygon and not use_floor_pointcloud:
            raise RuntimeError(
                "use_wall_polygon=True requires use_floor_pointcloud=True — "
                "wall_polygon piggy-backs on floor mask + z0 from the floor "
                "pointcloud step"
            )
        self._use_wall_polygon = use_wall_polygon
        self._wall_polygon_density_min_cell_hits = wall_polygon_density_min_cell_hits
        self._wall_polygon_density_morph_close_radius_cells = (
            wall_polygon_density_morph_close_radius_cells
        )
        self._wall_polygon_components_min_area_cells = (
            wall_polygon_components_min_area_cells
        )
        self._wall_polygon_line_min_linearity = wall_polygon_line_min_linearity
        self._wall_polygon_line_min_length_m = wall_polygon_line_min_length_m
        self._wall_polygon_snap_tolerance_deg = wall_polygon_snap_tolerance_deg
        self._wall_polygon_merge_offset_tolerance_m = (
            wall_polygon_merge_offset_tolerance_m
        )
        self._wall_polygon_merge_gap_fill_m = wall_polygon_merge_gap_fill_m
        self._wall_polygon_assembly_intersection_tolerance_m = (
            wall_polygon_assembly_intersection_tolerance_m
        )
        self._wall_polygon_assembly_use_alpha_shape = (
            wall_polygon_assembly_use_alpha_shape
        )
        self._wall_polygon_validate_floor_iou_min = wall_polygon_validate_floor_iou_min
        self._wall_polygon_validate_area_change_max_ratio = (
            wall_polygon_validate_area_change_max_ratio
        )
        self._wall_polygon_min_lines = wall_polygon_min_lines
        self._wall_polygon_max_lines = wall_polygon_max_lines
        self._wall_polygon_obstacle_height_min_m = wall_polygon_obstacle_height_min_m
        self._wall_polygon_obstacle_height_max_m = wall_polygon_obstacle_height_max_m
        self._display_navigation_grid_enabled = display_navigation_grid_enabled
        self._display_navigation_grid_params = DisplayNavigationGridParams(
            cell_m=display_navigation_grid_cell_m,
            clearance_m=display_navigation_grid_clearance_m,
            connectivity=display_navigation_grid_connectivity,
            poi_attach_k=display_navigation_grid_poi_attach_k,
        )

    async def execute(
        self,
        *,
        scan_id: UUID,
        build_job_id: UUID,
        keyframes: list[KeyframeRef],
        pois: list[POIMarkRow],
        rtabmap_nodes: list[RtabmapNode] | None = None,
        rtabmap_links: list[RtabmapLink] | None = None,
        rtabmap_features: list[RtabmapFeaturePoint] | None = None,
        rtabmap_frames: list[RtabmapDataFrame] | None = None,
        scan_manifest_metadata: dict[str, object] | None = None,
        progress_sink: ProgressSink,
        cancel_check: CancelCheck,
        debug_sink: BuildDebugSink | None = None,
    ) -> BuildOutcome:
        """
        FLOOR_SEG → BACK_PROJECT → WALKABLE_GRID → SKELETON →
        NODE_PLACEMENT → POI_PROJECTION → QUALITY_GATE → PERSIST(호출자 담당)
        단계 경계에서 cancel_check() 호출.
        """
        sink: BuildDebugSink = debug_sink if debug_sink is not None else NoopDebugSink()

        all_masks: list[KeyframeMasks] = []
        tz_values: list[float] = []
        keyframes_processed = len(keyframes)
        depth_maps_collected: list[DepthMap] = []
        footprint_geojson: dict[str, object] | None = None
        rtabmap_trajectory: dict[str, object] | None = None
        floor_pointcloud_meta: dict[str, object] | None = None
        floor_raster_meta: dict[str, object] | None = None
        # Sprint 47: floor_pointcloud 모드에서 eager rectification + CAD cleanup
        # 결과를 보존하여 2nd rectification 호출을 skip 한다.
        floor_pc_rectification_meta: dict[str, object] | None = None
        floor_pc_cad_cleanup_meta: dict[str, object] | None = None
        floor_pc_rectified_pre_cleanup: dict[str, object] | None = None
        already_rectified_floor_pc = False

        floor_pc_hint_meta: dict[str, object] | None = None
        floor_pc_cad_effect_pass: bool | None = None
        floor_pc_cad_effect_small_polygon_pass: bool | None = None
        wall_polygon_meta: dict[str, object] | None = None
        if self._use_floor_pointcloud:
            (
                z0,
                grid,
                walkable_cells,
                footprint_geojson,
                floor_pointcloud_meta,
                floor_raster_meta,
                floor_pc_rectification_meta,
                floor_pc_cad_cleanup_meta,
                floor_pc_rectified_pre_cleanup,
                already_rectified_floor_pc,
                floor_pc_hint_meta,
                floor_pc_cad_effect_pass,
                floor_pc_cad_effect_small_polygon_pass,
                wall_polygon_meta,
            ) = await self._run_floor_pointcloud_and_grid(
                rtabmap_nodes or [],
                rtabmap_links or [],
                rtabmap_frames or [],
                progress_sink,
                cancel_check,
            )
        elif self._use_rtabmap_trajectory:  # noqa: F811 (handled by Sprint 46 outer if)
            await progress_sink(BuildStep.FLOOR_SEG, 0.20)
            z0, grid, walkable_cells, footprint_geojson, rtabmap_trajectory = (
                await self._run_rtabmap_trajectory_and_grid(
                    rtabmap_nodes or [],
                    rtabmap_links or [],
                    rtabmap_features or [],
                    rtabmap_frames or [],
                    progress_sink,
                    cancel_check,
                )
            )
        else:
            # ── FLOOR_SEG ─────────────────────────────────────────────────────
            all_masks, tz_values = await self._run_floor_seg(
                keyframes, progress_sink, cancel_check
            )
            keyframes_processed = len(all_masks)

            try:
                sink.on_floor_segmentation(all_masks)
            except Exception as e:
                logger.warning("debug sink on_floor_segmentation failed: %s", e)

            # ── FLOOR_POLYGON_SIMPLIFY + FLOOR_LAYOUT_UNION (sink 활성일 때만) ──
            if not isinstance(sink, NoopDebugSink):
                poly_step = FloorPolygonSimplifyStep()
                polygons = await asyncio.to_thread(poly_step.run, all_masks)
                try:
                    sink.on_floor_polygons(polygons)
                except Exception as e:
                    logger.warning("debug sink on_floor_polygons failed: %s", e)

                z0_for_layout = WalkableGridStep().estimate_z0(tz_values)
                layout_step = FloorLayoutStep()
                layout = await asyncio.to_thread(layout_step.run, polygons, z0_for_layout)
                trajectory_pts = [(m.tx, m.ty) for m in all_masks]
                try:
                    sink.on_floor_layout(layout, trajectory_pts)
                except Exception as e:
                    logger.warning("debug sink on_floor_layout failed: %s", e)

                if await cancel_check():
                    return self._cancelled_outcome(keyframes_processed)

            if await cancel_check():
                return self._cancelled_outcome(keyframes_processed)

            # ── BACK_PROJECT + WALKABLE_GRID ──────────────────────────────────
            # 우선순위: adaptive_buffer > triangulation > depth_nn > baseline
            if self._use_adaptive_buffer:
                z0, grid, walkable_cells, footprint_geojson = (
                    await self._run_adaptive_buffer_and_grid(
                        all_masks, tz_values, progress_sink, cancel_check
                    )
                )
            elif self._use_triangulation and self._sp_lg_runner is not None:
                z0, grid, walkable_cells = await self._run_triangulation_and_grid(
                    all_masks, tz_values, progress_sink, cancel_check
                )
            elif self._use_depth_nn and self._depth_runner is not None:
                z0, grid, walkable_cells, depth_maps_collected = (
                    await self._run_depth_aware_back_project_and_grid(
                        all_masks, tz_values, progress_sink, cancel_check
                    )
                )
            else:
                z0, grid, walkable_cells = await self._run_back_project_and_grid(
                    all_masks, tz_values, progress_sink, cancel_check
                )

        try:
            sink.on_walkable_grid(grid)
        except Exception as e:
            logger.warning("debug sink on_walkable_grid failed: %s", e)

        # ── depth-aware 시각화 (depth_nn 경로 + sink 활성일 때만) ──────────────
        if self._use_depth_nn and depth_maps_collected and not isinstance(sink, NoopDebugSink):
            try:
                sink.on_depth_maps(depth_maps_collected)
            except Exception as e:
                logger.warning("debug sink on_depth_maps failed: %s", e)

            try:
                await self._emit_depth_aware_layout(
                    sink, grid, z0, all_masks
                )
            except Exception as e:
                logger.warning("debug sink on_depth_aware_layout failed: %s", e)

        # ── [신규] Fix 3: raster-fit FloorLayout (walkable_grid 기반, sink 활성일 때만) ──
        if not isinstance(sink, NoopDebugSink):
            trajectory_pts_for_raster = [(m.tx, m.ty) for m in all_masks]
            raster_step = FloorLayoutFromGridStep()
            raster_layout = await asyncio.to_thread(raster_step.run, grid, z0)
            try:
                sink.on_floor_layout_raster(raster_layout, trajectory_pts_for_raster)
            except Exception as e:
                logger.warning("debug sink on_floor_layout_raster failed: %s", e)

        if await cancel_check():
            return self._cancelled_outcome(keyframes_processed, walkable_cells)

        # ── SKELETON ──────────────────────────────────────────────────────────
        await progress_sink(BuildStep.SKELETON, 0.50)
        skel, skel_mask = await asyncio.to_thread(self._run_skeletonize, SkeletonizeStep(), grid)
        skel_px = skel.skeleton_pixel_count
        logger.info("skeleton done pixels=%d", skel_px)

        try:
            sink.on_skeleton(skel_mask, skel)
        except Exception as e:
            logger.warning("debug sink on_skeleton failed: %s", e)

        if await cancel_check():
            return self._cancelled_outcome(keyframes_processed, walkable_cells, skel_px)

        # ── [Sprint 34 Cycle 9] MANHATTAN RECTIFICATION (NODE_PLACEMENT 전으로 이동) ──
        # footprint_geojson이 있을 때만 실행 (adaptive_buffer 경로).
        # dominant_angle_deg와 rectified polygon을 NodePlacementStep에 전달하여
        # graph가 footprint와 동일한 회전축을 사용하도록 통일.
        # Sprint 47 W-6: floor_pointcloud 모드는 _run_floor_pointcloud_and_grid 안에서
        # 이미 rectification + CAD cleanup 적용했으므로 2nd 호출을 skip 한다.
        rectified_footprint_geojson: dict[str, object] | None = None
        rectification: dict[str, object] | None = None
        rectified_footprint_polygon: object | None = None  # shapely geometry
        footprint_dominant_angle: float | None = None

        if already_rectified_floor_pc:
            # floor_pointcloud 경로: eager 결과 재사용
            rectification = floor_pc_rectification_meta
            # NODE_PLACEMENT 직전 footprint는 cleanup 후 polygon 그대로 (footprint_geojson).
            rectified_footprint_geojson = footprint_geojson
            if rectification is not None:
                angle_val = rectification.get("dominant_angle_deg")
                if isinstance(angle_val, (int, float)):
                    footprint_dominant_angle = float(angle_val)
            if footprint_geojson is not None:
                try:
                    from shapely.geometry import shape as _shape

                    rectified_footprint_polygon = _shape(footprint_geojson)
                    if not hasattr(rectified_footprint_polygon, "exterior"):
                        from shapely.ops import unary_union as _unary_union

                        rectified_footprint_polygon = _unary_union(
                            rectified_footprint_polygon.geoms  # type: ignore[union-attr]
                        ).convex_hull
                except Exception as _e:
                    logger.warning(
                        "floor_pc rectified footprint polygon parse failed: %s", _e
                    )
                    rectified_footprint_polygon = None
            logger.info(
                "skip 2nd manhattan_rectification (already rectified by floor_pointcloud)"
            )
        elif footprint_geojson is not None:
            from indoor_server.application.building.steps.manhattan_rectification import (
                ManhattanRectificationStep,
            )

            rectified = await asyncio.to_thread(
                ManhattanRectificationStep().run,
                footprint_geojson,
            )
            rectified_footprint_geojson = rectified.rectified_geojson
            rectification = rectified.metadata()
            footprint_dominant_angle = rectified.dominant_angle_deg
            # rectified polygon — NodePlacementStep footprint snap용
            try:
                from shapely.geometry import shape as _shape

                rectified_footprint_polygon = _shape(rectified.rectified_geojson)
                # MultiPolygon의 경우 exterior를 직접 쓸 수 없으므로
                # convex_hull 또는 unary_union.convex_hull fallback
                if not hasattr(rectified_footprint_polygon, "exterior"):
                    from shapely.ops import unary_union as _unary_union

                    rectified_footprint_polygon = _unary_union(
                        rectified_footprint_polygon.geoms  # type: ignore[union-attr]
                    ).convex_hull
            except Exception as _e:
                logger.warning("footprint polygon parse failed: %s", _e)
                rectified_footprint_polygon = None
            logger.info(
                "manhattan_rectification done dominant_angle=%.2f accepted=%s area_change=%.3f",
                footprint_dominant_angle,
                rectified.accepted,
                rectified.area_change_ratio,
            )

        # Sprint 49 (Codex BLOCKER 1/2): ARKit↔RTABMap rigid SE(3) transform.
        # keyframe_meta.rtabmap_node_id 가 채워진 row 의 (tx, ty, tz) 와
        # RTABMap node pose translation 사이의 Kabsch SVD.
        from indoor_server.application.building.arkit_to_rtabmap_transform import (
            EstimateParams,
            TransformInputPair,
            estimate_arkit_to_rtabmap_transform,
        )

        # rtabmap_node_id 별 ARKit pose
        kf_pose_by_node_id: dict[int, tuple[float, float, float]] = {}
        for kf in keyframes:
            if kf.rtabmap_node_id is None:
                continue
            kf_pose_by_node_id.setdefault(
                int(kf.rtabmap_node_id),
                (float(kf.tx), float(kf.ty), float(kf.tz)),
            )

        # rtabmap nodes 에서 (node_id → translation)
        node_pose_by_id: dict[int, tuple[float, float, float]] = {}
        for node in rtabmap_nodes or []:
            # pose 는 4x4 matrix (row-major). translation = pose[i][3], i=0..2.
            try:
                tx = float(node.pose[0][3])
                ty = float(node.pose[1][3])
                tz = float(node.pose[2][3])
            except (IndexError, TypeError):
                continue
            node_pose_by_id[int(node.node_id)] = (tx, ty, tz)

        transform_pairs: list[TransformInputPair] = []
        for node_id, arkit_xyz in kf_pose_by_node_id.items():
            rtab_xyz = node_pose_by_id.get(node_id)
            if rtab_xyz is None:
                continue
            transform_pairs.append(
                TransformInputPair(
                    arkit_xyz=arkit_xyz,
                    rtabmap_xyz=rtab_xyz,
                )
            )

        transform_estimate = estimate_arkit_to_rtabmap_transform(
            transform_pairs,
            params=EstimateParams(),
        )
        logger.info(
            "arkit_to_rtabmap_transform pair_count=%d confidence=%s rms=%.4fm "
            "planar=%s reason=%s",
            transform_estimate.transform.pair_count,
            transform_estimate.transform.confidence,
            transform_estimate.transform.residual_rms_m
            if not (
                transform_estimate.transform.residual_rms_m
                != transform_estimate.transform.residual_rms_m  # NaN check
            )
            else float("nan"),
            transform_estimate.transform.planar_mode,
            transform_estimate.transform.fallback_reason,
        )

        # ── NODE_PLACEMENT ────────────────────────────────────────────────────
        await progress_sink(BuildStep.NODE_PLACEMENT, 0.65)
        original_node_count_pre_rect = len(skel.nodes)
        display_navigation_grid_meta: dict[str, object] | None = None
        use_display_navigation_grid = (
            self._display_navigation_grid_enabled
            and (rectified_footprint_geojson is not None or footprint_geojson is not None)
        )
        if use_display_navigation_grid:
            navigation_footprint = rectified_footprint_geojson or footprint_geojson
            assert navigation_footprint is not None
            display_step = DisplayNavigationGridStep(
                scan_id=scan_id,
                build_job_id=build_job_id,
                params=self._display_navigation_grid_params,
                arkit_to_rtabmap_transform=transform_estimate.transform,
            )
            display_result = await asyncio.to_thread(
                display_step.run,
                footprint_geojson=navigation_footprint,
                floor_z=z0,
                pois=pois,
            )
            if rectified_footprint_geojson is not None:
                rectified_footprint_geojson = display_result.footprint_geojson
            else:
                footprint_geojson = display_result.footprint_geojson
            nodes, edges = display_result.nodes, display_result.edges
            poi_world_poses = display_result.poi_world_poses
            display_navigation_grid_meta = display_result.metadata
            display_navigation_grid_meta["used"] = True
            logger.info(
                "display_navigation_grid done nodes=%d edges=%d pois=%d "
                "cell=%.2f clearance=%.2f",
                len(nodes),
                len(edges),
                len(poi_world_poses),
                self._display_navigation_grid_params.cell_m,
                self._display_navigation_grid_params.clearance_m,
            )

            # Sprint 49: poi_frame_transform metadata + per-POI position source.
            poi_frame_transform_meta: dict[str, object] = (
                transform_estimate.transform.to_metadata()
            )
            poi_frame_transform_meta["per_poi"] = (
                display_result.poi_position_metadata
            )
        else:
            placer = NodePlacementStep(
                scan_id=scan_id,
                build_job_id=build_job_id,
                force_rectilinear=True,
                dominant_angle_deg=footprint_dominant_angle,
                footprint_polygon=rectified_footprint_polygon,
            )
            nodes, edges = await asyncio.to_thread(placer.run, skel, grid.origin)
            logger.info(
                "node_placement done nodes=%d edges=%d "
                "(rectilinear=True dominant=%.2f footprint_snap=%s)",
                len(nodes),
                len(edges),
                footprint_dominant_angle
                if footprint_dominant_angle is not None
                else 0.0,
                rectified_footprint_polygon is not None,
            )

            # ── POI_PROJECTION ────────────────────────────────────────────────
            await progress_sink(BuildStep.POI_PROJECTION, 0.80)
            projector = POIProjectionStep(
                scan_id=scan_id,
                build_job_id=build_job_id,
                arkit_to_rtabmap_transform=transform_estimate.transform,
            )
            nodes, edges, poi_world_poses = await asyncio.to_thread(
                projector.run, pois, nodes, edges
            )
            logger.info("poi_projection done pois=%d", len(poi_world_poses))

            # Sprint 49: poi_frame_transform metadata + per-POI position source.
            poi_frame_transform_meta = transform_estimate.transform.to_metadata()
            poi_frame_transform_meta["per_poi"] = projector.poi_position_metadata
            display_navigation_grid_meta = {
                "graph_source": "display_navigation_grid",
                "used": False,
                "reason": "disabled_or_missing_footprint",
            }

        try:
            sink.on_node_placement(nodes, edges, grid.origin)
        except Exception as e:
            logger.warning("debug sink on_node_placement failed: %s", e)

        if await cancel_check():
            return self._cancelled_outcome(keyframes_processed, walkable_cells, skel_px)

        # ── QUALITY_GATE ──────────────────────────────────────────────────────
        await progress_sink(BuildStep.QUALITY_GATE, 0.90)
        from indoor_server.config import settings as _settings

        gate_step = QualityGateStep(
            min_coverage=_settings.quality_gate_min_coverage,
            max_components=_settings.quality_gate_max_components,
        )
        report = await asyncio.to_thread(gate_step.evaluate, grid, nodes, edges)

        from indoor_server.application.building.steps.node_placement import _get_dominant_angle

        counts = BuildCounts(
            keyframes_processed=keyframes_processed,
            walkable_cells=walkable_cells,
            skeleton_pixels=skel_px,
            map_nodes=len(nodes),
            map_edges=len(edges),
            pois_projected=len(poi_world_poses),
            walkable_coverage=report.walkable_coverage,
            connected_components=report.connected_components,
            floor_z0=z0,  # W-6: 실제 z0 값 저장
            footprint_geojson=footprint_geojson,  # Sprint 24: IMDF export용
            rectified_footprint_geojson=rectified_footprint_geojson,
            rectification=rectification,
            # Sprint 34 Cycle 9: footprint dominant_angle 주입 후 rectilinear graph
            graph_dominant_angle_deg=_get_dominant_angle(footprint_dominant_angle),
            graph_forced_rectilinear=True,
            graph_original_node_count=original_node_count_pre_rect,
            graph_rectified_node_count=len(nodes),
            build_source=_resolve_build_source(
                use_floor_pointcloud=self._use_floor_pointcloud,
                use_rtabmap_trajectory=self._use_rtabmap_trajectory,
            ),
            rtabmap_trajectory=rtabmap_trajectory,
            floor_pointcloud=floor_pointcloud_meta,
            floor_raster=floor_raster_meta,
            polygon_cad_cleanup=floor_pc_cad_cleanup_meta,
            dominant_angle_hint=floor_pc_hint_meta,
            cad_effect_pass=floor_pc_cad_effect_pass,
            cad_effect_small_polygon_pass=floor_pc_cad_effect_small_polygon_pass,
            # Sprint 49 (Codex BLOCKER 1/2/5/6)
            poi_frame_transform=poi_frame_transform_meta,
            scan_manifest=scan_manifest_metadata,
            # Sprint 50 (Codex BLOCKER 1~5): rectangle dictionary cover.
            rectangle_cover=(
                floor_raster_meta.get("rectangle_cover")
                if isinstance(floor_raster_meta, dict)
                else None
            ),
            # Sprint 51: wall-fitting polygon observer metadata (default OFF).
            wall_polygon=wall_polygon_meta,
            # Sprint 61: user-facing 2D display graph + POI/facility attach.
            display_navigation_grid=display_navigation_grid_meta,
        )

        if not report.passed:
            return BuildOutcome(
                nodes=nodes,
                edges=edges,
                poi_world_poses=poi_world_poses,
                counts=counts,
                passed_quality_gate=False,
                failure_reason=report.failure_reason,
            )

        return BuildOutcome(
            nodes=nodes,
            edges=edges,
            poi_world_poses=poi_world_poses,
            counts=counts,
            passed_quality_gate=True,
        )

    # ── private step runners ──────────────────────────────────────────────────

    async def _run_floor_seg(
        self,
        keyframes: list[KeyframeRef],
        progress_sink: ProgressSink,
        cancel_check: CancelCheck,
    ) -> tuple[list[KeyframeMasks], list[float]]:
        """FLOOR_SEG 단계 실행. (all_masks, tz_values) 반환."""
        await progress_sink(BuildStep.FLOOR_SEG, 0.0)
        seg_step = FloorSegmentationStep(self._segmenter, self._storage_root)
        all_masks: list[KeyframeMasks] = []
        tz_values: list[float] = []

        seg_iter = await seg_step.run(
            keyframes,
            progress=lambda p: progress_sink(BuildStep.FLOOR_SEG, p * 0.25),
        )
        async for mask in seg_iter:
            all_masks.append(mask)
            tz_values.append(mask.tz)

        logger.info("floor_seg done keyframes=%d", len(all_masks))
        return all_masks, tz_values

    async def _run_floor_pointcloud_and_grid(
        self,
        nodes: list[RtabmapNode],
        links: list[RtabmapLink],
        frames: list[RtabmapDataFrame],
        progress_sink: ProgressSink,
        cancel_check: CancelCheck,
    ) -> tuple[
        float,
        WalkableGrid,
        int,
        dict[str, object] | None,
        dict[str, object],
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
        dict[str, object] | None,
        bool,
        dict[str, object] | None,
        bool | None,
        bool | None,
        dict[str, object] | None,
    ]:
        """Sprint 46: floor segmentation point cloud → WalkableGrid + footprint.

        Sprint 47: rectification + CAD cleanup eager 적용 (W-6 idempotent skip).

        반환:
            (z0, grid, walkable_cells, footprint_geojson_for_node_placement,
             pointcloud_meta, raster_meta,
             rectification_meta, cad_cleanup_meta, rectified_footprint_geojson_pre_cleanup,
             already_rectified)
        """
        from indoor_server.application.building.steps.floor_point_cloud import (
            FloorPointCloudStep,
            FloorPointCloudStepParams,
        )
        from indoor_server.application.building.steps.floor_raster import (
            FloorRasterStep,
            FloorRasterStepParams,
        )
        from indoor_server.application.building.steps.rtabmap_image_evidence import (
            RtabmapImageEvidenceStep,
        )

        await progress_sink(BuildStep.FLOOR_SEG, 0.10)

        empty_grid = self._empty_walkable_grid(0.0)

        if not nodes or not frames:
            empty_pc_meta: dict[str, object] = {
                "build_source": "floor_pointcloud",
                "issues": ["missing_rtabmap_nodes_or_frames"],
                "node_count": len(nodes),
                "frame_count": len(frames),
                "auto_enabled_image_segmentation": (
                    self._floor_pointcloud_auto_enabled_image_seg
                ),
            }
            empty_raster_meta: dict[str, object] = {
                "build_source": "floor_pointcloud",
                "empty_reason": "missing_rtabmap_nodes_or_frames",
            }
            return (
                0.0,
                empty_grid,
                0,
                None,
                empty_pc_meta,
                empty_raster_meta,
                None,
                None,
                None,
                False,
                None,
                None,
                None,
                None,
            )

        node_pose_ids = {node.node_id for node in nodes}
        image_evidence = await RtabmapImageEvidenceStep().run(
            frames=frames,
            segmenter=self._segmenter,
            node_pose_ids=node_pose_ids,
            orientation_mode=self._rtabmap_image_orientation_mode,  # type: ignore[arg-type]
            floor_mask_min_ratio=self._rtabmap_image_floor_mask_min_ratio,
            wall_mask_max_ratio=self._rtabmap_image_wall_mask_max_ratio,
        )
        floor_masks_by_node_id = image_evidence.floor_masks_by_node_id
        wall_masks_by_node_id = image_evidence.wall_masks_by_node_id

        if await cancel_check():
            cancelled_meta: dict[str, object] = {
                "build_source": "floor_pointcloud",
                "issues": ["cancelled_after_floor_mask"],
                "image_evidence": image_evidence.metadata(),
                "auto_enabled_image_segmentation": (
                    self._floor_pointcloud_auto_enabled_image_seg
                ),
            }
            cancelled_raster_meta: dict[str, object] = {
                "build_source": "floor_pointcloud",
                "empty_reason": "cancelled_after_floor_mask",
            }
            return (
                0.0,
                empty_grid,
                0,
                None,
                cancelled_meta,
                cancelled_raster_meta,
                None,
                None,
                None,
                False,
                None,
                None,
                None,
                None,
            )

        await progress_sink(BuildStep.BACK_PROJECT, 0.25)

        pc_step = FloorPointCloudStep(
            FloorPointCloudStepParams(
                pixel_stride=self._floor_pointcloud_pixel_stride,
                height_tolerance_m=self._floor_pointcloud_height_tolerance_m,
            )
        )
        cloud = await asyncio.to_thread(
            pc_step.run,
            nodes=nodes,
            frames=frames,
            floor_masks_by_node_id=floor_masks_by_node_id,
        )

        await progress_sink(BuildStep.WALKABLE_GRID, 0.35)

        # Sprint 47 W-5: floor_raster default 변경 대신 호출부에서 명시 주입.
        # 다른 호출자 (FloorRasterStep() default) 영향 0.
        # Sprint 49 hotfix: ㄴ자/T자 corner 끊김 대응. keep_largest_component=False
        # + min_component_area_m2=3.0 으로 작은 noise 만 drop, 본 hall 보존.
        # morph_close 도 5로 키워 corner 부근 floor pixel 부족을 closing 으로 메움.
        raster_step = FloorRasterStep(
            FloorRasterStepParams(
                min_cell_hits=self._floor_pointcloud_min_cell_hits,
                morph_close_radius_cells=max(
                    5, self._floor_raster_cad_morph_close_cells
                ),
                keep_largest_component=False,
                min_component_area_m2=3.0,
            )
        )
        raster = await asyncio.to_thread(raster_step.run, cloud)

        await progress_sink(BuildStep.WALKABLE_GRID, 0.40)

        walkable_cells = int(raster.grid.mask.sum())
        pointcloud_meta = {
            "build_source": "floor_pointcloud",
            "image_evidence": image_evidence.metadata(),
            "point_cloud": cloud.metadata,
            "auto_enabled_image_segmentation": (
                self._floor_pointcloud_auto_enabled_image_seg
            ),
        }
        raster_meta = {
            "build_source": "floor_pointcloud",
            **raster.metadata,
        }
        logger.info(
            "floor_pointcloud done points=%d walkable_cells=%d z0=%.3f",
            int(cloud.points_xy.shape[0]),
            walkable_cells,
            cloud.z0,
        )

        # ── Sprint 50 (Codex BLOCKER 1~5) + v2 (axis pair + multi-component) ──
        # opt-in (production OFF). raster.grid 의 obs heatmap 을 rotation +
        # integral image + greedy 로 cover 시도. accepted=True 이면 footprint
        # 자체를 직사각형 union 으로 교체하고 hint chain / Manhattan / CAD
        # cleanup 모두 skip. accepted=False 이면 metadata 만 남기고 fallback
        # → Sprint 49 hint chain path 그대로 진행 (BLOCKER 5).
        # v2: axes_mode="pair" 시 RTABMap link / OBB 에서 dominant axis 추출 후
        # (primary, primary+90°) pair 로만 sweep. heatmap 을 connected component
        # 로 분리, 각 component 별 axis pair 추정 후 cover 결과 union.
        rectangle_cover_meta: dict[str, object] | None = None
        rectangle_cover_accepted = False
        rectangle_cover_footprint: dict[str, object] | None = None
        if (
            self._use_rectangle_dictionary_cover
            and raster.footprint_geojson is not None
        ):
            (
                rectangle_cover_meta,
                rectangle_cover_accepted,
                rectangle_cover_footprint,
            ) = await self._run_rectangle_cover_dispatch(
                raster=raster,
                nodes=nodes,
                links=links,
            )

        # ── Sprint 48: dominant_angle hint chain + candidate retry ──
        # rectangle cover accepted 면 hint chain skip (BLOCKER 5 fallback 계약).
        # 1) hint candidate list 산출 (RTABMap link → graph centerline → OBB)
        # 2) Manhattan rectification 을 hint=None 으로 1차 호출 → metadata 보존
        # 3) hint enabled 이면 hint candidate retry — 각 candidate 로 Manhattan
        #    재호출, area_change reject 시 다음 candidate. 모두 reject 시 1차
        #    fallback 결과 유지.
        # 4) cleanup 적용 (default OFF — opt-in 시에만)
        rectification_meta: dict[str, object] | None = None
        cad_cleanup_meta: dict[str, object] | None = None
        rectified_footprint_pre_cleanup: dict[str, object] | None = None
        hint_meta: dict[str, object] | None = None
        cad_effect_pass: bool | None = None
        cad_effect_small_polygon_pass: bool | None = None
        already_rectified = False
        final_footprint_geojson = raster.footprint_geojson
        if rectangle_cover_accepted and rectangle_cover_footprint is not None:
            # BLOCKER 5: cover accepted → final footprint 자체를 union polygon
            # 으로 교체. hint chain / Manhattan / CAD cleanup 호출은 skip.
            final_footprint_geojson = rectangle_cover_footprint
            already_rectified = True

        if (
            not rectangle_cover_accepted
            and raster.footprint_geojson is not None
        ):
            from indoor_server.application.building.dominant_angle_hint import (
                HintGateThresholds,
                compute_dominant_angle_hints,
                cross_check_hints,
                extract_link_segments_xy,
            )
            from indoor_server.application.building.steps.manhattan_rectification import (
                ManhattanRectificationStep,
            )

            gate_thresholds = HintGateThresholds(
                rtabmap_link_min_segments=(
                    self._dominant_angle_hint_rtabmap_min_segments
                ),
                rtabmap_link_min_total_length_m=(
                    self._dominant_angle_hint_rtabmap_min_total_length_m
                ),
                rtabmap_link_min_best_bin_ratio=(
                    self._dominant_angle_hint_rtabmap_min_best_bin_ratio
                ),
                obb_min_aspect_ratio=(
                    self._dominant_angle_hint_obb_min_aspect_ratio
                ),
                cross_check_max_diff_deg=(
                    self._dominant_angle_hint_cross_check_max_diff_deg
                ),
            )

            link_segments_xy = extract_link_segments_xy(nodes, links)
            hint_candidates = compute_dominant_angle_hints(
                rtabmap_links_with_node_xy=link_segments_xy,
                footprint_polygon=raster.footprint_geojson,
                graph_edges=None,
                gate_thresholds=gate_thresholds,
            )
            cross_check = cross_check_hints(
                hint_candidates, gate_thresholds=gate_thresholds
            )

            # 1차 (hint 없이) — internal contour confidence path
            rectified = await asyncio.to_thread(
                ManhattanRectificationStep().run,
                raster.footprint_geojson,
                manhattan_max_area_change_floor_pointcloud=(
                    self._manhattan_floor_pointcloud_max_area_change
                ),
                dominant_angle_snap_mode="four_way",
            )
            rectification_meta = rectified.metadata()
            rectified_footprint_pre_cleanup = rectified.rectified_geojson
            final_footprint_geojson = rectified.rectified_geojson
            already_rectified = True

            chosen_source: str | None = None
            chosen_angle: float | None = None
            rejected_count = 0

            # ── Sprint 49 (Codex BLOCKER 6 + W-1): 4-way trigger ──
            # Sprint 48 회귀 fix. accepted=True 라도 다음 trigger 중 하나라도
            # 발화하면 candidate retry. baseline (1차) 결과는 보존하면서 retry
            # 후보 중 best 만 채택.
            attempt_reasons: list[str] = []
            baseline_orthogonality = _compute_corner_orthogonality(
                rectified.rectified_geojson
            )
            ortho_threshold = (
                self._polygon_cad_hint_retry_orthogonality_threshold
            )
            if not rectified.accepted:
                attempt_reasons.append("first_pass_rejected")
            if rectified.snap_mode_used == "fallback":
                attempt_reasons.append("first_pass_snap_fallback")
            if rectified.low_angle_confidence:
                attempt_reasons.append("first_pass_low_angle_confidence")
            if baseline_orthogonality < ortho_threshold:
                attempt_reasons.append("first_pass_low_orthogonality")
            primary_reason = attempt_reasons[0] if attempt_reasons else "first_pass_strong"
            should_attempt_chain = bool(attempt_reasons)

            # candidate per-iteration metadata (Codex BLOCKER 6)
            candidate_dumps: list[dict[str, object]] = []
            baseline_meta = rectified.metadata()
            baseline_dump: dict[str, object] = {
                "source": "baseline_no_hint",
                "angle_deg": float(rectified.dominant_angle_deg),
                "retry_accepted": bool(rectified.accepted),
                "retry_orthogonality": float(baseline_orthogonality),
                "retry_area_change_ratio": float(rectified.area_change_ratio),
                "retry_vertex_reduction_ratio": _vertex_reduction_ratio(baseline_meta),
                "shape_guard_passed": _shape_guard_pass(
                    raster.footprint_geojson, rectified.rectified_geojson
                ),
                "shape_preservation_guard": _shape_guard_breakdown(
                    raster.footprint_geojson, rectified.rectified_geojson
                ),
                "candidate_score": _candidate_score(
                    orthogonality=baseline_orthogonality,
                    area_change_ratio=rectified.area_change_ratio,
                    vertex_reduction=_vertex_reduction_ratio(baseline_meta),
                    accepted=rectified.accepted,
                    cad_effect_candidate_pass=_cad_effect_candidate_pass(
                        accepted=rectified.accepted,
                        fallback_used=rectified.fallback_used,
                        forced_rectilinear_used=rectified.forced_rectilinear_used,
                        snap_mode_used=rectified.snap_mode_used,
                        orthogonality=baseline_orthogonality,
                    ),
                    shape_guard_passed=_shape_guard_pass(
                        raster.footprint_geojson, rectified.rectified_geojson
                    ),
                ),
                "cad_effect_candidate_pass": _cad_effect_candidate_pass(
                    accepted=rectified.accepted,
                    fallback_used=rectified.fallback_used,
                    forced_rectilinear_used=rectified.forced_rectilinear_used,
                    snap_mode_used=rectified.snap_mode_used,
                    orthogonality=baseline_orthogonality,
                ),
                "selected": True,  # 일단 baseline 으로 시작. retry 채택 시 false 로 갱신.
                "reject_reason": None,
            }
            candidate_dumps.append(baseline_dump)

            _baseline_score_raw = baseline_dump["candidate_score"]
            best_score: float = float(_baseline_score_raw)  # type: ignore[arg-type]
            best_dump_idx = 0  # baseline

            if self._dominant_angle_hint_enabled and should_attempt_chain:
                # candidate iterate (모두 시도, best 만 채택)
                for cand_idx, cand in enumerate(hint_candidates):
                    if cand.angle_deg is None or not cand.accepted:
                        # gate 미통과 candidate 도 dump (선택 안 됨)
                        continue
                    # source 별 hint 주입 retry
                    retry = await asyncio.to_thread(
                        ManhattanRectificationStep().run,
                        raster.footprint_geojson,
                        manhattan_max_area_change_floor_pointcloud=(
                            self._manhattan_floor_pointcloud_max_area_change
                        ),
                        dominant_angle_snap_mode="four_way",
                        dominant_angle_hint_deg=cand.angle_deg,
                        dominant_angle_hint_source=cand.source,
                    )
                    # candidate metadata update (area_change_ratio 채우기)
                    import dataclasses as _dc

                    hint_candidates[cand_idx] = _dc.replace(
                        cand,
                        area_change_ratio=retry.area_change_ratio,
                        accepted=retry.accepted,
                        reject_reason=(
                            None if retry.accepted else "area_change_exceeded"
                        ),
                    )
                    retry_meta = retry.metadata()
                    retry_orth = _compute_corner_orthogonality(retry.rectified_geojson)
                    retry_shape_pass = _shape_guard_pass(
                        raster.footprint_geojson, retry.rectified_geojson
                    )
                    retry_shape_breakdown = _shape_guard_breakdown(
                        raster.footprint_geojson, retry.rectified_geojson
                    )
                    retry_cad_pass = _cad_effect_candidate_pass(
                        accepted=retry.accepted,
                        fallback_used=retry.fallback_used,
                        forced_rectilinear_used=retry.forced_rectilinear_used,
                        snap_mode_used=retry.snap_mode_used,
                        orthogonality=retry_orth,
                    )
                    retry_score = _candidate_score(
                        orthogonality=retry_orth,
                        area_change_ratio=retry.area_change_ratio,
                        vertex_reduction=_vertex_reduction_ratio(retry_meta),
                        accepted=retry.accepted,
                        cad_effect_candidate_pass=retry_cad_pass,
                        shape_guard_passed=retry_shape_pass,
                    )
                    cand_dump: dict[str, object] = {
                        "source": cand.source,
                        "angle_deg": float(cand.angle_deg),
                        "retry_accepted": bool(retry.accepted),
                        "retry_orthogonality": float(retry_orth),
                        "retry_area_change_ratio": float(retry.area_change_ratio),
                        "retry_vertex_reduction_ratio": _vertex_reduction_ratio(retry_meta),
                        "shape_guard_passed": retry_shape_pass,
                        "shape_preservation_guard": retry_shape_breakdown,
                        "candidate_score": float(retry_score),
                        "cad_effect_candidate_pass": retry_cad_pass,
                        "selected": False,
                        "reject_reason": (
                            None
                            if (retry.accepted and retry_shape_pass)
                            else "score_or_shape_reject"
                        ),
                    }
                    candidate_dumps.append(cand_dump)
                    rejected_count += 0 if retry.accepted else 1

                    # best 갱신 정책: accepted AND shape_guard_passed AND score 가 더 큼.
                    if (
                        retry.accepted
                        and retry_shape_pass
                        and retry_score > best_score
                    ):
                        best_score = retry_score
                        best_dump_idx = len(candidate_dumps) - 1
                        rectified = retry
                        rectification_meta = retry.metadata()
                        rectified_footprint_pre_cleanup = retry.rectified_geojson
                        final_footprint_geojson = retry.rectified_geojson
                        chosen_source = cand.source
                        chosen_angle = cand.angle_deg

            # selected 표시 갱신: 최종 채택된 dump 만 selected=True.
            for i, dump in enumerate(candidate_dumps):
                dump["selected"] = (i == best_dump_idx)

            chosen_after_accepted = bool(
                best_dump_idx > 0  # baseline 아닌 candidate 가 채택됨
            )
            chosen_reason = (
                "candidate_better"
                if chosen_after_accepted
                else "no_candidate_better"
            )

            hint_meta = {
                "enabled": self._dominant_angle_hint_enabled,
                "candidates": [c.to_dict() for c in hint_candidates],
                "chosen_source": chosen_source,
                "chosen_angle_deg": chosen_angle,
                "rejected_count": int(rejected_count),
                "cross_check": cross_check,
                "params": gate_thresholds.to_metadata(),
                # Sprint 49 (Codex BLOCKER 6 + W-1)
                "first_pass_baseline": {
                    "accepted": bool(baseline_dump["retry_accepted"]),
                    "snap_mode_used": rectified.snap_mode_used,
                    "low_angle_confidence": rectified.low_angle_confidence,
                    "corner_orthogonality_ratio": float(baseline_orthogonality),
                },
                "hint_chain_attempt_reasons": list(attempt_reasons),
                "hint_chain_primary_reason": primary_reason,
                "hint_chain_chosen_after_accepted": chosen_after_accepted,
                "hint_chain_chosen_reason": chosen_reason,
                "hint_chain_baseline_orthogonality": float(baseline_orthogonality),
                "hint_chain_attempt_reason": primary_reason,  # backward-compat
                "candidate_dumps": candidate_dumps,
                "candidates_evaluated": len(candidate_dumps),
                "polygon_cad_hint_retry_orthogonality_threshold": float(
                    ortho_threshold
                ),
            }
            # Sprint 48: hint chain 결과를 rectification_meta 에도 노출.
            rectification_meta = dict(rectification_meta or {})
            rectification_meta["dominant_angle_hint_chain"] = hint_meta

            logger.info(
                "floor_pc manhattan_rectification done dominant=%.2f accepted=%s "
                "area_change=%.3f confidence=%.3f snap_mode=%s "
                "hint_source=%s hint_rejected=%d",
                rectified.dominant_angle_deg,
                rectified.accepted,
                rectified.area_change_ratio,
                rectified.dominant_angle_confidence,
                rectified.snap_mode_used,
                chosen_source if chosen_source is not None else "none",
                rejected_count,
            )

            # Sprint 47 fix: rectification fallback이어도 CAD cleanup은 수행한다.
            # 단 Codex F-3 권고로 default OFF — opt-in 시에만.
            if self._polygon_cad_cleanup_enabled:
                from indoor_server.application.building.steps.polygon_cad_cleanup import (
                    PolygonCadCleanupStep,
                    PolygonCadCleanupStepParams,
                )

                if rectified.accepted:
                    cad_cleanup_input = rectified.rectified_geojson
                    cad_input_source = "rectified"
                else:
                    cad_cleanup_input = raster.footprint_geojson
                    cad_input_source = "raw_fallback"

                cleanup = await asyncio.to_thread(
                    PolygonCadCleanupStep(
                        PolygonCadCleanupStepParams(
                            enabled=True,
                            collinear_angle_tol_deg=(
                                self._polygon_cad_collinear_angle_tol_deg
                            ),
                            short_edge_min_length_m=(
                                self._polygon_cad_short_edge_min_length_m
                            ),
                            near_vertex_merge_distance_m=(
                                self._polygon_cad_near_vertex_merge_distance_m
                            ),
                            orthogonality_angle_tol_deg=(
                                self._polygon_cad_orthogonality_angle_tol_deg
                            ),
                        )
                    ).run,
                    cad_cleanup_input,
                )
                cad_cleanup_meta = dict(cleanup.metadata)
                cad_cleanup_meta["input_source"] = cad_input_source
                final_footprint_geojson = cleanup.cleaned_geojson

                # Sprint 48 (Codex F-2): cad_effect_pass 계산.
                cad_effect_pass, cad_effect_small_polygon_pass = (
                    _compute_cad_effect_pass(
                        rectification_meta=rectification_meta,
                        cleanup_meta=cad_cleanup_meta,
                    )
                )

                meta_for_log = cleanup.metadata
                logger.info(
                    "polygon_cad_cleanup done input=%s vertex %s→%s "
                    "orthogonality=%s iou=%s centroid_shift=%sm "
                    "cad_effect_pass=%s small_polygon_pass=%s",
                    cad_input_source,
                    meta_for_log.get("vertex_count_before"),
                    meta_for_log.get("vertex_count_after"),
                    meta_for_log.get("corner_orthogonality_ratio"),
                    meta_for_log.get("iou_raw_rectified"),
                    meta_for_log.get("centroid_shift_m"),
                    cad_effect_pass,
                    cad_effect_small_polygon_pass,
                )

        # Sprint 50: rectangle cover metadata 를 raster_meta 안에 mount
        # (별도 BuildCounts 필드 없이 floor_raster.rectangle_cover 키 활용).
        if rectangle_cover_meta is not None:
            raster_meta = dict(raster_meta)
            raster_meta["rectangle_cover"] = rectangle_cover_meta

        # ── Sprint 51: Wall-fitting polygon (default OFF, observer only) ──
        # production polygon 변경 0. final_footprint_geojson, raster.grid 등
        # 모두 그대로 통과한다. metadata 만 wall_polygon_meta 로 dump.
        wall_polygon_meta: dict[str, object] | None = None
        if self._use_wall_polygon:
            wall_polygon_meta = await self._run_wall_polygon_observer(
                nodes=nodes,
                frames=frames,
                floor_masks_by_node_id=floor_masks_by_node_id,
                obstacle_masks_by_node_id=wall_masks_by_node_id,
                z0=cloud.z0,
                floor_polygon_geojson=final_footprint_geojson,
            )

        return (
            cloud.z0,
            raster.grid,
            walkable_cells,
            final_footprint_geojson,
            pointcloud_meta,
            raster_meta,
            rectification_meta,
            cad_cleanup_meta,
            rectified_footprint_pre_cleanup,
            already_rectified,
            hint_meta,
            cad_effect_pass,
            cad_effect_small_polygon_pass,
            wall_polygon_meta,
        )

    async def _run_wall_polygon_observer(
        self,
        *,
        nodes: list[RtabmapNode],
        frames: list[RtabmapDataFrame],
        floor_masks_by_node_id: dict[int, np.ndarray],
        obstacle_masks_by_node_id: dict[int, np.ndarray] | None = None,
        z0: float,
        floor_polygon_geojson: dict[str, object] | None,
    ) -> dict[str, object] | None:
        """Sprint 51 — Wall polygon observer dispatch.

        7-step pipeline 결과를 metadata 로만 반환. production polygon 무영향.
        실패 시에도 metadata 는 항상 채워서 caller 가 fail_reason 으로 진단 가능.
        """
        from indoor_server.application.building.steps.wall_polygon import (
            AngleSnapParams,
            ComponentSplitParams,
            DensityRefineParams,
            LineFitParams,
            MergeParams,
            ObstacleSourceStepParams,
            PolygonAssemblyParams,
            ValidateParams,
            WallPolygonFromObstacleStep,
            WallPolygonStepParams,
        )

        params = WallPolygonStepParams(
            obstacle_source=ObstacleSourceStepParams(
                pixel_stride=self._floor_pointcloud_pixel_stride,
                height_above_floor_min_m=self._wall_polygon_obstacle_height_min_m,
                height_above_floor_max_m=self._wall_polygon_obstacle_height_max_m,
                mask_mode=(
                    "direct_mask"
                    if obstacle_masks_by_node_id is not None
                    else "inverse_floor"
                ),
            ),
            density=DensityRefineParams(
                min_cell_hits=self._wall_polygon_density_min_cell_hits,
                morph_close_radius_cells=(
                    self._wall_polygon_density_morph_close_radius_cells
                ),
            ),
            components=ComponentSplitParams(
                min_area_cells=self._wall_polygon_components_min_area_cells,
            ),
            line_fit=LineFitParams(
                min_linearity=self._wall_polygon_line_min_linearity,
                min_length_m=self._wall_polygon_line_min_length_m,
            ),
            snap=AngleSnapParams(
                snap_tolerance_deg=self._wall_polygon_snap_tolerance_deg,
            ),
            merge=MergeParams(
                same_offset_tolerance_m=(
                    self._wall_polygon_merge_offset_tolerance_m
                ),
                gap_fill_max_m=self._wall_polygon_merge_gap_fill_m,
            ),
            assembly=PolygonAssemblyParams(
                intersection_tolerance_m=(
                    self._wall_polygon_assembly_intersection_tolerance_m
                ),
                use_alpha_shape_fallback=(
                    self._wall_polygon_assembly_use_alpha_shape
                ),
            ),
            validate=ValidateParams(
                floor_iou_min=self._wall_polygon_validate_floor_iou_min,
                area_change_max_ratio=(
                    self._wall_polygon_validate_area_change_max_ratio
                ),
            ),
            min_wall_lines=self._wall_polygon_min_lines,
            max_wall_lines=self._wall_polygon_max_lines,
        )

        step = WallPolygonFromObstacleStep(params)
        try:
            result = await asyncio.to_thread(
                step.run,
                nodes=nodes,
                frames=frames,
                floor_masks_by_node_id=floor_masks_by_node_id,
                obstacle_masks_by_node_id=obstacle_masks_by_node_id,
                z0=z0,
                floor_polygon_geojson=floor_polygon_geojson,
            )
        except Exception as e:
            logger.warning("wall_polygon dispatch failed: %s", e)
            return {
                "enabled": True,
                "accepted": False,
                "fail_reason": f"runtime_error:{type(e).__name__}",
            }
        meta = dict(result.metadata)
        heatmap_obj = result.stage_outputs.get("obstacle_source")
        try:
            from indoor_server.application.building.steps.wall_polygon import (
                HeatmapBoundaryStep,
            )
            from indoor_server.application.building.steps.wall_polygon.obstacle_source import (
                ObstacleHeatmap,
            )

            if isinstance(heatmap_obj, ObstacleHeatmap):
                boundary = await asyncio.to_thread(
                    HeatmapBoundaryStep().run,
                    heatmap_obj,
                )
                boundary_meta = dict(boundary.metadata)
                boundary_meta["accepted"] = bool(boundary.accepted)
                boundary_meta["fail_reason"] = boundary.fail_reason
                boundary_meta["boundary_geojson"] = boundary.boundary_geojson
                meta["heatmap_boundary"] = boundary_meta
        except Exception as e:
            logger.warning("heatmap_boundary observer failed: %s", e)
            meta["heatmap_boundary"] = {
                "accepted": False,
                "fail_reason": f"runtime_error:{type(e).__name__}",
            }
        heatmap_boundary_log = meta.get("heatmap_boundary")
        heatmap_boundary_accepted = (
            heatmap_boundary_log.get("accepted")
            if isinstance(heatmap_boundary_log, dict)
            else None
        )
        logger.info(
            "wall_polygon done accepted=%s fail_reason=%s line_count=%s "
            "vertex_count=%s orthogonality=%s iou=%s heatmap_boundary=%s",
            result.accepted,
            result.fail_reason,
            meta.get("line_count"),
            meta.get("vertex_count"),
            meta.get("corner_orthogonality_ratio"),
            meta.get("iou_with_floor"),
            heatmap_boundary_accepted,
        )
        return meta

    async def _run_rectangle_cover_dispatch(
        self,
        *,
        raster: object,
        nodes: list[RtabmapNode],
        links: list[RtabmapLink],
    ) -> tuple[dict[str, object] | None, bool, dict[str, object] | None]:
        """Sprint 50 v2 — rectangle cover dispatch (axis pair / multi-component).

        axes_mode="full18": v1 그대로 18 angle full sweep (whole heatmap).
        axes_mode="pair":
            1) RTABMap link length-weighted dominant angle 추출 (gate 0.10).
            2) 실패 시 footprint OBB long edge.
            3) 둘 다 실패 시 v1 18-angle full path 로 fallback.
            4) heatmap connected component 분리, 각 component 별 axis pair 추정
               + cover. 모든 component rectangle 결과를 union.

        Returns:
            (rectangle_cover_meta dict, accepted bool, footprint_geojson dict|None).
        """
        from shapely.geometry import MultiPolygon, Polygon
        from shapely.geometry import mapping as _mapping
        from shapely.geometry import shape as _shape
        from shapely.ops import unary_union as _unary_union

        from indoor_server.application.building.dominant_angle_hint import (
            extract_link_segments_xy,
        )
        from indoor_server.application.building.steps.rectangle_dictionary_cover import (  # noqa: E501
            RectangleDictionaryCoverParams,
            RectangleDictionaryCoverResult,
            RectangleDictionaryCoverStep,
            axes_pair_from_primary,
            estimate_dominant_axis_from_links,
            estimate_dominant_axis_from_obb,
            split_heatmap_by_components,
        )

        common_params: dict[str, object] = {
            "precision_threshold": self._rectangle_cover_precision_threshold,
            "recall_min": self._rectangle_cover_recall_min,
            "over_cover_max": self._rectangle_cover_over_cover_max,
            "time_budget_sec": self._rectangle_cover_time_budget_sec,
            "candidate_stride_cells": (
                self._rectangle_cover_candidate_stride_cells
            ),
            "max_candidates_per_dimension": (
                self._rectangle_cover_max_candidates_per_dimension
            ),
            "precision_threshold_dynamic": (
                self._rectangle_cover_precision_threshold_dynamic
            ),
            "precision_threshold_min": (
                self._rectangle_cover_precision_threshold_min
            ),
        }

        def _meta_from_result(
            res: RectangleDictionaryCoverResult,
            extra: dict[str, object] | None = None,
        ) -> dict[str, object]:
            base: dict[str, object] = {
                "attempted": True,
                "accepted": bool(res.accepted),
                "fallback_used": bool(res.fallback_used),
                "fallback_source": res.fallback_source,
                **res.metadata,
            }
            if extra:
                base.update(extra)
            return base

        # axes_mode "full18" — v1 그대로
        if self._rectangle_cover_axes_mode == "full18":
            cover_step = RectangleDictionaryCoverStep(
                RectangleDictionaryCoverParams(**common_params)  # type: ignore[arg-type]
            )
            cover_result = await asyncio.to_thread(cover_step.run, raster.grid)  # type: ignore[attr-defined]
            meta = _meta_from_result(
                cover_result,
                extra={"axes_mode": "full18", "v2_active": False},
            )
            self._log_cover(cover_result)
            footprint = (
                cover_result.footprint_geojson
                if cover_result.accepted
                else None
            )
            return meta, bool(cover_result.accepted), footprint

        # axes_mode "pair" — v2
        link_segments_xy = extract_link_segments_xy(nodes, links)
        global_axis_link = estimate_dominant_axis_from_links(
            link_segments_xy,
            min_best_bin_ratio=(
                self._rectangle_cover_axis_link_min_best_bin_ratio
            ),
        )

        global_axis_obb: tuple[float, float] | None = None
        if raster.footprint_geojson is not None:  # type: ignore[attr-defined]
            try:
                geom = _shape(raster.footprint_geojson)  # type: ignore[attr-defined]
                global_axis_obb = estimate_dominant_axis_from_obb(
                    geom,
                    min_aspect_ratio=(
                        self._rectangle_cover_axis_obb_min_aspect_ratio
                    ),
                )
            except Exception as e:
                logger.warning(
                    "rectangle_cover_v2 OBB axis extract failed: %s", e
                )

        if global_axis_link is None and global_axis_obb is None:
            # v2 fallback to v1 18-angle full sweep (Sprint 49 hint chain
            # 으로 가는 BLOCKER 5 fallback 은 cover_result.accepted=False 시).
            logger.info(
                "rectangle_cover_v2 no global axis (link=None obb=None) — "
                "falling back to full18 sweep"
            )
            cover_step = RectangleDictionaryCoverStep(
                RectangleDictionaryCoverParams(**common_params)  # type: ignore[arg-type]
            )
            cover_result = await asyncio.to_thread(cover_step.run, raster.grid)  # type: ignore[attr-defined]
            meta = _meta_from_result(
                cover_result,
                extra={
                    "axes_mode": "pair",
                    "v2_active": True,
                    "v2_axes_resolution": "full18_fallback",
                    "v2_global_axis_link_deg": None,
                    "v2_global_axis_obb_deg": None,
                },
            )
            self._log_cover(cover_result)
            footprint = (
                cover_result.footprint_geojson
                if cover_result.accepted
                else None
            )
            return meta, bool(cover_result.accepted), footprint

        # split heatmap into components
        heatmap = raster.grid.observation_count.astype(np.int64, copy=False)  # type: ignore[attr-defined]
        components = split_heatmap_by_components(
            heatmap,
            origin=raster.grid.origin,  # type: ignore[attr-defined]
            min_component_cells=self._rectangle_cover_min_component_cells,
        )
        if not components:
            # 통째로 0 hits — empty heatmap 처리에 위임
            cover_step = RectangleDictionaryCoverStep(
                RectangleDictionaryCoverParams(**common_params)  # type: ignore[arg-type]
            )
            cover_result = await asyncio.to_thread(cover_step.run, raster.grid)  # type: ignore[attr-defined]
            meta = _meta_from_result(
                cover_result,
                extra={
                    "axes_mode": "pair",
                    "v2_active": True,
                    "v2_axes_resolution": "no_components",
                },
            )
            return meta, False, None

        # global fallback axis pair (component link 추정 실패 시)
        global_axis_deg: float | None = None
        global_axis_source: str = "none"
        if global_axis_link is not None:
            global_axis_deg = global_axis_link[0]
            global_axis_source = "rtabmap_link_global"
        elif global_axis_obb is not None:
            global_axis_deg = global_axis_obb[0]
            global_axis_source = "footprint_obb_global"

        component_results: list[dict[str, object]] = []
        all_rectangles_polys: list[Polygon] = []
        any_accepted = False
        all_axes_in_pair_global = True
        all_axes_set: set[float] = set()
        cover_attempt_meta: list[dict[str, object]] = []

        for comp_idx, (sub_grid, bbox) in enumerate(components):
            # component-local link segments
            r0, c0, r1, c1 = bbox
            cs = float(raster.grid.origin.cell_size)  # type: ignore[attr-defined]
            xb0 = raster.grid.origin.x0 + c0 * cs  # type: ignore[attr-defined]
            yb0 = raster.grid.origin.y0 + r0 * cs  # type: ignore[attr-defined]
            xb1 = raster.grid.origin.x0 + c1 * cs  # type: ignore[attr-defined]
            yb1 = raster.grid.origin.y0 + r1 * cs  # type: ignore[attr-defined]

            local_link_segments = [
                (a, b)
                for (a, b) in link_segments_xy
                if (
                    xb0 <= a[0] <= xb1
                    and yb0 <= a[1] <= yb1
                    and xb0 <= b[0] <= xb1
                    and yb0 <= b[1] <= yb1
                )
            ]
            local_axis_link = estimate_dominant_axis_from_links(
                local_link_segments,
                min_best_bin_ratio=(
                    self._rectangle_cover_axis_link_min_best_bin_ratio
                ),
            )

            local_axis_deg: float | None = None
            local_axis_source: str
            if local_axis_link is not None:
                local_axis_deg = local_axis_link[0]
                local_axis_source = "rtabmap_link_local"
            else:
                # component OBB on its own mask
                from shapely.geometry import Polygon as _Poly
                comp_mask = sub_grid.observation_count > 0
                if comp_mask.any():
                    rows = np.where(comp_mask.any(axis=1))[0]
                    cols = np.where(comp_mask.any(axis=0))[0]
                    rr0, rr1 = int(rows.min()), int(rows.max()) + 1
                    cc0, cc1 = int(cols.min()), int(cols.max()) + 1
                    sg_x0 = sub_grid.origin.x0
                    sg_y0 = sub_grid.origin.y0
                    xy_corners = [
                        (sg_x0 + cc0 * cs, sg_y0 + rr0 * cs),
                        (sg_x0 + cc1 * cs, sg_y0 + rr0 * cs),
                        (sg_x0 + cc1 * cs, sg_y0 + rr1 * cs),
                        (sg_x0 + cc0 * cs, sg_y0 + rr1 * cs),
                    ]
                    comp_poly = _Poly(xy_corners)
                    local_obb = estimate_dominant_axis_from_obb(
                        comp_poly,
                        min_aspect_ratio=(
                            self._rectangle_cover_axis_obb_min_aspect_ratio
                        ),
                    )
                    if local_obb is not None:
                        local_axis_deg = local_obb[0]
                        local_axis_source = "component_obb"
                    elif global_axis_deg is not None:
                        local_axis_deg = global_axis_deg
                        local_axis_source = (
                            "global_fallback:" + global_axis_source
                        )
                    else:
                        local_axis_source = "none"
                elif global_axis_deg is not None:
                    local_axis_deg = global_axis_deg
                    local_axis_source = "global_fallback:" + global_axis_source
                else:
                    local_axis_source = "none"

            if local_axis_deg is None:
                cover_attempt_meta.append(
                    {
                        "component_idx": comp_idx,
                        "bbox": list(bbox),
                        "axis_source": local_axis_source,
                        "axis_deg": None,
                        "rectangle_count": 0,
                        "accepted": False,
                        "skip_reason": "no_axis",
                    }
                )
                continue

            axes_pair = axes_pair_from_primary(local_axis_deg)
            params_dict = dict(common_params)
            params_dict["axes_override"] = axes_pair
            comp_step = RectangleDictionaryCoverStep(
                RectangleDictionaryCoverParams(**params_dict)  # type: ignore[arg-type]
            )
            comp_result = await asyncio.to_thread(comp_step.run, sub_grid)
            cover_attempt_meta.append(
                {
                    "component_idx": comp_idx,
                    "bbox": list(bbox),
                    "axis_source": local_axis_source,
                    "axis_deg": float(local_axis_deg),
                    "axes_pair": [float(axes_pair[0]), float(axes_pair[1])],
                    "rectangle_count": _meta_int(
                        comp_result.metadata, "rectangle_count", 0
                    ),
                    "accepted": bool(comp_result.accepted),
                    "recall": comp_result.metadata.get("recall"),
                    "over_cover_ratio": comp_result.metadata.get("over_cover_ratio"),
                    "all_axes_in_pair": comp_result.metadata.get(
                        "all_axes_in_pair"
                    ),
                    "fallback_reason": comp_result.metadata.get(
                        "fallback_reason"
                    ),
                    "wall_time_sec": comp_result.metadata.get("wall_time_sec"),
                }
            )
            component_results.append({"idx": comp_idx, "meta": comp_result.metadata})
            if comp_result.accepted and comp_result.rectangles:
                any_accepted = True
                for rect in comp_result.rectangles:
                    all_rectangles_polys.append(rect.world_polygon)
                if not bool(
                    comp_result.metadata.get("all_axes_in_pair", False)
                ):
                    all_axes_in_pair_global = False
                all_axes_set.update(
                    float(round(p % 180.0, 4)) for p in axes_pair
                )

        # union all rectangles across components
        union_geom: Polygon | MultiPolygon | None = None
        union_geojson: dict[str, object] | None = None
        if all_rectangles_polys:
            union_geom = _unary_union(all_rectangles_polys)
            if union_geom.is_empty:
                union_geom = None
            else:
                union_geojson = dict(_mapping(union_geom))

        # aggregate metric
        total_hits_global = int(heatmap.sum())
        if union_geom is not None:
            from indoor_server.application.building.steps.rectangle_dictionary_cover import (  # noqa: E501
                _world_polygons_to_grid_hit_count,
            )
            polys_for_count: list[Polygon]
            if isinstance(union_geom, Polygon):
                polys_for_count = [union_geom]
            elif isinstance(union_geom, MultiPolygon):
                polys_for_count = list(union_geom.geoms)
            else:
                polys_for_count = []
            cover_mask, covered_hits = _world_polygons_to_grid_hit_count(
                polygons=polys_for_count,
                heatmap=heatmap,
                origin=raster.grid.origin,  # type: ignore[attr-defined]
            )
            union_area_cells = int(cover_mask.sum())
            target_cells = int((heatmap > 0).sum())
            over_cover_cells = max(0, union_area_cells - target_cells)
            over_cover_ratio = (
                over_cover_cells / float(union_area_cells)
                if union_area_cells > 0
                else 0.0
            )
            recall = (
                covered_hits / float(total_hits_global)
                if total_hits_global > 0
                else 0.0
            )
        else:
            covered_hits = 0
            union_area_cells = 0
            over_cover_cells = 0
            over_cover_ratio = 0.0
            recall = 0.0

        # global accept gate (per-component accepted ≥1 + recall_min + over_cover_max)
        global_accepted = (
            any_accepted
            and recall >= self._rectangle_cover_recall_min
            and over_cover_ratio <= self._rectangle_cover_over_cover_max
        )

        agg_meta: dict[str, object] = {
            "attempted": True,
            "axes_mode": "pair",
            "v2_active": True,
            "v2_axes_resolution": "multi_component",
            "v2_global_axis_link_deg": (
                global_axis_link[0] if global_axis_link is not None else None
            ),
            "v2_global_axis_link_best_bin_ratio": (
                global_axis_link[1] if global_axis_link is not None else None
            ),
            "v2_global_axis_obb_deg": (
                global_axis_obb[0] if global_axis_obb is not None else None
            ),
            "v2_global_axis_obb_aspect": (
                global_axis_obb[1] if global_axis_obb is not None else None
            ),
            "v2_component_count": len(components),
            "v2_components": cover_attempt_meta,
            "rectangle_count": len(all_rectangles_polys),
            "total_hits": total_hits_global,
            "covered_hits": int(covered_hits),
            "uncovered_hits": int(total_hits_global - covered_hits),
            "recall": float(recall),
            "over_cover_ratio": float(over_cover_ratio),
            "over_cover_cells": int(over_cover_cells),
            "union_area_cells": int(union_area_cells),
            "target_cells_in_heatmap": int((heatmap > 0).sum()),
            "all_axes_in_pair": bool(
                all_axes_in_pair_global and any_accepted
            ),
            "axes_used": sorted(all_axes_set),
            "accepted": bool(global_accepted),
            "fallback_used": not bool(global_accepted),
            "fallback_source": (
                None if global_accepted else "sprint49_hint_chain"
            ),
            "fallback_reason": (
                None if global_accepted else "v2_aggregate_gate_failed"
            ),
            "params": {
                "rectangle_cover_axes_mode": (
                    self._rectangle_cover_axes_mode
                ),
                "rectangle_cover_precision_threshold_dynamic": (
                    self._rectangle_cover_precision_threshold_dynamic
                ),
                "rectangle_cover_precision_threshold_min": (
                    self._rectangle_cover_precision_threshold_min
                ),
                "rectangle_cover_min_component_cells": (
                    self._rectangle_cover_min_component_cells
                ),
                "rectangle_cover_axis_link_min_best_bin_ratio": (
                    self._rectangle_cover_axis_link_min_best_bin_ratio
                ),
                "rectangle_cover_axis_obb_min_aspect_ratio": (
                    self._rectangle_cover_axis_obb_min_aspect_ratio
                ),
                "rectangle_cover_recall_min": (
                    self._rectangle_cover_recall_min
                ),
                "rectangle_cover_over_cover_max": (
                    self._rectangle_cover_over_cover_max
                ),
                "rectangle_cover_precision_threshold": (
                    self._rectangle_cover_precision_threshold
                ),
            },
        }
        logger.info(
            "rectangle_cover_v2 accepted=%s rects=%d recall=%.3f over=%.3f "
            "components=%d link_axis=%s obb_axis=%s all_axes_in_pair=%s",
            global_accepted,
            len(all_rectangles_polys),
            recall,
            over_cover_ratio,
            len(components),
            (
                f"{global_axis_link[0]:.2f}"
                if global_axis_link is not None
                else "none"
            ),
            (
                f"{global_axis_obb[0]:.2f}"
                if global_axis_obb is not None
                else "none"
            ),
            agg_meta["all_axes_in_pair"],
        )
        return (
            agg_meta,
            bool(global_accepted),
            union_geojson if global_accepted else None,
        )

    def _log_cover(
        self, result: object
    ) -> None:
        """v1/full18 path 의 accepted 로그."""
        try:
            meta = result.metadata  # type: ignore[attr-defined]
            if result.accepted:  # type: ignore[attr-defined]
                logger.info(
                    "rectangle_dictionary_cover ACCEPTED — skipping hint chain. "
                    "rects=%s recall=%.3f over=%.3f wall=%.2fs",
                    meta.get("rectangle_count"),
                    float(meta.get("recall", 0.0)),
                    float(meta.get("over_cover_ratio", 0.0)),
                    float(meta.get("wall_time_sec", 0.0)),
                )
            else:
                logger.warning(
                    "rectangle_dictionary_cover NOT_ACCEPTED reason=%s — "
                    "falling back to sprint49 hint chain",
                    meta.get("fallback_reason"),
                )
        except Exception:
            pass

    async def _run_rtabmap_trajectory_and_grid(
        self,
        nodes: list[RtabmapNode],
        links: list[RtabmapLink],
        features: list[RtabmapFeaturePoint],
        frames: list[RtabmapDataFrame],
        progress_sink: ProgressSink,
        cancel_check: CancelCheck,
    ) -> tuple[float, WalkableGrid, int, dict[str, object] | None, dict[str, object]]:
        """RTAB-Map Node/Link trajectory -> walkable grid.

        This path intentionally skips keyframe jpg segmentation and raw ARKit
        pose_matrix back-projection.
        """
        from indoor_server.application.building.steps.rtabmap_trajectory import (
            RtabmapTrajectoryRoadStep,
        )

        await progress_sink(BuildStep.BACK_PROJECT, 0.25)
        if await cancel_check():
            return 0.0, self._empty_walkable_grid(0.0), 0, None, {
                "source": "rtabmap_node_link",
                "issues": ["cancelled_before_rtabmap_trajectory"],
            }

        step = RtabmapTrajectoryRoadStep(
            half_width_m=self._rtabmap_trajectory_half_width_m,
            cell_size_m=0.10,
            use_feature_evidence=self._rtabmap_feature_evidence_enabled,
        )
        result = await asyncio.to_thread(
            step.run,
            nodes=nodes,
            links=links,
            features=features,
        )
        grid = result.grid
        footprint_geojson = result.footprint_geojson
        metadata = dict(result.metadata)
        depth_confidence: np.ndarray | None = None
        depth_avoid: np.ndarray | None = None
        floor_masks_by_node_id: dict[int, np.ndarray] | None = None
        avoid_masks_by_node_id: dict[int, np.ndarray] | None = None
        if (
            self._rtabmap_image_segmentation_enabled
            and frames
            and result.grid.mask.any()
        ):
            from indoor_server.application.building.steps.rtabmap_image_evidence import (
                RtabmapImageEvidenceStep,
            )

            node_pose_ids = {node.node_id for node in nodes}
            image_evidence = await RtabmapImageEvidenceStep().run(
                frames=frames,
                segmenter=self._segmenter,
                node_pose_ids=node_pose_ids,
                orientation_mode=self._rtabmap_image_orientation_mode,  # type: ignore[arg-type]
                floor_mask_min_ratio=self._rtabmap_image_floor_mask_min_ratio,
                wall_mask_max_ratio=self._rtabmap_image_wall_mask_max_ratio,
            )
            floor_masks_by_node_id = image_evidence.floor_masks_by_node_id
            avoid_masks_by_node_id = image_evidence.nonwalkable_masks_by_node_id
            metadata["image_evidence"] = image_evidence.metadata()
        if self._rtabmap_depth_evidence_enabled and frames and result.grid.mask.any():
            from indoor_server.application.building.steps.rtabmap_depth_evidence import (
                RtabmapDepthEvidenceStep,
            )

            depth_evidence = await asyncio.to_thread(
                RtabmapDepthEvidenceStep(
                    vertical_tolerance_m=self._rtabmap_depth_vertical_tolerance_m,
                    require_floor_mask=self._rtabmap_image_segmentation_enabled,
                ).run,
                frames=frames,
                nodes=nodes,
                grid=result.grid,
                floor_masks_by_node_id=floor_masks_by_node_id,
                avoid_masks_by_node_id=avoid_masks_by_node_id,
            )
            depth_confidence = depth_evidence.confidence
            depth_avoid = depth_evidence.avoid
            metadata["depth_evidence"] = depth_evidence.metadata()

        if self._rtabmap_rectilinear_cover_enabled and result.grid.mask.any():
            from indoor_server.application.building.steps.rectilinear_cover import (
                RectilinearWalkableCoverStep,
            )
            from indoor_server.application.building.steps.rtabmap_floor_guard import (
                RtabmapFloorGuardStep,
            )

            cover_input = result.grid
            if depth_confidence is not None or depth_avoid is not None:
                floor_guard = await asyncio.to_thread(
                    RtabmapFloorGuardStep().run,
                    result.grid,
                    confidence=depth_confidence,
                    avoid=depth_avoid,
                )
                cover_input = floor_guard.grid
                metadata["floor_guard"] = floor_guard.metadata

            cover = await asyncio.to_thread(
                RectilinearWalkableCoverStep(
                    dominant_angle_deg=(
                        _float_or_none(metadata.get("dominant_angle_deg"))
                        if self._rtabmap_rectilinear_cover_rotated_grid_enabled
                        else None
                    ),
                ).run,
                cover_input,
                confidence=depth_confidence,
                avoid=depth_avoid,
            )
            grid = cover.grid
            footprint_geojson = cover.footprint_geojson or result.footprint_geojson
            metadata["pre_rectilinear_walkable_cells"] = int(result.grid.mask.sum())
            metadata["pre_rectilinear_footprint_geojson"] = result.footprint_geojson
            metadata["rectilinear_cover"] = cover.metadata

        walkable_cells = int(grid.mask.sum())
        await progress_sink(BuildStep.WALKABLE_GRID, 0.40)
        return (
            result.z0,
            grid,
            walkable_cells,
            footprint_geojson,
            metadata,
        )

    async def _run_triangulation_and_grid(
        self,
        all_masks: list[KeyframeMasks],
        tz_values: list[float],
        progress_sink: ProgressSink,
        cancel_check: CancelCheck,
    ) -> tuple[float, WalkableGrid, int]:
        """BACK_PROJECT(triangulation) + WALKABLE_GRID. (z0, grid, walkable_cells) 반환."""
        from indoor_server.application.building.steps.back_projection import default_intrinsics
        from indoor_server.application.building.steps.triangulation import TriangulationStep

        assert self._sp_lg_runner is not None  # 생성자에서 보장

        await progress_sink(BuildStep.BACK_PROJECT, 0.25)
        grid_step = WalkableGridStep()
        z0 = grid_step.estimate_z0(tz_values)

        intrinsics_by_frame = [
            default_intrinsics(m.floor_mask.shape[1], m.floor_mask.shape[0])
            for m in all_masks
        ]

        step = TriangulationStep(
            sp_lg_runner=self._sp_lg_runner,
            window=self._triangulation_window,
            enable_floor_mask_gate=self._triangulation_floor_gate,
            min_match_score=self._triangulation_min_score,
            max_pair_matches=self._triangulation_max_matches,
        )

        async def _triang_progress(p: float) -> None:
            await progress_sink(BuildStep.BACK_PROJECT, 0.25 + p * 0.12)

        if await cancel_check():
            empty_cloud_pts = [np.zeros((0, 3), dtype=np.float64) for _ in all_masks]
            grid = grid_step.run(iter(empty_cloud_pts), [(m.tx, m.ty) for m in all_masks], z0=z0)
            return z0, grid, int(grid.mask.sum())

        cloud = await step.run(
            masks=all_masks,
            storage_root=self._storage_root,
            intrinsics_by_frame=intrinsics_by_frame,
            z0=z0,
            progress=_triang_progress,
        )

        trajectory_pts = [(m.tx, m.ty) for m in all_masks]

        def _points_gen() -> Generator[np.ndarray, None, None]:
            yield from cloud.points_per_frame

        grid = await asyncio.to_thread(
            lambda: grid_step.run(_points_gen(), trajectory_pts, z0=z0)
        )

        if self._use_trajectory_buffer:
            from indoor_server.application.building.steps.trajectory_buffer import (
                TrajectoryBufferStep,
            )

            buffer_step = TrajectoryBufferStep(
                buffer_m=self._trajectory_buffer_m,
                clip_to_triangulation_hull=False,  # Sprint 22 예정
            )
            triangulated_xy = (
                np.vstack([p[:, :2] for p in cloud.points_per_frame if len(p) > 0])
                if cloud.total_kept > 0
                else None
            )
            grid = await asyncio.to_thread(
                buffer_step.run, grid, trajectory_pts, triangulated_xy
            )

        walkable_cells = int(grid.mask.sum())
        logger.info(
            "triangulation walkable_grid done cells=%d z0=%.3f total_kept=%d buffer=%s",
            walkable_cells,
            z0,
            cloud.total_kept,
            f"{self._trajectory_buffer_m:.2f}m" if self._use_trajectory_buffer else "off",
        )
        await progress_sink(BuildStep.WALKABLE_GRID, 0.40)
        return z0, grid, walkable_cells

    async def _run_adaptive_buffer_and_grid(
        self,
        all_masks: list[KeyframeMasks],
        tz_values: list[float],
        progress_sink: ProgressSink,
        cancel_check: CancelCheck,
    ) -> tuple[float, WalkableGrid, int, dict[str, object] | None]:
        """FLOOR_SEG 결과 → adaptive buffer → WalkableGrid.
        (z0, grid, walkable_cells, footprint_geojson) 반환.

        BACK_PROJECT + WALKABLE_GRID 두 단계를 한 번에 처리.
        progress: BACK_PROJECT 0.25 → 0.33 → WALKABLE_GRID 0.40.
        footprint_geojson: shapely union polygon → GeoJSON dict (Sprint 24 IMDF export용).
        """
        from indoor_server.application.building.steps.adaptive_buffer import AdaptiveBufferStep
        from indoor_server.application.building.steps.back_projection import default_intrinsics

        await progress_sink(BuildStep.BACK_PROJECT, 0.25)
        grid_step = WalkableGridStep()
        z0 = grid_step.estimate_z0(tz_values)

        intrinsics_by_frame = [
            default_intrinsics(m.floor_mask.shape[1], m.floor_mask.shape[0])
            for m in all_masks
        ]

        if await cancel_check():
            return z0, self._empty_walkable_grid(z0), 0, None

        step = AdaptiveBufferStep(
            strip_bottom_fraction=self._adaptive_buffer_strip_fraction,
            min_buffer_m=self._adaptive_buffer_min_m,
            max_buffer_m=self._adaptive_buffer_max_m,
            cell_size_m=0.10,
            segmenter_pre_strips=False,
        )

        await progress_sink(BuildStep.BACK_PROJECT, 0.33)
        grid, footprint_geojson = await step.run_with_footprint(
            masks=all_masks, z0=z0, intrinsics_by_frame=intrinsics_by_frame
        )

        walkable_cells = int(grid.mask.sum())
        logger.info(
            "adaptive_buffer walkable_grid done cells=%d z0=%.3f frames=%d",
            walkable_cells, z0, len(all_masks),
        )
        await progress_sink(BuildStep.WALKABLE_GRID, 0.40)
        return z0, grid, walkable_cells, footprint_geojson

    def _empty_walkable_grid(self, z0: float) -> WalkableGrid:
        from indoor_server.domain.building.models import GridOrigin

        origin = GridOrigin(x0=0.0, y0=0.0, z0=z0, cell_size=0.10, w=1, h=1)
        return WalkableGrid(
            origin=origin,
            mask=np.zeros((1, 1), dtype=bool),
            observation_count=np.zeros((1, 1), dtype=np.uint16),
        )

    async def _run_back_project_and_grid(
        self,
        all_masks: list[KeyframeMasks],
        tz_values: list[float],
        progress_sink: ProgressSink,
        cancel_check: CancelCheck,
    ) -> tuple[float, WalkableGrid, int]:
        """BACK_PROJECT + WALKABLE_GRID 단계 실행. (z0, grid, walkable_cells) 반환."""
        await progress_sink(BuildStep.BACK_PROJECT, 0.25)
        bp_step = BackProjectionStep()
        grid_step = WalkableGridStep()
        z0 = grid_step.estimate_z0(tz_values)

        # W-3: cancel_check를 제너레이터 내부에서 호출할 수 없으므로
        #       asyncio.Event로 취소 신호를 전파한다.
        cancelled = False

        def _gen_points() -> Generator[np.ndarray, None, None]:
            nonlocal cancelled
            for m in all_masks:
                if cancelled:
                    return
                h, w = m.floor_mask.shape
                pts = bp_step.run(
                    floor_mask=m.floor_mask,
                    pose_bytes=m.pose_matrix,
                    image_w=w,
                    image_h=h,
                    z0=z0,
                )
                yield pts

        # W-3: BACK_PROJECT 구간 중 취소 체크
        # asyncio.to_thread + asyncio.create_task 모두 Task로 래핑해 asyncio.wait에 전달
        trajectory_pts = [(m.tx, m.ty) for m in all_masks]
        cancel_task: asyncio.Task[bool] = asyncio.create_task(cancel_check())
        grid_task: asyncio.Task[WalkableGrid] = asyncio.create_task(
            asyncio.to_thread(
                lambda: grid_step.run(
                    _gen_points(),
                    trajectory_pts,
                    z0=z0,  # W-7: estimate_z0 값 전달
                )
            )
        )

        done, pending = await asyncio.wait(
            {cancel_task, grid_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if cancel_task in done and cancel_task.result():
            # 취소 신호 — cancelled 플래그로 제너레이터 조기 종료 유도 후 대기
            cancelled = True
            grid = await grid_task
            walkable_cells = int(grid.mask.sum())
            logger.info("walkable_grid cancelled mid-backproject cells=%d", walkable_cells)
            return z0, grid, walkable_cells

        # 정상 완료 — cancel_task만 정리 (grid_task는 대기해야 하므로 취소 금지)
        if cancel_task in pending:
            cancel_task.cancel()
        grid = await grid_task

        walkable_cells = int(grid.mask.sum())
        logger.info("walkable_grid done cells=%d z0=%.3f", walkable_cells, z0)

        # W-2: WALKABLE_GRID progress_sink 발행
        await progress_sink(BuildStep.WALKABLE_GRID, 0.40)

        return z0, grid, walkable_cells

    async def _run_depth_aware_back_project_and_grid(
        self,
        all_masks: list[KeyframeMasks],
        tz_values: list[float],
        progress_sink: ProgressSink,
        cancel_check: CancelCheck,
    ) -> tuple[float, WalkableGrid, int, list[DepthMap]]:
        """BACK_PROJECT(depth-aware) + WALKABLE_GRID.

        multi-view scale 활성 시:
            Pass 1: depth map + per-frame 초기 scale 수집 (pts는 Pass 2에서 생성)
            SuperPoint+LightGlue 매칭 → scipy least_squares로 전역 scale 최적화
            Pass 2: 최적 scale로 재 back-project → world points

        비활성 시: 기존 per-frame scale 경로.

        반환: (z0, grid, walkable_cells, depth_maps)
        """
        from indoor_server.application.building.steps.back_projection import (
            default_intrinsics,
        )
        from indoor_server.application.building.steps.depth_back_projection import (
            DepthAwareBackProjectionStep,
            ScaleCalibrator,
        )

        assert self._depth_runner is not None  # type guard (생성자에서 보장)

        await progress_sink(BuildStep.BACK_PROJECT, 0.25)
        grid_step = WalkableGridStep()
        z0 = grid_step.estimate_z0(tz_values)
        use_rt = self._use_multiview_scale  # diag 결과 stored pose는 view matrix
        bp_step = DepthAwareBackProjectionStep(
            depth_runner=self._depth_runner,
            scale_calibrator=ScaleCalibrator(use_transposed_rotation=use_rt),
            use_transposed_rotation=use_rt,
        )

        # Pass 1: depth map + 초기 scale 수집
        depth_arr_by_seq: dict[int, np.ndarray] = {}
        initial_scales: dict[int, float] = {}
        pts_by_frame: dict[int, np.ndarray] = {}
        from indoor_server.application.building.steps.back_projection import Intrinsics
        intrinsics_by_frame: list[Intrinsics] = []
        valid_frame_indices: list[int] = []

        for idx, m in enumerate(all_masks):
            if await cancel_check():
                break

            img_rel = f"scans/{m.scan_id}/keyframes/{m.seq:06d}.jpg"
            img_path = self._storage_root / img_rel
            if not img_path.exists():
                logger.warning("depth_aware: image not found seq=%d", m.seq)
                h_m, w_m = m.floor_mask.shape
                intrinsics_by_frame.append(default_intrinsics(w_m, h_m))
                continue

            try:
                import cv2
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    logger.warning("depth_aware: cv2.imread failed seq=%d", m.seq)
                    h_m, w_m = m.floor_mask.shape
                    intrinsics_by_frame.append(default_intrinsics(w_m, h_m))
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.warning("depth_aware: image load error seq=%d: %s", m.seq, e)
                h_m, w_m = m.floor_mask.shape
                intrinsics_by_frame.append(default_intrinsics(w_m, h_m))
                continue

            h, w = m.floor_mask.shape
            intrin = default_intrinsics(w, h)
            intrinsics_by_frame.append(intrin)

            pts, depth_relative, used_scale = await bp_step.run(
                image=rgb,
                floor_mask=m.floor_mask,
                pose_bytes=m.pose_matrix,
                image_w=w,
                image_h=h,
                z0=z0,
            )
            if depth_relative is not None:
                depth_arr_by_seq[m.seq] = depth_relative
            if used_scale is not None and used_scale > 0:
                initial_scales[idx] = float(used_scale)
            pts_by_frame[idx] = pts
            valid_frame_indices.append(idx)

            frac = (idx + 1) / max(len(all_masks), 1)
            await progress_sink(BuildStep.BACK_PROJECT, 0.25 + frac * 0.08)

        # Multi-view scale optimization (opt-in)
        optimized_scales: dict[int, float] | None = None
        if self._use_multiview_scale and self._sp_lg_runner is not None and depth_arr_by_seq:
            logger.info(
                "multiview_scale: start — window=%d frames=%d initial_ok=%d",
                self._multiview_window,
                len(all_masks),
                len(initial_scales),
            )
            from indoor_server.application.building.steps.multiview_scale import (
                MultiViewScaleCalibrationStep,
            )

            mv_step = MultiViewScaleCalibrationStep(
                sp_lg_runner=self._sp_lg_runner,
                window=self._multiview_window,
            )

            async def _mv_progress(p: float) -> None:
                await progress_sink(BuildStep.BACK_PROJECT, 0.33 + p * 0.04)

            match_result = await mv_step.extract_matches(
                masks=all_masks,
                depth_maps_by_seq=depth_arr_by_seq,
                storage_root=self._storage_root,
                intrinsics_by_frame=intrinsics_by_frame,
                progress=_mv_progress,
            )
            optimized_scales = mv_step.optimize_scales(
                match_result=match_result,
                initial_scales=initial_scales,
                n_frames=len(all_masks),
            )

            # Pass 2: re-back-project with optimized scales
            pts_by_frame.clear()
            for idx, m in enumerate(all_masks):
                if await cancel_check():
                    break
                if m.seq not in depth_arr_by_seq:
                    continue
                scale_opt = optimized_scales.get(idx)
                if scale_opt is None or scale_opt <= 0:
                    continue

                # depth + pose + mask는 pass 1에서 있던 것 재사용 (depth_arr_by_seq)
                h, w = m.floor_mask.shape
                pts_2, _dmap, _scale_used = await bp_step.run(
                    image=np.zeros((h, w, 3), dtype=np.uint8),  # not used, depth precomputed
                    floor_mask=m.floor_mask,
                    pose_bytes=m.pose_matrix,
                    image_w=w,
                    image_h=h,
                    z0=z0,
                    override_scale=scale_opt,
                    precomputed_depth=depth_arr_by_seq[m.seq],
                )
                pts_by_frame[idx] = pts_2

                frac = (idx + 1) / max(len(all_masks), 1)
                await progress_sink(BuildStep.BACK_PROJECT, 0.37 + frac * 0.03)

        # DepthMap 레코드 구성
        depth_maps_collected: list[DepthMap] = []
        for m in all_masks:
            if m.seq not in depth_arr_by_seq:
                continue
            idx = next(i for i, mm in enumerate(all_masks) if mm.seq == m.seq)
            used = (
                optimized_scales.get(idx)
                if optimized_scales is not None
                else initial_scales.get(idx)
            )
            pts_len = int(pts_by_frame.get(idx, np.zeros((0, 3))).shape[0])
            depth_maps_collected.append(
                DepthMap(
                    seq=m.seq,
                    depth_relative=depth_arr_by_seq[m.seq],
                    scale=used,
                    valid_pixel_count=pts_len,
                )
            )

        empty = np.zeros((0, 3), dtype=np.float64)
        all_points = [pts_by_frame.get(i, empty) for i in range(len(all_masks))]

        def _points_gen() -> Generator[np.ndarray, None, None]:
            yield from all_points

        trajectory_pts = [(m.tx, m.ty) for m in all_masks]
        grid: WalkableGrid = await asyncio.to_thread(
            lambda: grid_step.run(_points_gen(), trajectory_pts, z0=z0)
        )

        walkable_cells = int(grid.mask.sum())
        logger.info(
            "depth_aware walkable_grid done cells=%d z0=%.3f frames=%d multiview=%s",
            walkable_cells,
            z0,
            len(all_points),
            "on" if optimized_scales is not None else "off",
        )
        await progress_sink(BuildStep.WALKABLE_GRID, 0.40)
        return z0, grid, walkable_cells, depth_maps_collected

    async def _emit_depth_aware_layout(
        self,
        sink: BuildDebugSink,
        grid: WalkableGrid,
        z0: float,
        all_masks: list[KeyframeMasks],
    ) -> None:
        """walkable_grid → DepthAwareLayout 생성 → sink.on_depth_aware_layout."""
        from indoor_server.application.building.steps.floor_layout import FloorLayoutFromGridStep
        from indoor_server.domain.building.models import DepthAwareLayout

        raster_step = FloorLayoutFromGridStep()
        floor_layout = await asyncio.to_thread(raster_step.run, grid, z0)
        trajectory_pts = [(m.tx, m.ty) for m in all_masks]

        depth_aware_layout = DepthAwareLayout(
            multipolygon=floor_layout.multipolygon,
            z0=floor_layout.z0,
            sub_polygon_count=floor_layout.sub_polygon_count,
            total_area_m2=floor_layout.total_area_m2,
        )
        sink.on_depth_aware_layout(depth_aware_layout, trajectory_pts)

    def _run_skeletonize(
        self,
        step: SkeletonizeStep,
        grid: WalkableGrid,
    ) -> tuple[SkeletonGraph, np.ndarray]:
        """
        SkeletonizeStep.run()을 실행하고 (SkeletonGraph, skeleton_bool_mask) 반환.
        skeleton mask는 composite/debug용으로만 사용 (DB 미저장).
        medial_axis를 한 번만 호출하기 위해 SkeletonizeStep 내부 결과를 재활용.
        """
        from skimage.morphology import medial_axis as _medial_axis

        result: tuple[np.ndarray, np.ndarray] = _medial_axis(  # type: ignore[no-untyped-call]
            grid.mask, return_distance=True
        )
        skeleton_bool: np.ndarray = result[0]
        skel_count = int(skeleton_bool.sum())

        # 빈 skeleton — _extract_topology 의 next(iter(empty)) 가 StopIteration 을
        # 던지면서 to_thread future 가 깨지는 것을 방지(SkeletonizeStep.run 과 동일한
        # 조기 종료 보강).
        if skel_count == 0:
            return (
                SkeletonGraph(nodes=[], edges=[], skeleton_pixel_count=0),
                skeleton_bool,
            )

        # SkeletonGraph는 skeleton_bool에서 graph/topology 추출
        graph = step._build_graph(skeleton_bool)
        nodes_topo, edges_topo = step._extract_topology(graph)
        skel_graph = SkeletonGraph(
            nodes=nodes_topo,
            edges=edges_topo,
            skeleton_pixel_count=skel_count,
        )
        return skel_graph, skeleton_bool

    def _cancelled_outcome(
        self,
        keyframes_processed: int = 0,
        walkable_cells: int = 0,
        skeleton_pixels: int = 0,
    ) -> BuildOutcome:
        return BuildOutcome(
            nodes=[],
            edges=[],
            poi_world_poses={},
            counts=BuildCounts(
                keyframes_processed=keyframes_processed,
                walkable_cells=walkable_cells,
                skeleton_pixels=skeleton_pixels,
            ),
            passed_quality_gate=False,
            failure_reason=BuildFailureReason.INTERNAL,
        )


def _resolve_build_source(
    *,
    use_floor_pointcloud: bool,
    use_rtabmap_trajectory: bool,
) -> str | None:
    """Sprint 46: build_source 라벨 결정. 우선순위 fp > traj > None."""
    if use_floor_pointcloud:
        return "floor_pointcloud"
    if use_rtabmap_trajectory:
        return "rtabmap_trajectory"
    return None


def _compute_cad_effect_pass(
    *,
    rectification_meta: dict[str, object] | None,
    cleanup_meta: dict[str, object] | None,
) -> tuple[bool | None, bool | None]:
    """Codex F-2: cad_effect_pass / cad_effect_small_polygon_pass 계산.

    cad_effect_pass:
      rectification.accepted is True
      AND fallback_used is False
      AND forced_rectilinear_used is True
      AND snap_mode_used in {"hint", "four_way"}
      AND polygon_cad_cleanup.input_source == "rectified"
      AND corner_orthogonality_ratio >= 0.90
      AND collinear_residual_count == 0
      AND (before >= 16 → after <= floor(before * 0.5))

    before<16 은 cad_effect_pass 에서 제외, cad_effect_small_polygon_pass 분리:
      cleanup_changed=True AND after<=before AND ortho>=0.90.
    """
    if rectification_meta is None or cleanup_meta is None:
        return None, None

    rec = rectification_meta
    cl = cleanup_meta
    # Sprint 48 evidence fix: vertex reduction은 raw raster contour →
    # final cleaned polygon 전체 감소율로 측정한다. cleanup_meta의
    # vertex_count_before는 "rectification 후 cleanup 직전" 값이라 rectification
    # 단계에서 발생한 큰 감소(57→10 같은)를 metric에서 빠뜨린다.
    raw_before_raw = rec.get("original_vertex_count")
    cleanup_before_raw = cl.get("vertex_count_before", 0)
    before_raw = (
        raw_before_raw
        if isinstance(raw_before_raw, (int, float)) and raw_before_raw > 0
        else cleanup_before_raw
    )
    after_raw = cl.get("vertex_count_after", 0)
    ortho_raw = cl.get("corner_orthogonality_ratio", 0.0)
    collinear_raw = cl.get("collinear_residual_count", 0)
    before = int(before_raw) if isinstance(before_raw, (int, float)) else 0
    after = int(after_raw) if isinstance(after_raw, (int, float)) else 0
    ortho = float(ortho_raw) if isinstance(ortho_raw, (int, float)) else 0.0
    collinear_residual = (
        int(collinear_raw) if isinstance(collinear_raw, (int, float)) else 0
    )
    cleanup_changed = bool(cl.get("cleanup_changed", False))
    input_source = cl.get("input_source")
    accepted = bool(rec.get("accepted") is True)
    fallback_used = bool(rec.get("fallback_used") is True)
    forced_rectilinear_used = bool(rec.get("forced_rectilinear_used") is True)
    snap_mode_used = rec.get("snap_mode_used")

    # 정상 polygon (>= 16 vertex) — strict pass
    base_required = (
        accepted
        and not fallback_used
        and forced_rectilinear_used
        and snap_mode_used in ("hint", "four_way")
        and input_source == "rectified"
        and ortho >= 0.90
        and collinear_residual == 0
    )

    if before >= 16:
        cad_pass = bool(base_required and after <= int(before * 0.5))
        small_pass = False
    elif before > 0:
        cad_pass = False
        # small polygon 분기 (Codex W-4)
        small_pass = bool(
            cleanup_changed
            and after <= before
            and ortho >= 0.90
        )
    else:
        cad_pass = False
        small_pass = False

    return cad_pass, small_pass


# ── Sprint 49 (Codex BLOCKER 6 + W-1/W-2/W-3/W-4): hint chain helpers ────────


def _compute_corner_orthogonality(geojson: dict[str, object]) -> float:
    """polygon corner 의 직각 비율. polygon_cad_cleanup._count_corners 와 동일.

    raster contour 그대로 쓰는 baseline 도 측정 가능하도록 shapely 직접 사용.
    """
    try:
        from math import acos, degrees, sqrt

        from shapely.geometry import shape

        from indoor_server.application.building.steps.polygon_cad_cleanup import (
            _iter_polygons,
            _ring_to_open_pts,
        )

        geom = shape(geojson)
        polys = _iter_polygons(geom)
        if not polys:
            return 0.0
        right_angle = 0
        total = 0
        tol = 5.0
        for poly in polys:
            pts = _ring_to_open_pts(list(poly.exterior.coords))
            n = len(pts)
            if n < 3:
                continue
            for i in range(n):
                prev_pt = pts[(i - 1) % n]
                cur = pts[i]
                nxt = pts[(i + 1) % n]
                v1x = prev_pt[0] - cur[0]
                v1y = prev_pt[1] - cur[1]
                v2x = nxt[0] - cur[0]
                v2y = nxt[1] - cur[1]
                n1 = sqrt(v1x * v1x + v1y * v1y)
                n2 = sqrt(v2x * v2x + v2y * v2y)
                if n1 < 1e-9 or n2 < 1e-9:
                    continue
                dot = (v1x * v2x + v1y * v2y) / (n1 * n2)
                dot = max(-1.0, min(1.0, dot))
                ang = float(degrees(acos(dot)))
                total += 1
                if abs(ang - 90.0) <= tol:
                    right_angle += 1
        if total == 0:
            return 0.0
        return float(right_angle) / float(total)
    except Exception:
        return 0.0


def _vertex_reduction_ratio(rectification_meta: dict[str, object]) -> float:
    """raw vs rectified vertex count reduction 비율 (1 - rectified/raw)."""
    raw = rectification_meta.get("original_vertex_count")
    rectified = rectification_meta.get("rectified_vertex_count")
    if not isinstance(raw, (int, float)) or not isinstance(rectified, (int, float)):
        return 0.0
    if raw <= 0:
        return 0.0
    return float(1.0 - float(rectified) / float(raw))


def _shape_guard_breakdown(
    raw_geojson: dict[str, object] | None,
    rectified_geojson: dict[str, object],
) -> dict[str, object]:
    """Codex W-4: candidate 별 shape preservation guard breakdown."""
    if raw_geojson is None:
        return {
            "iou": 1.0,
            "centroid_shift_m": 0.0,
            "bbox_width_ratio": 1.0,
            "bbox_height_ratio": 1.0,
            "iou_pass": True,
            "centroid_pass": True,
            "bbox_pass": True,
        }
    try:
        from shapely.geometry import shape

        from indoor_server.application.building.steps.polygon_cad_cleanup import (
            _compute_bbox_change_ratio,
            _compute_centroid_shift,
            _compute_iou,
        )

        raw = shape(raw_geojson)
        rect = shape(rectified_geojson)
        iou = _compute_iou(raw, rect)
        centroid_shift = _compute_centroid_shift(raw, rect)
        bbox_w_ratio, bbox_h_ratio = _compute_bbox_change_ratio(raw, rect)
        return {
            "iou": float(iou),
            "centroid_shift_m": float(centroid_shift),
            "bbox_width_ratio": float(bbox_w_ratio),
            "bbox_height_ratio": float(bbox_h_ratio),
            # Sprint 49 hotfix: hint candidate(회전 + 직각화 적용)가 raw raster
            # contour와 자연적으로 IOU가 0.5~0.7대로 떨어진다. 0.7 임계는
            # baseline(raw vs raw, IOU=1.0)만 통과시키는 baseline-bias라 사용자
            # 가치(직각화)에 역행한다. Codex evaluator-design WARN-1 권고대로
            # `0.65, 1.35` bbox + IOU 0.50 으로 완화. centroid는 폴리곤 면적의
            # 길이 기준이라 1m 그대로 유지.
            "iou_pass": bool(iou >= 0.50),
            "centroid_pass": bool(centroid_shift <= 1.0),
            "bbox_pass": bool(0.65 <= bbox_w_ratio <= 1.35 and 0.65 <= bbox_h_ratio <= 1.35),
        }
    except Exception:
        return {
            "iou": 0.0,
            "centroid_shift_m": float("inf"),
            "bbox_width_ratio": 0.0,
            "bbox_height_ratio": 0.0,
            "iou_pass": False,
            "centroid_pass": False,
            "bbox_pass": False,
        }


def _shape_guard_pass(
    raw_geojson: dict[str, object] | None,
    rectified_geojson: dict[str, object],
) -> bool:
    breakdown = _shape_guard_breakdown(raw_geojson, rectified_geojson)
    return bool(
        breakdown.get("iou_pass")
        and breakdown.get("centroid_pass")
        and breakdown.get("bbox_pass")
    )


def _cad_effect_candidate_pass(
    *,
    accepted: bool,
    fallback_used: bool,
    forced_rectilinear_used: bool,
    snap_mode_used: str,
    orthogonality: float,
) -> bool:
    """candidate 가 strict cad_effect 기준을 만족하는지 (cleanup 전 단계 약식)."""
    return bool(
        accepted
        and not fallback_used
        and forced_rectilinear_used
        and snap_mode_used in ("hint", "four_way")
        and orthogonality >= 0.50
    )


def _candidate_score(
    *,
    orthogonality: float,
    area_change_ratio: float,
    vertex_reduction: float,
    accepted: bool,
    cad_effect_candidate_pass: bool,
    shape_guard_passed: bool,
) -> float:
    """Codex W-3: composite score.

    핵심 항목 - orthogonality (가중 5.0)
    tie-breaker - cad_effect_candidate_pass + vertex_reduction + shape_guard
    penalty - area_change_ratio (negative)
    """
    if not accepted:
        return -1.0
    score = 5.0 * float(orthogonality)
    score += 1.0 if cad_effect_candidate_pass else 0.0
    score += 0.5 if shape_guard_passed else 0.0
    score += 1.0 * float(vertex_reduction)
    score -= 0.5 * float(area_change_ratio)
    return float(score)


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (float, int, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _meta_int(meta: dict[str, object], key: str, default: int) -> int:
    """metadata dict 에서 int 값 안전 추출."""
    v = meta.get(key, default)
    if isinstance(v, (int, float)):
        return int(v)
    return int(default)
