"""WorkerLoop — SKIP LOCKED claim → build → terminal. SIGTERM graceful."""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from indoor_server.application.building.build_service import BuildService
from indoor_server.application.building.pipeline import BuildPipeline
from indoor_server.config import settings
from indoor_server.infrastructure.jobs.build_claimer import BuildJobClaimer
from indoor_server.infrastructure.jobs.stale_watchdog import StaleJobWatchdog
from indoor_server.infrastructure.ml.model_cache import ModelCache
from indoor_server.infrastructure.ml.protocol import SegmentationOutput, SemanticSegmenter
from indoor_server.infrastructure.ml.segformer_onnx import SegformerOnnxSegmenter

logger = logging.getLogger(__name__)


def _worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


def _resolve_wall_polygon_enabled(
    *,
    wall_polygon_enabled: bool,
    use_floor_pointcloud: bool,
    log: logging.Logger,
) -> bool:
    """Sprint 52 W-4: warn instead of silently disabling wall_polygon.

    Returns the effective `use_wall_polygon` flag and logs a single WARNING
    when the operator enabled wall_polygon without floor_pointcloud (which is
    a hard requirement for the obstacle source).
    """
    if wall_polygon_enabled and not use_floor_pointcloud:
        log.warning(
            "wall_polygon_enabled=true but use_floor_pointcloud=false — "
            "wall_polygon disabled silently. set INDOOR_FLOOR_POINTCLOUD_ENABLED=true "
            "to activate."
        )
        return False
    return wall_polygon_enabled and use_floor_pointcloud


class _UnusedSegmenter:
    async def segment(self, image: np.ndarray) -> SegmentationOutput:
        raise RuntimeError("segmenter is unavailable in rtabmap trajectory mode")


class WorkerLoop:
    def __init__(self) -> None:
        self._shutdown = False
        self._worker_id = _worker_id()
        self._watchdog_interval = 60  # seconds

    async def run_forever(self) -> None:
        logger.info("worker starting worker_id=%s", self._worker_id)

        # Sprint 46: floor_pointcloud_enabled 우선. 켜져 있으면 trajectory mode를 끈다.
        use_floor_pointcloud = settings.floor_pointcloud_enabled
        use_rtabmap_trajectory = (
            settings.rtabmap_trajectory_enabled and not use_floor_pointcloud
        )
        use_adaptive_buffer = (
            not use_rtabmap_trajectory and not use_floor_pointcloud
        )
        # floor_pointcloud는 image segmentation을 강제로 켠다 (BuildPipeline.__init__
        # 에서도 보정되지만 여기서 segmenter 인스턴스화를 결정해야 한다).
        rtabmap_image_segmentation_enabled = (
            settings.rtabmap_image_segmentation_enabled or use_floor_pointcloud
        )

        segmenter: SemanticSegmenter
        if (
            use_rtabmap_trajectory
            and not rtabmap_image_segmentation_enabled
            and not use_floor_pointcloud
        ):
            segmenter = _UnusedSegmenter()
        else:
            # 모델 캐시 확인 + 로드
            model_cache = ModelCache(
                settings.model_cache_dir,
                repo_id=settings.segformer_model_repo_id,
                filename=settings.segformer_model_filename,
            )
            model_path = await asyncio.to_thread(model_cache.ensure)
            segmenter = SegformerOnnxSegmenter(model_path)

        engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        pipeline = BuildPipeline(
            segmenter=segmenter,
            storage_root=settings.storage_root,
            use_adaptive_buffer=use_adaptive_buffer,
            use_rtabmap_trajectory=use_rtabmap_trajectory,
            rtabmap_trajectory_half_width_m=settings.rtabmap_trajectory_half_width_m,
            rtabmap_feature_evidence_enabled=settings.rtabmap_feature_evidence_enabled,
            rtabmap_rectilinear_cover_enabled=settings.rtabmap_rectilinear_cover_enabled,
            rtabmap_rectilinear_cover_rotated_grid_enabled=(
                settings.rtabmap_rectilinear_cover_rotated_grid_enabled
            ),
            rtabmap_depth_evidence_enabled=settings.rtabmap_depth_evidence_enabled,
            rtabmap_depth_vertical_tolerance_m=(
                settings.rtabmap_depth_vertical_tolerance_m
            ),
            rtabmap_image_segmentation_enabled=rtabmap_image_segmentation_enabled,
            rtabmap_image_orientation_mode=settings.rtabmap_image_orientation_mode,
            rtabmap_image_floor_mask_min_ratio=(
                settings.rtabmap_image_floor_mask_min_ratio
            ),
            rtabmap_image_wall_mask_max_ratio=(
                settings.rtabmap_image_wall_mask_max_ratio
            ),
            use_floor_pointcloud=use_floor_pointcloud,
            floor_pointcloud_pixel_stride=settings.floor_pointcloud_pixel_stride,
            floor_pointcloud_height_tolerance_m=(
                settings.floor_pointcloud_height_tolerance_m
            ),
            floor_pointcloud_min_cell_hits=settings.floor_pointcloud_min_cell_hits,
            # Sprint 47: CAD cleanup config wiring
            polygon_cad_cleanup_enabled=settings.polygon_cad_cleanup_enabled,
            floor_raster_cad_morph_close_cells=(
                settings.floor_raster_cad_morph_close_cells
            ),
            manhattan_floor_pointcloud_max_area_change=(
                settings.manhattan_floor_pointcloud_max_area_change
            ),
            polygon_cad_collinear_angle_tol_deg=(
                settings.polygon_cad_collinear_angle_tol_deg
            ),
            polygon_cad_short_edge_min_length_m=(
                settings.polygon_cad_short_edge_min_length_m
            ),
            polygon_cad_near_vertex_merge_distance_m=(
                settings.polygon_cad_near_vertex_merge_distance_m
            ),
            polygon_cad_orthogonality_angle_tol_deg=(
                settings.polygon_cad_orthogonality_angle_tol_deg
            ),
            # Sprint 50 (Codex BLOCKER 4): production default OFF.
            use_rectangle_dictionary_cover=settings.rectangle_dictionary_cover_enabled,
            rectangle_cover_precision_threshold=(
                settings.rectangle_cover_precision_threshold
            ),
            rectangle_cover_recall_min=settings.rectangle_cover_recall_min,
            rectangle_cover_over_cover_max=settings.rectangle_cover_over_cover_max,
            rectangle_cover_time_budget_sec=(
                settings.rectangle_cover_time_budget_sec
            ),
            rectangle_cover_candidate_stride_cells=(
                settings.rectangle_cover_candidate_stride_cells
            ),
            rectangle_cover_max_candidates_per_dimension=(
                settings.rectangle_cover_max_candidates_per_dimension
            ),
            # Sprint 50 v2
            rectangle_cover_axes_mode=settings.rectangle_cover_axes_mode,
            rectangle_cover_precision_threshold_dynamic=(
                settings.rectangle_cover_precision_threshold_dynamic
            ),
            rectangle_cover_precision_threshold_min=(
                settings.rectangle_cover_precision_threshold_min
            ),
            rectangle_cover_min_component_cells=(
                settings.rectangle_cover_min_component_cells
            ),
            rectangle_cover_axis_link_min_best_bin_ratio=(
                settings.rectangle_cover_axis_link_min_best_bin_ratio
            ),
            rectangle_cover_axis_obb_min_aspect_ratio=(
                settings.rectangle_cover_axis_obb_min_aspect_ratio
            ),
            # Sprint 51: Wall-fitting polygon (default OFF). require floor_pointcloud.
            # Sprint 52 W-4: warn instead of silently disabling.
            use_wall_polygon=_resolve_wall_polygon_enabled(
                wall_polygon_enabled=settings.wall_polygon_enabled,
                use_floor_pointcloud=use_floor_pointcloud,
                log=logger,
            ),
            wall_polygon_density_min_cell_hits=(
                settings.wall_polygon_density_min_cell_hits
            ),
            wall_polygon_density_morph_close_radius_cells=(
                settings.wall_polygon_density_morph_close_radius_cells
            ),
            wall_polygon_components_min_area_cells=(
                settings.wall_polygon_components_min_area_cells
            ),
            wall_polygon_line_min_linearity=(
                settings.wall_polygon_line_min_linearity
            ),
            wall_polygon_line_min_length_m=(
                settings.wall_polygon_line_min_length_m
            ),
            wall_polygon_snap_tolerance_deg=(
                settings.wall_polygon_snap_tolerance_deg
            ),
            wall_polygon_merge_offset_tolerance_m=(
                settings.wall_polygon_merge_offset_tolerance_m
            ),
            wall_polygon_merge_gap_fill_m=(
                settings.wall_polygon_merge_gap_fill_m
            ),
            wall_polygon_assembly_intersection_tolerance_m=(
                settings.wall_polygon_assembly_intersection_tolerance_m
            ),
            wall_polygon_assembly_use_alpha_shape=(
                settings.wall_polygon_assembly_use_alpha_shape
            ),
            wall_polygon_validate_floor_iou_min=(
                settings.wall_polygon_validate_floor_iou_min
            ),
            wall_polygon_validate_area_change_max_ratio=(
                settings.wall_polygon_validate_area_change_max_ratio
            ),
            wall_polygon_min_lines=settings.wall_polygon_min_lines,
            wall_polygon_max_lines=settings.wall_polygon_max_lines,
            wall_polygon_obstacle_height_min_m=(
                settings.wall_polygon_obstacle_height_min_m
            ),
            wall_polygon_obstacle_height_max_m=(
                settings.wall_polygon_obstacle_height_max_m
            ),
            display_navigation_grid_enabled=(
                settings.display_navigation_grid_enabled
            ),
            display_navigation_grid_cell_m=settings.display_navigation_grid_cell_m,
            display_navigation_grid_clearance_m=(
                settings.display_navigation_grid_clearance_m
            ),
            display_navigation_grid_connectivity=(
                settings.display_navigation_grid_connectivity
            ),
            display_navigation_grid_poi_attach_k=(
                settings.display_navigation_grid_poi_attach_k
            ),
        )
        debug_root = settings.debug_root if settings.build_debug_dump else None
        build_service = BuildService(engine=engine, pipeline=pipeline, debug_root=debug_root)

        # SIGTERM 핸들러
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGTERM, self._handle_sigterm)
        loop.add_signal_handler(signal.SIGINT, self._handle_sigterm)

        # startup stale recover
        async with AsyncSession(engine) as session:
            async with session.begin():
                recovered = await StaleJobWatchdog(
                    session, settings.build_job_stale_after_sec
                ).recover_stale()
        logger.info("startup stale recovery recovered=%d", recovered)

        last_watchdog = asyncio.get_event_loop().time()

        while not self._shutdown:
            # 주기 watchdog
            now = asyncio.get_event_loop().time()
            if now - last_watchdog >= self._watchdog_interval:
                async with AsyncSession(engine) as session:
                    async with session.begin():
                        await StaleJobWatchdog(
                            session, settings.build_job_stale_after_sec
                        ).recover_stale()
                last_watchdog = now

            # claim
            async with AsyncSession(engine) as session:
                async with session.begin():
                    job = await BuildJobClaimer(session).claim_one(self._worker_id)

            if job is None:
                await asyncio.sleep(settings.build_worker_poll_interval_sec)
                continue

            logger.info(
                "job claimed build_job_id=%s scan_id=%s",
                job.build_job_id,
                job.scan_id,
            )
            await build_service.run(
                scan_id=str(job.scan_id),
                build_job_id=str(job.build_job_id),
            )

        logger.info("worker shutting down gracefully worker_id=%s", self._worker_id)

    def _handle_sigterm(self) -> None:
        logger.info("SIGTERM received — will stop after current job")
        self._shutdown = True
