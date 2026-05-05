"""Run wall/obstacle heatmap evidence on a real RTAB-Map database.

This is a Sprint 54 evidence harness for the wall-polygon path. Unlike
`sprint51_wall_polygon_evidence.py`, it does not use a synthetic heatmap:

  RTABMap Data.image -> Segformer floor/wall mask
  RTABMap Data.depth + calibration + Node.pose -> floor point cloud + z0
  wall mask + depth + pose -> obstacle heatmap
  obstacle heatmap -> wall-polygon facade + 7 evidence PNGs

Usage:
    uv run python scripts/run_real_wall_polygon_evidence.py \
      /path/to/rtabmap.db \
      --out-dir ../_workspace/sprint_54_server_wall_obs/evidence/real_wall_polygon
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # server/
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from indoor_server.application.building.steps.floor_point_cloud import (  # noqa: E402
    FloorPointCloudStep,
    FloorPointCloudStepParams,
)
from indoor_server.application.building.steps.floor_raster import (  # noqa: E402
    FloorRasterStep,
    FloorRasterStepParams,
)
from indoor_server.application.building.steps.node_placement import (  # noqa: E402
    NodePlacementStep,
)
from indoor_server.application.building.steps.rtabmap_image_evidence import (  # noqa: E402
    RtabmapImageEvidenceStep,
)
from indoor_server.application.building.steps.rtabmap_trajectory import (  # noqa: E402
    RtabmapTrajectoryRoadStep,
)
from indoor_server.application.building.steps.skeletonize import (  # noqa: E402
    SkeletonizeStep,
)
from indoor_server.application.building.steps.wall_polygon import (  # noqa: E402
    DensityRefineParams,
    HeatmapBoundaryParams,
    HeatmapBoundaryResult,
    HeatmapBoundaryStep,
    ObstacleSourceStepParams,
    WallInteriorFillParams,
    WallInteriorFillStep,
    WallPolygonFromObstacleStep,
    WallPolygonStepParams,
)
from indoor_server.application.building.steps.wall_polygon.evidence import (  # noqa: E402
    render_wall_polygon_evidence,
)
from indoor_server.application.building.steps.wall_polygon.obstacle_source import (  # noqa: E402
    ObstacleHeatmap,
)
from indoor_server.application.rtabmap.reader import RtabmapReader  # noqa: E402
from indoor_server.config import settings  # noqa: E402
from indoor_server.infrastructure.ml.model_cache import ModelCache  # noqa: E402
from indoor_server.infrastructure.ml.segformer_onnx import (  # noqa: E402
    SegformerOnnxSegmenter,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rtabmap_db", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--image-limit", type=int, default=None)
    parser.add_argument(
        "--image-orientation",
        choices=["sensor", "rotate_cw_90", "rotate_ccw_90", "rotate_180", "auto"],
        default="auto",
    )
    parser.add_argument("--floor-mask-min-ratio", type=float, default=0.0)
    parser.add_argument("--wall-mask-max-ratio", type=float, default=1.0)
    parser.add_argument(
        "--obstacle-mask-source",
        choices=["inverse_floor", "wall", "nonwalkable"],
        default="wall",
        help=(
            "Obstacle heatmap mask source. wall/nonwalkable use direct segmentation "
            "mask pixels; inverse_floor keeps the legacy non-floor path."
        ),
    )
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--cell-size-m", type=float, default=0.10)
    parser.add_argument("--min-cell-hits", type=int, default=4)
    parser.add_argument("--morph-close-radius-cells", type=int, default=1)
    parser.add_argument("--boundary-min-cell-hits", type=int, default=8)
    parser.add_argument("--boundary-gap-radius-cells", type=int, default=3)
    parser.add_argument("--boundary-close-radius-cells", type=int, default=1)
    parser.add_argument("--boundary-keep-components", type=int, default=3)
    parser.add_argument("--boundary-simplify-tolerance-m", type=float, default=0.05)
    parser.add_argument("--interior-wall-dilate-radius-cells", type=int, default=2)
    parser.add_argument("--interior-support-dilate-radius-cells", type=int, default=8)
    parser.add_argument("--interior-seed-radius-cells", type=int, default=4)
    parser.add_argument("--interior-close-radius-cells", type=int, default=3)
    parser.add_argument("--interior-rectilinear-area-change-limit", type=float, default=0.65)
    parser.add_argument("--disable-interior-rectilinear", action="store_true")
    parser.add_argument("--display-road-half-width-m", type=float, default=0.25)
    parser.add_argument("--display-max-edge-length-m", type=float, default=4.0)
    parser.add_argument("--height-min-m", type=float, default=0.30)
    parser.add_argument("--height-max-m", type=float, default=2.50)
    args = parser.parse_args()

    report = asyncio.run(_run(args))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = RtabmapReader()
    nodes = reader.load_nodes(args.rtabmap_db)
    links = reader.load_links(args.rtabmap_db)
    frames = reader.load_data_frames(args.rtabmap_db, limit=args.image_limit)

    model_cache = ModelCache(
        settings.model_cache_dir,
        repo_id=settings.segformer_model_repo_id,
        filename=settings.segformer_model_filename,
    )
    segmenter = SegformerOnnxSegmenter(model_cache.ensure())

    image_evidence = await RtabmapImageEvidenceStep().run(
        frames=frames,
        segmenter=segmenter,
        node_pose_ids={node.node_id for node in nodes},
        orientation_mode=args.image_orientation,
        floor_mask_min_ratio=args.floor_mask_min_ratio,
        wall_mask_max_ratio=args.wall_mask_max_ratio,
    )
    floor_masks = image_evidence.floor_masks_by_node_id
    obstacle_masks: dict[int, np.ndarray] | None = None
    obstacle_mask_mode: Literal["inverse_floor", "direct_mask"] = "inverse_floor"
    if args.obstacle_mask_source == "wall":
        obstacle_masks = image_evidence.wall_masks_by_node_id
        obstacle_mask_mode = "direct_mask"
    elif args.obstacle_mask_source == "nonwalkable":
        obstacle_masks = image_evidence.nonwalkable_masks_by_node_id
        obstacle_mask_mode = "direct_mask"

    floor_cloud = await asyncio.to_thread(
        FloorPointCloudStep(
            FloorPointCloudStepParams(pixel_stride=args.pixel_stride)
        ).run,
        nodes=nodes,
        frames=frames,
        floor_masks_by_node_id=floor_masks,
    )
    floor_raster = await asyncio.to_thread(
        FloorRasterStep(
            FloorRasterStepParams(
                min_cell_hits=2,
                morph_close_radius_cells=5,
                keep_largest_component=False,
                min_component_area_m2=3.0,
            )
        ).run,
        floor_cloud,
    )

    params = WallPolygonStepParams(
        obstacle_source=ObstacleSourceStepParams(
            pixel_stride=args.pixel_stride,
            cell_size_m=args.cell_size_m,
            height_above_floor_min_m=args.height_min_m,
            height_above_floor_max_m=args.height_max_m,
            mask_mode=obstacle_mask_mode,
        ),
        density=DensityRefineParams(
            min_cell_hits=args.min_cell_hits,
            morph_close_radius_cells=args.morph_close_radius_cells,
        ),
    )
    wall_step = WallPolygonFromObstacleStep(params)
    result = await asyncio.to_thread(
        wall_step.run,
        nodes=nodes,
        frames=frames,
        floor_masks_by_node_id=floor_masks,
        obstacle_masks_by_node_id=obstacle_masks,
        z0=floor_cloud.z0,
        floor_polygon_geojson=floor_raster.footprint_geojson,
    )

    paths = render_wall_polygon_evidence(
        result,
        output_dir=out_dir,
        scan_id=args.rtabmap_db.parent.name or args.rtabmap_db.stem,
        floor_polygon_geojson=floor_raster.footprint_geojson,
    )

    heatmap = result.stage_outputs.get("obstacle_source")
    boundary_report: dict[str, object] | None = None
    interior_report: dict[str, object] | None = None
    display_graph_report: dict[str, object] | None = None
    if isinstance(heatmap, ObstacleHeatmap):
        heatmap_npz_path = out_dir / "obstacle_heatmap_counts.npz"
        np.savez_compressed(
            heatmap_npz_path,
            counts=np.asarray(heatmap.counts, dtype=np.int32),
            origin_x=np.asarray([heatmap.origin_x], dtype=np.float64),
            origin_y=np.asarray([heatmap.origin_y], dtype=np.float64),
            cell_size_m=np.asarray([heatmap.cell_size_m], dtype=np.float64),
        )
        paths["obstacle_heatmap_counts_npz"] = heatmap_npz_path
        obs_path = out_dir / "00_obs_heatmap_reference_style.png"
        _render_reference_style_heatmap(heatmap, obs_path)
        paths["00_obs_heatmap_reference_style"] = obs_path
        boundary = HeatmapBoundaryStep(
            HeatmapBoundaryParams(
                min_cell_hits=args.boundary_min_cell_hits,
                bridge_gap_radius_cells=args.boundary_gap_radius_cells,
                close_radius_cells=args.boundary_close_radius_cells,
                keep_largest_components=args.boundary_keep_components,
                simplify_tolerance_m=args.boundary_simplify_tolerance_m,
            )
        ).run(heatmap)
        boundary_report = {
            "accepted": boundary.accepted,
            "fail_reason": boundary.fail_reason,
            **boundary.metadata,
        }
        boundary_metrics_path = out_dir / "heatmap_boundary_metrics.json"
        boundary_metrics_path.write_text(
            json.dumps(boundary_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths["heatmap_boundary_metrics"] = boundary_metrics_path
        if boundary.boundary_geojson is not None:
            boundary_geojson_path = out_dir / "heatmap_boundary.geojson"
            _write_geometry_geojson(
                boundary_geojson_path,
                geometry=boundary.boundary_geojson,
                properties=boundary.metadata,
            )
            paths["heatmap_boundary_geojson"] = boundary_geojson_path
        boundary_overlay_path = out_dir / "08_heatmap_boundary_overlay.png"
        _render_heatmap_boundary_overlay(heatmap, boundary, boundary_overlay_path)
        paths["08_heatmap_boundary_overlay"] = boundary_overlay_path
        boundary_masks_path = out_dir / "09_heatmap_boundary_masks.png"
        _render_heatmap_boundary_masks(boundary, boundary_masks_path)
        paths["09_heatmap_boundary_masks"] = boundary_masks_path

        trajectory_road_for_interior = RtabmapTrajectoryRoadStep(
            half_width_m=args.display_road_half_width_m,
            cell_size_m=floor_raster.grid.origin.cell_size,
            rectify_centerline=True,
            centerline_simplify_tolerance_m=0.35,
            centerline_min_segment_m=0.25,
        ).run(nodes=nodes, links=links)
        seed_points_xy = _extract_rectified_centerline_points(
            trajectory_road_for_interior.metadata
        )
        dominant_angle_raw = trajectory_road_for_interior.metadata.get(
            "dominant_angle_deg"
        )
        dominant_angle_deg = (
            float(dominant_angle_raw)
            if isinstance(dominant_angle_raw, int | float)
            else None
        )
        interior = WallInteriorFillStep(
            WallInteriorFillParams(
                wall_min_cell_hits=args.boundary_min_cell_hits,
                wall_bridge_gap_radius_cells=args.boundary_gap_radius_cells,
                wall_close_radius_cells=args.boundary_close_radius_cells,
                wall_dilate_radius_cells=args.interior_wall_dilate_radius_cells,
                support_dilate_radius_cells=args.interior_support_dilate_radius_cells,
                seed_radius_cells=args.interior_seed_radius_cells,
                interior_close_radius_cells=args.interior_close_radius_cells,
                rectilinear_enabled=not args.disable_interior_rectilinear,
                rectilinear_area_change_limit=(
                    args.interior_rectilinear_area_change_limit
                ),
            )
        ).run(
            heatmap=heatmap,
            seed_points_xy=seed_points_xy,
            support_grid=floor_raster.grid,
            dominant_angle_hint_deg=dominant_angle_deg,
        )
        interior_report = {
            "accepted": interior.accepted,
            "fail_reason": interior.fail_reason,
            **interior.metadata,
        }
        interior_metrics_path = out_dir / "wall_interior_metrics.json"
        interior_metrics_path.write_text(
            json.dumps(interior_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths["wall_interior_metrics"] = interior_metrics_path
        if interior.interior_geojson is not None:
            interior_geojson_path = out_dir / "wall_interior.geojson"
            _write_geometry_geojson(
                interior_geojson_path,
                geometry=interior.interior_geojson,
                properties=interior.metadata,
            )
            paths["wall_interior_geojson"] = interior_geojson_path
        interior_overlay_path = out_dir / "11_wall_interior_overlay.png"
        _render_wall_interior_overlay(
            heatmap=heatmap,
            interior=interior,
            seed_points_xy=seed_points_xy,
            out_path=interior_overlay_path,
        )
        paths["11_wall_interior_overlay"] = interior_overlay_path
        display_graph_report = _render_display_graph_overlay(
            heatmap=heatmap,
            floor_grid=floor_raster.grid,
            nodes=nodes,
            links=links,
            out_path=out_dir / "10_display_graph_overlay.png",
            scan_id=_uuid_from_text(args.rtabmap_db.parent.name),
            road_half_width_m=args.display_road_half_width_m,
            max_edge_length_m=args.display_max_edge_length_m,
        )
        paths["10_display_graph_overlay"] = out_dir / "10_display_graph_overlay.png"
        display_graph_metrics_path = out_dir / "display_graph_metrics.json"
        display_graph_metrics_path.write_text(
            json.dumps(display_graph_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths["display_graph_metrics"] = display_graph_metrics_path
        final_map_path = out_dir / "12_final_map_candidate.png"
        _render_wall_interior_final_map(
            heatmap=heatmap,
            interior=interior,
            seed_points_xy=seed_points_xy,
            out_path=final_map_path,
        )
        paths["12_final_map_candidate"] = final_map_path

    image_meta = _compact_image_evidence_metadata(image_evidence.metadata())
    floor_raster_meta = _compact_floor_raster_metadata(floor_raster.metadata)
    report: dict[str, Any] = {
        "db_path": str(args.rtabmap_db),
        "node_count": len(nodes),
        "link_count": len(links),
        "frame_count": len(frames),
        "obstacle_mask_source": args.obstacle_mask_source,
        "image_evidence": image_meta,
        "floor_pointcloud": floor_cloud.metadata,
        "floor_raster": floor_raster_meta,
        "wall_polygon": {
            "accepted": result.accepted,
            "fail_reason": result.fail_reason,
            "line_count": result.metadata.get("line_count"),
            "vertex_count": result.metadata.get("vertex_count"),
            "corner_orthogonality_ratio": result.metadata.get(
                "corner_orthogonality_ratio"
            ),
            "iou_with_floor": result.metadata.get("iou_with_floor"),
            "area_change_ratio": result.metadata.get("area_change_ratio"),
            "fallback_used": result.metadata.get("fallback_used"),
            "stages": result.metadata.get("stages", {}),
        },
        "heatmap_boundary": boundary_report,
        "wall_interior": interior_report,
        "display_graph": display_graph_report,
        "paths": {key: str(path) for key, path in sorted(paths.items())},
    }
    report_path = out_dir / "real_wall_polygon_evidence_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _compact_image_evidence_metadata(meta: dict[str, object]) -> dict[str, object]:
    """Keep the evidence report readable; the PNG/NPZ artifacts hold raw detail."""
    keys = [
        "frame_count",
        "decoded_count",
        "segmented_count",
        "floor_ratio_mean",
        "wall_ratio_mean",
        "stair_ratio_mean",
        "object_ratio_mean",
        "nonwalkable_ratio_mean",
        "floor_mask_node_count",
        "wall_mask_node_count",
        "stair_mask_node_count",
        "object_mask_node_count",
        "nonwalkable_mask_node_count",
        "orientation_mode",
        "selected_orientation_counts",
        "floor_mask_min_ratio",
        "wall_mask_max_ratio",
        "used_floor_ratio_mean",
        "used_wall_ratio_mean",
        "used_stair_ratio_mean",
        "used_object_ratio_mean",
        "used_nonwalkable_ratio_mean",
        "issues",
    ]
    return {key: meta.get(key) for key in keys}


def _compact_floor_raster_metadata(meta: dict[str, object]) -> dict[str, object]:
    """Drop heavy GeoJSON echo from the script-level report."""
    compact = dict(meta)
    compact.pop("footprint_geojson", None)
    return compact


def _write_geometry_geojson(
    out_path: Path,
    *,
    geometry: dict[str, object],
    properties: dict[str, object],
) -> None:
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": properties,
            }
        ],
    }
    out_path.write_text(
        json.dumps(feature_collection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _render_reference_style_heatmap(
    heatmap: ObstacleHeatmap,
    out_path: Path,
) -> None:
    """Render an obstacle heatmap with a dark, Habitat-like visual style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.asarray(heatmap.counts, dtype=np.float64)
    masked = np.ma.masked_where(counts <= 0, np.log1p(counts))

    fig, ax = plt.subplots(figsize=(7.0, 6.0), facecolor="black")
    ax.set_facecolor("black")
    # Use image-style coordinates so hallway bands visually match common
    # occupancy-map / Habitat "Obs Heatmap" examples: row 0 is at the top.
    ax.imshow(masked, cmap="inferno", origin="upper", interpolation="nearest")
    ax.set_title(
        f"Obs Heatmap · points={heatmap.metadata.get('world_obstacle_point_count', 0)} "
        f"cells={int(np.count_nonzero(counts))}",
        color="black",
        fontsize=11,
        pad=6,
        backgroundcolor="white",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _render_heatmap_boundary_overlay(
    heatmap: ObstacleHeatmap,
    boundary: HeatmapBoundaryResult,
    out_path: Path,
) -> None:
    """Render the accepted boundary directly on top of the obstacle heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.asarray(heatmap.counts, dtype=np.float64)
    masked = np.ma.masked_where(counts <= 0, np.log1p(counts))

    fig, ax = plt.subplots(figsize=(7.0, 6.0), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(masked, cmap="inferno", origin="upper", interpolation="nearest")

    boundary_overlay = np.zeros((*boundary.boundary_mask.shape, 4), dtype=np.float32)
    boundary_overlay[boundary.boundary_mask] = (0.0, 0.85, 1.0, 0.22)
    ax.imshow(boundary_overlay, origin="upper", interpolation="nearest")

    if boundary.boundary_geojson is not None:
        for ring in _iter_exterior_rings(boundary.boundary_geojson):
            cols = [
                (float(x) - heatmap.origin_x) / heatmap.cell_size_m
                for x, _y in ring
            ]
            rows = [
                (float(y) - heatmap.origin_y) / heatmap.cell_size_m
                for _x, y in ring
            ]
            ax.plot(cols, rows, color="#00e5ff", linewidth=1.6)

    title = "08 Boundary overlay"
    if not boundary.accepted:
        title += f" · FAIL {boundary.fail_reason}"
    else:
        title += (
            f" · area={boundary.metadata.get('polygon_area_m2', 0):.1f}m2"
            f" vertices={boundary.metadata.get('polygon_vertex_count', 0)}"
        )
    ax.set_title(
        title,
        color="black",
        fontsize=11,
        pad=6,
        backgroundcolor="white",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _render_heatmap_boundary_masks(
    boundary: HeatmapBoundaryResult,
    out_path: Path,
) -> None:
    """Render threshold mask vs gap-bridged final boundary mask."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), facecolor="white")
    panels = [
        ("threshold", boundary.threshold_mask),
        ("bridged boundary", boundary.boundary_mask),
    ]
    for ax, (title, mask) in zip(axes, panels, strict=True):
        ax.imshow(mask.astype(np.float32), cmap="gray_r", origin="upper")
        ax.set_title(f"{title} cells={int(mask.sum())}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"09 Boundary masks · accepted={boundary.accepted} "
        f"reason={boundary.fail_reason or 'ok'}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _render_wall_interior_overlay(
    *,
    heatmap: ObstacleHeatmap,
    interior: Any,
    seed_points_xy: list[tuple[float, float]],
    out_path: Path,
) -> None:
    """Render wall-constrained interior fill over the wall heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.asarray(heatmap.counts, dtype=np.float64)
    masked = np.ma.masked_where(counts <= 0, np.log1p(counts))

    fig, ax = plt.subplots(figsize=(7.0, 6.0), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(masked, cmap="inferno", origin="upper", interpolation="nearest")

    support_overlay = np.zeros((*interior.support_mask.shape, 4), dtype=np.float32)
    support_overlay[interior.support_mask] = (0.0, 0.95, 0.25, 0.10)
    ax.imshow(support_overlay, origin="upper", interpolation="nearest")

    wall_overlay = np.zeros((*interior.wall_mask.shape, 4), dtype=np.float32)
    wall_overlay[interior.wall_mask] = (1.0, 0.1, 0.0, 0.36)
    ax.imshow(wall_overlay, origin="upper", interpolation="nearest")

    fill_overlay = np.zeros((*interior.interior_mask.shape, 4), dtype=np.float32)
    fill_overlay[interior.interior_mask] = (0.0, 0.75, 1.0, 0.26)
    ax.imshow(fill_overlay, origin="upper", interpolation="nearest")

    if seed_points_xy:
        cols = [
            (point[0] - heatmap.origin_x) / heatmap.cell_size_m
            for point in seed_points_xy
        ]
        rows = [
            (point[1] - heatmap.origin_y) / heatmap.cell_size_m
            for point in seed_points_xy
        ]
        ax.plot(cols, rows, color="#40a9ff", linewidth=2.0, alpha=0.95)
        ax.scatter(cols, rows, s=14, color="#40a9ff", edgecolors="white", linewidths=0.4)

    if interior.interior_geojson is not None:
        for ring in _iter_exterior_rings(interior.interior_geojson):
            cols = [
                (float(x) - heatmap.origin_x) / heatmap.cell_size_m
                for x, _y in ring
            ]
            rows = [
                (float(y) - heatmap.origin_y) / heatmap.cell_size_m
                for _x, y in ring
            ]
            ax.plot(cols, rows, color="#ffe066", linewidth=2.2)

    title = "11 Wall-seeded interior"
    if not interior.accepted:
        title += f" · FAIL {interior.fail_reason}"
    else:
        title += (
            f" · area={interior.metadata.get('final_area_m2', 0):.1f}m2"
            f" vertices={interior.metadata.get('final_vertex_count', 0)}"
        )
    ax.set_title(
        title,
        color="black",
        fontsize=11,
        pad=6,
        backgroundcolor="white",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _render_display_graph_overlay(
    *,
    heatmap: ObstacleHeatmap,
    floor_grid: Any,
    nodes: list[Any],
    links: list[Any],
    out_path: Path,
    scan_id: UUID,
    road_half_width_m: float,
    max_edge_length_m: float,
) -> dict[str, object]:
    """Render a first-pass display graph over the obstacle heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trajectory_road = RtabmapTrajectoryRoadStep(
        half_width_m=road_half_width_m,
        cell_size_m=floor_grid.origin.cell_size,
        rectify_centerline=True,
        centerline_simplify_tolerance_m=0.35,
        centerline_min_segment_m=0.25,
    ).run(nodes=nodes, links=links)
    centerline_points = _extract_rectified_centerline_points(trajectory_road.metadata)
    graph_source = "rtabmap_trajectory_centerline"
    skeleton_pixel_count = 0
    if len(centerline_points) < 2:
        graph_grid = trajectory_road.grid if trajectory_road.grid.mask.any() else floor_grid
        graph_source = (
            "rtabmap_trajectory_rectified_skeleton"
            if trajectory_road.grid.mask.any()
            else "floor_raster_skeleton_fallback"
        )
        skeleton = SkeletonizeStep().run(graph_grid)
        graph_nodes, graph_edges = NodePlacementStep(
            scan_id=scan_id,
            build_job_id=uuid4(),
            max_edge_length_m=max_edge_length_m,
            force_rectilinear=True,
            dominant_angle_deg=None,
        ).run(skeleton, graph_grid.origin)
        skeleton_pixel_count = skeleton.skeleton_pixel_count
        graph_polylines = [edge.polyline for edge in graph_edges]
        graph_node_points = [(node.x, node.y) for node in graph_nodes]
        edge_lengths = [edge.length_m for edge in graph_edges]
    else:
        graph_polylines = [
            [
                (a[0], a[1], trajectory_road.z0),
                (b[0], b[1], trajectory_road.z0),
            ]
            for a, b in zip(centerline_points[:-1], centerline_points[1:], strict=True)
        ]
        graph_node_points = centerline_points
        edge_lengths = [
            float(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5)
            for a, b in zip(centerline_points[:-1], centerline_points[1:], strict=True)
        ]

    counts = np.asarray(heatmap.counts, dtype=np.float64)
    masked = np.ma.masked_where(counts <= 0, np.log1p(counts))
    fig, ax = plt.subplots(figsize=(7.0, 6.0), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(masked, cmap="inferno", origin="upper", interpolation="nearest")

    edge_obstacle_samples = 0
    edge_total_samples = 0
    for polyline in graph_polylines:
        cols = [
            (point[0] - heatmap.origin_x) / heatmap.cell_size_m
            for point in polyline
        ]
        rows = [
            (point[1] - heatmap.origin_y) / heatmap.cell_size_m
            for point in polyline
        ]
        ax.plot(cols, rows, color="#00ff6a", linewidth=1.8, alpha=0.9)
        hit, total = _count_polyline_heatmap_hits(heatmap, polyline)
        edge_obstacle_samples += hit
        edge_total_samples += total

    node_out_of_bounds = 0
    for x, y in graph_node_points:
        col = (x - heatmap.origin_x) / heatmap.cell_size_m
        row = (y - heatmap.origin_y) / heatmap.cell_size_m
        if row < 0 or col < 0 or row >= heatmap.shape[0] or col >= heatmap.shape[1]:
            node_out_of_bounds += 1
        ax.scatter(
            [col],
            [row],
            s=18,
            color="#40a9ff",
            edgecolors="white",
            linewidths=0.6,
        )

    hit_ratio = (
        edge_obstacle_samples / float(edge_total_samples)
        if edge_total_samples > 0
        else 0.0
    )
    ax.set_title(
        f"10 Display graph overlay · nodes={len(graph_node_points)} "
        f"edges={len(graph_polylines)} "
        f"wall-hit={hit_ratio:.2f}",
        color="black",
        fontsize=11,
        pad=6,
        backgroundcolor="white",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    return {
        "source": graph_source,
        "trajectory_road": trajectory_road.metadata,
        "skeleton_pixels": skeleton_pixel_count,
        "node_count": len(graph_node_points),
        "edge_count": len(graph_polylines),
        "node_out_of_heatmap_bounds": node_out_of_bounds,
        "edge_obstacle_sample_hit_ratio": hit_ratio,
        "edge_length_p50_m": float(np.percentile(edge_lengths, 50)) if edge_lengths else 0.0,
        "edge_length_p95_m": float(np.percentile(edge_lengths, 95)) if edge_lengths else 0.0,
    }


def _render_wall_interior_final_map(
    *,
    heatmap: ObstacleHeatmap,
    interior: Any,
    seed_points_xy: list[tuple[float, float]],
    out_path: Path,
) -> None:
    """Render only the final map candidate, without raw fill debug layers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.asarray(heatmap.counts, dtype=np.float64)
    masked = np.ma.masked_where(counts <= 0, np.log1p(counts))

    fig, ax = plt.subplots(figsize=(7.0, 6.0), facecolor="#f5f1df")
    ax.set_facecolor("#f5f1df")
    ax.imshow(
        masked,
        cmap="magma",
        origin="upper",
        interpolation="nearest",
        alpha=0.55,
    )

    if interior.interior_geojson is not None:
        for ring in _iter_exterior_rings(interior.interior_geojson):
            cols = [
                (float(x) - heatmap.origin_x) / heatmap.cell_size_m
                for x, _y in ring
            ]
            rows = [
                (float(y) - heatmap.origin_y) / heatmap.cell_size_m
                for _x, y in ring
            ]
            ax.fill(
                cols,
                rows,
                facecolor="#fffef7",
                edgecolor="#596056",
                linewidth=1.8,
                alpha=0.94,
                zorder=2,
            )

    if seed_points_xy:
        cols = [
            (point[0] - heatmap.origin_x) / heatmap.cell_size_m
            for point in seed_points_xy
        ]
        rows = [
            (point[1] - heatmap.origin_y) / heatmap.cell_size_m
            for point in seed_points_xy
        ]
        ax.plot(cols, rows, color="#0b7cff", linewidth=2.1, alpha=0.92, zorder=3)
        ax.scatter(
            cols,
            rows,
            s=18,
            color="#0b7cff",
            edgecolors="white",
            linewidths=0.6,
            zorder=4,
        )

    verdict = "PASS" if interior.accepted else f"FAIL {interior.fail_reason}"
    ax.set_title(
        "12 Final map candidate"
        f" · {verdict}"
        f" · source={interior.metadata.get('final_source', 'unknown')}",
        color="#202124",
        fontsize=10,
        pad=6,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _extract_rectified_centerline_points(
    metadata: dict[str, object],
) -> list[tuple[float, float]]:
    rectification = metadata.get("centerline_rectification")
    if not isinstance(rectification, dict):
        return []
    raw_points = rectification.get("rectified_points")
    if not isinstance(raw_points, list):
        return []
    points: list[tuple[float, float]] = []
    for item in raw_points:
        if isinstance(item, list | tuple) and len(item) >= 2:
            points.append((float(item[0]), float(item[1])))
    return points


def _count_polyline_heatmap_hits(
    heatmap: ObstacleHeatmap,
    polyline: list[tuple[float, float, float]],
) -> tuple[int, int]:
    if len(polyline) < 2:
        return 0, 0
    hits = 0
    total = 0
    for start, end in zip(polyline[:-1], polyline[1:], strict=True):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = float((dx * dx + dy * dy) ** 0.5)
        samples = max(2, int(length / max(heatmap.cell_size_m, 1e-6)))
        for idx in range(samples + 1):
            t = idx / float(samples)
            x = start[0] + dx * t
            y = start[1] + dy * t
            col = int((x - heatmap.origin_x) / heatmap.cell_size_m)
            row = int((y - heatmap.origin_y) / heatmap.cell_size_m)
            if 0 <= row < heatmap.shape[0] and 0 <= col < heatmap.shape[1]:
                total += 1
                if heatmap.counts[row, col] >= 8:
                    hits += 1
    return hits, total


def _uuid_from_text(text: str) -> UUID:
    try:
        return UUID(text)
    except ValueError:
        return uuid4()


def _iter_exterior_rings(
    geometry: dict[str, object],
) -> list[list[tuple[float, float]]]:
    geom_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    rings: list[list[tuple[float, float]]] = []
    if geom_type == "Polygon" and isinstance(coordinates, list):
        _append_polygon_exterior(rings, coordinates)
    elif geom_type == "MultiPolygon" and isinstance(coordinates, list):
        for polygon_coordinates in coordinates:
            if isinstance(polygon_coordinates, list):
                _append_polygon_exterior(rings, polygon_coordinates)
    return rings


def _append_polygon_exterior(
    rings: list[list[tuple[float, float]]],
    polygon_coordinates: list[object],
) -> None:
    if not polygon_coordinates:
        return
    exterior = polygon_coordinates[0]
    if not isinstance(exterior, list):
        return
    ring: list[tuple[float, float]] = []
    for pair in exterior:
        if isinstance(pair, list | tuple) and len(pair) >= 2:
            ring.append((float(pair[0]), float(pair[1])))
    if ring:
        rings.append(ring)


if __name__ == "__main__":
    raise SystemExit(main())
