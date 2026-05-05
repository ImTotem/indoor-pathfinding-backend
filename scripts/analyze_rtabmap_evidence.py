"""Inspect RTAB-Map feature/image evidence for map-building experiments.

Usage:
    uv run python scripts/analyze_rtabmap_evidence.py /path/to/rtabmap.db \
      --out-dir ../_workspace/sprint_39
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SERVER_ROOT = Path(__file__).resolve().parents[1]
_SRC = _SERVER_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from indoor_server.application.building.steps.rtabmap_image_evidence import (  # noqa: E402
    RtabmapImageEvidenceStep,
)
from indoor_server.application.building.steps.rtabmap_depth_evidence import (  # noqa: E402
    RtabmapDepthEvidenceStep,
)
from indoor_server.application.building.steps.rtabmap_floor_guard import (  # noqa: E402
    RtabmapFloorGuardStep,
)
from indoor_server.application.building.steps.rectilinear_cover import (  # noqa: E402
    RectilinearWalkableCoverStep,
)
from indoor_server.application.building.steps.rtabmap_trajectory import (  # noqa: E402
    RtabmapTrajectoryRoadStep,
    _FloorFrame,
)
from indoor_server.application.rtabmap.reader import RtabmapReader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rtabmap_db", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--image-limit", type=int, default=5)
    parser.add_argument("--feature-preview-limit", type=int, default=4000)
    parser.add_argument(
        "--segment-images",
        action="store_true",
        help="Run Segformer on RTAB-Map Data.image and feed masks into depth/cover evidence.",
    )
    parser.add_argument(
        "--feature-evidence",
        action="store_true",
        help=(
            "Use RTAB-Map Feature.depth_x/y/z to expand road half-width. "
            "Off by default because feature points are not semantic-floor filtered."
        ),
    )
    parser.add_argument(
        "--image-orientation",
        choices=["sensor", "rotate_cw_90", "rotate_ccw_90", "rotate_180", "auto"],
        default="sensor",
        help="Segformer input orientation policy. Masks are restored to sensor/depth coordinates.",
    )
    parser.add_argument("--floor-mask-min-ratio", type=float, default=0.0)
    parser.add_argument("--wall-mask-max-ratio", type=float, default=1.0)
    parser.add_argument("--depth-vertical-tolerance", type=float, default=0.35)
    parser.add_argument(
        "--allow-depth-without-floor-mask",
        action="store_true",
        help="Do not fail closed when segmentation is enabled but a node floor mask is suppressed/missing.",
    )
    parser.add_argument("--segmentation-preview-limit", type=int, default=12)
    parser.add_argument(
        "--rectilinear-cover-angle",
        type=float,
        default=None,
        help="Override dominant angle for rotated-grid rectangle cover. Defaults to RTAB-Map trajectory metadata.",
    )
    parser.add_argument(
        "--disable-rotated-rectilinear-cover",
        action="store_true",
        help="Use source-grid axis-aligned cover for before/after comparisons.",
    )
    args = parser.parse_args()

    reader = RtabmapReader()
    db_path = args.rtabmap_db
    nodes = reader.load_nodes(db_path)
    links = reader.load_links(db_path)
    features = reader.load_feature_points(db_path)
    data_frames = reader.load_data_frames(db_path, limit=args.image_limit)
    segmenter = None
    if args.segment_images:
        from indoor_server.config import settings  # noqa: E402
        from indoor_server.infrastructure.ml.model_cache import ModelCache  # noqa: E402
        from indoor_server.infrastructure.ml.segformer_onnx import (  # noqa: E402
            SegformerOnnxSegmenter,
        )

        model_cache = ModelCache(
            settings.model_cache_dir,
            repo_id=settings.segformer_model_repo_id,
            filename=settings.segformer_model_filename,
        )
        segmenter = SegformerOnnxSegmenter(model_cache.ensure())

    road = RtabmapTrajectoryRoadStep().run(
        nodes=nodes,
        links=links,
        features=features if args.feature_evidence else None,
    )
    image_evidence = asyncio.run(
        RtabmapImageEvidenceStep().run(
            data_frames,
            segmenter=segmenter,
            node_pose_ids={node.node_id for node in nodes},
            orientation_mode=args.image_orientation,
            floor_mask_min_ratio=args.floor_mask_min_ratio,
            wall_mask_max_ratio=args.wall_mask_max_ratio,
        )
    )
    avoid_masks_by_node_id = image_evidence.nonwalkable_masks_by_node_id
    depth_evidence = (
        RtabmapDepthEvidenceStep(
            vertical_tolerance_m=args.depth_vertical_tolerance,
            require_floor_mask=(
                args.segment_images and not args.allow_depth_without_floor_mask
            ),
        ).run(
            frames=data_frames,
            nodes=nodes,
            grid=road.grid,
            floor_masks_by_node_id=(
                image_evidence.floor_masks_by_node_id if args.segment_images else None
            ),
            avoid_masks_by_node_id=(
                avoid_masks_by_node_id if args.segment_images else None
            ),
        )
        if road.grid.mask.any()
        else None
    )
    floor_guard = (
        RtabmapFloorGuardStep().run(
            road.grid,
            confidence=(
                depth_evidence.confidence
                if depth_evidence is not None
                else None
            ),
            avoid=(
                depth_evidence.avoid
                if depth_evidence is not None
                else None
            ),
        )
        if road.grid.mask.any()
        else None
    )
    rectilinear_cover = (
        RectilinearWalkableCoverStep(
            dominant_angle_deg=(
                None
                if args.disable_rotated_rectilinear_cover
                else _cover_angle_override_or_metadata(
                    args.rectilinear_cover_angle,
                    road.metadata,
                )
            ),
        ).run(
            floor_guard.grid if floor_guard is not None else road.grid,
            confidence=(
                depth_evidence.confidence
                if depth_evidence is not None
                else None
            ),
            avoid=(
                depth_evidence.avoid
                if depth_evidence is not None
                else None
            ),
        )
        if road.grid.mask.any()
        else None
    )

    report: dict[str, Any] = {
        "db_path": str(db_path),
        "node_count": len(nodes),
        "link_count": len(links),
        "feature_count": len(features),
        "data_frame_sample_count": len(data_frames),
        "road": road.metadata,
        "rectilinear_cover": (
            rectilinear_cover.metadata if rectilinear_cover is not None else None
        ),
        "floor_guard": (
            floor_guard.metadata if floor_guard is not None else None
        ),
        "depth_evidence": (
            depth_evidence.metadata() if depth_evidence is not None else None
        ),
        "image_evidence": image_evidence.metadata(),
    }

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.out_dir / "rtabmap_evidence_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if road.footprint_geojson is not None:
            preview_path = args.out_dir / "rtabmap_feature_evidence_preview.png"
            _render_preview(
                nodes=nodes,
                links=links,
                features=features[: args.feature_preview_limit],
                footprint_geojson=road.footprint_geojson,
                metadata=road.metadata,
                out=preview_path,
            )
            report["preview_path"] = str(preview_path)
            if (
                rectilinear_cover is not None
                and rectilinear_cover.footprint_geojson is not None
            ):
                cover_preview_path = (
                    args.out_dir / "rtabmap_rectilinear_cover_preview.png"
                )
                _render_preview(
                    nodes=nodes,
                    links=links,
                    features=features[: args.feature_preview_limit],
                    footprint_geojson=rectilinear_cover.footprint_geojson,
                    metadata=rectilinear_cover.metadata,
                    out=cover_preview_path,
                )
                report["rectilinear_cover_preview_path"] = str(cover_preview_path)
            if (
                depth_evidence is not None
                and depth_evidence.confidence is not None
            ):
                confidence_path = args.out_dir / "rtabmap_depth_confidence_grid.png"
                _render_confidence_grid(
                    confidence=depth_evidence.confidence,
                    target=road.grid.mask,
                    out=confidence_path,
                    label="depth confidence",
                )
                report["depth_confidence_preview_path"] = str(confidence_path)
            if floor_guard is not None:
                guard_path = args.out_dir / "rtabmap_floor_guard_grid.png"
                _render_guard_grid(
                    source=road.grid.mask,
                    guarded=floor_guard.grid.mask,
                    confidence=(
                        depth_evidence.confidence
                        if depth_evidence is not None
                        else None
                    ),
                    avoid=(
                        depth_evidence.avoid
                        if depth_evidence is not None
                        else None
                    ),
                    out=guard_path,
                )
                report["floor_guard_preview_path"] = str(guard_path)
            if (
                depth_evidence is not None
                and depth_evidence.avoid is not None
            ):
                avoid_path = args.out_dir / "rtabmap_depth_avoid_grid.png"
                _render_confidence_grid(
                    confidence=depth_evidence.avoid,
                    target=road.grid.mask,
                    out=avoid_path,
                    label="depth avoid",
                )
                report["depth_avoid_preview_path"] = str(avoid_path)
            if args.segment_images:
                segmentation_path = args.out_dir / "rtabmap_segmentation_contact_sheet.png"
                _render_segmentation_contact_sheet(
                    frames=data_frames,
                    image_evidence=image_evidence,
                    out=segmentation_path,
                    limit=args.segmentation_preview_limit,
                )
                report["segmentation_contact_sheet_path"] = str(segmentation_path)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _render_preview(
    *,
    nodes: list[Any],
    links: list[Any],
    features: list[Any],
    footprint_geojson: dict[str, object],
    metadata: dict[str, object],
    out: Path,
) -> None:
    import cv2
    from shapely.geometry import MultiPolygon, Polygon, shape

    geom = shape(footprint_geojson)
    polygons = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
    frame = _FloorFrame.from_nodes(nodes)
    node_xy = {node.node_id: frame.project_node_xy(node) for node in nodes}
    feature_xy = _feature_xy(nodes=nodes, features=features, frame=frame)

    xs: list[float] = []
    ys: list[float] = []
    for polygon in polygons:
        min_x, min_y, max_x, max_y = polygon.bounds
        xs.extend([min_x, max_x])
        ys.extend([min_y, max_y])
    for x, y in feature_xy:
        xs.append(x)
        ys.append(y)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad_x = max((max_x - min_x) * 0.08, 0.5)
    pad_y = max((max_y - min_y) * 0.08, 0.5)
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    width, height = 1400, 1000
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)

    def to_px(point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        px = int(round((x - min_x) / (max_x - min_x) * (width - 1)))
        py = int(round((max_y - y) / (max_y - min_y) * (height - 1)))
        return px, py

    for polygon in polygons:
        _draw_polygon(canvas, polygon, to_px)
    if isinstance(geom, MultiPolygon):
        for polygon in geom.geoms:
            _draw_polygon(canvas, polygon, to_px)

    for link in links:
        if link.link_type != 0:
            continue
        if link.from_id not in node_xy or link.to_id not in node_xy:
            continue
        cv2.line(
            canvas,
            to_px(node_xy[link.from_id]),
            to_px(node_xy[link.to_id]),
            (30, 126, 243),
            4,
            cv2.LINE_AA,
        )

    for x, y in feature_xy:
        cv2.circle(canvas, to_px((x, y)), 1, (39, 174, 96), -1, cv2.LINE_AA)

    for xy in node_xy.values():
        cv2.circle(canvas, to_px(xy), 5, (50, 50, 50), -1, cv2.LINE_AA)

    half_width = metadata.get("resolved_half_width_m", metadata.get("half_width_m"))
    feature_meta = metadata.get("feature_evidence")
    lines = [
        f"nodes={len(nodes)} links={len(links)} features_preview={len(features)}",
    ]
    if half_width is not None:
        lines.append(f"resolved_half_width_m={half_width}")
    if metadata.get("source") == "rectilinear_rectangle_cover":
        lines.append(
            "rectilinear "
            f"rects={metadata.get('rectangle_count')} "
            f"coverage={metadata.get('coverage_ratio')} "
            f"over={metadata.get('overcovered_cells')}"
        )
    if isinstance(feature_meta, dict):
        lines.append(
            "feature p75="
            f"{feature_meta.get('distance_percentiles_m', {}).get('p75')}"
        )
    for idx, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (24, 44 + idx * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)


def _cover_angle_override_or_metadata(
    angle: float | None,
    metadata: dict[str, Any],
) -> float | None:
    if angle is not None:
        return angle
    value = metadata.get("dominant_angle_deg")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _render_segmentation_contact_sheet(
    *,
    frames: list[Any],
    image_evidence: Any,
    out: Path,
    limit: int,
) -> None:
    import cv2

    tiles: list[np.ndarray] = []
    frame_by_node = {frame.node_id: frame for frame in frames}
    frame_meta = {
        frame.node_id: frame
        for frame in image_evidence.frames
        if frame.floor_ratio is not None
    }
    for node_id in list(frame_meta)[:limit]:
        frame = frame_by_node.get(node_id)
        if frame is None or frame.image_bytes is None:
            continue
        image = _decode_rgb(frame.image_bytes)
        if image is None:
            continue
        floor = image_evidence.floor_masks_by_node_id.get(node_id)
        wall = image_evidence.wall_masks_by_node_id.get(node_id)
        stair = image_evidence.stair_masks_by_node_id.get(node_id)
        obj = image_evidence.object_masks_by_node_id.get(node_id)
        overlay = image.copy()
        if wall is not None:
            overlay[wall] = (255, 80, 80)
        if obj is not None:
            overlay[obj] = (200, 80, 255)
        if floor is not None:
            overlay[floor] = (40, 190, 90)
        if stair is not None:
            overlay[stair] = (255, 170, 40)
        blended = cv2.addWeighted(image, 0.55, overlay, 0.45, 0.0)
        meta = frame_meta[node_id]
        label = (
            f"node {node_id} {meta.selected_orientation or ''} "
            f"F {meta.floor_ratio:.2f} W {meta.wall_ratio:.2f}"
        )
        tile = cv2.resize(blended, (240, 180), interpolation=cv2.INTER_AREA)
        cv2.putText(
            tile,
            label,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            label,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    if not tiles:
        return
    cols = min(3, len(tiles))
    rows = int(np.ceil(len(tiles) / cols))
    canvas = np.full((rows * 180, cols * 240, 3), 245, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        canvas[row * 180:(row + 1) * 180, col * 240:(col + 1) * 240] = tile
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _decode_rgb(data: bytes) -> np.ndarray | None:
    import cv2

    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def _draw_polygon(
    canvas: np.ndarray,
    polygon: Any,
    to_px: Any,
) -> None:
    import cv2

    exterior = np.array([to_px((x, y)) for x, y in polygon.exterior.coords], dtype=np.int32)
    cv2.fillPoly(canvas, [exterior], color=(232, 232, 224))
    cv2.polylines(canvas, [exterior], isClosed=True, color=(160, 160, 150), thickness=2)
    for interior in polygon.interiors:
        hole = np.array([to_px((x, y)) for x, y in interior.coords], dtype=np.int32)
        cv2.fillPoly(canvas, [hole], color=(245, 245, 245))
        cv2.polylines(canvas, [hole], isClosed=True, color=(180, 180, 170), thickness=1)


def _render_confidence_grid(
    *,
    confidence: np.ndarray,
    target: np.ndarray,
    out: Path,
    label: str,
) -> None:
    import cv2

    if confidence.size == 0:
        return
    height, width = confidence.shape
    scale = max(1, min(12, int(round(900 / max(height, width)))))
    base = np.full((height, width, 3), 245, dtype=np.uint8)
    if np.max(confidence) > 0:
        heat = cv2.applyColorMap(
            np.clip(confidence * 255.0, 0, 255).astype(np.uint8),
            cv2.COLORMAP_TURBO,
        )
        mask = confidence > 0
        base[mask] = heat[mask]
    contours, _ = cv2.findContours(
        target.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(base, contours, -1, (40, 40, 40), 1)
    preview = cv2.resize(base, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
    cv2.putText(
        preview,
        f"{label} cells={int(np.count_nonzero(confidence))}",
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), preview)


def _render_guard_grid(
    *,
    source: np.ndarray,
    guarded: np.ndarray,
    confidence: np.ndarray | None,
    avoid: np.ndarray | None,
    out: Path,
) -> None:
    import cv2

    height, width = source.shape
    scale = max(1, min(12, int(round(900 / max(height, width)))))
    base = np.full((height, width, 3), 245, dtype=np.uint8)
    base[source.astype(bool)] = (220, 225, 225)
    if confidence is not None and confidence.shape == source.shape:
        base[confidence > 0] = (120, 220, 120)
    if avoid is not None and avoid.shape == source.shape:
        base[avoid > 0] = (180, 120, 220)
    base[guarded.astype(bool)] = (255, 255, 255)
    contours, _ = cv2.findContours(
        guarded.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(base, contours, -1, (20, 20, 20), 1)
    preview = cv2.resize(
        base,
        (width * scale, height * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.putText(
        preview,
        (
            f"floor guard cells={int(np.count_nonzero(guarded))} "
            f"from={int(np.count_nonzero(source))}"
        ),
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), preview)


def _feature_xy(
    *,
    nodes: list[Any],
    features: list[Any],
    frame: _FloorFrame,
) -> list[tuple[float, float]]:
    node_pose = {
        node.node_id: np.asarray(node.pose, dtype=np.float64)
        for node in nodes
    }
    points: list[tuple[float, float]] = []
    for feature in features:
        pose = node_pose.get(feature.node_id)
        if pose is None:
            continue
        local = np.asarray(
            [
                feature.local_xyz[0],
                feature.local_xyz[1],
                feature.local_xyz[2],
                1.0,
            ],
            dtype=np.float64,
        )
        world = pose @ local
        points.append(frame.project_xyz_xy(world[:3]))
    return points


if __name__ == "__main__":
    raise SystemExit(main())
