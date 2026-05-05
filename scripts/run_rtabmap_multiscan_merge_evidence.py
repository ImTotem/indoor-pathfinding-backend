"""RTAB-Map multi-scan merge evidence harness.

This is Sprint 75 Phase 0. It does not mutate uploaded scan folders. Source
RTAB-Map DBs are copied to an evidence working directory, provenance labels are
injected into the copies, then `rtabmap-reprocess -a "db1;db2"` is executed.

Usage:
    uv run python scripts/run_rtabmap_multiscan_merge_evidence.py \
      --scan-root /path/to/scanA \
      --scan-root /path/to/scanB \
      --output-dir ../_workspace/sprint_75_rtabmap_multiscan_pose_fusion/evidence/demo
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # server/
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from indoor_server.application.building.multiscan_pose_mapping import (  # noqa: E402
    build_node_pose_mapping,
)
from indoor_server.application.building.multiscan_rtabmap_merge import (  # noqa: E402
    MultiScanReprocessParams,
    MultiScanRtabmapReprocessRunner,
    SourceRtabmapScan,
)
from indoor_server.application.building.steps.back_projection import Intrinsics  # noqa: E402
from indoor_server.application.building.steps.floor_raster import (  # noqa: E402
    FloorRasterStep,
    FloorRasterStepParams,
)
from indoor_server.application.building.steps.multiscan_dense_video_polygon import (  # noqa: E402
    MultiScanDenseVideoPolygonParams,
    MultiScanDenseVideoPolygonStep,
    MultiScanDenseVideoSource,
)
from indoor_server.application.building.steps.multiscan_rtabmap_floor_polygon import (  # noqa: E402
    MultiScanRtabmapFloorPolygonParams,
    MultiScanRtabmapFloorPolygonStep,
    MultiScanRtabmapFloorSource,
)
from indoor_server.application.scan_manifest import parse_manifest_file  # noqa: E402
from indoor_server.config import settings  # noqa: E402
from indoor_server.infrastructure.ml.model_cache import ModelCache  # noqa: E402
from indoor_server.infrastructure.ml.segformer_onnx import SegformerOnnxSegmenter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan-root",
        action="append",
        type=Path,
        required=True,
        help="Scan directory containing rtabmap.db. Provide at least two.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rtabmap-reprocess-bin", type=str, default=None)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--stamp-tolerance-s", type=float, default=0.033)
    parser.add_argument("--feature-strategy", type=int, default=1)
    parser.add_argument("--loop-thr", type=float, default=None)
    parser.add_argument(
        "--skip-polygon",
        action="store_true",
        help="Only run RTAB-Map merge and node mapping; skip dense video floor polygon.",
    )
    parser.add_argument(
        "--polygon-source",
        choices=["auto", "video", "rtabmap"],
        default="auto",
        help="auto prefers video when all scans have scan.mp4+poses.bin, otherwise RTABMap Data.",
    )
    parser.add_argument("--polygon-z0", type=float, default=None)
    parser.add_argument("--polygon-stride", type=int, default=2)
    parser.add_argument("--polygon-pixel-stride", type=int, default=4)
    parser.add_argument("--polygon-min-cell-hits", type=int, default=2)
    parser.add_argument(
        "--polygon-source-timestamp-mode",
        choices=["raw_pose", "rebased_pose", "video"],
        default="raw_pose",
    )
    parser.add_argument(
        "--rtabmap-image-orientation-mode",
        choices=["sensor", "rotate_cw_90", "rotate_ccw_90", "rotate_180", "auto"],
        default="sensor",
    )
    parser.add_argument("--skip-wall-polygon", action="store_true")
    args = parser.parse_args()

    return asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> int:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    scan_roots: dict[str, Path] = {}
    source_scans: list[SourceRtabmapScan] = []
    for path in args.scan_root:
        source = _source_from_scan_root(path)
        scan_roots[source.scan_id] = path
        source_scans.append(source)
    params = MultiScanReprocessParams(
        feature_strategy=args.feature_strategy,
        extra_args=(
            (f"--Rtabmap/LoopThr={args.loop_thr}",)
            if args.loop_thr is not None
            else ()
        ),
    )
    runner = MultiScanRtabmapReprocessRunner(
        binary_path=args.rtabmap_reprocess_bin,
        default_timeout_s=args.timeout_s,
    )
    result = await runner.run(
        sources=source_scans,
        output_db=output_dir / "merged.db",
        work_dir=output_dir / "prepared_sources",
        params=params,
        timeout_s=args.timeout_s,
    )
    (output_dir / "rtabmap_reprocess.stdout.log").write_text(
        result.stdout_text,
        encoding="utf-8",
    )
    (output_dir / "rtabmap_reprocess.stderr.log").write_text(
        result.stderr_text,
        encoding="utf-8",
    )

    source_dbs = {
        scan.scan_id: scan.source_db_path
        for scan in result.prepared_scans
    }
    mapping = build_node_pose_mapping(
        source_dbs=source_dbs,
        merged_db=result.output_db_path,
        stamp_tolerance_s=args.stamp_tolerance_s,
    )
    _write_node_mapping_csv(output_dir / "node_mapping.csv", mapping)
    polygon_metrics = None
    if not args.skip_polygon:
        polygon_metrics = await _run_polygon_fusion(
            args=args,
            output_dir=output_dir,
            scan_roots=scan_roots,
            mapping=mapping,
            inter_session_loop_closure_count=result.loop_closure_count,
        )
    metrics = {
        "merge": result.to_metadata(),
        "pose_mapping": mapping.metrics.to_dict(),
        "polygon": polygon_metrics,
        "notes": {
            "polygon_fusion": (
                "Dense video floor polygon fusion runs when all scan roots include "
                "manifest.json, scan.mp4 and poses.bin. Use --skip-polygon to run "
                "only the merge/mapping stages."
            )
        },
    }
    (output_dir / "merge_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=_json_default))
    return 0


def _source_from_scan_root(scan_root: Path) -> SourceRtabmapScan:
    db_path = scan_root / "rtabmap.db"
    if not db_path.exists():
        raise FileNotFoundError(f"rtabmap.db not found under scan root: {scan_root}")
    return SourceRtabmapScan(scan_id=scan_root.name, db_path=db_path)


async def _run_polygon_fusion(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    scan_roots: dict[str, Path],
    mapping: Any,
    inter_session_loop_closure_count: int,
) -> dict[str, object]:
    polygon_source = _resolve_polygon_source(args=args, scan_roots=scan_roots)
    if polygon_source == "rtabmap":
        return await _run_rtabmap_polygon_fusion(
            args=args,
            output_dir=output_dir,
            scan_roots=scan_roots,
            mapping=mapping,
            inter_session_loop_closure_count=inter_session_loop_closure_count,
        )

    dense_sources: list[MultiScanDenseVideoSource] = []
    for scan_id, scan_root in scan_roots.items():
        dense_sources.append(_dense_video_source(scan_id=scan_id, scan_root=scan_root))

    model_cache = ModelCache(
        settings.model_cache_dir,
        repo_id=settings.segformer_model_repo_id,
        filename=settings.segformer_model_filename,
    )
    model_path = await asyncio.to_thread(model_cache.ensure)
    segmenter = SegformerOnnxSegmenter(model_path=model_path)
    raster_step = FloorRasterStep(
        FloorRasterStepParams(min_cell_hits=args.polygon_min_cell_hits)
    )
    step = MultiScanDenseVideoPolygonStep(
        segmenter=segmenter,
        params=MultiScanDenseVideoPolygonParams(
            stride=args.polygon_stride,
            pixel_stride=args.polygon_pixel_stride,
            source_timestamp_mode=args.polygon_source_timestamp_mode,
        ),
        raster_step=raster_step,
    )
    z0 = (
        float(args.polygon_z0)
        if args.polygon_z0 is not None
        else _estimate_z0_from_mapping(mapping)
    )
    result = await step.run(
        sources=dense_sources,
        mapping_result=mapping,
        z0=z0,
        inter_session_loop_closure_count=inter_session_loop_closure_count,
    )

    if result.raster.footprint_geojson is not None:
        (output_dir / "fused_floor_polygon.geojson").write_text(
            json.dumps(result.raster.footprint_geojson, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    np.savez_compressed(
        output_dir / "fused_floor_points.npz",
        points_xy=result.cloud.points_xy.astype(np.float32),
        z_values=result.cloud.z_values.astype(np.float32),
        grid_mask=result.raster.grid.mask.astype(np.uint8),
        observation_count=result.raster.grid.observation_count,
    )
    (output_dir / "fused_floor_metadata.json").write_text(
        json.dumps(
            {**result.metadata, "polygon_source": polygon_source},
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    return {**result.metadata, "polygon_source": polygon_source}


async def _run_rtabmap_polygon_fusion(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    scan_roots: dict[str, Path],
    mapping: Any,
    inter_session_loop_closure_count: int,
) -> dict[str, object]:
    model_cache = ModelCache(
        settings.model_cache_dir,
        repo_id=settings.segformer_model_repo_id,
        filename=settings.segformer_model_filename,
    )
    model_path = await asyncio.to_thread(model_cache.ensure)
    segmenter = SegformerOnnxSegmenter(model_path=model_path)
    raster_step = FloorRasterStep(
        FloorRasterStepParams(min_cell_hits=args.polygon_min_cell_hits)
    )
    step = MultiScanRtabmapFloorPolygonStep(
        segmenter=segmenter,
        params=MultiScanRtabmapFloorPolygonParams(
            floor_pixel_stride=args.polygon_pixel_stride,
            image_orientation_mode=args.rtabmap_image_orientation_mode,
            include_wall_polygon=not args.skip_wall_polygon,
            wall_pixel_stride=args.polygon_pixel_stride,
        ),
        raster_step=raster_step,
    )
    result = await step.run(
        sources=[
            MultiScanRtabmapFloorSource(scan_id=scan_id, db_path=scan_root / "rtabmap.db")
            for scan_id, scan_root in scan_roots.items()
        ],
        mapping_result=mapping,
        inter_session_loop_closure_count=inter_session_loop_closure_count,
    )
    if result.raster.footprint_geojson is not None:
        (output_dir / "fused_floor_polygon.geojson").write_text(
            json.dumps(result.raster.footprint_geojson, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if result.wall_polygon is not None and result.wall_polygon.final_polygon_geojson is not None:
        (output_dir / "fused_wall_polygon.geojson").write_text(
            json.dumps(
                result.wall_polygon.final_polygon_geojson,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if result.wall_heatmap is not None:
        np.savez_compressed(
            output_dir / "fused_wall_heatmap.npz",
            counts=result.wall_heatmap.counts,
        )
    np.savez_compressed(
        output_dir / "fused_floor_points.npz",
        points_xy=result.cloud.points_xy.astype(np.float32),
        z_values=result.cloud.z_values.astype(np.float32),
        grid_mask=result.raster.grid.mask.astype(np.uint8),
        observation_count=result.raster.grid.observation_count,
    )
    metadata = {**result.metadata, "polygon_source": "rtabmap"}
    (output_dir / "fused_floor_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return metadata


def _resolve_polygon_source(
    *,
    args: argparse.Namespace,
    scan_roots: dict[str, Path],
) -> str:
    if args.polygon_source != "auto":
        return str(args.polygon_source)
    if all(_scan_has_video_polygon_inputs(path) for path in scan_roots.values()):
        return "video"
    return "rtabmap"


def _scan_has_video_polygon_inputs(scan_root: Path) -> bool:
    return (
        (scan_root / "manifest.json").exists()
        and (scan_root / "scan.mp4").exists()
        and (scan_root / "poses.bin").exists()
    )


def _dense_video_source(*, scan_id: str, scan_root: Path) -> MultiScanDenseVideoSource:
    manifest = parse_manifest_file(scan_root / "manifest.json", expected_scan_id=scan_id)
    missing = [
        name
        for name, value in {
            "intrinsics_fx": manifest.intrinsics_fx,
            "intrinsics_fy": manifest.intrinsics_fy,
            "intrinsics_cx": manifest.intrinsics_cx,
            "intrinsics_cy": manifest.intrinsics_cy,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(f"manifest intrinsics missing for scan {scan_id}: {missing}")
    fx = manifest.intrinsics_fx
    fy = manifest.intrinsics_fy
    cx = manifest.intrinsics_cx
    cy = manifest.intrinsics_cy
    if fx is None or fy is None or cx is None or cy is None:
        raise ValueError(f"manifest intrinsics missing for scan {scan_id}: {missing}")
    video_path = scan_root / (manifest.video_path or "scan.mp4")
    poses_path = scan_root / (manifest.poses_path or "poses.bin")
    if not video_path.exists():
        raise FileNotFoundError(f"scan video missing for scan {scan_id}: {video_path}")
    if not poses_path.exists():
        raise FileNotFoundError(f"poses.bin missing for scan {scan_id}: {poses_path}")
    return MultiScanDenseVideoSource(
        scan_id=scan_id,
        video_path=video_path,
        poses_path=poses_path,
        intrinsics=Intrinsics(
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
        ),
    )


def _estimate_z0_from_mapping(mapping: Any) -> float:
    for item in mapping.mappings:
        if item.optimized_pose is not None:
            return float(item.optimized_pose[2][3]) - 1.5
    return 0.0


def _write_node_mapping_csv(path: Path, mapping: Any) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "source_scan_id",
                "source_node_id",
                "source_stamp",
                "mapping_source",
                "merged_node_id",
                "merged_stamp",
                "residual_s",
                "missing_reason",
            ],
        )
        writer.writeheader()
        for row in mapping.mappings:
            writer.writerow(
                {
                    "source_scan_id": row.source_scan_id,
                    "source_node_id": row.source_node_id,
                    "source_stamp": row.source_stamp,
                    "mapping_source": row.mapping_source,
                    "merged_node_id": row.merged_node_id,
                    "merged_stamp": row.merged_stamp,
                    "residual_s": row.residual_s,
                    "missing_reason": row.missing_reason,
                }
            )


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
