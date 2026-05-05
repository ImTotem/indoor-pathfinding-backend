"""Sprint 81 — Full pipeline entrypoint.

Usage:
  python run_pipeline.py --scan-a <scan_A_root> --scan-b <scan_B_root> \
      --out-dir <evidence_out_dir> [--sample-hz 2.0] [--jpeg-quality 92]

Steps:
  0. Pre-flight check
  1. Keyframe extraction (scan A + B)
  2. RGB-only RTAB-Map DB writer (scan A + B)
  3. rtabmap-reprocess multi-session merge → merged.db
  4a. ARKit SE(3) alignment → merged_v2.db (Cycle 2: pose patch)
  4b. Heatmap + polygon + visualization from merged_v2.db
  5. rtabmap-info for scan_A.db and scan_B.db

Output directory layout:
  <out-dir>/
    preflight.txt
    calib_A.json, calib_B.json
    kf_A/index.json + *.jpg
    kf_B/index.json + *.jpg
    scan_A.db, scan_B.db
    rtabmap_info_A.txt, rtabmap_info_B.txt
    merged.db, merge.log
    merged_v2.db
    merge_metrics_v2.json
    viz/pose_graph.png, heatmap_pose.png, heatmap_features.png
    polygon/floor_polygon.geojson, floor_polygon.png
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _run(cmd: list[str], log_path: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    logger.info("$ %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if log_path:
        log_path.write_text(result.stdout + result.stderr)
    if check and result.returncode != 0:
        logger.error("FAILED (exit=%d): %s\nstderr: %s", result.returncode, cmd, result.stderr[-2000:])
        raise RuntimeError(f"Command failed: {cmd}")
    return result


def step0_preflight(out_dir: Path) -> None:
    """Pre-flight: tool availability check."""
    import av
    import cv2
    import numpy as np
    import shapely

    lines = [
        f"rtabmap-reprocess: {subprocess.run(['rtabmap-reprocess', '--version'], capture_output=True, text=True).stderr.strip()[:80]}",
        f"av: {av.__version__}",
        f"cv2: {cv2.__version__}",
        f"numpy: {np.__version__}",
        f"shapely: {shapely.__version__}",
    ]
    (out_dir / "preflight.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        logger.info(line)


def step1_extract_keyframes(
    scan_root: Path,
    out_dir: Path,
    scan_label: str,
    sample_hz: float,
    jpeg_quality: int,
) -> Path:
    """Step 1: keyframe extraction."""
    script = Path(__file__).parent / "extract_keyframes.py"
    kf_dir = out_dir / f"kf_{scan_label}"
    _run([sys.executable, str(script), str(scan_root), str(kf_dir),
          "--sample-hz", str(sample_hz), "--jpeg-quality", str(jpeg_quality)])
    return kf_dir / "index.json"


def step2_write_db(
    scan_root: Path,
    kf_index: Path,
    out_db: Path,
    map_id: int,
    scan_label: str,
    out_dir: Path,
) -> None:
    """Step 2: RGB-only RTAB-Map DB."""
    script = Path(__file__).parent / "write_rtabmap_db.py"
    _run([sys.executable, str(script), str(scan_root), str(kf_index),
          str(out_db), "--map-id", str(map_id)])

    # rtabmap-info evidence
    info_path = out_dir / f"rtabmap_info_{scan_label}.txt"
    result = _run(["rtabmap-info", str(out_db)], log_path=info_path, check=False)
    logger.info("rtabmap-info %s: exit=%d", out_db.name, result.returncode)


def step3_merge(
    scan_a_db: Path,
    scan_b_db: Path,
    merged_db: Path,
    merge_log: Path,
) -> None:
    """Step 3: rtabmap-reprocess multi-session merge."""
    dbs_arg = f"{scan_a_db};{scan_b_db}"
    cmd = [
        "rtabmap-reprocess",
        "--Mem/IncrementalMemory", "true",
        "--Mem/InitWMWithAllNodes", "false",
        "--RGBD/Enabled", "false",
        "--Reg/Strategy", "0",
        "--Vis/EstimationType", "1",
        "--Kp/DetectorStrategy", "6",
        "--Rtabmap/DetectionRate", "0",
        "--Rtabmap/LoopThr", "0.11",
        "--Mem/STMSize", "30",
        "--uwarn",
        dbs_arg,
        str(merged_db),
    ]
    _run(cmd, log_path=merge_log)


def step4a_align(
    merged_db: Path,
    scan_a_db: Path,
    scan_b_db: Path,
    merged_v2_db: Path,
    metrics_json: Path,
) -> None:
    """Step 4a: ARKit SE(3) alignment → merged_v2.db."""
    script = Path(__file__).parent / "align_and_rebuild_merged.py"
    _run([sys.executable, str(script),
          str(merged_db), str(scan_a_db), str(scan_b_db),
          str(merged_v2_db), "--metrics-json", str(metrics_json)])


def step4b_visualize(
    merged_v2_db: Path,
    out_viz: Path,
    out_polygon: Path,
    metrics_json: Path,
) -> None:
    """Step 4b: heatmap + polygon from merged_v2.db."""
    script = Path(__file__).parent / "extract_heatmap_polygon_v2.py"
    _run([sys.executable, str(script),
          str(merged_v2_db), str(out_viz), str(out_polygon),
          "--metrics-json", str(metrics_json)])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Sprint 81 full pipeline")
    parser.add_argument("--scan-a", type=Path, required=True, help="Scan A root dir")
    parser.add_argument("--scan-b", type=Path, required=True, help="Scan B root dir")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output evidence directory")
    parser.add_argument("--sample-hz", type=float, default=2.0)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument(
        "--skip-steps", nargs="*", default=[],
        help="Steps to skip (0,1a,1b,2a,2b,3,4a,4b). Useful for partial re-runs."
    )
    args = parser.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # Step 0
    if "0" not in args.skip_steps:
        logger.info("=== Step 0: Pre-flight ===")
        step0_preflight(out)

    # Step 1
    kf_a_index = out / "kf_A" / "index.json"
    if "1a" not in args.skip_steps:
        logger.info("=== Step 1a: Keyframe extraction — Scan A ===")
        kf_a_index = step1_extract_keyframes(args.scan_a, out, "A", args.sample_hz, args.jpeg_quality)

    kf_b_index = out / "kf_B" / "index.json"
    if "1b" not in args.skip_steps:
        logger.info("=== Step 1b: Keyframe extraction — Scan B ===")
        kf_b_index = step1_extract_keyframes(args.scan_b, out, "B", args.sample_hz, args.jpeg_quality)

    # Step 2
    scan_a_db = out / "scan_A.db"
    if "2a" not in args.skip_steps:
        logger.info("=== Step 2a: RGB-only DB — Scan A ===")
        step2_write_db(args.scan_a, kf_a_index, scan_a_db, 0, "A", out)

    scan_b_db = out / "scan_B.db"
    if "2b" not in args.skip_steps:
        logger.info("=== Step 2b: RGB-only DB — Scan B ===")
        step2_write_db(args.scan_b, kf_b_index, scan_b_db, 1, "B", out)

    # Step 3
    merged_db = out / "merged.db"
    merge_log = out / "merge.log"
    if "3" not in args.skip_steps:
        logger.info("=== Step 3: rtabmap-reprocess multi-session merge ===")
        step3_merge(scan_a_db, scan_b_db, merged_db, merge_log)

    # Step 4a
    merged_v2_db = out / "merged_v2.db"
    metrics_json = out / "merge_metrics_v2.json"
    if "4a" not in args.skip_steps:
        logger.info("=== Step 4a: ARKit SE(3) alignment → merged_v2.db ===")
        step4a_align(merged_db, scan_a_db, scan_b_db, merged_v2_db, metrics_json)

    # Step 4b
    out_viz = out / "viz"
    out_polygon = out / "polygon"
    if "4b" not in args.skip_steps:
        logger.info("=== Step 4b: Heatmap + polygon visualization ===")
        step4b_visualize(merged_v2_db, out_viz, out_polygon, metrics_json)

    elapsed = time.time() - t_start
    logger.info("Pipeline complete in %.1fs", elapsed)
    logger.info("Output: %s", out)

    # Summary
    key_files = [
        out / "merged_v2.db",
        out / "merge_metrics_v2.json",
        out / "viz" / "pose_graph.png",
        out / "polygon" / "floor_polygon.geojson",
    ]
    for f in key_files:
        status = f"OK ({f.stat().st_size//1024}KB)" if f.exists() else "MISSING"
        logger.info("  %s: %s", f.name, status)


if __name__ == "__main__":
    main()
