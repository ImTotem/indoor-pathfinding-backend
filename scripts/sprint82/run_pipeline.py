"""Sprint 82 — Entrypoint: Full pipeline.

Pre-flight → refine polygon → build nav graph → visualize → metrics.

Usage:
    cd server/
    uv run python -m scripts.sprint82.run_pipeline \\
      --merged-db ../_workspace/sprint81-frame-pose-rtabmap-merge/evidence/cycle2/merged_v2.db \\
      --baseline-polygon ../_workspace/sprint81-frame-pose-rtabmap-merge/evidence/cycle2/polygon/floor_polygon.geojson \\
      --out-dir ../_workspace/sprint82-floor-polygon-navgraph/evidence/cycle1
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("sprint82.run_pipeline")


def step_preflight(merged_db: Path, out_dir: Path) -> dict:
    """AC1: Pre-flight environment check."""
    logger.info("=== Step A: Pre-flight ===")
    lines = []

    # shapely
    try:
        import shapely
        lines.append(f"shapely: {shapely.__version__}")
        logger.info("shapely OK: %s", shapely.__version__)
    except ImportError as e:
        lines.append(f"shapely: MISSING — {e}")
        logger.error("shapely import FAIL: %s", e)
        raise

    # networkx
    nx_mode = "fallback dict mode"
    try:
        import networkx as nx
        nx_mode = f"networkx {nx.__version__}"
        lines.append(f"networkx: {nx.__version__}")
        logger.info("networkx OK: %s", nx.__version__)
    except ImportError:
        lines.append("networkx: fallback dict mode")
        logger.info("networkx not available — fallback dict mode")

    # scipy
    try:
        import scipy
        lines.append(f"scipy: {scipy.__version__}")
    except ImportError:
        lines.append("scipy: MISSING")
        raise RuntimeError("scipy required for cKDTree")

    # matplotlib
    try:
        import matplotlib
        lines.append(f"matplotlib: {matplotlib.__version__}")
    except ImportError:
        lines.append("matplotlib: MISSING")

    # merged_v2.db
    if not merged_db.exists():
        raise FileNotFoundError(f"merged_v2.db not found: {merged_db}")

    conn = sqlite3.connect(str(merged_db))
    node_count = conn.execute("SELECT COUNT(*) FROM Node").fetchone()[0]
    link_count = conn.execute("SELECT COUNT(*) FROM Link").fetchone()[0]

    # Count non-zero poses
    import struct
    nodes = conn.execute("SELECT id, pose FROM Node").fetchall()
    nonzero = 0
    for nid, blob in nodes:
        if blob and len(blob) >= 48:
            v = struct.unpack("<12f", bytes(blob))
            tx, ty = v[3], v[7]
            if abs(tx) > 1e-9 or abs(ty) > 1e-9:
                nonzero += 1
            elif nid == 1:
                pass  # origin skip
    conn.close()

    lines.append(f"merged_v2.db: Node count={node_count}, pose non-zero count={nonzero}, link count={link_count}")
    logger.info("merged_v2.db: nodes=%d, non-zero poses=%d, links=%d", node_count, nonzero, link_count)

    out_dir.mkdir(parents=True, exist_ok=True)
    preflight_txt = out_dir / "preflight.txt"
    preflight_txt.write_text("\n".join(lines) + "\n")
    logger.info("Pre-flight OK → %s", preflight_txt)

    return {
        "node_count": node_count,
        "pose_nonzero_count": nonzero,
        "link_count": link_count,
        "networkx_mode": nx_mode,
    }


def step_refine_polygon(
    merged_db: Path,
    baseline_geojson: Path,
    out_dir: Path,
) -> tuple:
    """AC2 + AC3: Polygon refinement + overlay PNG."""
    logger.info("=== Step B: Refine polygon ===")
    from .refine_polygon import load_pose_xy, refine_floor_polygon, save_polygon_geojson, _remove_outliers

    xy, map_id, pose_ids = load_pose_xy(merged_db)

    # Outlier removal
    xy_clean, drop_ratio = _remove_outliers(xy, k=4, threshold_m=5.0)
    # Rebuild map_id/pose_ids for cleaned xy using mask
    from scipy.spatial import cKDTree
    import numpy as np
    tree = cKDTree(xy)
    _, idx_map = tree.query(xy_clean)
    map_id_clean = map_id[idx_map]
    pose_ids_clean = pose_ids[idx_map]

    logger.info("Pose count after outlier removal: %d (dropped %.1f%%)", len(xy_clean), drop_ratio * 100)

    poly, metrics = refine_floor_polygon(xy_clean)
    metrics["outlier_drop_ratio"] = round(drop_ratio, 4)

    # Save GeoJSON
    out_polygon_dir = out_dir / "polygon"
    save_polygon_geojson(
        poly, metrics,
        out_polygon_dir / "floor_polygon_refined.geojson",
        xy_clean, map_id_clean,
    )

    # Save metrics
    out_polygon_dir.mkdir(parents=True, exist_ok=True)
    (out_polygon_dir / "polygon_metrics.json").write_text(json.dumps(metrics, indent=2))

    # AC3: Overlay PNG
    from .visualize import render_polygon_overlay
    render_polygon_overlay(
        baseline_geojson, poly, xy_clean, map_id_clean,
        out_polygon_dir / "floor_polygon_overlay.png",
    )

    logger.info(
        "Polygon refined: area=%.1f m², containment=%.3f, vertices %d→%d",
        metrics["area_m2"], metrics["containment_ratio"],
        metrics["vertex_count_before"], metrics["vertex_count_after"],
    )
    return poly, xy_clean, map_id_clean, pose_ids_clean, metrics


def step_build_navgraph(
    merged_db: Path,
    polygon,
    xy: "np.ndarray",
    map_id: "np.ndarray",
    pose_ids: "np.ndarray",
    out_dir: Path,
) -> tuple:
    """AC4: Nav graph build."""
    logger.info("=== Step C: Build nav graph ===")
    from .build_nav_graph import (
        downsample_nodes, build_edges, assign_components,
        compute_polygon_edge_containment, load_rtabmap_links,
        save_nav_geojson, save_nav_graph_json,
    )
    import numpy as np

    rtabmap_links = load_rtabmap_links(merged_db)
    logger.info("RTABMap links loaded: %d", len(rtabmap_links))

    nodes = downsample_nodes(xy, map_id, pose_ids, polygon, cell_m=2.0)
    logger.info("Nav nodes (pre-component): %d", len(nodes))

    edges = build_edges(nodes, polygon, rtabmap_links, edge_max_m=3.0)
    comp_stats = assign_components(nodes, edges, polygon, cross_session_bridge_m=5.0)

    active_nodes = [n for n in nodes if n.component_id >= 0]
    active_edges = [e for e in edges if
                    any(n.node_id == e.from_id for n in active_nodes) and
                    any(n.node_id == e.to_id for n in active_nodes)]

    # Polygon containment ratio for edges
    containment = compute_polygon_edge_containment(active_edges, active_nodes, polygon)

    from collections import Counter
    comp_sizes = Counter(n.component_id for n in active_nodes)
    avg_deg = (
        2 * len(active_edges) / len(active_nodes)
        if active_nodes else 0.0
    )

    navgraph_metrics = {
        "node_count": len(active_nodes),
        "edge_count": len(active_edges),
        "components": comp_stats["components"],
        "largest_component_size": comp_stats["largest_component_size"],
        "largest_component_ratio": comp_stats["largest_component_ratio"],
        "avg_degree": round(avg_deg, 2),
        "polygon_containment_ratio": round(containment, 4),
        "bridge_attempted": comp_stats["bridge_attempted"],
        "bridge_added": comp_stats.get("bridge_added", False),
    }

    out_nav_dir = out_dir / "navgraph"
    out_nav_dir.mkdir(parents=True, exist_ok=True)

    save_nav_geojson(
        active_nodes, active_edges,
        out_nav_dir / "nav_nodes.geojson",
        out_nav_dir / "nav_edges.geojson",
    )
    save_nav_graph_json(
        active_nodes, active_edges, navgraph_metrics,
        out_nav_dir / "nav_graph.json",
    )
    (out_nav_dir / "navgraph_metrics.json").write_text(json.dumps(navgraph_metrics, indent=2))

    logger.info(
        "Nav graph: %d nodes, %d edges, %d components, containment=%.3f",
        len(active_nodes), len(active_edges),
        comp_stats["components"], containment,
    )
    return active_nodes, active_edges, navgraph_metrics


def step_visualize_navgraph(
    polygon,
    nodes,
    edges,
    out_dir: Path,
) -> None:
    """AC5: Nav graph PNG."""
    logger.info("=== Step D: Visualize nav graph ===")
    from .visualize import render_nav_graph, render_combined

    out_nav_dir = out_dir / "navgraph"
    render_nav_graph(polygon, nodes, edges, out_nav_dir / "nav_graph.png")

    # Combined
    render_combined(
        out_dir / "polygon" / "floor_polygon_overlay.png",
        out_nav_dir / "nav_graph.png",
        out_dir / "combined.png",
    )


def main(merged_db: Path, baseline_polygon: Path, out_dir: Path) -> dict:
    """Full pipeline. Returns aggregated metrics dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step A: Pre-flight
    preflight = step_preflight(merged_db, out_dir)

    # Step B: Polygon
    polygon, xy, map_id, pose_ids, poly_metrics = step_refine_polygon(
        merged_db, baseline_polygon, out_dir
    )

    # Step C: Nav graph
    nodes, edges, nav_metrics = step_build_navgraph(
        merged_db, polygon, xy, map_id, pose_ids, out_dir
    )

    # Step D: Visualize
    step_visualize_navgraph(polygon, nodes, edges, out_dir)

    # Aggregate metrics
    metrics = {
        "preflight": preflight,
        "polygon": {
            "area_m2": poly_metrics["area_m2"],
            "containment_ratio": poly_metrics["containment_ratio"],
            "vertex_count_before": poly_metrics["vertex_count_before"],
            "vertex_count_after": poly_metrics["vertex_count_after"],
            "vertex_reduction_ratio": round(
                1.0 - poly_metrics["vertex_count_after"] / max(poly_metrics["vertex_count_before"], 1), 4
            ),
            "adopted_ratio": poly_metrics["adopted_ratio"],
            "geometry_type": poly_metrics["geometry_type"],
            "outlier_drop_ratio": poly_metrics.get("outlier_drop_ratio", 0.0),
        },
        "navgraph": nav_metrics,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Pipeline complete — metrics.json written → %s", out_dir / "metrics.json")

    return metrics


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Sprint 82: floor polygon + nav graph pipeline")
    parser.add_argument("--merged-db", required=True, type=Path)
    parser.add_argument("--baseline-polygon", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    metrics = main(args.merged_db, args.baseline_polygon, args.out_dir)

    print("\n=== Sprint 82 Pipeline Complete ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    _cli()
