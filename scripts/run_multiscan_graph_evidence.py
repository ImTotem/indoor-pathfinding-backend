"""Multi-scan fused floor polygon/points → display navigation graph evidence harness.

Sprint 77 Cycle 1 — Step C.

입력: Step B 산출물 디렉터리
  - 우선 경로 A: fused_floor_polygon.geojson (area ≥ 5m² 이면 사용)
  - Fallback 경로 B: fused_floor_points.npz → WalkableGridStep → alpha-shape polygon →
    DisplayNavigationGridStep

출력: walkable_grid.npz, nodes.json, edges.json, graph_metadata.json

좌표계: fused floor points/polygon 은 multiscan_rtabmap_floor_polygon 스텝이
world frame 에서 back-projection 했으므로 추가 변환 불필요.

Usage:
    cd server/
    uv run python scripts/run_multiscan_graph_evidence.py \\
      --merge-dir ../_workspace/.../merge_floor \\
      --out-dir  ../_workspace/.../graph
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # server/
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ─────────────────────────────────────────────
# serialization helpers
# ─────────────────────────────────────────────

def _node_to_dict(node: object) -> dict[str, object]:
    nt = getattr(node, "node_type", None)
    return {
        "node_id": str(getattr(node, "node_id", "")),
        "scan_id": str(getattr(node, "scan_id", "")),
        "build_job_id": str(getattr(node, "build_job_id", "")),
        "type": nt.value if hasattr(nt, "value") else str(nt),
        "x": getattr(node, "x", 0.0),
        "y": getattr(node, "y", 0.0),
        "z": getattr(node, "z", 0.0),
        "label": getattr(node, "label", None),
    }


def _edge_to_dict(edge: object) -> dict[str, object]:
    et = getattr(edge, "edge_type", None)
    return {
        "edge_id": str(getattr(edge, "edge_id", "")),
        "scan_id": str(getattr(edge, "scan_id", "")),
        "build_job_id": str(getattr(edge, "build_job_id", "")),
        "from": str(getattr(edge, "from_node_id", "")),
        "to": str(getattr(edge, "to_node_id", "")),
        "type": et.value if hasattr(et, "value") else str(et),
        "length_m": getattr(edge, "length_m", 0.0),
        "polyline": [list(pt) for pt in getattr(edge, "polyline", [])],
    }


# ─────────────────────────────────────────────
# walkable_grid.npz 합성 (render_graph_only.py 호환)
# ─────────────────────────────────────────────

def _rasterize_footprint(footprint_geojson: dict | object, cell_size: float, z0: float):
    """footprint GeoJSON 또는 shapely geom → (mask, obs, x0, y0, z0, cell)."""
    from shapely.geometry import Point
    from shapely.geometry.base import BaseGeometry

    if isinstance(footprint_geojson, BaseGeometry):
        geom = footprint_geojson
    else:
        from shapely.geometry import shape
        from shapely.ops import unary_union
        gj = footprint_geojson
        features = gj.get("features", []) if isinstance(gj, dict) else []
        if features:
            geom = unary_union([shape(f["geometry"]) for f in features])
        else:
            # Raw geometry (not FeatureCollection)
            geom = shape(gj)

    if geom.is_empty:
        return None

    b = geom.bounds  # minx, miny, maxx, maxy
    x_min, y_min, x_max, y_max = b
    x0_val = x_min - cell_size
    y0_val = y_min - cell_size
    w = max(1, int(np.ceil((x_max - x_min) / cell_size)) + 2)
    h = max(1, int(np.ceil((y_max - y_min) / cell_size)) + 2)

    mask = np.zeros((h, w), dtype=bool)
    obs = np.zeros((h, w), dtype=np.uint16)
    for gy in range(h):
        for gx in range(w):
            px = x0_val + (gx + 0.5) * cell_size
            py = y0_val + (gy + 0.5) * cell_size
            if geom.contains(Point(px, py)):
                mask[gy, gx] = True
                obs[gy, gx] = 3
    return mask, obs, x0_val, y0_val, z0, cell_size


def _save_walkable_grid(out_dir: Path, raster_result, z0: float, cell_size: float) -> None:
    if raster_result is None:
        print("[WARN] footprint rasterization produced no cells; walkable_grid.npz not written")
        return
    mask, obs, x0_val, y0_val, z0_val, cell = raster_result
    grid_path = out_dir / "walkable_grid.npz"
    np.savez_compressed(
        str(grid_path),
        mask=mask,
        observation_count=obs,
        origin_x0=np.float64(x0_val),
        origin_y0=np.float64(y0_val),
        origin_z0=np.float64(z0_val),
        origin_cell_size=np.float64(cell),
        origin_w=np.int64(mask.shape[1]),
        origin_h=np.int64(mask.shape[0]),
    )
    print(f"[INFO] wrote {grid_path} (mask cells={int(mask.sum())})")


# ─────────────────────────────────────────────
# Fallback B: fused_floor_points.npz → alpha-shape polygon
# ─────────────────────────────────────────────

def _points_npz_to_footprint_geojson(npz_path: Path, z0: float, alpha: float = 2.0) -> dict:
    """fused_floor_points.npz의 XY 포인트 → alpha-shape polygon → GeoJSON dict."""
    from shapely.geometry import mapping, MultiPoint
    from shapely.ops import unary_union

    data = np.load(str(npz_path))
    pts_xy = data["points_xy"].astype(np.float64)

    if len(pts_xy) == 0:
        raise ValueError("fused_floor_points.npz has 0 points")

    # alpha-shape via shapely buffer trick (concave hull approximation)
    # shapely 2.0+: Polygon.concave_hull
    mp = MultiPoint(pts_xy)
    try:
        footprint = mp.concave_hull(ratio=0.1)
    except AttributeError:
        # shapely < 2.0 fallback: convex hull
        footprint = mp.convex_hull

    if footprint.is_empty or footprint.area < 0.5:
        # last resort: convex hull
        footprint = mp.convex_hull

    geojson = mapping(footprint)
    print(f"[INFO] points→footprint: area={footprint.area:.2f}m² type={footprint.geom_type} n_pts={len(pts_xy)}")
    return dict(geojson), footprint


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-scan fused floor → graph evidence")
    parser.add_argument(
        "--merge-dir", type=Path, required=True,
        help="Step B output dir (contains fused_floor_polygon.geojson + fused_floor_points.npz)",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cell-m", type=float, default=1.0, help="Graph tile cell size in meters (default 1.0)")
    parser.add_argument("--clearance-m", type=float, default=0.3, help="Wall clearance in meters (default 0.3)")
    parser.add_argument("--z0", type=float, default=None, help="Floor z height override")
    parser.add_argument("--grid-cell-size", type=float, default=0.10, help="Walkable grid raster cell size (default 0.10m)")
    parser.add_argument("--min-polygon-area-m2", type=float, default=5.0,
                        help="Minimum polygon area to use GeoJSON directly (fallback to points if below)")
    args = parser.parse_args()

    merge_dir: Path = args.merge_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 입력 파일 ──────────────────────────────────────────
    polygon_path = merge_dir / "fused_floor_polygon.geojson"
    points_path = merge_dir / "fused_floor_points.npz"
    metadata_path = merge_dir / "fused_floor_metadata.json"

    # ── 2. z0 결정 ─────────────────────────────────────────────
    z0 = args.z0
    if z0 is None and metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        # raster origin z0
        raster_origin = meta.get("raster", {}).get("origin", {})
        z0 = raster_origin.get("z0")
        if z0 is not None:
            z0 = float(z0)
        else:
            # per_scan z0 평균
            per_scan = meta.get("per_scan", {})
            z0s = [v.get("z0") for v in per_scan.values() if v.get("z0") is not None]
            if z0s:
                z0 = float(np.mean(z0s))
        if z0 is not None:
            print(f"[INFO] z0 from metadata: {z0:.3f}")
    if z0 is None:
        z0 = 0.0
        print(f"[WARN] z0 not found in metadata; defaulting to {z0}")

    # ── 3. footprint 결정 ─────────────────────────────────────
    footprint_geojson: dict | None = None
    footprint_geom = None
    polygon_path_used = "none"

    if polygon_path.exists():
        raw_gj = json.loads(polygon_path.read_text(encoding="utf-8"))
        from shapely.geometry import shape
        from shapely.ops import unary_union
        if raw_gj.get("type") == "FeatureCollection":
            features = raw_gj.get("features", [])
            geom = unary_union([shape(f["geometry"]) for f in features]) if features else shape(raw_gj)
        else:
            geom = shape(raw_gj)
        area = float(geom.area)
        print(f"[INFO] fused_floor_polygon.geojson area={area:.2f}m²")
        if area >= args.min_polygon_area_m2:
            footprint_geojson = raw_gj
            footprint_geom = geom
            polygon_path_used = str(polygon_path)
        else:
            print(f"[WARN] polygon area {area:.2f}m² < min {args.min_polygon_area_m2}m² → fallback to points npz")

    if footprint_geojson is None and points_path.exists():
        print(f"[INFO] Building footprint from fused_floor_points.npz …")
        footprint_geojson, footprint_geom = _points_npz_to_footprint_geojson(points_path, z0)
        polygon_path_used = str(points_path) + " (alpha-shape)"
        # Save derived polygon for reference
        from shapely.geometry import mapping
        derived_path = out_dir / "derived_footprint.geojson"
        derived_path.write_text(
            json.dumps(dict(mapping(footprint_geom)), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[INFO] saved derived footprint: {derived_path}")

    if footprint_geojson is None:
        print("[ERROR] No usable footprint source found", file=sys.stderr)
        return 1

    # ── 4. DisplayNavigationGridStep 호출 ─────────────────────
    from indoor_server.application.building.steps.display_navigation_grid import (
        DisplayNavigationGridParams,
        DisplayNavigationGridStep,
    )

    scan_id = uuid4()
    build_job_id = uuid4()
    params = DisplayNavigationGridParams(
        cell_m=args.cell_m,
        clearance_m=args.clearance_m,
    )
    step = DisplayNavigationGridStep(
        scan_id=scan_id,
        build_job_id=build_job_id,
        params=params,
    )

    print(f"[INFO] Running DisplayNavigationGridStep (cell={args.cell_m}m, clearance={args.clearance_m}m) …")
    result = step.run(
        footprint_geojson=footprint_geojson,
        floor_z=z0,
        pois=[],
    )

    nodes = result.nodes
    edges = result.edges
    print(f"[INFO] graph: nodes={len(nodes)} edges={len(edges)}")

    # ── 5. nodes.json / edges.json 저장 ──────────────────────
    (out_dir / "nodes.json").write_text(
        json.dumps([_node_to_dict(n) for n in nodes], ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "edges.json").write_text(
        json.dumps([_edge_to_dict(e) for e in edges], ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[INFO] wrote nodes.json ({len(nodes)} nodes), edges.json ({len(edges)} edges)")

    # ── 6. walkable_grid.npz 합성 ─────────────────────────────
    print(f"[INFO] Rasterizing footprint → walkable_grid.npz (cell={args.grid_cell_size}m) …")
    raster_result = _rasterize_footprint(footprint_geom or footprint_geojson, args.grid_cell_size, z0)
    _save_walkable_grid(out_dir, raster_result, z0, args.grid_cell_size)

    # ── 7. graph_metadata.json ────────────────────────────────
    graph_meta = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "z0": z0,
        "cell_m": args.cell_m,
        "clearance_m": args.clearance_m,
        "polygon_path_used": polygon_path_used,
        "graph_metadata": result.metadata,
    }
    (out_dir / "graph_metadata.json").write_text(
        json.dumps(graph_meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    if len(nodes) == 0:
        print("[ERROR] 0 nodes — check footprint geometry or cell_m/clearance_m params", file=sys.stderr)
        return 1

    print(f"[DONE] graph evidence written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
