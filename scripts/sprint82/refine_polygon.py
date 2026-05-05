"""Sprint 82 — Step 1: Floor polygon refinement.

merged_v2.db Node.pose XY → refined floor polygon (concave_hull ratio sweep +
morphological close/open + simplify).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import struct
from pathlib import Path
from typing import Union

import numpy as np
import shapely
from shapely.geometry import MultiPoint, mapping
from shapely.ops import unary_union
from shapely.validation import make_valid

# shapely >= 2.0 uses shapely.concave_hull(geom, ratio=...) instead of geom.concave_hull(ratio=...)
_SHAPELY_CONCAVE_HULL_FUNC = hasattr(shapely, "concave_hull")


def _concave_hull(mp: "MultiPoint", ratio: float):
    """Compatibility wrapper for shapely.concave_hull / geom.concave_hull."""
    if _SHAPELY_CONCAVE_HULL_FUNC:
        return shapely.concave_hull(mp, ratio=ratio)
    return mp.concave_hull(ratio=ratio)  # shapely < 2.x fallback

try:
    from shapely.geometry import Polygon, MultiPolygon
except ImportError:
    pass

logger = logging.getLogger(__name__)

ShapelyPolygon = Union["Polygon", "MultiPolygon"]


def _blob_to_T(blob: bytes) -> np.ndarray:
    """3x4 float32 row-major blob → 4x4 homogeneous transform."""
    v = struct.unpack("<12f", bytes(blob))
    T = np.eye(4)
    T[0, :3] = v[0:3];  T[0, 3] = v[3]
    T[1, :3] = v[4:7];  T[1, 3] = v[7]
    T[2, :3] = v[8:11]; T[2, 3] = v[11]
    return T


def load_pose_xy(merged_db: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Node.pose from merged_v2.db.

    Returns
    -------
    xy       : np.ndarray (N, 2)   — ARKit XY floor plane
    map_id   : np.ndarray (N,)     — session index {0, 1}
    pose_ids : np.ndarray (N,)     — original Node.id
    """
    conn = sqlite3.connect(str(merged_db))
    conn.row_factory = sqlite3.Row
    nodes = conn.execute("SELECT id, map_id, pose FROM Node ORDER BY id").fetchall()
    links = conn.execute("SELECT from_id, to_id, type FROM Link").fetchall()
    conn.close()

    xy_list, mid_list, pid_list = [], [], []
    zero_skipped = 0
    for n in nodes:
        nid = n["id"]
        blob = n["pose"]
        if blob is None or len(blob) < 48:
            zero_skipped += 1
            continue
        v = struct.unpack("<12f", bytes(blob))
        tx, ty = v[3], v[7]
        # id=1 origin (all-zero translation) 제외
        if abs(tx) < 1e-9 and abs(ty) < 1e-9 and nid <= 1:
            zero_skipped += 1
            continue
        if abs(tx) < 1e-9 and abs(ty) < 1e-9:
            zero_skipped += 1
            continue
        xy_list.append([tx, ty])
        mid_list.append(n["map_id"])
        pid_list.append(nid)

    logger.info(
        "load_pose_xy: %d valid poses, %d zero-pose skipped, %d links",
        len(xy_list), zero_skipped, len(links),
    )

    xy = np.array(xy_list, dtype=float)
    map_id = np.array(mid_list, dtype=int)
    pose_ids = np.array(pid_list, dtype=int)
    return xy, map_id, pose_ids


def _remove_outliers(xy: np.ndarray, k: int = 4, threshold_m: float = 5.0) -> tuple[np.ndarray, float]:
    """KDTree 기반 outlier 제거. k-nearest 평균거리 > threshold_m 인 점 제거."""
    from scipy.spatial import cKDTree

    if len(xy) <= k + 1:
        return xy, 0.0

    tree = cKDTree(xy)
    dists, _ = tree.query(xy, k=k + 1)  # includes self
    mean_dists = dists[:, 1:].mean(axis=1)  # exclude self
    mask = mean_dists <= threshold_m
    drop_ratio = float((~mask).sum()) / len(xy)
    if drop_ratio > 0.05:
        logger.warning("Outlier drop ratio %.1f%% > 5%% — possible alignment issue", drop_ratio * 100)
    logger.info("Outlier removal: dropped %d / %d points (%.1f%%)", (~mask).sum(), len(xy), drop_ratio * 100)
    return xy[mask], drop_ratio


def _containment_ratio(poly: "ShapelyPolygon", xy: np.ndarray) -> float:
    """polygon 안에 들어가는 pose 수 / 전체 pose 수."""
    from shapely.geometry import Point
    from shapely.prepared import prep

    prepped = prep(poly)
    inside = sum(1 for pt in xy if prepped.contains(Point(pt)))
    return inside / len(xy) if len(xy) > 0 else 0.0


def refine_floor_polygon(
    xy: np.ndarray,
    *,
    concave_ratios: list[float] | None = None,
    buffer_close_m: float = 0.6,
    buffer_open_m: float = 0.4,
    simplify_tol_m: float = 0.20,
    min_containment_ratio: float = 0.95,
) -> tuple["ShapelyPolygon", dict]:
    """Refine floor polygon via ratio sweep + morphological close/open + simplify.

    Returns
    -------
    polygon : Shapely Polygon or MultiPolygon (valid geometry)
    metrics : dict with sweep results and adopted hyperparameters
    """
    if concave_ratios is None:
        concave_ratios = [0.10, 0.15, 0.20, 0.30]

    mp = MultiPoint(xy.tolist())
    sweep_results = []
    adopted = None
    adopted_poly = None

    # Extend sweep range — include finer-grain and up to 0.80 to handle
    # corridor-shaped pose distributions where tight ratios undercount containment
    extended_ratios = concave_ratios + [0.05, 0.08, 0.40, 0.50, 0.60, 0.70, 0.80]
    seen: list[float] = []
    for r in extended_ratios:
        if r in seen:
            continue
        seen.append(r)

    all_ratios = sorted(seen)

    for ratio in all_ratios:
        try:
            hull = _concave_hull(mp, ratio=ratio)
        except Exception as e:
            logger.warning("concave_hull ratio=%.2f failed: %s", ratio, e)
            continue

        # Morphological close: fill small gaps
        closed = hull.buffer(buffer_close_m).buffer(-buffer_close_m)
        # Morphological open: remove spurs
        opened = closed.buffer(-buffer_open_m).buffer(buffer_open_m)
        # Simplify
        simplified = opened.simplify(simplify_tol_m, preserve_topology=True)
        # Validate
        if not simplified.is_valid:
            simplified = make_valid(simplified)

        if simplified.is_empty:
            logger.warning("concave_hull ratio=%.2f → empty after morphology", ratio)
            continue

        cr = _containment_ratio(simplified, xy)
        vb = len(list(hull.exterior.coords)) if hasattr(hull, "exterior") else sum(
            len(list(g.exterior.coords)) for g in hull.geoms
        ) if hasattr(hull, "geoms") else 0
        va = len(list(simplified.exterior.coords)) if hasattr(simplified, "exterior") else sum(
            len(list(g.exterior.coords)) for g in simplified.geoms
        ) if hasattr(simplified, "geoms") else 0

        result = {
            "ratio": ratio,
            "area_m2": round(simplified.area, 2),
            "containment_ratio": round(cr, 4),
            "vertex_before": vb,
            "vertex_after": va,
            "is_valid": simplified.is_valid,
        }
        sweep_results.append(result)
        logger.info(
            "ratio=%.2f: area=%.1f m², containment=%.3f, vertices %d→%d",
            ratio, simplified.area, cr, vb, va,
        )

        if cr >= min_containment_ratio and adopted is None:
            adopted = ratio
            adopted_poly = simplified

    if adopted_poly is None:
        logger.warning(
            "No ratio achieved containment >= %.2f — using baseline (largest area result)",
            min_containment_ratio,
        )
        # Fallback: pick result with highest containment
        if sweep_results:
            best = max(sweep_results, key=lambda x: x["containment_ratio"])
            adopted = best["ratio"]
            hull = _concave_hull(mp, ratio=adopted)
            closed = hull.buffer(buffer_close_m).buffer(-buffer_close_m)
            opened = closed.buffer(-buffer_open_m).buffer(buffer_open_m)
            simplified = opened.simplify(simplify_tol_m, preserve_topology=True)
            if not simplified.is_valid:
                simplified = make_valid(simplified)
            adopted_poly = simplified
        else:
            # Last resort: convex hull
            adopted_poly = mp.convex_hull
            adopted = None

    # Final metrics
    poly = adopted_poly
    cr_final = _containment_ratio(poly, xy)

    vb_final = sweep_results[0]["vertex_before"] if sweep_results else 0
    va_final = (
        len(list(poly.exterior.coords))
        if hasattr(poly, "exterior")
        else sum(len(list(g.exterior.coords)) for g in poly.geoms)
        if hasattr(poly, "geoms")
        else 0
    )

    metrics = {
        "sweep_results": sweep_results,
        "adopted_ratio": adopted,
        "adopted_buffer_close_m": buffer_close_m,
        "adopted_buffer_open_m": buffer_open_m,
        "adopted_simplify_tol_m": simplify_tol_m,
        "area_m2": round(poly.area, 2),
        "containment_ratio": round(cr_final, 4),
        "vertex_count_before": vb_final,
        "vertex_count_after": va_final,
        "is_valid": poly.is_valid,
        "geometry_type": poly.geom_type,
    }

    return poly, metrics


def save_polygon_geojson(
    poly: "ShapelyPolygon",
    metrics: dict,
    out_path: Path,
    xy: np.ndarray,
    map_id: np.ndarray,
) -> None:
    """Save refined polygon as GeoJSON with properties."""
    feature = {
        "type": "Feature",
        "geometry": mapping(poly),
        "properties": {
            "area_m2": metrics["area_m2"],
            "vertex_count_before": metrics["vertex_count_before"],
            "vertex_count_after": metrics["vertex_count_after"],
            "containment_ratio": metrics["containment_ratio"],
            "concave_ratio": metrics["adopted_ratio"],
            "buffer_close_m": metrics["adopted_buffer_close_m"],
            "buffer_open_m": metrics["adopted_buffer_open_m"],
            "simplify_tol_m": metrics["adopted_simplify_tol_m"],
            "coordinate_system": "merged_v2 (A frame, ARKit XY plane)",
            "pose_count": len(xy),
            "session_0_count": int((map_id == 0).sum()),
            "session_1_count": int((map_id == 1).sum()),
        },
    }
    fc = {"type": "FeatureCollection", "features": [feature]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fc, indent=2))
    logger.info("Saved refined polygon GeoJSON → %s (area=%.1f m²)", out_path, metrics["area_m2"])
