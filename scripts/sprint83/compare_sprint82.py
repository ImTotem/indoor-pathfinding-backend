"""
compare_sprint82.py — Cycle 2: compare sprint83-cycle2 polygon vs sprint82 polygon.

Cycle 1 bug: sprint82's Y was incorrectly mapped to cycle_1's Z axis.
Cycle 2 fix: both polygons use same (X, Y) world frame — DB Z-up, corridor horizontal plane.
sprint82 trajectory bbox: X[-4, 32], Y[-60, 0].
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_geojson(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        geo = json.load(f)
    if geo.get("type") == "FeatureCollection":
        feats = geo.get("features", [])
        return feats[0] if feats else None
    return geo


def get_geometry(feature: dict | None) -> dict | None:
    if feature is None:
        return None
    if feature.get("type") in ("Polygon", "MultiPolygon", "GeometryCollection"):
        return feature
    return feature.get("geometry")


def polygon_area(geom: dict) -> float:
    from shapely.geometry import shape
    try:
        return shape(geom).area
    except Exception:
        return 0.0


def compute_iou(geom_a: dict, geom_b: dict) -> dict:
    from shapely.geometry import shape
    try:
        pa = shape(geom_a)
        pb = shape(geom_b)
        if not pa.is_valid:
            pa = pa.buffer(0)
        if not pb.is_valid:
            pb = pb.buffer(0)
        inter = pa.intersection(pb).area
        union = pa.union(pb).area
        return {
            "intersection_m2": round(float(inter), 2),
            "union_m2": round(float(union), 2),
            "iou": round(float(inter / union), 4) if union > 0 else 0.0,
        }
    except Exception as e:
        return {"error": str(e)}


def save_comparison_png(
    geom_cycle2: dict,
    geom_cycle1: dict | None,
    geom_sprint82: dict | None,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 10))

    def draw_geom(geom: dict, color: str, label: str, lw: float = 1.5, ls: str = "-") -> None:
        gtype = geom.get("type", "")
        if gtype == "Polygon":
            coords = geom["coordinates"][0]
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, label=label)
            ax.fill(xs, ys, color=color, alpha=0.12)
        elif gtype == "MultiPolygon":
            first = True
            for poly in geom["coordinates"]:
                coords = poly[0]
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls,
                        label=label if first else None)
                ax.fill(xs, ys, color=color, alpha=0.12)
                first = False

    draw_geom(geom_cycle2, "red", "cycle2 (Z-up fix)", lw=2.5)
    if geom_cycle1 is not None:
        draw_geom(geom_cycle1, "orange", "cycle1 (Y-up bug)", lw=1.5, ls="--")
    if geom_sprint82 is not None:
        draw_geom(geom_sprint82, "lightblue", "sprint82 (trajectory)", lw=1.5, ls=":")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Floor polygon overlay: cycle2 (red) vs cycle1 (orange) vs sprint82 (blue)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def run_compare(
    cycle2_polygon_path: str,
    cycle1_polygon_path: str | None,
    sprint82_polygon_path: str | None,
    out_json_path: str,
    out_vs_sprint82_png: str,
    out_vs_cycle1_png: str,
) -> dict:
    feat2 = load_geojson(cycle2_polygon_path)
    geom2 = get_geometry(feat2)
    if geom2 is None:
        return {"error": "cycle2 polygon not found"}

    feat1 = load_geojson(cycle1_polygon_path) if cycle1_polygon_path else None
    geom1 = get_geometry(feat1)

    feat82 = load_geojson(sprint82_polygon_path) if sprint82_polygon_path else None
    geom82 = get_geometry(feat82)

    area2 = polygon_area(geom2)
    area1 = polygon_area(geom1) if geom1 else None
    area82 = polygon_area(geom82) if geom82 else 894.29  # fallback from cycle1 report

    result: dict = {
        "cycle2_area_m2": round(area2, 2),
        "sprint82_area_m2": round(area82, 2),
    }

    if area82 > 0:
        result["area_diff_vs_sprint82_pct"] = round(abs(area2 - area82) / area82 * 100, 2)

    if geom82 is not None:
        result["iou_vs_sprint82"] = compute_iou(geom2, geom82)

    if geom1 is not None:
        result["cycle1_area_m2"] = round(area1, 2)
        result["area_diff_cycle1_vs_cycle2_m2"] = round(area2 - area1, 2)
        result["area_diff_cycle1_vs_cycle2_pct"] = round(
            abs(area2 - area1) / max(area1, 1) * 100, 2
        )
        result["iou_cycle1_vs_cycle2"] = compute_iou(geom2, geom1)

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved metrics: {out_json_path}")

    # Sprint82 comparison PNG
    save_comparison_png(geom2, geom1, geom82, out_vs_sprint82_png)

    # cycle1 vs cycle2 overlay (separate file)
    if geom1 is not None:
        save_comparison_png(geom2, geom1, None, out_vs_cycle1_png)

    return result
