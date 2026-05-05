"""
build_heatmap_polygon.py — Cycle 2: density grid + polygon from world (X, Y) points.

Axis change from cycle_1:
  - cycle_1: output (X, Z) — used ARKit +Y-up frame
  - cycle_2: output (X, Y) — DB Z-up frame, corridor horizontal plane
  - heatmap axes: X (m) horizontal, Y (m) vertical
  - sprint82 trajectory bbox: X[-4,32], Y[-60,0] — same (X,Y) plane
"""

from __future__ import annotations

import json
import math
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Density grid ──────────────────────────────────────────────────────────────

def build_density_grid(
    points: np.ndarray,   # (N, 2) float — columns = (X, Y)
    bin_size: float = 0.2,
) -> tuple[np.ndarray, tuple]:
    """
    Build 2D histogram grid in (X, Y) plane.
    Returns (grid (ny, nx) counts, origin (xmin, ymin, bin_size)).
    """
    if len(points) == 0:
        raise ValueError("No points to build density grid")

    xs = points[:, 0]
    ys = points[:, 1]

    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())

    nx = max(1, int(math.ceil((xmax - xmin) / bin_size)) + 1)
    ny = max(1, int(math.ceil((ymax - ymin) / bin_size)) + 1)

    H, xedges, yedges = np.histogram2d(
        xs, ys,
        bins=[nx, ny],
        range=[[xmin, xmax + bin_size], [ymin, ymax + bin_size]],
    )
    grid = H.T  # (ny, nx), yi=row, xi=col
    origin = (xmin, ymin, float(bin_size))
    return grid.astype(np.float32), origin


# ── Polygon extraction ─────────────────────────────────────────────────────────

def extract_polygon(
    grid: np.ndarray,    # (ny, nx) float
    origin: tuple,       # (xmin, ymin, bin_size)
    min_count: int = 3,
    min_area_m2: float = 5.0,
    morph_kernel: int = 5,
    morph_iters: int = 2,
    epsilon_ratio: float = 1.5,
) -> tuple[dict | None, dict]:
    """
    Grid → occupancy → contour → GeoJSON Polygon.
    Output coordinates in (X, Y) world frame.
    """
    xmin, ymin, bin_size = origin
    ny, nx = grid.shape

    occupancy = (grid >= min_count).astype(np.uint8)
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    for _ in range(morph_iters):
        occupancy = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(occupancy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    min_area_cells = min_area_m2 / (bin_size ** 2)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area_cells]
    valid_contours.sort(key=cv2.contourArea, reverse=True)

    metrics = {
        "total_contours": len(contours),
        "valid_contours": len(valid_contours),
        "occupancy_cells": int(occupancy.sum()),
        "min_count": min_count,
        "bin_size_m": bin_size,
        "morph_kernel": morph_kernel,
        "morph_iters": morph_iters,
    }

    if not valid_contours:
        return None, metrics

    epsilon_cells = bin_size * epsilon_ratio / bin_size
    all_polygons = []

    for c in valid_contours[:3]:
        approx = cv2.approxPolyDP(c, epsilon_cells, closed=True)
        coords = approx.squeeze(1)  # (M, 2): (xi, yi) pixel indices
        world_coords = [
            [round(float(xmin + xi * bin_size), 4),
             round(float(ymin + yi * bin_size), 4)]
            for xi, yi in coords
        ]
        if world_coords[0] != world_coords[-1]:
            world_coords.append(world_coords[0])
        all_polygons.append(world_coords)

    primary = all_polygons[0]
    area_cells = float(cv2.contourArea(valid_contours[0]))
    area_m2 = area_cells * (bin_size ** 2)

    coords_arr = np.array(primary[:-1])
    bbox = {
        "xmin": round(float(coords_arr[:, 0].min()), 3),
        "xmax": round(float(coords_arr[:, 0].max()), 3),
        "ymin": round(float(coords_arr[:, 1].min()), 3),
        "ymax": round(float(coords_arr[:, 1].max()), 3),
    }

    metrics.update({
        "polygon_vertex_count": len(primary) - 1,
        "polygon_area_m2": round(area_m2, 2),
        "polygon_bbox": bbox,
        "num_polygons_kept": len(all_polygons),
    })

    if len(all_polygons) == 1:
        geometry = {"type": "Polygon", "coordinates": [primary]}
    else:
        geometry = {"type": "MultiPolygon", "coordinates": [[ring] for ring in all_polygons]}

    geojson = {
        "type": "Feature",
        "geometry": geometry,
        "properties": {
            "bin_size_m": bin_size,
            "min_count": min_count,
            "area_m2": round(area_m2, 2),
            "vertex_count": len(primary) - 1,
            "num_components": len(all_polygons),
            "source": "sprint83-cycle2-segformer-raycast-Zup",
            "coordinate_frame": "DB Z-up world (X=corridor width, Y=corridor length)",
        },
    }
    return geojson, metrics


# ── Visualization ──────────────────────────────────────────────────────────────

def save_density_heatmap(
    grid: np.ndarray,
    origin: tuple,
    out_path: str,
    title: str = "Floor density",
    log_scale: bool = True,
) -> None:
    xmin, ymin, bin_size = origin
    ny, nx = grid.shape
    xmax = xmin + nx * bin_size
    ymax = ymin + ny * bin_size

    fig, ax = plt.subplots(figsize=(10, 8))
    data = np.log1p(grid) if log_scale else grid
    im = ax.imshow(
        data, origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        cmap="viridis", aspect="equal", interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="log(count+1)" if log_scale else "count")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_per_scan_heatmap(
    grid_A: np.ndarray, origin_A: tuple,
    grid_B: np.ndarray, origin_B: tuple,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, grid, origin, label in [
        (axes[0], grid_A, origin_A, "Scan A (cycle2)"),
        (axes[1], grid_B, origin_B, "Scan B (cycle2)"),
    ]:
        xmin, ymin, bs = origin
        ny, nx = grid.shape
        im = ax.imshow(
            np.log1p(grid), origin="lower",
            extent=[xmin, xmin + nx * bs, ymin, ymin + ny * bs],
            cmap="viridis", aspect="equal",
        )
        plt.colorbar(im, ax=ax)
        ax.set_title(label)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_polygon_overlay(
    grid: np.ndarray,
    origin: tuple,
    polygon_geojson_path: str,
    sprint82_geojson_path: str | None,
    out_path: str,
    title: str = "Floor polygon comparison (cycle2 vs sprint82)",
) -> None:
    xmin, ymin, bin_size = origin
    ny, nx = grid.shape
    xmax = xmin + nx * bin_size
    ymax = ymin + ny * bin_size

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(
        np.log1p(grid), origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        cmap="viridis", aspect="equal", interpolation="nearest",
    )

    with open(polygon_geojson_path) as f:
        geo83 = json.load(f)
    _draw_polygon(ax, geo83["geometry"], color="red", label="cycle2 (Z-up fix)", lw=2)

    if sprint82_geojson_path and os.path.exists(sprint82_geojson_path):
        with open(sprint82_geojson_path) as f:
            geo82 = json.load(f)
        geom82 = _extract_geometry(geo82)
        if geom82:
            _draw_polygon(ax, geom82, color="lightgray", label="sprint82 (trajectory)", lw=1.5, linestyle="--")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _extract_geometry(geo: dict) -> dict | None:
    if geo.get("type") == "FeatureCollection":
        feats = geo.get("features", [])
        return feats[0].get("geometry") if feats else None
    elif geo.get("type") == "Feature":
        return geo.get("geometry")
    return geo


def _draw_polygon(ax, geom: dict, color: str, label: str, lw: float = 1.5, linestyle: str = "-") -> None:
    gtype = geom.get("type", "")
    if gtype == "Polygon":
        coords = geom["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        ax.plot(xs, ys, color=color, linewidth=lw, linestyle=linestyle, label=label)
        ax.fill(xs, ys, color=color, alpha=0.15)
    elif gtype == "MultiPolygon":
        first = True
        for poly in geom["coordinates"]:
            coords = poly[0]
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            ax.plot(xs, ys, color=color, linewidth=lw, linestyle=linestyle,
                    label=label if first else None)
            ax.fill(xs, ys, color=color, alpha=0.15)
            first = False
    elif gtype == "GeometryCollection":
        for sub in geom.get("geometries", []):
            _draw_polygon(ax, sub, color, label, lw, linestyle)


def compute_iou(poly_a_coords: list, poly_b_coords: list) -> float:
    from shapely.geometry import Polygon
    pa = Polygon(poly_a_coords[0])
    pb = Polygon(poly_b_coords[0])
    if not pa.is_valid:
        pa = pa.buffer(0)
    if not pb.is_valid:
        pb = pb.buffer(0)
    intersection = pa.intersection(pb).area
    union = pa.union(pb).area
    return round(float(intersection / union), 4) if union > 0 else 0.0
