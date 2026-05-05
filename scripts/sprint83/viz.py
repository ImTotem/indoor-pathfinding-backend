"""
viz.py — visualization helpers: mask overlay PNGs, density heatmaps, polygon overlays.
"""

from __future__ import annotations

import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image


def save_mask_overlay(
    image_path: str,
    mask: np.ndarray,  # bool (H, W)
    out_path: str,
    alpha: float = 0.4,
    color_bgr: tuple = (0, 255, 0),  # green overlay
) -> None:
    """Save original image with floor mask overlay as PNG."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")
    overlay = img.copy()
    green_layer = np.zeros_like(img)
    green_layer[mask] = color_bgr
    blended = cv2.addWeighted(img, 1.0 - alpha, green_layer, alpha, 0)
    # Only apply blend where mask is True
    result = img.copy()
    result[mask] = blended[mask]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 6])


def save_density_heatmap(
    grid: np.ndarray,    # (nz, nx) counts
    origin: tuple,       # (xmin, zmin, bin_size)
    out_path: str,
    title: str = "Floor density",
    log_scale: bool = True,
) -> None:
    """Save density grid as heatmap PNG (viridis, axes in meters)."""
    xmin, zmin, bin_size = origin
    nz, nx = grid.shape
    xmax = xmin + nx * bin_size
    zmax = zmin + nz * bin_size

    fig, ax = plt.subplots(figsize=(10, 8))
    data = np.log1p(grid) if log_scale else grid
    im = ax.imshow(
        data,
        origin="lower",
        extent=[xmin, xmax, zmin, zmax],
        cmap="viridis",
        aspect="equal",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="log(count+1)" if log_scale else "count")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_polygon_overlay(
    grid: np.ndarray,
    origin: tuple,
    polygon_geojson_path: str,
    sprint82_geojson_path: str | None,
    out_path: str,
) -> None:
    """
    Overlay sprint83 polygon (red) and optional sprint82 polygon (gray) on density heatmap.
    """
    import json
    from shapely.geometry import shape as shapely_shape

    xmin, zmin, bin_size = origin
    nz, nx = grid.shape
    xmax = xmin + nx * bin_size
    zmax = zmin + nz * bin_size

    fig, ax = plt.subplots(figsize=(12, 10))
    data = np.log1p(grid)
    ax.imshow(
        data,
        origin="lower",
        extent=[xmin, xmax, zmin, zmax],
        cmap="viridis",
        aspect="equal",
        interpolation="nearest",
    )

    # Load and draw sprint83 polygon
    with open(polygon_geojson_path) as f:
        geo83 = json.load(f)
    geom83 = geo83["geometry"]
    _draw_geojson_polygon(ax, geom83, color="red", label="sprint83 (segformer)", lw=2)

    # Draw sprint82 polygon if available
    if sprint82_geojson_path and os.path.exists(sprint82_geojson_path):
        with open(sprint82_geojson_path) as f:
            geo82 = json.load(f)
        # Support FeatureCollection, Feature, or raw geometry
        if geo82.get("type") == "FeatureCollection":
            features = geo82.get("features", [])
            if features:
                geom82 = features[0].get("geometry", features[0])
            else:
                geom82 = None
        elif geo82.get("type") == "Feature":
            geom82 = geo82.get("geometry")
        else:
            geom82 = geo82
        if geom82:
            _draw_geojson_polygon(ax, geom82, color="lightgray", label="sprint82 (trajectory)", lw=1.5, linestyle="--")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Floor polygon comparison: sprint83 vs sprint82")
    ax.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _draw_geojson_polygon(ax, geom: dict, color: str, label: str, lw: float = 1.5, linestyle: str = "-") -> None:
    gtype = geom.get("type", "")
    if gtype == "Polygon":
        coords = geom["coordinates"][0]
        xs = [c[0] for c in coords]
        zs = [c[1] for c in coords]
        ax.plot(xs, zs, color=color, linewidth=lw, linestyle=linestyle, label=label)
        ax.fill(xs, zs, color=color, alpha=0.15)
    elif gtype == "MultiPolygon":
        first = True
        for poly in geom["coordinates"]:
            coords = poly[0]
            xs = [c[0] for c in coords]
            zs = [c[1] for c in coords]
            ax.plot(xs, zs, color=color, linewidth=lw, linestyle=linestyle,
                    label=label if first else None)
            ax.fill(xs, zs, color=color, alpha=0.15)
            first = False
    elif gtype == "GeometryCollection":
        for sub in geom.get("geometries", []):
            _draw_geojson_polygon(ax, sub, color, label, lw, linestyle)
