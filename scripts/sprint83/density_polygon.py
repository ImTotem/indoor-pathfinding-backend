"""
density_polygon.py — density grid from world (X,Z) points → occupancy → contour polygon.
"""

from __future__ import annotations

import json
import math

import cv2
import numpy as np


def build_density_grid(
    points: np.ndarray,  # (N, 2) float, columns = (X, Z)
    bin_size: float = 0.2,
) -> tuple[np.ndarray, tuple]:
    """
    Build 2D histogram grid.
    Returns (H grid (nz, nx) counts, origin tuple (xmin, zmin, bin_size)).
    Axes: grid[zi, xi] = count of points in that cell.
    """
    if len(points) == 0:
        raise ValueError("No points to build density grid")

    xs = points[:, 0]
    zs = points[:, 1]

    xmin, xmax = xs.min(), xs.max()
    zmin, zmax = zs.min(), zs.max()

    nx = max(1, int(math.ceil((xmax - xmin) / bin_size)) + 1)
    nz = max(1, int(math.ceil((zmax - zmin) / bin_size)) + 1)

    # np.histogram2d: shape is (nx, nz) — we want (nz, nx) image convention
    H, xedges, zedges = np.histogram2d(
        xs, zs,
        bins=[nx, nz],
        range=[[xmin, xmax + bin_size], [zmin, zmax + bin_size]],
    )
    # H shape: (nx, nz) → transpose to (nz, nx) for image coords
    grid = H.T  # (nz, nx), zi=row, xi=col

    origin = (float(xmin), float(zmin), float(bin_size))
    return grid.astype(np.float32), origin


def extract_polygon(
    grid: np.ndarray,      # (nz, nx) float
    origin: tuple,         # (xmin, zmin, bin_size)
    min_count: int = 3,
    min_area_m2: float = 5.0,
    morph_kernel: int = 5,
    morph_iters: int = 2,
    epsilon_ratio: float = 1.5,
) -> tuple[dict | None, dict]:
    """
    Grid → occupancy → contour → GeoJSON Polygon.
    Returns (geojson_dict or None, metrics dict).
    """
    xmin, zmin, bin_size = origin
    nz, nx = grid.shape

    occupancy = (grid >= min_count).astype(np.uint8)
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    for _ in range(morph_iters):
        occupancy = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(occupancy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Filter by area
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

    # Simplify and convert to world coords
    epsilon = bin_size * epsilon_ratio
    epsilon_cells = epsilon / bin_size
    all_polygons = []

    for c in valid_contours[:3]:  # keep max 3 largest
        approx = cv2.approxPolyDP(c, epsilon_cells, closed=True)
        coords = approx.squeeze(1)  # (M, 2): (xi, zi)
        world_coords = [
            [round(float(xmin + xi * bin_size), 4),
             round(float(zmin + zi * bin_size), 4)]
            for xi, zi in coords
        ]
        # Close polygon
        if world_coords[0] != world_coords[-1]:
            world_coords.append(world_coords[0])
        all_polygons.append(world_coords)

    # Primary polygon = largest
    primary = all_polygons[0]
    area_cells = float(cv2.contourArea(valid_contours[0]))
    area_m2 = area_cells * (bin_size ** 2)

    # Compute bbox in world coords
    coords_arr = np.array(primary[:-1])
    bbox = {
        "xmin": round(float(coords_arr[:, 0].min()), 3),
        "xmax": round(float(coords_arr[:, 0].max()), 3),
        "zmin": round(float(coords_arr[:, 1].min()), 3),
        "zmax": round(float(coords_arr[:, 1].max()), 3),
    }

    metrics.update({
        "polygon_vertex_count": len(primary) - 1,
        "polygon_area_m2": round(area_m2, 2),
        "polygon_bbox": bbox,
        "num_polygons_kept": len(all_polygons),
    })

    # Build GeoJSON
    if len(all_polygons) == 1:
        geometry = {
            "type": "Polygon",
            "coordinates": [primary],
        }
    else:
        # MultiPolygon: each polygon = [exterior_ring] = [[coord, ...]]
        geometry = {
            "type": "MultiPolygon",
            "coordinates": [[ring] for ring in all_polygons],
        }

    geojson = {
        "type": "Feature",
        "geometry": geometry,
        "properties": {
            "bin_size_m": bin_size,
            "min_count": min_count,
            "area_m2": round(area_m2, 2),
            "vertex_count": len(primary) - 1,
            "num_components": len(all_polygons),
            "source": "sprint83-segformer-raycast",
        },
    }
    return geojson, metrics


def compute_iou(poly_a_coords: list, poly_b_coords: list) -> float:
    """IoU between two GeoJSON polygon coordinate rings."""
    from shapely.geometry import Polygon
    pa = Polygon(poly_a_coords[0])
    pb = Polygon(poly_b_coords[0])
    if not pa.is_valid:
        pa = pa.buffer(0)
    if not pb.is_valid:
        pb = pb.buffer(0)
    intersection = pa.intersection(pb).area
    union = pa.union(pb).area
    if union == 0:
        return 0.0
    return round(float(intersection / union), 4)
