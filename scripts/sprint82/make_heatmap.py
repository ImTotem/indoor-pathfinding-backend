#!/usr/bin/env python3
"""Density heatmap of merged_v2.db poses with refined polygon overlay.

Goal: diagnose why the refined floor polygon shape feels off — render a 2D
density map of pose XY split by map_id (scan A=0, scan B=1) and overlay the
adopted polygon + sweep alternatives so the user can see where the hull is
cutting too tight or too loose.
"""

from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon as MplPolygon

ROOT = Path("/Users/leehyeonsu/home/koreatech/graduate_project/Indoor-pathfinding-v2")
DB = ROOT / "_workspace/sprint81-frame-pose-rtabmap-merge/evidence/cycle2/merged_v2.db"
POLY_DIR = ROOT / "_workspace/sprint82-floor-polygon-navgraph/evidence/cycle1/polygon"
OUT_DIR = ROOT / "_workspace/sprint82-floor-polygon-navgraph/evidence/cycle1/heatmap"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_poses() -> tuple[np.ndarray, np.ndarray]:
    """Return (xy[N,2], map_id[N]) from merged_v2.db Node table."""
    con = sqlite3.connect(DB)
    rows = con.execute("SELECT id, map_id, pose FROM Node WHERE pose IS NOT NULL").fetchall()
    con.close()
    xs, ys, mids = [], [], []
    for nid, mid, blob in rows:
        if blob is None or len(blob) < 12 * 4:
            continue
        # RTAB-Map stores 3x4 row-major float32 [R|t]; tx/ty/tz are at indices 3,7,11.
        floats = struct.unpack("<12f", blob[: 12 * 4])
        tx, ty = floats[3], floats[7]
        if abs(tx) < 1e-9 and abs(ty) < 1e-9 and nid != 1:
            continue
        xs.append(tx)
        ys.append(ty)
        mids.append(mid)
    return np.column_stack([xs, ys]), np.array(mids)


def load_polygon(name: str = "floor_polygon_refined.geojson") -> np.ndarray:
    g = json.loads((POLY_DIR / name).read_text())
    coords = g["features"][0]["geometry"]["coordinates"][0]
    return np.array(coords)


def render_combined(xy: np.ndarray, mids: np.ndarray, poly: np.ndarray, out: Path) -> None:
    bin_size = 0.5  # meters
    pad = 5.0
    x0, y0 = xy[:, 0].min() - pad, xy[:, 1].min() - pad
    x1, y1 = xy[:, 0].max() + pad, xy[:, 1].max() + pad
    nx = int(np.ceil((x1 - x0) / bin_size))
    ny = int(np.ceil((y1 - y0) / bin_size))
    H, xedges, yedges = np.histogram2d(
        xy[:, 0], xy[:, 1], bins=[nx, ny], range=[[x0, x1], [y0, y1]]
    )

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Panel 1: log density combined.
    ax = axes[0]
    im = ax.imshow(
        H.T,
        origin="lower",
        extent=[x0, x1, y0, y1],
        cmap="inferno",
        norm=LogNorm(vmin=1, vmax=max(H.max(), 2)),
        aspect="equal",
    )
    ax.add_patch(MplPolygon(poly, closed=True, fill=False, edgecolor="cyan", lw=2.0, label="adopted polygon (ratio=0.70)"))
    ax.set_title(f"Pose density (log) — bin={bin_size}m, total={len(xy)} poses")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="poses / bin")

    # Panel 2: per-scan scatter with polygon.
    ax = axes[1]
    for mid, color, label in [(0, "tab:blue", "scan A (map_id=0)"), (1, "tab:orange", "scan B (map_id=1)")]:
        mask = mids == mid
        ax.scatter(xy[mask, 0], xy[mask, 1], s=8, c=color, alpha=0.55, label=f"{label} n={int(mask.sum())}")
    ax.add_patch(MplPolygon(poly, closed=True, fill=False, edgecolor="black", lw=1.8, label="polygon"))
    ax.set_title("Per-scan pose scatter (after T_AB alignment)")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: kernel-ish density (gaussian blur on histogram) + polygon ambiguity check.
    from scipy.ndimage import gaussian_filter
    Hs = gaussian_filter(H, sigma=2.0)
    ax = axes[2]
    im = ax.imshow(
        Hs.T,
        origin="lower",
        extent=[x0, x1, y0, y1],
        cmap="viridis",
        aspect="equal",
    )
    ax.add_patch(MplPolygon(poly, closed=True, fill=False, edgecolor="red", lw=2.0))
    ax.set_title(f"Smoothed density (σ=2 bins) — polygon area=894.3m²")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    fig.colorbar(im, ax=ax, label="smoothed count")

    fig.suptitle("sprint82 — pose density vs refined polygon (diagnose hull tightness)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out} ({out.stat().st_size} bytes)")


def render_zoom_with_alts(xy: np.ndarray, mids: np.ndarray, out: Path) -> None:
    """Overlay multiple ratio sweep polygons to see how shape changes."""
    sweep_path = ROOT / "_workspace/sprint82-floor-polygon-navgraph/evidence/cycle1/polygon/polygon_metrics.json"
    metrics = json.loads(sweep_path.read_text())
    sweep = metrics.get("sweep_results", [])
    if not sweep:
        print("no sweep_results found, skipping alt overlay")
        return

    fig, ax = plt.subplots(figsize=(11, 9))
    bin_size = 0.5
    pad = 4.0
    x0, y0 = xy[:, 0].min() - pad, xy[:, 1].min() - pad
    x1, y1 = xy[:, 0].max() + pad, xy[:, 1].max() + pad
    nx = int(np.ceil((x1 - x0) / bin_size))
    ny = int(np.ceil((y1 - y0) / bin_size))
    H, *_ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[nx, ny], range=[[x0, x1], [y0, y1]])

    ax.imshow(
        H.T,
        origin="lower",
        extent=[x0, x1, y0, y1],
        cmap="gray_r",
        norm=LogNorm(vmin=1, vmax=max(H.max(), 2)),
        aspect="equal",
    )
    from shapely import concave_hull
    from shapely.geometry import MultiPoint
    pts = MultiPoint([(float(p[0]), float(p[1])) for p in xy])
    cmap = plt.get_cmap("plasma")
    ratios = sorted({s["ratio"] for s in sweep})
    for i, r in enumerate(ratios):
        s = next(x for x in sweep if x["ratio"] == r)
        try:
            hull = concave_hull(pts, ratio=r)
            geom = hull.geoms[0] if hasattr(hull, "geoms") else hull
            arr = np.array(list(geom.exterior.coords))
        except Exception as exc:
            print(f"ratio={r} concave_hull failed: {exc}")
            continue
        color = cmap(i / max(1, len(ratios) - 1))
        ax.plot(
            np.r_[arr[:, 0], arr[0, 0]],
            np.r_[arr[:, 1], arr[0, 1]],
            color=color,
            lw=1.4,
            alpha=0.85,
            label=f"ratio={r:.2f} area={s.get('area_m2', float('nan')):.0f}m² v={s.get('vertex_after', s.get('vertex_before','?'))}",
        )
    ax.set_title("Ratio sweep overlay — pick where the hull stops cutting through dense pose mass")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out} ({out.stat().st_size} bytes)")


def main() -> None:
    xy, mids = load_poses()
    poly = load_polygon()
    print(f"loaded {len(xy)} poses, polygon vertices={len(poly)}")
    print(f"map_id distribution: {dict(zip(*np.unique(mids, return_counts=True)))}")
    print(f"x range: [{xy[:,0].min():.2f}, {xy[:,0].max():.2f}]")
    print(f"y range: [{xy[:,1].min():.2f}, {xy[:,1].max():.2f}]")
    render_combined(xy, mids, poly, OUT_DIR / "heatmap_combined.png")
    render_zoom_with_alts(xy, mids, OUT_DIR / "heatmap_ratio_sweep.png")
    summary = {
        "total_poses": int(len(xy)),
        "map_id_counts": {int(k): int(v) for k, v in zip(*np.unique(mids, return_counts=True))},
        "bbox": {
            "x_min": float(xy[:, 0].min()),
            "x_max": float(xy[:, 0].max()),
            "y_min": float(xy[:, 1].min()),
            "y_max": float(xy[:, 1].max()),
        },
        "polygon_vertices": int(len(poly)),
    }
    (OUT_DIR / "heatmap_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
