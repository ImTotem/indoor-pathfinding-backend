"""Sprint 82 — Step 3: Visualization.

Renders polygon_overlay.png, nav_graph.png, combined.png.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

if TYPE_CHECKING:
    from shapely.geometry import Polygon, MultiPolygon
    from .build_nav_graph import NavNode, NavEdge

logger = logging.getLogger(__name__)

ShapelyPolygon = Union["Polygon", "MultiPolygon"]


def _plot_polygon(ax: plt.Axes, poly: "ShapelyPolygon", **kwargs) -> None:
    """Plot a Polygon or MultiPolygon on ax."""
    from shapely.geometry import MultiPolygon as MP

    polys = list(poly.geoms) if isinstance(poly, MP) else [poly]
    for p in polys:
        if p.is_empty:
            continue
        x, y = p.exterior.xy
        ax.fill(x, y, **{k: v for k, v in kwargs.items() if k not in ("edgecolor", "linestyle", "linewidth", "label")})
        ax.plot(x, y, color=kwargs.get("edgecolor", "black"),
                linestyle=kwargs.get("linestyle", "-"),
                linewidth=kwargs.get("linewidth", 1.0),
                label=kwargs.get("label", ""))


def render_polygon_overlay(
    baseline_geojson: Path,
    refined: "ShapelyPolygon",
    xy: np.ndarray,
    map_id: np.ndarray,
    out_png: Path,
) -> None:
    """Overlay: baseline polygon (gray) + refined (royalblue) + pose dots."""
    from shapely.geometry import shape

    fig, ax = plt.subplots(figsize=(12, 10))

    # Baseline polygon
    if baseline_geojson.exists():
        try:
            fc = json.loads(baseline_geojson.read_text())
            for feat in fc.get("features", []):
                base_poly = shape(feat["geometry"])
                _plot_polygon(
                    ax, base_poly,
                    color="gray", alpha=0.10,
                    edgecolor="gray", linestyle="--", linewidth=1.0,
                    label="baseline",
                )
        except Exception as e:
            logger.warning("Failed to load baseline GeoJSON: %s", e)

    # Refined polygon
    _plot_polygon(
        ax, refined,
        color="royalblue", alpha=0.20,
        edgecolor="navy", linestyle="-", linewidth=1.5,
        label="refined",
    )

    # Pose dots by session
    for mid, color, label in [(0, "royalblue", "session 0"), (1, "crimson", "session 1")]:
        mask = map_id == mid
        ax.scatter(xy[mask, 0], xy[mask, 1], s=6, c=color, alpha=0.6, label=label, zorder=3)

    # Legend patches
    patches = [
        mpatches.Patch(color="gray", alpha=0.4, label="baseline polygon"),
        mpatches.Patch(color="royalblue", alpha=0.4, label="refined polygon"),
        mpatches.Patch(color="royalblue", label="session 0 pose"),
        mpatches.Patch(color="crimson", label="session 1 pose"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("Floor Polygon: Baseline vs Refined")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close(fig)
    size_kb = out_png.stat().st_size // 1024
    logger.info("Saved polygon overlay PNG → %s (%d KB)", out_png, size_kb)


def render_nav_graph(
    polygon: "ShapelyPolygon",
    nodes: "list[NavNode]",
    edges: "list[NavEdge]",
    out_png: Path,
) -> None:
    """Nav graph: refined polygon + nodes (colored by component) + edges."""
    import matplotlib.cm as cm

    active = [n for n in nodes if n.component_id >= 0]
    if not active:
        logger.warning("No active nodes to render")
        return

    comp_ids = sorted(set(n.component_id for n in active))
    cmap = cm.get_cmap("tab10", max(len(comp_ids), 1))
    comp_color = {cid: cmap(i) for i, cid in enumerate(comp_ids)}

    node_map = {n.node_id: (n.x, n.y) for n in active}
    active_ids = {n.node_id for n in active}

    fig, ax = plt.subplots(figsize=(14, 11))

    # Polygon fill
    _plot_polygon(
        ax, polygon,
        color="royalblue", alpha=0.15,
        edgecolor="navy", linestyle="-", linewidth=1.5,
    )

    # Edges
    grid_plotted = loop_plotted = 0
    for e in edges:
        if e.from_id not in active_ids or e.to_id not in active_ids:
            continue
        a = node_map[e.from_id]
        b = node_map[e.to_id]
        if e.kind in ("loop_closure", "neighbor"):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="lime", lw=1.0, alpha=0.6, zorder=2)
            loop_plotted += 1
        else:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="gray", lw=0.5, alpha=0.5, zorder=2)
            grid_plotted += 1

    # Nodes colored by component
    for n in active:
        color = comp_color[n.component_id]
        ax.scatter(n.x, n.y, s=25, c=[color], zorder=4, edgecolors="white", linewidths=0.3)

    # Legend
    comp_sizes = Counter(n.component_id for n in active)
    patches = []
    for cid in comp_ids[:8]:  # max 8 legend entries
        patches.append(
            mpatches.Patch(color=comp_color[cid], label=f"comp {cid} (n={comp_sizes[cid]})")
        )
    patches += [
        mpatches.Patch(color="gray", alpha=0.6, label=f"grid edge ({grid_plotted})"),
        mpatches.Patch(color="lime", alpha=0.8, label=f"loop/neighbor edge ({loop_plotted})"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=7)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(f"Navigation Graph ({len(active)} nodes, {len(edges)} edges, {len(comp_ids)} components)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close(fig)
    size_kb = out_png.stat().st_size // 1024
    logger.info("Saved nav_graph PNG → %s (%d KB)", out_png, size_kb)


def render_combined(
    polygon_png: Path,
    navgraph_png: Path,
    out_png: Path,
) -> None:
    """Side-by-side combined PNG."""
    from PIL import Image

    imgs = []
    for p in [polygon_png, navgraph_png]:
        if p.exists():
            imgs.append(np.array(Image.open(str(p))))

    if not imgs:
        logger.warning("No source PNGs for combined render")
        return

    if len(imgs) == 1:
        combined = imgs[0]
    else:
        h = max(im.shape[0] for im in imgs)
        padded = []
        for im in imgs:
            if im.shape[0] < h:
                pad = np.full((h - im.shape[0], im.shape[1], im.shape[2]), 255, dtype=np.uint8)
                im = np.vstack([im, pad])
            padded.append(im)
        combined = np.hstack(padded)

    fig, ax = plt.subplots(figsize=(combined.shape[1] / 150, combined.shape[0] / 150))
    ax.imshow(combined)
    ax.axis("off")
    ax.set_title("Sprint 82 — Floor Polygon + Navigation Graph")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close(fig)
    size_kb = out_png.stat().st_size // 1024
    logger.info("Saved combined PNG → %s (%d KB)", out_png, size_kb)
