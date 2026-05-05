"""Run a trained U-Net on a real RTABMap obstacle heatmap.

Input  : obstacle_heatmap_counts.npz (counts, origin_x, origin_y, cell_size_m)
Output : <out>/predicted_mask.png, <out>/polygon_overlay.png, <out>/polygon.geojson
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from uuid import uuid4

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw
from scipy import ndimage
from shapely.geometry import MultiPolygon, Polygon, mapping
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SPRINT_59 = PROJECT_ROOT / "_workspace" / "sprint_59_polygon_to_graph"
if SPRINT_59.exists() and str(SPRINT_59) not in sys.path:
    sys.path.insert(0, str(SPRINT_59))

from indoor_server.application.building.steps.node_placement import (  # noqa: E402
    NodePlacementStep,
)
from indoor_server.application.building.steps.skeletonize import (  # noqa: E402
    SkeletonizeStep,
)
from indoor_server.domain.building.models import GridOrigin, WalkableGrid  # noqa: E402

from model import build_unet, select_device  # noqa: E402
from path_corridor_graph import (  # type: ignore[import-not-found]  # noqa: E402
    PathGraph,
    chaikin_smooth_path_inside,
    extract_path_graph_from_skeleton,
    graph_metrics,
    graph_to_geojson,
    prune_short_leaf_edges,
)

INFERNO = colormaps["inferno"]
IMAGE_PIXEL_TO_M = 0.05
GRAPH_CELL_M = 0.10
SIMPLIFY_TOL_PX = 4.0
MIN_AREA_PX = 1500


def render_heatmap_as_inferno(counts: np.ndarray, min_counts: int = 0) -> np.ndarray:
    """RTABMap counts -> inferno BGR like the generator's input style.

    Args:
        counts: int array of obstacle-hit counts per cell.
        min_counts: cells with counts < min_counts are set to 0 (black).
            Use this to drop low-intensity 'smear' (people/desks) and keep
            only walls.
    """
    counts_used = counts.copy()
    if min_counts > 0:
        counts_used[counts_used < min_counts] = 0
    log_counts = np.log1p(counts_used.astype(np.float32))
    normed = log_counts / max(log_counts.max(), 1e-6)
    rgba = INFERNO(normed)
    bgr = (rgba[..., [2, 1, 0]] * 255).astype(np.uint8)
    bgr[counts_used <= 0] = 0
    return bgr


def strip_dark_purple_bgr(img_bgr: np.ndarray, g_threshold: int = 60) -> np.ndarray:
    """Sprint 60 parity: zero rendered inferno pixels whose G channel is low."""
    out = img_bgr.copy()
    out[out[:, :, 1] < g_threshold] = 0
    return out


def _dominant_angle(ring: list[tuple[float, float]]) -> float:
    """Length-weighted mean angle after folding horizontal/vertical into one bin."""
    if len(ring) < 2:
        return 0.0
    total_w = 0.0
    total_aw = 0.0
    for i, (x1, y1) in enumerate(ring):
        x2, y2 = ring[(i + 1) % len(ring)]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        angle = math.atan2(dy, dx) % (math.pi / 2.0)
        if angle > math.pi / 4.0:
            angle -= math.pi / 2.0
        total_aw += angle * length
        total_w += length
    if total_w < 1e-6:
        return 0.0
    return total_aw / total_w


def _cluster_1d(values: list[float], tolerance: float) -> dict[float, float]:
    if not values:
        return {}
    sorted_values = sorted(set(values))
    clusters: list[list[float]] = [[sorted_values[0]]]
    for value in sorted_values[1:]:
        if value - clusters[-1][-1] <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    result: dict[float, float] = {}
    for cluster in clusters:
        mean = sum(cluster) / len(cluster)
        for value in cluster:
            result[value] = mean
    return result


def _force_rectilinear(
    ring_rot: list[tuple[float, float]],
    *,
    cluster_tol_m: float,
) -> list[tuple[float, float]]:
    if len(ring_rot) < 3:
        return ring_rot
    xs = [x for x, _ in ring_rot]
    ys = [y for _, y in ring_rot]
    x_map = _cluster_1d(xs, cluster_tol_m)
    y_map = _cluster_1d(ys, cluster_tol_m)
    return [(x_map[x], y_map[y]) for x, y in ring_rot]


def manhattan_rectify(
    poly: Polygon,
    *,
    grid_m: float = 0.10,
    cluster_tol_m: float = 0.40,
) -> tuple[Polygon | None, dict[str, object]]:
    """Rectify a neural polygon with Sprint 60 dominant-mean Manhattan cleanup."""
    ring = list(poly.exterior.coords)[:-1]
    if len(ring) < 4:
        return None, {"fail_reason": "too_few_vertices"}

    theta = _dominant_angle(ring)
    cos_t = math.cos(-theta)
    sin_t = math.sin(-theta)
    rotated = [(x * cos_t - y * sin_t, x * sin_t + y * cos_t) for x, y in ring]
    rectified = _force_rectilinear(rotated, cluster_tol_m=cluster_tol_m)
    snapped = [
        (round(x / grid_m) * grid_m, round(y / grid_m) * grid_m)
        for x, y in rectified
    ]

    deduped: list[tuple[float, float]] = []
    for point in snapped:
        if not deduped or point != deduped[-1]:
            deduped.append(point)
    if deduped and deduped[0] == deduped[-1]:
        deduped.pop()
    if len(deduped) < 3:
        return None, {"fail_reason": "deduped_too_few_vertices"}

    merged: list[tuple[float, float]] = []
    for i, cur in enumerate(deduped):
        prev = deduped[(i - 1) % len(deduped)]
        nxt = deduped[(i + 1) % len(deduped)]
        v1 = (cur[0] - prev[0], cur[1] - prev[1])
        v2 = (nxt[0] - cur[0], nxt[1] - cur[1])
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        if abs(cross) < 1e-9 and dot > 0:
            continue
        merged.append(cur)
    if len(merged) < 3:
        return None, {"fail_reason": "merged_too_few_vertices"}

    cos_b = math.cos(theta)
    sin_b = math.sin(theta)
    back = [(x * cos_b - y * sin_b, x * sin_b + y * cos_b) for x, y in merged]
    out = Polygon(back)
    if not out.is_valid:
        out = out.buffer(0)
    if isinstance(out, MultiPolygon):
        out = max(out.geoms, key=lambda p: p.area)
    if out.is_empty or not isinstance(out, Polygon):
        return None, {"fail_reason": "invalid_rectified_polygon"}
    return out, {
        "dominant_angle_deg": math.degrees(theta),
        "rectify_grid_m": grid_m,
        "rectify_cluster_tol_m": cluster_tol_m,
        "vertex_count_after_cluster": len(rectified),
        "vertex_count_after_dedup": len(deduped),
        "vertex_count_after_collinear_merge": len(merged),
    }


def _polygon_feature(poly: Polygon, properties: dict[str, object]) -> dict[str, object]:
    return {
        "type": "Feature",
        "geometry": mapping(poly),
        "properties": properties,
    }


def _write_single_polygon_geojson(
    out_path: Path,
    *,
    poly: Polygon,
    properties: dict[str, object],
) -> None:
    out_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [_polygon_feature(poly, properties)],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _mask700_to_polygon(mask_700: np.ndarray, *, pixel_to_m: float) -> tuple[Polygon | None, dict[str, object]]:
    """Sprint 60 polygon extraction: largest mask_700 contour -> local meters."""
    contours, _ = cv2.findContours(
        mask_700.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    info: dict[str, object] = {"contour_count": len(contours)}
    contours = [contour for contour in contours if cv2.contourArea(contour) >= MIN_AREA_PX]
    if not contours:
        return None, info
    largest = max(contours, key=cv2.contourArea)
    info["largest_area_px"] = float(cv2.contourArea(largest))
    simplified = cv2.approxPolyDP(largest, SIMPLIFY_TOL_PX, True).reshape(-1, 2)
    info["polygon_vertex_count_px"] = int(len(simplified))
    if len(simplified) < 4:
        return None, info
    image_h = mask_700.shape[0]
    coords_m = [
        (float(col) * pixel_to_m, float(image_h - 1 - row) * pixel_to_m)
        for col, row in simplified
    ]
    poly = Polygon(coords_m)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda p: p.area)
    if poly.is_empty or not isinstance(poly, Polygon):
        return None, info
    info["polygon_area_m2"] = float(poly.area)
    return poly, info


def _rasterize_local_polygon(
    poly: Polygon,
    *,
    cell_m: float,
    pad_m: float = 0.5,
) -> tuple[np.ndarray, object]:
    xmin, ymin, xmax, ymax = poly.bounds
    xmin -= pad_m
    ymin -= pad_m
    xmax += pad_m
    ymax += pad_m
    width = max(2, int(math.ceil((xmax - xmin) / cell_m)))
    height = max(2, int(math.ceil((ymax - ymin) / cell_m)))
    image = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(image)
    pix = [
        ((x - xmin) / cell_m, height - 1 - (y - ymin) / cell_m)
        for x, y in poly.exterior.coords
    ]
    drawer.polygon(pix, outline=1, fill=1)
    for inner in poly.interiors:
        ipix = [
            ((x - xmin) / cell_m, height - 1 - (y - ymin) / cell_m)
            for x, y in inner.coords
        ]
        drawer.polygon(ipix, outline=0, fill=0)
    mask = np.flipud(np.array(image, dtype=bool))
    class LocalHeatmap:
        def __init__(self) -> None:
            self.counts = mask.astype(np.int32)
            self.origin_x = xmin
            self.origin_y = ymin
            self.cell_size_m = cell_m

        def cell_to_world(self, row: float, col: float) -> tuple[float, float]:
            return (
                self.origin_x + (float(col) + 0.5) * self.cell_size_m,
                self.origin_y + (float(row) + 0.5) * self.cell_size_m,
            )

    hm = LocalHeatmap()
    return mask, hm


def _extract_sprint60_graph(poly: Polygon) -> tuple[PathGraph, dict[str, object]]:
    mask, hm = _rasterize_local_polygon(poly, cell_m=GRAPH_CELL_M)
    erode_iter = 4
    skel_mask = (
        ndimage.binary_erosion(
            mask,
            structure=np.ones((3, 3), dtype=bool),
            iterations=erode_iter,
        )
        if mask.any()
        else mask
    )
    skel = skeletonize(skel_mask)
    raw_graph, path_meta = extract_path_graph_from_skeleton(skel, hm, polygon=poly)
    pruned = prune_short_leaf_edges(raw_graph, min_length_m=0.6)
    return pruned, {
        "raster_corridor_cells": int(mask.sum()),
        "skeleton_cells": int(skel.sum()),
        "raw_nodes": len(raw_graph.nodes),
        "raw_edges": len(raw_graph.edges),
        "graph_nodes": len(pruned.nodes),
        "graph_edges": len(pruned.edges),
        **graph_metrics(pruned, poly),
        **path_meta,
    }


def _render_sprint60_overlay(
    out_path: Path,
    *,
    input_img_rgb: np.ndarray,
    mask_700: np.ndarray,
    poly: Polygon,
    graph: PathGraph,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=140)
    axes[0].imshow(input_img_rgb)
    axes[0].contour(mask_700.astype(np.uint8), levels=[0.5], colors=["#22d3ee"], linewidths=1.6)
    axes[0].set_title("input + predicted mask")
    axes[0].axis("off")

    xs, ys = poly.exterior.xy
    axes[1].fill(xs, ys, color="#e8e8ec", edgecolor="#3b4a66", linewidth=1.4)
    for inner in poly.interiors:
        ix, iy = inner.xy
        axes[1].fill(ix, iy, color="white", edgecolor="#3b4a66", linewidth=1.0)
    axes[1].set_title(f"polygon · {len(poly.exterior.coords) - 1} verts · {poly.area:.1f} m²")
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.25, linestyle="--")

    axes[2].fill(xs, ys, color="#e8e8ec", edgecolor="#3b4a66", linewidth=1.0)
    for inner in poly.interiors:
        ix, iy = inner.xy
        axes[2].fill(ix, iy, color="white", edgecolor="#3b4a66", linewidth=0.8)
    for edge in graph.edges:
        path = chaikin_smooth_path_inside(edge.path, poly, iterations=2)
        if len(path) < 2:
            continue
        xp, yp = zip(*path, strict=True)
        axes[2].plot(xp, yp, "-", color="#cf2e3a", linewidth=2.0, zorder=3)
    if graph.nodes:
        import networkx as nx

        topology = nx.Graph()
        topology.add_nodes_from(range(len(graph.nodes)))
        for edge in graph.edges:
            topology.add_edge(edge.start, edge.end)
        landmark = [node for node in topology.nodes if topology.degree[node] != 2]
        if landmark:
            lx, ly = zip(*[graph.nodes[idx] for idx in landmark], strict=True)
            axes[2].scatter(
                lx,
                ly,
                c="#cf2e3a",
                s=28,
                zorder=4,
                edgecolors="white",
                linewidths=1.0,
            )
    axes[2].set_title(f"graph · {len(graph.nodes)} nodes / {len(graph.edges)} edges")
    axes[2].set_aspect("equal")
    axes[2].grid(True, alpha=0.2, linestyle="--")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def _extract_display_graph(
    poly: Polygon,
    *,
    cell_size_m: float,
    dominant_angle_deg: float | None,
) -> tuple[dict[str, object], dict[str, object], list[list[tuple[float, float, float]]]]:
    """Build a display/routing graph from the rectified polygon footprint."""
    pad_m = 0.5
    xmin, ymin, xmax, ymax = poly.bounds
    x0 = xmin - pad_m
    y0 = ymin - pad_m
    w = max(2, int(math.ceil((xmax - x0 + pad_m) / cell_size_m)))
    h = max(2, int(math.ceil((ymax - y0 + pad_m) / cell_size_m)))

    mask = np.zeros((h, w), dtype=np.uint8)
    exterior = np.asarray(
        [
            [
                int(round((x - x0) / cell_size_m)),
                int(round((y - y0) / cell_size_m)),
            ]
            for x, y in poly.exterior.coords
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [exterior], 1)
    for interior in poly.interiors:
        hole = np.asarray(
            [
                [
                    int(round((x - x0) / cell_size_m)),
                    int(round((y - y0) / cell_size_m)),
                ]
                for x, y in interior.coords
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [hole], 0)

    origin = GridOrigin(x0=x0, y0=y0, z0=0.0, cell_size=cell_size_m, w=w, h=h)
    grid = WalkableGrid(
        origin=origin,
        mask=mask.astype(bool),
        observation_count=mask.astype(np.uint16),
    )
    skeleton = SkeletonizeStep().run(grid)
    nodes, edges = NodePlacementStep(
        scan_id=uuid4(),
        build_job_id=uuid4(),
        max_edge_length_m=4.0,
        force_rectilinear=True,
        dominant_angle_deg=dominant_angle_deg,
        footprint_polygon=poly,
        snap_to_footprint_threshold_m=0.5,
    ).run(skeleton, origin)

    node_index = {node.node_id: idx for idx, node in enumerate(nodes)}
    features: list[dict[str, object]] = []
    for idx, node in enumerate(nodes):
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "kind": "node",
                    "id": idx,
                    "node_type": getattr(node.node_type, "value", str(node.node_type)),
                },
                "geometry": {"type": "Point", "coordinates": [node.x, node.y]},
            }
        )
    edge_polylines: list[list[tuple[float, float, float]]] = []
    for idx, edge in enumerate(edges):
        edge_polylines.append(edge.polyline)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "kind": "edge",
                    "id": idx,
                    "from": node_index.get(edge.from_node_id),
                    "to": node_index.get(edge.to_node_id),
                    "length_m": edge.length_m,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[x, y] for x, y, _z in edge.polyline],
                },
            }
        )

    lengths = [edge.length_m for edge in edges]
    metrics = {
        "skeleton_pixels": skeleton.skeleton_pixel_count,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "edge_length_total_m": float(sum(lengths)),
        "edge_length_p50_m": float(np.percentile(lengths, 50)) if lengths else 0.0,
        "edge_length_p95_m": float(np.percentile(lengths, 95)) if lengths else 0.0,
        "raster_cells": int(mask.sum()),
        "cell_size_m": cell_size_m,
    }
    return {"type": "FeatureCollection", "features": features}, metrics, edge_polylines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=Path, help="obstacle_heatmap_counts.npz")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resize-to", type=int, default=None,
                        help="If set, override checkpoint's resize")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--simplify-tol-m", type=float, default=0.30)
    parser.add_argument("--filter-min-counts", type=int, default=0,
                        help="Mask out heatmap cells with counts < N (drop interior smear)")
    parser.add_argument(
        "--polygon-mode",
        choices=["image700", "native"],
        default="image700",
        help=(
            "image700 matches Sprint 60 exactly: contour on 700px mask with "
            "0.05m/px local coordinates. native crops back to heatmap cells."
        ),
    )
    parser.add_argument(
        "--image-pixel-to-m",
        type=float,
        default=IMAGE_PIXEL_TO_M,
        help="Meter scale for --polygon-mode image700.",
    )
    parser.add_argument(
        "--strip-purple",
        action="store_true",
        help="Zero out rendered inferno pixels with G<threshold before inference.",
    )
    parser.add_argument(
        "--purple-g-threshold",
        type=int,
        default=60,
        help="G-channel cutoff used by Sprint 60 strip_dark_purple.",
    )
    parser.add_argument(
        "--rectify",
        action="store_true",
        help="Apply Sprint 60 dominant-mean Manhattan rectification.",
    )
    parser.add_argument(
        "--rectify-grid-m",
        type=float,
        default=0.10,
        help="Grid snap interval for Manhattan rectification.",
    )
    parser.add_argument(
        "--rectify-cluster-tol-m",
        type=float,
        default=0.40,
        help="1D wall-coordinate clustering tolerance for rectification.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.npz_path)
    counts = np.asarray(data["counts"], dtype=np.int32)
    origin_x = float(data["origin_x"][0])
    origin_y = float(data["origin_y"][0])
    cell_size_m = float(data["cell_size_m"][0])
    H_real, W_real = counts.shape
    print(f"[heatmap] {W_real}×{H_real} cells @ {cell_size_m}m, origin=({origin_x:.2f}, {origin_y:.2f})")

    bgr = render_heatmap_as_inferno(counts, min_counts=args.filter_min_counts)
    overlay_bgr_native = bgr.copy()
    # generator was trained at 700x700 nominal — pad heatmap to its larger side, then resize.
    target_native = max(H_real, W_real)
    pad_y = target_native - H_real
    pad_x = target_native - W_real
    bgr_padded = np.pad(
        bgr, ((0, pad_y), (0, pad_x), (0, 0)), mode="constant", constant_values=0
    )
    bgr_padded = cv2.resize(bgr_padded, (700, 700), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(args.out_dir / "input_inferno.png"), bgr_padded)
    strip_metrics: dict[str, object] | None = None
    if args.strip_purple:
        before_nonzero = int((bgr_padded.max(axis=2) > 0).sum())
        bgr_padded = strip_dark_purple_bgr(
            bgr_padded,
            g_threshold=args.purple_g_threshold,
        )
        overlay_bgr_native = strip_dark_purple_bgr(
            overlay_bgr_native,
            g_threshold=args.purple_g_threshold,
        )
        after_nonzero = int((bgr_padded.max(axis=2) > 0).sum())
        cv2.imwrite(str(args.out_dir / "input_after_strip.png"), bgr_padded)
        strip_metrics = {
            "enabled": True,
            "green_channel_threshold": args.purple_g_threshold,
            "before_nonzero_pixels": before_nonzero,
            "after_nonzero_pixels": after_nonzero,
            "kept_ratio": after_nonzero / float(max(1, before_nonzero)),
        }
        print(
            f"[strip-purple] kept {after_nonzero}/{before_nonzero} pixels "
            f"({strip_metrics['kept_ratio'] * 100.0:.1f}%)"
        )

    device = select_device(args.device)
    print(f"[device] {device}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder = ckpt.get("encoder", "resnet34")
    resize_to = args.resize_to or ckpt.get("resize_to", 512)
    print(f"[ckpt] encoder={encoder} resize_to={resize_to} val_iou={ckpt.get('val_iou')}")
    model = build_unet(encoder_name=encoder, encoder_weights=None).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    img_rgb_700 = cv2.cvtColor(bgr_padded, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        img_rgb_700,
        (int(resize_to), int(resize_to)),
        interpolation=cv2.INTER_AREA,
    )
    rgb = resized.astype(np.float32) / 255.0
    rgb = np.transpose(rgb, (2, 0, 1))[None, ...]
    tensor = torch.from_numpy(rgb).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)
    probs_np = probs[0, 0].detach().cpu().numpy()
    probs_700 = cv2.resize(probs_np, (700, 700), interpolation=cv2.INTER_LINEAR)

    pred_700 = (probs_700 >= args.threshold).astype(np.uint8) * 255
    cv2.imwrite(str(args.out_dir / "predicted_mask_700.png"), pred_700)

    if args.polygon_mode == "image700":
        mask_700 = pred_700 > 0
        raw_poly_700, mask_info = _mask700_to_polygon(
            mask_700,
            pixel_to_m=args.image_pixel_to_m,
        )
        if raw_poly_700 is None:
            print(f"[done] no image700 polygon; info={mask_info}")
            return 1
        final_poly_700 = raw_poly_700
        final_info = dict(mask_info)
        final_info["rectified"] = False
        if args.rectify:
            rectified, rect_meta = manhattan_rectify(
                raw_poly_700,
                grid_m=args.rectify_grid_m,
                cluster_tol_m=args.rectify_cluster_tol_m,
            )
            final_info["rectification"] = rect_meta
            if rectified is None:
                print(f"[rectify] failed: {rect_meta}")
            else:
                before_area = float(raw_poly_700.area)
                area_change_ratio = abs(rectified.area - before_area) / max(
                    before_area,
                    1e-6,
                )
                final_poly_700 = rectified
                final_info.update(
                    {
                        "rectified": True,
                        "rectify_grid_m": args.rectify_grid_m,
                        "polygon_vertex_count_before_rectify": len(
                            raw_poly_700.exterior.coords
                        )
                        - 1,
                        "polygon_area_m2_before_rectify": before_area,
                        "polygon_vertex_count_after_rectify": len(
                            final_poly_700.exterior.coords
                        )
                        - 1,
                        "polygon_area_m2": float(final_poly_700.area),
                        "area_change_ratio": float(area_change_ratio),
                    }
                )
                print(
                    "[rectify] "
                    f"vertices {final_info['polygon_vertex_count_before_rectify']}→"
                    f"{final_info['polygon_vertex_count_after_rectify']} "
                    f"area {before_area:.1f}→{final_poly_700.area:.1f}m2 "
                    f"delta={area_change_ratio * 100.0:.1f}% "
                    f"angle={rect_meta.get('dominant_angle_deg', 0.0):.2f}°"
                )

        _write_single_polygon_geojson(
            args.out_dir / "polygon.geojson",
            poly=final_poly_700,
            properties={"source": "best-v2.pt", "polygon_mode": "image700", **final_info},
        )
        _write_single_polygon_geojson(
            args.out_dir / "raw_polygon.geojson",
            poly=raw_poly_700,
            properties={
                "source": "best-v2.pt",
                "polygon_mode": "image700",
                **mask_info,
            },
        )
        graph, graph_info = _extract_sprint60_graph(final_poly_700)
        (args.out_dir / "graph.geojson").write_text(
            json.dumps(graph_to_geojson(graph, final_poly_700), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        title = (
            "best-v2.pt → polygon → graph · scan 56A8698C · "
            f"val IoU {float(ckpt.get('val_iou', 0.0)):.3f}"
        )
        _render_sprint60_overlay(
            args.out_dir / "overlay.png",
            input_img_rgb=img_rgb_700,
            mask_700=mask_700,
            poly=final_poly_700,
            graph=graph,
            title=title,
        )
        metrics = {
            "checkpoint": str(args.checkpoint),
            "input": {
                "npz_path": str(args.npz_path),
                "heatmap_shape": [int(H_real), int(W_real)],
                "cell_size_m": cell_size_m,
                "filter_min_counts": args.filter_min_counts,
                "strip_purple": strip_metrics
                or {
                    "enabled": False,
                    "green_channel_threshold": args.purple_g_threshold,
                },
            },
            "checkpoint_meta": {
                "encoder": encoder,
                "resize_to": int(resize_to),
                "val_iou": (
                    float(ckpt["val_iou"])
                    if isinstance(ckpt.get("val_iou"), int | float)
                    else ckpt.get("val_iou")
                ),
            },
            "prediction": {
                "prob_threshold": args.threshold,
                "mask_700_pixels": int(mask_700.sum()),
                "polygon_mode": "image700",
                "image_pixel_to_m": args.image_pixel_to_m,
            },
            "polygon": final_info,
            "graph": graph_info,
        }
        (args.out_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[polygon-mode] image700 scale={args.image_pixel_to_m}m/px")
        print(f"[polygon] {final_info}")
        print(
            f"[done] graph={graph_info['graph_nodes']}n/"
            f"{graph_info['graph_edges']}e"
        )
        return 0

    # Crop back to the real heatmap region.
    pred_native = cv2.resize(pred_700, (target_native, target_native), interpolation=cv2.INTER_NEAREST)
    pred_native = pred_native[:H_real, :W_real]
    cv2.imwrite(str(args.out_dir / "predicted_mask.png"), pred_native)

    # Extract polygon contour.
    contours, _ = cv2.findContours(pred_native, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_polygons: list[Polygon] = []
    for c in contours:
        if cv2.contourArea(c) < 50:
            continue
        epsilon = (args.simplify_tol_m / cell_size_m)
        approx = cv2.approxPolyDP(c, epsilon, closed=True)
        ring_world = []
        for pt in approx[:, 0, :]:
            x_world = origin_x + (pt[0] + 0.5) * cell_size_m
            y_world = origin_y + (pt[1] + 0.5) * cell_size_m
            ring_world.append((x_world, y_world))
        if len(ring_world) >= 3:
            poly = Polygon(ring_world)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda p: p.area)
            if isinstance(poly, Polygon) and not poly.is_empty:
                raw_polygons.append(poly)

    if not raw_polygons:
        print("[done] polygons=0 vertices=0")
        return 1

    raw_poly = max(raw_polygons, key=lambda p: p.area)
    raw_properties: dict[str, object] = {
        "source": "best-v2.pt",
        "vertex_count": len(raw_poly.exterior.coords) - 1,
        "area_m2": float(raw_poly.area),
        "contour_count": len(contours),
        "kept_contour_count": len(raw_polygons),
    }
    _write_single_polygon_geojson(
        args.out_dir / "raw_polygon.geojson",
        poly=raw_poly,
        properties=raw_properties,
    )

    final_poly = raw_poly
    final_properties = dict(raw_properties)
    final_properties["rectified"] = False
    if args.rectify:
        rectified, rect_meta = manhattan_rectify(
            raw_poly,
            grid_m=args.rectify_grid_m,
            cluster_tol_m=args.rectify_cluster_tol_m,
        )
        final_properties["rectification"] = rect_meta
        if rectified is None:
            print(f"[rectify] failed: {rect_meta}")
        else:
            area_change_ratio = abs(rectified.area - raw_poly.area) / max(
                raw_poly.area,
                1e-6,
            )
            final_poly = rectified
            final_properties.update(
                {
                    "rectified": True,
                    "vertex_count_raw": len(raw_poly.exterior.coords) - 1,
                    "vertex_count": len(final_poly.exterior.coords) - 1,
                    "area_m2_raw": float(raw_poly.area),
                    "area_m2": float(final_poly.area),
                    "area_change_ratio": float(area_change_ratio),
                }
            )
            print(
                "[rectify] "
                f"vertices {final_properties['vertex_count_raw']}→"
                f"{final_properties['vertex_count']} "
                f"area {raw_poly.area:.1f}→{final_poly.area:.1f}m2 "
                f"delta={area_change_ratio * 100.0:.1f}% "
                f"angle={rect_meta.get('dominant_angle_deg', 0.0):.2f}°"
            )

    _write_single_polygon_geojson(
        args.out_dir / "polygon.geojson",
        poly=final_poly,
        properties=final_properties,
    )
    graph_geojson, graph_metrics, graph_polylines = _extract_display_graph(
        final_poly,
        cell_size_m=cell_size_m,
        dominant_angle_deg=(
            float(final_properties["rectification"]["dominant_angle_deg"])
            if isinstance(final_properties.get("rectification"), dict)
            and isinstance(
                final_properties["rectification"].get("dominant_angle_deg"),
                int | float,
            )
            else None
        ),
    )
    (args.out_dir / "graph.geojson").write_text(
        json.dumps(graph_geojson, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Overlay
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="black")
    ax.set_facecolor("black")
    if args.strip_purple:
        ax.imshow(
            cv2.cvtColor(overlay_bgr_native, cv2.COLOR_BGR2RGB),
            origin="upper",
            interpolation="nearest",
        )
    else:
        ax.imshow(
            np.log1p(counts).astype(np.float32),
            cmap="inferno",
            origin="upper",
            interpolation="nearest",
        )
    overlay = np.zeros((*pred_native.shape, 4), dtype=np.float32)
    overlay[pred_native > 0] = (0.0, 0.85, 1.0, 0.16)
    ax.imshow(overlay, origin="upper", interpolation="nearest")
    for poly, color, width, label in [
        (raw_poly, "#00e5ff", 1.2, "raw"),
        (final_poly, "#ffe066", 2.0, "final"),
    ]:
        cols = [(x - origin_x) / cell_size_m for x, _ in poly.exterior.coords]
        rows = [(y - origin_y) / cell_size_m for _, y in poly.exterior.coords]
        ax.plot(cols, rows, color=color, linewidth=width, label=label)
    for polyline in graph_polylines:
        cols = [(x - origin_x) / cell_size_m for x, _y, _z in polyline]
        rows = [(y - origin_y) / cell_size_m for _x, y, _z in polyline]
        ax.plot(cols, rows, color="#2f80ed", linewidth=1.6, alpha=0.95)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(
        f"Predicted polygon · raw={len(raw_poly.exterior.coords)-1}v "
        f"final={len(final_poly.exterior.coords)-1}v "
        f"graph={graph_metrics['node_count']}n/{graph_metrics['edge_count']}e "
        f"prob_thr={args.threshold}",
        color="black", fontsize=11, pad=6, backgroundcolor="white",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(args.out_dir / "polygon_overlay.png", dpi=140, bbox_inches="tight",
                pad_inches=0.04)
    plt.close(fig)

    metrics = {
        "checkpoint": str(args.checkpoint),
        "input": {
            "npz_path": str(args.npz_path),
            "heatmap_shape": [int(H_real), int(W_real)],
            "cell_size_m": cell_size_m,
            "filter_min_counts": args.filter_min_counts,
            "strip_purple": strip_metrics
            or {
                "enabled": False,
                "green_channel_threshold": args.purple_g_threshold,
            },
        },
        "checkpoint_meta": {
            "encoder": encoder,
            "resize_to": int(resize_to),
            "val_iou": (
                float(ckpt["val_iou"])
                if isinstance(ckpt.get("val_iou"), int | float)
                else ckpt.get("val_iou")
            ),
        },
        "prediction": {
            "prob_threshold": args.threshold,
            "native_mask_pixels": int((pred_native > 0).sum()),
            "contour_count": len(contours),
            "kept_contour_count": len(raw_polygons),
        },
        "polygon": final_properties,
        "graph": graph_metrics,
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[done] polygons={len(raw_polygons)} "
        f"raw_vertices={len(raw_poly.exterior.coords)-1} "
        f"final_vertices={len(final_poly.exterior.coords)-1} "
        f"graph={graph_metrics['node_count']}n/{graph_metrics['edge_count']}e"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
