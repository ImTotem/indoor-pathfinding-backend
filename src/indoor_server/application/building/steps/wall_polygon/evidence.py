"""Sprint 51 (labels Englishified in Sprint 52) — Wall polygon evidence PNG renderer.

Dumps the 7-PNG evidence pack via matplotlib (Agg backend). Sprint 52
replaced Korean labels with English equivalents to eliminate matplotlib
glyph warnings on minimal CI fonts; Korean is still the preferred
documentation language elsewhere but evaluator-facing PNGs need stable
glyph coverage.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from indoor_server.application.building.steps.wall_polygon.assembly import (
    AssembledPolygon,
)
from indoor_server.application.building.steps.wall_polygon.components import (
    FragmentSet,
)
from indoor_server.application.building.steps.wall_polygon.density import DensityRefined
from indoor_server.application.building.steps.wall_polygon.facade import (
    WallPolygonResult,
)
from indoor_server.application.building.steps.wall_polygon.line_fit import (
    LineSegmentSet,
)
from indoor_server.application.building.steps.wall_polygon.merge import WallLineSet
from indoor_server.application.building.steps.wall_polygon.obstacle_source import (
    ObstacleHeatmap,
)
from indoor_server.application.building.steps.wall_polygon.snap import (
    SnappedSegmentSet,
)
from indoor_server.application.building.steps.wall_polygon.validate import (
    WallPolygonValidation,
)

logger = logging.getLogger(__name__)


def _ensure_mpl() -> Any:
    """import matplotlib with non-interactive backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _draw_placeholder(plt: Any, output_path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(title, fontsize=11)
    ax.text(
        0.5, 0.5, str(message),
        ha="center", va="center", fontsize=10, color="gray",
    )
    ax.set_axis_off()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _render_density(
    plt: Any,
    heatmap: ObstacleHeatmap,
    refined: DensityRefined | None,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    counts = np.asarray(heatmap.counts, dtype=np.float64)
    log_counts = np.log1p(counts)
    ax.imshow(
        log_counts,
        cmap="gray_r",
        origin="lower",
        interpolation="nearest",
    )
    if refined is not None:
        mask = refined.occupancy_mask
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask] = (1.0, 0.2, 0.2, 0.45)
        ax.imshow(overlay, origin="lower", interpolation="nearest")
        params_obj = refined.metadata.get("params", {})
        min_hits = (
            params_obj.get("min_cell_hits", "n/a")
            if isinstance(params_obj, dict)
            else "n/a"
        )
        ax.set_title(
            "01 — Density refinement (gray=count log scale, red=mask after threshold)\n"
            f"cell={heatmap.cell_size_m}m  "
            f"hits>={min_hits}  "
            f"mask cells={refined.metadata.get('cells_after_morph', 0)}",
            fontsize=10,
        )
    else:
        ax.set_title(
            "01 — Raw obstacle heatmap (refinement skipped)", fontsize=10
        )
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _render_components(plt: Any, fragments: FragmentSet, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    label_map = fragments.label_map
    n_comp = int(label_map.max())
    if n_comp == 0:
        ax.text(
            0.5, 0.5, "(no components)", ha="center", va="center", color="gray"
        )
        ax.set_title("02 — Components (empty)", fontsize=10)
    else:
        # color via hashing — distinct colors
        rng = np.random.default_rng(42)
        colors = rng.uniform(0.2, 1.0, size=(n_comp + 1, 3))
        colors[0] = (1.0, 1.0, 1.0)  # background
        rgb = colors[label_map]
        ax.imshow(rgb, origin="lower", interpolation="nearest")
        for frag in fragments.fragments:
            r0, r1 = frag.bbox_rows
            c0, c1 = frag.bbox_cols
            cy = (r0 + r1) / 2.0
            cx = (c0 + c1) / 2.0
            ax.text(
                cx, cy, str(frag.component_id),
                color="black", fontsize=8, ha="center", va="center",
                bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 1},
            )
        ax.set_title(
            f"02 — Components (kept={len(fragments.fragments)} / "
            f"total={fragments.metadata.get('total_components_before_filter', 0)})",
            fontsize=10,
        )
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _render_components_from_mask(
    plt: Any,
    refined: DensityRefined,
    output_path: Path,
) -> None:
    """Sprint 52 substitute when facade does not produce FragmentSet.

    Shows the refined density mask. The Sprint 51 connected-component label
    map is no longer in the facade pipeline, but evaluators still expect a
    "stage 02" panel.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    mask = refined.occupancy_mask
    if not mask.any():
        ax.text(
            0.5, 0.5, "(empty mask)", ha="center", va="center", color="gray"
        )
    else:
        ax.imshow(mask.astype(np.float32), cmap="Greys", origin="lower")
    cells_raw = refined.metadata.get("cells_after_morph", int(mask.sum()))
    if isinstance(cells_raw, int | float):
        cells = int(cells_raw)
    else:
        cells = int(mask.sum())
    ax.set_title(
        f"02 — Refined mask preview (cells={cells})",
        fontsize=10,
    )
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _render_line_fits(
    plt: Any,
    refined_mask: np.ndarray | None,
    lines: LineSegmentSet,
    heatmap: ObstacleHeatmap,
    output_path: Path,
    *,
    source: str = "skeleton_only",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    cell = heatmap.cell_size_m
    ox = heatmap.origin_x
    oy = heatmap.origin_y
    if refined_mask is not None:
        bg = refined_mask.astype(np.float32) * 0.5
        h_cells, w_cells = refined_mask.shape
        extent = (ox, ox + w_cells * cell, oy, oy + h_cells * cell)
        ax.imshow(bg, cmap="Greys", origin="lower", extent=extent, alpha=0.6)
    primary_raw = lines.metadata.get("primary_count", 0)
    fallback_raw = lines.metadata.get("fallback_count", 0)
    primary_count = (
        int(primary_raw) if isinstance(primary_raw, int | float) else 0
    )
    fallback_count = (
        int(fallback_raw) if isinstance(fallback_raw, int | float) else 0
    )
    for idx, seg in enumerate(lines.segments):
        if source == "skeleton+hough":
            color = "green" if idx < primary_count else "blue"
        elif source == "hough_only":
            color = "blue"
        else:
            color = "green"
        ax.plot(
            [seg.x1, seg.x2],
            [seg.y1, seg.y2],
            color=color,
            linewidth=2.0,
            alpha=0.9,
        )
    if source == "hough_only":
        ax.text(
            0.02,
            0.98,
            "fallback (Hough)",
            transform=ax.transAxes,
            fontsize=9,
            color="blue",
            verticalalignment="top",
        )
    ax.set_title(
        f"03 — Line fits (accept={len(lines.segments)} "
        f"primary={primary_count} fallback={fallback_count} source={source})",
        fontsize=10,
    )
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _render_after_snap(
    plt: Any,
    snapped: SnappedSegmentSet,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for s in snapped.segments:
        ax.plot(
            [s.base.x1, s.base.x2],
            [s.base.y1, s.base.y2],
            color="blue", linewidth=2.0, alpha=0.9,
        )
    for r in snapped.rejected:
        ax.plot(
            [r.x1, r.x2],
            [r.y1, r.y2],
            color="gray", linewidth=1.0, alpha=0.5, linestyle=":",
        )
    ax.set_title(
        f"04 — After angle snap (pass={len(snapped.segments)} "
        f"reject={len(snapped.rejected)} ratio={snapped.snap_pass_ratio:.2f}, "
        f"dominant={snapped.dominant_angle_deg:.1f} deg)",
        fontsize=10,
    )
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _render_after_merge(
    plt: Any,
    merged: WallLineSet,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, line in enumerate(merged.lines):
        ax.plot(
            [line.world_x1, line.world_x2],
            [line.world_y1, line.world_y2],
            color="black", linewidth=3.0, alpha=0.9,
        )
        mx = (line.world_x1 + line.world_x2) / 2.0
        my = (line.world_y1 + line.world_y2) / 2.0
        ax.text(
            mx, my, f"L{i}",
            color="darkred", fontsize=9, ha="center", va="bottom",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1},
        )
    ax.set_title(
        f"05 — After collinear merge (lines={len(merged.lines)}, "
        f"gap_filled={merged.metadata.get('gap_filled_count', 0)})",
        fontsize=10,
    )
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _render_polygon(
    plt: Any,
    assembled: AssembledPolygon,
    merged: WallLineSet | None,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    if assembled.polygon_geojson is None:
        ax.text(
            0.5, 0.5, "(polygon not generated)",
            ha="center", va="center", color="gray",
        )
    else:
        from shapely.geometry import shape

        geom = shape(assembled.polygon_geojson)
        polys = []
        if geom.geom_type == "Polygon":
            polys.append(geom)
        elif geom.geom_type == "MultiPolygon":
            polys.extend(list(geom.geoms))
        for poly in polys:
            ext = list(poly.exterior.coords)
            xs = [p[0] for p in ext]
            ys = [p[1] for p in ext]
            ax.fill(xs, ys, color="lightblue", alpha=0.5, edgecolor="navy", linewidth=2)
            # corner dots
            params_obj = assembled.metadata.get("params", {})
            tol_raw: object = (
                params_obj.get("orthogonality_angle_tol_deg", 5.0)
                if isinstance(params_obj, dict)
                else 5.0
            )
            try:
                tol = float(tol_raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                tol = 5.0
            # mark per-vertex orthogonality
            import math as _math

            n = len(ext) - 1  # closing dup
            for i in range(n):
                a = ext[(i - 1) % n]
                b = ext[i]
                c = ext[(i + 1) % n]
                v1 = (a[0] - b[0], a[1] - b[1])
                v2 = (c[0] - b[0], c[1] - b[1])
                n1 = _math.hypot(*v1)
                n2 = _math.hypot(*v2)
                if n1 <= 1e-9 or n2 <= 1e-9:
                    continue
                cos_a = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
                ang = _math.degrees(_math.acos(cos_a))
                col = "green" if abs(ang - 90.0) <= tol else "red"
                ax.plot(b[0], b[1], "o", color=col, markersize=6)
        # avoid lint: _corner_orthogonality_ratio import is intentional for vertex coloring
    if merged is not None:
        for line in merged.lines:
            ax.plot(
                [line.world_x1, line.world_x2],
                [line.world_y1, line.world_y2],
                color="lightgray", linewidth=1.0, alpha=0.5,
            )
    title = (
        f"06 — Assembled polygon (vertex={assembled.vertex_count} "
        f"orthogonality={assembled.corner_orthogonality_ratio:.2f} "
        f"fallback={assembled.fallback_used or 'none'})"
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _render_iou_with_floor(
    plt: Any,
    assembled: AssembledPolygon,
    floor_polygon_geojson: dict[str, object] | None,
    validation: WallPolygonValidation,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    from shapely.geometry import shape

    if floor_polygon_geojson is not None:
        try:
            floor = shape(floor_polygon_geojson)
            polys = (
                [floor] if floor.geom_type == "Polygon"
                else (list(floor.geoms) if floor.geom_type == "MultiPolygon" else [])
            )
            for poly in polys:
                ext = list(poly.exterior.coords)
                xs = [p[0] for p in ext]
                ys = [p[1] for p in ext]
                ax.plot(xs, ys, color="navy", linewidth=2.0, label="floor (point cloud)")
        except Exception:
            pass
    if assembled.polygon_geojson is not None:
        try:
            wall = shape(assembled.polygon_geojson)
            polys = (
                [wall] if wall.geom_type == "Polygon"
                else (list(wall.geoms) if wall.geom_type == "MultiPolygon" else [])
            )
            for poly in polys:
                ext = list(poly.exterior.coords)
                xs = [p[0] for p in ext]
                ys = [p[1] for p in ext]
                ax.fill(xs, ys, color="green", alpha=0.4, label="wall polygon")
        except Exception:
            pass
    title = (
        f"07 — IoU vs floor polygon (IoU={validation.iou_with_floor:.3f} "
        f"area_change={validation.area_change_ratio:.3f} "
        f"accepted={validation.accepted})"
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_wall_polygon_evidence(
    result: WallPolygonResult,
    output_dir: Path,
    *,
    scan_id: str,
    floor_polygon_geojson: dict[str, object] | None = None,
) -> dict[str, Path]:
    """Render the 7-PNG evidence pack for a single wall polygon run.

    `output_dir` is created if missing. Returns mapping of filename → Path.
    """
    plt = _ensure_mpl()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    stages = result.stage_outputs

    # 01 density
    p1 = output_dir / "01_density.png"
    paths["01_density"] = p1
    heatmap = stages.get("obstacle_source")
    refined = stages.get("density")
    if isinstance(heatmap, ObstacleHeatmap):
        _render_density(plt, heatmap, refined if isinstance(refined, DensityRefined) else None, p1)
    else:
        _draw_placeholder(plt, p1, "01 — density", "obstacle source 단계 데이터 없음")

    # 02 components — Sprint 52: facade no longer produces FragmentSet, but
    # we keep the slot for backward compat with sprint 51 dumps. If absent,
    # fall back to the refined density mask as a "component preview".
    p2 = output_dir / "02_components.png"
    paths["02_components"] = p2
    fragments = stages.get("components")
    if isinstance(fragments, FragmentSet):
        _render_components(plt, fragments, p2)
    elif isinstance(refined, DensityRefined):
        _render_components_from_mask(plt, refined, p2)
    else:
        _draw_placeholder(
            plt, p2, "02 — components", "component split stage skipped"
        )

    # 03 line_fits
    p3 = output_dir / "03_line_fits.png"
    paths["03_line_fits"] = p3
    lines = stages.get("line_fit")
    if (
        isinstance(lines, LineSegmentSet)
        and isinstance(heatmap, ObstacleHeatmap)
    ):
        source_obj = lines.metadata.get("source", "skeleton_only")
        source = str(source_obj) if isinstance(source_obj, str) else "skeleton_only"
        refined_mask = (
            refined.occupancy_mask if isinstance(refined, DensityRefined) else None
        )
        _render_line_fits(plt, refined_mask, lines, heatmap, p3, source=source)
    else:
        _draw_placeholder(
            plt, p3, "03 — line_fits", "line fit stage skipped"
        )

    # 04 after_snap
    p4 = output_dir / "04_after_snap.png"
    paths["04_after_snap"] = p4
    snapped = stages.get("snap")
    if isinstance(snapped, SnappedSegmentSet):
        _render_after_snap(plt, snapped, p4)
    else:
        _draw_placeholder(
            plt, p4, "04 — after_snap", "snap stage skipped"
        )

    # 05 after_merge
    p5 = output_dir / "05_after_merge.png"
    paths["05_after_merge"] = p5
    merged = stages.get("merge")
    if isinstance(merged, WallLineSet):
        _render_after_merge(plt, merged, p5)
    else:
        _draw_placeholder(
            plt, p5, "05 — after_merge", "merge stage skipped"
        )

    # 06 polygon
    p6 = output_dir / "06_polygon.png"
    paths["06_polygon"] = p6
    assembled = stages.get("assembly")
    if isinstance(assembled, AssembledPolygon):
        _render_polygon(
            plt,
            assembled,
            merged if isinstance(merged, WallLineSet) else None,
            p6,
        )
    else:
        _draw_placeholder(
            plt, p6, "06 — polygon", "polygon assembly stage skipped"
        )

    # 07 iou_with_floor
    p7 = output_dir / "07_iou_with_floor.png"
    paths["07_iou_with_floor"] = p7
    validation = stages.get("validate")
    if (
        isinstance(validation, WallPolygonValidation)
        and isinstance(assembled, AssembledPolygon)
    ):
        _render_iou_with_floor(plt, assembled, floor_polygon_geojson, validation, p7)
    else:
        _draw_placeholder(
            plt, p7, "07 — iou_with_floor", "validate stage skipped"
        )

    # also write json report
    report_path = output_dir / "wall_polygon_report.json"
    paths["report_json"] = report_path
    report_path.write_text(
        json.dumps(_build_json_report(result, scan_id=scan_id), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return paths


def _build_json_report(result: WallPolygonResult, *, scan_id: str) -> dict[str, object]:
    """Human-readable + machine-readable mirror of result.metadata."""
    meta = dict(result.metadata)
    return {
        "scan_id": scan_id,
        "accepted": bool(result.accepted),
        "fail_reason": result.fail_reason,
        "params": meta.get("params"),
        "stages": meta.get("stages", {}),
        "line_count": meta.get("line_count"),
        "vertex_count": meta.get("vertex_count"),
        "corner_orthogonality_ratio": meta.get("corner_orthogonality_ratio"),
        "iou_with_floor": meta.get("iou_with_floor"),
        "area_change_ratio": meta.get("area_change_ratio"),
        "self_intersection_count": meta.get("self_intersection_count"),
        "fallback_used": meta.get("fallback_used"),
        "final_polygon_geojson": result.final_polygon_geojson,
    }


def _maybe_dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return obj
