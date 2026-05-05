"""Sprint 52 — Step 2~3 primary: skeletonize + endpoint/junction split.

Replaces Sprint 51 PCA-per-component which structurally rejects multi-axis
blobs (ㄱ/T/+ corridor). The skeleton split does:

  1. binary mask → `skimage.morphology.skeletonize(method=...)` → 1-pixel skeleton
  2. skeleton pixels → networkx 8-connectivity Graph
  3. classify by degree: endpoint=1, junction>=3, straight=2
  4. cut at junctions → polylines from endpoint→endpoint or endpoint→junction
  5. spur prune: leaf branches whose length < ratio × main edge are dropped
  6. RDP simplify per polyline (epsilon = `rdp_epsilon_cells`)
  7. polyline-to-segment: multi-vertex polyline expanded into 2-vertex segments
  8. straightness/length filter, cell→world conversion
  9. emit `LineSegment` list (compatible with line_fit dataclass for downstream)

Output schema mirrors `line_fit.LineSegmentSet` so the facade can swap the
Step 2~3 dispatch with no Step 4~7 changes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from skimage.morphology import skeletonize

from indoor_server.application.building.steps.wall_polygon.density import (
    DensityRefined,
)
from indoor_server.application.building.steps.wall_polygon.line_fit import (
    LineSegment,
    _fold_angle_deg,
)
from indoor_server.application.building.steps.wall_polygon.obstacle_source import (
    ObstacleHeatmap,
)


@dataclass(frozen=True)
class SkeletonSplitParams:
    """primary algorithm parameters."""

    skeletonize_method: str = "lee"
    rdp_epsilon_cells: float = 1.5
    min_pixels_per_segment: int = 6
    min_length_m: float = 0.5
    min_straightness: float = 0.85
    spur_prune_length_ratio: float = 0.20
    spur_prune_max_iter: int = 3

    def to_metadata(self) -> dict[str, object]:
        return {
            "skeletonize_method": self.skeletonize_method,
            "rdp_epsilon_cells": self.rdp_epsilon_cells,
            "min_pixels_per_segment": self.min_pixels_per_segment,
            "min_length_m": self.min_length_m,
            "min_straightness": self.min_straightness,
            "spur_prune_length_ratio": self.spur_prune_length_ratio,
            "spur_prune_max_iter": self.spur_prune_max_iter,
        }


@dataclass(frozen=True)
class SkeletonSplitResult:
    """Step 2~3 primary output — schema compatible with `LineSegmentSet`."""

    segments: list[LineSegment]
    skeleton_pixel_count: int
    endpoint_count: int
    junction_count: int
    spurs_pruned_count: int
    rejected_short_count: int
    rejected_curved_count: int
    metadata: dict[str, object] = field(default_factory=dict)


def _rdp_simplify(
    polyline: list[tuple[int, int]], epsilon: float
) -> list[tuple[int, int]]:
    """Iterative Ramer-Douglas-Peucker simplification on pixel polyline."""
    if len(polyline) <= 2 or epsilon <= 0:
        return list(polyline)

    keep = [False] * len(polyline)
    keep[0] = True
    keep[-1] = True
    stack: list[tuple[int, int]] = [(0, len(polyline) - 1)]
    while stack:
        i_start, i_end = stack.pop()
        if i_end - i_start < 2:
            continue
        ax, ay = polyline[i_start]
        bx, by = polyline[i_end]
        dx = bx - ax
        dy = by - ay
        seg_len = math.hypot(dx, dy)
        max_dist = -1.0
        max_idx = -1
        if seg_len <= 1e-9:
            for k in range(i_start + 1, i_end):
                px, py = polyline[k]
                d = math.hypot(px - ax, py - ay)
                if d > max_dist:
                    max_dist = d
                    max_idx = k
        else:
            inv = 1.0 / seg_len
            nx_ = -dy * inv
            ny_ = dx * inv
            for k in range(i_start + 1, i_end):
                px, py = polyline[k]
                d = abs((px - ax) * nx_ + (py - ay) * ny_)
                if d > max_dist:
                    max_dist = d
                    max_idx = k
        if max_idx >= 0 and max_dist > epsilon:
            keep[max_idx] = True
            stack.append((i_start, max_idx))
            stack.append((max_idx, i_end))
    return [polyline[i] for i in range(len(polyline)) if keep[i]]


class SkeletonSplitStep:
    """Step 2~3 primary — skeletonize + endpoint/junction split."""

    def __init__(self, params: SkeletonSplitParams | None = None) -> None:
        self._params = params if params is not None else SkeletonSplitParams()
        self._validate()

    def _validate(self) -> None:
        p = self._params
        if p.skeletonize_method not in ("lee", "zhang"):
            raise ValueError(
                f"skeletonize_method must be 'lee' or 'zhang', got {p.skeletonize_method}"
            )
        if p.rdp_epsilon_cells < 0:
            raise ValueError("rdp_epsilon_cells must be >= 0")
        if p.min_pixels_per_segment < 2:
            raise ValueError("min_pixels_per_segment must be >= 2")
        if p.min_length_m < 0:
            raise ValueError("min_length_m must be >= 0")
        if not (0.0 < p.min_straightness <= 1.0):
            raise ValueError("min_straightness must be in (0, 1]")
        if not (0.0 <= p.spur_prune_length_ratio < 1.0):
            raise ValueError("spur_prune_length_ratio must be in [0, 1)")
        if p.spur_prune_max_iter < 0:
            raise ValueError("spur_prune_max_iter must be >= 0")

    @property
    def params(self) -> SkeletonSplitParams:
        return self._params

    def run(
        self,
        refined: DensityRefined,
        *,
        heatmap: ObstacleHeatmap,
    ) -> SkeletonSplitResult:
        params = self._params
        mask = np.asarray(refined.occupancy_mask, dtype=bool)
        if not mask.any():
            return SkeletonSplitResult(
                segments=[],
                skeleton_pixel_count=0,
                endpoint_count=0,
                junction_count=0,
                spurs_pruned_count=0,
                rejected_short_count=0,
                rejected_curved_count=0,
                metadata={
                    "skeleton_pixel_count": 0,
                    "endpoint_count": 0,
                    "junction_count": 0,
                    "polyline_count": 0,
                    "spurs_pruned_count": 0,
                    "rejected_short_count": 0,
                    "rejected_curved_count": 0,
                    "params": params.to_metadata(),
                },
            )

        skel = skeletonize(  # type: ignore[no-untyped-call]
            mask, method=params.skeletonize_method
        )
        skel = np.asarray(skel, dtype=bool)
        skel_count = int(skel.sum())
        if skel_count == 0:
            return SkeletonSplitResult(
                segments=[],
                skeleton_pixel_count=0,
                endpoint_count=0,
                junction_count=0,
                spurs_pruned_count=0,
                rejected_short_count=0,
                rejected_curved_count=0,
                metadata={
                    "skeleton_pixel_count": 0,
                    "endpoint_count": 0,
                    "junction_count": 0,
                    "polyline_count": 0,
                    "spurs_pruned_count": 0,
                    "rejected_short_count": 0,
                    "rejected_curved_count": 0,
                    "params": params.to_metadata(),
                },
            )

        graph = self._build_graph(skel)
        spurs_pruned = self._prune_spurs(graph, params)
        polylines = self._extract_polylines(graph)
        endpoint_count = sum(1 for n in graph.nodes() if graph.degree(n) == 1)
        junction_count = sum(1 for n in graph.nodes() if graph.degree(n) >= 3)

        segments: list[LineSegment] = []
        rejected_short = 0
        rejected_curved = 0
        seg_id = 0
        cell = float(heatmap.cell_size_m)
        ox = float(heatmap.origin_x)
        oy = float(heatmap.origin_y)

        for polyline in polylines:
            if len(polyline) < 2:
                continue
            if len(polyline) < params.min_pixels_per_segment:
                rejected_short += 1
                continue
            simplified = _rdp_simplify(polyline, params.rdp_epsilon_cells)
            # decompose polyline into 2-vertex sub-segments
            for i in range(len(simplified) - 1):
                p0 = simplified[i]
                p1 = simplified[i + 1]
                # cell (col, row) → world
                x1 = ox + (float(p0[0]) + 0.5) * cell
                y1 = oy + (float(p0[1]) + 0.5) * cell
                x2 = ox + (float(p1[0]) + 0.5) * cell
                y2 = oy + (float(p1[1]) + 0.5) * cell
                length = float(math.hypot(x2 - x1, y2 - y1))
                if length < params.min_length_m:
                    rejected_short += 1
                    continue
                # straightness on this sub-segment relative to the original
                # pixel chain that lives between simplified[i] and simplified[i+1].
                straightness = self._straightness(
                    polyline,
                    simplified,
                    i,
                )
                if straightness < params.min_straightness:
                    rejected_curved += 1
                    continue
                angle_deg = _fold_angle_deg(
                    math.degrees(math.atan2(y2 - y1, x2 - x1))
                )
                pixel_count = self._chain_pixel_count(polyline, simplified, i)
                segments.append(
                    LineSegment(
                        component_id=seg_id,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        length_m=length,
                        angle_deg=angle_deg,
                        linearity=float(straightness),
                        pixel_count=int(pixel_count),
                    )
                )
                seg_id += 1

        metadata: dict[str, object] = {
            "skeleton_pixel_count": skel_count,
            "endpoint_count": int(endpoint_count),
            "junction_count": int(junction_count),
            "polyline_count": len(polylines),
            "segment_count": len(segments),
            "spurs_pruned_count": int(spurs_pruned),
            "rejected_short_count": int(rejected_short),
            "rejected_curved_count": int(rejected_curved),
            "params": params.to_metadata(),
        }
        return SkeletonSplitResult(
            segments=segments,
            skeleton_pixel_count=skel_count,
            endpoint_count=int(endpoint_count),
            junction_count=int(junction_count),
            spurs_pruned_count=int(spurs_pruned),
            rejected_short_count=int(rejected_short),
            rejected_curved_count=int(rejected_curved),
            metadata=metadata,
        )

    def _build_graph(self, skel: np.ndarray) -> nx.Graph:
        graph: nx.Graph = nx.Graph()
        ys, xs = np.where(skel)
        for y, x in zip(ys, xs, strict=True):
            graph.add_node((int(x), int(y)))
        for y, x in zip(ys, xs, strict=True):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx_ = int(y) + dy, int(x) + dx
                    if 0 <= ny < skel.shape[0] and 0 <= nx_ < skel.shape[1]:
                        if skel[ny, nx_]:
                            graph.add_edge((int(x), int(y)), (nx_, ny))
        return graph

    def _prune_spurs(self, graph: nx.Graph, params: SkeletonSplitParams) -> int:
        """Iteratively trim short leaf branches.

        For each iteration, find each leaf path (endpoint → first junction).
        Compute global longest leaf length. Drop leaf paths whose length is
        below `spur_prune_length_ratio × longest_leaf`.
        """
        pruned = 0
        for _ in range(params.spur_prune_max_iter):
            endpoints = [n for n in graph.nodes() if graph.degree(n) == 1]
            if not endpoints:
                break
            leaf_paths: list[list[tuple[int, int]]] = []
            for endpoint in endpoints:
                path = self._walk_to_first_junction(graph, endpoint)
                if path:
                    leaf_paths.append(path)
            if not leaf_paths:
                break
            longest = max(len(p) for p in leaf_paths)
            cutoff = max(2, int(longest * params.spur_prune_length_ratio))
            removed_any = False
            for path in leaf_paths:
                if len(path) < cutoff:
                    # Drop all interior pixels except the last (junction). If
                    # the junction itself becomes degree 1 after removal, the
                    # next iteration handles it.
                    for px in path[:-1]:
                        if graph.has_node(px):
                            graph.remove_node(px)
                            pruned += 1
                            removed_any = True
            if not removed_any:
                break
        return pruned

    def _walk_to_first_junction(
        self, graph: nx.Graph, start: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """Walk from a degree-1 endpoint along the chain until degree != 2."""
        if graph.degree(start) != 1:
            return []
        path = [start]
        prev: tuple[int, int] | None = None
        cur = start
        # bound by graph size to avoid pathological loops in degenerate input.
        bound = graph.number_of_nodes() + 1
        while bound > 0:
            bound -= 1
            neighbors = [n for n in graph.neighbors(cur) if n != prev]
            if len(neighbors) != 1 and cur != start:
                break
            if not neighbors:
                break
            nxt = neighbors[0]
            path.append(nxt)
            if graph.degree(nxt) != 2:
                break
            prev, cur = cur, nxt
        return path

    def _extract_polylines(
        self, graph: nx.Graph
    ) -> list[list[tuple[int, int]]]:
        """Cut at junctions / endpoints → list of pixel polylines."""
        polylines: list[list[tuple[int, int]]] = []
        if graph.number_of_edges() == 0:
            return polylines
        key_nodes = {n for n in graph.nodes() if graph.degree(n) != 2}
        if not key_nodes:
            # pure cycle → break at arbitrary node
            anchor = next(iter(graph.nodes()))
            key_nodes = {anchor}
        visited: set[frozenset[tuple[tuple[int, int], tuple[int, int], int]]] = set()
        for anchor in key_nodes:
            for neighbor in graph.neighbors(anchor):
                path = [anchor, neighbor]
                prev = anchor
                cur = neighbor
                while cur not in key_nodes:
                    nexts = [n for n in graph.neighbors(cur) if n != prev]
                    if not nexts:
                        break
                    prev, cur = cur, nexts[0]
                    path.append(cur)
                if len(path) < 2:
                    continue
                # multi-edge tolerant key: pair of endpoints + path length.
                edge_key: frozenset[
                    tuple[tuple[int, int], tuple[int, int], int]
                ] = frozenset(
                    {
                        (path[0], path[-1], len(path)),
                        (path[-1], path[0], len(path)),
                    }
                )
                if edge_key in visited:
                    continue
                visited.add(edge_key)
                polylines.append(path)
        return polylines

    def _straightness(
        self,
        full_polyline: list[tuple[int, int]],
        simplified: list[tuple[int, int]],
        sub_idx: int,
    ) -> float:
        """chord length / polyline length within sub-segment."""
        p0 = simplified[sub_idx]
        p1 = simplified[sub_idx + 1]
        try:
            i0 = full_polyline.index(p0)
            i1 = full_polyline.index(p1, i0 + 1) if p1 != p0 else i0
        except ValueError:
            return 1.0
        if i1 < i0:
            i0, i1 = i1, i0
        chord = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        path_len = 0.0
        for k in range(i0, i1):
            a = full_polyline[k]
            b = full_polyline[k + 1]
            path_len += math.hypot(b[0] - a[0], b[1] - a[1])
        if path_len <= 1e-9:
            return 1.0
        return float(min(1.0, chord / path_len))

    def _chain_pixel_count(
        self,
        full_polyline: list[tuple[int, int]],
        simplified: list[tuple[int, int]],
        sub_idx: int,
    ) -> int:
        p0 = simplified[sub_idx]
        p1 = simplified[sub_idx + 1]
        try:
            i0 = full_polyline.index(p0)
            i1 = full_polyline.index(p1, i0 + 1) if p1 != p0 else i0
        except ValueError:
            return max(2, abs(sub_idx) + 2)
        if i1 < i0:
            i0, i1 = i1, i0
        return max(2, i1 - i0 + 1)
