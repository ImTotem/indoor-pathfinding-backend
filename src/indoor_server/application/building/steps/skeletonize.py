"""Skeletonize step — medial axis → networkx skeleton graph."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from indoor_server.domain.building.models import WalkableGrid

logger = logging.getLogger(__name__)


@dataclass
class SkeletonNode:
    pixel_x: int
    pixel_y: int
    degree: int


@dataclass
class SkeletonEdge:
    u: tuple[int, int]   # (px, py) pixel
    v: tuple[int, int]
    polyline_px: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class SkeletonGraph:
    nodes: list[SkeletonNode]
    edges: list[SkeletonEdge]
    skeleton_pixel_count: int


class SkeletonizeStep:
    """medial_axis → networkx → SkeletonGraph."""

    def run(self, grid: WalkableGrid) -> SkeletonGraph:
        """
        1. skimage.morphology.medial_axis
        2. skeleton bool → networkx Graph (8-connectivity)
        3. degree 분류 + 2-degree 흡수
        """
        from skimage.morphology import medial_axis

        result: tuple[np.ndarray, np.ndarray] = medial_axis(  # type: ignore[no-untyped-call]
            grid.mask, return_distance=True
        )
        skeleton = result[0]
        skel_count = int(skeleton.sum())
        logger.info("skeleton pixels=%d", skel_count)

        if skel_count == 0:
            return SkeletonGraph(nodes=[], edges=[], skeleton_pixel_count=0)

        graph = self._build_graph(skeleton)
        nodes, edges = self._extract_topology(graph)
        return SkeletonGraph(nodes=nodes, edges=edges, skeleton_pixel_count=skel_count)

    def _build_graph(self, skeleton: np.ndarray) -> object:
        import networkx as nx

        graph = nx.Graph()
        ys, xs = np.where(skeleton)
        for y, x in zip(ys, xs, strict=True):
            graph.add_node((x, y))

        # 8-연결
        for y, x in zip(ys, xs, strict=True):
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx_ = y + dy, x + dx
                    if 0 <= ny < skeleton.shape[0] and 0 <= nx_ < skeleton.shape[1]:
                        if skeleton[ny, nx_]:
                            graph.add_edge((x, y), (nx_, ny))
        return graph

    def _extract_topology(
        self, graph: object
    ) -> tuple[list[SkeletonNode], list[SkeletonEdge]]:
        """degree != 2 인 픽셀이 node. degree=2 픽셀은 edge polyline으로 흡수."""
        import networkx as nx

        assert isinstance(graph, nx.Graph)

        key_nodes = {n for n in graph.nodes() if graph.degree(n) != 2}
        if not key_nodes:
            # 순환 skeleton — degree=2만 있음 → 임의 한 점을 기준 노드로
            key_nodes = {next(iter(graph.nodes()))}

        skel_nodes = [
            SkeletonNode(pixel_x=px, pixel_y=py, degree=graph.degree((px, py)))
            for px, py in key_nodes
        ]

        edges: list[SkeletonEdge] = []
        visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

        for start in key_nodes:
            for neighbor in graph.neighbors(start):
                # 2-degree 체인 따라가기
                path = [start, neighbor]
                prev = start
                cur = neighbor
                while cur not in key_nodes:
                    nexts = [n for n in graph.neighbors(cur) if n != prev]
                    if not nexts:
                        break
                    prev, cur = cur, nexts[0]
                    path.append(cur)

                end = path[-1]
                edge_key = (min(start, end), max(start, end))
                if edge_key in visited_edges:
                    continue
                visited_edges.add(edge_key)
                edges.append(
                    SkeletonEdge(
                        u=start,
                        v=end,
                        polyline_px=path,
                    )
                )

        return skel_nodes, edges
