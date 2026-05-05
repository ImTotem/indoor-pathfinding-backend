"""Sprint 82 — Step 2: Navigation graph extraction.

Voxel-grid node downsampling from pose XY + polygon containment filter +
KDTree edge building + RTABMap link boost edges + connectivity repair.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import struct
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from shapely.geometry import LineString, Point, mapping
from shapely.prepared import prep

logger = logging.getLogger(__name__)

ShapelyPolygon = Union["Polygon", "MultiPolygon"]

try:
    from shapely.geometry import Polygon, MultiPolygon  # noqa: F401
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NavNode:
    node_id: str
    x: float
    y: float
    map_id_dominant: int
    source_pose_ids: list[int]
    component_id: int = -1

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class NavEdge:
    edge_id: str
    from_id: str
    to_id: str
    weight_m: float
    kind: str  # "grid" | "loop_closure" | "neighbor"
    source_link_count: int = 0


# ---------------------------------------------------------------------------
# RTABMap link loader
# ---------------------------------------------------------------------------

def load_rtabmap_links(merged_db: Path) -> list[tuple[int, int, int]]:
    """Load (from_id, to_id, type) from merged_v2.db Link table."""
    conn = sqlite3.connect(str(merged_db))
    rows = conn.execute("SELECT from_id, to_id, type FROM Link").fetchall()
    conn.close()
    return [(r[0], r[1], r[2]) for r in rows]


# ---------------------------------------------------------------------------
# Node downsampling
# ---------------------------------------------------------------------------

def downsample_nodes(
    xy: np.ndarray,
    map_id: np.ndarray,
    pose_ids: np.ndarray,
    polygon: "ShapelyPolygon",
    *,
    cell_m: float = 2.0,
) -> list[NavNode]:
    """Voxel-grid downsample + polygon containment filter.

    Each (x//cell_m, y//cell_m) grid cell keeps the first pose that falls
    inside the refined polygon.
    """
    prepped = prep(polygon)
    cell_to_first: dict[tuple[int, int], int] = {}  # cell → index in xy
    cell_to_poses: dict[tuple[int, int], list[int]] = {}
    cell_to_map: dict[tuple[int, int], list[int]] = {}

    for i, (pt, mid, pid) in enumerate(zip(xy, map_id, pose_ids)):
        if not prepped.contains(Point(pt)):
            continue
        cx = int(pt[0] // cell_m)
        cy = int(pt[1] // cell_m)
        key = (cx, cy)
        if key not in cell_to_first:
            cell_to_first[key] = i
            cell_to_poses[key] = []
            cell_to_map[key] = []
        cell_to_poses[key].append(int(pid))
        cell_to_map[key].append(int(mid))

    nodes: list[NavNode] = []
    for key, idx in cell_to_first.items():
        mids = cell_to_map[key]
        dom_map = int(np.bincount(mids).argmax())
        nodes.append(
            NavNode(
                node_id=str(uuid.uuid4()),
                x=float(xy[idx, 0]),
                y=float(xy[idx, 1]),
                map_id_dominant=dom_map,
                source_pose_ids=cell_to_poses[key],
            )
        )

    logger.info(
        "downsample_nodes: cell_m=%.1f → %d nav nodes",
        cell_m, len(nodes),
    )
    logger.info("Nav nodes: %d", len(nodes))
    return nodes


# ---------------------------------------------------------------------------
# Edge building
# ---------------------------------------------------------------------------

def build_edges(
    nodes: list[NavNode],
    polygon: "ShapelyPolygon",
    rtabmap_links: list[tuple[int, int, int]],
    *,
    edge_max_m: float = 3.0,
) -> list[NavEdge]:
    """Build nav edges via KDTree proximity + polygon containment check + RTABMap boost."""
    from scipy.spatial import cKDTree

    if not nodes:
        return []

    prepped = prep(polygon)
    coords = np.array([[n.x, n.y] for n in nodes])
    node_ids = [n.node_id for n in nodes]
    tree = cKDTree(coords)

    edges: list[NavEdge] = []
    seen: set[frozenset[str]] = set()

    def _add_edge(i: int, j: int, kind: str, link_count: int = 0) -> None:
        key = frozenset([node_ids[i], node_ids[j]])
        if key in seen:
            return
        seg = LineString([coords[i], coords[j]])
        # Polygon containment check: covers allows boundary touching
        if not prepped.covers(seg):
            return
        d = float(np.linalg.norm(coords[i] - coords[j]))
        edges.append(
            NavEdge(
                edge_id=str(uuid.uuid4()),
                from_id=node_ids[i],
                to_id=node_ids[j],
                weight_m=round(d, 4),
                kind=kind,
                source_link_count=link_count,
            )
        )
        seen.add(key)

    # Grid edges: KDTree radius query
    pairs = tree.query_pairs(edge_max_m)
    for i, j in pairs:
        _add_edge(i, j, "grid")

    logger.info("Grid edges after polygon filter: %d", len(edges))

    # RTABMap boost edges
    # Build pose_id → node index mapping
    pose_to_node: dict[int, int] = {}
    for ni, node in enumerate(nodes):
        for pid in node.source_pose_ids:
            pose_to_node[pid] = ni

    boost_added = 0
    for from_id, to_id, link_type in rtabmap_links:
        ni = pose_to_node.get(from_id)
        nj = pose_to_node.get(to_id)
        if ni is None or nj is None or ni == nj:
            continue
        kind = "loop_closure" if link_type in (1, 2, 3) else "neighbor"
        before = len(edges)
        _add_edge(ni, nj, kind, link_count=1)
        if len(edges) > before:
            boost_added += 1

    logger.info("RTABMap boost edges added: %d (total: %d)", boost_added, len(edges))
    return edges


# ---------------------------------------------------------------------------
# Component assignment + connectivity repair
# ---------------------------------------------------------------------------

def _union_find_components(
    n_nodes: int, adj: dict[int, list[int]]
) -> list[int]:
    """Simple union-find for connected components. Returns component id per node."""
    parent = list(range(n_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, neighbors in adj.items():
        for j in neighbors:
            union(i, j)

    return [find(i) for i in range(n_nodes)]


def assign_components(
    nodes: list[NavNode],
    edges: list[NavEdge],
    polygon: "ShapelyPolygon",
    *,
    cross_session_bridge_m: float = 5.0,
) -> dict:
    """Assign component_id to each node. Attempt cross-session bridge if needed."""
    if not nodes:
        return {"components": 0, "largest_component_size": 0, "bridge_attempted": False}

    node_index = {n.node_id: i for i, n in enumerate(nodes)}
    adj: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    for e in edges:
        i = node_index.get(e.from_id)
        j = node_index.get(e.to_id)
        if i is not None and j is not None:
            adj[i].append(j)
            adj[j].append(i)

    raw_comp = _union_find_components(len(nodes), adj)
    comp_ids_unique = list(set(raw_comp))
    # Remap to 0-based contiguous
    comp_remap = {old: new for new, old in enumerate(comp_ids_unique)}
    comp_list = [comp_remap[c] for c in raw_comp]
    for node, cid in zip(nodes, comp_list):
        node.component_id = cid

    from collections import Counter
    comp_sizes = Counter(comp_list)
    n_components = len(comp_sizes)
    largest_size = max(comp_sizes.values())
    largest_comp_id = comp_sizes.most_common(1)[0][0]
    largest_ratio = largest_size / len(nodes)

    logger.info(
        "Components: %d, largest=%d/%d (%.1f%%)",
        n_components, largest_size, len(nodes), largest_ratio * 100,
    )

    bridge_attempted = False
    bridge_added = False

    if largest_ratio < 0.80 and n_components >= 2:
        logger.info("Largest component < 80%% — attempting cross-session bridge")
        bridge_attempted = True
        from scipy.spatial import cKDTree
        from shapely.prepared import prep as sprep

        prepped = sprep(polygon)
        coords = np.array([[n.x, n.y] for n in nodes])

        # Find two largest components
        top2 = comp_sizes.most_common(2)
        cid_a, cid_b = top2[0][0], top2[1][0]
        idx_a = [i for i, c in enumerate(comp_list) if c == cid_a]
        idx_b = [i for i, c in enumerate(comp_list) if c == cid_b]

        tree_b = cKDTree(coords[idx_b])
        best_dist = float("inf")
        best_pair = None
        for ia in idx_a:
            dists, neighbors = tree_b.query(coords[ia], k=1)
            if np.isscalar(dists) and dists < best_dist:
                ib_local = int(neighbors)
                ib = idx_b[ib_local]
                seg = LineString([coords[ia], coords[ib]])
                if dists <= cross_session_bridge_m and prepped.covers(seg):
                    best_dist = float(dists)
                    best_pair = (ia, ib)

        if best_pair is not None:
            ia, ib = best_pair
            bridge_edge = NavEdge(
                edge_id=str(uuid.uuid4()),
                from_id=nodes[ia].node_id,
                to_id=nodes[ib].node_id,
                weight_m=round(best_dist, 4),
                kind="neighbor",
                source_link_count=0,
            )
            edges.append(bridge_edge)
            bridge_added = True
            logger.info("Cross-session bridge added: dist=%.2f m", best_dist)

            # Recompute components
            adj[ia].append(ib)
            adj[ib].append(ia)
            raw_comp2 = _union_find_components(len(nodes), adj)
            comp_remap2 = {old: new for new, old in enumerate(set(raw_comp2))}
            comp_list2 = [comp_remap2[c] for c in raw_comp2]
            for node, cid in zip(nodes, comp_list2):
                node.component_id = cid
            comp_sizes2 = Counter(comp_list2)
            n_components = len(comp_sizes2)
            largest_size = max(comp_sizes2.values())
            largest_ratio = largest_size / len(nodes)
            logger.info(
                "After bridge: components=%d, largest=%d/%d (%.1f%%)",
                n_components, largest_size, len(nodes), largest_ratio * 100,
            )
        else:
            logger.warning(
                "Cross-session bridge failed: no polygon-internal segment ≤ %.1f m found",
                cross_session_bridge_m,
            )

    # Prune noise components (≤ 2 nodes)
    if True:
        noise_nodes = {i for i, c in enumerate(comp_list) if comp_sizes.get(c, 0) <= 2}
        # Recount with updated comp_list from nodes
        final_comp_list = [n.component_id for n in nodes]
        final_comp_sizes = Counter(final_comp_list)
        noise_nodes_final = {i for i, c in enumerate(final_comp_list) if final_comp_sizes.get(c, 0) <= 2}
        if noise_nodes_final:
            logger.info("Pruning %d noise nodes (component size ≤ 2)", len(noise_nodes_final))
            keep_ids = {nodes[i].node_id for i in range(len(nodes)) if i not in noise_nodes_final}
            for n in nodes:
                if n.node_id not in keep_ids:
                    n.component_id = -1  # mark as pruned
            edges[:] = [e for e in edges if e.from_id in keep_ids and e.to_id in keep_ids]

    # Final stats
    active_nodes = [n for n in nodes if n.component_id >= 0]
    active_comps = Counter(n.component_id for n in active_nodes)
    n_final_components = len(active_comps)
    largest_final = max(active_comps.values()) if active_comps else 0
    largest_ratio_final = largest_final / len(active_nodes) if active_nodes else 0.0

    return {
        "components": n_final_components,
        "largest_component_size": largest_final,
        "largest_component_ratio": round(largest_ratio_final, 4),
        "bridge_attempted": bridge_attempted,
        "bridge_added": bridge_added,
    }


# ---------------------------------------------------------------------------
# GeoJSON / JSON serialization
# ---------------------------------------------------------------------------

def compute_polygon_edge_containment(
    edges: list[NavEdge],
    nodes: list[NavNode],
    polygon: "ShapelyPolygon",
) -> float:
    """Fraction of edge segments fully inside refined polygon."""
    if not edges:
        return 1.0
    prepped = prep(polygon)
    node_map = {n.node_id: (n.x, n.y) for n in nodes}
    inside = 0
    for e in edges:
        a = node_map.get(e.from_id)
        b = node_map.get(e.to_id)
        if a and b:
            if prepped.covers(LineString([a, b])):
                inside += 1
    return inside / len(edges)


def save_nav_geojson(
    nodes: list[NavNode],
    edges: list[NavEdge],
    out_nodes: Path,
    out_edges: Path,
) -> None:
    """Save nav_nodes.geojson and nav_edges.geojson."""
    # Degree computation
    degree: dict[str, int] = {n.node_id: 0 for n in nodes}
    for e in edges:
        if e.from_id in degree:
            degree[e.from_id] += 1
        if e.to_id in degree:
            degree[e.to_id] += 1

    active = [n for n in nodes if n.component_id >= 0]
    node_features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [n.x, n.y]},
            "properties": {
                "node_id": n.node_id,
                "degree": degree.get(n.node_id, 0),
                "map_id_dominant": n.map_id_dominant,
                "source_pose_count": len(n.source_pose_ids),
                "component_id": n.component_id,
            },
        }
        for n in active
    ]

    node_map = {n.node_id: (n.x, n.y) for n in active}
    active_ids = {n.node_id for n in active}
    edge_features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [list(node_map[e.from_id]), list(node_map[e.to_id])],
            },
            "properties": {
                "edge_id": e.edge_id,
                "from_node": e.from_id,
                "to_node": e.to_id,
                "weight_m": e.weight_m,
                "kind": e.kind,
                "source_link_count": e.source_link_count,
            },
        }
        for e in edges
        if e.from_id in active_ids and e.to_id in active_ids
    ]

    out_nodes.parent.mkdir(parents=True, exist_ok=True)
    out_nodes.write_text(
        json.dumps({"type": "FeatureCollection", "features": node_features}, indent=2)
    )
    out_edges.write_text(
        json.dumps({"type": "FeatureCollection", "features": edge_features}, indent=2)
    )
    logger.info(
        "Saved nav GeoJSON: %d nodes → %s, %d edges → %s",
        len(node_features), out_nodes, len(edge_features), out_edges,
    )


def save_nav_graph_json(
    nodes: list[NavNode],
    edges: list[NavEdge],
    stats: dict,
    out_path: Path,
) -> None:
    """Save nav_graph.json in node-link format compatible with networkx.node_link_graph."""
    active = [n for n in nodes if n.component_id >= 0]
    active_ids = {n.node_id for n in active}

    # Compute degrees
    degree: dict[str, int] = {n.node_id: 0 for n in active}
    for e in edges:
        if e.from_id in degree:
            degree[e.from_id] += 1
        if e.to_id in degree:
            degree[e.to_id] += 1

    node_link = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": [
            {
                "id": n.node_id,
                "x": n.x,
                "y": n.y,
                "map_id_dominant": n.map_id_dominant,
                "source_pose_count": len(n.source_pose_ids),
                "component_id": n.component_id,
                "degree": degree.get(n.node_id, 0),
            }
            for n in active
        ],
        "links": [
            {
                "source": e.from_id,
                "target": e.to_id,
                "weight": e.weight_m,
                "kind": e.kind,
                "source_link_count": e.source_link_count,
            }
            for e in edges
            if e.from_id in active_ids and e.to_id in active_ids
        ],
        "stats": stats,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(node_link, indent=2))
    logger.info("Saved nav_graph.json → %s (%d nodes, %d edges)", out_path, len(active), len(edges))
