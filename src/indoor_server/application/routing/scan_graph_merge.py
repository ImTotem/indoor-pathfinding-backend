"""In-memory merge for multiple RTABMap scan routing graphs."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from uuid import NAMESPACE_URL, UUID, uuid5

import networkx as nx
import numpy as np

from indoor_server.application.routing.vertical_connectors import connector_key_for_node
from indoor_server.domain.routing.models import MapEdgeRow, MapNodeRow


@dataclass(frozen=True)
class ScanGraphFragment:
    scan_id: str
    build_job_id: UUID
    nodes: list[MapNodeRow]
    edges: list[MapEdgeRow]


@dataclass(frozen=True)
class RigidTransform2D:
    rotation_rad: float = 0.0
    tx: float = 0.0
    ty: float = 0.0

    def apply(self, x: float, y: float) -> tuple[float, float]:
        cos_v = math.cos(self.rotation_rad)
        sin_v = math.sin(self.rotation_rad)
        return (
            cos_v * x - sin_v * y + self.tx,
            sin_v * x + cos_v * y + self.ty,
        )

    def to_metadata(self) -> dict[str, float]:
        return {
            "rotation_deg": math.degrees(self.rotation_rad),
            "tx": self.tx,
            "ty": self.ty,
        }


@dataclass(frozen=True)
class ScanMergeReport:
    fragment_count: int
    merged_node_count: int
    merged_edge_count: int
    overlap_bridge_count: int
    transforms: dict[str, dict[str, object]]


@dataclass(frozen=True)
class ScanMergeResult:
    graph: nx.Graph
    node_lookup: dict[UUID, MapNodeRow]
    report: ScanMergeReport


class ScanGraphMerger:
    """Merge scan graphs by overlap without persisting a new graph artifact."""

    def __init__(
        self,
        *,
        overlap_tolerance_m: float = 0.75,
        min_overlap_pairs: int = 2,
    ) -> None:
        self._overlap_tolerance_m = overlap_tolerance_m
        self._min_overlap_pairs = min_overlap_pairs

    def merge(
        self,
        fragments: list[ScanGraphFragment],
        *,
        merge_overlaps: bool,
    ) -> ScanMergeResult:
        graph: nx.Graph = nx.Graph()
        node_lookup: dict[UUID, MapNodeRow] = {}
        transforms: dict[str, dict[str, object]] = {}
        overlap_bridge_count = 0

        for index, fragment in enumerate(fragments):
            base_nodes = list(node_lookup.values())
            transform = RigidTransform2D()
            accepted = index == 0 or not merge_overlaps
            overlap_pairs: list[tuple[UUID, UUID, float]] = []
            source = "identity" if index == 0 else "disabled"

            if index > 0 and merge_overlaps and base_nodes:
                transform, source = self._estimate_transform(base_nodes, fragment.nodes)
                transformed_nodes = [
                    _transform_node(node, transform, source=source)
                    for node in fragment.nodes
                ]
                overlap_pairs = self._find_overlap_pairs(base_nodes, transformed_nodes)
                accepted = len(overlap_pairs) >= self._min_overlap_pairs
            else:
                transformed_nodes = [
                    _transform_node(node, transform, source=source)
                    for node in fragment.nodes
                ]

            if not accepted and index > 0:
                transform = RigidTransform2D()
                transformed_nodes = [
                    _transform_node(node, transform, source="unmerged_identity")
                    for node in fragment.nodes
                ]
                source = "unmerged_identity"
                overlap_pairs = []

            self._add_fragment(graph, node_lookup, transformed_nodes, fragment.edges)
            for first_id, second_id, distance_m in overlap_pairs:
                graph.add_edge(
                    first_id,
                    second_id,
                    length_m=max(distance_m, 0.05),
                    source_ref={
                        "edge_kind": "scan_overlap_bridge",
                        "source_scan_id": fragment.scan_id,
                        "transform_source": source,
                    },
                    edge_id=uuid5(
                        NAMESPACE_URL,
                        f"indoor-route-scan-merge:{first_id}:{second_id}",
                    ),
                )
                overlap_bridge_count += 1

            transforms[fragment.scan_id] = {
                "accepted": accepted,
                "source": source,
                "overlap_pair_count": len(overlap_pairs),
                "transform": transform.to_metadata(),
            }

        return ScanMergeResult(
            graph=graph,
            node_lookup=node_lookup,
            report=ScanMergeReport(
                fragment_count=len(fragments),
                merged_node_count=graph.number_of_nodes(),
                merged_edge_count=graph.number_of_edges(),
                overlap_bridge_count=overlap_bridge_count,
                transforms=transforms,
            ),
        )

    def _add_fragment(
        self,
        graph: nx.Graph,
        node_lookup: dict[UUID, MapNodeRow],
        nodes: list[MapNodeRow],
        edges: list[MapEdgeRow],
    ) -> None:
        for node in nodes:
            graph.add_node(
                node.node_id,
                x=node.x,
                y=node.y,
                z=node.z,
                node_type=node.node_type,
                label=node.label,
                poi_mark_id=node.poi_mark_id,
                source_ref=node.source_ref,
                scan_id=str(node.scan_id) if node.scan_id is not None else None,
                level_id=node.level_id,
            )
            node_lookup[node.node_id] = node
        for edge in edges:
            if edge.from_node_id not in node_lookup or edge.to_node_id not in node_lookup:
                continue
            graph.add_edge(
                edge.from_node_id,
                edge.to_node_id,
                length_m=edge.length_m,
                edge_id=edge.edge_id,
                source_ref={"edge_kind": "scan_internal"},
            )

    def _estimate_transform(
        self,
        base_nodes: list[MapNodeRow],
        moving_nodes: list[MapNodeRow],
    ) -> tuple[RigidTransform2D, str]:
        semantic_pairs = _semantic_correspondences(base_nodes, moving_nodes)
        if semantic_pairs:
            return _rigid_from_pairs(semantic_pairs), "semantic_correspondence"
        return _centroid_translation(base_nodes, moving_nodes), "centroid_translation"

    def _find_overlap_pairs(
        self,
        base_nodes: list[MapNodeRow],
        moving_nodes: list[MapNodeRow],
    ) -> list[tuple[UUID, UUID, float]]:
        base_candidates = _walkable_nodes(base_nodes)
        moving_candidates = _walkable_nodes(moving_nodes)
        pairs: list[tuple[UUID, UUID, float]] = []
        used_base: set[UUID] = set()
        for moving in moving_candidates:
            nearest: tuple[MapNodeRow, float] | None = None
            for base in base_candidates:
                if base.node_id in used_base:
                    continue
                distance_m = _distance_2d(base, moving)
                if nearest is None or distance_m < nearest[1]:
                    nearest = (base, distance_m)
            if nearest is None or nearest[1] > self._overlap_tolerance_m:
                continue
            used_base.add(nearest[0].node_id)
            pairs.append((nearest[0].node_id, moving.node_id, nearest[1]))
        return pairs


def _transform_node(
    node: MapNodeRow,
    transform: RigidTransform2D,
    *,
    source: str,
) -> MapNodeRow:
    x, y = transform.apply(node.x, node.y)
    source_ref = dict(node.source_ref or {})
    source_ref["scan_merge_transform"] = {
        "source": source,
        **transform.to_metadata(),
    }
    return replace(node, x=x, y=y, source_ref=source_ref)


def _semantic_correspondences(
    base_nodes: list[MapNodeRow],
    moving_nodes: list[MapNodeRow],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    base_by_key = _semantic_key_index(base_nodes)
    moving_by_key = _semantic_key_index(moving_nodes)
    pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for key in sorted(base_by_key.keys() & moving_by_key.keys()):
        base = base_by_key[key]
        moving = moving_by_key[key]
        pairs.append(((moving.x, moving.y), (base.x, base.y)))
    return pairs


def _semantic_key_index(nodes: list[MapNodeRow]) -> dict[str, MapNodeRow]:
    out: dict[str, MapNodeRow] = {}
    for node in nodes:
        key = connector_key_for_node(node)
        if key is None and node.label:
            key = f"label:{_normalize_label(node.label)}"
        if key is None:
            continue
        out.setdefault(key, node)
    return out


def _rigid_from_pairs(
    pairs: list[tuple[tuple[float, float], tuple[float, float]]],
) -> RigidTransform2D:
    if len(pairs) == 1:
        (sx, sy), (tx, ty) = pairs[0]
        return RigidTransform2D(tx=tx - sx, ty=ty - sy)

    source = np.array([pair[0] for pair in pairs], dtype=float)
    target = np.array([pair[1] for pair in pairs], dtype=float)
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_centroid - rotation @ source_centroid
    rotation_rad = math.atan2(rotation[1, 0], rotation[0, 0])
    return RigidTransform2D(
        rotation_rad=rotation_rad,
        tx=float(translation[0]),
        ty=float(translation[1]),
    )


def _centroid_translation(
    base_nodes: list[MapNodeRow],
    moving_nodes: list[MapNodeRow],
) -> RigidTransform2D:
    base = _walkable_nodes(base_nodes) or base_nodes
    moving = _walkable_nodes(moving_nodes) or moving_nodes
    if not base or not moving:
        return RigidTransform2D()
    base_x = sum(node.x for node in base) / len(base)
    base_y = sum(node.y for node in base) / len(base)
    moving_x = sum(node.x for node in moving) / len(moving)
    moving_y = sum(node.y for node in moving) / len(moving)
    return RigidTransform2D(tx=base_x - moving_x, ty=base_y - moving_y)


def _walkable_nodes(nodes: list[MapNodeRow]) -> list[MapNodeRow]:
    return [
        node
        for node in nodes
        if node.node_type not in {"poi", "poi_attach"}
    ]


def _distance_2d(first: MapNodeRow, second: MapNodeRow) -> float:
    return math.hypot(first.x - second.x, first.y - second.y)


def _normalize_label(label: str) -> str:
    return "".join(ch.lower() for ch in label.strip() if not ch.isspace())
