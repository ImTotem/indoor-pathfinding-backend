"""Scan graph merge tests (Sprint 62)."""
from __future__ import annotations

from uuid import UUID, uuid4

import networkx as nx

from indoor_server.application.routing.scan_graph_merge import (
    ScanGraphFragment,
    ScanGraphMerger,
)
from indoor_server.domain.routing.models import MapEdgeRow, MapNodeRow


def _node(node_id: UUID, x: float, y: float, *, scan_id: UUID) -> MapNodeRow:
    return MapNodeRow(
        node_id=node_id,
        x=x,
        y=y,
        z=0.0,
        node_type="corridor",
        label=None,
        poi_mark_id=None,
        scan_id=scan_id,
        build_job_id=uuid4(),
    )


def _edge(a: UUID, b: UUID) -> MapEdgeRow:
    return MapEdgeRow(edge_id=uuid4(), from_node_id=a, to_node_id=b, length_m=1.0)


def _fragment(scan_id: UUID, xs: list[float]) -> ScanGraphFragment:
    ids = [uuid4() for _ in xs]
    nodes = [
        _node(node_id, x, 0.0, scan_id=scan_id)
        for node_id, x in zip(ids, xs, strict=True)
    ]
    return ScanGraphFragment(
        scan_id=str(scan_id),
        build_job_id=uuid4(),
        nodes=nodes,
        edges=[_edge(ids[0], ids[1]), _edge(ids[1], ids[2])],
    )


def test_overlap_merge_adds_bridge_edges_after_centroid_alignment() -> None:
    first = _fragment(uuid4(), [0.0, 1.0, 2.0])
    second = _fragment(uuid4(), [10.0, 11.0, 12.0])

    result = ScanGraphMerger(overlap_tolerance_m=0.2).merge(
        [first, second],
        merge_overlaps=True,
    )

    assert result.report.fragment_count == 2
    assert result.report.overlap_bridge_count >= 2
    assert result.report.transforms[second.scan_id]["accepted"] is True
    assert nx.number_connected_components(result.graph) == 1


def test_merge_overlaps_false_keeps_scan_components_separate() -> None:
    first = _fragment(uuid4(), [0.0, 1.0, 2.0])
    second = _fragment(uuid4(), [10.0, 11.0, 12.0])

    result = ScanGraphMerger(overlap_tolerance_m=0.2).merge(
        [first, second],
        merge_overlaps=False,
    )

    assert result.report.overlap_bridge_count == 0
    assert nx.number_connected_components(result.graph) == 2
