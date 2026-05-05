"""Vertical connector routing tests (Sprint 62)."""
from __future__ import annotations

from uuid import UUID, uuid4

import networkx as nx

from indoor_server.application.routing.vertical_connectors import (
    VerticalEdge,
    add_vertical_edges,
    connector_key_for_node,
)
from indoor_server.domain.routing.models import MapNodeRow


def _node(
    node_id: UUID,
    *,
    scan_id: UUID,
    label: str,
    facility_type: str,
    connector_key: str | None = None,
) -> MapNodeRow:
    source_ref: dict[str, object] = {
        "graph_source": "display_navigation_grid",
        "facility_type": facility_type,
    }
    if connector_key is not None:
        source_ref["connector_key"] = connector_key
    return MapNodeRow(
        node_id=node_id,
        x=0.0,
        y=0.0,
        z=0.0,
        node_type="poi",
        label=label,
        poi_mark_id=None,
        source_ref=source_ref,
        scan_id=scan_id,
    )


def test_connector_key_normalizes_stair_and_elevator_labels() -> None:
    scan_id = uuid4()
    stair = _node(
        uuid4(),
        scan_id=scan_id,
        label="STAIR_A 동쪽 계단",
        facility_type="stair",
    )
    elevator = _node(
        uuid4(),
        scan_id=scan_id,
        label="ELEV_CENTER 중앙 엘리베이터",
        facility_type="elevator",
    )

    assert connector_key_for_node(stair) == "stair:a"
    assert connector_key_for_node(elevator) == "elevator:center"


def test_add_vertical_edges_connects_same_connector_across_scans() -> None:
    scan_a = uuid4()
    scan_b = uuid4()
    first_id = uuid4()
    second_id = uuid4()
    nodes = {
        first_id: _node(
            first_id,
            scan_id=scan_a,
            label="ELEV_CENTER 중앙 엘리베이터",
            facility_type="elevator",
            connector_key="elevator:center",
        ),
        second_id: _node(
            second_id,
            scan_id=scan_b,
            label="ELEV_CENTER 중앙 엘리베이터",
            facility_type="elevator",
            connector_key="elevator:center",
        ),
    }
    graph = nx.Graph()
    for node in nodes.values():
        graph.add_node(node.node_id, x=node.x, y=node.y, z=node.z)

    added = add_vertical_edges(graph, nodes)

    assert added == 1
    edge_data = graph.get_edge_data(first_id, second_id)
    assert edge_data is not None
    assert edge_data["length_m"] == 8.0
    assert edge_data["source_ref"]["connector_key"] == "elevator:center"


def test_explicit_escalator_edge_is_preserved() -> None:
    scan_a = uuid4()
    scan_b = uuid4()
    first_id = uuid4()
    second_id = uuid4()
    nodes = {
        first_id: _node(first_id, scan_id=scan_a, label="ES-1", facility_type=""),
        second_id: _node(second_id, scan_id=scan_b, label="ES-1", facility_type=""),
    }
    graph = nx.Graph()
    for node in nodes.values():
        graph.add_node(node.node_id, x=node.x, y=node.y, z=node.z)

    added = add_vertical_edges(
        graph,
        nodes,
        explicit_edges=[
            VerticalEdge(
                from_node_id=first_id,
                to_node_id=second_id,
                length_m=12.0,
                source_ref={
                    "edge_kind": "vertical_connector",
                    "connector_type": "escalator",
                    "connector_key": "es_1",
                },
            )
        ],
    )

    assert added == 1
    edge_data = graph.get_edge_data(first_id, second_id)
    assert edge_data["source_ref"]["connector_type"] == "escalator"
