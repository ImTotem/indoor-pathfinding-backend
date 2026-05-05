"""RouteService multi-scan and multi-floor behavior (Sprint 62)."""
from __future__ import annotations

from uuid import UUID, uuid4

import networkx as nx
import pytest

from indoor_server.application.routing.route_service import RouteService
from indoor_server.domain.routing.models import MapNodeRow
from indoor_server.interfaces.api.schemas import RouteEndpoint


class _MultiLoader:
    def __init__(self, graph: nx.Graph, nodes: dict[UUID, MapNodeRow]) -> None:
        self._graph = graph
        self._nodes = nodes
        self._build_job_ids = [uuid4(), uuid4()]

    async def load_many(
        self,
        *,
        scan_ids: list[str],
        merge_overlaps: bool = False,
    ):
        return self._graph, self._nodes, self._build_job_ids, None


def _node(
    node_id: UUID,
    x: float,
    *,
    scan_id: UUID,
    node_type: str = "corridor",
    label: str | None = None,
) -> MapNodeRow:
    return MapNodeRow(
        node_id=node_id,
        x=x,
        y=0.0,
        z=0.0,
        node_type=node_type,
        label=label,
        poi_mark_id=None,
        scan_id=scan_id,
        source_ref={"graph_source": "display_navigation_grid"}
        if node_type == "corridor"
        else {
            "graph_source": "display_navigation_grid",
            "facility_type": "elevator",
            "connector_key": "elevator:center",
        },
    )


@pytest.mark.asyncio
async def test_multiscan_route_uses_vertical_transition_cost() -> None:
    scan_a = uuid4()
    scan_b = uuid4()
    start = uuid4()
    elev_a = uuid4()
    elev_b = uuid4()
    goal = uuid4()
    nodes = {
        start: _node(start, 0.0, scan_id=scan_a),
        elev_a: _node(elev_a, 1.0, scan_id=scan_a, node_type="poi", label="ELEV_CENTER"),
        elev_b: _node(elev_b, 1.0, scan_id=scan_b, node_type="poi", label="ELEV_CENTER"),
        goal: _node(goal, 2.0, scan_id=scan_b),
    }
    graph = nx.Graph()
    for node in nodes.values():
        graph.add_node(
            node.node_id,
            x=node.x,
            y=node.y,
            z=node.z,
            node_type=node.node_type,
            source_ref=node.source_ref,
        )
    graph.add_edge(start, elev_a, length_m=1.0)
    graph.add_edge(elev_a, elev_b, length_m=8.0)
    graph.add_edge(elev_b, goal, length_m=1.0)

    loader = _MultiLoader(graph, nodes)
    service = RouteService(loader, heuristic_use_z=False)  # type: ignore[arg-type]

    path = await service.compute(
        scan_id=str(scan_a),
        scan_ids=[str(scan_a), str(scan_b)],
        start=RouteEndpoint(node_id=start),
        goal=RouteEndpoint(node_id=goal),
    )

    assert [node.node_id for node in path.nodes_in_order] == [start, elev_a, elev_b, goal]
    assert path.length_m == pytest.approx(10.0)
    assert path.scan_ids == [scan_a, scan_b]
    assert path.build_job_ids == loader._build_job_ids
    assert path.route_metadata is not None
    assert path.route_metadata["route_scope"] == "multi_scan"
