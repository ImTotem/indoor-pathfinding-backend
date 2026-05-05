"""RouteService display-grid behavior (Sprint 61)."""
from __future__ import annotations

from uuid import UUID, uuid4

import networkx as nx
import pytest

from indoor_server.application.routing.route_service import RouteService
from indoor_server.domain.routing.models import MapNodeRow
from indoor_server.interfaces.api.schemas import RouteEndpoint


class _Loader:
    def __init__(self, graph: nx.Graph, nodes: dict[UUID, MapNodeRow]) -> None:
        self._graph = graph
        self._nodes = nodes
        self._build_job_id = uuid4()

    async def load(self, *, scan_id: str):
        return self._graph, self._nodes, self._build_job_id


def _node(node_id: UUID, x: float, y: float) -> MapNodeRow:
    return MapNodeRow(
        node_id=node_id,
        x=x,
        y=y,
        z=0.0,
        node_type="corridor",
        label=None,
        poi_mark_id=None,
        source_ref={"graph_source": "display_navigation_grid"},
    )


@pytest.mark.asyncio
async def test_display_navigation_grid_route_keeps_raw_polygon_safe_polyline() -> None:
    a, b, c = uuid4(), uuid4(), uuid4()
    nodes = {
        a: _node(a, 0.0, 0.0),
        b: _node(b, 1.0, 1.0),
        c: _node(c, 2.0, 2.0),
    }
    graph = nx.Graph()
    for node in nodes.values():
        graph.add_node(
            node.node_id,
            x=node.x,
            y=node.y,
            z=node.z,
            node_type=node.node_type,
            label=node.label,
            poi_mark_id=node.poi_mark_id,
            source_ref=node.source_ref,
        )
    graph.add_edge(a, b, length_m=1.414)
    graph.add_edge(b, c, length_m=1.414)

    service = RouteService(_Loader(graph, nodes), heuristic_use_z=False)  # type: ignore[arg-type]

    path = await service.compute(
        scan_id=str(uuid4()),
        start=RouteEndpoint(node_id=a),
        goal=RouteEndpoint(node_id=c),
    )

    assert path.polyline == [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 2.0, 0.0)]
