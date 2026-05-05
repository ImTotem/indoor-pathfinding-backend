"""Sprint 84 — interfloor_mark must not become synthetic negative POI FKs."""
from __future__ import annotations

from uuid import UUID, uuid4

from indoor_server.application.building.build_service import (
    _append_interfloor_connector_nodes,
)
from indoor_server.domain.building.enums import EdgeType, NodeType
from indoor_server.domain.building.models import MapEdgeVO, MapNodeVO
from indoor_server.infrastructure.db.repositories.interfloor_mark_repo import (
    InterfloorMarkDbRow,
)


def _mark(connector_type: str, prefix: str, idx: int = 0) -> InterfloorMarkDbRow:
    return InterfloorMarkDbRow(
        id=10 + idx,
        scan_id="00000000-0000-0000-0000-000000000000",
        keyframe_seq=1,
        created_at=1234567890000,
        connector_type=connector_type,
        prefix=prefix,
        pose_matrix=b"\x00" * 64,
        tx=1.0,
        ty=2.0,
        tz=0.0,
    )


def _base_node(scan_id: UUID, build_job_id: UUID) -> MapNodeVO:
    return MapNodeVO(
        node_id=uuid4(),
        scan_id=scan_id,
        build_job_id=build_job_id,
        node_type=NodeType.CORRIDOR,
        x=1.5,
        y=2.0,
        z=0.0,
    )


def test_connector_node_uses_source_ref_not_negative_poi_fk() -> None:
    scan_id = uuid4()
    build_job_id = uuid4()
    nodes, edges = _append_interfloor_connector_nodes(
        interfloor_marks=[_mark("escalator", "ES-1")],
        nodes=[_base_node(scan_id, build_job_id)],
        edges=[],
        scan_id=scan_id,
        build_job_id=build_job_id,
    )

    connector = next(
        node
        for node in nodes
        if (node.source_ref or {}).get("role") == "vertical_connector_stop"
    )
    assert connector.poi_mark_id is None
    assert connector.source_ref is not None
    assert connector.source_ref["connector_type"] == "escalator"
    assert connector.source_ref["connector_key"] == "es_1"

    assert len(edges) == 1
    assert edges[0].edge_type == EdgeType.POI_SPUR
    assert edges[0].from_node_id == connector.node_id


def test_empty_interfloor_marks_preserve_graph() -> None:
    scan_id = uuid4()
    build_job_id = uuid4()
    node = _base_node(scan_id, build_job_id)
    edge = MapEdgeVO(
        edge_id=uuid4(),
        scan_id=scan_id,
        build_job_id=build_job_id,
        from_node_id=node.node_id,
        to_node_id=node.node_id,
        edge_type=EdgeType.SKELETON,
        polyline=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        length_m=1.0,
    )

    nodes, edges = _append_interfloor_connector_nodes(
        interfloor_marks=[],
        nodes=[node],
        edges=[edge],
        scan_id=scan_id,
        build_job_id=build_job_id,
    )

    assert nodes == [node]
    assert edges == [edge]
