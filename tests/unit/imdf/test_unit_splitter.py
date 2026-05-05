"""UnitSplitter — unit polygon 완전 제거로 인해 deprecated.

UnitSplitter와 CartographicLayoutBuilder 코드는 코드베이스에 남아 있으나
ImdfBuilder/ExportService 경로에서 호출되지 않으며 unit.geojson은 항상 빈 배열.
이 파일은 기존 API 호환성 확인만 수행한다.
"""
from __future__ import annotations

from uuid import UUID

from indoor_server.application.imdf.unit_splitter import UnitSplitter
from indoor_server.domain.routing.models import MapEdgeRow, MapNodeRow

_SCAN_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def _node(node_id: str, x: float, y: float) -> MapNodeRow:
    return MapNodeRow(
        node_id=UUID(node_id),
        x=x,
        y=y,
        z=0.0,
        node_type="corridor",
        label=None,
        poi_mark_id=None,
    )


def test_splitter_api_still_callable() -> None:
    """UnitSplitter.split()은 여전히 호출 가능해야 한다 (API 호환성)."""
    footprint = {
        "type": "Polygon",
        "coordinates": [[
            [0.0, 0.0], [10.0, 0.0], [10.0, 6.0], [0.0, 6.0], [0.0, 0.0],
        ]],
    }
    n1 = _node("00000000-0000-0000-0000-000000000001", 1.0, 3.0)
    n2 = _node("00000000-0000-0000-0000-000000000002", 9.0, 3.0)
    edge = MapEdgeRow(
        edge_id=UUID("00000000-0000-0000-0000-000000000003"),
        from_node_id=n1.node_id,
        to_node_id=n2.node_id,
        length_m=8.0,
    )

    result = UnitSplitter().split(
        scan_id=_SCAN_ID,
        footprint_geojson=footprint,
        nodes=[n1, n2],
        edges=[edge],
    )

    # UnitSplitter는 내부적으로 CartographicLayoutBuilder를 통해 결과를 생성하지만
    # ImdfBuilder에서는 units 파라미터가 무시되므로 unit.geojson은 항상 빈 배열.
    assert isinstance(result.units, list)
    assert isinstance(result.metadata, dict)
