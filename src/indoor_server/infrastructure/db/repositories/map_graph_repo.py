"""MapGraphRepository — map_node/map_edge bulk insert + soft delete + routing load."""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.domain.building.models import MapEdgeVO, MapNodeVO
from indoor_server.domain.routing.models import MapEdgeRow, MapNodeRow
from indoor_server.infrastructure.db import tables as t

logger = logging.getLogger(__name__)


def _node_geom_wkt(node: MapNodeVO) -> str:
    """POINT Z EWKT (SRID=0 prefix). geoalchemy2가 ST_GeomFromEWKT로 자동 래핑."""
    return f"SRID=0;POINTZ({node.x} {node.y} {node.z})"


def _edge_geom_wkt(edge: MapEdgeVO) -> str:
    """LINESTRING Z EWKT (SRID=0 prefix). 점 사이는 쉼표로 구분."""
    pts = ", ".join(f"{x} {y} {z}" for x, y, z in edge.polyline)
    return f"SRID=0;LINESTRINGZ({pts})"


class MapGraphRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def replace_graph(
        self,
        *,
        scan_id: str,
        build_job_id: str,
        nodes: Sequence[MapNodeVO],
        edges: Sequence[MapEdgeVO],
    ) -> None:
        """
        soft delete → bulk insert → stale 삭제. 트랜잭션 1개.
        호출 측이 트랜잭션을 열어야 한다.
        """
        # 1. 기존 active rows → is_stale=true
        await self._session.execute(
            sa.update(t.map_edge)
            .where(
                t.map_edge.c.scan_id == scan_id,
                t.map_edge.c.is_stale == sa.false(),
            )
            .values(is_stale=True)
        )
        await self._session.execute(
            sa.update(t.map_node)
            .where(
                t.map_node.c.scan_id == scan_id,
                t.map_node.c.is_stale == sa.false(),
            )
            .values(is_stale=True)
        )

        # 2. 새 rows 삽입
        if nodes:
            node_rows = [
                {
                    "node_id": str(n.node_id),
                    "scan_id": str(n.scan_id),
                    "build_job_id": str(n.build_job_id),
                    "node_type": n.node_type.value,
                    "geom": _node_geom_wkt(n),
                    "label": n.label,
                    "poi_mark_id": n.poi_mark_id,
                    "source_ref": n.source_ref,
                    "is_stale": False,
                }
                for n in nodes
            ]
            await self._session.execute(sa.insert(t.map_node), node_rows)

        if edges:
            edge_rows = [
                {
                    "edge_id": str(e.edge_id),
                    "scan_id": str(e.scan_id),
                    "build_job_id": str(e.build_job_id),
                    "from_node_id": str(e.from_node_id),
                    "to_node_id": str(e.to_node_id),
                    "edge_type": e.edge_type.value,
                    "geom": _edge_geom_wkt(e),
                    "length_m": e.length_m,
                    "is_stale": False,
                }
                for e in edges
            ]
            await self._session.execute(sa.insert(t.map_edge), edge_rows)

        # 3. 이전 stale rows 삭제 (새 job이 아닌 이전 job의 stale rows만)
        await self._session.execute(
            sa.delete(t.map_edge).where(
                t.map_edge.c.scan_id == scan_id,
                t.map_edge.c.is_stale == sa.true(),
                t.map_edge.c.build_job_id != build_job_id,
            )
        )
        await self._session.execute(
            sa.delete(t.map_node).where(
                t.map_node.c.scan_id == scan_id,
                t.map_node.c.is_stale == sa.true(),
                t.map_node.c.build_job_id != build_job_id,
            )
        )

        logger.info(
            "graph replaced scan_id=%s nodes=%d edges=%d",
            scan_id,
            len(nodes),
            len(edges),
        )

    async def load_graph_geojson(self, scan_id: str) -> list[dict[str, object]]:
        """scan_id의 non-stale nodes/edges → GeoJSON Feature 목록."""
        features: list[dict[str, object]] = []

        node_rows = (
            await self._session.execute(
                sa.select(
                    t.map_node.c.node_id,
                    t.map_node.c.node_type,
                    t.map_node.c.label,
                    t.map_node.c.poi_mark_id,
                    t.map_node.c.source_ref,
                    sa.func.ST_AsGeoJSON(t.map_node.c.geom).label("geom_json"),
                ).where(
                    t.map_node.c.scan_id == scan_id,
                    t.map_node.c.is_stale == sa.false(),
                )
            )
        ).fetchall()

        for row in node_rows:
            import json

            geom = json.loads(row.geom_json)
            props: dict[str, object] = {
                "node_id": row.node_id,
                "node_type": row.node_type,
                "label": row.label,
            }
            if row.poi_mark_id is not None:
                props["poi_mark_id"] = row.poi_mark_id
            if row.source_ref is not None:
                props["source_ref"] = row.source_ref
            features.append({"type": "Feature", "geometry": geom, "properties": props})

        edge_rows = (
            await self._session.execute(
                sa.select(
                    t.map_edge.c.edge_id,
                    t.map_edge.c.from_node_id,
                    t.map_edge.c.to_node_id,
                    t.map_edge.c.edge_type,
                    t.map_edge.c.length_m,
                    sa.func.ST_AsGeoJSON(t.map_edge.c.geom).label("geom_json"),
                ).where(
                    t.map_edge.c.scan_id == scan_id,
                    t.map_edge.c.is_stale == sa.false(),
                )
            )
        ).fetchall()

        for row in edge_rows:
            import json

            geom = json.loads(row.geom_json)
            features.append(
                {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {
                        "edge_id": row.edge_id,
                        "from_node_id": row.from_node_id,
                        "to_node_id": row.to_node_id,
                        "edge_type": row.edge_type,
                        "length_m": row.length_m,
                    },
                }
            )

        return features

    async def load_graph_for_routing(
        self,
        scan_id: str,
    ) -> tuple[list[MapNodeRow], list[MapEdgeRow], UUID | None]:
        """routing 용도로 non-stale map_node/map_edge를 로드.

        ST_AsGeoJSON 없이 geom 좌표를 직접 SELECT — networkx 변환에 최적화.

        Returns:
            (nodes, edges, latest_succeeded_build_job_id)
            build 완료 job이 없으면 build_job_id=None.
        """
        # 최신 succeeded build_job_id 조회
        job_row = (
            await self._session.execute(
                sa.select(t.build_job.c.build_job_id)
                .where(
                    t.build_job.c.scan_id == scan_id,
                    t.build_job.c.state == "succeeded",
                )
                .order_by(t.build_job.c.enqueued_at.desc())
                .limit(1)
            )
        ).first()
        build_job_id: UUID | None = UUID(job_row.build_job_id) if job_row else None

        node_rows = (
            await self._session.execute(
                sa.select(
                    t.map_node.c.node_id,
                    t.map_node.c.scan_id,
                    t.map_node.c.build_job_id,
                    t.map_node.c.node_type,
                    t.map_node.c.label,
                    t.map_node.c.poi_mark_id,
                    t.map_node.c.source_ref,
                    sa.func.ST_X(t.map_node.c.geom).label("x"),
                    sa.func.ST_Y(t.map_node.c.geom).label("y"),
                    sa.func.ST_Z(t.map_node.c.geom).label("z"),
                ).where(
                    t.map_node.c.scan_id == scan_id,
                    t.map_node.c.is_stale == sa.false(),
                )
            )
        ).fetchall()

        edge_rows = (
            await self._session.execute(
                sa.select(
                    t.map_edge.c.edge_id,
                    t.map_edge.c.scan_id,
                    t.map_edge.c.build_job_id,
                    t.map_edge.c.from_node_id,
                    t.map_edge.c.to_node_id,
                    t.map_edge.c.length_m,
                ).where(
                    t.map_edge.c.scan_id == scan_id,
                    t.map_edge.c.is_stale == sa.false(),
                )
            )
        ).fetchall()

        nodes = [
            MapNodeRow(
                node_id=UUID(r.node_id),
                x=float(r.x) if r.x is not None else 0.0,
                y=float(r.y) if r.y is not None else 0.0,
                z=float(r.z) if r.z is not None else 0.0,
                node_type=r.node_type,
                label=r.label,
                poi_mark_id=r.poi_mark_id,
                source_ref=r.source_ref,
                scan_id=UUID(r.scan_id),
                build_job_id=UUID(r.build_job_id),
                level_id=_level_id_from_source_ref(r.source_ref),
            )
            for r in node_rows
        ]
        edges = [
            MapEdgeRow(
                edge_id=UUID(r.edge_id),
                from_node_id=UUID(r.from_node_id),
                to_node_id=UUID(r.to_node_id),
                length_m=float(r.length_m),
                scan_id=UUID(r.scan_id),
                build_job_id=UUID(r.build_job_id),
            )
            for r in edge_rows
        ]
        return nodes, edges, build_job_id

    async def load_explicit_vertical_edges(
        self,
        scan_ids: Sequence[str],
    ) -> list[tuple[UUID, UUID, float, dict[str, object]]]:
        """Load vertical connector stop pairs for the selected scan graphs.

        The current DB schema stores vertical connector stops as route node IDs.
        Route-time multi-floor routing can therefore add in-memory transition
        edges without extending the persisted map_edge enum.
        """
        if not scan_ids:
            return []

        rows = (
            await self._session.execute(
                sa.select(
                    t.vertical_connector.c.connector_id,
                    t.vertical_connector.c.connector_type,
                    t.vertical_connector.c.connector_key,
                    t.vertical_connector.c.name,
                    t.vertical_connector_stop.c.level_id,
                    t.vertical_connector_stop.c.route_node_id,
                )
                .select_from(
                    t.vertical_connector.join(
                        t.vertical_connector_stop,
                        t.vertical_connector.c.connector_id
                        == t.vertical_connector_stop.c.connector_id,
                    ).join(
                        t.map_node,
                        t.map_node.c.node_id == t.vertical_connector_stop.c.route_node_id,
                    )
                )
                .where(
                    t.map_node.c.scan_id.in_(list(scan_ids)),
                    t.map_node.c.is_stale == sa.false(),
                    t.vertical_connector_stop.c.route_node_id.is_not(None),
                )
                .order_by(
                    t.vertical_connector.c.connector_id,
                    t.vertical_connector_stop.c.level_id,
                )
            )
        ).fetchall()

        by_connector: dict[str, list[Any]] = {}
        for row in rows:
            by_connector.setdefault(row.connector_id, []).append(row)

        edges: list[tuple[UUID, UUID, float, dict[str, object]]] = []
        for connector_rows in by_connector.values():
            for index, first in enumerate(connector_rows):
                for second in connector_rows[index + 1:]:
                    first_id = UUID(first.route_node_id)
                    second_id = UUID(second.route_node_id)
                    connector_type = str(first.connector_type)
                    cost = _vertical_transition_cost_m(connector_type)
                    edges.append(
                        (
                            first_id,
                            second_id,
                            cost,
                            {
                                "edge_kind": "vertical_connector",
                                "source": "vertical_connector_stop",
                                "connector_id": str(first.connector_id),
                                "connector_type": connector_type,
                                "connector_key": str(first.connector_key),
                                "name": first.name,
                                "from_level_id": first.level_id,
                                "to_level_id": second.level_id,
                            },
                        )
                    )
        return edges


def _level_id_from_source_ref(source_ref: dict[str, object] | None) -> str | None:
    if source_ref is None:
        return None
    value = source_ref.get("level_id") or source_ref.get("floor_id")
    if value is None:
        return None
    return str(value)


def _vertical_transition_cost_m(connector_type: str) -> float:
    value = connector_type.strip().lower()
    if value == "elevator":
        return 8.0
    if value in {"stair", "stairs", "escalator"}:
        return 12.0
    return 10.0
