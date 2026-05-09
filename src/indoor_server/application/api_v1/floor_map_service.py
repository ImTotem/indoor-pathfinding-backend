"""V1 floor map service — floor 단위 2D 지도 (polygon + graph) 다운로드.

좌표는 모두 server world frame (meter). 측위 결과 / route polyline / map_node
와 동일 frame 이라 클라가 추가 transform 없이 같은 viewport 에 겹쳐 그릴 수 있음.
"""
from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.application.api_v1.building_floor_service import BuildingFloorService
from indoor_server.application.api_v1.errors import V1ServiceError
from indoor_server.config import settings
from indoor_server.domain.building.enums import BuildState
from indoor_server.domain.routing.models import MapEdgeRow, MapNodeRow
from indoor_server.infrastructure.db.repositories.build_job_repo import BuildJobRepository
from indoor_server.infrastructure.db.repositories.map_graph_repo import MapGraphRepository
from indoor_server.interfaces.api.v1_schemas import (
    FloorMapBounds,
    FloorMapConnector,
    FloorMapCoordinateSystem,
    FloorMapEdge,
    FloorMapNode,
    FloorMapResponse,
)

logger = logging.getLogger(__name__)

_BACKBONE_TYPES = {"corridor", "poi_attach", "junction", "endpoint"}


class FloorMapService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def load(self, floor_id: UUID) -> FloorMapResponse:
        svc = BuildingFloorService(self._session)
        active = await svc.get_active_scan_for_floor(floor_id)
        if active is None:
            raise V1ServiceError(
                404,
                "ACTIVE_SCAN_NOT_FOUND",
                "floor has no active scan.",
                {"floorId": str(floor_id)},
            )

        build_job = await BuildJobRepository(self._session).get_latest(active.scan_id)
        if build_job is None or build_job.state != BuildState.SUCCEEDED:
            state_str = build_job.state.value if build_job else "not_started"
            raise V1ServiceError(
                422,
                "GRAPH_NOT_READY",
                "build has not succeeded yet.",
                {"state": state_str},
            )

        polygon = self._load_polygon(str(build_job.build_job_id))
        nodes_rows, edges_rows, _ = await MapGraphRepository(
            self._session
        ).load_graph_for_routing(active.scan_id)

        nodes_out = [_to_node(n) for n in nodes_rows]
        node_type_by_id: dict[UUID, str] = {n.node_id: n.node_type for n in nodes_rows}
        edges_out = [_to_edge(e, node_type_by_id) for e in edges_rows]

        bounds = _compute_bounds(nodes_rows, polygon)

        return FloorMapResponse(
            floor_id=UUID(active.floor_id),
            building_id=UUID(active.building_id),
            scan_id=UUID(active.scan_id),
            floor_level=active.floor_level,
            floor_name=active.floor_name,
            build_job_id=build_job.build_job_id,
            coordinate_system=FloorMapCoordinateSystem(),
            bounds=bounds,
            polygon=polygon,
            nodes=nodes_out,
            edges=edges_out,
            etag=str(build_job.build_job_id),
        )

    @staticmethod
    def _load_polygon(build_job_id: str) -> dict[str, Any]:
        path = (
            settings.storage_root
            / "builds"
            / build_job_id
            / "polygon_v2"
            / "floor_polygon.geojson"
        )
        if not path.exists():
            logger.info(
                "floor_polygon.geojson missing for build=%s — empty FC", build_job_id
            )
            return {"type": "FeatureCollection", "features": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
                logger.warning(
                    "floor_polygon.geojson is not a FeatureCollection (build=%s) — wrapping",
                    build_job_id,
                )
                return {"type": "FeatureCollection", "features": []}
            return data
        except Exception as exc:
            logger.warning(
                "failed to read floor_polygon.geojson build=%s: %s", build_job_id, exc
            )
            return {"type": "FeatureCollection", "features": []}


# ── helpers ──────────────────────────────────────────────────────────────────


def _to_node(n: MapNodeRow) -> FloorMapNode:
    connector: FloorMapConnector | None = None
    src = n.source_ref or {}
    ctype = src.get("connector_type")
    if isinstance(ctype, str) and ctype:
        connector = FloorMapConnector(
            type=str(ctype).lower(),
            key=str(src.get("connector_key")) if src.get("connector_key") else None,
        )

    return FloorMapNode(
        id=n.node_id,
        type=n.node_type,
        x=n.x,
        y=n.y,
        z=n.z,
        label=n.label,
        connector=connector,
    )


def _to_edge(e: MapEdgeRow, node_type_by_id: dict[UUID, str]) -> FloorMapEdge:
    a = node_type_by_id.get(e.from_node_id, "")
    b = node_type_by_id.get(e.to_node_id, "")
    if a in _BACKBONE_TYPES and b in _BACKBONE_TYPES:
        edge_type = "corridor"
    else:
        edge_type = "spur"
    return FloorMapEdge(
        id=e.edge_id,
        from_id=e.from_node_id,
        to_id=e.to_node_id,
        length_m=float(e.length_m),
        type=edge_type,
    )


def _compute_bounds(
    nodes: list[MapNodeRow], polygon: dict[str, Any]
) -> FloorMapBounds:
    xs: list[float] = [n.x for n in nodes]
    ys: list[float] = [n.y for n in nodes]

    for feat in polygon.get("features", []) or []:
        geom = feat.get("geometry") or {}
        for x, y in _iter_coords(geom):
            xs.append(x)
            ys.append(y)

    if not xs or not ys:
        return FloorMapBounds(
            min_x=0.0, min_y=0.0, max_x=0.0, max_y=0.0, width_m=0.0, height_m=0.0
        )

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return FloorMapBounds(
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        width_m=max_x - min_x,
        height_m=max_y - min_y,
    )


def _iter_coords(geom: dict[str, Any]):
    """GeoJSON geometry → flat (x, y) iterator."""
    t = geom.get("type")
    coords = geom.get("coordinates")
    if coords is None:
        return
    if t == "Point":
        x, y = coords[0], coords[1]
        yield float(x), float(y)
    elif t == "LineString" or t == "MultiPoint":
        for p in coords:
            yield float(p[0]), float(p[1])
    elif t == "Polygon" or t == "MultiLineString":
        for ring in coords:
            for p in ring:
                yield float(p[0]), float(p[1])
    elif t == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                for p in ring:
                    yield float(p[0]), float(p[1])
