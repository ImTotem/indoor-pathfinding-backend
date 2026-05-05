"""POI catalog and mockable enrichment service for Sprint 84."""
from __future__ import annotations

import math
from typing import Any, Protocol
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.application.api_v1.building_floor_service import (
    BuildingFloorService,
    point_to_dict,
)
from indoor_server.application.api_v1.errors import V1ServiceError
from indoor_server.application.semantic.mock_analyzer import MockSemanticAnalyzer
from indoor_server.domain.semantic.models import SemanticAnalysis
from indoor_server.infrastructure.db import tables as t
from indoor_server.interfaces.api.schemas import RouteEndpoint
from indoor_server.interfaces.api.v1_schemas import POICreateRequest, POIResponse


class POIAnalyzer(Protocol):
    async def analyze(
        self,
        *,
        label: str | None,
        class_name: str | None,
        image_bytes: bytes | None,
    ) -> SemanticAnalysis:
        """Analyze one POI label/photo without making route code depend on a real LLM."""


class MockPOIAnalyzer:
    name = "mock"

    def __init__(self) -> None:
        self._inner = MockSemanticAnalyzer()

    async def analyze(
        self,
        *,
        label: str | None,
        class_name: str | None,
        image_bytes: bytes | None,
    ) -> SemanticAnalysis:
        _ = image_bytes
        return self._inner.analyze(label=label, class_name=class_name, source="poi_photo")


class POICatalogService:
    def __init__(
        self,
        session: AsyncSession,
        analyzer: POIAnalyzer | None = None,
    ) -> None:
        self._session = session
        self._analyzer = analyzer or MockPOIAnalyzer()

    async def list_pois(self, building_id: UUID) -> list[POIResponse]:
        rows = await self._poi_rows(building_id=building_id, query=None)
        return [_poi_response(row) for row in rows]

    async def search_pois(self, building_id: UUID, query: str | None) -> list[POIResponse]:
        rows = await self._poi_rows(building_id=building_id, query=query)
        return [_poi_response(row) for row in rows]

    async def create_poi(
        self,
        building_id: UUID,
        request: POICreateRequest,
    ) -> POIResponse:
        floor_id = await self._resolve_floor_id(building_id, request)
        route_node_id, display = await self._nearest_route_node(
            floor_id=floor_id,
            x=request.x,
            y=request.y,
            z=request.z,
        )
        display_wkt = (
            _point_wkt(display["x"], display["y"], display["z"]) if display is not None else None
        )
        row = (
            await self._session.execute(
                sa.insert(t.poi_canonical)
                .values(
                    building_id=str(building_id),
                    floor_id=str(floor_id),
                    scan_id=None,
                    level_id=f"floor:{floor_id}",
                    category=request.category,
                    name=request.name,
                    label=request.name,
                    route_node_id=str(route_node_id) if route_node_id is not None else None,
                    display_point=display_wkt,
                    source_mark_ids=[],
                    needs_review=False,
                    llm_confidence=None,
                )
                .returning(
                    t.poi_canonical.c.canonical_id,
                    t.poi_canonical.c.building_id,
                    t.poi_canonical.c.floor_id,
                    t.poi_canonical.c.name,
                    t.poi_canonical.c.label,
                    t.poi_canonical.c.category,
                    t.poi_canonical.c.route_node_id,
                    sa.func.ST_AsGeoJSON(t.poi_canonical.c.display_point).label(
                        "display_point_json"
                    ),
                    t.poi_canonical.c.needs_review,
                    t.poi_canonical.c.llm_confidence,
                )
            )
        ).first()
        await self._session.commit()
        assert row is not None
        return _poi_response(row)

    async def sync_scan_pois(self, *, scan_id: str, build_job_id: str) -> int:
        _ = build_job_id
        floor_row = await self._floor_for_scan(scan_id)
        if floor_row is None:
            return 0

        await self._session.execute(
            sa.delete(t.poi_canonical).where(t.poi_canonical.c.scan_id == scan_id)
        )

        node_rows = (
            await self._session.execute(
                sa.select(
                    t.map_node.c.node_id,
                    t.map_node.c.poi_mark_id,
                    t.map_node.c.label,
                    t.map_node.c.source_ref,
                    sa.func.ST_X(t.map_node.c.geom).label("x"),
                    sa.func.ST_Y(t.map_node.c.geom).label("y"),
                    sa.func.ST_Z(t.map_node.c.geom).label("z"),
                    sa.func.ST_AsGeoJSON(t.map_node.c.geom).label("display_point_json"),
                ).where(
                    t.map_node.c.scan_id == scan_id,
                    t.map_node.c.is_stale == sa.false(),
                    t.map_node.c.poi_mark_id.is_not(None),
                )
            )
        ).fetchall()

        synced = 0
        for node in node_rows:
            poi_mark_id = int(node.poi_mark_id)
            if poi_mark_id < 0:
                continue
            mark = await self._poi_mark(poi_mark_id)
            if mark is None:
                continue
            photo = await self._latest_photo(poi_mark_id)
            class_name = str(photo.class_name) if photo is not None else None
            image_blob = bytes(photo.image_blob) if photo is not None and photo.image_blob else None
            analysis = await self._analyzer.analyze(
                label=mark.label,
                class_name=class_name,
                image_bytes=image_blob,
            )
            canonical_id = (
                await self._session.execute(
                    sa.insert(t.poi_canonical)
                    .values(
                        building_id=str(floor_row.building_id),
                        floor_id=str(floor_row.floor_id),
                        scan_id=scan_id,
                        level_id=f"floor:{floor_row.floor_id}",
                        category=analysis.category,
                        name=analysis.name or mark.label or node.label,
                        label=mark.label or node.label,
                        route_node_id=str(node.node_id),
                        display_point=_point_wkt(float(node.x), float(node.y), float(node.z)),
                        source_mark_ids=[poi_mark_id],
                        needs_review=analysis.needs_review,
                        llm_confidence=analysis.confidence,
                    )
                    .returning(t.poi_canonical.c.canonical_id)
                )
            ).scalar_one()
            await self._session.execute(
                sa.update(t.poi_mark)
                .where(t.poi_mark.c.id == poi_mark_id)
                .values(canonical_id=canonical_id)
            )
            await self._session.execute(
                sa.insert(t.poi_analysis_job).values(
                    poi_mark_id=poi_mark_id,
                    analyzer=analysis.analyzer,
                    state="succeeded",
                    result_json=analysis.as_metadata(),
                )
            )
            await self._session.execute(
                sa.insert(t.place_label_candidate).values(
                    canonical_id=canonical_id,
                    source=analysis.analyzer,
                    label=analysis.name or mark.label,
                    category=analysis.category,
                    confidence=analysis.confidence,
                    raw_json=analysis.as_metadata(),
                )
            )
            synced += 1
        return synced

    async def destination_endpoint(
        self,
        *,
        building_id: UUID,
        destination_name: str,
    ) -> RouteEndpoint:
        pattern = f"%{destination_name.lower()}%"
        active_scan_ids = await self._active_scan_ids(building_id)
        canonical = (
            await self._session.execute(
                sa.select(t.poi_canonical.c.route_node_id, t.poi_canonical.c.source_mark_ids)
                .select_from(
                    t.poi_canonical.outerjoin(
                        t.map_node,
                        t.poi_canonical.c.route_node_id == t.map_node.c.node_id,
                    )
                )
                .where(
                    t.poi_canonical.c.building_id == str(building_id),
                    self._active_poi_scope(active_scan_ids),
                    sa.or_(
                        sa.func.lower(t.poi_canonical.c.name).like(pattern),
                        sa.func.lower(t.poi_canonical.c.label).like(pattern),
                    ),
                )
                .order_by(t.poi_canonical.c.needs_review.asc())
                .limit(1)
            )
        ).first()
        if canonical is not None:
            if canonical.route_node_id is not None:
                return RouteEndpoint(node_id=UUID(str(canonical.route_node_id)))
            source_mark_ids = canonical.source_mark_ids or []
            if source_mark_ids:
                return RouteEndpoint(poi_mark_id=int(source_mark_ids[0]))

        if active_scan_ids:
            node = (
                await self._session.execute(
                    sa.select(t.map_node.c.node_id)
                    .where(
                        t.map_node.c.scan_id.in_(active_scan_ids),
                        t.map_node.c.is_stale == sa.false(),
                        sa.func.lower(t.map_node.c.label).like(pattern),
                    )
                    .limit(1)
                )
            ).first()
            if node is not None:
                return RouteEndpoint(node_id=UUID(str(node.node_id)))

            mark = (
                await self._session.execute(
                    sa.select(t.poi_mark.c.id)
                    .where(
                        t.poi_mark.c.scan_id.in_(active_scan_ids),
                        sa.func.lower(t.poi_mark.c.label).like(pattern),
                    )
                    .limit(1)
                )
            ).first()
            if mark is not None:
                return RouteEndpoint(poi_mark_id=int(mark.id))

        raise V1ServiceError(
            status_code=404,
            code="DESTINATION_NOT_FOUND",
            message="destinationName did not match any POI or route node.",
            detail={"destinationName": destination_name},
        )

    async def _poi_rows(self, *, building_id: UUID, query: str | None) -> list[Any]:
        active_scan_ids = await self._active_scan_ids(building_id)
        filters = [
            t.poi_canonical.c.building_id == str(building_id),
            self._active_poi_scope(active_scan_ids),
        ]
        if query:
            pattern = f"%{query.lower()}%"
            filters.append(
                sa.or_(
                    sa.func.lower(t.poi_canonical.c.name).like(pattern),
                    sa.func.lower(t.poi_canonical.c.label).like(pattern),
                    sa.func.lower(t.poi_canonical.c.category).like(pattern),
                )
            )
        rows = (
            await self._session.execute(
                sa.select(
                    t.poi_canonical.c.canonical_id,
                    t.poi_canonical.c.building_id,
                    t.poi_canonical.c.floor_id,
                    t.poi_canonical.c.name,
                    t.poi_canonical.c.label,
                    t.poi_canonical.c.category,
                    t.poi_canonical.c.route_node_id,
                    sa.func.ST_AsGeoJSON(t.poi_canonical.c.display_point).label(
                        "display_point_json"
                    ),
                    t.poi_canonical.c.needs_review,
                    t.poi_canonical.c.llm_confidence,
                )
                .select_from(
                    t.poi_canonical.outerjoin(
                        t.map_node,
                        t.poi_canonical.c.route_node_id == t.map_node.c.node_id,
                    )
                )
                .where(*filters)
                .order_by(t.poi_canonical.c.name.asc().nullslast())
            )
        ).fetchall()
        return list(rows)

    async def _resolve_floor_id(self, building_id: UUID, request: POICreateRequest) -> UUID:
        if request.floor_id is not None:
            row = (
                await self._session.execute(
                    sa.select(t.building_floor.c.floor_id).where(
                        t.building_floor.c.floor_id == str(request.floor_id),
                        t.building_floor.c.building_id == str(building_id),
                    )
                )
            ).first()
            if row is None:
                raise V1ServiceError(
                    404,
                    "FLOOR_NOT_FOUND",
                    "floor not found for building",
                    {"floorId": str(request.floor_id), "buildingId": str(building_id)},
                )
            return request.floor_id
        floor_filters = [t.building_floor.c.building_id == str(building_id)]
        if request.floor_level is not None:
            floor_filters.append(t.building_floor.c.level == request.floor_level)
        row = (
            await self._session.execute(
                sa.select(t.building_floor.c.floor_id)
                .where(*floor_filters)
                .order_by(t.building_floor.c.level.asc())
                .limit(1)
            )
        ).first()
        if row is None:
            raise V1ServiceError(404, "FLOOR_NOT_FOUND", "floor not found")
        return UUID(str(row.floor_id))

    async def _nearest_route_node(
        self,
        *,
        floor_id: UUID,
        x: float | None,
        y: float | None,
        z: float | None,
    ) -> tuple[UUID | None, dict[str, float] | None]:
        active = await BuildingFloorService(self._session).get_active_scan_for_floor(floor_id)
        if active is None:
            return None, _display_point_from_request(x, y, z)
        rows = (
            await self._session.execute(
                sa.select(
                    t.map_node.c.node_id,
                    sa.func.ST_X(t.map_node.c.geom).label("x"),
                    sa.func.ST_Y(t.map_node.c.geom).label("y"),
                    sa.func.ST_Z(t.map_node.c.geom).label("z"),
                ).where(
                    t.map_node.c.scan_id == active.scan_id,
                    t.map_node.c.is_stale == sa.false(),
                )
            )
        ).fetchall()
        if not rows:
            return None, _display_point_from_request(x, y, z)
        target = _display_point_from_request(x, y, z)
        if target is None:
            first = rows[0]
            return UUID(str(first.node_id)), {
                "x": float(first.x),
                "y": float(first.y),
                "z": float(first.z),
            }
        best = min(
            rows,
            key=lambda row: math.hypot(float(row.x) - target["x"], float(row.y) - target["y"]),
        )
        return UUID(str(best.node_id)), {
            "x": float(best.x),
            "y": float(best.y),
            "z": float(best.z),
        }

    async def _floor_for_scan(self, scan_id: str) -> Any | None:
        return (
            await self._session.execute(
                sa.select(
                    t.building_floor.c.floor_id,
                    t.building_floor.c.building_id,
                    t.building_floor.c.level,
                )
                .select_from(
                    t.floor_scan.join(
                        t.building_floor,
                        t.floor_scan.c.floor_id == t.building_floor.c.floor_id,
                    )
                )
                .where(t.floor_scan.c.scan_id == scan_id)
                .order_by(t.floor_scan.c.active.desc(), t.floor_scan.c.created_at.desc())
                .limit(1)
            )
        ).first()

    async def _poi_mark(self, poi_mark_id: int) -> Any | None:
        return (
            await self._session.execute(
                sa.select(t.poi_mark).where(t.poi_mark.c.id == poi_mark_id)
            )
        ).first()

    async def _latest_photo(self, poi_mark_id: int) -> Any | None:
        return (
            await self._session.execute(
                sa.select(t.poi_photo)
                .where(t.poi_photo.c.poi_mark_id == poi_mark_id)
                .order_by(t.poi_photo.c.captured_at.desc())
                .limit(1)
            )
        ).first()

    async def _active_scan_ids(self, building_id: UUID) -> list[str]:
        active_scans = await BuildingFloorService(self._session).get_active_scans_for_building(
            building_id
        )
        return [scan.scan_id for scan in active_scans]

    def _active_poi_scope(self, active_scan_ids: list[str]) -> Any:
        active_scan_source = (
            t.poi_canonical.c.scan_id.in_(active_scan_ids) if active_scan_ids else sa.false()
        )
        active_route_node = (
            sa.and_(
                t.map_node.c.scan_id.in_(active_scan_ids),
                t.map_node.c.is_stale == sa.false(),
            )
            if active_scan_ids
            else sa.false()
        )
        return sa.and_(
            sa.or_(t.poi_canonical.c.scan_id.is_(None), active_scan_source),
            sa.or_(t.poi_canonical.c.route_node_id.is_(None), active_route_node),
        )


def _poi_response(row: Any) -> POIResponse:
    return POIResponse(
        poi_id=UUID(str(row.canonical_id)),
        building_id=UUID(str(row.building_id)) if row.building_id else None,
        floor_id=UUID(str(row.floor_id)) if row.floor_id else None,
        name=str(row.name) if row.name is not None else None,
        label=str(row.label) if row.label is not None else None,
        category=str(row.category),
        route_node_id=UUID(str(row.route_node_id)) if row.route_node_id else None,
        display_point=point_to_dict(row.display_point_json),
        needs_review=bool(row.needs_review),
        llm_confidence=float(row.llm_confidence) if row.llm_confidence is not None else None,
    )


def _display_point_from_request(
    x: float | None,
    y: float | None,
    z: float | None,
) -> dict[str, float] | None:
    if x is None or y is None:
        return None
    return {"x": x, "y": y, "z": z or 0.0}


def _point_wkt(x: float, y: float, z: float) -> str:
    return f"SRID=0;POINTZ({x} {y} {z})"
