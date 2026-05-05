"""Building/floor compatibility service for `/api/v1/*`."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.application.api_v1.errors import V1ServiceError
from indoor_server.config import settings
from indoor_server.domain.building.enums import BuildState
from indoor_server.infrastructure.db import tables as t
from indoor_server.infrastructure.db.repositories.build_job_repo import BuildJobRepository
from indoor_server.infrastructure.db.repositories.map_graph_repo import MapGraphRepository
from indoor_server.interfaces.api.v1_schemas import (
    BuildingCreateRequest,
    BuildingDetailResponse,
    BuildingResponse,
    BuildingStatusRequest,
    BuildingUpdateRequest,
    FloorCreateRequest,
    FloorPathResponse,
    FloorResponse,
    FloorUpdateRequest,
    PassageSegment,
    VerticalPassageResponse,
)


@dataclass(frozen=True)
class ActiveScan:
    scan_id: str
    floor_id: str
    building_id: str
    floor_level: int
    floor_name: str
    storage_path: str | None = None
    keyframe_count: int = 0
    created_at: object | None = None


class BuildingFloorService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_buildings(self, *, status_filter: str | None = None) -> list[BuildingResponse]:
        stmt = sa.select(t.building).order_by(t.building.c.created_at.desc())
        if status_filter is not None:
            stmt = stmt.where(t.building.c.status == status_filter)
        rows = (await self._session.execute(stmt)).fetchall()
        return [_building_response(row) for row in rows]

    async def create_building(self, request: BuildingCreateRequest) -> BuildingResponse:
        from indoor_server.config import settings

        result = await self._session.execute(
            sa.insert(t.building)
            .values(
                name=request.name,
                description=request.description,
                latitude=request.latitude,
                longitude=request.longitude,
            )
            .returning(t.building)
        )
        row = result.first()
        assert row is not None
        building_id = str(row.building_id)

        # Sprint 78 A-5: mock passage seed (env gate)
        if settings.enable_mock_passage:
            await self._seed_initial_floors_and_passage(building_id)

        await self._session.commit()
        return _building_response(row)

    async def _seed_initial_floors_and_passage(self, building_id: str) -> None:
        """Sprint 78: building 생성 시 1F/2F floor + elevator mock passage 자동 seed.

        prod 환경에서는 INDOOR_SERVER_ENABLE_MOCK_PASSAGE=false 로 비활성화.
        is_mock=true row는 응답에 mock 플래그로 노출해 prod 데이터와 구분 가능.
        """
        import logging

        from sqlalchemy.dialects.postgresql import insert as pg_insert

        _log = logging.getLogger(__name__)

        # 1F, 2F floor 자동 생성
        for floor_num, floor_name in [(1, "1F"), (2, "2F")]:
            existing = (
                await self._session.execute(
                    sa.select(t.building_floor.c.floor_id)
                    .where(
                        t.building_floor.c.building_id == building_id,
                        t.building_floor.c.level == floor_num,
                    )
                )
            ).first()
            if existing is None:
                await self._session.execute(
                    sa.insert(t.building_floor).values(
                        building_id=building_id,
                        name=floor_name,
                        level=floor_num,
                    )
                )

        # elevator mock vertical_connector (is_mock=true)
        connector_type = "elevator"
        connector_key = "ev-1"
        insert_stmt = pg_insert(t.vertical_connector).values(
            building_id=building_id,
            connector_type=connector_type,
            connector_key=connector_key,
            name="ELEVATOR EV-1 (mock)",
            is_mock=True,
        )
        stmt = insert_stmt.on_conflict_do_nothing(
            index_elements=[
                t.vertical_connector.c.building_id,
                t.vertical_connector.c.connector_type,
                t.vertical_connector.c.connector_key,
            ],
        )
        await self._session.execute(stmt)
        _log.info("mock passage seed: building_id=%s elevator ev-1 seeded", building_id)

    async def get_building(self, building_id: UUID) -> BuildingDetailResponse:
        row = await self._get_building_row(building_id)
        floors = await self.list_floors(building_id)
        passages = await self.list_passages(building_id)
        base = _building_response(row)
        return BuildingDetailResponse(
            **base.model_dump(),
            floors=floors,
            vertical_passages=passages,
        )

    async def update_building(
        self,
        building_id: UUID,
        request: BuildingUpdateRequest,
    ) -> BuildingResponse:
        values = request.model_dump(exclude_unset=True)
        if not values:
            return _building_response(await self._get_building_row(building_id))
        values["updated_at"] = sa.func.now()
        row = (
            await self._session.execute(
                sa.update(t.building)
                .where(t.building.c.building_id == str(building_id))
                .values(**values)
                .returning(t.building)
            )
        ).first()
        if row is None:
            raise _not_found("BUILDING_NOT_FOUND", "building not found")
        await self._session.commit()
        return _building_response(row)

    async def delete_building(self, building_id: UUID) -> None:
        result = await self._session.execute(
            sa.delete(t.building).where(t.building.c.building_id == str(building_id))
        )
        if int(getattr(result, "rowcount", 0)) == 0:
            raise _not_found("BUILDING_NOT_FOUND", "building not found")
        await self._session.commit()

    async def patch_status(
        self,
        building_id: UUID,
        request: BuildingStatusRequest,
    ) -> BuildingResponse:
        row = (
            await self._session.execute(
                sa.update(t.building)
                .where(t.building.c.building_id == str(building_id))
                .values(status=request.status, updated_at=sa.func.now())
                .returning(t.building)
            )
        ).first()
        if row is None:
            raise _not_found("BUILDING_NOT_FOUND", "building not found")
        await self._session.commit()
        return _building_response(row)

    async def list_floors(self, building_id: UUID) -> list[FloorResponse]:
        await self._ensure_building(building_id)
        rows = (
            await self._session.execute(
                sa.select(t.building_floor)
                .where(t.building_floor.c.building_id == str(building_id))
                .order_by(t.building_floor.c.level.asc())
            )
        ).fetchall()
        return [await self._floor_response(row) for row in rows]

    async def create_floor(
        self,
        building_id: UUID,
        request: FloorCreateRequest,
    ) -> FloorResponse:
        await self._ensure_building(building_id)
        try:
            row = (
                await self._session.execute(
                    sa.insert(t.building_floor)
                    .values(
                        building_id=str(building_id),
                        name=request.name,
                        level=request.level,
                        height=request.height,
                    )
                    .returning(t.building_floor)
                )
            ).first()
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise
        assert row is not None
        return await self._floor_response(row)

    async def get_floor(self, floor_id: UUID) -> FloorResponse:
        return await self._floor_response(await self._get_floor_row(floor_id))

    async def update_floor(self, floor_id: UUID, request: FloorUpdateRequest) -> FloorResponse:
        values = request.model_dump(exclude_unset=True)
        if not values:
            return await self.get_floor(floor_id)
        values["updated_at"] = sa.func.now()
        row = (
            await self._session.execute(
                sa.update(t.building_floor)
                .where(t.building_floor.c.floor_id == str(floor_id))
                .values(**values)
                .returning(t.building_floor)
            )
        ).first()
        if row is None:
            raise _not_found("FLOOR_NOT_FOUND", "floor not found")
        await self._session.commit()
        return await self._floor_response(row)

    async def delete_floor(self, floor_id: UUID) -> None:
        result = await self._session.execute(
            sa.delete(t.building_floor).where(t.building_floor.c.floor_id == str(floor_id))
        )
        if int(getattr(result, "rowcount", 0)) == 0:
            raise _not_found("FLOOR_NOT_FOUND", "floor not found")
        await self._session.commit()

    async def get_active_scan_for_floor(self, floor_id: UUID) -> ActiveScan | None:
        row = (
            await self._session.execute(
                sa.select(
                    t.floor_scan.c.scan_id,
                    t.building_floor.c.floor_id,
                    t.building_floor.c.building_id,
                    t.building_floor.c.level,
                    t.building_floor.c.name,
                    t.scan_ingest.c.storage_path,
                    t.scan_session.c.keyframe_count,
                    t.scan_ingest.c.ingested_at,
                )
                .select_from(
                    t.floor_scan.join(
                        t.building_floor,
                        t.floor_scan.c.floor_id == t.building_floor.c.floor_id,
                    )
                    .join(t.scan_ingest, t.floor_scan.c.scan_id == t.scan_ingest.c.scan_id)
                    .outerjoin(t.scan_session, t.scan_ingest.c.scan_id == t.scan_session.c.scan_id)
                )
                .where(
                    t.floor_scan.c.floor_id == str(floor_id),
                    t.floor_scan.c.active == sa.true(),
                )
                .order_by(t.floor_scan.c.created_at.desc())
                .limit(1)
            )
        ).first()
        if row is None:
            return None
        return ActiveScan(
            scan_id=str(row.scan_id),
            floor_id=str(row.floor_id),
            building_id=str(row.building_id),
            floor_level=int(row.level),
            floor_name=str(row.name),
            storage_path=str(row.storage_path) if row.storage_path else None,
            keyframe_count=int(row.keyframe_count or 0),
            created_at=row.ingested_at,
        )

    async def get_active_scans_for_building(self, building_id: UUID) -> list[ActiveScan]:
        rows = (
            await self._session.execute(
                sa.select(
                    t.floor_scan.c.scan_id,
                    t.building_floor.c.floor_id,
                    t.building_floor.c.building_id,
                    t.building_floor.c.level,
                    t.building_floor.c.name,
                    t.scan_ingest.c.storage_path,
                    t.scan_session.c.keyframe_count,
                    t.scan_ingest.c.ingested_at,
                )
                .select_from(
                    t.floor_scan.join(
                        t.building_floor,
                        t.floor_scan.c.floor_id == t.building_floor.c.floor_id,
                    )
                    .join(t.scan_ingest, t.floor_scan.c.scan_id == t.scan_ingest.c.scan_id)
                    .outerjoin(t.scan_session, t.scan_ingest.c.scan_id == t.scan_session.c.scan_id)
                )
                .where(
                    t.building_floor.c.building_id == str(building_id),
                    t.floor_scan.c.active == sa.true(),
                )
                .order_by(t.building_floor.c.level.asc())
            )
        ).fetchall()
        return [
            ActiveScan(
                scan_id=str(row.scan_id),
                floor_id=str(row.floor_id),
                building_id=str(row.building_id),
                floor_level=int(row.level),
                floor_name=str(row.name),
                storage_path=str(row.storage_path) if row.storage_path else None,
                keyframe_count=int(row.keyframe_count or 0),
                created_at=row.ingested_at,
            )
            for row in rows
        ]

    async def get_floor_path(self, floor_id: UUID) -> FloorPathResponse:
        floor = await self._get_floor_row(floor_id)
        active = await self.get_active_scan_for_floor(floor_id)
        if active is None:
            return FloorPathResponse(floor_id=UUID(floor.floor_id), nodes=[], edges=[])
        build_job = await BuildJobRepository(self._session).get_latest(active.scan_id)
        if build_job is None or build_job.state != BuildState.SUCCEEDED:
            return FloorPathResponse(
                floor_id=UUID(floor.floor_id),
                scan_id=UUID(active.scan_id),
                nodes=[],
                edges=[],
            )
        features = await MapGraphRepository(self._session).load_graph_geojson(active.scan_id)
        raw_nodes = [feature for feature in features if _feature_geometry_type(feature) == "Point"]
        raw_edges = [
            feature for feature in features
            if _feature_geometry_type(feature) == "LineString"
        ]

        # A-3: kind 필드 추가 (node_type → kind 매핑)
        nodes = [_enrich_node_kind(n) for n in raw_nodes]
        edges = [_enrich_edge_kind(e) for e in raw_edges]

        return FloorPathResponse(
            floor_id=UUID(floor.floor_id),
            scan_id=UUID(active.scan_id),
            build_job_id=build_job.build_job_id,
            nodes=nodes,
            edges=edges,
            bounds=_bounds_from_node_features(nodes),
        )

    async def pointcloud_path(self, floor_id: UUID) -> Path:
        active = await self.get_active_scan_for_floor(floor_id)
        if active is None:
            raise _not_found("POINTCLOUD_NOT_FOUND", "active scan not found")
        scan_root = settings.storage_root / "scans" / active.scan_id
        for candidate in [
            scan_root / "pointcloud.ply",
            scan_root / "floor_points.ply",
            scan_root / "world_points.npz",
        ]:
            if candidate.exists():
                return candidate
        debug_candidates = list(settings.debug_root.glob(f"{active.scan_id}/**/world_points.npz"))
        if debug_candidates:
            return debug_candidates[0]
        raise _not_found("POINTCLOUD_NOT_FOUND", "pointcloud artifact not found")

    async def list_passages(self, building_id: UUID) -> list[VerticalPassageResponse]:
        await self._ensure_building(building_id)
        rows = (
            await self._session.execute(
                sa.select(t.vertical_connector)
                .where(t.vertical_connector.c.building_id == str(building_id))
                .order_by(
                    t.vertical_connector.c.connector_type,
                    t.vertical_connector.c.connector_key,
                )
            )
        ).fetchall()
        return [await self._passage_response(row) for row in rows]

    async def get_passage(self, passage_id: UUID) -> VerticalPassageResponse:
        row = (
            await self._session.execute(
                sa.select(t.vertical_connector).where(
                    t.vertical_connector.c.connector_id == str(passage_id)
                )
            )
        ).first()
        if row is None:
            raise _not_found("PASSAGE_NOT_FOUND", "passage not found")
        return await self._passage_response(row)

    async def _get_building_row(self, building_id: UUID) -> Any:
        row = (
            await self._session.execute(
                sa.select(t.building).where(t.building.c.building_id == str(building_id))
            )
        ).first()
        if row is None:
            raise _not_found("BUILDING_NOT_FOUND", "building not found")
        return row

    async def _ensure_building(self, building_id: UUID) -> None:
        await self._get_building_row(building_id)

    async def _get_floor_row(self, floor_id: UUID) -> Any:
        row = (
            await self._session.execute(
                sa.select(t.building_floor).where(t.building_floor.c.floor_id == str(floor_id))
            )
        ).first()
        if row is None:
            raise _not_found("FLOOR_NOT_FOUND", "floor not found")
        return row

    async def _floor_response(self, row: Any) -> FloorResponse:
        floor_id = UUID(str(row.floor_id))
        active = await self.get_active_scan_for_floor(floor_id)
        has_path = False
        active_scan_uuid: UUID | None = None
        has_ply = False
        if active is not None:
            active_scan_uuid = UUID(active.scan_id)
            latest = await BuildJobRepository(self._session).get_latest(active.scan_id)
            has_path = latest is not None and latest.state == BuildState.SUCCEEDED
            scan_root = settings.storage_root / "scans" / active.scan_id
            has_ply = any(
                (scan_root / name).exists()
                for name in ("pointcloud.ply", "floor_points.ply", "world_points.npz")
            )
        return FloorResponse(
            floor_id=floor_id,
            building_id=UUID(str(row.building_id)),
            name=str(row.name),
            level=int(row.level),
            height=float(row.height) if row.height is not None else None,
            has_path=has_path,
            has_ply=has_ply,
            active_scan_id=active_scan_uuid,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    async def _passage_response(self, row: Any) -> VerticalPassageResponse:
        stop_rows = (
            await self._session.execute(
                sa.select(
                    t.vertical_connector_stop.c.connector_stop_id,
                    t.vertical_connector_stop.c.level_id,
                    t.vertical_connector_stop.c.route_node_id,
                )
                .where(t.vertical_connector_stop.c.connector_id == row.connector_id)
                .order_by(t.vertical_connector_stop.c.level_id)
            )
        ).fetchall()
        segments: list[PassageSegment] = []
        for stop in stop_rows:
            # route_node 에서 좌표 조회 (map_nodes 테이블)
            seg_x: float | None = None
            seg_y: float | None = None
            if stop.route_node_id:
                node_row = (
                    await self._session.execute(
                        sa.select(
                            sa.func.ST_X(t.map_node.c.geom).label("x"),
                            sa.func.ST_Y(t.map_node.c.geom).label("y"),
                        ).where(
                            t.map_node.c.node_id == str(stop.route_node_id)
                        )
                    )
                ).first()
                if node_row is not None:
                    seg_x = float(node_row.x) if node_row.x is not None else None
                    seg_y = float(node_row.y) if node_row.y is not None else None

            segments.append(
                PassageSegment(
                    stop_id=str(stop.connector_stop_id),
                    level_id=str(stop.level_id),
                    route_node_id=str(stop.route_node_id) if stop.route_node_id else None,
                    x=seg_x,
                    y=seg_y,
                    kind=str(row.connector_type).upper(),
                )
            )
        is_mock = bool(getattr(row, "is_mock", False))
        return VerticalPassageResponse(
            passage_id=UUID(str(row.connector_id)),
            building_id=UUID(str(row.building_id)) if row.building_id else None,
            connector_type=str(row.connector_type).upper(),
            connector_key=str(row.connector_key),
            name=str(row.name) if row.name else None,
            mock=is_mock,
            segments=segments,
        )


def _building_response(row: Any) -> BuildingResponse:
    return BuildingResponse(
        building_id=UUID(str(row.building_id)),
        name=str(row.name),
        description=str(row.description) if row.description is not None else None,
        latitude=float(row.latitude) if row.latitude is not None else None,
        longitude=float(row.longitude) if row.longitude is not None else None,
        status=row.status,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _not_found(code: str, message: str) -> V1ServiceError:
    return V1ServiceError(status_code=404, code=code, message=message)


# Sprint 78 A-3: node_type → kind 매핑 헬퍼

_NODE_TYPE_TO_KIND: dict[str, str] = {
    "CORRIDOR": "corridor",
    "POI": "poi",
    "POI_ATTACH": "corridor",  # attach point는 corridor처럼 취급
    "PASSAGE": "passage",
    # vertical_connector_stop role의 노드는 source_ref로 판단
}

_EDGE_TYPE_TO_KIND: dict[str, str] = {
    "SKELETON": "corridor",
    "POI_SPUR": "poi_spur",
    "PASSAGE_SPUR": "passage_spur",
    "CONNECTOR": "passage_spur",
}


def _enrich_node_kind(feature: dict[str, Any]) -> dict[str, Any]:
    """node feature에 kind / label / passage_kind 필드를 추가한다."""
    import copy
    feat = copy.deepcopy(feature)
    props = feat.get("properties") or {}

    node_type_raw = str(props.get("node_type", "")).upper()
    source_ref = props.get("source_ref") or {}

    # vertical_connector_stop role → passage
    if isinstance(source_ref, dict) and source_ref.get("role") == "vertical_connector_stop":
        kind = "passage"
        connector_type = str(source_ref.get("connector_type", "")).lower()
        _passage_kinds = ("elevator", "stairs", "escalator", "staircase")
        passage_kind = connector_type if connector_type in _passage_kinds else None
        props["passage_kind"] = passage_kind
    else:
        kind = _NODE_TYPE_TO_KIND.get(node_type_raw, "corridor")
        props["passage_kind"] = None

    props["kind"] = kind
    # H-2: label 키가 없을 경우 None으로 보장 (안전망)
    props["label"] = props.get("label")
    feat["properties"] = props
    return feat


def _enrich_edge_kind(feature: dict[str, Any]) -> dict[str, Any]:
    """edge feature에 kind 필드를 추가한다."""
    import copy
    feat = copy.deepcopy(feature)
    props = feat.get("properties") or {}

    edge_type_raw = str(props.get("edge_type", "")).upper()
    props["kind"] = _EDGE_TYPE_TO_KIND.get(edge_type_raw, "corridor")
    feat["properties"] = props
    return feat


def _bounds_from_node_features(nodes: list[dict[str, Any]]) -> dict[str, float] | None:
    points: list[tuple[float, float]] = []
    for node in nodes:
        coords = node.get("geometry", {}).get("coordinates")
        if not isinstance(coords, list) or len(coords) < 2:
            continue
        try:
            points.append((float(coords[0]), float(coords[1])))
        except (TypeError, ValueError):
            continue
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    diagonal = math.hypot(max_x - min_x, max_y - min_y)
    return {
        "minX": min_x,
        "minY": min_y,
        "maxX": max_x,
        "maxY": max_y,
        "diagonalM": diagonal,
    }


def _feature_geometry_type(feature: dict[str, Any]) -> str | None:
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        return None
    value = geometry.get("type")
    return str(value) if value is not None else None


def point_to_dict(geom_json: str | None) -> dict[str, float] | None:
    if geom_json is None:
        return None
    geom = json.loads(geom_json)
    coords = geom.get("coordinates")
    if not isinstance(coords, list) or len(coords) < 2:
        return None
    return {
        "x": float(coords[0]),
        "y": float(coords[1]),
        "z": float(coords[2]) if len(coords) > 2 else 0.0,
    }
