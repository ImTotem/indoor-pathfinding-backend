"""Persist interfloor connector stops for route-time multi-floor edges."""
from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.infrastructure.db import tables as t


class VerticalConnectorCatalogService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def sync_scan_interfloor_marks(self, *, scan_id: str, build_job_id: str) -> int:
        _ = build_job_id
        floor = await self._floor_for_scan(scan_id)
        if floor is None:
            return 0

        old_stop_ids = (
            await self._session.execute(
                sa.select(t.vertical_connector_stop.c.connector_stop_id)
                .select_from(
                    t.vertical_connector_stop.join(
                        t.map_node,
                        t.vertical_connector_stop.c.route_node_id == t.map_node.c.node_id,
                    )
                )
                .where(t.map_node.c.scan_id == scan_id)
            )
        ).fetchall()
        if old_stop_ids:
            await self._session.execute(
                sa.delete(t.vertical_connector_stop).where(
                    t.vertical_connector_stop.c.connector_stop_id.in_(
                        [row.connector_stop_id for row in old_stop_ids]
                    )
                )
            )

        nodes = (
            await self._session.execute(
                sa.select(t.map_node.c.node_id, t.map_node.c.source_ref).where(
                    t.map_node.c.scan_id == scan_id,
                    t.map_node.c.is_stale == sa.false(),
                    t.map_node.c.source_ref.is_not(None),
                )
            )
        ).fetchall()

        synced = 0
        for node in nodes:
            source_ref = node.source_ref or {}
            if source_ref.get("role") != "vertical_connector_stop":
                continue
            connector_type = str(source_ref.get("connector_type") or "").lower()
            connector_key = str(source_ref.get("connector_key") or "").lower()
            if not connector_type or not connector_key:
                continue
            connector_id = await self._upsert_connector(
                building_id=str(floor.building_id),
                connector_type=connector_type,
                connector_key=connector_key,
            )
            await self._upsert_stop(
                connector_id=connector_id,
                level_id=f"floor:{floor.floor_id}",
                route_node_id=str(node.node_id),
            )
            synced += 1
        return synced

    async def _floor_for_scan(self, scan_id: str) -> Any | None:
        return (
            await self._session.execute(
                sa.select(t.building_floor.c.floor_id, t.building_floor.c.building_id)
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

    async def _upsert_connector(
        self,
        *,
        building_id: str,
        connector_type: str,
        connector_key: str,
    ) -> str:
        insert_stmt = pg_insert(t.vertical_connector).values(
            building_id=building_id,
            connector_type=connector_type,
            connector_key=connector_key,
            name=f"{connector_type.upper()} {connector_key}",
        )
        stmt = insert_stmt.on_conflict_do_update(
            index_elements=[
                t.vertical_connector.c.building_id,
                t.vertical_connector.c.connector_type,
                t.vertical_connector.c.connector_key,
            ],
            set_={"name": f"{connector_type.upper()} {connector_key}"},
        ).returning(t.vertical_connector.c.connector_id)
        return str((await self._session.execute(stmt)).scalar_one())

    async def _upsert_stop(
        self,
        *,
        connector_id: str,
        level_id: str,
        route_node_id: str,
    ) -> None:
        insert_stmt = pg_insert(t.vertical_connector_stop).values(
            connector_id=connector_id,
            level_id=level_id,
            route_node_id=route_node_id,
            poi_canonical_id=None,
        )
        stmt = insert_stmt.on_conflict_do_update(
            index_elements=[
                t.vertical_connector_stop.c.connector_id,
                t.vertical_connector_stop.c.level_id,
            ],
            set_={"route_node_id": route_node_id, "poi_canonical_id": None},
        )
        await self._session.execute(stmt)
