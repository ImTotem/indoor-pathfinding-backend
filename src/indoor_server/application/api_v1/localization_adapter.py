"""Mockable VPS/localization adapter for V1 compatibility."""
from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.application.api_v1.building_floor_service import (
    ActiveScan,
    BuildingFloorService,
)
from indoor_server.application.api_v1.errors import V1ServiceError
from indoor_server.config import settings
from indoor_server.infrastructure.db import tables as t
from indoor_server.infrastructure.db.repositories.build_job_repo import BuildJobRepository
from indoor_server.interfaces.api.v1_schemas import (
    LocalizeResponse,
    NodeImagesRequest,
    NodeImagesResponse,
    SlamMetadataResponse,
    SlamStatusResponse,
)


@dataclass(frozen=True)
class LocalizeResult:
    map_id: str | None
    pose: dict[str, Any]
    confidence: float
    candidates: list[dict[str, Any]]


class VPSLocalizer(Protocol):
    async def localize(
        self,
        *,
        building_id: UUID,
        images: list[bytes],
        candidate_map_ids: list[str],
        shared_db_paths: list[str],
    ) -> LocalizeResult:
        """Return local metric pose for uploaded camera images."""


class MockVPSLocalizer:
    async def localize(
        self,
        *,
        building_id: UUID,
        images: list[bytes],
        candidate_map_ids: list[str],
        shared_db_paths: list[str],
    ) -> LocalizeResult:
        _ = (building_id, images, shared_db_paths)
        map_id = candidate_map_ids[0] if candidate_map_ids else None
        return LocalizeResult(
            map_id=map_id,
            pose={"x": 0.0, "y": 0.0, "z": 0.0},
            confidence=0.5 if map_id else 0.0,
            candidates=[{"mapId": value, "score": 0.5} for value in candidate_map_ids[:3]],
        )


class SLAMV3Localizer:
    async def localize(
        self,
        *,
        building_id: UUID,
        images: list[bytes],
        candidate_map_ids: list[str],
        shared_db_paths: list[str],
    ) -> LocalizeResult:
        _ = (candidate_map_ids, shared_db_paths)
        from models.slam_api import SLAMLocalizeRequest
        from routes import slam_routes

        encoded_images = [base64.b64encode(image).decode("ascii") for image in images]
        response = await slam_routes._localize_impl(
            SLAMLocalizeRequest(
                map_id=str(building_id),
                images=encoded_images,
                camera_intrinsics={},
            ),
            mask_persons=False,
        )
        map_id = response.floorId or response.mapId or str(building_id)
        return LocalizeResult(
            map_id=map_id,
            pose={
                **response.pose,
                "floorLevel": response.floorLevel,
            },
            confidence=response.confidence,
            candidates=[
                {
                    "mapId": map_id,
                    "score": response.confidence,
                    "numMatches": response.numMatches,
                    "matchedImageIndex": response.matchedImageIndex,
                }
            ],
        )


class LocalizationAdapter:
    def __init__(
        self,
        session: AsyncSession,
        localizer: VPSLocalizer | None = None,
    ) -> None:
        self._session = session
        self._localizer = localizer or _default_localizer()

    async def localize(self, *, building_id: UUID, images: list[bytes]) -> LocalizeResponse:
        scans = await BuildingFloorService(self._session).get_active_scans_for_building(
            building_id
        )
        if not scans:
            raise V1ServiceError(404, "ACTIVE_SCAN_NOT_FOUND", "building has no active scans")
        if not images:
            raise V1ServiceError(422, "IMAGES_REQUIRED", "at least one image is required")

        result = await self._localizer.localize(
            building_id=building_id,
            images=images,
            candidate_map_ids=[scan.scan_id for scan in scans],
            shared_db_paths=[
                str(settings.storage_root / "scans" / scan.scan_id / "rtabmap.db")
                for scan in scans
            ],
        )
        floor_level = _floor_for_map_id(result.map_id, scans)
        pose = dict(result.pose)
        if floor_level is not None:
            pose["floorLevel"] = floor_level
        return LocalizeResponse(
            building_id=building_id,
            map_id=result.map_id,
            pose=pose,
            confidence=result.confidence,
            candidates=result.candidates,
        )

    async def slam_status(self, building_id: UUID) -> SlamStatusResponse:
        scans = await BuildingFloorService(self._session).get_active_scans_for_building(
            building_id
        )
        items: list[dict[str, Any]] = []
        latest_status = "NOT_STARTED"
        for scan in scans:
            job = await BuildJobRepository(self._session).get_latest(scan.scan_id)
            status_value = job.state.value.upper() if job is not None else "NOT_STARTED"
            latest_status = status_value
            items.append(
                {
                    "scanId": scan.scan_id,
                    "floorId": scan.floor_id,
                    "floorLevel": scan.floor_level,
                    "status": status_value,
                    "buildJobId": str(job.build_job_id) if job is not None else None,
                }
            )
        return SlamStatusResponse(
            building_id=building_id,
            active_scan_count=len(scans),
            latest_status=latest_status,
            scans=items,
        )

    async def slam_metadata(self, building_id: UUID) -> SlamMetadataResponse:
        scans = await BuildingFloorService(self._session).get_active_scans_for_building(
            building_id
        )
        first = scans[0] if scans else None
        return SlamMetadataResponse(
            building_id=building_id,
            active_scan_id=UUID(first.scan_id) if first else None,
            keyframe_count=first.keyframe_count if first else 0,
            created_at=first.created_at if first is not None else None,
        )

    async def node_images(
        self,
        *,
        building_id: UUID,
        request: NodeImagesRequest,
    ) -> NodeImagesResponse:
        scans = await BuildingFloorService(self._session).get_active_scans_for_building(
            building_id
        )
        if not scans:
            return NodeImagesResponse(building_id=building_id, images=[])
        rows = (
            await self._session.execute(
                sa.select(
                    t.map_node.c.node_id,
                    t.map_node.c.scan_id,
                    t.map_node.c.label,
                    t.map_node.c.source_ref,
                    sa.func.ST_X(t.map_node.c.geom).label("x"),
                    sa.func.ST_Y(t.map_node.c.geom).label("y"),
                    sa.func.ST_Z(t.map_node.c.geom).label("z"),
                ).where(
                    t.map_node.c.scan_id.in_([scan.scan_id for scan in scans]),
                    t.map_node.c.is_stale == sa.false(),
                )
            )
        ).fetchall()
        if not rows:
            return NodeImagesResponse(building_id=building_id, images=[])
        row = _nearest_node(list(rows), request)
        scan_by_id = {scan.scan_id: scan for scan in scans}
        scan = scan_by_id.get(str(row.scan_id))
        return NodeImagesResponse(
            building_id=building_id,
            images=[
                {
                    "nodeId": str(row.node_id),
                    "scanId": str(row.scan_id),
                    "floorLevel": scan.floor_level if scan is not None else None,
                    "label": row.label,
                    "position": {"x": float(row.x), "y": float(row.y), "z": float(row.z)},
                    "imagePath": None,
                    "sourceRef": row.source_ref,
                }
            ],
        )


def _default_localizer() -> VPSLocalizer:
    if settings.vps_localizer_mode == "mock":
        return MockVPSLocalizer()
    if settings.vps_localizer_mode in {"slam_v3", "api_slam_v3"}:
        return SLAMV3Localizer()
    if settings.vps_localizer_mode != "mock":
        raise V1ServiceError(
            status_code=503,
            code="VPS_SERVICE_ERROR",
            message="Supported VPS localizer modes: mock, slam_v3.",
        )
    return MockVPSLocalizer()


def _floor_for_map_id(map_id: str | None, scans: list[ActiveScan]) -> int | None:
    if map_id is None:
        return None
    for scan in scans:
        if scan.scan_id == map_id:
            return scan.floor_level
    return None


def _nearest_node(rows: list[Any], request: NodeImagesRequest) -> Any:
    x = request.x
    y = request.y
    if x is None or y is None:
        return rows[0]
    return min(
        rows,
        key=lambda row: math.hypot(float(row.x) - x, float(row.y) - y),
    )
