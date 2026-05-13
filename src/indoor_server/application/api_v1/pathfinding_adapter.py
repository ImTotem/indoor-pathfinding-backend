"""V1 pathfinding adapter over the existing RouteService."""
from __future__ import annotations

import logging
import math
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.application.api_v1.building_floor_service import (
    ActiveScan,
    BuildingFloorService,
)
from indoor_server.application.api_v1.errors import V1ServiceError
from indoor_server.application.api_v1.poi_catalog_service import POICatalogService
from indoor_server.application.routing.graph_loader import GraphLoader
from indoor_server.application.routing.route_service import RouteService
from indoor_server.domain.building.enums import BuildState
from indoor_server.domain.routing.errors import (
    EndpointNodeNotFoundError,
    GraphNotReadyError,
    PathNotFoundError,
    POINodeNotFoundError,
    ScanNotFoundError,
    SnapDistanceExceededError,
)
from indoor_server.domain.routing.models import MapNodeRow
from indoor_server.infrastructure.db.repositories.build_job_repo import BuildJobRepository

_log = logging.getLogger(__name__)
from indoor_server.interfaces.api.schemas import RouteEndpoint
from indoor_server.interfaces.api.v1_schemas import (
    CoordinatePoint,
    FloorCoordinateRouteRequest,
    FloorCoordinateRouteResponse,
    FloorTransitionResponse,
    PathfindingRequest,
    PathfindingResponse,
    PathStepResponse,
    V1Position,
    V1RouteSnapInfo,
)


class PathfindingAdapter:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def compute(
        self,
        *,
        building_id: UUID,
        request: PathfindingRequest,
    ) -> PathfindingResponse:
        building_service = BuildingFloorService(self._session)
        scans = await building_service.get_active_scans_for_building(building_id)
        if not scans:
            raise V1ServiceError(404, "ACTIVE_SCAN_NOT_FOUND", "building has no active scans")

        # 시작 scan 자동 결정. start_floor_level override 없으면 좌표 z 기반.
        if request.start_floor_level is not None:
            start_scan = _scan_for_level(scans, request.start_floor_level)
            if start_scan is None:
                raise V1ServiceError(
                    404,
                    "START_FLOOR_NOT_FOUND",
                    "startFloorLevel has no active scan.",
                    {"startFloorLevel": request.start_floor_level},
                )
        else:
            start_scan = _scan_for_z(scans, request.start_z)

        goal = await POICatalogService(self._session).destination_endpoint(
            building_id=building_id,
            destination_name=request.destination_name,
        )
        start = RouteEndpoint(
            coordinate=(request.start_x, request.start_y, request.start_z),
        )

        # Multi-scan A* requires every fragment to have a SUCCEEDED build,
        # otherwise GraphNotReady poisons the whole compute. Filter unbuilt
        # active scans (e.g. floors whose only active scan is OPEN with no
        # build_job yet) so cross-floor routing still works for floors that
        # *are* built. The user-selected start_scan is kept regardless — if
        # its graph isn't ready we surface that as the error, which is right.
        built_scan_ids = await self._filter_built_scan_ids(scans, start_scan.scan_id)

        route_service = RouteService(GraphLoader(self._session))
        try:
            route = await route_service.compute(
                scan_id=start_scan.scan_id,
                scan_ids=built_scan_ids,
                merge_overlaps=False,
                start=start,
                goal=goal,
                vertical_preference=request.vertical_preference,
            )
        except (
            ScanNotFoundError,
            GraphNotReadyError,
            SnapDistanceExceededError,
            EndpointNodeNotFoundError,
            POINodeNotFoundError,
            PathNotFoundError,
        ) as e:
            raise _route_error_to_v1(e) from e

        level_by_scan_id = {scan.scan_id: scan.floor_level for scan in scans}
        steps = [
            _step_response(index=index, node=node, level_by_scan_id=level_by_scan_id)
            for index, node in enumerate(route.nodes_in_order, start=1)
        ]
        metadata = dict(route.route_metadata or {})
        metadata["vertical_preference"] = request.vertical_preference
        metadata["start_scan_id"] = start_scan.scan_id
        metadata["start_floor_level"] = start_scan.floor_level
        if request.preference != "SHORTEST":
            metadata["preference_ignored"] = True
            metadata["requested_preference"] = request.preference
        return PathfindingResponse(
            building_id=building_id,
            total_distance=route.length_m,
            estimated_time_seconds=math.ceil(route.length_m / 1.2),
            steps=steps,
            floor_transitions=_floor_transitions(route.nodes_in_order, level_by_scan_id),
            route_metadata=metadata,
        )

    async def _filter_built_scan_ids(
        self,
        scans: list[ActiveScan],
        start_scan_id: str,
    ) -> list[str]:
        """active scans 중 latest build = SUCCEEDED 인 것만 반환.

        start_scan_id 는 무조건 포함 (built 가 아니면 GraphLoader 가
        GraphNotReadyError 를 던져 사용자 입장에서 명확한 에러로 표면화).
        """
        repo = BuildJobRepository(self._session)
        built: list[str] = []
        skipped: list[tuple[str, int, str | None]] = []
        for scan in scans:
            latest = await repo.get_latest(scan.scan_id)
            ok = latest is not None and latest.state == BuildState.SUCCEEDED
            if ok:
                built.append(scan.scan_id)
            else:
                skipped.append(
                    (scan.scan_id, scan.floor_level, latest.state.value if latest else None)
                )
        if start_scan_id not in built:
            built.insert(0, start_scan_id)
        if skipped:
            _log.info(
                "pathfinding: %d active scan(s) skipped (no SUCCEEDED build): %s",
                len(skipped),
                [(sid[:12], lvl, st) for sid, lvl, st in skipped],
            )
        return built

    async def compute_floor_coordinate_route(
        self,
        *,
        building_id: UUID,
        floor_id: UUID,
        request: FloorCoordinateRouteRequest,
    ) -> FloorCoordinateRouteResponse:
        building_service = BuildingFloorService(self._session)
        floor = await building_service.get_floor(floor_id)
        if floor.building_id != building_id:
            raise V1ServiceError(
                422,
                "FLOOR_BUILDING_MISMATCH",
                "floorId does not belong to buildingId.",
                {"buildingId": str(building_id), "floorId": str(floor_id)},
            )
        active = await building_service.get_active_scan_for_floor(floor_id)
        if active is None:
            raise V1ServiceError(
                404,
                "ACTIVE_SCAN_NOT_FOUND",
                "floor has no active scan.",
                {"floorId": str(floor_id)},
            )

        route_service = RouteService(GraphLoader(self._session))
        try:
            route = await route_service.compute(
                scan_id=active.scan_id,
                scan_ids=None,
                merge_overlaps=False,
                start=_coordinate_endpoint(request.start),
                goal=_coordinate_endpoint(request.goal),
            )
        except (
            ScanNotFoundError,
            GraphNotReadyError,
            SnapDistanceExceededError,
            EndpointNodeNotFoundError,
            POINodeNotFoundError,
            PathNotFoundError,
        ) as e:
            raise _route_error_to_v1(e) from e

        return FloorCoordinateRouteResponse(
            building_id=building_id,
            floor_id=floor_id,
            scan_id=UUID(active.scan_id),
            path_geometry={
                "type": "LineString",
                "coordinates": [[x, y, z] for x, y, z in route.polyline],
            },
            length_m=route.length_m,
            node_count=len(route.nodes_in_order),
            snap_info=V1RouteSnapInfo(
                start_snap_distance_m=route.snap.start_snap_distance_m,
                goal_snap_distance_m=route.snap.goal_snap_distance_m,
            ),
            route_metadata=dict(route.route_metadata or {}),
        )


def _scan_for_level(scans: list[ActiveScan], floor_level: int) -> ActiveScan | None:
    for scan in scans:
        if scan.floor_level == floor_level:
            return scan
    return None


def _scan_for_z(scans: list[ActiveScan], z: float, floor_height_m: float = 3.0) -> ActiveScan:
    """좌표 z → 가장 가까운 floor 의 active scan.

    각 floor 의 expected z = floor_level * floor_height_m (1층 기준 z=0).
    실제 ARKit world frame 별 origin 차이가 있어 정확하지 않을 수 있으나,
    snap 단계가 multi-scan node 후보 중 nearest 를 다시 고르므로 시작 scan 잘못
    잡혀도 cross-floor edge 로 도달 가능. tie 면 floor_level=1 (지상층) 우선.
    """
    def key(scan: ActiveScan) -> tuple[float, int]:
        expected_z = (scan.floor_level - 1) * floor_height_m
        return (abs(z - expected_z), abs(scan.floor_level - 1))
    return min(scans, key=key)


def _coordinate_endpoint(point: CoordinatePoint) -> RouteEndpoint:
    return RouteEndpoint(coordinate=(point.x, point.y, point.z))


def _route_error_to_v1(
    error: (
        ScanNotFoundError
        | GraphNotReadyError
        | SnapDistanceExceededError
        | EndpointNodeNotFoundError
        | POINodeNotFoundError
        | PathNotFoundError
    ),
) -> V1ServiceError:
    if isinstance(error, ScanNotFoundError):
        return V1ServiceError(404, "SCAN_NOT_FOUND", "scan_id does not exist.")
    if isinstance(error, GraphNotReadyError):
        return V1ServiceError(
            422,
            "GRAPH_NOT_READY",
            "build has not succeeded yet.",
            {"state": error.state, "buildJobId": error.build_job_id},
        )
    if isinstance(error, SnapDistanceExceededError):
        return V1ServiceError(
            422,
            "SNAP_DISTANCE_EXCEEDED",
            f"{error.side} coordinate too far from any map node.",
            {"distanceM": error.distance_m, "thresholdM": error.threshold_m},
        )
    if isinstance(error, EndpointNodeNotFoundError):
        return V1ServiceError(
            422,
            "ENDPOINT_NODE_NOT_FOUND",
            f"{error.side} node_id not found in graph.",
            {"nodeId": error.node_id},
        )
    if isinstance(error, POINodeNotFoundError):
        return V1ServiceError(
            422,
            "POI_NODE_NOT_FOUND",
            f"{error.side} poi_mark_id has no projected map_node.",
            {"poiMarkId": error.poi_mark_id},
        )
    return V1ServiceError(
        422,
        "PATH_NOT_FOUND",
        "no path between start and goal.",
        {"startNodeId": error.start_node_id, "goalNodeId": error.goal_node_id},
    )


def _step_response(
    *,
    index: int,
    node: MapNodeRow,
    level_by_scan_id: dict[str, int],
) -> PathStepResponse:
    floor_level = _floor_level_for_node(node, level_by_scan_id)
    if node.node_type == "virtual_snap":
        instruction = "Start."
    else:
        label = node.label or node.node_type
        instruction = f"Proceed to {label}."
    return PathStepResponse(
        step_number=index,
        floor_level=floor_level,
        position=V1Position(x=node.x, y=node.y, z=node.z, floor_level=floor_level),
        instruction=instruction,
        node_id=node.node_id,
    )


def _floor_level_for_node(
    node: MapNodeRow,
    level_by_scan_id: dict[str, int],
) -> int | None:
    if node.level_id is not None and node.level_id.startswith("floor:"):
        # The actual floor level is resolved by scan_id below when possible.
        pass
    if node.scan_id is not None:
        return level_by_scan_id.get(str(node.scan_id))
    return None


def _floor_transitions(
    nodes: list[MapNodeRow],
    level_by_scan_id: dict[str, int],
) -> list[FloorTransitionResponse]:
    transitions: list[FloorTransitionResponse] = []
    for prev, current in zip(nodes, nodes[1:], strict=False):
        prev_level = _floor_level_for_node(prev, level_by_scan_id)
        current_level = _floor_level_for_node(current, level_by_scan_id)
        if prev_level is None or current_level is None or prev_level == current_level:
            continue
        source_ref = current.source_ref or prev.source_ref or {}
        transitions.append(
            FloorTransitionResponse(
                from_floor_level=prev_level,
                to_floor_level=current_level,
                connector_type=(
                    str(source_ref.get("connector_type")).upper()
                    if source_ref.get("connector_type")
                    else None
                ),
                connector_key=(
                    str(source_ref.get("connector_key"))
                    if source_ref.get("connector_key")
                    else None
                ),
            )
        )
    return transitions
