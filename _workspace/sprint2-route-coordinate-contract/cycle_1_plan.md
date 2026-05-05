# Cycle 1 Plan: route coordinate external contract

## 판정

현재 답은 "아직 아니다"이다. 좌표 기반 라우팅 코어와 `scan_id` 기반 mobile wrapper는 이미 있지만, 외부 앱이 `building_id + floor_id + 출발좌표 + 목적지좌표`만 보내는 계약은 아직 없다.

## 현재 동작

- 내부 route API: `POST /route`는 `RouteRequest.scan_id`를 필수로 받고, 선택적으로 `scan_ids`, `merge_overlaps`, `start`, `goal`을 받는다. 근거: `src/indoor_server/interfaces/api/schemas.py:136`, `src/indoor_server/interfaces/api/router.py:354`, `src/indoor_server/interfaces/api/router.py:361`.
- mobile 좌표 API: `POST /mobile/v1/routes/coordinates`는 `start_coordinate`, `goal_coordinate`를 받지만 여전히 `scan_id`가 필수다. 핸들러는 이를 `RouteRequest`로 감싸 `post_route()`에 위임한다. 근거: `src/indoor_server/interfaces/api/schemas.py:146`, `src/indoor_server/interfaces/api/schemas.py:152`, `src/indoor_server/interfaces/api/router.py:482`, `src/indoor_server/interfaces/api/router.py:497`.
- 실제 라우팅: `RouteService.compute()`는 `scan_id`/`scan_ids`로 그래프를 로드하고, 좌표 endpoint는 nearest node snap 후 A*를 수행한다. 근거: `src/indoor_server/application/routing/route_service.py:43`, `src/indoor_server/application/routing/route_service.py:61`, `src/indoor_server/application/routing/route_service.py:95`, `src/indoor_server/application/routing/route_service.py:103`.
- V1 pathfinding API: `POST /api/v1/buildings/{buildingId}/pathfinding`은 `building_id`를 path parameter로 받지만 body는 `startFloorLevel + startX/Y/Z + destinationName` 구조다. 목적지는 좌표가 아니라 POI 이름으로 resolve한다. 근거: `src/indoor_server/interfaces/api/v1_router.py:411`, `src/indoor_server/interfaces/api/v1_schemas.py:126`, `src/indoor_server/application/api_v1/pathfinding_adapter.py:59`.
- floor active scan 조회는 이미 있다. `BuildingFloorService.get_active_scan_for_floor()`가 `floor_id`에서 active `scan_id`를 찾는다. 근거: `src/indoor_server/application/api_v1/building_floor_service.py:259`.

## Gap

외부 앱이 알아야 하는 식별자가 지금은 두 갈래로 나뉜다.

| 원하는 계약 | 현재 API | Gap |
|---|---|---|
| `building_id + floor_id + start coordinate + goal coordinate` | 없음 | scan/map 선택을 클라이언트가 몰라도 되는 endpoint 필요 |
| 좌표 → 좌표 길찾기 | `/mobile/v1/routes/coordinates` | `scan_id` 필수라 외부 앱 계약으로는 부적합 |
| building 기반 길찾기 | `/api/v1/buildings/{buildingId}/pathfinding` | 목적지가 `destinationName`이고 `floor_id`, goal coordinate가 없음 |
| floor 기반 route | `/api/v1/floors/{floorId}/route?from=&to=` | node id 기반이고 좌표 snap 계약이 없음 |

## Target External Contract

새 외부 계약은 V1 호환 surface에 둔다.

`POST /api/v1/buildings/{buildingId}/floors/{floorId}/routes/coordinates`

Request:

```json
{
  "start": {"x": 0.0, "y": 0.0, "z": 0.0},
  "goal": {"x": 10.0, "y": 0.0, "z": 0.0}
}
```

처리 규칙:

1. `floorId`를 조회하고 해당 floor의 `buildingId`가 path의 `buildingId`와 다르면 `404 FLOOR_NOT_FOUND` 또는 `422 FLOOR_BUILDING_MISMATCH` 중 하나로 명확히 실패시킨다. 추천은 `422 FLOOR_BUILDING_MISMATCH`.
2. `BuildingFloorService.get_active_scan_for_floor(floor_id)`로 active `scan_id`를 resolve한다.
3. active scan이 없으면 `404 ACTIVE_SCAN_NOT_FOUND`.
4. `RouteService.compute(scan_id=active.scan_id, scan_ids=None, merge_overlaps=False, start=coordinate, goal=coordinate)`로 위임한다.
5. response는 앱이 바로 그릴 수 있게 `pathGeometry`, `lengthM`, `nodeCount`, `snapInfo`, `routeMetadata`를 포함하는 V1 camelCase schema로 둔다. 내부 `RouteResponse`를 그대로 노출하지 말고 V1 schema를 별도로 둔다.

## Compatibility Rule

- 기존 `POST /route`는 내부/관리자/디버그용 canonical low-level API로 유지한다. `scan_id`, `scan_ids`, `node_id`, `poi_mark_id`, `merge_overlaps` 기능을 제거하거나 의미 변경하지 않는다.
- 기존 `POST /mobile/v1/routes/coordinates`는 backcompat로 유지한다. request/response shape와 `scan_id` 필수 조건을 바꾸지 않는다.
- 새 V1 endpoint만 `scan_id`를 숨긴 외부 앱 계약으로 삼는다. 내부적으로 active scan을 resolve하지만 request에는 `scan_id`/`scan_ids`를 허용하지 않는다.
- `scan_ids` multi-scan 경로는 기존 API에 남긴다. 이번 target은 단일 floor 좌표 경로이므로 floor active scan 1개만 사용한다. 다층 경로는 기존 `PathfindingAdapter` 계열의 별도 확장으로 분리한다.

## Implementation Plan

1. `src/indoor_server/interfaces/api/v1_schemas.py`에 좌표 request/response schema 추가
   - `CoordinatePoint(x, y, z=0.0)`
   - `FloorCoordinateRouteRequest(start, goal)`
   - `FloorCoordinateRouteResponse(buildingId, floorId, scanId, pathGeometry, lengthM, nodeCount, snapInfo, routeMetadata)`
2. `src/indoor_server/application/api_v1/pathfinding_adapter.py` 또는 새 `floor_coordinate_route_adapter.py`에 adapter 추가
   - floor 조회와 building mismatch 검증
   - active scan resolve
   - `RouteEndpoint(coordinate=(x, y, z))` 생성
   - `RouteService.compute()` 예외를 V1 error code로 매핑
3. `src/indoor_server/interfaces/api/v1_router.py`에 endpoint 추가
   - `POST /buildings/{buildingId}/floors/{floorId}/routes/coordinates`
   - `_V1_ERRORS` 형식 유지
4. OpenAPI smoke 기준 확인
   - `/v3/api-docs`에 새 path가 보이는지 확인
   - 기존 `/route`, `/mobile/v1/routes/coordinates`, `/api/v1/buildings/{buildingId}/pathfinding` path가 그대로 남는지 확인

## Acceptance Criteria

- 외부 앱은 `buildingId`, `floorId`, `start`, `goal`만 제공해 200 route 응답을 받는다.
- request body에 `scanId`, `scanIds`, `destinationName`, `nodeId`, `poiMarkId`가 필요 없다.
- floor가 building에 속하지 않으면 명확한 V1 error가 반환된다.
- active scan이 없으면 `ACTIVE_SCAN_NOT_FOUND`가 반환된다.
- build 미완료, snap 거리 초과, path 없음은 기존 route API와 같은 의미의 V1 error로 매핑된다.
- 기존 `scan_id` 기반 `/route`와 `/mobile/v1/routes/coordinates` 테스트는 수정 없이 통과한다.

## Focused Tests

추가할 테스트:

- `tests/unit/application/test_v1_compat_services.py`
  - adapter가 `buildingId + floorId`로 active scan을 찾고 `RouteService.compute(scan_id=active.scan_id, start=coordinate, goal=coordinate)`를 호출하는지 검증
  - floor/building mismatch 에러 검증
  - active scan 없음 에러 검증
- `tests/integration/api/test_v1_pathfinding_api.py` 또는 새 `test_v1_coordinate_route_api.py`
  - `POST /api/v1/buildings/{buildingId}/floors/{floorId}/routes/coordinates`가 V1 camelCase shape로 200을 반환하는지 검증
  - request에 `scanId` 없이 동작하는지 검증
- `tests/integration/api/test_route.py`
  - 기존 `test_mobile_coordinate_route_wraps_route_service`는 그대로 유지해 scan_id 기반 mobile wrapper backcompat를 지킨다.

실행할 명령:

```bash
uv run ruff check src/indoor_server/interfaces/api/v1_schemas.py src/indoor_server/interfaces/api/v1_router.py src/indoor_server/application/api_v1/pathfinding_adapter.py tests/unit/application/test_v1_compat_services.py tests/integration/api/test_v1_pathfinding_api.py tests/integration/api/test_route.py
uv run pytest tests/unit/application/test_v1_compat_services.py tests/integration/api/test_v1_pathfinding_api.py tests/integration/api/test_route.py -q
```
