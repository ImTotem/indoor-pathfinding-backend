## Status

done

## Summary

`buildingId + floorId + start/goal 좌표`만으로 V1 route를 요청할 수 있게 추가했고, 기존 `scan_id` 기반 `/route`, `/mobile/v1/routes/coordinates` 계약은 유지했다.

## Detail

- `src/indoor_server/interfaces/api/v1_schemas.py`
  - `CoordinatePoint`, `FloorCoordinateRouteRequest`, `V1RouteSnapInfo`, `FloorCoordinateRouteResponse`를 추가했다.
  - 새 request는 `start`, `goal` 좌표만 받으며 `scanId` 같은 extra field는 허용하지 않도록 `extra="forbid"`로 제한했다.
  - response는 V1 camelCase 기준으로 `buildingId`, `floorId`, `scanId`, `pathGeometry`, `lengthM`, `nodeCount`, `snapInfo`, `routeMetadata`를 반환한다.
- `src/indoor_server/application/api_v1/pathfinding_adapter.py`
  - `compute_floor_coordinate_route()`를 추가했다.
  - `floorId`가 `buildingId`에 속하는지 검증하고, `BuildingFloorService.get_active_scan_for_floor()`로 active `scan_id`를 내부 resolve한다.
  - route 실행은 기존 `RouteService.compute(scan_id=active.scan_id, scan_ids=None, merge_overlaps=False, start=coordinate, goal=coordinate)`에 위임한다.
  - 기존 pathfinding route exception mapping을 `_route_error_to_v1()` helper로 정리해 새 좌표 route와 공유하게 했다.
- `src/indoor_server/interfaces/api/v1_router.py`
  - `POST /api/v1/buildings/{buildingId}/floors/{floorId}/routes/coordinates` endpoint를 추가했다.
- `tests/unit/application/test_v1_compat_services.py`
  - floor coordinate route가 active scan을 내부 resolve하고 `RouteService.compute()`에 `scan_ids=None`, `merge_overlaps=False`로 위임하는지 검증했다.
  - floor/building mismatch와 active scan 없음 에러를 검증했다.
- `tests/integration/api/test_v1_pathfinding_api.py`
  - 새 endpoint가 request body에 `scanId` 없이 V1 camelCase shape를 반환하는지 검증했다.
  - OpenAPI path 노출 smoke test를 추가했다.
- `tests/integration/api/test_swagger_docs.py`
  - `/v3/api-docs` alias가 새 V1 coordinate route path를 포함하는지 검증 대상에 추가했다.

## Evidence

- Ruff
  - Command: `uv run ruff check src/indoor_server/interfaces/api/v1_schemas.py src/indoor_server/interfaces/api/v1_router.py src/indoor_server/application/api_v1/pathfinding_adapter.py tests/unit/application/test_v1_compat_services.py tests/integration/api/test_v1_pathfinding_api.py tests/integration/api/test_route.py tests/integration/api/test_swagger_docs.py`
  - Log: `_workspace/sprint2-route-coordinate-contract/cycle_1_ruff.log`
  - Result: `All checks passed!`
- Pytest
  - Command: `uv run pytest tests/unit/application/test_v1_compat_services.py tests/integration/api/test_v1_pathfinding_api.py tests/integration/api/test_route.py tests/integration/api/test_swagger_docs.py -q`
  - Log: `_workspace/sprint2-route-coordinate-contract/cycle_1_pytest.log`
  - Result: `28 passed, 4 warnings in 0.71s`
  - Note: warnings are FastAPI `HTTP_422_UNPROCESSABLE_ENTITY` deprecation warnings from existing route error tests.

## Next

- 다층 좌표 route가 필요해지면 이번 endpoint를 확장하지 말고 기존 `PathfindingAdapter`의 multi-scan/vertical transition 계열 계약으로 별도 설계하는 편이 안전하다.
