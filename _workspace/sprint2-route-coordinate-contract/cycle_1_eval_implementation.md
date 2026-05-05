## Status

done

## Summary

PASS: 새 V1 좌표 route 계약은 `buildingId + floorId + start/goal`만으로 동작하도록 구현됐고, 기존 `scan_id` 기반 `/route`와 `/mobile/v1/routes/coordinates` 계약도 유지된다.

## Detail

### Critical (0개)

- 없음

### High (0개)

- 없음

### Medium (0개)

- 없음

### Low (0개)

- 없음

## Evidence

- 설계 요구사항 매핑
  - `src/indoor_server/interfaces/api/v1_schemas.py:141`: `FloorCoordinateRouteRequest`는 `start`, `goal`만 갖고 `extra="forbid"`로 root-level `scanId`/`scanIds` 추가 입력을 거부한다.
  - `src/indoor_server/interfaces/api/v1_schemas.py:153`: `FloorCoordinateRouteResponse`는 V1 camelCase 응답 필드 `buildingId`, `floorId`, `scanId`, `pathGeometry`, `lengthM`, `nodeCount`, `snapInfo`, `routeMetadata`를 제공한다.
  - `src/indoor_server/application/api_v1/pathfinding_adapter.py:107`: `compute_floor_coordinate_route()`가 floor 조회, building/floor mismatch 검증, active scan resolve, `RouteService.compute()` 위임을 수행한다.
  - `src/indoor_server/application/api_v1/pathfinding_adapter.py:116`: floor가 path의 building에 속하지 않으면 `422 FLOOR_BUILDING_MISMATCH`로 실패한다.
  - `src/indoor_server/application/api_v1/pathfinding_adapter.py:123`: active scan이 없으면 `404 ACTIVE_SCAN_NOT_FOUND`로 실패한다.
  - `src/indoor_server/application/api_v1/pathfinding_adapter.py:134`: route 위임은 `scan_id=active.scan_id`, `scan_ids=None`, `merge_overlaps=False`, 좌표 start/goal로 수행된다.
  - `src/indoor_server/interfaces/api/v1_router.py:432`: `POST /api/v1/buildings/{buildingId}/floors/{floorId}/routes/coordinates` endpoint가 등록됐다.
  - `src/indoor_server/interfaces/api/router.py:354`: 기존 `POST /route`는 `scan_id`, `scan_ids`, `merge_overlaps`, endpoint type 계약을 유지한다.
  - `src/indoor_server/interfaces/api/router.py:482`: 기존 `POST /mobile/v1/routes/coordinates`는 여전히 `CoordinateRouteRequest`를 받아 `RouteRequest`로 감싸 기존 `/route` 처리에 위임한다.
- 테스트 근거
  - `tests/unit/application/test_v1_compat_services.py:156`: active scan resolve 후 `RouteService.compute()` 위임 파라미터를 검증한다.
  - `tests/unit/application/test_v1_compat_services.py:224`: floor/building mismatch 에러를 검증한다.
  - `tests/unit/application/test_v1_compat_services.py:263`: active scan 없음 에러를 검증한다.
  - `tests/integration/api/test_v1_pathfinding_api.py:80`: 새 API가 body에 `scanId` 없이 V1 camelCase 응답을 반환하는지 검증한다.
  - `tests/integration/api/test_v1_pathfinding_api.py:125`: OpenAPI에 새 path가 노출되는지 검증한다.
  - `tests/integration/api/test_route.py:245`: 기존 `/route` multi-scan 계약 보존을 검증한다.
  - `tests/integration/api/test_route.py:278`: 기존 `/mobile/v1/routes/coordinates`가 `scan_id`, `scan_ids`, `merge_overlaps`를 유지하는지 검증한다.
- 실행 결과
  - `git status --short`: `fatal: not a git repository (or any of the parent directories): .git`
    - 변경 파일 목록은 git diff가 아니라 build report의 파일 목록 기준으로 추론했다.
  - `uv run ruff check src/indoor_server/interfaces/api/v1_schemas.py src/indoor_server/interfaces/api/v1_router.py src/indoor_server/application/api_v1/pathfinding_adapter.py tests/unit/application/test_v1_compat_services.py tests/integration/api/test_v1_pathfinding_api.py tests/integration/api/test_route.py tests/integration/api/test_swagger_docs.py`
    - 결과: `All checks passed!`
  - `uv run pytest tests/unit/application/test_v1_compat_services.py tests/integration/api/test_v1_pathfinding_api.py tests/integration/api/test_route.py tests/integration/api/test_swagger_docs.py -q`
    - 결과: `28 passed, 4 warnings in 0.75s`
    - 경고: 기존 route error 테스트에서 FastAPI `HTTP_422_UNPROCESSABLE_ENTITY` deprecation warning 4건. 이번 구현의 실패 증거는 아니다.
  - `uv run python - <<'PY' ... FloorCoordinateRouteRequest.model_validate(..., scanId=...) ... PY`
    - 결과: `extra_forbidden`
    - 의미: 새 request schema가 `scanId` root field를 거부한다.

## Next

- rollback 필요 없음.
