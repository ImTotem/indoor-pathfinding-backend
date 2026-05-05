## Status

done

## Summary

PASS: FastAPI 앱을 임시 포트 `18080`에서 실행해 새 building/floor coordinate route의 OpenAPI 노출, 실제 200 응답, validation/error 경계를 HTTP로 검증했다.

## Detail

| Scenario | Result | Evidence |
|---|---:|---|
| App startup on separate port | PASS | `evidence/cycle_1_eval_server.log` |
| `GET /v3/api-docs` exposes target path | PASS | `evidence/openapi.json`, `evidence/openapi_summary.json` |
| Existing compatibility paths remain exposed (`/route`, `/mobile/v1/routes/coordinates`, `/api/v1/buildings/{buildingId}/pathfinding`) | PASS | `evidence/openapi_summary.json` |
| Request schema requires only `start`, `goal` and forbids extra fields | PASS | `evidence/openapi_summary.json` |
| Runtime readiness (`/healthz`, `/readyz`) | PASS | `evidence/healthz.json`, `evidence/readyz.json` |
| Read-only discovery of an existing routable floor path | PASS | `evidence/buildings.json`, `evidence/existing_floor_candidates.json`, `evidence/floor_path.json` |
| `POST /api/v1/buildings/{buildingId}/floors/{floorId}/routes/coordinates` succeeds without request `scanId` | PASS | `evidence/request_success_from_existing_path.json`, `evidence/success_route.json` |
| Floor/building mismatch returns V1 error | PASS | `evidence/building_mismatch.json` |
| Matching floor with no active scan returns `ACTIVE_SCAN_NOT_FOUND` | PASS | `evidence/active_scan_missing.json` |
| Unknown valid floor UUID returns `FLOOR_NOT_FOUND` | PASS | `evidence/valid_missing_floor.json` |
| Forbidden body `scanId` returns `VALIDATION_ERROR` / `extra_forbidden` | PASS | `evidence/request_extra_scanid.json`, `evidence/extra_scanid.json` |
| Missing `start`/`goal` and invalid UUID path return V1 validation envelope | PASS | `evidence/missing_fields.json`, `evidence/invalid_uuid.json` |

Coverage: 12/12 executed scenarios passed. The happy-path route response returned `200`, `buildingId`, `floorId`, internally resolved `scanId`, `pathGeometry.type=LineString`, `lengthM=33.281499999999994`, `nodeCount=17`, and `snapInfo`, while the request body contained only `start` and `goal`.

Verdict: PASS

Origin: none

Recommended rollback: none

## Evidence

- `_workspace/sprint2-route-coordinate-contract/evidence/http_summary.json`
- `_workspace/sprint2-route-coordinate-contract/evidence/openapi_summary.json`
- `_workspace/sprint2-route-coordinate-contract/evidence/success_route.json`
- `_workspace/sprint2-route-coordinate-contract/evidence/building_mismatch.json`
- `_workspace/sprint2-route-coordinate-contract/evidence/active_scan_missing.json`
- `_workspace/sprint2-route-coordinate-contract/evidence/valid_missing_floor.json`
- `_workspace/sprint2-route-coordinate-contract/evidence/extra_scanid.json`
- `_workspace/sprint2-route-coordinate-contract/evidence/missing_fields.json`
- `_workspace/sprint2-route-coordinate-contract/evidence/invalid_uuid.json`
- `_workspace/sprint2-route-coordinate-contract/evidence/cycle_1_eval_server.log`
