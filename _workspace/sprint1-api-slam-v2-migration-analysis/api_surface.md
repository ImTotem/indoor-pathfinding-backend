# API Surface After Migration

## 결론

current repo의 서버 표면은 v2 API 전체 + legacy `/api/slam/*` 호환 API다. SLAM 위치추정 계열은 base64 JSON이 아니라 `multipart/form-data` 이미지 파일 업로드를 외부 계약으로 사용한다.

## 노출 유지

- `/api/slam/v3/localize`
- `/api/slam/process`
- `/api/slam/status/{building_id}`
- `/api/slam/health`
- `/api/slam/maps/{building_id}/metadata`
- `/api/slam/localize`
- `/api/slam/v2/localize`
- `/api/slam/v1/debug/matches`
- `/api/slam/v2/debug/mask`
- `/api/slam/v2/debug/matches`
- `/api/slam/v3/debug/matches`
- `/api/v1/buildings`
- `/api/v1/buildings/{buildingId}`
- `/api/v1/buildings/{buildingId}/floors`
- `/api/v1/buildings/{buildingId}/localize`
- `/api/v1/buildings/{buildingId}/node-images`
- `/api/v1/buildings/{buildingId}/passages`
- `/api/v1/buildings/{buildingId}/pathfinding`
- `/api/v1/buildings/{buildingId}/pois`
- `/api/v1/buildings/{buildingId}/pois/search`
- `/api/v1/buildings/{buildingId}/slam/metadata`
- `/api/v1/buildings/{buildingId}/slam/status`
- `/api/v1/buildings/{buildingId}/status`
- `/api/v1/floors/{floorId}`
- `/api/v1/floors/{floorId}/path`
- `/api/v1/floors/{floorId}/pointcloud`
- `/api/v1/floors/{floorId}/process`
- `/api/v1/floors/{floorId}/process/status`
- `/api/v1/floors/{floorId}/route`
- `/api/v1/floors/{floorId}/scans/chunks`
- `/api/v1/floors/{floorId}/scans/chunks/{chunkId}`
- `/api/v1/floors/{floorId}/scans/merge`
- `/api/v1/floors/{floorId}/scans/merge/status`
- `/api/v1/passages/{passageId}`
- `/mobile/v1/routes/coordinates`
- `/route`
- `/scan/upload`
- `/scan/{scan_id}`
- `/scan/{scan_id}/build`
- `/scan/{scan_id}/graph`
- `/scan/{scan_id}/imdf`
- `/healthz`
- `/readyz`
- `/v3/api-docs`
- `/swagger-ui/index.html`

## 보존 기준

- `/api/slam` 아래 listed legacy 호환 API는 제거하지 않는다.
- `/api/slam/*/localize` 외부 요청은 `building_id` 또는 legacy `map_id` form field + 이미지 파일 업로드를 받는다.
- `map_id`는 별도 map table ID가 아니라 건물 ID로 해석해 활성 floor map을 조회한다.

## 연결 흐름

`/api/v1/buildings/{buildingId}/localize`
→ `LocalizationAdapter`
→ `SLAMV3Localizer` when `INDOOR_VPS_LOCALIZER_MODE=slam_v3`
→ legacy `be.routes.slam_routes._localize_impl`
→ `PostgresAdapter.get_floor_maps(buildingId)`
→ v2 `building_floor` + `floor_scan` + `scan_ingest`
→ `var/storage/scans/{scan_id}/rtabmap.db`
→ SuperPoint + LightGlue localization
