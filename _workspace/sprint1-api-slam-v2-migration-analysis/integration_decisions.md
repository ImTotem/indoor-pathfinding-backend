# Integration Decisions

## 결론

v2 서버를 기준 구현으로 두고, current repo 앱 진입점은 `src/indoor_server.main:app`로 유지한다. legacy `be`는 listed `/api/slam/*` 호환 API를 current app에 붙인다.

## 확정 결정

| 항목 | 결정 | 반영 위치 |
|---|---|---|
| 앱 기준 | v2 `src/indoor_server` 유지 | `src/indoor_server/main.py` |
| SLAM 위치 | legacy `be`에 남김 | `be/routes/slam_routes.py` |
| 노출 SLAM API | listed `/api/slam/*` 호환 API 전체 노출 | `be/routes/slam_routes.py`, `src/indoor_server/main.py` |
| VPS 연결 | v2 V1 localize adapter가 SLAM v3 구현을 호출 | `src/indoor_server/application/api_v1/localization_adapter.py` |
| DB 기준 | v2 schema 기준: `building_floor` + `floor_scan` + `scan_ingest` | `be/storage/postgres_adapter.py` |
| localize 요청 | 외부 API는 `multipart/form-data` 이미지 파일 업로드 사용 | `be/routes/slam_routes.py` |
| map_id 처리 | `map_id`는 건물 ID 호환 alias로 처리하고 floor map은 building_id로 조회 | `be/routes/slam_routes.py`, `be/storage/postgres_adapter.py` |
| 환경 기준 | Docker는 compose env로, 로컬은 `.env`로 주입 | `docker-compose.yml`, `.env.example` |
| current repo workspace | 통합 분석/결정 문서만 유지 | `_workspace/sprint1-api-slam-v2-migration-analysis/` |

## 권장 기본값으로 처리한 항목

- `INDOOR_VPS_LOCALIZER_MODE` 기본값은 `mock` 유지: 기존 테스트와 문서 스모크 보호.
- Docker compose의 `server`는 `INDOOR_VPS_LOCALIZER_MODE=slam_v3` 주입: 배포/시연에서는 실제 SLAM v3 경로 사용.
- legacy `/api/slam/process`, `/api/slam/status`, `/api/slam/health`, `/api/slam/maps/*/metadata`, `/api/slam/*/debug/*`는 호환 표면으로 노출.
- legacy SLAM queue는 DB adapter 초기화에 성공하면 current app lifespan에서 시작한다. DB가 없으면 process endpoint는 상태 기반 fallback만 반환한다.
- Swagger는 `SLAM - 처리`, `SLAM - 위치추정`, `SLAM - 디버그` 태그로 분리하고 설명은 한글로 유지한다.

## 주의

현재 파일 트리에는 통합 도중 삭제된 기존 `.git` 메타데이터가 없다. 원격 clone으로 확인한 `20HyeonsuLee/indoor-pathfinding-backend`는 삭제 전 local FastAPI/SLAM repo와 구조가 맞지 않아 복구 소스로 사용하지 않았다. 커밋/PR 작업 전에는 `.git` 복구 또는 새 repo 초기화 방식을 결정해야 한다.
