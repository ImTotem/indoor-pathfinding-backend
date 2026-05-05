# Migration Status

## 현재 완성 기준

- v2 서버 파일이 current repo의 실행 기준이다.
- FastAPI app import 기준 노출 확인:
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
  - `/api/slam/v3/localize`
  - `/api/v1/buildings/{buildingId}/localize`
  - `/mobile/v1/routes/coordinates`
  - `/route`, `/scan/upload`, `/scan/{scan_id}/build`, `/scan/{scan_id}/graph`, `/scan/{scan_id}/imdf`
- `/api/slam` listed 호환 API는 모두 노출한다.
- `/api/slam/*/localize`와 debug upload API는 Swagger에서 `multipart/form-data` 이미지 파일 업로드로 표시한다.
- Swagger 태그는 `SLAM - 처리`, `SLAM - 위치추정`, `SLAM - 디버그`로 분리했다.
- `be/config`, `be/models`, `be/routes`, `be/storage`는 명시 패키지로 고정해 외부 `config` 패키지 충돌을 차단했다.
- legacy `be`의 잘린 RTAB-Map 파일은 `/api/slam/*` 호환에 필요한 최소 surface로 대체했다:
  - `RTABMapEngine.extract_intrinsics_from_db()`
  - `RTABMapEngine.scale_intrinsics()`
  - `RTABMapEngine.database_parser`
  - `RTABMapEngine._load_map_file()`
  - RTAB-Map descriptor map manager는 SuperPoint debug 경로 기준 stub 처리
- `uv.lock`은 `opencv-python-headless` 제거, `opencv-contrib-python` 추가 상태로 갱신했다.

## 검증 결과

```bash
PYTHONPATH=src:be python3 -m compileall -q src be
```

결과: PASS

```bash
PYTHONPATH=src:be python3 -m pytest \
  tests/integration/api/test_route.py \
  tests/integration/api/test_v1_localization_api.py \
  tests/unit/application/test_v1_compat_services.py \
  -q
```

결과: `20 passed, 4 warnings`

```bash
PYTHONPATH=src:be python3 - <<'PY'
from indoor_server.main import app
print([p for p in sorted({route.path for route in app.routes}) if p.startswith('/api/slam')])
PY
```

결과:

```text
['/api/slam/health',
 '/api/slam/localize',
 '/api/slam/maps/{building_id}/metadata',
 '/api/slam/process',
 '/api/slam/status/{building_id}',
 '/api/slam/v1/debug/matches',
 '/api/slam/v2/debug/mask',
 '/api/slam/v2/debug/matches',
 '/api/slam/v2/localize',
 '/api/slam/v3/debug/matches',
 '/api/slam/v3/localize']
```

## 남은 작업

| 우선순위 | 작업 | 이유 |
|---|---|---|
| P0 | `.git` 메타데이터 복구/재초기화 결정 | 현재 상태로는 diff/commit/rollback 추적 불가 |
| P0 | 실제 Postgres + 실제 `rtabmap.db`로 `/api/slam/*/localize` e2e | 현재 검증은 import/route/unit 중심이고 실제 SuperPoint 매칭은 미검증 |
| P0 | 실제 이미지 파일 업로드로 iOS localize 요청 검증 | 외부 계약을 base64 JSON에서 multipart 파일 업로드로 바꿨기 때문 |
| P1 | Docker image build smoke | `torch`, `torchvision`, `opencv-contrib-python`, `lightglue` 조합이 이미지에서 설치되는지 확인 필요 |
| P1 | v2 scan ingest 후 active floor map discovery 검증 | `scan_ingest.storage_path` → `var/storage/scans/{scan_id}/rtabmap.db` 연결 확인 필요 |
| P2 | legacy `be` 불필요 파일 추가 pruning | 호환 surface 복구 과정에서 일부 import-compat stub과 비사용 파일은 남아 있음 |

## 로컬 테스트 환경

```bash
cp .env.example .env
docker compose up -d db
PYTHONPATH=src:be uv run alembic upgrade head
PYTHONPATH=src:be uv run uvicorn indoor_server.main:app --reload
```

로컬에서 실제 SLAM 위치추정을 쓰려면 `.env`에 다음 값이 필요하다.

```dotenv
INDOOR_VPS_LOCALIZER_MODE=slam_v3
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432
POSTGRES_DB=indoor
POSTGRES_USER=indoor
POSTGRES_PASSWORD=indoor
DATA_DIR=./be/data
RTABMAP_PATH=/opt/homebrew/bin/rtabmap-reprocess
```
