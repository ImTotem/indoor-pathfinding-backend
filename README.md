# indoor-server

Indoor Pathfinding 서버. FastAPI + PostgreSQL(PostGIS + pgvector) 기반 스캔 수신 서버.

## 빠른 시작

```bash
# 1. uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 의존성 설치
uv sync --extra dev

# 3. 환경 변수
cp .env.example .env

# 4. DB 기동
docker compose up db -d

# 5. 마이그레이션
uv run alembic upgrade head

# 6. 서버 기동
uv run uvicorn indoor_server.main:app --reload
```

## 검증

```bash
uv run ruff check .
uv run mypy .
uv run pytest                   # 단위 + mock 기반 통합 테스트 (DB 불필요)
uv run pytest -m integration    # Postgres 기동 후 실행 (docker compose up db -d)
```

## 주요 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| POST | `/scan/upload` | ZIP 업로드 (Bearer 토큰 필요) |
| GET  | `/scan/{scan_id}` | 수신 기록 조회 |
| GET  | `/healthz` | Liveness 체크 |
| GET  | `/readyz` | DB + 스토리지 Readiness 체크 |
| GET  | `/docs` | Swagger UI |
