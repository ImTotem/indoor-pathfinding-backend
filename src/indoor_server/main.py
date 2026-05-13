"""FastAPI 앱 factory."""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from indoor_server.config import settings
from indoor_server.domain.building.errors import BuildDomainError
from indoor_server.domain.scan.errors import ScanDomainError
from indoor_server.interfaces.api.router import router
from indoor_server.interfaces.api.v1_router import v1_router

_SERVER_ROOT = Path(__file__).resolve().parents[2]
_LEGACY_BE_ROOT = _SERVER_ROOT / "be"
if _LEGACY_BE_ROOT.exists() and str(_LEGACY_BE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEGACY_BE_ROOT))

logging.basicConfig(level=settings.log_level)

logger = logging.getLogger(__name__)


OPENAPI_TAGS = [
    {"name": "scan", "description": "스캔 업로드, 조회, 수집 메타데이터 API"},
    {"name": "build", "description": "스캔 빌드 작업 실행과 결과 조회 API"},
    {"name": "route", "description": "좌표 기반 실내 경로 탐색 API"},
    {"name": "mobile", "description": "모바일 앱에서 바로 쓰는 얇은 호환 API"},
    {"name": "imdf", "description": "IMDF/지도 내보내기 API"},
    {
        "name": "사용자 앱 API",
        "description": "사용자 앱이 직접 호출하는 API: 건물조회, v3 localize, POI 조회, 경로 탐색",
    },
    {"name": "SLAM - 처리", "description": "`/api/slam` 처리, 상태, 헬스, 메타데이터 API"},
    {"name": "SLAM - 위치추정", "description": "이미지 파일 업로드 기반 VPS/SLAM 위치 추정 API"},
    {"name": "SLAM - 디버그", "description": "마스킹과 매칭 결과를 확인하는 디버그 API"},
    {"name": "V1 - 건물", "description": "기존 앱 호환 건물 CRUD와 상태 API"},
    {"name": "V1 - 층", "description": "기존 앱 호환 층 CRUD와 건물별 층 목록 API"},
    {"name": "V1 - 지도 데이터", "description": "층 경로, pointcloud 같은 지도 산출물 조회 API"},
    {"name": "V1 - 스캔/처리", "description": "층 스캔 chunk 업로드, 병합, 처리 상태 API"},
    {"name": "V1 - 길찾기", "description": "건물/층 기준 길찾기와 좌표 route API"},
    {"name": "V1 - 위치추정", "description": "이미지 위치추정과 주변 node image 조회 API"},
    {"name": "V1 - SLAM", "description": "기존 앱 호환 SLAM 상태와 메타데이터 API"},
    {"name": "V1 - POI", "description": "건물 POI 목록, 생성, 검색 API"},
    {"name": "V1 - 통로", "description": "수직 통로 목록과 상세 조회 API"},
    {"name": "health", "description": "서버 헬스 체크 API"},
    {"name": "dev-viewer", "description": "로컬 개발용 지도 뷰어 API"},
]


async def _warmup_active_floor_caches(pool: asyncpg.Pool) -> None:
    """Preload SuperPoint cache for every floor whose active scan is READY.

    Runs once at server startup. Each floor's indexing dispatched to a worker
    thread (sync) so the lifespan startup doesn't block. After uvicorn reload
    this restores the cache that the in-memory `SuperPointMapManager`
    singleton lost on the process restart.
    """
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT fs.floor_id::text AS floor_id,
                       fs.scan_id::text  AS scan_id,
                       si.storage_path   AS storage_path
                FROM floor_scan fs
                JOIN scan_ingest si ON si.scan_id = fs.scan_id
                WHERE fs.active = true AND fs.status = 'READY'
                """
            )
    except Exception as exc:
        logger.warning("warmup: query active floors failed: %s", exc)
        return
    if not rows:
        logger.info("warmup: no active READY scans to warm up")
        return

    storage_root = Path(os.getenv("STORAGE_ROOT", "/app/var/storage"))
    loop = asyncio.get_event_loop()
    for r in rows:
        floor_id = r["floor_id"]
        storage_path = r["storage_path"]
        reproc = storage_root / storage_path / "rtabmap_reprocessed.db"
        raw = storage_root / storage_path / "rtabmap.db"
        if reproc.exists() and reproc.stat().st_size > 0:
            db_path = reproc
        elif raw.exists():
            db_path = raw
        else:
            logger.warning(
                "warmup: no rtabmap db for floor=%s storage=%s",
                floor_id, storage_path,
            )
            continue
        logger.info(
            "warmup: queueing floor=%s db=%s", floor_id, db_path.name,
        )
        loop.run_in_executor(None, _do_floor_warmup, floor_id, str(db_path))


def _do_floor_warmup(map_id: str, db_path: str) -> None:
    try:
        from be.slam_engines.superpoint.map_manager import SuperPointMapManager
        SuperPointMapManager().get_or_load(map_id, db_path)
        logger.info("[startup warmup] cache ready map_id=%s", map_id)
    except Exception as exc:
        logger.warning(
            "[startup warmup] failed map_id=%s err=%s", map_id, exc,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize `/api/slam/*` dependencies (메인 사용자 앱 endpoint v3)."""

    slam_pool = None
    slam_queue = None
    slam_routes_module = None
    try:
        from config.settings import settings as slam_settings
        from routes import slam_routes
        from slam_engines.rtabmap.engine import RTABMapEngine
        from storage.postgres_adapter import PostgresAdapter
        from utils.job_queue import SLAMJobQueue

        slam_routes_module = slam_routes

        try:
            slam_pool = await asyncpg.create_pool(
                host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                database=os.getenv("POSTGRES_DB", "indoor"),
                user=os.getenv("POSTGRES_USER", "indoor"),
                password=os.getenv("POSTGRES_PASSWORD", "indoor"),
                min_size=1,
                max_size=10,
            )
            adapter = PostgresAdapter(slam_pool)
            slam_routes.postgres_adapter = adapter
            try:
                slam_queue = SLAMJobQueue(
                    adapter,
                    RTABMapEngine(),
                    slam_settings.MAPS_DIR,
                )
                await slam_queue.start_worker()
                slam_routes.job_queue = slam_queue
            except Exception as exc:
                logger.warning("SLAM job queue disabled: %s", exc)
                slam_routes.job_queue = None
        except Exception as exc:
            logger.warning("SLAM DB adapter disabled: %s", exc)
            slam_routes.postgres_adapter = None
            slam_routes.job_queue = None
    except Exception as exc:
        logger.warning("SLAM router initialized without dependencies: %s", exc)

    # Kick off SuperPoint cache warmup for every active READY floor scan so
    # the first /localize after a server restart doesn't pay the cold-start
    # cost. fire-and-forget — actual indexing runs in worker threads.
    if slam_pool is not None:
        try:
            asyncio.create_task(_warmup_active_floor_caches(slam_pool))
        except Exception as exc:
            logger.warning("warmup task schedule failed: %s", exc)

    yield

    if slam_queue is not None:
        await slam_queue.shutdown()
    if slam_pool is not None:
        await slam_pool.close()
    if slam_routes_module is not None:
        slam_routes_module.postgres_adapter = None
        slam_routes_module.job_queue = None


app = FastAPI(
    title="Indoor Pathfinding Server",
    version="0.1.0",
    description="Scan upload & metadata ingest API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(v1_router)

try:
    from routes import slam_routes

    app.include_router(slam_routes.router)
except Exception as exc:
    logger.warning("SLAM router not registered: %s", exc)


@app.get("/v3/api-docs", include_in_schema=False)
async def springdoc_openapi_alias() -> JSONResponse:
    """Springdoc-compatible OpenAPI JSON alias for V1 client/backend comparison."""
    return JSONResponse(app.openapi())


@app.get("/swagger-ui", include_in_schema=False)
async def swagger_ui_redirect() -> RedirectResponse:
    return RedirectResponse(url="/swagger-ui/index.html")


@app.get("/swagger-ui/", include_in_schema=False)
async def swagger_ui_slash_redirect() -> RedirectResponse:
    return RedirectResponse(url="/swagger-ui/index.html")


@app.get("/swagger-ui/index.html", include_in_schema=False)
async def swagger_ui_index() -> HTMLResponse:
    """Swagger UI path compatible with the previous Spring Boot deployment."""
    return get_swagger_ui_html(
        openapi_url="/v3/api-docs",
        title=f"{app.title} - Swagger UI",
    )

# ── Dev Viewer (INDOOR_DEV_VIEWER_ENABLED=true 일 때만 활성) ──────────────────
if settings.dev_viewer_enabled:
    from indoor_server.interfaces.dev.viewer_router import dev_router

    app.include_router(dev_router)

    _viewer_dir = Path(__file__).resolve().parents[2] / "static" / "imdf_viewer"
    if _viewer_dir.exists():
        app.mount(
            "/dev/viewer",
            StaticFiles(directory=str(_viewer_dir), html=True),
            name="imdf-viewer",
        )
        logger.info("Dev IMDF viewer 활성: /dev/viewer/")
    else:
        logger.warning("Dev viewer 디렉터리 없음: %s", _viewer_dir)


@app.exception_handler(BuildDomainError)
async def build_domain_error_handler(request: Request, exc: BuildDomainError) -> JSONResponse:
    status_map = {
        "BUILD_ALREADY_RUNNING": 409,
        "BUILD_NOT_FOUND": 404,
        "GRAPH_NOT_READY": 422,
        "MODEL_LOAD_FAILED": 500,
    }
    status_code = status_map.get(exc.code, 500)
    return JSONResponse(
        status_code=status_code,
        content={"code": exc.code, "message": exc.message, "detail": exc.detail or None},
    )


@app.exception_handler(ScanDomainError)
async def domain_error_handler(request: Request, exc: ScanDomainError) -> JSONResponse:
    status_map = {
        "ZIP_STRUCTURE_INVALID": 400,
        "MISSING_REQUIRED_FILE": 400,
        "SCAN_ID_MISMATCH": 400,
        "INVALID_SCAN_ID": 422,
        "SIDECAR_SCHEMA_MISMATCH": 422,
        # Sprint 49 (Codex BLOCKER 4): manifest v4/v5 contract 위반 → 422.
        "MANIFEST_VERSION_MISMATCH": 422,
        "MANIFEST_STRUCTURE_INVALID": 422,
        "SCAN_CONFLICT": 409,
        "PAYLOAD_TOO_LARGE": 413,
        "UNAUTHORIZED": 401,
    }
    status_code = status_map.get(exc.code, 500)
    return JSONResponse(
        status_code=status_code,
        content={"code": exc.code, "message": exc.message, "detail": exc.detail or None},
    )


@app.exception_handler(HTTPException)
async def v1_http_exception_handler(request: Request, exc: HTTPException) -> Response:
    if request.url.path.startswith("/api/v1/") and _is_v1_error_detail(exc.detail):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail,
            headers=exc.headers,
        )
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def v1_request_validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> Response:
    if request.url.path.startswith("/api/v1/"):
        return JSONResponse(
            status_code=422,
            content={
                "code": "VALIDATION_ERROR",
                "message": "request validation failed",
                "detail": {"errors": jsonable_encoder(exc.errors())},
            },
        )
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """예상치 못한 예외를 500 ApiError 포맷으로 반환."""
    logger.error("unhandled exception", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"code": "INTERNAL", "message": "내부 서버 오류", "detail": None},
    )


def _is_v1_error_detail(detail: object) -> bool:
    return (
        isinstance(detail, dict)
        and isinstance(detail.get("code"), str)
        and isinstance(detail.get("message"), str)
        and "detail" in detail
    )


def run() -> None:
    uvicorn.run("indoor_server.main:app", host="0.0.0.0", port=8000, reload=True)
