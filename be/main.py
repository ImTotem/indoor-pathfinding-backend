# main.py
import os
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from slam_interface.factory import SLAMEngineFactory
import slam_engines  # 엔진 자동 등록

from routes import scan, localize, path, viewer, maps, slam_routes
from storage.postgres_adapter import PostgresAdapter
from utils.job_queue import SLAMJobQueue
from utils.temp_file_manager import cleanup_orphaned_temps


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown events.
    
    Startup:
    - Initialize asyncpg connection pool
    - Create PostgresAdapter for database operations
    - Create SLAMJobQueue for sequential job processing
    - Inject dependencies into slam_routes module
    - Start background job queue worker
    - Clean up orphaned temporary files
    
    Shutdown:
    - Gracefully shutdown job queue worker
    - Close asyncpg connection pool
    
    IMPORTANT: Run with single worker to guarantee singleton queue behavior
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
    """
    # Startup
    pool = await asyncpg.create_pool(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "slam_db"),
        user=os.getenv("POSTGRES_USER", "slam_service"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        min_size=1,
        max_size=10
    )
    
    postgres_adapter = PostgresAdapter(pool)
    slam_routes.postgres_adapter = postgres_adapter
    
    slam_engine = SLAMEngineFactory.create(settings.SLAM_ENGINE_TYPE)
    job_queue = SLAMJobQueue(postgres_adapter, slam_engine)
    slam_routes.job_queue = job_queue
    
    await job_queue.start_worker()
    cleanup_orphaned_temps()
    
    yield  # Server runs
    
    # Shutdown
    await job_queue.shutdown()
    await pool.close()


# FastAPI 앱
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"\n{'='*60}")
print(f"  {settings.API_TITLE}")
print(f"  SLAM Engine: {settings.SLAM_ENGINE_TYPE}")
print(f"  Available: {SLAMEngineFactory.list_engines()}")
print(f"{'='*60}\n")

# 라우터 등록
app.include_router(scan.router)
app.include_router(localize.router)
app.include_router(path.router)
app.include_router(viewer.router)
app.include_router(maps.router)
app.include_router(slam_routes.router)

@app.get("/")
async def root():
    """헬스 체크"""
    return {
        "service": settings.API_TITLE,
        "status": "running",
        "version": settings.API_VERSION,
        "slam_engine": settings.SLAM_ENGINE_TYPE,
        "available_engines": SLAMEngineFactory.list_engines(),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level=settings.LOG_LEVEL.lower())

