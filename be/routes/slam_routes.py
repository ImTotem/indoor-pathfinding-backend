# routes/slam_routes.py
import base64
import io
from fastapi import APIRouter, HTTPException, status
from pathlib import Path
from PIL import Image

from models.slam_api import (
    SLAMProcessRequest,
    SLAMProcessResponse,
    SLAMLocalizeRequest,
    SLAMLocalizeResponse,
    MapMetadata,
    HealthResponse
)
from slam_interface.factory import SLAMEngineFactory
from config.settings import Settings
from utils import logger
from slam_engines.rtabmap.database_parser import DatabaseParser

settings = Settings()

router = APIRouter(prefix="/api/slam", tags=["SLAM"])

# TODO: Inject via lifespan in Task 7
# These will be initialized during app startup
postgres_adapter = None
job_queue = None


@router.post(
    "/process",
    response_model=SLAMProcessResponse,
    status_code=status.HTTP_200_OK,
    summary="Enqueue SLAM processing job",
    description="Creates a new SLAM processing job and enqueues it for background processing"
)
async def process_slam(request: SLAMProcessRequest):
    """
    Enqueue SLAM processing for a completed scan session.
    
    Returns map_id and queue position for status tracking.
    """
    # Check dependencies initialized
    if postgres_adapter is None:
        logger.error("[SLAM-PROCESS] PostgresAdapter not initialized")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not initialized"
        )
    
    if job_queue is None:
        logger.error("[SLAM-PROCESS] SLAMJobQueue not initialized")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Job queue not initialized"
        )
    
    # Check session exists
    session_path = Path(f"data/sessions/{request.session_id}")
    if not session_path.exists():
        logger.warning(f"[SLAM-PROCESS] Session not found: {request.session_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {request.session_id} not found"
        )
    
    logger.info(f"[SLAM-PROCESS] Creating job for session: {request.session_id}")
    
    try:
        # Create job in database
        map_id = await postgres_adapter.create_job(request.session_id)
        logger.info(f"[SLAM-PROCESS] Job created: map_id={map_id}")
        
        # Enqueue for processing
        await job_queue.enqueue(map_id, request.session_id)
        logger.info(f"[SLAM-PROCESS] Job enqueued: map_id={map_id}")
        
        # Get queue position
        queue_position = job_queue.get_queue_length()
        
        return SLAMProcessResponse(
            map_id=map_id,
            status="pending",
            queue_position=queue_position
        )
        
    except Exception as e:
        logger.error(f"[SLAM-PROCESS] Error creating job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create SLAM job: {str(e)}"
        )


@router.get(
    "/status/{map_id}",
    response_model=dict,
    summary="Get SLAM job status",
    description="Retrieve current status and metadata for a SLAM processing job"
)
async def get_slam_status(map_id: str):
    """
    Get job status by map_id.
    
    Returns job metadata including status, session_id, timestamps, and error messages.
    """
    # Check dependencies initialized
    if postgres_adapter is None:
        logger.error("[SLAM-STATUS] PostgresAdapter not initialized")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not initialized"
        )
    
    logger.debug(f"[SLAM-STATUS] Fetching status for map_id: {map_id}")
    
    try:
        job_info = await postgres_adapter.get_job_status(map_id)
        
        if not job_info:
            logger.warning(f"[SLAM-STATUS] Job not found: {map_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {map_id} not found"
            )
        
        # Convert timestamps to ISO format strings if present
        if job_info.get("created_at"):
            job_info["created_at"] = job_info["created_at"].isoformat()
        if job_info.get("updated_at"):
            job_info["updated_at"] = job_info["updated_at"].isoformat()
        
        logger.debug(f"[SLAM-STATUS] Job status: {job_info['status']}")
        return job_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SLAM-STATUS] Error fetching job status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch job status: {str(e)}"
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    description="Check PostgreSQL connection and queue status"
)
async def health_check():
    """
    Check system health.
    
    Returns overall status, PostgreSQL connection state, and current queue length.
    """
    # Check PostgreSQL connection
    postgres_status = "disconnected"
    
    if postgres_adapter is None:
        postgres_status = "not_initialized"
    else:
        try:
            # Try a simple query to verify connection
            # Adapter will retry on connection failure
            _ = await postgres_adapter.get_job_status("health_check_test")
            # If we get here, connection is working (even if job doesn't exist)
            postgres_status = "connected"
        except Exception as e:
            logger.warning(f"[SLAM-HEALTH] PostgreSQL health check failed: {e}")
            postgres_status = f"error: {str(e)}"
    
    # Get queue length
    queue_length = 0
    if job_queue is not None:
        try:
            queue_length = job_queue.get_queue_length()
        except Exception as e:
            logger.warning(f"[SLAM-HEALTH] Queue length check failed: {e}")
    
    # Determine overall status
    overall_status = "healthy" if postgres_status == "connected" else "degraded"
    
    logger.debug(f"[SLAM-HEALTH] Status: {overall_status}, Postgres: {postgres_status}, Queue: {queue_length}")
    
    return HealthResponse(
        status=overall_status,
        postgres=postgres_status,
        queue_length=queue_length
    )


@router.get(
    "/maps/{map_id}/metadata",
    response_model=MapMetadata,
    summary="Get map metadata",
    description="Retrieve metadata for a completed SLAM map (placeholder - keyframes count not yet implemented)"
)
async def get_map_metadata(map_id: str):
    """
    Get map metadata.
    
    Returns map information including session_id, status, and creation time.
    Note: num_keyframes is currently a placeholder (0) - actual implementation pending.
    """
    # Check dependencies initialized
    if postgres_adapter is None:
        logger.error("[SLAM-METADATA] PostgresAdapter not initialized")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not initialized"
        )
    
    logger.debug(f"[SLAM-METADATA] Fetching metadata for map_id: {map_id}")
    
    try:
        job_info = await postgres_adapter.get_job_status(map_id)
        
        if not job_info:
            logger.warning(f"[SLAM-METADATA] Map not found: {map_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Map {map_id} not found"
            )
        
        # Format timestamps
        created_at = job_info["created_at"].isoformat() if job_info.get("created_at") else ""
        
        # Extract actual keyframe count from RTAB-Map database
        try:
            parser = DatabaseParser()
            db_path = settings.MAPS_DIR / f"{map_id}.db"
            if db_path.exists():
                parsed = await parser.parse_database(str(db_path))
                num_keyframes = parsed.get('num_keyframes', 0)
            else:
                logger.warning(f"[SLAM-METADATA] Database file not found: {db_path}")
                num_keyframes = 0
        except Exception as e:
            logger.warning(f"[SLAM-METADATA] Failed to parse database: {e}")
            num_keyframes = 0
        
        logger.debug(f"[SLAM-METADATA] Map metadata: status={job_info['status']}, keyframes={num_keyframes}")
        
        return MapMetadata(
            map_id=map_id,
            session_id=job_info["session_id"],
            num_keyframes=num_keyframes,
            created_at=created_at,
            status=job_info["status"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SLAM-METADATA] Error fetching map metadata: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch map metadata: {str(e)}"
        )


@router.post(
    "/localize",
    response_model=SLAMLocalizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Localize image in existing SLAM map",
    description="Real-time localization using base64-encoded image against existing map database"
)
async def localize_in_map(request: SLAMLocalizeRequest):
    """Localize an image in an existing SLAM map.
    
    Parses base64 image, extracts camera intrinsics from map database,
    and returns estimated camera pose with confidence score.
    """
    logger.info(f"[SLAM-LOCALIZE] Localize request for map_id: {request.map_id}")
    
    # 1. Decode base64 images (1-5)
    try:
        image_bytes_list = []
        for i, img_b64 in enumerate(request.images):
            try:
                img_bytes = base64.b64decode(img_b64)
                image_bytes_list.append(img_bytes)
            except Exception as e:
                raise ValueError(f"Invalid base64 in image {i+1}: {e}")
    except ValueError as e:
        logger.error(f"[SLAM-LOCALIZE] {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[SLAM-LOCALIZE] Failed to decode images: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to decode images: {e}")
    
    # 2. Create SLAM engine
    slam_engine = SLAMEngineFactory.create(settings.SLAM_ENGINE_TYPE)
    
    # 3. Extract intrinsics from map database
    db_path = settings.MAPS_DIR / f"{request.map_id}.db"
    try:
        intrinsics = slam_engine.extract_intrinsics_from_db(str(db_path))
    except FileNotFoundError:
        logger.warning(f"[SLAM-LOCALIZE] Map not found: {request.map_id}")
        raise HTTPException(status_code=404, detail=f"Map not found: {request.map_id}")
    except Exception as e:
        logger.error(f"[SLAM-LOCALIZE] Failed to extract intrinsics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract intrinsics: {e}")
    
    # 4. Get FIRST image resolution and scale intrinsics if needed
    try:
        img = Image.open(io.BytesIO(image_bytes_list[0]))
        img_width, img_height = img.size
    except Exception as e:
        logger.error(f"[SLAM-LOCALIZE] Invalid image data: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
    
    if (img_width, img_height) != (intrinsics['width'], intrinsics['height']):
        try:
            intrinsics = slam_engine.scale_intrinsics(intrinsics, img_width, img_height)
        except Exception as e:
            logger.error(f"[SLAM-LOCALIZE] Failed to scale intrinsics: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to scale intrinsics: {e}")
    
    # 5. Call localize
    try:
        result = await slam_engine.localize(request.map_id, image_bytes_list, intrinsics=intrinsics)
        return SLAMLocalizeResponse(pose=result['pose'], confidence=result['confidence'])
    except FileNotFoundError as e:
        logger.error(f"[SLAM-LOCALIZE] Map not found during localization: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except TimeoutError:
        logger.error("[SLAM-LOCALIZE] Localization timeout")
        raise HTTPException(status_code=503, detail="Localization timeout (30s)")
    except ValueError as e:
        logger.error(f"[SLAM-LOCALIZE] Localization failed: {e}")
        raise HTTPException(status_code=503, detail=f"Localization failed: {e}")
    except Exception as e:
        logger.error(f"[SLAM-LOCALIZE] Server error: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
