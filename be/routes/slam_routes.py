import base64
import io
from fastapi import APIRouter, HTTPException, status
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

postgres_adapter = None
job_queue = None


@router.post(
    "/process",
    response_model=SLAMProcessResponse,
    status_code=status.HTTP_200_OK,
)
async def process_slam(request: SLAMProcessRequest):
    if postgres_adapter is None or job_queue is None:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    sessions = await postgres_adapter.get_sessions_by_building_id(request.building_id)
    if not sessions:
        raise HTTPException(status_code=404, detail=f"No sessions found for building {request.building_id}")
    
    session_db_pairs = []
    for session in sessions:
        file_path = session.get("file_path")
        if not file_path:
            logger.warning(f"[SLAM-PROCESS] Session {session['id']} has no file_path, skipping")
            continue
        session_db_pairs.append((session["id"], file_path))
    
    if not session_db_pairs:
        raise HTTPException(status_code=404, detail=f"No uploadedfiles found for building {request.building_id}")
    
    logger.info(f"[SLAM-PROCESS] Enqueuing building: {request.building_id}, sessions: {len(session_db_pairs)}")
    
    try:
        await job_queue.enqueue(request.building_id, session_db_pairs)
        
        return SLAMProcessResponse(
            map_id=request.building_id,
            status="PROCESSING",
            queue_position=job_queue.get_queue_length()
        )
    except Exception as e:
        logger.error(f"[SLAM-PROCESS] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{building_id}")
async def get_slam_status(building_id: str):
    if postgres_adapter is None:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    sessions = await postgres_adapter.get_sessions_by_building_id(building_id)
    if not sessions:
        raise HTTPException(status_code=404, detail=f"No sessions found for building {building_id}")
    
    for s in sessions:
        if s.get("created_at"):
            s["created_at"] = s["created_at"].isoformat()
        if s.get("updated_at"):
            s["updated_at"] = s["updated_at"].isoformat()
    
    statuses = [s.get("status") for s in sessions]
    if all(st == "COMPLETED" for st in statuses):
        overall_status = "COMPLETED"
    elif any(st == "FAILED" for st in statuses):
        overall_status = "FAILED"
    elif any(st == "PROCESSING" for st in statuses):
        overall_status = "PROCESSING"
    else:
        overall_status = "UPLOADED"
    
    return {
        "building_id": building_id,
        "overall_status": overall_status,
        "sessions": sessions,
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    postgres_status = "not_initialized"
    if postgres_adapter is not None:
        postgres_status = await postgres_adapter.health_check()
    
    queue_length = job_queue.get_queue_length() if job_queue else 0
    overall_status = "healthy" if postgres_status == "connected" else "degraded"
    
    return HealthResponse(
        status=overall_status,
        postgres=postgres_status,
        queue_length=queue_length
    )


@router.get("/maps/{building_id}/metadata", response_model=MapMetadata)
async def get_map_metadata(building_id: str):
    if postgres_adapter is None:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    sessions = await postgres_adapter.get_sessions_by_building_id(building_id)
    if not sessions:
        raise HTTPException(status_code=404, detail=f"No sessions found for building {building_id}")
    
    num_keyframes = 0
    db_path = settings.MAPS_DIR / f"{building_id}.db"
    if db_path.exists():
        try:
            parser = DatabaseParser()
            parsed = await parser.parse_database(str(db_path))
            num_keyframes = parsed.get('num_keyframes', 0)
        except Exception as e:
            logger.warning(f"[SLAM-METADATA] Failed to parse database: {e}")
    
    earliest_created = min(
        (s["created_at"] for s in sessions if s.get("created_at")),
        default=None,
    )
    created_at = earliest_created.isoformat() if earliest_created else ""
    
    statuses = [s.get("status") for s in sessions]
    if all(st == "COMPLETED" for st in statuses):
        overall_status = "COMPLETED"
    elif any(st == "FAILED" for st in statuses):
        overall_status = "FAILED"
    elif any(st == "PROCESSING" for st in statuses):
        overall_status = "PROCESSING"
    else:
        overall_status = "UPLOADED"
    
    return MapMetadata(
        map_id=building_id,
        building_id=building_id,
        num_keyframes=num_keyframes,
        created_at=created_at,
        status=overall_status,
    )


@router.post("/localize", response_model=SLAMLocalizeResponse, status_code=status.HTTP_200_OK)
async def localize_in_map(request: SLAMLocalizeRequest):
    logger.info(f"[SLAM-LOCALIZE] map_id: {request.map_id}")
    
    try:
        image_bytes_list = []
        for i, img_b64 in enumerate(request.images):
            try:
                image_bytes_list.append(base64.b64decode(img_b64))
            except Exception as e:
                raise ValueError(f"Invalid base64 in image {i+1}: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    slam_engine = SLAMEngineFactory.create(settings.SLAM_ENGINE_TYPE)
    
    db_path = settings.MAPS_DIR / f"{request.map_id}.db"
    try:
        intrinsics = slam_engine.extract_intrinsics_from_db(str(db_path))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Map not found: {request.map_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract intrinsics: {e}")
    
    try:
        img = Image.open(io.BytesIO(image_bytes_list[0]))
        img_width, img_height = img.size
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
    
    if (img_width, img_height) != (intrinsics['width'], intrinsics['height']):
        try:
            intrinsics = slam_engine.scale_intrinsics(intrinsics, img_width, img_height)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to scale intrinsics: {e}")
    
    try:
        result = await slam_engine.localize(request.map_id, image_bytes_list, intrinsics=intrinsics)
        return SLAMLocalizeResponse(pose=result['pose'], confidence=result['confidence'])
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except TimeoutError:
        raise HTTPException(status_code=503, detail="Localization timeout (30s)")
    except ValueError as e:
        raise HTTPException(status_code=503, detail=f"Localization failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
