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
    
    session = await postgres_adapter.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found")
    
    file_path = session.get("file_path")
    if not file_path:
        raise HTTPException(status_code=404, detail=f"No file_path for session {request.session_id}")
    
    logger.info(f"[SLAM-PROCESS] Enqueuing session: {request.session_id}, file: {file_path}")
    
    try:
        await job_queue.enqueue(request.session_id, file_path)
        
        return SLAMProcessResponse(
            map_id=request.session_id,
            status=session.get("status", "UPLOADED"),
            queue_position=job_queue.get_queue_length()
        )
    except Exception as e:
        logger.error(f"[SLAM-PROCESS] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{session_id}")
async def get_slam_status(session_id: str):
    if postgres_adapter is None:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    session = await postgres_adapter.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    if session.get("created_at"):
        session["created_at"] = session["created_at"].isoformat()
    if session.get("updated_at"):
        session["updated_at"] = session["updated_at"].isoformat()
    
    return session


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


@router.get("/maps/{session_id}/metadata", response_model=MapMetadata)
async def get_map_metadata(session_id: str):
    if postgres_adapter is None:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    session = await postgres_adapter.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    num_keyframes = 0
    db_path = settings.MAPS_DIR / f"{session_id}.db"
    if db_path.exists():
        try:
            parser = DatabaseParser()
            parsed = await parser.parse_database(str(db_path))
            num_keyframes = parsed.get('num_keyframes', 0)
        except Exception as e:
            logger.warning(f"[SLAM-METADATA] Failed to parse database: {e}")
    
    created_at = session["created_at"].isoformat() if session.get("created_at") else ""
    
    return MapMetadata(
        map_id=session_id,
        session_id=session_id,
        num_keyframes=num_keyframes,
        created_at=created_at,
        status=session.get("status", "")
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
