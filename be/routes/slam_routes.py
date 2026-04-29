import base64
import io
import functools
import asyncio
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, status
from PIL import Image, ImageOps

from models.slam_api import (
    SLAMProcessRequest,
    SLAMProcessResponse,
    SLAMLocalizeRequest,
    SLAMLocalizeResponse,
    MapMetadata,
    HealthResponse,
    MaskDebugRequest,
    MaskDebugResponse,
    MaskDebugImage,
    MatchDebugResponse,
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
        # Spring Boot stores relative paths like ./storage/uploads/UUID.db
        # In the web container, uploads are mounted at /app/storage/uploads/
        if file_path.startswith("./storage/uploads/") or file_path.startswith("storage/uploads/"):
            filename = file_path.split("/")[-1]
            file_path = f"/app/storage/uploads/{filename}"
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


async def _localize_impl(request: SLAMLocalizeRequest, mask_persons: bool = False) -> SLAMLocalizeResponse:
    """Core localization logic shared by v1 and v2 endpoints."""
    import asyncio

    building_id = request.map_id
    logger.info(f"[SLAM-LOCALIZE] building_id: {building_id}, mask_persons: {mask_persons}")

    # --- decode images ---
    try:
        image_bytes_list = []
        for i, img_b64 in enumerate(request.images):
            try:
                image_bytes_list.append(base64.b64decode(img_b64))
            except Exception as e:
                raise ValueError(f"Invalid base64 in image {i+1}: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # --- discover floor maps ---
    floor_maps = []
    if postgres_adapter is not None:
        floor_maps = await postgres_adapter.get_floor_maps(building_id)

    if not floor_maps:
        single_db = settings.MAPS_DIR / f"{building_id}.db"
        if single_db.exists():
            floor_maps = [{"floor_id": "", "floor_name": "", "level": 0, "file_path": str(single_db)}]
        else:
            raise HTTPException(status_code=404, detail=f"No maps found for building {building_id}")

    # --- resolve DB paths ---
    slam_engine = SLAMEngineFactory.create(settings.SLAM_ENGINE_TYPE)

    resolved_floors = []
    for fm in floor_maps:
        fp = fm["file_path"]
        if fp.startswith("./storage/uploads/") or fp.startswith("storage/uploads/"):
            filename = fp.split("/")[-1]
            fp = f"/app/storage/uploads/{filename}"
        resolved_floors.append({**fm, "file_path": fp})

    # --- extract intrinsics from first available DB ---
    intrinsics = None
    for fm in resolved_floors:
        try:
            intrinsics = slam_engine.extract_intrinsics_from_db(fm["file_path"])
            break
        except Exception:
            continue

    if intrinsics is None:
        raise HTTPException(status_code=500, detail="Failed to extract intrinsics from any floor DB")

    # Resize query images to DB resolution for feature matching at the correct scale
    db_w, db_h = intrinsics['width'], intrinsics['height']
    resized = []
    for img_bytes in image_bytes_list:
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes)))
            if img.size != (db_w, db_h):
                img = img.resize((db_w, db_h), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=95)
            resized.append(buf.getvalue())
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
    image_bytes_list = resized

    # --- localize against all floors in parallel ---
    async def _localize_floor(fm: dict) -> dict:
        try:
            result = await slam_engine.localize(
                fm["floor_id"], image_bytes_list,
                intrinsics=intrinsics, db_path=fm["file_path"],
                mask_persons=mask_persons,
            )
            result["floor_id"] = fm["floor_id"]
            result["floor_name"] = fm["floor_name"]
            result["floor_level"] = fm["level"]
            return result
        except (FileNotFoundError, ValueError) as e:
            logger.debug(f"[SLAM-LOCALIZE] Floor {fm['floor_name']}: {e}")
            return None
        except Exception as e:
            logger.warning(f"[SLAM-LOCALIZE] Floor {fm['floor_name']} error: {e}")
            return None

    results = await asyncio.gather(*[_localize_floor(fm) for fm in resolved_floors])
    valid = [r for r in results if r is not None]

    if not valid:
        raise HTTPException(status_code=503, detail="Localization failed on all floors")

    best = max(valid, key=lambda r: (r["confidence"], r.get("num_matches", 0)))

    logger.info(
        f"[SLAM-LOCALIZE] Best: floor={best.get('floor_name')}, "
        f"confidence={best['confidence']:.2f}, matches={best.get('num_matches', 0)}"
    )

    return SLAMLocalizeResponse(
        pose=best["pose"],
        confidence=best["confidence"],
        mapId=building_id,
        numMatches=best.get("num_matches", 0),
        matchedImageIndex=best.get("matched_image_index", 0),
        floorId=best.get("floor_id", ""),
        floorLevel=best.get("floor_level", 0),
    )


@router.post("/localize", response_model=SLAMLocalizeResponse, status_code=status.HTTP_200_OK)
async def localize_in_map(request: SLAMLocalizeRequest):
    """Localize against all floors of a building in parallel."""
    return await _localize_impl(request, mask_persons=False)


@router.post("/v2/localize", response_model=SLAMLocalizeResponse, status_code=status.HTTP_200_OK)
async def localize_in_map_v2(request: SLAMLocalizeRequest):
    """Localize with YOLO-based person masking for improved accuracy in crowded spaces."""
    return await _localize_impl(request, mask_persons=True)


@router.post("/v2/debug/mask", response_model=MaskDebugResponse, status_code=status.HTTP_200_OK)
async def debug_person_mask(request: MaskDebugRequest):
    """Return original images and YOLO-annotated versions for debugging person masking.

    For each input image, returns:
    - original_b64: the original image re-encoded as JPEG
    - annotated_b64: same image with green bounding boxes drawn around detected persons
    - persons_detected: number of persons found

    Intended for before/after comparison and PPT slides.
    """
    from slam_engines.rtabmap.person_masker import PersonMasker

    masker = PersonMasker()
    loop = asyncio.get_event_loop()
    results = []

    for i, img_b64 in enumerate(request.images):
        try:
            img_bytes = base64.b64decode(img_b64)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid base64 in image {i + 1}")

        bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail=f"Cannot decode image {i + 1}")

        boxes = await loop.run_in_executor(None, functools.partial(masker.detect_boxes, img_bytes))

        # annotated: person regions blacked out (mirrors what localize actually does)
        annotated = bgr.copy()
        for (x1, y1, x2, y2) in boxes:
            annotated[y1:y2, x1:x2] = 0

        _, orig_buf = cv2.imencode(".jpg", bgr)
        _, ann_buf = cv2.imencode(".jpg", annotated)

        results.append(MaskDebugImage(
            index=i,
            original_b64=base64.b64encode(orig_buf.tobytes()).decode(),
            annotated_b64=base64.b64encode(ann_buf.tobytes()).decode(),
            persons_detected=len(boxes),
        ))

    return MaskDebugResponse(total_images=len(results), results=results)


@router.post("/v2/debug/matches", response_model=MatchDebugResponse, status_code=status.HTTP_200_OK)
async def debug_matches(request: SLAMLocalizeRequest):
    """Visualize feature matches between the query image and the best-matched DB keyframe.

    Returns:
    - query_b64: query image with detected keypoints
    - db_frame_b64: matched DB keyframe image (if stored in DB)
    - matches_b64: side-by-side image with match lines
    """
    from slam_engines.rtabmap.match_debugger import visualize_matches

    building_id = request.map_id

    try:
        img_bytes = base64.b64decode(request.images[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")

    # --- discover floor maps (same as _localize_impl) ---
    floor_maps = []
    if postgres_adapter is not None:
        floor_maps = await postgres_adapter.get_floor_maps(building_id)
    if not floor_maps:
        single_db = settings.MAPS_DIR / f"{building_id}.db"
        if single_db.exists():
            floor_maps = [{"floor_id": "", "floor_name": "", "level": 0, "file_path": str(single_db)}]
        else:
            raise HTTPException(status_code=404, detail=f"No maps found for building {building_id}")

    # --- resolve DB paths ---
    slam_engine = SLAMEngineFactory.create(settings.SLAM_ENGINE_TYPE)
    resolved_floors = []
    for fm in floor_maps:
        fp = fm["file_path"]
        if fp.startswith("./storage/uploads/") or fp.startswith("storage/uploads/"):
            fp = f"/app/storage/uploads/{fp.split('/')[-1]}"
        resolved_floors.append({**fm, "file_path": fp})

    # --- resize query image to DB resolution ---
    intrinsics = None
    for fm in resolved_floors:
        try:
            intrinsics = slam_engine.extract_intrinsics_from_db(fm["file_path"])
            break
        except Exception:
            continue
    if intrinsics is None:
        raise HTTPException(status_code=500, detail="Failed to extract intrinsics")

    db_w, db_h = intrinsics["width"], intrinsics["height"]
    pil_img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes)))
    if pil_img.size != (db_w, db_h):
        pil_img = pil_img.resize((db_w, db_h), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    img_bytes = buf.getvalue()

    # --- run visualize_matches on each floor, keep best ---
    best_result = None
    best_floor = None
    for fm in resolved_floors:
        map_id_key = fm["floor_id"] or building_id
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                functools.partial(visualize_matches, fm["file_path"], map_id_key, img_bytes),
            )
            if best_result is None or result["num_node_matches"] > best_result["num_node_matches"]:
                best_result = result
                best_floor = fm
        except Exception as e:
            logger.debug(f"[DEBUG-MATCHES] Floor {fm.get('floor_name', '')}: {e}")

    if best_result is None:
        raise HTTPException(status_code=503, detail="No matches found on any floor")

    def _enc(bgr: np.ndarray) -> str:
        _, buf = cv2.imencode(".jpg", bgr)
        return base64.b64encode(buf.tobytes()).decode()

    return MatchDebugResponse(
        query_b64=_enc(best_result["query_bgr"]),
        matches_b64=_enc(best_result["vis_bgr"]),
        db_frame_b64=_enc(best_result["db_bgr"]) if best_result["has_db_image"] else None,
        best_node_id=best_result["best_node_id"],
        num_good_matches=best_result["num_good_matches"],
        num_node_matches=best_result["num_node_matches"],
        floor_id=best_floor.get("floor_id", ""),
        floor_name=best_floor.get("floor_name", ""),
        has_db_image=best_result["has_db_image"],
    )
