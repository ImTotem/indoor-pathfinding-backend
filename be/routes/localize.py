# routes/localize.py
from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from typing import List
from PIL import Image
import io

from slam_interface.factory import SLAMEngineFactory
from config.settings import settings

router = APIRouter(prefix="/api", tags=["localize"])

@router.post("/localize")
async def localize(
    map_id: str = Form(...),
    images: List[UploadFile] = File(...)
):
    """Localize current position using 1-5 camera images against existing map.
    
    Request:
        - map_id: ID of the RTABMap database (form field)
        - images: List of 1-5 image files (multipart)
    
    Response:
        - pose: {x, y, z, qx, qy, qz, qw}
        - confidence: float (0.0-1.0)
        - map_id: str (echoed from request)
        - num_matches: int
    """
    
    if not (1 <= len(images) <= 5):
        raise HTTPException(
            status_code=400,
            detail="1개에서 5개 사이의 이미지를 제공해야 합니다"
        )
    
    try:
        image_bytes = []
        for img in images:
            content = await img.read()
            if len(content) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Empty image file: {img.filename}"
                )
            image_bytes.append(content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read images: {str(e)}"
        )
    
    slam_engine = SLAMEngineFactory.create(settings.SLAM_ENGINE_TYPE)
    
    # Extract intrinsics from DB
    db_path = settings.MAPS_DIR / f"{map_id}.db"
    try:
        intrinsics = slam_engine.extract_intrinsics_from_db(str(db_path))
        print(f"[Localize] Extracted intrinsics: {intrinsics}")
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Failed to extract intrinsics from map {map_id}: {str(e)}"
        )
    
    # Get first image resolution
    try:
        first_image = Image.open(io.BytesIO(image_bytes[0]))
        img_width, img_height = first_image.size
        print(f"[Localize] Query image resolution: {img_width}x{img_height}")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read image resolution: {str(e)}"
        )
    
    # Scale intrinsics if resolution differs
    if (img_width, img_height) != (intrinsics['width'], intrinsics['height']):
        print(f"[Localize] Resolution mismatch: DB={intrinsics['width']}x{intrinsics['height']}, Query={img_width}x{img_height}")
        try:
            intrinsics = slam_engine.scale_intrinsics(intrinsics, img_width, img_height)
            print(f"[Localize] Scaled intrinsics: {intrinsics}")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to scale intrinsics: {str(e)}"
            )
    else:
        print(f"[Localize] Resolution matches, no scaling needed")
    
    try:
        result = await slam_engine.localize(map_id, image_bytes, intrinsics=intrinsics)
        return result
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    
    except TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="위치 인식 실패: RTABMap processing timeout (exceeded 30 seconds)"
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail=f"위치 인식 실패: {str(e)}"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"서버 오류: {str(e)}"
        )
