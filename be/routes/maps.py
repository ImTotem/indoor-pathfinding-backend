# routes/maps.py
import logging
from fastapi import APIRouter, HTTPException
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from config.settings import settings
from slam_engines.rtabmap.database_parser import DatabaseParser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["maps"])

@router.get("/maps")
async def get_maps():
    """List available RTABMap databases for relocalization"""
    # 고정 맵 모드
    if settings.USE_FIXED_MAP:
        fixed_map_path = settings.MAPS_DIR / f"{settings.FIXED_MAP_ID}.db"
        
        if not fixed_map_path.exists():
            logger.warning(f"Fixed map not found: {fixed_map_path}")
            return {"maps": []}
        
        # DB 파싱
        db_parser = DatabaseParser()
        metadata = await db_parser.parse_database(str(fixed_map_path))
        
        # 좌표 로그 출력 (첫 5개 키프레임)
        if metadata.get('keyframes'):
            logger.info(f"Fixed Map: {settings.FIXED_MAP_ID}")
            logger.info(f"Total Keyframes: {len(metadata['keyframes'])}")
            for i, kf in enumerate(metadata['keyframes'][:5]):
                pos = kf.get('position', [0, 0, 0])
                logger.info(f"  Keyframe {kf['id']}: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
        
        return {
            "maps": [{
                "id": settings.FIXED_MAP_ID,
                "name": settings.FIXED_MAP_ID,
                "created_at": "2025-02-02T20:22:40Z",
                "keyframe_count": metadata.get('num_keyframes', 0)
            }]
        }
    
    maps_dir = settings.MAPS_DIR
    
    # Handle directory not existing gracefully
    if not maps_dir.exists():
        return {"maps": []}
    
    maps = []
    db_parser = DatabaseParser()
    
    try:
        # Scan for .db files
        for db_file in sorted(maps_dir.glob("*.db")):
            try:
                map_id = db_file.stem
                
                # Parse created_at from filename (map_YYYYMMDD_HHMMSS pattern)
                created_at = _parse_map_timestamp(map_id)
                
                # Get keyframe count from database
                db_metadata = await db_parser.parse_database(str(db_file))
                keyframe_count = db_metadata.get('num_keyframes', 0)
                
                # Use map_id as name (can be enhanced later with metadata file)
                maps.append({
                    "id": map_id,
                    "name": map_id,
                    "created_at": created_at,
                    "keyframe_count": keyframe_count
                })
            except Exception as e:
                # Skip individual map if parsing fails
                print(f"[Maps] Failed to parse map {db_file.name}: {e}")
                continue
        
        return {"maps": maps}
        
    except PermissionError:
        raise HTTPException(
            status_code=500,
            detail="Failed to list maps: permission denied"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list maps: {str(e)}"
        )


def _parse_map_timestamp(map_id: str) -> str:
    """
    Parse timestamp from map_id (format: map_YYYYMMDD_HHMMSS).
    
    Args:
        map_id: Map identifier (e.g., "map_20250207_143022")
        
    Returns:
        ISO 8601 timestamp string (e.g., "2025-02-07T14:30:22Z")
        Falls back to current time if parsing fails
    """
    try:
        # Remove 'map_' prefix
        timestamp_part = map_id.replace("map_", "")
        
        # Split by underscore to get date and time
        parts = timestamp_part.split("_")
        
        if len(parts) >= 2:
            date_str = parts[0]  # YYYYMMDD
            time_str = parts[1]  # HHMMSS
            
            # Parse to datetime
            dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
            return dt.isoformat() + "Z"
    except Exception:
        pass
    
    # Fallback: use current time
    return datetime.now().isoformat() + "Z"
