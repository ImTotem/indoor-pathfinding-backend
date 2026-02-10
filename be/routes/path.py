# routes/path.py
from fastapi import APIRouter

from models.request_models import PathRequest
from models.response_models import PathResponse

router = APIRouter(prefix="/api/path", tags=["path"])

@router.post("/calculate", response_model=PathResponse)
async def calculate_path(request: PathRequest):
    """경로 계산 (A*)"""
    
    # 더미 경로
    start = request.start_position
    end = [start[0] + 5.0, start[1], start[2] + 5.0]
    
    path = []
    for i in range(10):
        t = i / 9.0
        point = [
            start[0] + (end[0] - start[0]) * t,
            start[1] + (end[1] - start[1]) * t,
            start[2] + (end[2] - start[2]) * t,
        ]
        path.append(point)
    
    return PathResponse(
        map_id=request.map_id,
        path=path,
        distance=7.07,
        instruction="5m 직진 후 우회전",
    )

