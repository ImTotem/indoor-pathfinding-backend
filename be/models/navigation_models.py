from pydantic import BaseModel
from typing import List, Optional


class SessionStartRequest(BaseModel):
    user_id: str
    map_id: str
    start: List[float]  # [x, y, z]
    goal: List[float]   # [x, y, z]
    preference: Optional[str] = "SHORTEST"


class SessionStartResponse(BaseModel):
    session_id: str
    status: str
    initial_path: List[List[float]]


class PositionFrameRequest(BaseModel):
    session_id: str
    position: List[float]
    image_base64: Optional[str] = None
    timestamp: Optional[int] = None


class PositionFrameResponse(BaseModel):
    session_id: str
    position: List[float]
    path_index: int
    on_path: bool
    deviation_distance: float
    next_instruction: str
    remaining_path: List[List[float]]
    status: str
    arrival: Optional[bool] = False


class ErrorResponse(BaseModel):
    type: str = "error"
    message: str
    session_id: Optional[str] = None
