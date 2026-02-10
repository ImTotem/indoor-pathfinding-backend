from pydantic import BaseModel, Field, field_validator
from typing import List


class SLAMProcessRequest(BaseModel):
    session_id: str = Field(..., json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"})

    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('session_id cannot be empty')
        return v.strip()


class SLAMProcessResponse(BaseModel):
    map_id: str
    status: str
    queue_position: int


class SLAMLocalizeRequest(BaseModel):
    map_id: str = Field(..., json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"})
    images: List[str] = Field(..., json_schema_extra={"example": ["base64_img1"]})
    camera_intrinsics: dict = Field(..., json_schema_extra={"example": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}})
    
    @field_validator('images')
    @classmethod
    def validate_images(cls, v: list) -> list:
        if not (1 <= len(v) <= 5):
            raise ValueError('Must provide 1-5 images')
        return v


class SLAMLocalizeResponse(BaseModel):
    pose: dict
    confidence: float


class MapMetadata(BaseModel):
    map_id: str
    session_id: str
    num_keyframes: int
    created_at: str
    status: str


class HealthResponse(BaseModel):
    status: str
    postgres: str
    queue_length: int
