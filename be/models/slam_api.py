# models/slam_api.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List


class SLAMProcessRequest(BaseModel):
    """POST /api/slam/process request - Enqueue SLAM processing job"""
    session_id: str = Field(..., json_schema_extra={"example": "session_abc123"})

    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        if not v or not v.strip():
            raise ValueError('session_id cannot be empty')
        return v


class SLAMProcessResponse(BaseModel):
    """POST /api/slam/process response - Job enqueued"""
    map_id: str = Field(..., json_schema_extra={"example": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"})
    status: str = Field(..., json_schema_extra={"example": "pending"})
    queue_position: int = Field(..., json_schema_extra={"example": 0})


class SLAMLocalizeRequest(BaseModel):
    """POST /api/slam/localize request - Localize images in map (1-5 images)"""
    map_id: str = Field(..., json_schema_extra={"example": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"})
    images: List[str] = Field(..., json_schema_extra={"example": ["base64_img1", "base64_img2"]})
    camera_intrinsics: dict = Field(..., json_schema_extra={"example": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}})
    
    @field_validator('images')
    @classmethod
    def validate_images(cls, v):
        if not (1 <= len(v) <= 5):
            raise ValueError('Must provide 1-5 images')
        return v


class SLAMLocalizeResponse(BaseModel):
    """POST /api/slam/localize response - Localization result (future stub)"""
    pose: dict = Field(..., json_schema_extra={"example": {"x": 0.0, "y": 0.0, "z": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}})
    confidence: float = Field(..., json_schema_extra={"example": 0.95})


class MapMetadata(BaseModel):
    """GET /api/slam/maps/{map_id}/metadata response - Map information"""
    map_id: str = Field(..., json_schema_extra={"example": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"})
    session_id: str = Field(..., json_schema_extra={"example": "session_abc123"})
    num_keyframes: int = Field(..., json_schema_extra={"example": 150})
    created_at: str = Field(..., json_schema_extra={"example": "2026-02-10T12:00:00Z"})
    status: str = Field(..., json_schema_extra={"example": "success"})


class HealthResponse(BaseModel):
    """GET /api/slam/health response - System health status"""
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    postgres: str = Field(..., json_schema_extra={"example": "connected"})
    queue_length: int = Field(..., json_schema_extra={"example": 5})
