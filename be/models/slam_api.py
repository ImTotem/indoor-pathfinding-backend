from pydantic import BaseModel, Field, field_validator
from typing import List


class SLAMProcessRequest(BaseModel):
    building_id: str = Field(..., json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"})

    @field_validator('building_id')
    @classmethod
    def validate_building_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('building_id cannot be empty')
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
        if len(v) < 1:
            raise ValueError('Must provide at least 1 image')
        return v


class MatchedNode(BaseModel):
    query_index: int
    node_id: int
    match_count: int
    confidence: float
    pose: dict


class SLAMLocalizeResponse(BaseModel):
    pose: dict
    confidence: float
    map_id: str = ""
    matched_node_id: int = 0
    all_matches: List[MatchedNode] = []


class MapMetadata(BaseModel):
    map_id: str
    building_id: str
    num_keyframes: int
    created_at: str
    status: str


class HealthResponse(BaseModel):
    status: str
    postgres: str
    queue_length: int
