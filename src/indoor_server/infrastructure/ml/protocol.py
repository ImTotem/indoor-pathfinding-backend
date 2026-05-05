"""SemanticSegmenter Protocol — onnxruntime import는 구현체 모듈로 국한."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict


class SegmentationOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    class_mask: np.ndarray  # (H, W) int32 — ADE20K class index


@runtime_checkable
class SemanticSegmenter(Protocol):
    async def segment(self, image: np.ndarray) -> SegmentationOutput:
        """image: HxWx3 uint8 RGB. 반환: HxW int32 class_id (ADE20K)."""
        ...
