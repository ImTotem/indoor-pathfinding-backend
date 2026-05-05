"""Floor segmentation step — keyframe 이미지마다 Segformer 추론."""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from pathlib import Path
from uuid import UUID

import numpy as np
from pydantic import BaseModel, ConfigDict

from indoor_server.infrastructure.ml.protocol import SemanticSegmenter
from indoor_server.infrastructure.ml.segformer_onnx import extract_floor_stair_masks

logger = logging.getLogger(__name__)


class KeyframeRef(BaseModel):
    """파이프라인이 keyframe 하나를 참조하기 위한 VO."""

    model_config = ConfigDict(frozen=True)

    scan_id: UUID
    seq: int
    image_path: str  # storage_root 기준 상대 경로
    tx: float
    ty: float
    tz: float
    pose_matrix: bytes
    rtabmap_node_id: int | None = None


class KeyframeMasks(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    scan_id: UUID
    seq: int
    floor_mask: np.ndarray   # (H, W) bool
    stair_mask: np.ndarray   # (H, W) bool
    tx: float
    ty: float
    tz: float
    pose_matrix: bytes


ProgressFn = Callable[[float], Coroutine[object, object, None]]


class FloorSegmentationStep:
    def __init__(
        self,
        segmenter: SemanticSegmenter,
        storage_root: Path,
    ) -> None:
        self._segmenter = segmenter
        self._storage_root = storage_root

    async def run(
        self,
        keyframes: Sequence[KeyframeRef],
        progress: ProgressFn,
    ) -> AsyncIterator[KeyframeMasks]:
        """
        W-4: async generator로 직접 yield — `async def` + `yield` 패턴.
        호출부: `seg_iter = await seg_step.run(...)` → `async for mask in seg_iter:`
        """
        # coroutine이 async generator를 반환하는 대신 직접 위임한다.
        # pipeline.py는 `await seg_step.run(...)` 후 `async for`로 소비한다.
        return self._stream(keyframes, progress)

    async def _stream(
        self,
        keyframes: Sequence[KeyframeRef],
        progress: ProgressFn,
    ) -> AsyncIterator[KeyframeMasks]:
        total = len(keyframes)
        for i, kf in enumerate(keyframes):
            image = self._load_image(kf.image_path)
            seg_out = await self._segmenter.segment(image)
            floor_mask, stair_mask = extract_floor_stair_masks(seg_out.class_mask)
            yield KeyframeMasks(
                scan_id=kf.scan_id,
                seq=kf.seq,
                floor_mask=floor_mask,
                stair_mask=stair_mask,
                tx=kf.tx,
                ty=kf.ty,
                tz=kf.tz,
                pose_matrix=kf.pose_matrix,
            )
            if total > 0:
                await progress((i + 1) / total)

    def _load_image(self, image_path: str) -> np.ndarray:
        """image_path를 읽어 HxWx3 uint8 RGB ndarray 반환."""
        import cv2

        full_path = self._storage_root / image_path
        bgr = cv2.imread(str(full_path))
        if bgr is None:
            raise FileNotFoundError(f"keyframe image not found: {full_path}")
        rgb: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb
