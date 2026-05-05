"""Segformer-B0 ADE20K ONNX 추론 래퍼.

onnxruntime import는 이 파일 내부에서만 — 앱 기동 시 강제 로드 금지.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np

from indoor_server.infrastructure.ml.protocol import SegmentationOutput

logger = logging.getLogger(__name__)

# ADE20K 150-class 중 walkable/stair/object class index (0-based)
# ADE20K class list: https://github.com/CSAILVision/ADE20K
_FLOOR_CLASS_IDX = 3      # floor
_RUG_CLASS_IDX = 28       # rug, carpet, ... (NVIDIA SegFormer ADE config)
_STAIRS_CLASS_IDX = 53    # stairs, stairway (0-based: 54-1=53)
_STAIRWAY_CLASS_IDX = 59  # stairway (0-based: 60-1=59)
_OBJECT_AVOID_CLASS_IDXS = frozenset(
    {
        7,   # bed
        10,  # cabinet
        15,  # table
        19,  # chair
        23,  # sofa
        24,  # shelf
        30,  # armchair
        31,  # seat
        33,  # desk
        35,  # wardrobe
        39,  # cushion
        44,  # chest of drawers
        45,  # counter
        50,  # refrigerator
        55,  # case
        56,  # pool table
        57,  # pillow
        62,  # bookcase
        64,  # coffee table
        69,  # bench
        70,  # countertop
        73,  # kitchen island
        75,  # swivel chair
        89,  # television receiver
        97,  # ottoman
        99,  # buffet
        107, # washer
        110, # stool
        117, # cradle
        118, # oven
    }
)

# ADE20K pretrained model 입력 normalization (ImageNet)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INPUT_SIZE = 512


class SegformerOnnxSegmenter:
    """ONNX Runtime 기반 Segformer-B0 ADE20K 추론."""

    def __init__(
        self,
        model_path: Path,
        providers: list[str] | None = None,
    ) -> None:
        # onnxruntime은 이 메서드 내에서만 import
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime 패키지가 설치되지 않았습니다.") from exc

        _providers = providers or ["CPUExecutionProvider"]
        self._session: Any = ort.InferenceSession(str(model_path), providers=_providers)
        self._input_name: str = self._session.get_inputs()[0].name
        logger.info(
            "SegformerOnnxSegmenter loaded path=%s providers=%s",
            model_path,
            _providers,
        )

    async def segment(self, image: np.ndarray) -> SegmentationOutput:
        """image: (H, W, 3) uint8 RGB → SegmentationOutput(class_mask: H×W int32)."""
        result = await asyncio.to_thread(self._infer, image)
        return result

    def _infer(self, image: np.ndarray) -> SegmentationOutput:
        h_orig, w_orig = image.shape[:2]
        preprocessed = self._preprocess(image)
        outputs: list[Any] = self._session.run(None, {self._input_name: preprocessed})
        class_mask = self._postprocess(outputs, h_orig, w_orig)
        return SegmentationOutput(class_mask=class_mask)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """HxWx3 uint8 RGB → 1x3x512x512 float32 (ImageNet normalize)."""
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python-headless 패키지가 필요합니다.") from exc

        resized: np.ndarray = cv2.resize(
            image, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR
        )
        normalized = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
        # HWC → CHW → NCHW
        chw = normalized.transpose(2, 0, 1)
        result: np.ndarray = chw[np.newaxis, ...]
        return result

    def _postprocess(
        self, outputs: list[Any], h_orig: int, w_orig: int
    ) -> np.ndarray:
        """모델 출력(logits) → 원본 해상도 class_mask (H, W) int32."""
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python-headless 패키지가 필요합니다.") from exc

        logits: np.ndarray = outputs[0]  # (1, num_classes, H_out, W_out)
        # argmax → class index
        pred = np.argmax(logits[0], axis=0).astype(np.int32)  # (H_out, W_out)
        if pred.shape != (h_orig, w_orig):
            pred = cv2.resize(
                pred,
                (w_orig, h_orig),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)
        result2: np.ndarray = pred
        return result2


def extract_floor_stair_masks(
    class_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """ADE20K class_mask → (floor_mask, stair_mask) bool ndarray (A-3)."""
    floor_mask = (class_mask == _FLOOR_CLASS_IDX) | (class_mask == _RUG_CLASS_IDX)
    stair_mask = (class_mask == _STAIRS_CLASS_IDX) | (class_mask == _STAIRWAY_CLASS_IDX)
    return floor_mask.astype(bool), stair_mask.astype(bool)


def extract_object_avoid_mask(class_mask: np.ndarray) -> np.ndarray:
    """ADE20K class_mask → object/furniture non-walkable avoid mask."""
    return np.isin(class_mask, tuple(_OBJECT_AVOID_CLASS_IDXS)).astype(bool)
