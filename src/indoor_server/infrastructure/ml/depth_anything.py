"""Depth Anything v2 ONNX 추론 래퍼.

onnxruntime import는 __init__ 내부에서만 — 앱 기동 강제 로드 금지.
M4 Mac은 CoreMLExecutionProvider, Linux 서버는 CPUExecutionProvider fallback.
"""
from __future__ import annotations

import asyncio
import logging
import platform
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Depth Anything v2 입력 normalize (ImageNet)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def default_providers() -> list[Any]:
    """플랫폼별 provider 자동 선택."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime 패키지가 설치되지 않았습니다.") from exc

    available = ort.get_available_providers()
    if "CoreMLExecutionProvider" in available and platform.system() == "Darwin":
        logger.info("depth_anything: using CoreMLExecutionProvider")
        return [
            ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}),
            "CPUExecutionProvider",
        ]
    if "CUDAExecutionProvider" in available:
        logger.info("depth_anything: using CUDAExecutionProvider")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    logger.info("depth_anything: using CPUExecutionProvider")
    return ["CPUExecutionProvider"]


class DepthAnythingV2Runner:
    """Depth Anything v2 ONNX 추론 래퍼.

    estimate() 반환: (H, W) float32 relative inverse depth.
    값이 클수록 카메라에 가까움 (affine-invariant). Metric 변환은 ScaleCalibrator 책임.
    """

    def __init__(
        self,
        model_path: Path,
        providers: list[Any] | None = None,
        input_size: int = 518,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime 패키지가 설치되지 않았습니다.") from exc

        _providers = providers if providers is not None else default_providers()
        self._session: Any = ort.InferenceSession(str(model_path), providers=_providers)
        self._input_name: str = self._session.get_inputs()[0].name
        self._input_size = input_size
        logger.info(
            "DepthAnythingV2Runner loaded path=%s input_size=%d",
            model_path,
            input_size,
        )

    async def estimate(self, image: np.ndarray) -> np.ndarray:
        """image: (H, W, 3) uint8 RGB → (H, W) float32 relative inverse depth."""
        return await asyncio.to_thread(self._infer, image)

    async def estimate_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """배치 추론 — 1차 구현은 단건 루프."""
        results: list[np.ndarray] = []
        for img in images:
            results.append(await self.estimate(img))
        return results

    def _infer(self, image: np.ndarray) -> np.ndarray:
        h_orig, w_orig = image.shape[:2]
        inp = self._preprocess(image)
        outputs: list[Any] = self._session.run(None, {self._input_name: inp})
        return self._postprocess(outputs, h_orig, w_orig)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 RGB → (1, 3, input_size, input_size) float32 ImageNet-normalized."""
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python-headless 패키지가 필요합니다.") from exc

        s = self._input_size
        resized: np.ndarray = cv2.resize(image, (s, s), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
        chw = normalized.transpose(2, 0, 1)
        return chw[np.newaxis, ...]  # (1, 3, s, s)

    def _postprocess(
        self,
        outputs: list[Any],
        h_orig: int,
        w_orig: int,
    ) -> np.ndarray:
        """모델 출력 → 원본 해상도 (H, W) float32 relative depth."""
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python-headless 패키지가 필요합니다.") from exc

        raw: np.ndarray = outputs[0]
        # onnx-community 모델 출력: (1, H, W) 또는 (H, W)
        if raw.ndim == 3:
            raw = raw[0]
        depth = raw.astype(np.float32)

        if depth.shape != (h_orig, w_orig):
            depth = cv2.resize(
                depth, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)

        return depth
