"""SuperPoint + LightGlue fused ONNX 추론 래퍼 (Sprint 20).

단일 ONNX graph로 (imgA, imgB) pair 입력 → (keypointsA, keypointsB, matches, scores) 반환.
provider auto-select는 DepthAnythingV2Runner 패턴을 mirror: CoreML(Darwin) → CUDA(Linux GPU) → CPU.

모델 I/O (thomasonzhou/superpoint-lightglue):
    input  images:    (B=2, 1, H, W) float32 grayscale [0, 1]
    output keypoints: (B=2, K, 2) int64 (x, y) — K는 입력 이미지에 따라 가변, 최대 1024
    output matches:   (N, 3) int64 (batch_idx, kp_idx_0, kp_idx_1)
                       batch_idx는 항상 0 (pair 1개만 넣으므로)
                       kp_idx_0 == image0 keypoint, kp_idx_1 == image1 keypoint
    output mscores:   (N,) float32
"""
from __future__ import annotations

import asyncio
import logging
import platform
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def default_providers() -> list[Any]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime 패키지가 설치되지 않았습니다.") from exc

    available = ort.get_available_providers()
    if "CoreMLExecutionProvider" in available and platform.system() == "Darwin":
        logger.info("superpoint_lightglue: using CoreMLExecutionProvider")
        return [
            ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}),
            "CPUExecutionProvider",
        ]
    if "CUDAExecutionProvider" in available:
        logger.info("superpoint_lightglue: using CUDAExecutionProvider")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    logger.info("superpoint_lightglue: using CPUExecutionProvider")
    return ["CPUExecutionProvider"]


class MatchedPair:
    """Pair match 결과. 추적/디버깅 용이."""

    __slots__ = ("kpts_a", "kpts_b", "matches", "scores", "size_a", "size_b")

    def __init__(
        self,
        kpts_a: np.ndarray,
        kpts_b: np.ndarray,
        matches: np.ndarray,
        scores: np.ndarray,
        size_a: tuple[int, int],
        size_b: tuple[int, int],
    ) -> None:
        self.kpts_a = kpts_a  # (Ka, 2) int — model resolution 좌표
        self.kpts_b = kpts_b  # (Kb, 2) int
        self.matches = matches  # (N, 2) int (idx_a, idx_b)
        self.scores = scores  # (N,) float32
        self.size_a = size_a  # (H_model, W_model) — kpts 좌표 스케일
        self.size_b = size_b


class SuperPointLightGlueRunner:
    """SuperPoint+LightGlue fused ONNX 추론 래퍼.

    match(image_a, image_b) → MatchedPair.
    입력 이미지는 grayscale로 변환 후 input_size로 resize.
    Keypoint 좌표는 model input 좌표계 기준 — 원본 해상도로 돌리려면
    호출자가 (x * W_orig / input_size, y * H_orig / input_size) 스케일링 필요.
    """

    def __init__(
        self,
        model_path: Path,
        providers: list[Any] | None = None,
        input_size: int = 384,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime 패키지가 설치되지 않았습니다.") from exc

        _providers = providers if providers is not None else default_providers()
        self._session: Any = ort.InferenceSession(str(model_path), providers=_providers)
        self._input_name: str = self._session.get_inputs()[0].name
        self._input_size = input_size

        # CoreML은 NonZero op이 runtime에서 0 elements를 뱉으면 실패 (zero-match pair).
        # 동일 모델의 CPU-only 세션을 사전 로드해두고 fallback 시 사용.
        needs_fallback = any(
            (p if isinstance(p, str) else p[0]) != "CPUExecutionProvider"
            for p in _providers
        )
        self._cpu_session: Any | None = None
        if needs_fallback:
            try:
                self._cpu_session = ort.InferenceSession(
                    str(model_path), providers=["CPUExecutionProvider"]
                )
            except Exception as e:
                logger.warning("CPU fallback session init failed: %s", e)

        logger.info(
            "SuperPointLightGlueRunner loaded path=%s input_size=%d providers=%s fallback=%s",
            model_path,
            input_size,
            self._session.get_providers(),
            "CPU" if self._cpu_session is not None else "none",
        )

    @property
    def input_size(self) -> int:
        return self._input_size

    async def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchedPair:
        return await asyncio.to_thread(self._match_sync, image_a, image_b)

    def _match_sync(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchedPair:
        s = self._input_size
        a_g = _to_gray_normalized(image_a, s)  # (1, s, s)
        b_g = _to_gray_normalized(image_b, s)
        batch = np.stack([a_g, b_g], axis=0).astype(np.float32)  # (2, 1, s, s)

        try:
            outputs: list[Any] = self._session.run(None, {self._input_name: batch})
        except Exception as e:
            msg = str(e)
            if self._cpu_session is not None and ("zero elements" in msg or "CoreML" in msg):
                # CoreML이 zero-match pair에서 터질 때 CPU 세션으로 재시도
                logger.debug("CoreML match failed, retrying on CPU: %s", msg[:120])
                outputs = self._cpu_session.run(None, {self._input_name: batch})
            else:
                raise
        kpts, matches_raw, mscores = outputs
        # kpts: (2, K, 2) — 두 이미지 공통 K (model 내부에서 padding 처리됨)
        kpts_a = np.asarray(kpts[0], dtype=np.int64)  # (K, 2)
        kpts_b = np.asarray(kpts[1], dtype=np.int64)

        # matches_raw: (N, 3) — (batch_idx, idx_a, idx_b)
        matches_raw = np.asarray(matches_raw, dtype=np.int64)
        if matches_raw.size == 0:
            matches = np.empty((0, 2), dtype=np.int64)
            scores = np.empty((0,), dtype=np.float32)
        else:
            # batch_idx는 pair 1개일 때 모두 0 — 안전하게 필터
            batch_idx = matches_raw[:, 0]
            keep = batch_idx == 0
            matches = matches_raw[keep, 1:]  # (N, 2)
            scores = np.asarray(mscores, dtype=np.float32)[keep]

        return MatchedPair(
            kpts_a=kpts_a,
            kpts_b=kpts_b,
            matches=matches,
            scores=scores,
            size_a=(s, s),
            size_b=(s, s),
        )


def _to_gray_normalized(image: np.ndarray, size: int) -> np.ndarray:
    """(H, W, 3) RGB/BGR uint8 또는 (H, W) gray → (1, size, size) float32 [0,1]."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless 패키지가 필요합니다.") from exc

    if image.ndim == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image[..., 0]
    else:
        gray = image

    resized: np.ndarray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)[np.newaxis, ...]
