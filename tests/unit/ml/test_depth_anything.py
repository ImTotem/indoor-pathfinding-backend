"""DepthAnythingV2Runner 단위 테스트 — dummy ONNX 세션 mock."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from indoor_server.infrastructure.ml.depth_anything import DepthAnythingV2Runner

import numpy as np
import pytest


class TestDefaultProviders:
    """default_providers() 플랫폼별 반환 검증."""

    def test_darwin_with_coreml_returns_coreml(self) -> None:
        from indoor_server.infrastructure.ml.depth_anything import default_providers

        with (
            patch("platform.system", return_value="Darwin"),
            patch(
                "onnxruntime.get_available_providers",
                return_value=["CoreMLExecutionProvider", "CPUExecutionProvider"],
            ),
        ):
            providers = default_providers()
        assert any(
            (isinstance(p, tuple) and p[0] == "CoreMLExecutionProvider")
            or p == "CoreMLExecutionProvider"
            for p in providers
        )

    def test_linux_without_coreml_returns_cpu(self) -> None:
        from indoor_server.infrastructure.ml.depth_anything import default_providers

        with (
            patch("platform.system", return_value="Linux"),
            patch(
                "onnxruntime.get_available_providers",
                return_value=["CPUExecutionProvider"],
            ),
        ):
            providers = default_providers()
        assert "CPUExecutionProvider" in providers
        assert not any(
            (isinstance(p, tuple) and p[0] == "CoreMLExecutionProvider")
            or p == "CoreMLExecutionProvider"
            for p in providers
        )

    def test_cuda_available_returns_cuda(self) -> None:
        from indoor_server.infrastructure.ml.depth_anything import default_providers

        with (
            patch("platform.system", return_value="Linux"),
            patch(
                "onnxruntime.get_available_providers",
                return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
            ),
        ):
            providers = default_providers()
        assert "CUDAExecutionProvider" in providers


class TestDepthAnythingV2Runner:
    """DepthAnythingV2Runner — ONNX session mock으로 shape/normalize 검증."""

    def _make_runner(self, h_out: int = 518, w_out: int = 518) -> DepthAnythingV2Runner:
        from indoor_server.infrastructure.ml.depth_anything import DepthAnythingV2Runner

        # 더미 ONNX 세션 — 입력 수신 + 출력 반환
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "pixel_values"
        mock_session.get_inputs.return_value = [mock_input]

        # 출력: (1, H, W) float32
        fake_depth = np.ones((1, h_out, w_out), dtype=np.float32) * 0.5
        mock_session.run.return_value = [fake_depth]

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            runner = DepthAnythingV2Runner(
                model_path=Path("dummy.onnx"),
                providers=["CPUExecutionProvider"],
            )

        runner._session = mock_session
        return runner

    @pytest.mark.asyncio
    async def test_estimate_returns_correct_shape(self) -> None:
        """estimate() 반환이 원본 해상도 (H, W) float32."""
        runner = self._make_runner()
        h, w = 480, 640
        image = np.zeros((h, w, 3), dtype=np.uint8)
        result = await runner.estimate(image)
        assert result.shape == (h, w)
        assert result.dtype == np.float32

    @pytest.mark.asyncio
    async def test_estimate_input_shape_is_518x518(self) -> None:
        """_preprocess가 518×518 1×3×518×518 입력을 ONNX에 전달."""
        runner = self._make_runner()
        h, w = 240, 320
        image = np.zeros((h, w, 3), dtype=np.uint8)

        captured: list[dict[str, np.ndarray]] = []

        def mock_run(output_names: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            captured.append(feed)
            return [np.ones((1, 518, 518), dtype=np.float32)]

        runner._session.run = mock_run
        await runner.estimate(image)

        assert len(captured) == 1
        inp = captured[0]["pixel_values"]
        assert inp.shape == (1, 3, 518, 518), f"expected (1,3,518,518) got {inp.shape}"

    @pytest.mark.asyncio
    async def test_preprocess_imagenet_normalization(self) -> None:
        """흰 이미지(255) normalize → (255/255 - mean) / std."""
        runner = self._make_runner()
        white_img = np.full((10, 10, 3), 255, dtype=np.uint8)
        preprocessed = runner._preprocess(white_img)  # (1, 3, 518, 518)
        imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        expected = (1.0 - imagenet_mean) / imagenet_std  # 채널별 기댓값
        for c in range(3):
            np.testing.assert_allclose(preprocessed[0, c, 0, 0], expected[c], atol=1e-4)

    @pytest.mark.asyncio
    async def test_estimate_batch_returns_list(self) -> None:
        """estimate_batch는 list[ndarray] 반환."""
        runner = self._make_runner()
        images = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
        results = await runner.estimate_batch(images)
        assert len(results) == 3
        for r in results:
            assert r.shape == (64, 64)
