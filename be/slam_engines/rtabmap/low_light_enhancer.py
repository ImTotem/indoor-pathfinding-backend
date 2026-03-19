# slam_engines/rtabmap/low_light_enhancer.py
"""Zero-DCE low-light image enhancer for nighttime localization.

Uses the Zero-Reference Deep Curve Estimation (Zero-DCE) network to
enhance dark query images before feature extraction, improving
relocalization accuracy under low-light conditions.

Reference: Guo et al., "Zero-Reference Deep Curve Estimation for
Low-Light Image Enhancement", CVPR 2020.
"""

import logging
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Images with mean brightness below this threshold are enhanced
_DARK_THRESHOLD = 80

_WEIGHTS_PATH = Path(__file__).parent / "weights" / "zero_dce.pth"

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not available; low-light enhancement disabled")


def _build_model():
    """Build Zero-DCE network (only called when torch is available)."""

    class ZeroDCE(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU(inplace=True)
            nf = 32
            self.e_conv1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            self.e_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.e_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.e_conv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.e_conv5 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
            self.e_conv6 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
            self.e_conv7 = nn.Conv2d(nf * 2, 24, 3, 1, 1, bias=True)

        def forward(self, x):
            x1 = self.relu(self.e_conv1(x))
            x2 = self.relu(self.e_conv2(x1))
            x3 = self.relu(self.e_conv3(x2))
            x4 = self.relu(self.e_conv4(x3))
            x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
            x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
            x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

            # 8 iterations of curve enhancement
            r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
            x = x + r1 * (torch.pow(x, 2) - x)
            x = x + r2 * (torch.pow(x, 2) - x)
            x = x + r3 * (torch.pow(x, 2) - x)
            x = x + r4 * (torch.pow(x, 2) - x)
            x = x + r5 * (torch.pow(x, 2) - x)
            x = x + r6 * (torch.pow(x, 2) - x)
            x = x + r7 * (torch.pow(x, 2) - x)
            x = x + r8 * (torch.pow(x, 2) - x)
            return x

    return ZeroDCE()


class LowLightEnhancer:
    """Singleton low-light image enhancer using Zero-DCE.

    Lazy-loads the model on first use. Falls back gracefully if
    PyTorch is unavailable or model weights are missing.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._model = None
                    inst._device = None
                    inst._ready = False
                    cls._instance = inst
        return cls._instance

    def _load_model(self):
        if not _TORCH_AVAILABLE:
            return

        if not _WEIGHTS_PATH.exists():
            logger.warning(f"Zero-DCE weights not found: {_WEIGHTS_PATH}")
            return

        try:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._model = _build_model()
            self._model.load_state_dict(
                torch.load(_WEIGHTS_PATH, map_location=self._device, weights_only=True)
            )
            self._model.to(self._device)
            self._model.eval()
            self._ready = True
            logger.info(f"Zero-DCE loaded on {self._device}")
        except Exception as e:
            logger.error(f"Failed to load Zero-DCE: {e}")
            self._model = None

    def enhance(self, rgb_image: np.ndarray) -> np.ndarray:
        """Enhance a low-light RGB image. Returns enhanced RGB.

        Skips enhancement if:
        - Model not available
        - Image is already bright enough (mean > _DARK_THRESHOLD)
        """
        mean_brightness = np.mean(
            cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        )
        if mean_brightness > _DARK_THRESHOLD:
            return rgb_image

        if not self._ready:
            with self._lock:
                if not self._ready:
                    self._load_model()
            if not self._ready:
                return rgb_image

        try:
            img = rgb_image.astype(np.float32) / 255.0
            tensor = (
                torch.from_numpy(img.transpose(2, 0, 1))
                .unsqueeze(0)
                .to(self._device)
            )

            with torch.no_grad():
                enhanced = self._model(tensor)

            result = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            result = np.clip(result * 255, 0, 255).astype(np.uint8)

            logger.debug(
                f"Low-light enhanced: brightness {mean_brightness:.0f} → "
                f"{np.mean(cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)):.0f}"
            )
            return result
        except Exception as e:
            logger.error(f"Zero-DCE inference failed: {e}")
            return rgb_image
