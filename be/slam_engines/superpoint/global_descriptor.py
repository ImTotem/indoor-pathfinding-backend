"""DINOv2-based global descriptor for visual place recognition.

Replaces mean-pooled SuperPoint descriptors with DINOv2 CLS token features,
which are specifically designed for semantic image retrieval and significantly
outperform mean SuperPoint for top-K candidate selection.
"""
import threading

import numpy as np
import torch
from PIL import Image

from utils import logger

_MODEL_NAME = 'dinov2_vits14'  # 384-dim, fast; upgrade to dinov2_vitb14 (768-dim) for more accuracy
_IMG_SIZE = 224


class GlobalDescExtractor:
    """Singleton DINOv2 extractor. First caller sets the device."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, device: torch.device = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._setup(device or torch.device('cpu'))
                    cls._instance = inst
        return cls._instance

    def _setup(self, device: torch.device):
        import torchvision.transforms as T

        self._device = device
        logger.info(f"[DINOv2] Loading {_MODEL_NAME} on {device} ...")
        self._model = torch.hub.load(
            'facebookresearch/dinov2', _MODEL_NAME, verbose=False
        )
        self._model.eval().to(device)
        self._transform = T.Compose([
            T.Resize(_IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(_IMG_SIZE),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        logger.info(f"[DINOv2] Ready (dim=384)")

    @property
    def dim(self) -> int:
        return 384

    def extract(self, gray_uint8: np.ndarray) -> torch.Tensor:
        """Return 384-dim L2-normalised descriptor (CPU tensor) from a grayscale uint8 image."""
        pil = Image.fromarray(gray_uint8).convert('RGB')
        tensor = self._transform(pil).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feat = self._model(tensor)   # (1, 384)
        feat = feat[0].cpu()
        feat = feat / (feat.norm() + 1e-8)
        return feat
