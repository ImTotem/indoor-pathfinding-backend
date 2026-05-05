from __future__ import annotations

import numpy as np


class LowLightEnhancer:
    """No-op compatibility class for legacy RTABMap map loading."""

    def enhance(self, image: np.ndarray) -> np.ndarray:
        return image
