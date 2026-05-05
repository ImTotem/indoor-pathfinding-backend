"""PyTorch Dataset wrapping the synthetic generator."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data_generator import ALL_STYLES, H, W, generate_pair


class SyntheticHeatmapDataset(Dataset):
    """Generates (input image, target mask) pairs on-the-fly.

    Input  : 3xHxW float32 in [0, 1] (BGR→RGB).
    Target : 1xHxW float32 in {0, 1} (building footprint).
    """

    def __init__(
        self,
        n_samples: int,
        seed: int = 0,
        styles: list[str] | None = None,
        augment: bool = False,
        cache_dir: Path | None = None,
    ) -> None:
        self.n_samples = int(n_samples)
        self.seed = int(seed)
        self.styles = list(styles) if styles else list(ALL_STYLES)
        self.augment = augment
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return self.n_samples

    def _generate(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        style = self.styles[idx % len(self.styles)]
        for attempt in range(5):
            result = generate_pair(rng, style=style)
            if result is not None:
                canvas, mask, _, _ = result
                return canvas, mask
            rng = np.random.default_rng(self.seed * 1_000_003 + idx * 31 + attempt)
        # ultimate fallback — empty pair
        return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W), dtype=np.uint8)

    def __getitem__(self, idx: int):
        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{idx:06d}.npz"
            if cache_path.exists():
                data = np.load(cache_path)
                canvas, mask = data["canvas"], data["mask"]
            else:
                canvas, mask = self._generate(idx)
                np.savez_compressed(cache_path, canvas=canvas, mask=mask)
        else:
            canvas, mask = self._generate(idx)

        if self.augment:
            canvas, mask = _augment(canvas, mask, np.random.default_rng(idx * 7 + 1))

        # BGR uint8 -> RGB float32 [0,1], CHW
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = np.transpose(rgb, (2, 0, 1))
        target = (mask.astype(np.float32) / 255.0)[None, :, :]
        return torch.from_numpy(rgb), torch.from_numpy(target)


def _augment(
    canvas: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    # Horizontal flip
    if rng.random() < 0.5:
        canvas = canvas[:, ::-1, :].copy()
        mask = mask[:, ::-1].copy()
    # Vertical flip
    if rng.random() < 0.5:
        canvas = canvas[::-1, :, :].copy()
        mask = mask[::-1, :].copy()
    # 90-degree rotation (0/1/2/3)
    k = int(rng.integers(0, 4))
    if k > 0:
        canvas = np.rot90(canvas, k=k).copy()
        mask = np.rot90(mask, k=k).copy()
    # Intensity jitter ±20%
    if rng.random() < 0.5:
        scale = float(rng.uniform(0.8, 1.2))
        canvas = np.clip(canvas.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return canvas, mask
