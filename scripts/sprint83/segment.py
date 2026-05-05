"""
segment.py — SegFormer-b0 ADE20K floor mask inference.

Floor class: ADE20K id=3 ("floor;flooring").
Device: MPS (Mac) → CPU fallback.
"""

from __future__ import annotations

import time
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

FLOOR_CLASS_IDS = [3]  # ADE20K "floor;flooring"
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_segformer(device: str) -> tuple:
    """Load processor + model. Returns (processor, model, actual_device)."""
    t0 = time.time()
    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    elapsed = time.time() - t0
    print(f"[segment] model loaded on {device} in {elapsed:.1f}s")
    return processor, model


def infer_floor_mask(
    processor,
    model,
    image_path: str,
    device: str,
    floor_class_ids: list[int] = FLOOR_CLASS_IDS,
) -> np.ndarray:
    """
    Returns bool mask (H, W) at original image resolution.
    True = floor pixel.
    """
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size  # PIL: (W, H)

    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.logits shape: (1, num_classes, H/4, W/4) approx
    logits = outputs.logits  # (1, C, h, w)
    # Upsample to original resolution
    logits_up = torch.nn.functional.interpolate(
        logits,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )
    pred = logits_up.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W) int

    mask = np.zeros_like(pred, dtype=bool)
    for cid in floor_class_ids:
        mask |= pred == cid

    return mask


def infer_floor_masks_batch(
    processor,
    model,
    image_paths: list[str],
    device: str,
    batch_size: int = 8,
    floor_class_ids: list[int] = FLOOR_CLASS_IDS,
    verbose: bool = True,
) -> tuple[list[np.ndarray], dict]:
    """
    Batched inference over all image_paths.
    Returns (masks list of bool (H,W), stats dict).
    """
    total = len(image_paths)
    masks = []
    t0 = time.time()
    floor_ratios = []

    for batch_start in range(0, total, batch_size):
        batch_paths = image_paths[batch_start : batch_start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        orig_sizes = [img.size for img in images]  # (W, H)

        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits  # (B, C, h, w)

        for i, (logit_single, (ow, oh)) in enumerate(
            zip(logits, orig_sizes)
        ):
            logit_up = torch.nn.functional.interpolate(
                logit_single.unsqueeze(0),
                size=(oh, ow),
                mode="bilinear",
                align_corners=False,
            )
            pred = logit_up.argmax(dim=1).squeeze(0).cpu().numpy()
            mask = np.zeros_like(pred, dtype=bool)
            for cid in floor_class_ids:
                mask |= pred == cid
            masks.append(mask)
            floor_ratios.append(float(mask.mean()))

        if verbose and (batch_start // batch_size) % 10 == 0:
            elapsed = time.time() - t0
            done = min(batch_start + batch_size, total)
            print(
                f"[segment] {done}/{total} frames, "
                f"{elapsed:.1f}s, {elapsed/max(done,1)*1000:.0f}ms/frame"
            )

    elapsed_total = time.time() - t0
    stats = {
        "total_frames": total,
        "elapsed_s": round(elapsed_total, 2),
        "ms_per_frame": round(elapsed_total / max(total, 1) * 1000, 1),
        "floor_ratio_mean": round(float(np.mean(floor_ratios)), 4),
        "floor_ratio_std": round(float(np.std(floor_ratios)), 4),
        "floor_ratio_min": round(float(np.min(floor_ratios)), 4),
        "floor_ratio_max": round(float(np.max(floor_ratios)), 4),
        "frames_near_zero": int(sum(r < 0.05 for r in floor_ratios)),
        "frames_near_one": int(sum(r > 0.95 for r in floor_ratios)),
        "device": device,
        "batch_size": batch_size,
    }
    print(
        f"[segment] done: {total} frames in {elapsed_total:.1f}s "
        f"({stats['ms_per_frame']}ms/frame), "
        f"floor_ratio mean={stats['floor_ratio_mean']:.3f}"
    )
    return masks, stats
