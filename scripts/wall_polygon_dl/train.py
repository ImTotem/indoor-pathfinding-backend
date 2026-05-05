"""Train U-Net for synthetic heatmap → polygon-mask segmentation.

Usage (M4 Pro MPS, PoC defaults):
    uv run python train.py --train 900 --val 100 --epochs 30 --batch 4 \
        --out runs/poc_v1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import SyntheticHeatmapDataset
from model import build_unet, select_device


class BinaryFocalLoss(nn.Module):
    """Custom focal loss — works on MPS (smp.losses.FocalLoss currently doesn't)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * target + (1.0 - p) * (1.0 - target)
        loss = self.alpha * (1.0 - pt).pow(self.gamma) * bce
        return loss.mean()


def iou_score(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(logits) >= 0.5).float()
    target = (target >= 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))
    iou = (inter + eps) / (union + eps)
    return float(iou.mean().item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=900)
    parser.add_argument("--val", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--out", type=Path, default=Path("runs/poc_v1"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--resize-to", type=int, default=512,
                        help="Resize 700x700 → NxN to fit memory (must be multiple of 32)")
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"[device] {device}")

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "train": args.train,
                "val": args.val,
                "epochs": args.epochs,
                "batch": args.batch,
                "lr": args.lr,
                "encoder": args.encoder,
                "device": str(device),
                "seed": args.seed,
                "resize_to": args.resize_to,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    train_ds = SyntheticHeatmapDataset(
        n_samples=args.train,
        seed=args.seed,
        augment=True,
        cache_dir=(args.cache_dir / "train") if args.cache_dir else None,
    )
    val_ds = SyntheticHeatmapDataset(
        n_samples=args.val,
        seed=args.seed + 999,
        augment=False,
        cache_dir=(args.cache_dir / "val") if args.cache_dir else None,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    )

    model = build_unet(encoder_name=args.encoder).to(device)
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    focal_loss = BinaryFocalLoss(alpha=0.25, gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    resize_to = args.resize_to
    best_iou = -1.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        train_iou_sum = 0.0
        n_train = 0
        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if resize_to and (imgs.shape[-1] != resize_to):
                imgs = torch.nn.functional.interpolate(
                    imgs, size=(resize_to, resize_to), mode="bilinear", align_corners=False
                )
                masks = torch.nn.functional.interpolate(
                    masks, size=(resize_to, resize_to), mode="nearest"
                )
            optimizer.zero_grad()
            logits = model(imgs)
            loss = 0.5 * dice_loss(logits, masks) + 0.5 * focal_loss(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * imgs.size(0)
            train_iou_sum += iou_score(logits, masks) * imgs.size(0)
            n_train += imgs.size(0)
        scheduler.step()

        model.eval()
        val_loss_sum = 0.0
        val_iou_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                if resize_to and (imgs.shape[-1] != resize_to):
                    imgs = torch.nn.functional.interpolate(
                        imgs, size=(resize_to, resize_to), mode="bilinear", align_corners=False
                    )
                    masks = torch.nn.functional.interpolate(
                        masks, size=(resize_to, resize_to), mode="nearest"
                    )
                logits = model(imgs)
                loss = 0.5 * dice_loss(logits, masks) + 0.5 * focal_loss(logits, masks)
                val_loss_sum += float(loss.item()) * imgs.size(0)
                val_iou_sum += iou_score(logits, masks) * imgs.size(0)
                n_val += imgs.size(0)

        train_loss = train_loss_sum / max(1, n_train)
        train_iou = train_iou_sum / max(1, n_train)
        val_loss = val_loss_sum / max(1, n_val)
        val_iou = val_iou_sum / max(1, n_val)
        elapsed = time.time() - t0
        print(
            f"[ep {epoch:03d}/{args.epochs}] "
            f"train loss={train_loss:.4f} iou={train_iou:.3f} | "
            f"val loss={val_loss:.4f} iou={val_iou:.3f} | "
            f"{elapsed:.1f}s lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_iou": train_iou,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": elapsed,
            }
        )
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "encoder": args.encoder,
                    "resize_to": resize_to,
                    "val_iou": val_iou,
                },
                out_dir / "best.pt",
            )
            print(f"  → saved best.pt val_iou={val_iou:.3f}")

    print(f"Done. best val IoU={best_iou:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
