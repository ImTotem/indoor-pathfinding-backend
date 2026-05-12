#!/usr/bin/env python3
"""Zero-DCE before/after experiment for v3 localization.

Usage:
    python scripts/zero_dce_v3_experiment.py \
        --dark ./dark.jpg \
        --bright ./bright.jpg \
        --building-id f9693bd8-fbff-41b9-be88-cd4b0af3690a \
        --base-url http://localhost:5000

Outputs:
    response/zero_dce_v3_YYYYMMDD_HHMMSS/
        dark_original.jpg
        dark_zero_dce.jpg
        bright_reference.jpg
        comparison.jpg
        metrics.json
        v3_dark_original.json
        v3_dark_zero_dce.json
        v3_bright_reference.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageOps


def _default_weights_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]  # be2/
    workspace_root = repo_root.parent
    candidates = [
        repo_root / "be/slam_engines/rtabmap/weights/zero_dce.pth",
        workspace_root / "be/slam_engines/rtabmap/weights/zero_dce.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        for r in torch.split(x_r, 3, dim=1):
            x = x + r * (torch.pow(x, 2) - x)
        return x


def load_rgb(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def enhance_zero_dce(img: Image.Image, weights_path: Path, device: torch.device) -> Image.Image:
    if not weights_path.exists():
        raise FileNotFoundError(f"Zero-DCE weights not found: {weights_path}")

    model = ZeroDCE().to(device).eval()
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor).clamp(0, 1)
    out_arr = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(out_arr, mode="RGB")


def image_metrics(img: Image.Image) -> dict[str, float]:
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    return {
        "mean_brightness": float(gray.mean()),
        "std_contrast": float(gray.std()),
        "p05": float(np.percentile(gray, 5)),
        "p50": float(np.percentile(gray, 50)),
        "p95": float(np.percentile(gray, 95)),
        "dark_pixel_ratio_lt_50": float((gray < 50).mean()),
        "bright_pixel_ratio_gt_200": float((gray > 200).mean()),
    }


def make_comparison(dark: Image.Image, enhanced: Image.Image, bright: Image.Image) -> Image.Image:
    target_h = 420
    panels: list[tuple[str, Image.Image]] = [
        ("dark original", dark),
        ("dark + Zero-DCE", enhanced),
        ("bright reference", bright),
    ]
    resized = []
    for title, img in panels:
        w = round(img.width * (target_h / img.height))
        canvas = Image.new("RGB", (w, target_h + 42), "white")
        frame = img.resize((w, target_h), Image.Resampling.LANCZOS)
        canvas.paste(frame, (0, 42))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 12), title, fill=(0, 0, 0))
        resized.append(canvas)

    total_w = sum(img.width for img in resized)
    out = Image.new("RGB", (total_w, target_h + 42), "white")
    x = 0
    for img in resized:
        out.paste(img, (x, 0))
        x += img.width
    return out


def post_v3_localize(
    base_url: str,
    building_id: str,
    image_path: Path,
    timeout: int,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/api/slam/v3/localize"
    with image_path.open("rb") as f:
        response = requests.post(
            url,
            data={"building_id": building_id, "map_id": building_id},
            files=[("images", (image_path.name, f, "image/jpeg"))],
            timeout=timeout,
        )

    result: dict[str, Any] = {
        "status_code": response.status_code,
        "endpoint": url,
        "image": str(image_path),
    }
    try:
        result["body"] = response.json()
    except Exception:
        result["body"] = response.text[:1000]
    return result


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def print_localize_summary(label: str, result: dict[str, Any]) -> None:
    body = result.get("body")
    print(f"\n[{label}] status={result['status_code']}")
    if isinstance(body, dict):
        print(f"  confidence : {body.get('confidence')}")
        print(f"  numMatches : {body.get('numMatches')}")
        print(f"  floorId    : {body.get('floorId')}")
        print(f"  floorLevel : {body.get('floorLevel')}")
        print(f"  pose       : {body.get('pose')}")
    else:
        print(f"  body       : {body}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Zero-DCE and v3 localization")
    parser.add_argument("--dark", required=True, type=Path, help="저조도 입력 이미지")
    parser.add_argument("--bright", required=True, type=Path, help="밝은 기준 이미지")
    parser.add_argument("--building-id", "--map-id", required=True, dest="building_id")
    parser.add_argument("--base-url", default="http://localhost:5000")
    parser.add_argument("--weights", type=Path, default=_default_weights_path())
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--skip-localize", action="store_true")
    args = parser.parse_args()

    out_dir = Path("response") / f"zero_dce_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Output  : {out_dir}")
    print(f"Device  : {device}")
    print(f"Weights : {args.weights}")

    dark = load_rgb(args.dark)
    bright = load_rgb(args.bright)
    enhanced = enhance_zero_dce(dark, args.weights, device)

    dark_path = out_dir / "dark_original.jpg"
    enhanced_path = out_dir / "dark_zero_dce.jpg"
    bright_path = out_dir / "bright_reference.jpg"
    comparison_path = out_dir / "comparison.jpg"

    dark.save(dark_path, quality=95)
    enhanced.save(enhanced_path, quality=95)
    bright.save(bright_path, quality=95)
    make_comparison(dark, enhanced, bright).save(comparison_path, quality=95)

    metrics = {
        "dark_original": image_metrics(dark),
        "dark_zero_dce": image_metrics(enhanced),
        "bright_reference": image_metrics(bright),
    }
    save_json(out_dir / "metrics.json", metrics)

    print("\n[brightness]")
    for name, values in metrics.items():
        print(
            f"  {name:16s} mean={values['mean_brightness']:.2f}, "
            f"std={values['std_contrast']:.2f}, "
            f"dark_ratio={values['dark_pixel_ratio_lt_50']:.3f}"
        )
    print(f"\nSaved comparison: {comparison_path}")

    if args.skip_localize:
        return

    results = {
        "dark_original": post_v3_localize(args.base_url, args.building_id, dark_path, args.timeout),
        "dark_zero_dce": post_v3_localize(args.base_url, args.building_id, enhanced_path, args.timeout),
        "bright_reference": post_v3_localize(args.base_url, args.building_id, bright_path, args.timeout),
    }
    for name, result in results.items():
        save_json(out_dir / f"v3_{name}.json", result)
        print_localize_summary(name, result)


if __name__ == "__main__":
    main()
