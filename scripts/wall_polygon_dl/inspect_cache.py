"""Visualize cached NPZ samples as PNG side-by-sides.

Each cached file `000XXX.npz` contains (canvas BGR, mask uint8).
This script writes (canvas | mask) PNGs and an optional grid sheet.

Usage:
    # 20개 random sample을 PNG로 (각 파일 + 1장 sheet)
    python inspect_cache.py --cache-dir <DIR> --n 20 --out-dir <PNG_DIR>

    # 처음 N개 순차
    python inspect_cache.py --cache-dir <DIR> --n 12 --out-dir <PNG_DIR> --mode head

    # 특정 인덱스 1개만
    python inspect_cache.py --cache-dir <DIR> --idx 42 --out-dir <PNG_DIR>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    canvas = np.asarray(data["canvas"], dtype=np.uint8)  # BGR HxWx3
    mask = np.asarray(data["mask"], dtype=np.uint8)  # HxW
    return canvas, mask


def render_pair(canvas: np.ndarray, mask: np.ndarray, label: str) -> np.ndarray:
    h, w = canvas.shape[:2]
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    sep = np.full((h, 4, 3), 60, dtype=np.uint8)
    row = np.hstack([canvas, sep, mask_bgr])
    cv2.putText(
        row, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(row, "INPUT (heatmap)", (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(row, "TARGET (mask)", (w + 14, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (180, 180, 180), 1, cv2.LINE_AA)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=20, help="number of samples to render")
    parser.add_argument("--mode", choices=["random", "head"], default="random")
    parser.add_argument("--idx", type=int, default=None,
                        help="if set, render just this single sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--row-height", type=int, default=240,
                        help="grid sheet row height in pixels")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.cache_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no npz files in {args.cache_dir}")

    if args.idx is not None:
        path = args.cache_dir / f"{args.idx:06d}.npz"
        if not path.exists():
            raise SystemExit(f"{path} not found")
        canvas, mask = load_npz(path)
        out_path = args.out_dir / f"{path.stem}.png"
        cv2.imwrite(str(out_path), render_pair(canvas, mask, path.stem))
        print(f"wrote {out_path}")
        return 0

    if args.mode == "random":
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(files), size=min(args.n, len(files)), replace=False)
        indices.sort()
        chosen = [files[i] for i in indices]
    else:
        chosen = files[: args.n]

    rows = []
    for path in chosen:
        canvas, mask = load_npz(path)
        row = render_pair(canvas, mask, path.stem)
        out_path = args.out_dir / f"{path.stem}.png"
        cv2.imwrite(str(out_path), row)
        # downscale for the grid sheet
        scale = args.row_height / row.shape[0]
        rows.append(cv2.resize(row, (int(row.shape[1] * scale), args.row_height)))

    if rows:
        max_w = max(r.shape[1] for r in rows)
        rows = [
            cv2.copyMakeBorder(r, 0, 0, 0, max_w - r.shape[1],
                               cv2.BORDER_CONSTANT, value=0)
            for r in rows
        ]
        sheet = np.vstack(rows)
        cv2.imwrite(str(args.out_dir / "_sheet.png"), sheet)
        print(f"wrote {len(rows)} sample PNGs + _sheet.png → {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
