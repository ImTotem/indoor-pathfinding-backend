"""View cached NPZ samples as PNG sheets.

Usage:
    # 50 samples → 50 individual PNGs + a stitched sheet
    python view_cache.py --cache-dir cache_v1 --n 50 --out-dir _sheet

    # all 1000 samples (큰 sheet)
    python view_cache.py --cache-dir cache_v1 --n -1 --out-dir _sheet

    # specific indices
    python view_cache.py --cache-dir cache_v1 --indices 0 100 500 999 --out-dir _sheet
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=50,
                        help="number of samples to render (use -1 for all, evenly spaced)")
    parser.add_argument("--indices", type=int, nargs="*", default=None,
                        help="explicit indices to render")
    parser.add_argument("--per-row", type=int, default=5,
                        help="how many sample pairs per sheet row")
    parser.add_argument("--cell-h", type=int, default=200,
                        help="height (px) of each sample row in the sheet")
    args = parser.parse_args()

    cache: Path = args.cache_dir
    if not cache.exists():
        raise SystemExit(f"cache dir not found: {cache}")
    files = sorted(cache.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz under {cache}")
    print(f"[cache] {len(files)} samples in {cache}")

    if args.indices:
        targets = [i for i in args.indices if 0 <= i < len(files)]
    elif args.n == -1:
        targets = list(range(len(files)))
    else:
        n = min(args.n, len(files))
        targets = [round(i * (len(files) - 1) / max(n - 1, 1)) for i in range(n)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in targets:
        npz = np.load(files[idx])
        canvas = npz["canvas"]
        mask = npz["mask"]
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        sep = np.full((canvas.shape[0], 4, 3), 60, dtype=np.uint8)
        pair = np.hstack([canvas, sep, mask_bgr])
        cv2.putText(
            pair,
            f"#{idx:04d} mask={int((mask > 0).sum())}px",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        # individual PNG
        cv2.imwrite(str(args.out_dir / f"{idx:06d}.png"), pair)
        rows.append((idx, pair))

    # Stitched sheet (downscale rows to cell_h, group into per_row columns)
    if rows:
        H = args.cell_h
        scaled = []
        for idx, p in rows:
            scale = H / p.shape[0]
            scaled.append((idx, cv2.resize(p, (int(p.shape[1] * scale), H))))
        max_w = max(p.shape[1] for _, p in scaled)
        scaled = [
            (i, cv2.copyMakeBorder(p, 0, 0, 0, max_w - p.shape[1],
                                    cv2.BORDER_CONSTANT, value=0))
            for i, p in scaled
        ]
        per_row = args.per_row
        # group every per_row row entries side-by-side
        sheet_rows = []
        for r in range(0, len(scaled), per_row):
            chunk = [p for _, p in scaled[r:r + per_row]]
            while len(chunk) < per_row:
                chunk.append(np.zeros_like(chunk[0]))
            sheet_rows.append(np.hstack(chunk))
        max_row_w = max(r.shape[1] for r in sheet_rows)
        sheet_rows = [
            cv2.copyMakeBorder(r, 0, 0, 0, max_row_w - r.shape[1],
                                cv2.BORDER_CONSTANT, value=0)
            for r in sheet_rows
        ]
        sheet = np.vstack(sheet_rows)
        sheet_path = args.out_dir / "_sheet.png"
        cv2.imwrite(str(sheet_path), sheet)
        print(f"[sheet] {sheet_path} ({sheet.shape[1]}×{sheet.shape[0]} px, "
              f"{len(rows)} samples)")
    print(f"[done] wrote {len(targets)} png + sheet to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
