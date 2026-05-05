"""Generate N samples and dump a quick visual sheet to confirm generator works."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from data_generator import ALL_STYLES, generate_pair


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    rows = []
    for i in range(args.n):
        style = ALL_STYLES[i % len(ALL_STYLES)]
        rng = np.random.default_rng(args.seed + i * 41 + 13)
        result = generate_pair(rng, style=style)
        if result is None:
            failures += 1
            continue
        canvas, mask, polygon, _ = result
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        sep = np.full((canvas.shape[0], 4, 3), 60, dtype=np.uint8)
        row = np.hstack([canvas, sep, mask_bgr])
        cv2.putText(row, f"{i:02d} {style} v={len(polygon)}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(args.out_dir / f"{i:02d}_{style}.png"), row)
        rows.append(row)
    if rows:
        # stacked sheet (downscale rows to 200 high)
        sheet_rows = []
        for r in rows:
            scale = 200 / r.shape[0]
            sheet_rows.append(cv2.resize(r, (int(r.shape[1] * scale), 200)))
        max_w = max(r.shape[1] for r in sheet_rows)
        sheet_rows = [
            cv2.copyMakeBorder(r, 0, 0, 0, max_w - r.shape[1], cv2.BORDER_CONSTANT, value=0)
            for r in sheet_rows
        ]
        sheet = np.vstack(sheet_rows)
        cv2.imwrite(str(args.out_dir / "_sheet.png"), sheet)
    print(f"generated {args.n - failures}/{args.n} samples → {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
