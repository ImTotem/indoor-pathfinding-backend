"""Pre-generate synthetic samples to disk so external GPU machines can train
without re-running the (CPU-bound) generator.

Each sample is saved as compressed NPZ:
  {idx:06d}.npz containing 'canvas' (BGR uint8 H×W×3) and 'mask' (uint8 H×W)

Usage:
    python precompute_dataset.py --n 1000 --out-dir cache_v1
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from data_generator import ALL_STYLES, generate_pair


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True, help="number of samples")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--styles", nargs="*", default=None,
                        help="restrict to a subset of style names")
    parser.add_argument("--smear-off-prob", type=float, default=0.0,
                        help="probability of generating a smear-disabled sample "
                             "(augmentation for real-RTABMap-after-filter). "
                             "0.5 mixes 50/50 with smear-on.")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    styles = args.styles if args.styles else list(ALL_STYLES)

    skipped = 0
    failed = 0
    t0 = time.time()
    for i in range(args.n):
        out_path = out_dir / f"{i:06d}.npz"
        if out_path.exists():
            skipped += 1
            continue
        style = styles[i % len(styles)]
        rng = np.random.default_rng(args.seed * 1_000_003 + i)
        # Decide smear-off using a per-sample deterministic flip.
        flip_rng = np.random.default_rng(args.seed * 7919 + i)
        disable_smear = bool(flip_rng.random() < args.smear_off_prob)
        result = None
        for attempt in range(5):
            r = generate_pair(rng, style=style, disable_smear=disable_smear)
            if r is not None:
                result = r
                break
            rng = np.random.default_rng(args.seed * 1_000_003 + i * 31 + attempt)
        if result is None:
            failed += 1
            continue
        canvas, mask, _, _ = result
        np.savez_compressed(
            out_path, canvas=canvas, mask=mask,
            disable_smear=np.asarray([int(disable_smear)], dtype=np.int32),
        )
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (args.n - i - 1) / max(rate, 1e-6)
            print(f"[{i + 1:5d}/{args.n}] {rate:.1f} samples/s · eta {eta:.0f}s")

    elapsed = time.time() - t0
    print(
        f"Done. wrote={args.n - skipped - failed} skipped={skipped} failed={failed} "
        f"elapsed={elapsed:.1f}s out={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
