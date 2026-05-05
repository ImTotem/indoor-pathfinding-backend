"""SuperPoint + LightGlue fused ONNX 추론 latency 벤치마크.

목적:
    Sprint 20 multi-view scale calibration 도입 전, M4/CUDA 환경별로
    pair당 inference latency / keypoint 수 / match 수를 실측한다.

사용:
    # synthetic gradient image로 각 provider 벤치 (warmup 3 + measure 20)
    uv run python scripts/bench_superpoint_lightglue.py

    # 실제 keyframe 디렉터리에서 인접 pair 샘플링해 벤치
    uv run python scripts/bench_superpoint_lightglue.py --keyframes var/storage/scans/<scan_id>/keyframes

    # 특정 provider만 (GPU 서버에서 CUDA만 돌리고 싶을 때)
    uv run python scripts/bench_superpoint_lightglue.py --providers CUDAExecutionProvider

결과 해석:
    • mean/p50/p95 latency per pair (ms)
    • keypoints per image / matches per pair 평균
    • 227 keyframe × window=5 가정 시 예상 총 소요 시간
"""
from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indoor_server.config import settings

_MODEL_NAME = "superpoint_lightglue.onnx"


def _available_providers() -> list[str]:
    import onnxruntime as ort

    return list(ort.get_available_providers())


def _provider_config(name: str) -> Any:
    """Provider name → onnxruntime provider config tuple.

    CoreML: ANE 우선(ALL), CUDA: default options, CPU: string.
    """
    if name == "CoreMLExecutionProvider":
        return (name, {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"})
    return name


def _load_session(model_path: Path, provider: str) -> Any:
    import onnxruntime as ort

    providers: list[Any] = [_provider_config(provider)]
    if provider != "CPUExecutionProvider":
        providers.append("CPUExecutionProvider")  # fallback 연쇄
    return ort.InferenceSession(str(model_path), providers=providers)


def _prepare_gray(image: np.ndarray, size: int) -> np.ndarray:
    """(H, W, 3) uint8 RGB 또는 (H, W) uint8 gray → (1, size, size) float32 [0,1]."""
    import cv2

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)[np.newaxis, ...]


def _synthetic_pair(size: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """gradient + noise 기반 synthetic pair. 두 번째 이미지는 약간 shift."""
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 255, size * size, dtype=np.float32).reshape(size, size)
    base = base + rng.normal(0, 10, base.shape).astype(np.float32)
    base = np.clip(base, 0, 255).astype(np.uint8)
    shifted = np.roll(base, shift=8, axis=1)
    return base, shifted


def _load_real_pairs(keyframe_dir: Path, size: int, n_pairs: int, stride: int) -> list[tuple[np.ndarray, np.ndarray]]:
    import cv2

    paths = sorted(keyframe_dir.glob("*.jpg")) + sorted(keyframe_dir.glob("*.png"))
    if len(paths) < stride + 1:
        raise RuntimeError(f"keyframe too few: {len(paths)} in {keyframe_dir}")

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(min(n_pairs, len(paths) - stride)):
        a = cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(str(paths[i + stride]), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            continue
        pairs.append((a, b))
    return pairs


def _bench_provider(
    model_path: Path,
    provider: str,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    input_size: int,
    warmup: int,
    measure: int,
) -> dict[str, Any]:
    try:
        session = _load_session(model_path, provider)
    except Exception as e:
        return {"provider": provider, "error": str(e)}

    input_name = session.get_inputs()[0].name

    def _run(pair: tuple[np.ndarray, np.ndarray]) -> tuple[float, int, int]:
        a_g = _prepare_gray(pair[0], input_size)
        b_g = _prepare_gray(pair[1], input_size)
        batch = np.stack([a_g, b_g], axis=0)  # (2, 1, H, W)

        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: batch})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        kpts, matches, _mscores = outputs
        kp_count = int(kpts.shape[1]) if kpts.ndim == 3 else 0
        match_count = int(matches.shape[0]) if matches.ndim == 2 else 0
        return elapsed_ms, kp_count, match_count

    # warmup
    for i in range(warmup):
        _run(pairs[i % len(pairs)])

    latencies: list[float] = []
    kp_counts: list[int] = []
    match_counts: list[int] = []
    for i in range(measure):
        ms, kp, mc = _run(pairs[i % len(pairs)])
        latencies.append(ms)
        kp_counts.append(kp)
        match_counts.append(mc)

    latencies_sorted = sorted(latencies)
    return {
        "provider": provider,
        "actual_providers": session.get_providers(),
        "mean_ms": statistics.fmean(latencies),
        "p50_ms": latencies_sorted[len(latencies_sorted) // 2],
        "p95_ms": latencies_sorted[int(len(latencies_sorted) * 0.95)],
        "kp_mean": statistics.fmean(kp_counts),
        "match_mean": statistics.fmean(match_counts),
        "measure_n": measure,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframes", type=Path, default=None, help="실제 keyframe 디렉터리 (없으면 synthetic)")
    parser.add_argument("--pairs", type=int, default=20, help="측정할 pair 수")
    parser.add_argument("--stride", type=int, default=1, help="인접 pair stride (keyframe 간격)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--input-size", type=int, default=None, help="default: settings.superpoint_input_size")
    parser.add_argument(
        "--providers",
        type=str,
        default=None,
        help="comma-separated provider 화이트리스트 (예: CUDAExecutionProvider,CPUExecutionProvider)",
    )
    args = parser.parse_args()

    model_path = settings.model_cache_dir / _MODEL_NAME
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}\n  먼저 `uv run python scripts/fetch_superpoint_lightglue.py` 실행", file=sys.stderr)
        sys.exit(1)

    input_size = args.input_size or settings.superpoint_input_size

    print("=" * 72)
    print(f"host: {platform.platform()}")
    print(f"model: {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")
    print(f"input_size: {input_size}")

    if args.keyframes is not None:
        print(f"pairs: real from {args.keyframes} stride={args.stride}")
        pairs = _load_real_pairs(args.keyframes, input_size, args.pairs, args.stride)
    else:
        print("pairs: synthetic gradient")
        pairs = [_synthetic_pair(input_size, seed=i) for i in range(max(args.pairs, args.warmup + 1))]
    print(f"loaded {len(pairs)} pair(s)")
    print()

    available = _available_providers()
    if args.providers:
        candidates = [p.strip() for p in args.providers.split(",") if p.strip()]
    else:
        # darwin: CoreML 우선, linux+GPU: CUDA, 마지막 CPU
        candidates = []
        if platform.system() == "Darwin" and "CoreMLExecutionProvider" in available:
            candidates.append("CoreMLExecutionProvider")
        if "CUDAExecutionProvider" in available:
            candidates.append("CUDAExecutionProvider")
        candidates.append("CPUExecutionProvider")

    candidates = [c for c in candidates if c in available]
    print(f"available providers: {available}")
    print(f"benchmarking: {candidates}")
    print()

    results: list[dict[str, Any]] = []
    for prov in candidates:
        print(f"-- {prov} --")
        r = _bench_provider(model_path, prov, pairs, input_size, args.warmup, args.pairs)
        results.append(r)
        if "error" in r:
            print(f"  FAILED: {r['error']}")
            continue
        print(f"  actual providers: {r['actual_providers']}")
        print(f"  mean={r['mean_ms']:.1f}ms  p50={r['p50_ms']:.1f}ms  p95={r['p95_ms']:.1f}ms  n={r['measure_n']}")
        print(f"  kp/img={r['kp_mean']:.0f}  matches/pair={r['match_mean']:.0f}")
        print()

    # 227 keyframe × window=5 가정 투영
    print("=" * 72)
    print("projection (227 keyframe × window=5 = 1135 pair):")
    for r in results:
        if "error" in r:
            continue
        total_s = r["mean_ms"] * 1135 / 1000.0
        print(f"  {r['provider']:30s} ≈ {total_s:6.1f}s ({total_s/60:.1f} min)")


if __name__ == "__main__":
    main()
