"""Segformer-B0 ADE20K ONNX 모델 최초 다운로드 스크립트.

사용:
    uv run python scripts/fetch_segformer.py

var/models/segformer_b0_ade20k.onnx 에 저장.
이미 존재하면 skip.
"""
from __future__ import annotations

import sys
from pathlib import Path

# server/ 루트에서 실행 가정
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indoor_server.config import settings
from indoor_server.infrastructure.ml.model_cache import ModelCache


def main() -> None:
    cache = ModelCache(
        cache_dir=settings.model_cache_dir,
        repo_id=settings.segformer_model_repo_id,
        filename=settings.segformer_model_filename,
    )
    path = cache.ensure()
    print(f"model ready: {path}")


if __name__ == "__main__":
    main()
