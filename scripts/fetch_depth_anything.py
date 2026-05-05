"""Depth Anything v2-Small ONNX 모델 다운로드 스크립트.

1차: onnx-community/depth-anything-v2-small (onnx/model.onnx)
실패 시 alternate repo 목록을 순서대로 시도.

사용:
    uv run python scripts/fetch_depth_anything.py

var/models/depth_anything_v2_small.onnx 에 저장.
이미 존재하면 skip.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indoor_server.config import settings

_LOCAL_NAME = "depth_anything_v2_small.onnx"


def _try_download(repo_id: str, filename: str, dest: Path) -> bool:
    """HF Hub에서 다운로드 시도. 성공 시 True, 실패 시 False."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface-hub 패키지가 설치되지 않았습니다.", file=sys.stderr)
        return False

    try:
        print(f"  trying repo={repo_id} file={filename} ...")
        tmp_path = hf_hub_download(repo_id=repo_id, filename=filename)
        import shutil

        shutil.copy2(tmp_path, dest)
        return True
    except Exception as e:
        print(f"  failed: {e}")
        return False


def main() -> None:
    cache_dir = settings.model_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / _LOCAL_NAME

    if dest.exists():
        print(f"model cache hit: {dest} ({dest.stat().st_size:,} bytes)")
        return

    print(f"downloading Depth Anything v2-Small ONNX → {dest}")

    # 1차 시도
    primary_repo = settings.depth_anything_model_repo_id
    primary_file = settings.depth_anything_model_filename
    if _try_download(primary_repo, primary_file, dest):
        print(f"model ready: {dest}")
        return

    # alternate fallback
    for alt_repo in settings.depth_anything_model_alternate_repos:
        print(f"fallback: {alt_repo}")
        if _try_download(alt_repo, primary_file, dest):
            print(f"model ready (alternate): {dest}")
            return

    print(
        "\nERROR: 모든 repo에서 다운로드 실패.\n"
        "수동 다운로드:\n"
        "  https://huggingface.co/onnx-community/depth-anything-v2-small/tree/main/onnx\n"
        f"  → {dest} 에 저장\n",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
