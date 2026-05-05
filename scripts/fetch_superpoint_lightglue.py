"""SuperPoint + LightGlue fused ONNX 모델 다운로드 스크립트.

1차: thomasonzhou/superpoint-lightglue (model.onnx, 단일 파일 fused)
  입력: images (B, 1, H, W) float32 grayscale — B=2 pair 기준
  출력: keypoints (B, 1024, 2), matches (N, 3) [batch,kp0,kp1], mscores (N,)

실패 시 alternate repo 목록을 순서대로 시도.

사용:
    uv run python scripts/fetch_superpoint_lightglue.py

var/models/superpoint_lightglue.onnx 에 저장. 이미 존재하면 skip.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indoor_server.config import settings

_LOCAL_NAME = "superpoint_lightglue.onnx"


def _try_download(repo_id: str, filename: str, dest: Path) -> bool:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface-hub 패키지가 설치되지 않았습니다.", file=sys.stderr)
        return False

    try:
        print(f"  trying repo={repo_id} file={filename} ...")
        tmp_path = hf_hub_download(repo_id=repo_id, filename=filename)
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

    print(f"downloading SuperPoint+LightGlue fused ONNX → {dest}")

    primary_repo = settings.superpoint_lightglue_model_repo_id
    primary_file = settings.superpoint_lightglue_model_filename
    if _try_download(primary_repo, primary_file, dest):
        print(f"model ready: {dest}")
        return

    for alt_repo in settings.superpoint_lightglue_model_alternate_repos:
        print(f"fallback: {alt_repo}")
        if _try_download(alt_repo, primary_file, dest):
            print(f"model ready (alternate): {dest}")
            return

    print(
        "\nERROR: 모든 repo에서 다운로드 실패.\n"
        "수동 다운로드:\n"
        "  https://huggingface.co/thomasonzhou/superpoint-lightglue/blob/main/model.onnx\n"
        f"  → {dest} 에 저장\n",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
