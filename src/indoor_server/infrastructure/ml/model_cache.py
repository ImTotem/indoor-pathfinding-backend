"""on-demand HuggingFace Hub 모델 다운로드 + 로컬 캐시 관리.

onnxruntime import는 이 모듈 내부에서만.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ADE20K segformer-b0 ONNX — HF Hub snapshot
_DEFAULT_REPO_ID = "nickmuchi/segformer-b0-finetuned-ade-512-512"
_DEFAULT_FILENAME = "onnx/model.onnx"
_MODEL_LOCAL_NAME = "segformer_b0_ade20k.onnx"


class ModelCache:
    """모델 파일을 cache_dir에 관리. 없으면 HF Hub에서 pull."""

    def __init__(
        self,
        cache_dir: Path,
        repo_id: str = _DEFAULT_REPO_ID,
        filename: str = _DEFAULT_FILENAME,
    ) -> None:
        self._cache_dir = cache_dir
        self._repo_id = repo_id
        self._filename = filename
        self._local_path: Path | None = None

    @property
    def model_path(self) -> Path:
        if self._local_path is None:
            raise RuntimeError("ModelCache.ensure() 먼저 호출 필요")
        return self._local_path

    def ensure(self) -> Path:
        """캐시에 없으면 HF Hub에서 다운로드. 있으면 즉시 반환."""
        dest = self._cache_dir / _MODEL_LOCAL_NAME
        if dest.exists():
            logger.info("model cache hit path=%s", dest)
            self._local_path = dest
            return dest

        logger.info(
            "model not cached — downloading repo_id=%s filename=%s",
            self._repo_id,
            self._filename,
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._download(dest)
        self._local_path = dest
        logger.info("model downloaded path=%s size=%d", dest, dest.stat().st_size)
        return dest

    def _download(self, dest: Path) -> None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RuntimeError("huggingface-hub 패키지가 설치되지 않았습니다.") from e

        tmp_path = hf_hub_download(
            repo_id=self._repo_id,
            filename=self._filename,
        )
        import shutil

        shutil.copy2(tmp_path, dest)

    def sha256(self) -> str:
        """캐시된 모델 파일의 sha256 (검증용)."""
        path = self.model_path
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 18), b""):
                h.update(chunk)
        return h.hexdigest()
