"""ModelCache 단위 테스트 — HF 실제 다운로드 금지 (mock)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from indoor_server.infrastructure.ml.model_cache import _MODEL_LOCAL_NAME, ModelCache


def test_cache_hit(tmp_path: Path):
    """캐시 파일이 존재하면 다운로드 없이 즉시 반환."""
    dest = tmp_path / _MODEL_LOCAL_NAME
    dest.write_bytes(b"fake_model_data")

    cache = ModelCache(cache_dir=tmp_path)
    with patch.object(cache, "_download") as mock_dl:
        result = cache.ensure()
        mock_dl.assert_not_called()

    assert result == dest


def test_cache_miss_triggers_download(tmp_path: Path):
    """캐시 파일이 없으면 _download가 호출되어야 함."""
    cache = ModelCache(cache_dir=tmp_path)

    def _fake_download(dest: Path) -> None:
        dest.write_bytes(b"fake_downloaded_model")

    with patch.object(cache, "_download", side_effect=_fake_download) as mock_dl:
        result = cache.ensure()
        mock_dl.assert_called_once()

    assert result.exists()
    assert cache.model_path == result


def test_model_path_before_ensure_raises():
    """ensure() 호출 전 model_path 접근 → RuntimeError."""
    cache = ModelCache(cache_dir=Path("/nonexistent"))
    with pytest.raises(RuntimeError, match="ensure"):
        _ = cache.model_path


def test_fake_path_injection(tmp_path: Path):
    """외부에서 fake path 주입해 segmenter 테스트 지원."""
    dest = tmp_path / _MODEL_LOCAL_NAME
    dest.write_bytes(b"fake_onnx_model")
    cache = ModelCache(cache_dir=tmp_path)
    path = cache.ensure()
    assert path == dest
    assert path.exists()
