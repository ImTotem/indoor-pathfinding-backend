"""Dev viewer 통합 테스트.

테스트 케이스:
  V1 viewer 비활성 (INDOOR_DEV_VIEWER_ENABLED=false) → /dev/api/scans 404
  V2 viewer 활성 → /dev/api/scans 200 + 목록 (빌드 완료된 것만)
  V3 viewer 활성 + 토큰 누락 → 401
  V4 HTML smoke: static/imdf_viewer/index.html 파일 존재 확인
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

# ── 상수 ──────────────────────────────────────────────────────────────────────

TOKEN = "dev-token"
SCAN_ID = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"
BUILD_JOB_ID = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")

_VIEWER_ROUTER_PATH = "indoor_server.interfaces.dev.viewer_router"


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def _make_viewer_app() -> FastAPI:
    """dev_router만 등록한 격리 FastAPI 앱 반환."""
    from indoor_server.interfaces.dev.viewer_router import dev_router

    _app = FastAPI()
    _app.include_router(dev_router)
    return _app


# ── V1: viewer 비활성 → 404 ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_v1_viewer_disabled_returns_404() -> None:
    """INDOOR_DEV_VIEWER_ENABLED=false(기본값) 상태에서 /dev/api/scans → 404."""
    from indoor_server.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/dev/api/scans", headers=_auth_headers())

    assert res.status_code == 404


# ── V2: viewer 활성 → 200 + scan 목록 ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_v2_viewer_enabled_returns_scan_list() -> None:
    """dev_router 직접 앱에 등록 → /dev/api/scans 200 + 목록 반환."""
    test_app = _make_viewer_app()

    mock_row = AsyncMock()
    mock_row.scan_id = SCAN_ID
    mock_row.build_job_id = str(BUILD_JOB_ID)
    mock_row.counts = {"walkable_cells": 56832, "map_nodes": 19, "map_edges": 19}
    mock_row.finished_at = datetime.now(tz=UTC)

    from unittest.mock import MagicMock

    mock_result = MagicMock()
    mock_result.fetchall.return_value = [mock_row]

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)

    async def _override_session():  # type: ignore[return]
        yield mock_session

    from indoor_server.infrastructure.db.engine import get_session

    test_app.dependency_overrides[get_session] = _override_session

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        res = await client.get("/dev/api/scans", headers=_auth_headers())

    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["scan_id"] == SCAN_ID
    assert data[0]["walkable_cells"] == 56832


# ── V3: 토큰 누락 → 401 ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_v3_viewer_no_token_returns_401() -> None:
    """dev_router 앱에서 Bearer 토큰 없으면 401."""
    test_app = _make_viewer_app()

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        res = await client.get("/dev/api/scans")  # 토큰 없음

    assert res.status_code == 401


# ── V4: HTML 파일 존재 smoke test ────────────────────────────────────────────────


def test_v4_static_files_exist() -> None:
    """static/imdf_viewer/ 에 필수 파일이 존재하는지 확인."""
    server_root = Path(__file__).resolve().parents[3]  # server/
    viewer_dir = server_root / "static" / "imdf_viewer"

    assert viewer_dir.exists(), f"viewer 디렉토리 없음: {viewer_dir}"
    assert (viewer_dir / "index.html").exists(), "index.html 없음"
    assert (viewer_dir / "viewer.js").exists(), "viewer.js 없음"
    assert (viewer_dir / "viewer.css").exists(), "viewer.css 없음"

    viewer_js = (viewer_dir / "viewer.js").read_text()
    assert "props.display_point" in viewer_js
    assert "hud-rectified" in viewer_js  # unit 제거 후 남은 HUD 항목
    assert "_manhattanizeCoords(coords, 0.25).map(_pt)" in viewer_js
    assert "function _manhattanizeCoords" in viewer_js
