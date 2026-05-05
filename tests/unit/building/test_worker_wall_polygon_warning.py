"""Sprint 52 W-4 — worker `_resolve_wall_polygon_enabled` warning tests."""
from __future__ import annotations

import logging

import pytest

from indoor_server.worker.loop import _resolve_wall_polygon_enabled


def test_both_enabled_returns_true_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log = logging.getLogger("test_worker.wall_polygon.both_enabled")
    caplog.set_level(logging.WARNING, logger=log.name)
    result = _resolve_wall_polygon_enabled(
        wall_polygon_enabled=True,
        use_floor_pointcloud=True,
        log=log,
    )
    assert result is True
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warnings == []


def test_wall_enabled_floor_disabled_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log = logging.getLogger("test_worker.wall_polygon.silent_disable")
    caplog.set_level(logging.WARNING, logger=log.name)
    result = _resolve_wall_polygon_enabled(
        wall_polygon_enabled=True,
        use_floor_pointcloud=False,
        log=log,
    )
    assert result is False
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "wall_polygon_enabled=true" in warnings[0].message
    assert "use_floor_pointcloud=false" in warnings[0].message


def test_wall_disabled_returns_false_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log = logging.getLogger("test_worker.wall_polygon.wall_disabled")
    caplog.set_level(logging.WARNING, logger=log.name)
    result = _resolve_wall_polygon_enabled(
        wall_polygon_enabled=False,
        use_floor_pointcloud=True,
        log=log,
    )
    assert result is False
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warnings == []


def test_both_disabled_returns_false_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log = logging.getLogger("test_worker.wall_polygon.both_disabled")
    caplog.set_level(logging.WARNING, logger=log.name)
    result = _resolve_wall_polygon_enabled(
        wall_polygon_enabled=False,
        use_floor_pointcloud=False,
        log=log,
    )
    assert result is False
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warnings == []
