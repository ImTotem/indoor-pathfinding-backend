"""snap.py 단위 테스트."""
from __future__ import annotations

import math
from uuid import UUID

import pytest

from indoor_server.application.routing.snap import snap_coordinate_to_node
from indoor_server.domain.routing.errors import SnapDistanceExceededError
from indoor_server.domain.routing.models import MapNodeRow

_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


def _node(nid: UUID, x: float, y: float, z: float = 0.0) -> MapNodeRow:
    return MapNodeRow(
        node_id=nid, x=x, y=y, z=z,
        node_type="corridor", label=None, poi_mark_id=None,
    )


# ── 행복 경로 ─────────────────────────────────────────────────────────────────


def test_snap_happy_nearest() -> None:
    """좌표가 A 노드에 더 가까우면 A를 선택."""
    lookup = {
        _A: _node(_A, x=0.0, y=0.0),
        _B: _node(_B, x=10.0, y=0.0),
    }
    nid, dist = snap_coordinate_to_node((0.5, 0.0, 0.0), lookup, threshold_m=5.0)
    assert nid == _A
    assert math.isclose(dist, 0.5, rel_tol=1e-5)


def test_snap_exact_match() -> None:
    """좌표가 노드와 정확히 일치 → distance=0."""
    lookup = {_A: _node(_A, x=3.0, y=4.0, z=1.5)}
    nid, dist = snap_coordinate_to_node((3.0, 4.0, 1.5), lookup, threshold_m=0.01)
    assert nid == _A
    assert dist == pytest.approx(0.0, abs=1e-9)


def test_snap_3d_distance() -> None:
    """z 차이도 포함해 3D 거리 계산."""
    lookup = {_A: _node(_A, x=0.0, y=0.0, z=0.0)}
    # (3, 4, 0) → dist = 5
    nid, dist = snap_coordinate_to_node((3.0, 4.0, 0.0), lookup, threshold_m=6.0)
    assert nid == _A
    assert math.isclose(dist, 5.0, rel_tol=1e-5)


# ── 실패 경로 ─────────────────────────────────────────────────────────────────


def test_snap_exceeds_threshold_raises() -> None:
    """가장 가까운 노드도 threshold 초과 → SnapDistanceExceededError."""
    lookup = {_A: _node(_A, x=100.0, y=0.0)}
    with pytest.raises(SnapDistanceExceededError) as exc_info:
        snap_coordinate_to_node((0.0, 0.0, 0.0), lookup, threshold_m=5.0, side="goal")
    err = exc_info.value
    assert err.distance_m > 5.0
    assert err.threshold_m == pytest.approx(5.0)
    assert err.side == "goal"


def test_snap_empty_lookup_raises() -> None:
    """node_lookup이 비어 있으면 → SnapDistanceExceededError."""
    with pytest.raises(SnapDistanceExceededError):
        snap_coordinate_to_node((0.0, 0.0, 0.0), {}, threshold_m=5.0)
