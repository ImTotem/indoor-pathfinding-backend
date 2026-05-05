"""Sprint 65 Issue 2 fix — match_node_ids_by_stamp 알고리즘 단위 테스트."""
from __future__ import annotations

from indoor_server.application.building.keyframe_node_id_backfill import (
    match_node_ids_by_stamp,
)


def test_match_basic_1to1() -> None:
    keyframes = [(1, 1000.0), (2, 1001.0), (3, 1002.0)]
    nodes = [(101, 1000.05), (102, 1001.05), (103, 1002.05)]
    updates = match_node_ids_by_stamp(keyframes=keyframes, nodes=nodes)
    assert sorted(updates) == [(1, 101), (2, 102), (3, 103)]


def test_match_skips_when_diff_exceeds_threshold() -> None:
    keyframes = [(1, 1000.0), (2, 1001.0), (3, 1002.0)]
    nodes = [(101, 1000.05), (102, 1500.0)]  # 102 is 499s off
    updates = match_node_ids_by_stamp(
        keyframes=keyframes, nodes=nodes, threshold_s=1.0
    )
    assert updates == [(1, 101)]


def test_match_greedy_does_not_double_assign() -> None:
    """동일 keyframe 이 두 node 모두에 가까워도 1:1 만 매칭."""
    keyframes = [(1, 1000.0)]
    nodes = [(101, 1000.05), (102, 1000.06)]
    updates = match_node_ids_by_stamp(keyframes=keyframes, nodes=nodes)
    # 1 keyframe 만 있으므로 가장 가까운 node 1 개만 매칭
    assert len(updates) == 1
    assert updates[0][0] == 1
    assert updates[0][1] in (101, 102)


def test_match_empty_inputs() -> None:
    assert match_node_ids_by_stamp(keyframes=[], nodes=[]) == []
    assert match_node_ids_by_stamp(keyframes=[(1, 1.0)], nodes=[]) == []
    assert match_node_ids_by_stamp(keyframes=[], nodes=[(101, 1.0)]) == []


def test_match_threshold_boundary_exclusive() -> None:
    """threshold 와 정확히 같은 diff 는 매칭하지 않는다 (exclusive)."""
    keyframes = [(1, 1000.0)]
    nodes = [(101, 1001.0)]  # diff = 1.0
    updates = match_node_ids_by_stamp(
        keyframes=keyframes, nodes=nodes, threshold_s=1.0
    )
    assert updates == []
    # threshold 1.001 이면 매칭
    updates2 = match_node_ids_by_stamp(
        keyframes=keyframes, nodes=nodes, threshold_s=1.001
    )
    assert updates2 == [(1, 101)]


def test_match_picks_closest_when_multiple_candidates() -> None:
    keyframes = [(1, 1000.0), (2, 1000.5)]
    # node stamp = 1000.4 — keyframe 2 (diff 0.1) 가 keyframe 1 (diff 0.4) 보다 가까움
    nodes = [(101, 1000.4)]
    updates = match_node_ids_by_stamp(keyframes=keyframes, nodes=nodes)
    assert updates == [(2, 101)]
