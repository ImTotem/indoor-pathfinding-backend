"""Sprint 67 — PoseMatcher binary roundtrip + PTS 매칭 검증.

iOS PoseFileWriter 가 만드는 72B/record 포맷을 정확히 디코드해야 한다.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from indoor_server.application.building.pose_matcher import (
    DEFAULT_THRESHOLD_NS,
    RECORD_SIZE_BYTES,
    PoseMatcher,
)


def _write_record(buf: bytearray, pts_ns: int, transform_4x4_col_major: np.ndarray) -> None:
    """iOS PoseFileWriter 와 동일한 layout 으로 쓰기.

    transform: column-major 4x4. iOS 가 floats 배열로 cols.0(xyzw), cols.1(xyzw), ... 순서로 쓴다.
    numpy column-major matrix 의 .T.flatten() 이 그 순서.
    """
    buf.extend(struct.pack("<q", pts_ns))
    floats = transform_4x4_col_major.T.astype(np.float32).flatten()
    buf.extend(struct.pack("<16f", *floats))


def test_record_size_contract() -> None:
    """iOS PoseFileWriter.recordSize 와 동일."""
    assert RECORD_SIZE_BYTES == 72


def test_load_roundtrip_basic(tmp_path: Path) -> None:
    """1 record write → read → 동일."""
    poses_path = tmp_path / "poses.bin"
    buf = bytearray()

    transform = np.eye(4, dtype=np.float32)
    transform[0, 3] = 1.5
    transform[1, 3] = 2.5
    transform[2, 3] = 3.5
    _write_record(buf, 1_000_000_000, transform)

    poses_path.write_bytes(bytes(buf))

    matcher = PoseMatcher(poses_path)
    assert len(matcher) == 1

    sample = matcher.find_pose(1_000_000_000)
    assert sample is not None
    assert sample.pts_ns == 1_000_000_000
    assert sample.translation[0] == pytest.approx(1.5)
    assert sample.translation[1] == pytest.approx(2.5)
    assert sample.translation[2] == pytest.approx(3.5)


def test_find_pose_returns_closest_within_threshold(tmp_path: Path) -> None:
    """ ε 차이 PTS 도 매칭."""
    poses_path = tmp_path / "poses.bin"
    buf = bytearray()
    _write_record(buf, 1_000_000_000, np.eye(4, dtype=np.float32))
    _write_record(buf, 1_016_666_666, np.eye(4, dtype=np.float32))  # +16.67ms = next 60fps frame
    poses_path.write_bytes(bytes(buf))

    matcher = PoseMatcher(poses_path)

    # 정확히 매칭
    s = matcher.find_pose(1_000_000_000)
    assert s is not None and s.pts_ns == 1_000_000_000

    # 5ms 차이 (threshold 16.67ms 이내) → 1 record 매칭
    s = matcher.find_pose(1_005_000_000)
    assert s is not None and s.pts_ns == 1_000_000_000

    # 5ms 차이 (1_000 까지 5ms, 1_016 까지 11.7ms) — 1_000 선택
    s = matcher.find_pose(1_005_000_000)
    assert s is not None and s.pts_ns == 1_000_000_000

    # 10ms 차이 (1_000 까지 10ms, 1_016 까지 6.67ms) — 1_016 선택
    s = matcher.find_pose(1_010_000_000)
    assert s is not None and s.pts_ns == 1_016_666_666

    # threshold 초과 → None
    s = matcher.find_pose(2_000_000_000, threshold_ns=DEFAULT_THRESHOLD_NS)
    assert s is None


def test_find_pose_picks_closest_when_two_candidates(tmp_path: Path) -> None:
    poses_path = tmp_path / "poses.bin"
    buf = bytearray()
    _write_record(buf, 1_000_000_000, np.eye(4, dtype=np.float32))
    _write_record(buf, 1_020_000_000, np.eye(4, dtype=np.float32))  # 20ms 후
    poses_path.write_bytes(bytes(buf))

    matcher = PoseMatcher(poses_path)
    # query 1_011_000_000 — 1_000 까지 11ms, 1_020 까지 9ms → 1_020 선택
    s = matcher.find_pose(1_011_000_000, threshold_ns=15_000_000)
    assert s is not None and s.pts_ns == 1_020_000_000


def test_corrupted_file_size_throws(tmp_path: Path) -> None:
    poses_path = tmp_path / "poses.bin"
    poses_path.write_bytes(b"\x00" * 71)  # 72 의 배수 아님
    with pytest.raises(ValueError, match="not a multiple"):
        PoseMatcher(poses_path)


def test_empty_file_yields_zero_records(tmp_path: Path) -> None:
    poses_path = tmp_path / "poses.bin"
    poses_path.write_bytes(b"")
    matcher = PoseMatcher(poses_path)
    assert len(matcher) == 0
    assert matcher.find_pose(0) is None


def test_translation_extraction_column_major(tmp_path: Path) -> None:
    """iOS column-major layout 검증.

    iOS 의 simd_float4x4.columns.3 == translation 인데
    iOS PoseFileWriter 는 col.x, col.y, col.z, col.w 순서로 쓰기 때문에
    파일 byte offset 8 + 12*4 = 56 부터 4 floats 가 (tx, ty, tz, tw=1).
    PoseMatcher.translation 은 transform[:3, 3] 이어야 함 (numpy col 0 row 0~2).
    """
    poses_path = tmp_path / "poses.bin"
    buf = bytearray()

    # column-major: identity but col 3 = (7, -3.5, 2.25, 1)
    transform = np.eye(4, dtype=np.float32)
    transform[0, 3] = 7.0
    transform[1, 3] = -3.5
    transform[2, 3] = 2.25
    _write_record(buf, 1_500_000_000_000, transform)
    poses_path.write_bytes(bytes(buf))

    matcher = PoseMatcher(poses_path)
    s = matcher.find_pose(1_500_000_000_000)
    assert s is not None
    np.testing.assert_array_almost_equal(s.translation, [7.0, -3.5, 2.25])
