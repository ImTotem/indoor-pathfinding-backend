"""Sprint 65 Phase 4c — reprocess_service 단위 테스트.

실제 rtabmap-reprocess binary 가 PATH 에 있어야 PASS. 없으면 SKIP.
"""
from __future__ import annotations

import shutil
import sqlite3
import struct
from pathlib import Path

import pytest

from indoor_server.application.building.reprocess_service import (
    OptimizedPose,
    RTABMapReprocessRunner,
    extract_optimized_poses,
)

REPROCESS_BIN = shutil.which("rtabmap-reprocess")
_REPO_ROOT = Path(__file__).resolve().parents[3] / ".."
SAMPLE_DB = (
    _REPO_ROOT / "_workspace" / "sprint_65" / "evidence" / "sample_input.db"
).resolve()


def _make_minimal_rtabmap_db(path: Path, *, node_count: int = 3) -> None:
    """test fixture: Node 테이블만 가진 minimal RTAB-Map db.

    실제 RTAB-Map db schema 의 일부만 흉내. extract_optimized_poses 만 검증할 때 사용.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE Node (
            id INTEGER NOT NULL,
            map_id INTEGER NOT NULL,
            weight INTEGER,
            stamp FLOAT,
            pose BLOB,
            ground_truth_pose BLOB,
            velocity BLOB,
            label TEXT,
            gps BLOB,
            env_sensors BLOB,
            time_enter DATE,
            PRIMARY KEY (id)
        )
        """
    )
    for i in range(1, node_count + 1):
        # 간단한 identity rotation + translation = (i, 0, 0) 인 3x4 row-major 행렬
        pose_floats = [
            1.0, 0.0, 0.0, float(i),  # row 0
            0.0, 1.0, 0.0, 0.0,       # row 1
            0.0, 0.0, 1.0, 0.0,       # row 2
        ]
        pose_blob = struct.pack("<12f", *pose_floats)
        conn.execute(
            "INSERT INTO Node (id, map_id, stamp, pose) VALUES (?, ?, ?, ?)",
            (i, 0, 1000.0 + i, pose_blob),
        )
    conn.commit()
    conn.close()


def test_extract_optimized_poses_converts_3x4_rowmajor_to_4x4_colmajor(tmp_path: Path) -> None:
    db = tmp_path / "minimal.db"
    _make_minimal_rtabmap_db(db, node_count=2)

    poses = extract_optimized_poses(db)

    assert len(poses) == 2
    p1, p2 = poses
    assert isinstance(p1, OptimizedPose)
    assert p1.node_id == 1
    assert p1.stamp == pytest.approx(1001.0)
    assert len(p1.pose_matrix) == 64

    # column-major 4x4: column 3 (idx 12,13,14) = translation = (1, 0, 0)
    floats = struct.unpack("<16f", p1.pose_matrix)
    assert floats[12] == pytest.approx(1.0)
    assert floats[13] == pytest.approx(0.0)
    assert floats[14] == pytest.approx(0.0)
    assert floats[15] == pytest.approx(1.0)
    # column 0 = (1, 0, 0, 0)
    assert floats[0] == pytest.approx(1.0)
    assert floats[1] == pytest.approx(0.0)

    assert p2.node_id == 2
    p2_translation = struct.unpack("<16f", p2.pose_matrix)[12:15]
    assert p2_translation == pytest.approx((2.0, 0.0, 0.0))


def test_extract_optimized_poses_returns_empty_on_missing_file(tmp_path: Path) -> None:
    poses = extract_optimized_poses(tmp_path / "does_not_exist.db")
    assert poses == []


def test_runner_is_available_reflects_path() -> None:
    runner_with = RTABMapReprocessRunner(binary_path="/usr/bin/false")
    assert runner_with.binary_path == "/usr/bin/false"
    runner_none = RTABMapReprocessRunner(binary_path="/no/such/binary")
    # binary 가 실제 없으면 is_available False
    assert not runner_none.is_available()


@pytest.mark.skipif(
    REPROCESS_BIN is None or not SAMPLE_DB.exists(),
    reason="rtabmap-reprocess binary 또는 sample db 없음",
)
@pytest.mark.asyncio
async def test_runner_run_with_real_sample(tmp_path: Path) -> None:
    """실제 rtabmap-reprocess + sample db 로 e2e (15초 내).

    Phase 0 게이트에서 측정한 default reprocess (8초) 와 비슷하게 끝나야 한다.
    """
    runner = RTABMapReprocessRunner(default_timeout_s=120.0, feature_strategy=0)
    output = tmp_path / "reprocessed.db"
    result = await runner.run(input_db=SAMPLE_DB, output_db=output)
    assert output.exists()
    assert result.duration_s < 90.0
    assert len(result.optimized_poses) > 0
    # 대부분 nodeid + stamp 가 채워져 있어야 함
    sample = result.optimized_poses[0]
    assert sample.node_id > 0
    assert sample.stamp > 0
    assert len(sample.pose_matrix) == 64
