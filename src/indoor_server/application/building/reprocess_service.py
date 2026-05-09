"""rtabmap-reprocess subprocess runner (Sprint 65).

iOS raw recording 모드로 만든 rtabmap.db (Kp/MaxFeatures=-1, loop closure 비활성)를
서버가 desktop `rtabmap-reprocess` CLI 로 재처리하여 SIFT 재추출 + loop closure 검출 +
graph optimization 결과 db 를 생성한다.

호출 흐름 (Phase 4c):
    1. ingest 직후 또는 build pipeline 직전에 worker 가 RTABMapReprocessRunner.run() 호출.
    2. 결과 db 의 Node 테이블에서 (id, stamp, optimized 4x4 pose) 추출.
    3. pose_backfill 이 keyframe_meta.rtabmap_node_id 매칭으로 optimized pose 영속화.

설계 결정:
- subprocess: asyncio.create_subprocess_exec — 동기 worker thread 차단 방지.
- timeout: 기본 5분. scan 길이 비례 동적 조정은 호출자 책임.
- binary: PATH 에서 `rtabmap-reprocess` 자동 검색. 별도 경로는 `RTABMAP_REPROCESS_BIN` 환경변수.
- 옵션: `--Kp/DetectorStrategy 1 --Vis/FeatureType 1` (SIFT 재추출). raw mode db 에는
  feature 가 없으므로 desktop 에서 새로 뽑는다 (Issue #942 matlabbe 답변 근거).
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


class RTABMapReprocessError(Exception):
    """rtabmap-reprocess 실행 실패 (binary 없음, timeout, exit code != 0)."""


@dataclass(frozen=True)
class OptimizedPose:
    """rtabmap-reprocess 후 RTAB-Map graph optimization 결과 pose.

    pose_matrix 는 row-major 3x4 (12 float) → column-major 4x4 simd 형식 (16 float, 64 bytes).
    iOS keyframe_meta.pose_matrix 와 동일 직렬화 → server keyframe_meta.pose_matrix 와 호환.
    """

    node_id: int
    stamp: float  # Unix epoch seconds
    pose_matrix: bytes  # 64 bytes, column-major 4x4 float


@dataclass(frozen=True)
class ReprocessResult:
    """Reprocess 산출물."""

    output_db_path: Path
    duration_s: float
    optimized_poses: list[OptimizedPose]
    total_loop_closures: int = 0
    stdout_tail: str = ""
    stderr_tail: str = ""
    extra: dict[str, object] = field(default_factory=dict)


class RTABMapReprocessRunner:
    """rtabmap-reprocess CLI 호출 + 결과 파싱."""

    def __init__(
        self,
        *,
        binary_path: str | None = None,
        default_timeout_s: float = 300.0,
        feature_strategy: int = 1,  # 1 = SIFT
    ) -> None:
        self._binary_path = binary_path or os.environ.get(
            "RTABMAP_REPROCESS_BIN"
        ) or shutil.which("rtabmap-reprocess")
        self._default_timeout_s = default_timeout_s
        self._feature_strategy = feature_strategy

    @property
    def binary_path(self) -> str | None:
        return self._binary_path

    def is_available(self) -> bool:
        return self._binary_path is not None and Path(self._binary_path).exists()

    async def run(
        self,
        *,
        input_db: Path,
        output_db: Path,
        timeout_s: float | None = None,
        rgbd_enabled: bool = False,
        detector_strategy: int | None = None,
        loop_thr: float = 0.11,
    ) -> ReprocessResult:
        """rtabmap-reprocess 실행. visual feature 재추출 + graph optimization.

        Args:
            rgbd_enabled: depth 있는 db (live_rtabmap 모드) 면 True. RGBD loop closure 활성.
            detector_strategy: None 이면 default (RGBD: 6=GFTT/BRIEF, RGB-only: SIFT).
                sprint81 검증으로 단조 환경에선 6 이 효과 ↑.
            loop_thr: 0.11 (sprint81 표준, default 보다 lenient — loop closure 검출 ↑).

        Raises:
            RTABMapReprocessError: binary 없음, exit code != 0, timeout.
        """
        if self._binary_path is None:
            raise RTABMapReprocessError(
                "rtabmap-reprocess binary 가 PATH 또는 RTABMAP_REPROCESS_BIN 에 없습니다."
            )
        if not input_db.exists():
            raise RTABMapReprocessError(f"input db 가 없습니다: {input_db}")

        output_db.parent.mkdir(parents=True, exist_ok=True)
        if output_db.exists():
            output_db.unlink()

        # detector strategy: RGBD 면 GFTT/BRIEF (6, sprint81), 아니면 SIFT (1) default
        ds = detector_strategy if detector_strategy is not None else (
            6 if rgbd_enabled else self._feature_strategy
        )

        # introlab iOS 앱 호환: RGB↔depth 비정수 비율 (1920×1440 / 256×192 = 7.5×) 처리.
        # ImagePreDecimation=1 + DepthAsMask=true 이면 RTABMap 가 internal interpolate.
        # Hard error 는 decimation factor 가 depth 차원을 나눠떨어뜨리지 않을 때만.
        args = [
            self._binary_path,
            "--Mem/IncrementalMemory", "true",
            "--Mem/InitWMWithAllNodes", "true",
            "--Reg/Strategy", "0",
            "--Vis/EstimationType", "1",
            "--Rtabmap/DetectionRate", "0",
            "--Rtabmap/LoopThr", str(loop_thr),
            "--Mem/STMSize", "30",
            "--RGBD/Enabled", "true" if rgbd_enabled else "false",
            "--Mem/ImagePreDecimation", "1",   # 명시 — depth 비정수 비율도 OK
            "--Mem/DepthAsMask", "true",        # depth 를 RGB 에 맞춰 interpolate
            f"--Kp/DetectorStrategy={ds}",
            f"--Vis/FeatureType={ds}",
            "--uwarn",
            str(input_db),
            str(output_db),
        ]
        timeout = timeout_s if timeout_s is not None else self._default_timeout_s

        loop = asyncio.get_event_loop()
        start = loop.time()
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except TimeoutError as e:
            proc.kill()
            await proc.wait()
            raise RTABMapReprocessError(
                f"rtabmap-reprocess timeout > {timeout}s"
            ) from e
        duration = loop.time() - start

        stdout_text = stdout_b.decode(errors="replace") if stdout_b else ""
        stderr_text = stderr_b.decode(errors="replace") if stderr_b else ""

        if proc.returncode != 0:
            raise RTABMapReprocessError(
                f"rtabmap-reprocess exit code {proc.returncode}\n"
                f"stderr tail: {stderr_text[-1000:]}"
            )

        optimized = extract_optimized_poses(output_db)
        loop_closures = _count_loop_closures(output_db)

        logger.info(
            "rtabmap-reprocess complete",
            extra={
                "input_db": str(input_db),
                "output_db": str(output_db),
                "duration_s": round(duration, 3),
                "optimized_node_count": len(optimized),
                "loop_closures": loop_closures,
            },
        )

        return ReprocessResult(
            output_db_path=output_db,
            duration_s=duration,
            optimized_poses=optimized,
            total_loop_closures=loop_closures,
            stdout_tail=stdout_text[-2000:],
            stderr_tail=stderr_text[-2000:],
        )


def extract_optimized_poses(db_path: Path) -> list[OptimizedPose]:
    """rtabmap.db Node 테이블에서 (id, stamp, pose) 를 읽어 4x4 column-major 직렬화.

    RTAB-Map Node.pose = 3x4 row-major 12 floats (48 bytes).
    [r00 r01 r02 tx, r10 r11 r12 ty, r20 r21 r22 tz]

    iOS/server keyframe_meta.pose_matrix 는 column-major 4x4 16 floats (64 bytes).
    columns: [right(3+0), up(3+0), -forward(3+0), pos(3+1)] in simd_float4x4 layout.

    변환:
        col0 = (r00, r10, r20, 0)
        col1 = (r01, r11, r21, 0)
        col2 = (r02, r12, r22, 0)
        col3 = (tx,  ty,  tz,  1)
    """
    import sqlite3

    if not db_path.exists():
        return []

    uri = f"file:{db_path}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cur = conn.execute(
            "SELECT id, stamp, pose FROM Node WHERE pose IS NOT NULL ORDER BY id"
        )
        result: list[OptimizedPose] = []
        for node_id, stamp, pose_blob in cur.fetchall():
            if pose_blob is None or len(pose_blob) != 48:
                continue
            r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz = struct.unpack(
                "<12f", pose_blob
            )
            # column-major 4x4 (16 floats)
            cm = [
                r00, r10, r20, 0.0,  # col 0
                r01, r11, r21, 0.0,  # col 1
                r02, r12, r22, 0.0,  # col 2
                tx,  ty,  tz,  1.0,  # col 3
            ]
            packed = struct.pack("<16f", *cm)
            result.append(
                OptimizedPose(
                    node_id=int(node_id),
                    stamp=float(stamp) if stamp is not None else 0.0,
                    pose_matrix=packed,
                )
            )
        return result
    finally:
        conn.close()


def _count_loop_closures(db_path: Path) -> int:
    """Link 테이블에서 type 1 (Global loop closure) + 2 (Local space) row 수."""
    import sqlite3

    if not db_path.exists():
        return 0
    uri = f"file:{db_path}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM Link WHERE type IN (1, 2)"
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0
    finally:
        conn.close()
