"""rtabmap-reprocess 결과 optimized pose 를 도메인 테이블에 backfill (Sprint 65).

흐름:
    1. RTABMapReprocessRunner 가 OptimizedPose 리스트 생성 (rtabmap.db Node).
    2. backfill_keyframe_poses 가 keyframe_meta.rtabmap_node_id (Sprint 35 backfill 로 채워진) 와
       매칭하여 keyframe_meta.pose_matrix / tx / ty / tz 를 optimized 로 UPDATE.
    3. backfill_dependent_poses 가 같은 keyframe_seq 를 참조하는 poi_mark / branch_mark /
       interfloor_mark 의 pose 를 keyframe pre/post 변환 delta 로 보정.

설계 결정:
- iOS sidecar 가 정한 ARKit raw pose 를 서버가 reprocess 후 덮어쓰기.
  ARKit drift 보정이 build_pipeline 모든 단계에 자동 반영.
- raw pose 가 별도로 필요하면 향후 alembic 으로 keyframe_meta.raw_pose_matrix 컬럼 추가.
- pose backfill 은 reprocess 가 PASS 한 경우에만 호출. 실패 시 raw pose 그대로 두어 fallback.
"""
from __future__ import annotations

import logging
import struct
from dataclasses import dataclass

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.application.building.reprocess_service import OptimizedPose
from indoor_server.infrastructure.db import tables as t

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PoseBackfillStats:
    keyframe_updated: int
    keyframe_skipped_no_node_id: int
    poi_updated: int
    branch_updated: int
    interfloor_updated: int


def _pose_translation(pose_matrix: bytes) -> tuple[float, float, float]:
    """column-major 4x4 (64 bytes) 에서 columns.3.x/y/z 추출."""
    if len(pose_matrix) != 64:
        return (0.0, 0.0, 0.0)
    floats = struct.unpack("<16f", pose_matrix)
    # column-major: [c0(0..3), c1(4..7), c2(8..11), c3(12..15)] → translation = c3 (idx 12,13,14)
    return (float(floats[12]), float(floats[13]), float(floats[14]))


async def backfill_keyframe_poses(
    session: AsyncSession,
    *,
    scan_id: str,
    optimized: list[OptimizedPose],
) -> tuple[int, int]:
    """keyframe_meta.rtabmap_node_id 와 OptimizedPose.node_id 를 매칭하여 pose UPDATE.

    Returns:
        (updated, skipped_no_match): UPDATE 수행 행 수 / rtabmap_node_id NULL 또는
        매칭 안 된 OptimizedPose 수.
    """
    if not optimized:
        return (0, 0)

    by_node_id = {p.node_id: p for p in optimized}

    rows = (
        await session.execute(
            sa.select(
                t.keyframe_meta.c.scan_id,
                t.keyframe_meta.c.seq,
                t.keyframe_meta.c.rtabmap_node_id,
            ).where(t.keyframe_meta.c.scan_id == scan_id)
        )
    ).fetchall()

    updated = 0
    skipped = 0
    for row in rows:
        node_id = row.rtabmap_node_id
        if node_id is None:
            skipped += 1
            continue
        opt = by_node_id.get(int(node_id))
        if opt is None:
            skipped += 1
            continue
        tx, ty, tz = _pose_translation(opt.pose_matrix)
        await session.execute(
            sa.update(t.keyframe_meta)
            .where(
                sa.and_(
                    t.keyframe_meta.c.scan_id == scan_id,
                    t.keyframe_meta.c.seq == row.seq,
                )
            )
            .values(
                pose_matrix=opt.pose_matrix,
                tx=tx,
                ty=ty,
                tz=tz,
            )
        )
        updated += 1

    return (updated, skipped)


async def backfill_dependent_poses(
    session: AsyncSession,
    *,
    scan_id: str,
) -> tuple[int, int, int]:
    """poi_mark / branch_mark / interfloor_mark 의 pose 를 같은 keyframe 의 optimized pose 로 정렬.

    단순화 전략: iOS 가 마킹한 pose 는 그 시점 keyframe 의 transform 과 동일하다고 가정.
    keyframe_meta 가 reprocessed pose 로 갱신된 후, 동일 keyframe_seq 의 mark 도 그 값으로
    덮어쓰기. (drift 보정이 작은 경우 차이 미미.)

    Returns:
        (poi_updated, branch_updated, interfloor_updated)
    """
    # keyframe_seq → (pose_matrix, tx, ty, tz). reprocessed (rtabmap_node_id NOT NULL) 만.
    # ARKit drift 로 일부 keyframe 의 raw pose 가 jump 되어 있을 수 있어, RTABMap 가
    # keyframe 으로 채택해 graph optimization 통과한 것만 trusted source 로.
    rows = (
        await session.execute(
            sa.select(
                t.keyframe_meta.c.seq,
                t.keyframe_meta.c.pose_matrix,
                t.keyframe_meta.c.tx,
                t.keyframe_meta.c.ty,
                t.keyframe_meta.c.tz,
            )
            .where(t.keyframe_meta.c.scan_id == scan_id)
            .where(t.keyframe_meta.c.rtabmap_node_id.isnot(None))
        )
    ).fetchall()
    by_seq = {row.seq: (row.pose_matrix, row.tx, row.ty, row.tz) for row in rows}

    poi_updated = await _backfill_table(
        session,
        scan_id=scan_id,
        table=t.poi_mark,
        by_seq=by_seq,
    )
    branch_updated = await _backfill_table(
        session,
        scan_id=scan_id,
        table=t.branch_mark,
        by_seq=by_seq,
    )
    interfloor_updated = await _backfill_table(
        session,
        scan_id=scan_id,
        table=t.interfloor_mark,
        by_seq=by_seq,
    )
    return (poi_updated, branch_updated, interfloor_updated)


# iOS RTABMap library applies this rotation to convert ARKit-world poses
# into the RTABMap base frame stored in raw rtabmap.db Node.pose:
#   ARKit (X, Y, Z) → RTABMap base (-Z, -X, Y)
# Empirically derived from comparing sidecar.keyframe_meta.pose_matrix (ARKit)
# with raw rtabmap.db Node.pose (RTABMap base) for the same keyframe.
#
# iOS sends `dx/dy/dz_local = inverse(T_kf_arkit) @ p_mark_arkit_world`,
# i.e. coordinates in the ARKit camera frame. The optimized keyframe.pose
# we store in keyframe_meta is in the RTABMap base frame, so we rotate the
# ARKit-frame offset into the base frame before applying the keyframe pose.
_IOS_ARKIT_TO_RTABMAP_BASE_ROT = (
    (0.0, 0.0, -1.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)


async def _backfill_table(
    session: AsyncSession,
    *,
    scan_id: str,
    table: sa.Table,
    by_seq: dict[int, tuple[bytes, float, float, float]],
) -> int:
    """mark world pose 를 optimized keyframe pose × axis-swapped local offset 으로 갱신.

    keyframe_meta.pose 는 base→world (4x4 col-major bytes, 64B), RTABMap base
    frame 의 표현. 그러나 iOS 가 보낸 dx/dy/dz_local 은 ARKit camera frame 좌표
    (`inverse(T_kf_arkit) @ p_mark_arkit_world`) 라서, 두 frame 사이의 회전을
    적용한 뒤 keyframe.R @ local + keyframe.t 로 world 좌표 계산.
    """
    import bisect
    import numpy as np
    if not by_seq:
        return 0
    has_local = all(c in table.c for c in ("dx_local", "dy_local", "dz_local"))
    select_cols = [table.c.id, table.c.keyframe_seq]
    if has_local:
        select_cols += [table.c.dx_local, table.c.dy_local, table.c.dz_local]
    rows = (await session.execute(
        sa.select(*select_cols).where(table.c.scan_id == scan_id)
    )).fetchall()

    sorted_seqs = sorted(by_seq.keys())
    swap = np.array(_IOS_ARKIT_TO_RTABMAP_BASE_ROT, dtype=np.float64)

    def _nearest_seq(target: int) -> int | None:
        if not sorted_seqs:
            return None
        idx = bisect.bisect_left(sorted_seqs, target)
        candidates = []
        if idx < len(sorted_seqs):
            candidates.append(sorted_seqs[idx])
        if idx > 0:
            candidates.append(sorted_seqs[idx - 1])
        return min(candidates, key=lambda s: abs(s - target))

    updated = 0
    for row in rows:
        kf = by_seq.get(row.keyframe_seq)
        if kf is None:
            # ARKit drift 로 reprocess 안 된 keyframe 에 마크된 케이스 — nearest reprocessed seq fallback.
            nearest = _nearest_seq(row.keyframe_seq)
            if nearest is None:
                continue
            kf = by_seq[nearest]
        pose_matrix, kf_tx, kf_ty, kf_tz = kf

        dx = getattr(row, "dx_local", None) if has_local else None
        dy = getattr(row, "dy_local", None) if has_local else None
        dz = getattr(row, "dz_local", None) if has_local else None

        if dx is not None and dy is not None and dz is not None and pose_matrix:
            T = np.frombuffer(pose_matrix, dtype=np.float32).reshape(4, 4, order="F")
            local_arkit = np.array([dx, dy, dz], dtype=np.float64)
            local_rtab = swap @ local_arkit
            world_xyz = (
                T[:3, :3].astype(np.float64) @ local_rtab
                + T[:3, 3].astype(np.float64)
            )
            tx, ty, tz = float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])
        else:
            tx, ty, tz = kf_tx, kf_ty, kf_tz

        await session.execute(
            sa.update(table).where(table.c.id == row.id)
            .values(pose_matrix=pose_matrix, tx=tx, ty=ty, tz=tz)
        )
        updated += 1
    return updated


async def run_full_backfill(
    session: AsyncSession,
    *,
    scan_id: str,
    optimized: list[OptimizedPose],
) -> PoseBackfillStats:
    """reprocess 완료 후 모든 도메인 pose 를 일관성 있게 갱신."""
    keyframe_updated, keyframe_skipped = await backfill_keyframe_poses(
        session, scan_id=scan_id, optimized=optimized
    )
    poi_u, branch_u, interfloor_u = await backfill_dependent_poses(
        session, scan_id=scan_id
    )
    stats = PoseBackfillStats(
        keyframe_updated=keyframe_updated,
        keyframe_skipped_no_node_id=keyframe_skipped,
        poi_updated=poi_u,
        branch_updated=branch_u,
        interfloor_updated=interfloor_u,
    )
    logger.info("pose backfill complete scan_id=%s stats=%s", scan_id, stats)
    return stats
