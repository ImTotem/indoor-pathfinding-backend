"""rtabmap.db Node.stamp ↔ keyframe_meta.captured_at 매칭으로 rtabmap_node_id backfill (Sprint 65).

배경:
    Sprint 65 raw recording 모드(`Mem/STMSize=1`, `Mem/RehearsalSimilarity=1.0`)에서는
    RTABMap working memory 가 비어 있어 iOS finalize 시점의 `getAllNodeIdsAndStampsNative()` 가
    빈 결과를 반환한다. 결과적으로 sidecar `keyframe_meta.rtabmap_node_id` 가 모두 NULL 인 채로
    서버에 ingest 된다.

    그러나 raw recording mode + `Mem/NotLinkedNodesKept=true` 옵션으로 모든 frame 이 rtabmap.db
    의 Node 테이블에는 정상 저장된다. 따라서 서버가 ingest 직후 rtabmap.db Node.stamp 와
    keyframe_meta.captured_at 을 매칭해 NULL 행을 채우면 Sprint 35 iOS backfill 과 동일한 결과를
    얻을 수 있다.

알고리즘 (Sprint 35 iOS backfillNodeIDsFromGraph 그대로 옮김):
    1. NULL rtabmap_node_id 인 keyframe_meta 행 read.
    2. rtabmap.db Node.stamp 오름차순 정렬.
    3. 각 node 에 대해 가장 가까운 captured_at 행 매칭 (이미 사용된 seq 제외).
    4. abs(captured_at - stamp) < threshold(default 1.0s) 면 UPDATE.

이 backfill 은 reprocess 전 / 후 모두 호출해도 idempotent. Build pipeline 시작 시점에 1 회 실행.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.infrastructure.db import tables as t

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeIdBackfillStats:
    matched: int
    skipped_no_match: int
    already_set: int


def _read_keyframe_rows(
    db_path: Path,
) -> list[tuple[int, float]]:
    """rtabmap.db Node 에서 (id, stamp_seconds) 추출."""
    if not db_path.exists():
        return []
    uri = f"file:{db_path}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            "SELECT id, stamp FROM Node WHERE stamp IS NOT NULL ORDER BY stamp"
        ).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]
    finally:
        conn.close()


def match_node_ids_by_stamp(
    *,
    keyframes: list[tuple[int, float]],
    nodes: list[tuple[int, float]],
    threshold_s: float = 1.0,
) -> list[tuple[int, int]]:
    """순수 함수: greedy 1:1 매칭.

    Args:
        keyframes: [(seq, captured_at_seconds), ...]. NULL 인 keyframe 만 입력.
        nodes: [(node_id, stamp_seconds), ...]. stamp 오름차순 정렬 가정.
        threshold_s: 매칭 허용 시간 차이.

    Returns:
        [(seq, node_id), ...] — UPDATE 대상.
    """
    if not keyframes or not nodes:
        return []

    used_seqs: set[int] = set()
    updates: list[tuple[int, int]] = []
    sorted_nodes = sorted(nodes, key=lambda n: n[1])

    for node_id, stamp in sorted_nodes:
        best_seq: int | None = None
        best_diff = threshold_s + 1.0
        for seq, captured_at in keyframes:
            if seq in used_seqs:
                continue
            diff = abs(captured_at - stamp)
            if diff < best_diff:
                best_diff = diff
                best_seq = seq
        if best_seq is not None and best_diff < threshold_s:
            used_seqs.add(best_seq)
            updates.append((best_seq, node_id))
    return updates


async def backfill_keyframe_node_ids(
    session: AsyncSession,
    *,
    scan_id: str,
    rtabmap_db_path: Path,
    threshold_s: float = 1.0,
) -> NodeIdBackfillStats:
    """server keyframe_meta.rtabmap_node_id 가 NULL 인 행을 rtabmap.db Node.stamp 매칭으로 채움.

    이미 set 된 행은 건드리지 않는다 (idempotent).
    """
    if not rtabmap_db_path.exists():
        logger.warning(
            "rtabmap.db 가 없어 nodeID backfill skip scan_id=%s path=%s",
            scan_id, rtabmap_db_path,
        )
        return NodeIdBackfillStats(matched=0, skipped_no_match=0, already_set=0)

    # 1. NULL 행 read
    null_rows = (
        await session.execute(
            sa.select(t.keyframe_meta.c.seq, t.keyframe_meta.c.captured_at)
            .where(
                sa.and_(
                    t.keyframe_meta.c.scan_id == scan_id,
                    t.keyframe_meta.c.rtabmap_node_id.is_(None),
                )
            )
            .order_by(t.keyframe_meta.c.seq)
        )
    ).fetchall()

    already_set = (
        await session.execute(
            sa.select(sa.func.count())
            .select_from(t.keyframe_meta)
            .where(
                sa.and_(
                    t.keyframe_meta.c.scan_id == scan_id,
                    t.keyframe_meta.c.rtabmap_node_id.isnot(None),
                )
            )
        )
    ).scalar() or 0

    if not null_rows:
        return NodeIdBackfillStats(matched=0, skipped_no_match=0, already_set=int(already_set))

    keyframes_in: list[tuple[int, float]] = [
        (int(r.seq), float(r.captured_at) / 1000.0) for r in null_rows
    ]

    # 2. rtabmap.db Node.stamp read
    nodes_in = _read_keyframe_rows(rtabmap_db_path)
    if not nodes_in:
        logger.warning(
            "rtabmap.db Node 테이블이 비어 있어 nodeID backfill 0 건 scan_id=%s",
            scan_id,
        )
        return NodeIdBackfillStats(
            matched=0, skipped_no_match=len(keyframes_in), already_set=int(already_set)
        )

    # 3. greedy 매칭
    updates = match_node_ids_by_stamp(
        keyframes=keyframes_in, nodes=nodes_in, threshold_s=threshold_s
    )

    # 4. UPDATE 실행
    for seq, node_id in updates:
        await session.execute(
            sa.update(t.keyframe_meta)
            .where(
                sa.and_(
                    t.keyframe_meta.c.scan_id == scan_id,
                    t.keyframe_meta.c.seq == seq,
                )
            )
            .values(rtabmap_node_id=node_id)
        )

    matched = len(updates)
    skipped = len(keyframes_in) - matched
    logger.info(
        "keyframe nodeID backfill scan_id=%s matched=%d skipped=%d already_set=%d",
        scan_id, matched, skipped, already_set,
    )
    return NodeIdBackfillStats(
        matched=matched,
        skipped_no_match=skipped,
        already_set=int(already_set),
    )
