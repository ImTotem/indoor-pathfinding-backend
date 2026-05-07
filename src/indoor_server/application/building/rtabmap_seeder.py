"""빈 rtabmap.db 를 mp4 + poses.bin + manifest 로 채우는 seeder.

ARKit 클라가 RTAB-Map step 을 생략하고 raw 데이터만 보낸 경우에 사용.
sprint81 의 extract_keyframes / write_rtabmap_db 를 wrap.
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# scripts/sprint81 을 import path 에 추가 (worker 컨테이너 기준 /app/scripts/sprint81)
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPTS_DIR = _REPO_ROOT / "scripts" / "sprint81"
if _SCRIPTS_DIR.exists() and str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def is_empty_rtabmap_db(db_path: Path) -> bool:
    """0 byte 또는 Node 테이블 자체가 없으면 True."""
    if not db_path.exists():
        return True
    if db_path.stat().st_size == 0:
        return True
    import sqlite3
    try:
        con = sqlite3.connect(str(db_path))
        try:
            row = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='Node'"
            ).fetchone()
            if row is None:
                return True
            n = con.execute("SELECT COUNT(*) FROM Node").fetchone()[0]
            return n == 0
        finally:
            con.close()
    except sqlite3.DatabaseError:
        return True


def seed_rtabmap_db_from_video(
    scan_root: Path,
    output_db: Path,
    *,
    sample_hz: float = 6.0,
    jpeg_quality: int = 92,
) -> int:
    """scan.mp4 + poses.bin + manifest.json 로 rtabmap.db 의 Node/Data/Link 채움.

    Args:
        scan_root: scan.mp4, poses.bin, manifest.json 이 있는 디렉터리.
        output_db: 결과 rtabmap.db (기존 파일 있으면 덮어씀).
        sample_hz: keyframe sampling 빈도 (Hz).
        jpeg_quality: image JPEG 인코딩 quality.

    Returns:
        생성된 Node 개수.
    """
    from extract_keyframes import extract  # type: ignore
    from write_rtabmap_db import (  # type: ignore
        Intrinsics,
        write_rgb_only_db,
    )

    manifest_path = scan_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    intr = Intrinsics(
        fx=float(manifest["intrinsics_fx"]),
        fy=float(manifest["intrinsics_fy"]),
        cx=float(manifest["intrinsics_cx"]),
        cy=float(manifest["intrinsics_cy"]),
        width=1920,
        height=1080,
    )

    logger.info(
        "rtabmap_seeder: START scan_root=%s output=%s sample_hz=%.1f",
        scan_root, output_db, sample_hz,
    )

    with tempfile.TemporaryDirectory(prefix="rtabmap_seed_") as tmpdir:
        kf_dir = Path(tmpdir)
        records = extract(
            scan_root=scan_root,
            out_dir=kf_dir,
            sample_hz=sample_hz,
            jpeg_quality=jpeg_quality,
        )
        if not records:
            raise RuntimeError(
                "extract_keyframes: no records produced (pose 매칭 실패 또는 mp4 frame 부족)"
            )

        index_json = kf_dir / "index.json"
        if not index_json.exists():
            raise RuntimeError(f"extract_keyframes did not write index.json at {index_json}")

        write_rgb_only_db(
            keyframes_json=index_json,
            image_dir=kf_dir,
            intrinsics=intr,
            out_db_path=output_db,
            map_id=0,
        )

    logger.info(
        "rtabmap_seeder: DONE nodes=%d output=%s", len(records), output_db,
    )
    return len(records)


async def backfill_keyframe_node_ids_by_position(
    *,
    session,
    scan_id: str,
    rtabmap_db_path: Path,
    max_distance_m: float = 0.5,
) -> dict[str, int]:
    """keyframe_meta.tx/ty/tz ↔ Node.pose translation 의 nearest-neighbor 매칭으로
    rtabmap_node_id 를 채움.

    seed 한 Node.stamp 가 mp4 relative timestamp 라 기존 stamp-based backfill 과
    매칭 안 되는 raw_video_recording 시나리오 전용 fallback.
    """
    import sqlite3
    import struct
    import sqlalchemy as sa
    import numpy as np
    from indoor_server.infrastructure.db import tables as t

    if not rtabmap_db_path.exists():
        return {"matched": 0, "skipped": 0}

    # 1. NULL 인 keyframe_meta 행 (seq, tx, ty, tz)
    null_rows = (
        await session.execute(
            sa.select(
                t.keyframe_meta.c.seq,
                t.keyframe_meta.c.tx,
                t.keyframe_meta.c.ty,
                t.keyframe_meta.c.tz,
            )
            .where(
                sa.and_(
                    t.keyframe_meta.c.scan_id == scan_id,
                    t.keyframe_meta.c.rtabmap_node_id.is_(None),
                )
            )
            .order_by(t.keyframe_meta.c.seq)
        )
    ).fetchall()
    if not null_rows:
        return {"matched": 0, "skipped": 0}

    # 2. rtabmap.db Node.pose translation 추출
    con = sqlite3.connect(str(rtabmap_db_path))
    try:
        node_rows = con.execute(
            "SELECT id, pose FROM Node WHERE pose IS NOT NULL ORDER BY id"
        ).fetchall()
    finally:
        con.close()
    if not node_rows:
        return {"matched": 0, "skipped": len(null_rows)}

    node_ids: list[int] = []
    node_xyz: list[tuple[float, float, float]] = []
    for nid, blob in node_rows:
        if not blob or len(blob) != 48:
            continue
        vals = struct.unpack('<12f', blob)
        # 3x4 row-major: pose[:, 3] = (vals[3], vals[7], vals[11])
        node_ids.append(int(nid))
        node_xyz.append((float(vals[3]), float(vals[7]), float(vals[11])))
    if not node_xyz:
        return {"matched": 0, "skipped": len(null_rows)}

    # 3. AXIS_SWAP: keyframe_meta 는 ARKit (Y-up) frame, Node.pose 는 AXIS_SWAP 적용된 RTABMap (Z-up)
    # rec.tx, ty, tz (ARKit) → AXIS_SWAP @ (tx, ty, tz) → RTABMap world translation
    # AXIS_SWAP_3 = [[1,0,0],[0,0,-1],[0,1,0]]
    def arkit_to_rtab(p: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = p
        return (x, -z, y)

    node_xyz_arr = np.asarray(node_xyz, dtype=np.float64)

    # 4. greedy nearest match
    matched = 0
    skipped = 0
    used = set()
    for r in null_rows:
        seq = int(r.seq)
        kf_xyz = arkit_to_rtab((float(r.tx), float(r.ty), float(r.tz)))
        diffs = node_xyz_arr - np.asarray(kf_xyz)
        dists = np.linalg.norm(diffs, axis=1)
        # masked
        for ui in used:
            dists[ui] = np.inf
        j = int(np.argmin(dists))
        if dists[j] > max_distance_m:
            skipped += 1
            continue
        used.add(j)
        await session.execute(
            sa.update(t.keyframe_meta)
            .where(
                sa.and_(
                    t.keyframe_meta.c.scan_id == scan_id,
                    t.keyframe_meta.c.seq == seq,
                )
            )
            .values(rtabmap_node_id=node_ids[j])
        )
        matched += 1

    logger.info(
        "rtabmap_seeder: position-based backfill scan_id=%s matched=%d skipped=%d max_dist=%.2fm",
        scan_id, matched, skipped, max_distance_m,
    )
    return {"matched": matched, "skipped": skipped}
