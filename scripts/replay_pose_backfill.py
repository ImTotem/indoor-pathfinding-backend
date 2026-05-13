"""Replay pose_backfill against an already-reprocessed scan.

Usage:
    python scripts/replay_pose_backfill.py <scan_id>

Mirrors `BuildService._maybe_reprocess` 's already_reprocessed branch — reads
`rtabmap_reprocessed.db`, extracts optimized poses, and runs `run_full_backfill`
to update keyframe_meta / poi_mark / branch_mark / interfloor_mark in Postgres
with the latest `_backfill_table` formula.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from indoor_server.application.building.pose_backfill import run_full_backfill
from indoor_server.application.building.reprocess_service import extract_optimized_poses
from indoor_server.config import settings


async def main(scan_id: str) -> None:
    rtab_db = settings.storage_root / "scans" / scan_id / "rtabmap_reprocessed.db"
    if not rtab_db.exists():
        raise SystemExit(f"rtabmap_reprocessed.db not found: {rtab_db}")
    print(f"reading optimized poses from {rtab_db}")
    optimized = await asyncio.to_thread(extract_optimized_poses, rtab_db)
    print(f"  extracted {len(optimized)} optimized poses")

    db_url = settings.database_url
    engine = create_async_engine(db_url, future=True)
    try:
        async with AsyncSession(engine) as session:
            async with session.begin():
                stats = await run_full_backfill(
                    session, scan_id=scan_id, optimized=optimized,
                )
        print(f"  backfill stats: {stats}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    asyncio.run(main(sys.argv[1]))
