"""End-to-end integrity check for a streaming-uploaded scan.

Validates (in order):
1. rtabmap.db (raw, what streaming writer accumulated)
2. rtabmap_reprocessed.db (reprocess pipeline output)
3. scan_metadata.db (sidecar)
4. Postgres ingest counts vs sidecar
5. Node ↔ keyframe timestamp coherence (rtabmap_node_id mapping correctness)
"""
from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path

import asyncpg

SCAN_ID = sys.argv[1] if len(sys.argv) > 1 else "83238578-da2b-4e80-9298-cc84bb30e769"
ROOT = Path("/app/var/storage/scans") / SCAN_ID
RTAB_RAW = ROOT / "rtabmap.db"
RTAB_REP = ROOT / "rtabmap_reprocessed.db"
SIDE = ROOT / "scan_metadata.db"


def hr(title: str) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)


def check_rtabmap_raw() -> dict:
    hr("(1) rtabmap.db (RAW — streaming-pushed)")
    con = sqlite3.connect(str(RTAB_RAW))
    tables = ("Node", "Data", "Link", "Word", "Feature",
              "GlobalDescriptor", "Info", "Statistics", "Admin")
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    print(f"  counts: {counts}")

    img = con.execute(
        "SELECT COUNT(*) FROM Data WHERE image IS NOT NULL AND length(image)>1024"
    ).fetchone()[0]
    depth = con.execute(
        "SELECT COUNT(*) FROM Data WHERE depth IS NOT NULL AND length(depth)>256"
    ).fetchone()[0]
    calib = con.execute(
        "SELECT COUNT(*) FROM Data WHERE calibration IS NOT NULL AND length(calibration)=164"
    ).fetchone()[0]
    pose = con.execute(
        "SELECT COUNT(*) FROM Node WHERE pose IS NOT NULL AND length(pose)=48"
    ).fetchone()[0]
    n = counts["Node"]
    print(
        f"  per-node blob present: image={img}/{n}  depth={depth}/{n}  "
        f"calib={calib}/{n}  pose48B={pose}/{n}"
    )

    node_ids = {r[0] for r in con.execute("SELECT id FROM Node").fetchall()}
    orphan = sum(
        1 for r in con.execute("SELECT from_id, to_id FROM Link").fetchall()
        if r[0] not in node_ids or r[1] not in node_ids
    )
    print(f"  orphan links (from/to missing Node): {orphan}/{counts['Link']}")

    stamps = [
        r[0] for r in con.execute("SELECT stamp FROM Node ORDER BY id").fetchall()
        if r[0] is not None
    ]
    mono = all(stamps[i] <= stamps[i + 1] for i in range(len(stamps) - 1))
    print(
        f"  Node.stamp monotonic by id: {mono}  "
        f"range=[{min(stamps):.3f}, {max(stamps):.3f}]  "
        f"span={max(stamps) - min(stamps):.1f}s"
    )
    con.close()
    return counts


def check_reprocessed(raw_counts: dict) -> None:
    hr("(2) rtabmap_reprocessed.db (REPROCESSED)")
    if not RTAB_REP.exists():
        print("  (missing — reprocess did not run)")
        return
    con = sqlite3.connect(str(RTAB_REP))
    tables = ("Node", "Data", "Link", "Word", "Feature",
              "GlobalDescriptor", "Info", "Statistics", "Admin")
    rcounts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    print(f"  counts: {rcounts}")
    print(
        f"  delta vs raw: Node={rcounts['Node'] - raw_counts['Node']}  "
        f"Link={rcounts['Link'] - raw_counts['Link']:+d}  "
        f"Word=+{rcounts['Word']}  Feature=+{rcounts['Feature']}"
    )

    lc = con.execute("SELECT COUNT(*) FROM Link WHERE type=1").fetchone()[0]
    pr = con.execute("SELECT COUNT(*) FROM Link WHERE type=2").fetchone()[0]
    nb = con.execute("SELECT COUNT(*) FROM Link WHERE type=0").fetchone()[0]
    print(f"  Link types: neighbor=0:{nb}  loop_closure=1:{lc}  proximity=2:{pr}")

    feat_with_depth = con.execute(
        "SELECT COUNT(*) FROM Feature "
        "WHERE depth_x IS NOT NULL AND depth_y IS NOT NULL AND depth_z IS NOT NULL"
    ).fetchone()[0]
    pct = 100 * feat_with_depth / max(rcounts["Feature"], 1)
    print(
        f"  Feature with depth back-projection: {feat_with_depth}/{rcounts['Feature']} "
        f"({pct:.1f}%)"
    )

    row = con.execute("SELECT parameters FROM Info LIMIT 1").fetchone()
    if row and row[0]:
        key_substrings = (
            "Vis/FeatureType", "Kp/DetectorStrategy", "RGBD/Enabled",
            "Mem/IncrementalMemory", "Rtabmap/LoopThr",
        )
        keys = [kv for kv in row[0].split(";") if any(k in kv for k in key_substrings)]
        print(f"  reprocess params: {keys}")
    con.close()


def check_sidecar() -> tuple[dict, int, int, set]:
    hr("(3) scan_metadata.db (SIDECAR)")
    con = sqlite3.connect(str(SIDE))
    con.row_factory = sqlite3.Row
    tables = (
        "scan_session", "keyframe_meta", "poi_mark", "poi_photo",
        "branch_mark", "branch_edge", "interfloor_mark",
    )
    counts: dict = {}
    for t in tables:
        try:
            counts[t] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except sqlite3.OperationalError:
            counts[t] = "(absent)"
    print(f"  counts: {counts}")

    ss = con.execute(
        "SELECT id, state, keyframe_count, device_model, app_version "
        "FROM scan_session"
    ).fetchone()
    print(
        f"  scan_session: id={ss['id'][:8]}...  state={ss['state']}  "
        f"kf_count={ss['keyframe_count']}  device={ss['device_model']}  app={ss['app_version']}"
    )

    kf_total = counts["keyframe_meta"]
    kf_set = con.execute(
        "SELECT COUNT(*) FROM keyframe_meta WHERE rtabmap_node_id IS NOT NULL"
    ).fetchone()[0]
    print(
        f"  keyframe_meta.rtabmap_node_id set: {kf_set}/{kf_total} "
        f"({100*kf_set/max(kf_total,1):.1f}%)"
    )

    raw_con = sqlite3.connect(str(RTAB_RAW))
    raw_node_ids = {r[0] for r in raw_con.execute("SELECT id FROM Node").fetchall()}
    raw_con.close()
    refs = [
        r[0] for r in con.execute(
            "SELECT DISTINCT rtabmap_node_id FROM keyframe_meta "
            "WHERE rtabmap_node_id IS NOT NULL"
        ).fetchall()
    ]
    invalid = [nid for nid in refs if nid not in raw_node_ids]
    print(
        f"  rtabmap_node_id ↔ Node.id distinct values: {len(refs)} total, "
        f"{len(invalid)} invalid"
    )
    if invalid:
        print(f"    invalid (no such Node.id): {invalid[:5]}")
    con.close()
    return counts, kf_total, kf_set, raw_node_ids


async def check_postgres(side_counts: dict, kf_total: int) -> None:
    hr("(4) Postgres ingest")
    c = await asyncpg.connect(
        host="db", port=5432, user="indoor", password="indoor", database="indoor",
    )
    km = await c.fetchval(
        "SELECT count(*) FROM keyframe_meta WHERE scan_id::text=$1", SCAN_ID,
    )
    km_set = await c.fetchval(
        "SELECT count(*) FROM keyframe_meta WHERE scan_id::text=$1 "
        "AND rtabmap_node_id IS NOT NULL", SCAN_ID,
    )
    pm = await c.fetchval(
        "SELECT count(*) FROM poi_mark WHERE scan_id::text=$1", SCAN_ID,
    )
    bm_n = await c.fetchval(
        "SELECT count(*) FROM branch_mark WHERE scan_id::text=$1", SCAN_ID,
    )
    be_n = await c.fetchval(
        "SELECT count(*) FROM branch_edge WHERE scan_id::text=$1", SCAN_ID,
    )
    print(f"  keyframe_meta: PG={km}  sidecar={kf_total}  match={km == kf_total}")
    print(
        f"  keyframe_meta.rtabmap_node_id set: PG={km_set} "
        f"({100*km_set/max(km,1):.1f}%)"
    )
    print(
        f"  poi_mark:    PG={pm}  sidecar={side_counts['poi_mark']}  "
        f"match={pm == side_counts['poi_mark']}"
    )
    print(
        f"  branch_mark: PG={bm_n}  sidecar={side_counts['branch_mark']}  "
        f"match={bm_n == side_counts['branch_mark']}"
    )
    print(
        f"  branch_edge: PG={be_n}  sidecar={side_counts['branch_edge']}  "
        f"match={be_n == side_counts['branch_edge']}"
    )

    row = await c.fetchrow(
        "SELECT seq, rtabmap_node_id, tx, ty, tz FROM keyframe_meta "
        "WHERE scan_id::text=$1 AND seq=2", SCAN_ID,
    )
    if row:
        print(
            f"  sample seq=2 PG: rtabmap_node_id={row['rtabmap_node_id']}  "
            f"t=({row['tx']:.3f}, {row['ty']:.3f}, {row['tz']:.3f})"
        )
    await c.close()


def check_timestamp_coherence(raw_node_ids: set) -> None:
    hr("(5) Node ↔ keyframe TIMESTAMP coherence")
    raw_con = sqlite3.connect(str(RTAB_RAW))
    node_stamp = {r[0]: r[1] for r in raw_con.execute("SELECT id, stamp FROM Node").fetchall()}
    raw_con.close()

    con = sqlite3.connect(str(SIDE))
    con.row_factory = sqlite3.Row
    deltas: list[tuple[int, int, float]] = []
    for r in con.execute(
        "SELECT seq, captured_at, rtabmap_node_id FROM keyframe_meta "
        "WHERE rtabmap_node_id IS NOT NULL ORDER BY seq"
    ).fetchall():
        nid = r["rtabmap_node_id"]
        if nid not in node_stamp:
            continue
        cap_s = r["captured_at"] / 1000.0
        # Node.stamp is RTAB-Map's stamp_s (relative or unix — we just look at relative delta).
        deltas.append((r["seq"], nid, abs(cap_s - node_stamp[nid])))
    con.close()
    if deltas:
        vals = sorted(d[2] for d in deltas)
        print(f"  mappings checked: {len(deltas)}")
        print(
            f"  |delta| range: min={vals[0]:.4f}s  "
            f"median={vals[len(vals)//2]:.4f}s  max={vals[-1]:.4f}s"
        )
        worst = sorted(deltas, key=lambda x: -x[2])[:3]
        print("  worst 3 (seq, node_id, |delta|s):")
        for seq, nid, d in worst:
            print(f"    seq={seq}  node_id={nid}  delta={d:.4f}s")


async def main() -> None:
    raw_counts = check_rtabmap_raw()
    print()
    check_reprocessed(raw_counts)
    print()
    side_counts, kf_total, _kf_set, raw_node_ids = check_sidecar()
    print()
    await check_postgres(side_counts, kf_total)
    print()
    check_timestamp_coherence(raw_node_ids)


if __name__ == "__main__":
    asyncio.run(main())
