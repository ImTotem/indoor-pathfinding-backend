"""End-to-end replay: create new floor → streaming push of b86941c0's data
→ finalize (with manifest + sidecar) → build → verify graph.

Validates:
- streaming start/frames/finalize round-trip with real iOS RTAB-Map data
- pose_backfill axis-swap fix puts branch_marks on the floor plane
- map_node graph 와 floor_z0 정합
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import shutil
import sqlite3
import struct
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import asyncpg

REPO = Path(__file__).resolve().parents[1]
SOURCE_SCAN_DIR = REPO / "var/storage/scans/b86941c0-a71c-4b2c-b427-88b3f88781cf"
BUILDING_ID = "e30f31ea-5bbe-42df-9031-fa371bb7a7b3"
BASE = "http://localhost:8080/api/v1"
NEW_FLOOR_LEVEL = 7  # avoid collision with existing levels
K = 5


def post_json(path: str, body: dict, timeout: int = 60) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{BASE}{path}", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def post_multipart(
    path: str, parts: list[tuple[str, str, str, bytes]],
) -> tuple[int, dict]:
    boundary = "----e2e-replay"
    body = io.BytesIO()
    for field, filename, ctype, payload in parts:
        body.write(f"--{boundary}\r\n".encode())
        body.write(
            f'Content-Disposition: form-data; name="{field}"; '
            f'filename="{filename}"\r\n'.encode()
        )
        body.write(f"Content-Type: {ctype}\r\n\r\n".encode())
        body.write(payload)
        body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(f"{BASE}{path}", data=body.getvalue(), method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def b64(v: bytes | None) -> str | None:
    return base64.b64encode(v).decode() if v is not None else None


def main() -> None:
    print("=" * 60)
    print("(1) create new floor")
    print("=" * 60)
    status, resp = post_json(
        f"/buildings/{BUILDING_ID}/floors",
        {"name": "E2E Test 7F", "level": NEW_FLOOR_LEVEL},
    )
    print(f"  POST /floors → {status}: {resp}")
    if status not in (200, 201):
        raise SystemExit("floor create failed")
    new_floor_id = resp["floorId"] if "floorId" in resp else resp["id"]
    print(f"  new_floor_id = {new_floor_id}")
    print()

    print("=" * 60)
    print("(2) /scans/start on new floor")
    print("=" * 60)
    status, resp = post_json(f"/floors/{new_floor_id}/scans/start", {})
    print(f"  → {status}: {resp}")
    assert status == 201, resp
    scan_id = resp["scanId"]
    print()

    print("=" * 60)
    print(f"(3) push frames from b86941c0 in batches of K={K}")
    print("=" * 60)
    src = sqlite3.connect(str(SOURCE_SCAN_DIR / "rtabmap.db"))
    src.row_factory = sqlite3.Row
    nodes = src.execute(
        "SELECT n.id, n.map_id, n.weight, n.stamp, n.pose, n.ground_truth_pose, "
        "       n.velocity, n.label, n.gps, n.env_sensors, "
        "       d.image, d.depth, d.depth_confidence, d.calibration, "
        "       d.scan, d.scan_info, d.user_data "
        "FROM Node n LEFT JOIN Data d ON d.id = n.id ORDER BY n.id"
    ).fetchall()
    links = src.execute(
        "SELECT from_id, to_id, type, transform, information_matrix, user_data "
        "FROM Link ORDER BY from_id, to_id"
    ).fetchall()
    src.close()
    print(f"  source: {len(nodes)} nodes, {len(links)} links")

    batches: list[dict] = []
    for start in range(0, len(nodes), K):
        chunk = nodes[start : start + K]
        batches.append({
            "frames": [
                {
                    "nodeId": int(n["id"]),
                    "mapId": int(n["map_id"]),
                    "weight": int(n["weight"] or 0),
                    "stamp": float(n["stamp"] or 0.0),
                    "pose": b64(n["pose"]),
                    "image": b64(n["image"]),
                    "calibration": b64(n["calibration"]),
                    "depth": b64(n["depth"]),
                    "depthConfidence": b64(n["depth_confidence"]),
                    "groundTruthPose": b64(n["ground_truth_pose"]),
                    "velocity": b64(n["velocity"]),
                    "gps": b64(n["gps"]),
                    "envSensors": b64(n["env_sensors"]),
                    "label": n["label"],
                    "userData": b64(n["user_data"]),
                    "scan": b64(n["scan"]),
                    "scanInfo": b64(n["scan_info"]),
                }
                for n in chunk
            ],
            "links": [],
        })
    batches[-1]["links"] = [
        {
            "fromId": int(lk["from_id"]),
            "toId": int(lk["to_id"]),
            "type": int(lk["type"]),
            "transform": b64(lk["transform"]),
            "informationMatrix": b64(lk["information_matrix"]),
            "userData": b64(lk["user_data"]),
        }
        for lk in links
    ]
    for i, batch in enumerate(batches):
        st, rs = post_json(f"/scans/{scan_id}/frames", batch)
        if st != 200:
            raise SystemExit(f"batch {i} failed: {rs}")
    print(f"  pushed {len(batches)} batches")
    print()

    print("=" * 60)
    print("(4) finalize with manifest + sidecar")
    print("=" * 60)
    # Rewrite sidecar scan_id to match streaming scan_id.
    tmp_side = Path(tempfile.gettempdir()) / f"sidecar_{scan_id}.db"
    shutil.copy(SOURCE_SCAN_DIR / "scan_metadata.db", tmp_side)
    con = sqlite3.connect(str(tmp_side))
    old_id = "B86941C0-A71C-4B2C-B427-88B3F88781CF"
    con.execute("UPDATE scan_session SET id = ? WHERE id = ?", (scan_id, old_id))
    for tbl in ("keyframe_meta", "poi_mark", "poi_photo", "branch_mark",
                "interfloor_mark", "branch_edge"):
        try:
            con.execute(
                f"UPDATE {tbl} SET scan_id = ? WHERE scan_id = ?",
                (scan_id, old_id),
            )
        except sqlite3.OperationalError:
            pass
    con.commit()
    con.close()

    # Reuse the original manifest (rewrite scan_id only).
    manifest = json.loads((SOURCE_SCAN_DIR / "manifest.json").read_text())
    manifest["scan_id"] = scan_id
    manifest_bytes = json.dumps(manifest).encode()

    st, rs = post_multipart(
        f"/scans/{scan_id}/finalize",
        [
            ("manifest", "manifest.json", "application/json", manifest_bytes),
            ("metadata", "scan_metadata.db", "application/octet-stream",
             tmp_side.read_bytes()),
        ],
    )
    print(f"  → {st}: {rs}")
    assert st == 200, rs
    print()

    print("=" * 60)
    print(f"(5) trigger build for new floor {new_floor_id}")
    print("=" * 60)
    st, rs = post_json(f"/floors/{new_floor_id}/build", {})
    print(f"  → {st}: {rs}")
    build_job_id = rs.get("buildJobId")

    # Poll until build completes.
    print()
    print("=" * 60)
    print("(6) poll build status")
    print("=" * 60)
    deadline = time.time() + 300
    last_status = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"{BASE}/floors/{new_floor_id}/process/status", timeout=10,
            ) as r:
                s = json.loads(r.read())
        except Exception as e:
            print(f"  poll error: {e}")
            time.sleep(2)
            continue
        if s != last_status:
            print(f"  [{int(time.time()-deadline+300):3d}s] {s}")
            last_status = s
        if s.get("status") in ("COMPLETED", "FAILED"):
            break
        time.sleep(3)

    print()
    print("=" * 60)
    print("(7) verify graph")
    print("=" * 60)
    asyncio.run(verify_graph(scan_id, new_floor_id))
    print()
    print(f"DONE. new_floor_id={new_floor_id}  scan_id={scan_id}")


async def verify_graph(scan_id: str, floor_id: str) -> None:
    con = await asyncpg.connect(
        host="127.0.0.1", port=5432, user="indoor", password="indoor",
        database="indoor",
    )
    try:
        bj = await con.fetchrow(
            "SELECT state, current_step, progress, failure_reason, counts "
            "FROM build_job WHERE scan_id::text=$1 ORDER BY enqueued_at DESC LIMIT 1",
            scan_id,
        )
        if bj:
            counts = json.loads(bj["counts"]) if bj["counts"] else {}
            print(f"  build_job: state={bj['state']} step={bj['current_step']} "
                  f"progress={bj['progress']} reason={bj['failure_reason']}")
            print(f"  floor_z0:  {counts.get('floor_z0')}")
            print(f"  map_nodes: {counts.get('map_nodes')}  "
                  f"map_edges: {counts.get('map_edges')}")

        rows = await con.fetch(
            "SELECT node_type, label, ST_AsText(geom) AS g FROM map_node "
            "WHERE scan_id::text=$1 ORDER BY node_type, label",
            scan_id,
        )
        print(f"\n  map_node ({len(rows)}):")
        for r in rows:
            print(f"    {r['node_type']:<12} {r['label'] or '-':<10} {r['g']}")

        rows = await con.fetch(
            "SELECT id, tx, ty, tz FROM branch_mark "
            "WHERE scan_id::text=$1 ORDER BY id",
            scan_id,
        )
        zs = [r['tz'] for r in rows]
        if zs:
            print(f"\n  branch_mark Z stats: min={min(zs):.3f} max={max(zs):.3f}"
                  f"  median={sorted(zs)[len(zs)//2]:.3f}")
    finally:
        await con.close()


if __name__ == "__main__":
    main()
