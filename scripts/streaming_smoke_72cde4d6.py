"""Real-data smoke test of the streaming flow.

Take 72CDE4D6/rtabmap.db + scan_metadata.db, push the rows in K=3 batches
through /scans/start -> /scans/{id}/frames -> /scans/{id}/finalize, then
verify the resulting rtabmap.db is byte-equivalent (per-row blob hash) to
the source.

Run on host (not inside container) — talks to localhost:8080.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import sqlite3
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC_DIR = REPO / "var/storage/scans/72CDE4D6-12C8-4388-8582-E2CFFB54562A"
SRC_RTAB = SRC_DIR / "rtabmap.db"
SRC_SIDE = SRC_DIR / "scan_metadata.db"
FLOOR = "9cddc60e-6cf9-4ada-ad45-b9533f88e1a7"
BASE = "http://localhost:8080/api/v1"
K = 3


def post_json(path: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{BASE}{path}", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def post_multipart_parts(
    path: str, parts: list[tuple[str, str, str, bytes]]
) -> tuple[int, dict]:
    """parts: list of (field_name, filename, content_type, payload)."""
    boundary = "----formdata-smoke"
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
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def b64(v: bytes | None) -> str | None:
    return base64.b64encode(v).decode() if v is not None else None


def hexh(b: bytes | None) -> str:
    return hashlib.sha256(b).hexdigest()[:16] if b else "<null>"


def main() -> None:
    # 1. Load source rows.
    src = sqlite3.connect(str(SRC_RTAB))
    src.row_factory = sqlite3.Row
    nodes = src.execute(
        "SELECT n.id, n.map_id, n.weight, n.stamp, n.pose, n.ground_truth_pose, "
        "       n.velocity, n.label, n.gps, n.env_sensors, "
        "       d.image, d.depth, d.depth_confidence, d.calibration, "
        "       d.scan, d.scan_info, d.user_data "
        "FROM Node n LEFT JOIN Data d ON d.id = n.id "
        "ORDER BY n.id"
    ).fetchall()
    links = src.execute(
        "SELECT from_id, to_id, type, transform, information_matrix, user_data "
        "FROM Link ORDER BY from_id, to_id"
    ).fetchall()
    print(f"source: Node/Data={len(nodes)} rows, Link={len(links)} rows")
    src.close()

    # 2. /scans/start
    status, resp = post_json(f"/floors/{FLOOR}/scans/start", {})
    print("START:", status, resp)
    assert status == 201, resp
    scan_id = resp["scanId"]

    # 3. K-batch frames; all links in the final batch.
    batches: list[dict] = []
    for start in range(0, len(nodes), K):
        chunk = nodes[start : start + K]
        batches.append(
            {
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
            }
        )
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
        status, resp = post_json(f"/scans/{scan_id}/frames", batch)
        print(f"FRAMES batch {i} (K={len(batch['frames'])}, links={len(batch['links'])}):", status, resp)
        assert status == 200, resp

    # Idempotency: resend batch 0 — every nodeId already present, should skip all.
    status, resp = post_json(f"/scans/{scan_id}/frames", batches[0])
    print("FRAMES retry batch 0:", status, resp)
    assert resp["framesApplied"] == 0, resp
    assert resp["framesSkipped"] == len(batches[0]["frames"]), resp

    # 4. Compare on-disk rtabmap.db to source.
    out_rtab = REPO / "var/storage/scans" / scan_id / "rtabmap.db"
    out = sqlite3.connect(str(out_rtab))
    out_counts = {
        t: out.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ["Node", "Data", "Link"]
    }
    print("output rtabmap.db counts:", out_counts)
    assert out_counts["Node"] == len(nodes), out_counts
    assert out_counts["Data"] == len(nodes), out_counts
    assert out_counts["Link"] == len(links), out_counts

    src = sqlite3.connect(str(SRC_RTAB))
    src_rows = src.execute(
        "SELECT n.id, n.pose, n.stamp, d.image, d.depth, d.calibration "
        "FROM Node n LEFT JOIN Data d ON d.id = n.id ORDER BY n.id"
    ).fetchall()
    out_rows = out.execute(
        "SELECT n.id, n.pose, n.stamp, d.image, d.depth, d.calibration "
        "FROM Node n LEFT JOIN Data d ON d.id = n.id ORDER BY n.id"
    ).fetchall()
    mismatches = 0
    for s, o in zip(src_rows, out_rows):
        for i, col in enumerate(["id", "pose", "stamp", "image", "depth", "calibration"]):
            a, b = s[i], o[i]
            if a != b:
                mismatches += 1
                aa = hexh(a) if isinstance(a, (bytes, bytearray)) else a
                bb = hexh(b) if isinstance(b, (bytes, bytearray)) else b
                print(f"  MISMATCH node={s[0]} col={col}: src={aa}  out={bb}")
    print(f"per-node blob comparison: {len(src_rows)} nodes, mismatches={mismatches}")

    src_links = src.execute(
        "SELECT from_id, to_id, type, transform, information_matrix "
        "FROM Link ORDER BY from_id, to_id, type"
    ).fetchall()
    out_links = out.execute(
        "SELECT from_id, to_id, type, transform, information_matrix "
        "FROM Link ORDER BY from_id, to_id, type"
    ).fetchall()
    link_mis = sum(1 for a, b in zip(src_links, out_links) if a != b)
    print(f"per-link blob comparison: {len(src_links)} links, mismatches={link_mis}")
    src.close()
    out.close()

    # 5. Rewrite scan_id in sidecar copy + POST /finalize.
    tmp_side = Path(tempfile.gettempdir()) / f"sidecar_{scan_id}.db"
    shutil.copy(SRC_SIDE, tmp_side)
    con = sqlite3.connect(str(tmp_side))
    old_id = "72CDE4D6-12C8-4388-8582-E2CFFB54562A"
    con.execute("UPDATE scan_session SET id = ? WHERE id = ?", (scan_id, old_id))
    for tbl in (
        "keyframe_meta",
        "poi_mark",
        "poi_photo",
        "branch_mark",
        "interfloor_mark",
        "branch_edge",
    ):
        try:
            con.execute(f"UPDATE {tbl} SET scan_id = ? WHERE scan_id = ?", (scan_id, old_id))
        except sqlite3.OperationalError:
            pass
    con.commit()
    con.close()

    manifest_body = json.dumps({
        "metadata_version": 9,
        "scan_id": scan_id,
        "client_app_version": "smoke",
        "mode": "live_rtabmap",
        "keyframes_included": False,
        "keyframe_image_source": "rtabmap_db",
        "poi_image_source": "poi_photo_image_blob",
        "rtabmap_accepted_frame_count": 0,
        "rtabmap_reprocessed": False,
        "sidecar_keyframe_meta_count": len(nodes),
        "dropped_reject_frame_image_count": 0,
    }).encode()
    status, resp = post_multipart_parts(
        f"/scans/{scan_id}/finalize",
        [
            ("manifest", "manifest.json", "application/json", manifest_body),
            ("metadata", "scan_metadata.db", "application/octet-stream", tmp_side.read_bytes()),
        ],
    )
    print("FINALIZE:", status, resp)
    assert status == 200, resp

    # 6. Verify Postgres metadata ingestion.
    import asyncio
    import asyncpg

    async def check_pg() -> None:
        con = await asyncpg.connect(
            host="127.0.0.1", port=5432, user="indoor", password="indoor", database="indoor"
        )
        km = await con.fetchval(
            "SELECT count(*) FROM keyframe_meta WHERE scan_id::text=$1", scan_id
        )
        pm = await con.fetchval(
            "SELECT count(*) FROM poi_mark WHERE scan_id::text=$1", scan_id
        )
        ss = await con.fetchrow(
            "SELECT keyframe_count, state FROM scan_session WHERE scan_id::text=$1",
            scan_id,
        )
        fs = await con.fetchrow(
            "SELECT status, active, file_size FROM floor_scan WHERE scan_id::text=$1",
            scan_id,
        )
        sha = await con.fetchval(
            "SELECT payload_sha256 FROM scan_ingest WHERE scan_id::text=$1", scan_id
        )
        await con.close()
        print(
            f"  postgres keyframe_meta={km}  poi_mark={pm}  "
            f"scan_session={dict(ss) if ss else None}  "
            f"floor_scan={dict(fs) if fs else None}  "
            f"sha256={sha[:16] if sha else None}..."
        )

    asyncio.run(check_pg())
    print()
    print("DONE — smoke test scan_id =", scan_id)


if __name__ == "__main__":
    main()
