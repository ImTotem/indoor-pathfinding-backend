"""Merge smoke test using two streaming-finalized scans.

1) For each source scan dir on disk, push its rtabmap.db rows through
   /scans/start -> /scans/{id}/frames -> /scans/{id}/finalize.
2) Call POST /floors/{id}/scans/merge with the two new READY scan_ids.
3) Verify the merge response + per-source POI/keyframe union into the merged
   scan_id via Postgres.

Source scan dirs are picked so they have non-trivial POI / keyframe counts.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import shutil
import sqlite3
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import asyncpg

REPO = Path(__file__).resolve().parents[1]
FLOOR = "9cddc60e-6cf9-4ada-ad45-b9533f88e1a7"
BASE = "http://localhost:8080/api/v1"
K = 3

# Two source scan dirs we cloned and pushed through streaming.
SOURCES = [
    REPO / "var/storage/scans/72CDE4D6-12C8-4388-8582-E2CFFB54562A",
    REPO / "var/storage/scans/9FCE079B-17E2-4FA4-B7A1-05A9592917FF",
]


def post_json(path: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{BASE}{path}", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def post_multipart_parts(
    path: str, parts: list[tuple[str, str, str, bytes]]
) -> tuple[int, dict]:
    boundary = "----formdata-merge"
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


def push_source_as_streaming(src_dir: Path) -> str:
    """Re-upload one existing scan via the streaming endpoints. Returns the
    new server-assigned scan_id (READY)."""
    src_rtab = src_dir / "rtabmap.db"
    src_side = src_dir / "scan_metadata.db"

    # start
    status, resp = post_json(f"/floors/{FLOOR}/scans/start", {})
    assert status == 201, resp
    scan_id = resp["scanId"]
    print(f"  START: {scan_id}")

    # frames
    src = sqlite3.connect(str(src_rtab))
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
    for i, b in enumerate(batches):
        status, resp = post_json(f"/scans/{scan_id}/frames", b)
        assert status == 200, resp
    print(f"  FRAMES: {len(nodes)} nodes, {len(links)} links")

    # finalize — rewrite scan_id in a sidecar copy
    tmp_side = Path(tempfile.gettempdir()) / f"sidecar_merge_{scan_id}.db"
    shutil.copy(src_side, tmp_side)
    con = sqlite3.connect(str(tmp_side))
    old_id = src_dir.name
    con.execute("UPDATE scan_session SET id = ? WHERE id = ?", (scan_id, old_id))
    for tbl in (
        "keyframe_meta", "poi_mark", "poi_photo", "branch_mark",
        "interfloor_mark", "branch_edge",
    ):
        try:
            con.execute(
                f"UPDATE {tbl} SET scan_id = ? WHERE scan_id = ?",
                (scan_id, old_id),
            )
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
    assert status == 200, resp
    print(
        f"  FINALIZE: keyframes={resp['keyframeCount']}, "
        f"pois={resp['poiMarkCount']}, sha256={resp['payloadSha256'][:16]}..."
    )
    return scan_id


async def pg_counts(scan_ids: list[str]) -> dict:
    con = await asyncpg.connect(
        host="127.0.0.1", port=5432, user="indoor", password="indoor",
        database="indoor",
    )
    out: dict = {}
    for s in scan_ids:
        out[s] = {
            "keyframe_meta": await con.fetchval(
                "SELECT count(*) FROM keyframe_meta WHERE scan_id::text=$1", s
            ),
            "poi_mark": await con.fetchval(
                "SELECT count(*) FROM poi_mark WHERE scan_id::text=$1", s
            ),
            "poi_photo": await con.fetchval(
                "SELECT count(*) FROM poi_photo WHERE scan_id::text=$1", s
            ),
        }
    await con.close()
    return out


def main() -> None:
    print("== push source #1 ==")
    scan_a = push_source_as_streaming(SOURCES[0])
    print("== push source #2 ==")
    scan_b = push_source_as_streaming(SOURCES[1])
    print()
    print("== Pre-merge per-scan Postgres counts ==")
    pre = asyncio.run(pg_counts([scan_a, scan_b]))
    for s, c in pre.items():
        print(f"  {s}: {c}")
    print()

    print("== POST /scans/merge ==")
    status, resp = post_json(
        f"/floors/{FLOOR}/scans/merge",
        {"chunkIds": [scan_a, scan_b]},
    )
    print("MERGE:", status, resp)
    if status != 200:
        return
    merged = resp["activeScanId"]
    print()

    print("== Post-merge counts ==")
    post = asyncio.run(pg_counts([scan_a, scan_b, merged]))
    for s, c in post.items():
        print(f"  {s}: {c}")

    expected_kf = pre[scan_a]["keyframe_meta"] + pre[scan_b]["keyframe_meta"]
    expected_poi = pre[scan_a]["poi_mark"] + pre[scan_b]["poi_mark"]
    print()
    print(
        f"union check: expected kf={expected_kf} pois={expected_poi}  "
        f"got kf={post[merged]['keyframe_meta']} pois={post[merged]['poi_mark']}"
    )
    print("DONE merged_scan_id =", merged)


if __name__ == "__main__":
    main()
