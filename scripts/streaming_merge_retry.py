"""Just retry the merge endpoint on the two already-READY source scans."""
from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request

import asyncpg

FLOOR = "9cddc60e-6cf9-4ada-ad45-b9533f88e1a7"
BASE = "http://localhost:8080/api/v1"
SCAN_A = "5c21c939-f423-4baf-865d-a6d3e032a019"
SCAN_B = "1ab604d5-b167-4089-8709-b9d2eedc7d2e"


def post_json(path: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{BASE}{path}", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


async def counts(scan_ids: list[str]) -> dict:
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
    print("Pre-merge:")
    pre = asyncio.run(counts([SCAN_A, SCAN_B]))
    for s, c in pre.items():
        print(f"  {s}: {c}")

    print()
    print("POST /scans/merge ...")
    status, resp = post_json(
        f"/floors/{FLOOR}/scans/merge",
        {"chunkIds": [SCAN_A, SCAN_B]},
    )
    print("MERGE:", status, resp)
    if status != 200:
        return
    merged = resp["activeScanId"]

    print()
    print("Post-merge:")
    post = asyncio.run(counts([SCAN_A, SCAN_B, merged]))
    for s, c in post.items():
        print(f"  {s}: {c}")
    expected_kf = pre[SCAN_A]["keyframe_meta"] + pre[SCAN_B]["keyframe_meta"]
    expected_poi = pre[SCAN_A]["poi_mark"] + pre[SCAN_B]["poi_mark"]
    print()
    print(
        f"union check: expected kf>={expected_kf} pois>={expected_poi}  "
        f"got kf={post[merged]['keyframe_meta']} pois={post[merged]['poi_mark']}"
    )
    print("DONE merged_scan_id =", merged)


if __name__ == "__main__":
    main()
