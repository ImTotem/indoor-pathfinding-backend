#!/usr/bin/env python3
"""
Usage:
    python test_localize.py <domain> <map_id> image1.jpg [image2.jpg ...]
    python test_localize.py <domain> <map_id> image1.jpg --debug

    domain  : host[:port]  e.g. indoor-pathfinding-web.orb.local:5000
    map_id  : building UUID
    --debug : also run mask-debug and match-debug endpoints

Outputs are saved to ./response/<timestamp>/
"""
import argparse
import base64
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests

RESPONSE_DIR = Path(__file__).parent / "response"
DUMMY_INTRINSICS = {"fx": 1, "fy": 1, "cx": 1, "cy": 1}


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def save_image_b64(b64: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))


def _safe_error(text: str, max_len: int = 300) -> str:
    if len(text) > max_len:
        return text[:max_len] + f"... [truncated, total {len(text)} chars]"
    return text


def _strip_b64(obj, max_str: int = 200):
    if isinstance(obj, dict):
        return {k: _strip_b64(v, max_str) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_b64(v, max_str) for v in obj]
    if isinstance(obj, str) and len(obj) > max_str:
        return f"<base64, {len(obj)} chars>"
    return obj


def run_mask_debug(base_url: str, images_b64: list, tag: str):
    print(f"\n[mask-debug] Sending {len(images_b64)} image(s)...")
    resp = requests.post(
        f"{base_url}/v2/debug/mask",
        json={"images": images_b64},
        timeout=60,
    )
    if resp.status_code != 200:
        print(f"[mask-debug] ERROR {resp.status_code}: {_safe_error(resp.text)}")
        return

    data = resp.json()
    out_dir = RESPONSE_DIR / tag / "mask_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    for item in data["results"]:
        i = item["index"]
        save_image_b64(item["original_b64"], out_dir / f"original_{i}.jpg")
        save_image_b64(item["annotated_b64"], out_dir / f"masked_{i}.jpg")
        print(f"  image {i}: {item['persons_detected']} person(s) detected")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"total_images": data["total_images"],
                   "persons_per_image": [r["persons_detected"] for r in data["results"]]}, f, indent=2)
    print(f"[mask-debug] Saved to {out_dir}")


def run_match_debug(base_url: str, map_id: str, images_b64: list, tag: str):
    print(f"\n[match-debug] Sending first image for match visualization...")
    resp = requests.post(
        f"{base_url}/v2/debug/matches",
        json={
            "map_id": map_id,
            "images": images_b64[:1],
            "camera_intrinsics": DUMMY_INTRINSICS,
        },
        timeout=120,
    )
    if resp.status_code != 200:
        print(f"[match-debug] ERROR {resp.status_code}: {_safe_error(resp.text)}")
        return

    data = resp.json()
    out_dir = RESPONSE_DIR / tag / "match_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_image_b64(data["query_b64"], out_dir / "query_keypoints.jpg")
    save_image_b64(data["matches_b64"], out_dir / "matches.jpg")
    if data.get("db_frame_b64"):
        save_image_b64(data["db_frame_b64"], out_dir / "db_frame.jpg")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({k: v for k, v in data.items() if not k.endswith("_b64")}, f, indent=2, ensure_ascii=False)

    print(f"  best_node_id    : {data.get('best_node_id')}")
    print(f"  num_good_matches: {data.get('num_good_matches')}")
    print(f"  num_node_matches: {data.get('num_node_matches')}")
    print(f"  floor_name      : {data.get('floor_name')}")
    print(f"  has_db_image    : {data.get('has_db_image')}")
    print(f"[match-debug] Saved to {out_dir}")


def run_localize(base_url: str, map_id: str, images_b64: list, tag: str, version: str = "v1"):
    path = {"v1": "localize", "v2": "v2/localize", "v3": "v3/localize"}[version]
    endpoint = f"{base_url}/{path}"
    label = f"{version}_localize"
    print(f"\n[{label}] Sending {len(images_b64)} image(s)...")

    resp = requests.post(
        endpoint,
        json={
            "map_id": map_id,
            "images": images_b64,
            "camera_intrinsics": DUMMY_INTRINSICS,
        },
        timeout=120,
    )

    out_dir = RESPONSE_DIR / tag / label
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "status_code": resp.status_code,
        "timestamp": datetime.now().isoformat(),
        "endpoint": endpoint,
        "map_id": map_id,
    }

    if resp.status_code == 200:
        body = resp.json()
        result.update(body)
        conf = body.get("confidence")
        print(f"  confidence : {conf:.4f}" if isinstance(conf, float) else f"  confidence : {conf}")
        print(f"  numMatches : {body.get('numMatches')}")
        print(f"  floorId    : {body.get('floorId')}")
        print(f"  floorLevel : {body.get('floorLevel')}")
        print(f"  pose       : {body.get('pose')}")
    else:
        result["error"] = _safe_error(resp.text)
        print(f"  ERROR {resp.status_code}: {_safe_error(resp.text)}")

    with open(out_dir / "result.json", "w") as f:
        json.dump(_strip_b64(result), f, indent=2, ensure_ascii=False)
    print(f"[{label}] Saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Test SLAM localization endpoints")
    parser.add_argument("domain", help="Server hostname e.g. indoor-pathfinding-web.orb.local")
    parser.add_argument("map_id", help="Building UUID")
    parser.add_argument("images", nargs="+", help="Input image file paths")
    parser.add_argument("--debug", action="store_true", help="Also run mask-debug and match-debug")
    args = parser.parse_args()

    base_url = f"http://{args.domain}:5000/api/slam"

    missing = [p for p in args.images if not os.path.exists(p)]
    if missing:
        print(f"ERROR: Files not found: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"Base URL : {base_url}")
    print(f"Map ID   : {args.map_id}")
    print(f"Images   : {args.images}")
    print(f"Debug    : {args.debug}")

    images_b64 = [encode_image(p) for p in args.images]
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.debug:
        run_mask_debug(base_url, images_b64, tag)
        run_match_debug(base_url, args.map_id, images_b64, tag)

    run_localize(base_url, args.map_id, images_b64, tag, version="v1")
    run_localize(base_url, args.map_id, images_b64, tag, version="v2")
    run_localize(base_url, args.map_id, images_b64, tag, version="v3")

    print(f"\nAll outputs saved under: {RESPONSE_DIR / tag}")


if __name__ == "__main__":
    main()
