"""덤프 디렉토리에서 nodes+edges만 고해상도 PNG로 렌더 (그래프 구조 검수용)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

_PIXEL_PER_METER = 40  # 그래프 출력 해상도 (4cm/px)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: render_graph_only.py <dump_dir> [--out PATH]", file=sys.stderr)
        return 2
    dump = Path(sys.argv[1])
    out = Path(sys.argv[3]) if "--out" in sys.argv else dump / "graph_only.png"

    grid_npz = np.load(dump / "walkable_grid.npz")
    mask = grid_npz["mask"]
    x0 = float(grid_npz["origin_x0"])
    y0 = float(grid_npz["origin_y0"])
    cell = float(grid_npz["origin_cell_size"])
    grid_h, grid_w = mask.shape

    # world bbox
    x_min = x0
    y_min = y0
    x_max = x0 + grid_w * cell
    y_max = y0 + grid_h * cell
    img_w = int(round((x_max - x_min) * _PIXEL_PER_METER))
    img_h = int(round((y_max - y_min) * _PIXEL_PER_METER))
    img = np.full((img_h, img_w, 3), 18, dtype=np.uint8)

    # walkable mask 옅은 회색으로
    mask_u8 = (mask.astype(np.uint8) * 60)
    mask_resized = cv2.resize(mask_u8, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    bg = np.stack([mask_resized] * 3, axis=2)
    img = np.maximum(img, bg)

    def world_to_px(x: float, y: float) -> tuple[int, int]:
        return (
            int(round((x - x_min) * _PIXEL_PER_METER)),
            int(round((y - y_min) * _PIXEL_PER_METER)),
        )

    edges = json.loads((dump / "edges.json").read_text())
    nodes = json.loads((dump / "nodes.json").read_text())
    node_index = {n["node_id"]: n for n in nodes}

    # edges 흰색 1px
    for e in edges:
        poly = e.get("polyline") or []
        if len(poly) >= 2:
            pts = np.array([world_to_px(p[0], p[1]) for p in poly], dtype=np.int32)
            cv2.polylines(img, [pts], False, (220, 220, 220), 1, cv2.LINE_AA)
        else:
            n1 = node_index.get(e["from"])
            n2 = node_index.get(e["to"])
            if n1 and n2:
                cv2.line(
                    img,
                    world_to_px(n1["x"], n1["y"]),
                    world_to_px(n2["x"], n2["y"]),
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

    # 노드: junction 빨강 r=5, endpoint 노랑 r=4, corridor 회색 r=2, poi 마젠타 r=6
    type_color_radius = {
        "junction": ((30, 30, 230), 5),
        "endpoint": ((40, 220, 240), 4),
        "corridor": ((130, 130, 130), 2),
        "poi": ((230, 60, 230), 6),
        "branch": ((50, 200, 50), 5),
        "poi_attach_virtual": ((140, 80, 200), 4),
    }
    for n in nodes:
        color, r = type_color_radius.get(n["type"], ((180, 180, 180), 2))
        cv2.circle(img, world_to_px(n["x"], n["y"]), r, color, -1, cv2.LINE_AA)

    # legend
    legend = [
        ("JUNCTION", (30, 30, 230)),
        ("ENDPOINT", (40, 220, 240)),
        ("CORRIDOR", (130, 130, 130)),
        ("POI", (230, 60, 230)),
    ]
    pad = 12
    for i, (label, color) in enumerate(legend):
        y = pad + i * 22
        cv2.circle(img, (pad + 6, y + 6), 5, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (pad + 20, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (240, 240, 240), 1, cv2.LINE_AA)

    footer = (f"nodes={len(nodes)} edges={len(edges)} "
              f"world={x_max-x_min:.1f}m x {y_max-y_min:.1f}m  "
              f"cell={cell*100:.0f}cm  scale={_PIXEL_PER_METER}px/m")
    cv2.putText(img, footer, (pad, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imwrite(str(out), img)
    print(f"wrote {out} ({img_w}x{img_h})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
