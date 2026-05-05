"""render_composite_panel — cv2/numpy 4-패널 시각화."""
from __future__ import annotations

import logging

import numpy as np

from indoor_server.domain.building.enums import NodeType
from indoor_server.domain.building.models import GridOrigin, MapEdgeVO, MapNodeVO

logger = logging.getLogger(__name__)

_PANEL_LONG = 480   # 패널 long side 픽셀
_GUTTER = 8         # 패널 사이 gutter
_HEADER_H = 24      # 상단 header 높이
_FOOTER_H = 16      # 하단 footer 높이

# 색상 (BGR, cv2 기본)
_BG_DARK = (32, 32, 32)           # #202020
_WALKABLE = (240, 240, 240)        # #F0F0F0
_SKEL_BG = (64, 64, 64)           # #404040
_SKEL_CYAN = (255, 255, 0)         # #00FFFF (BGR)
_EDGE_WHITE = (255, 255, 255)
_JUNCTION_RED = (0, 0, 255)        # #FF0000 (BGR)
_ENDPOINT_YELLOW = (0, 255, 255)   # #FFFF00 (BGR)
_CORRIDOR_GRAY = (128, 128, 128)
_POI_MAGENTA = (255, 0, 255)       # #FF00FF (BGR)
_TEXT_BLACK = (0, 0, 0)
_TEXT_WHITE = (255, 255, 255)


def _scale_to_long(img: np.ndarray, long_side: int = _PANEL_LONG) -> np.ndarray:
    """이미지를 long side=long_side 픽셀로 등비 스케일."""
    import cv2

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((long_side, long_side, 3), dtype=np.uint8)
    scale = long_side / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized: np.ndarray = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return resized


def _make_panel_1_walkable(mask: np.ndarray) -> np.ndarray:
    """(1) walkable mask: bg 어둠 + walkable 흰색."""
    h, w = mask.shape
    panel = np.full((h, w, 3), _BG_DARK, dtype=np.uint8)
    panel[mask] = _WALKABLE
    return _scale_to_long(panel)


def _make_panel_2_skeleton(mask: np.ndarray, skeleton_mask: np.ndarray | None) -> np.ndarray:
    """(2) skeleton overlay: walkable 어둠 회색 + skeleton 시안 1px."""
    h, w = mask.shape
    panel = np.full((h, w, 3), _BG_DARK, dtype=np.uint8)
    panel[mask] = _SKEL_BG
    if skeleton_mask is not None and skeleton_mask.shape == mask.shape:
        panel[skeleton_mask] = _SKEL_CYAN
    return _scale_to_long(panel)


def _node_pixel(node: MapNodeVO, origin: GridOrigin) -> tuple[int, int]:
    """world 좌표 → 패널 픽셀 좌표 (y-flip 없음)."""
    px = (node.x - origin.x0) / origin.cell_size
    py = (node.y - origin.y0) / origin.cell_size
    return int(round(px)), int(round(py))


def _make_panel_3_graph(
    mask: np.ndarray,
    nodes: list[MapNodeVO] | None,
    edges: list[MapEdgeVO] | None,
    origin: GridOrigin,
) -> np.ndarray:
    """(3) nodes+edges graph."""
    import cv2

    h, w = mask.shape
    panel = np.full((h, w, 3), (24, 24, 24), dtype=np.uint8)
    panel[mask] = (32, 32, 32)

    if edges:
        _draw_edges(panel, edges, origin)

    if nodes:
        _draw_nodes(panel, nodes, origin)

    # 범례 텍스트 (우상단 1줄)
    legend = "J=RED E=YEL C=GRY P=MAG"
    cv2.putText(panel, legend, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, _TEXT_WHITE, 1, cv2.LINE_AA)

    return _scale_to_long(panel)


def _draw_edges(panel: np.ndarray, edges: list[MapEdgeVO], origin: GridOrigin) -> None:
    import cv2

    for edge in edges:
        if len(edge.polyline) < 2:
            continue
        pts = [
            (
                int(round((x - origin.x0) / origin.cell_size)),
                int(round((y - origin.y0) / origin.cell_size)),
            )
            for x, y, *_ in edge.polyline
        ]
        np_pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(panel, [np_pts], isClosed=False, color=_EDGE_WHITE, thickness=1)


def _draw_nodes(panel: np.ndarray, nodes: list[MapNodeVO], origin: GridOrigin) -> None:
    import cv2

    # Sprint 49: POI/POI_ATTACH는 다른 노드 위에 그려야 시각적으로 가려지지 않는다.
    # 1차 패스: junction/endpoint/corridor 그리기. 2차 패스: POI 큰 마커 위에 덮기.
    h, w = panel.shape[:2]
    poi_pixels: list[tuple[int, int]] = []
    for node in nodes:
        px, py = _node_pixel(node, origin)
        if not (0 <= px < w and 0 <= py < h):
            continue
        if node.node_type == NodeType.JUNCTION:
            cv2.circle(panel, (px, py), 4, _JUNCTION_RED, -1)
        elif node.node_type == NodeType.ENDPOINT:
            cv2.circle(panel, (px, py), 3, _ENDPOINT_YELLOW, -1)
        elif node.node_type == NodeType.POI or node.node_type == NodeType.POI_ATTACH:
            poi_pixels.append((px, py))
        else:
            cv2.circle(panel, (px, py), 2, _CORRIDOR_GRAY, -1)
    # 2차: POI 마젠타 큰 십자 + 외곽 원 (다른 노드 위에 덮기).
    for px, py in poi_pixels:
        cv2.circle(panel, (px, py), 9, _POI_MAGENTA, 2)
        cv2.line(panel, (px - 9, py), (px + 9, py), _POI_MAGENTA, 2)
        cv2.line(panel, (px, py - 9), (px, py + 9), _POI_MAGENTA, 2)


def _make_panel_4_obs(mask: np.ndarray, observation_count: np.ndarray) -> np.ndarray:
    """(4) observation heatmap: log(obs+1) → COLORMAP_INFERNO."""
    import cv2

    obs_f = np.log1p(observation_count.astype(np.float32))
    max_val = obs_f.max()
    if max_val > 0:
        obs_f = obs_f / max_val
    gray = (obs_f * 255).astype(np.uint8)
    heat: np.ndarray = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    return _scale_to_long(heat)


def _add_header(panel: np.ndarray, title: str) -> np.ndarray:
    """패널 상단에 title header 추가."""
    import cv2

    h, w = panel.shape[:2]
    header = np.full((_HEADER_H, w, 3), 255, dtype=np.uint8)
    cv2.putText(header, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _TEXT_BLACK, 1, cv2.LINE_AA)
    return np.vstack([header, panel])


def _pad_to_same_height(panels: list[np.ndarray]) -> list[np.ndarray]:
    """패널 높이를 최대값으로 맞춤 (하단 패딩)."""
    max_h = max(p.shape[0] for p in panels)
    result = []
    for p in panels:
        h, w = p.shape[:2]
        if h < max_h:
            pad = np.zeros((max_h - h, w, 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        result.append(p)
    return result


def _pad_to_same_width(panels: list[np.ndarray]) -> list[np.ndarray]:
    """패널 너비를 최대값으로 맞춤 (우측 패딩)."""
    max_w = max(p.shape[1] for p in panels)
    result = []
    for p in panels:
        h, w = p.shape[:2]
        if w < max_w:
            pad = np.zeros((h, max_w - w, 3), dtype=np.uint8)
            p = np.hstack([p, pad])
        result.append(p)
    return result


def render_composite_panel(
    walkable_mask: np.ndarray,
    observation_count: np.ndarray,
    skeleton_mask: np.ndarray | None,
    nodes: list[MapNodeVO] | None,
    edges: list[MapEdgeVO] | None,
    origin: GridOrigin,
    *,
    scan_id: str = "",
    job_id: str = "",
) -> np.ndarray:
    """
    4-패널 composite PNG 생성.

    반환: (H, W, 3) uint8 BGR.
    패널 배치:
        [1 walkable | 2 skeleton]
        [3 graph    | 4 obs heatmap]
    """
    import cv2

    # 각 패널 생성
    p1 = _add_header(_make_panel_1_walkable(walkable_mask), "(1) Walkable Mask")
    p2 = _add_header(_make_panel_2_skeleton(walkable_mask, skeleton_mask), "(2) Skeleton Overlay")
    p3 = _add_header(
        _make_panel_3_graph(walkable_mask, nodes, edges, origin), "(3) Graph Nodes+Edges"
    )
    p4 = _add_header(_make_panel_4_obs(walkable_mask, observation_count), "(4) Obs Heatmap")

    # 행별 높이 통일
    row1 = _pad_to_same_height([p1, p2])
    row2 = _pad_to_same_height([p3, p4])

    # 열 너비 통일 (4개 전체)
    all4 = _pad_to_same_width(row1 + row2)
    row1 = all4[:2]
    row2 = all4[2:]

    # gutter 삽입 (열 사이)
    gutter_v = np.full((row1[0].shape[0], _GUTTER, 3), 255, dtype=np.uint8)
    top = np.hstack([row1[0], gutter_v, row1[1]])

    gutter_v2 = np.full((row2[0].shape[0], _GUTTER, 3), 255, dtype=np.uint8)
    bottom = np.hstack([row2[0], gutter_v2, row2[1]])

    # 행 사이 gutter
    gutter_h = np.full((_GUTTER, top.shape[1], 3), 255, dtype=np.uint8)
    composite = np.vstack([top, gutter_h, bottom])

    # footer
    n_nodes = len(nodes) if nodes else 0
    n_edges = len(edges) if edges else 0
    cells = int(walkable_mask.sum())
    footer_text = (
        f"scan={scan_id[:8]} job={job_id[:8]} cells={cells} nodes={n_nodes} edges={n_edges}"
    )
    footer = np.full((_FOOTER_H, composite.shape[1], 3), 220, dtype=np.uint8)
    cv2.putText(
        footer, footer_text, (4, 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, _TEXT_BLACK, 1, cv2.LINE_AA
    )
    composite = np.vstack([composite, footer])

    return composite
