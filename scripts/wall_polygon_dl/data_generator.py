"""Synthetic obstacle-heatmap → polygon-mask data generator.

User-provided pipeline (Sprint 57):
  make_layout(rng, style)                    # 20 floor-plan styles
    └─ axis-aligned: rect union → mitre buffer cleanup
    └─ free-form: rect/circle/diagonal + penetration cap + round buffer
            ↓
  polygon (clean GT)
            ↓
    ├── make_mask                            → TARGET MASK (filled solid)
    └── densify → normals → perturb_outline
                ↓
        add_interior_smear (rect + Gaussian blur)
                ↓
        add_exterior_smear (rect + Gaussian blur)
                ↓
        draw_inferno_stroke (halo + core + faint)
                ↓
                                             → INPUT (raw heatmap, BGR)

Output of `generate_pair(rng, style)`:
  (canvas BGR uint8 700×700, mask uint8 700×700, polygon Nx2 float, style str)
"""
from __future__ import annotations

import cv2
import numpy as np
from matplotlib import colormaps
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

H, W = 700, 700
GRID = 10
inferno_cmap = colormaps["inferno"]


def inferno_bgr(t: float) -> tuple[int, int, int]:
    rgba = inferno_cmap(max(0.0, min(1.0, float(t))))
    return (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))


def snap(v: float, g: int = GRID) -> int:
    return int(round(v / g) * g)


def smooth_1d(x: np.ndarray, sigma: float = 5) -> np.ndarray:
    k = max(3, int(sigma * 4) | 1)
    kernel = np.exp(-((np.arange(k) - k // 2) ** 2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return np.convolve(np.concatenate([x[-k:], x, x[:k]]), kernel, mode="same")[k:-k]


# ---------------------------------------------------------------------------
# 1. Polygon cleanup
# ---------------------------------------------------------------------------
def clean_polygon(geom, min_feature: float = 30, axis_aligned: bool = True) -> np.ndarray | None:
    """Remove sawtooth artifacts via morphological open+close on polygon."""
    if geom.is_empty:
        return None
    if axis_aligned:
        try:
            e = geom.buffer(-min_feature / 2, join_style=2, mitre_limit=20)
            opened = (
                e.buffer(min_feature / 2, join_style=2, mitre_limit=20) if not e.is_empty else geom
            )
        except Exception:
            opened = geom
        try:
            d = opened.buffer(min_feature / 2, join_style=2, mitre_limit=20)
            closed = d.buffer(-min_feature / 2, join_style=2, mitre_limit=20)
        except Exception:
            closed = opened
    else:
        try:
            e = geom.buffer(-min_feature / 4, join_style=1)
            opened = e.buffer(min_feature / 4, join_style=1) if not e.is_empty else geom
        except Exception:
            opened = geom
        try:
            d = opened.buffer(min_feature / 4, join_style=1)
            closed = d.buffer(-min_feature / 4, join_style=1)
        except Exception:
            closed = opened
    if closed.is_empty:
        return None
    if closed.geom_type == "MultiPolygon":
        closed = max(closed.geoms, key=lambda p: p.area)
    coords = np.array(closed.exterior.coords[:-1], dtype=float)
    if axis_aligned:
        coords = np.array([[snap(c[0]), snap(c[1])] for c in coords], dtype=float)
    deduped = []
    min_dist = 2.0 if axis_aligned else 3.0
    for c in coords:
        if not deduped or np.linalg.norm(c - deduped[-1]) >= min_dist:
            deduped.append(c)
    if len(deduped) >= 2 and np.linalg.norm(deduped[0] - deduped[-1]) < min_dist:
        deduped.pop()
    if len(deduped) < 4:
        return None
    coords = np.array(deduped)
    threshold = 0.5 if axis_aligned else 5.0
    cleaned_v = []
    n = len(coords)
    for i in range(n):
        v1 = coords[i] - coords[(i - 1) % n]
        v2 = coords[(i + 1) % n] - coords[i]
        if abs(v1[0] * v2[1] - v1[1] * v2[0]) > threshold:
            cleaned_v.append(coords[i])
    return np.array(cleaned_v) if len(cleaned_v) >= 4 else None


# ---------------------------------------------------------------------------
# 2. Layout generators — 20 styles
# ---------------------------------------------------------------------------
ALL_STYLES = [
    "simple_rect",
    "simple_L",
    "corridor_long",
    "corridor_T",
    "corridor_plus",
    "school_top",
    "school_both",
    "school_partial",
    "corridor_E",
    "corridor_U",
    "corner_room",
    "branching",
    "complex_school",
    "h_shape",
    "cross_with_rooms",
    "diagonal_wing",
    "diagonal_branch",
    "circle_atrium",
    "octagon",
    "pure_circle",
]


def make_layout(rng: np.random.Generator, style: str | None = None):
    if style is None:
        style = rng.choice(ALL_STYLES)
    parts: list[tuple[int, int, int, int]] = []
    cw = snap(int(rng.integers(36, 50)))
    region_w = snap(rng.integers(300, 480))
    region_h = snap(rng.integers(300, 480))
    rx = snap(rng.integers(80, max(81, W - region_w - 80)))
    ry = snap(rng.integers(80, max(81, H - region_h - 80)))

    def add(x1, y1, x2, y2):
        x1, x2 = sorted([snap(x1), snap(x2)])
        y1, y2 = sorted([snap(y1), snap(y2)])
        if x2 > x1 and y2 > y1:
            parts.append((x1, y1, x2, y2))

    if style == "simple_rect":
        w = snap(rng.integers(160, 360))
        h = snap(rng.integers(120, 280))
        add(rx + (region_w - w) // 2, ry + (region_h - h) // 2,
            rx + (region_w - w) // 2 + w, ry + (region_h - h) // 2 + h)
    elif style == "simple_L":
        a = snap(rng.integers(int(region_w * 0.7), region_w))
        b = snap(rng.integers(int(region_h * 0.4), int(region_h * 0.5)))
        c = snap(rng.integers(int(region_w * 0.4), int(region_w * 0.6)))
        d = snap(rng.integers(int(region_h * 0.4), int(region_h * 0.6)))
        if rng.random() < 0.5:
            add(rx, ry, rx + a, ry + b)
            add(rx, ry + b, rx + c, ry + b + d)
        else:
            add(rx, ry, rx + a, ry + b)
            add(rx + a - c, ry + b, rx + a, ry + b + d)
    elif style == "corridor_long":
        if rng.random() < 0.5:
            cl = snap(rng.integers(int(region_w * 0.7), region_w))
            cy = snap(ry + region_h // 2 - cw // 2)
            add(rx, cy, rx + cl, cy + cw)
        else:
            cl = snap(rng.integers(int(region_h * 0.7), region_h))
            cx = snap(rx + region_w // 2 - cw // 2)
            add(cx, ry, cx + cw, ry + cl)
    elif style == "corridor_T":
        cl_h = snap(rng.integers(int(region_w * 0.7), region_w))
        cy = snap(ry + region_h // 4)
        add(rx, cy, rx + cl_h, cy + cw)
        cl_v = snap(rng.integers(int(region_h * 0.5), int(region_h * 0.7)))
        cx = snap(rx + cl_h // 2 - cw // 2)
        add(cx, cy, cx + cw, cy + cl_v)
    elif style == "corridor_plus":
        cl_h = snap(rng.integers(int(region_w * 0.7), region_w))
        cl_v = snap(rng.integers(int(region_h * 0.7), region_h))
        cx_pos = snap(rx + (region_w - cl_h) // 2)
        cy_pos = snap(ry + (region_h - cl_v) // 2)
        cy_h = snap(cy_pos + cl_v // 2 - cw // 2)
        add(cx_pos, cy_h, cx_pos + cl_h, cy_h + cw)
        cx_v = snap(cx_pos + cl_h // 2 - cw // 2)
        add(cx_v, cy_pos, cx_v + cw, cy_pos + cl_v)
    elif style in ("school_top", "school_both", "school_partial"):
        cl = snap(rng.integers(int(region_w * 0.85), region_w))
        cx_main = snap(rx + (region_w - cl) // 2)
        cy_main = snap(ry + region_h // 2 - cw // 2)
        add(cx_main, cy_main, cx_main + cl, cy_main + cw)
        n_slots = int(rng.integers(3, 6))
        slot_w = cl // n_slots
        rh_top = snap(rng.integers(80, 140))
        rh_bot = snap(rng.integers(80, 140))
        skip = 0.30 if style == "school_partial" else 0.0
        for i in range(n_slots):
            sx = cx_main + i * slot_w
            sxe = cx_main + (i + 1) * slot_w if i < n_slots - 1 else cx_main + cl
            if style in ("school_top", "school_both", "school_partial") and rng.random() > skip:
                add(sx, cy_main - rh_top, sxe, cy_main)
            if style != "school_top" and rng.random() > skip:
                add(sx, cy_main + cw, sxe, cy_main + cw + rh_bot)
    elif style == "corridor_E":
        cl_v = snap(rng.integers(int(region_h * 0.85), region_h))
        cx = snap(rx + GRID * 2)
        cy = snap(ry + (region_h - cl_v) // 2)
        add(cx, cy, cx + cw, cy + cl_v)
        cl_b = snap(rng.integers(int(region_w * 0.5), int(region_w * 0.85)))
        for frac in [0.0, 0.5, 1.0]:
            by = snap(cy + frac * (cl_v - cw))
            add(cx, by, cx + cl_b, by + cw)
    elif style == "corridor_U":
        cl_v = snap(rng.integers(int(region_h * 0.7), region_h))
        cy = snap(ry + (region_h - cl_v) // 2)
        cl_h = snap(rng.integers(int(region_w * 0.6), int(region_w * 0.85)))
        cx = snap(rx + (region_w - cl_h) // 2)
        add(cx, cy, cx + cw, cy + cl_v)
        add(cx + cl_h - cw, cy, cx + cl_h, cy + cl_v)
        add(cx, cy + cl_v - cw, cx + cl_h, cy + cl_v)
    elif style == "corner_room":
        cl_h = snap(rng.integers(int(region_w * 0.7), region_w))
        cl_v = snap(rng.integers(int(region_h * 0.7), region_h))
        cy = snap(ry + GRID * 2)
        add(rx, cy, rx + cl_h, cy + cw)
        right = rng.random() < 0.5
        cx_v = (rx + cl_h - cw) if right else rx
        add(cx_v, cy, cx_v + cw, cy + cl_v)
        room_w = snap(rng.integers(80, 130))
        room_h = snap(rng.integers(80, 130))
        if right:
            add(rx + cl_h - room_w, cy + cw, rx + cl_h, cy + cw + room_h)
        else:
            add(rx, cy + cw, rx + room_w, cy + cw + room_h)
    elif style == "branching":
        cl = snap(rng.integers(int(region_w * 0.85), region_w))
        cy = snap(ry + region_h // 2 - cw // 2)
        add(rx, cy, rx + cl, cy + cw)
        n_br = int(rng.integers(4, 7))
        for i in range(n_br):
            bx = snap(rx + (i + 0.5) * cl / n_br - cw // 2)
            bl = snap(rng.integers(80, 200))
            if rng.random() < 0.5:
                add(bx, cy - bl, bx + cw, cy)
            else:
                add(bx, cy + cw, bx + cw, cy + cw + bl)
    elif style == "complex_school":
        cl = snap(rng.integers(int(region_w * 0.85), region_w))
        cx_main = snap(rx + (region_w - cl) // 2)
        cy_main = snap(ry + region_h // 2 - cw // 2)
        add(cx_main, cy_main, cx_main + cl, cy_main + cw)
        sub_cl = snap(rng.integers(int(cl * 0.6), int(cl * 0.9)))
        sub_off_t = snap(rng.integers(60, 110))
        sub_off_b = snap(rng.integers(60, 110))
        sub_cx_t = snap(cx_main + (cl - sub_cl) // 2)
        add(sub_cx_t, cy_main - sub_off_t, sub_cx_t + sub_cl, cy_main - sub_off_t + cw)
        sub_cx_b = snap(cx_main + (cl - sub_cl) // 2)
        add(sub_cx_b, cy_main + cw + sub_off_b - cw, sub_cx_b + sub_cl, cy_main + cw + sub_off_b)
        for f in [0.25, 0.75]:
            cx_link = snap(sub_cx_t + f * sub_cl - cw // 2)
            add(cx_link, cy_main - sub_off_t, cx_link + cw, cy_main)
            add(cx_link, cy_main + cw, cx_link + cw, cy_main + cw + sub_off_b)
    elif style == "h_shape":
        cl_v = snap(rng.integers(int(region_h * 0.85), region_h))
        cy = snap(ry + (region_h - cl_v) // 2)
        cl_h = snap(rng.integers(int(region_w * 0.6), int(region_w * 0.85)))
        cx = snap(rx + (region_w - cl_h) // 2)
        add(cx, cy, cx + cw, cy + cl_v)
        add(cx + cl_h - cw, cy, cx + cl_h, cy + cl_v)
        cy_mid = snap(cy + cl_v // 2 - cw // 2)
        add(cx, cy_mid, cx + cl_h, cy_mid + cw)
    elif style == "cross_with_rooms":
        cl_h = snap(rng.integers(int(region_w * 0.85), region_w))
        cl_v = snap(rng.integers(int(region_h * 0.85), region_h))
        cx_pos = snap(rx + (region_w - cl_h) // 2)
        cy_pos = snap(ry + (region_h - cl_v) // 2)
        cy_h = snap(cy_pos + cl_v // 2 - cw // 2)
        add(cx_pos, cy_h, cx_pos + cl_h, cy_h + cw)
        cx_v = snap(cx_pos + cl_h // 2 - cw // 2)
        add(cx_v, cy_pos, cx_v + cw, cy_pos + cl_v)
        rw = snap(rng.integers(80, 120))
        rh = snap(rng.integers(80, 120))
        for dx, dy in [(0, 0), (cl_h - rw, 0), (0, cl_v - rh), (cl_h - rw, cl_v - rh)]:
            if rng.random() < 0.7:
                add(cx_pos + dx, cy_pos + dy, cx_pos + dx + rw, cy_pos + dy + rh)
    elif style == "diagonal_wing":
        mw = snap(rng.integers(240, 320))
        mh = snap(rng.integers(220, 280))
        mx = snap(rng.integers(80, 200))
        my = snap(rng.integers(100, H - mh - 100))
        main = box(mx, my, mx + mw, my + mh)
        angle = float(rng.uniform(np.pi / 8, np.pi / 3)) * (1 if rng.random() < 0.5 else -1)
        direction = np.array([np.cos(angle), np.sin(angle)])
        perp = np.array([-np.sin(angle), np.cos(angle)])
        length = float(rng.integers(160, 240))
        sy = my + mh // 2 + int(rng.integers(-mh // 6, mh // 6))
        entry = np.array([mx + mw, sy], dtype=float)
        penetration = float(min(60.0, mw * 0.25))
        start = entry - direction * penetration
        end = entry + direction * length
        cwf = float(cw)
        diag = Polygon([start + perp * cwf / 2, end + perp * cwf / 2,
                        end - perp * cwf / 2, start - perp * cwf / 2])
        return clean_polygon(main.union(diag), 25, axis_aligned=False), False
    elif style == "diagonal_branch":
        cl = snap(rng.integers(380, 500))
        rxx = snap(rng.integers(80, max(81, W - cl - 80)))
        cy = snap(rng.integers(150, H - 200))
        main = box(rxx, cy, rxx + cl, cy + cw)
        all_geoms = [main]
        n_br = int(rng.integers(2, 4))
        for i in range(n_br):
            angle = (
                float(rng.uniform(-np.pi / 2.5, -np.pi / 6))
                if rng.random() < 0.5
                else float(rng.uniform(np.pi / 6, np.pi / 2.5))
            )
            direction = np.array([np.cos(angle), np.sin(angle)])
            perp = np.array([-np.sin(angle), np.cos(angle)])
            bl = float(rng.integers(150, 220))
            sx = rxx + (i + 1) * cl / (n_br + 1)
            entry = np.array([sx, cy if direction[1] > 0 else cy + cw], dtype=float)
            sin_a = abs(np.sin(angle))
            penetration = float(min(40.0, (cw / sin_a) * 0.85))
            start = entry - direction * bl
            end = entry + direction * penetration
            cwf = float(cw)
            all_geoms.append(
                Polygon([start + perp * cwf / 2, end + perp * cwf / 2,
                         end - perp * cwf / 2, start - perp * cwf / 2])
            )
        return clean_polygon(unary_union(all_geoms), 25, axis_aligned=False), False
    elif style == "circle_atrium":
        cx_pos = W // 2 + int(rng.integers(-50, 50))
        cy_pos = H // 2 + int(rng.integers(-50, 50))
        radius = float(rng.integers(110, 160))
        npts = 48
        theta = np.linspace(0, 2 * np.pi, npts, endpoint=False)
        circle_pts = [(cx_pos + radius * np.cos(t), cy_pos + radius * np.sin(t)) for t in theta]
        all_geoms = [Polygon(circle_pts)]
        n_arms = int(rng.choice([2, 3, 4]))
        start_angle = float(rng.uniform(0, 2 * np.pi))
        for k in range(n_arms):
            arm_angle = start_angle + k * 2 * np.pi / n_arms
            direction = np.array([np.cos(arm_angle), np.sin(arm_angle)])
            perp = np.array([-np.sin(arm_angle), np.cos(arm_angle)])
            arm_len = float(rng.integers(140, 200))
            entry = np.array([cx_pos, cy_pos], dtype=float) + direction * radius
            penetration = float(min(50, radius * 0.4))
            start = entry - direction * penetration
            end = entry + direction * arm_len
            cwf = float(cw)
            all_geoms.append(
                Polygon([start + perp * cwf / 2, end + perp * cwf / 2,
                         end - perp * cwf / 2, start - perp * cwf / 2])
            )
        return clean_polygon(unary_union(all_geoms), 20, axis_aligned=False), False
    elif style == "octagon":
        cx_pos = W // 2 + int(rng.integers(-30, 30))
        cy_pos = H // 2 + int(rng.integers(-30, 30))
        r = float(rng.integers(140, 220))
        rotation = float(rng.uniform(0, np.pi / 4))
        theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        pts = [(cx_pos + r * np.cos(t + rotation), cy_pos + r * np.sin(t + rotation))
               for t in theta]
        return clean_polygon(Polygon(pts), 20, axis_aligned=False), False
    elif style == "pure_circle":
        cx_pos = W // 2 + int(rng.integers(-30, 30))
        cy_pos = H // 2 + int(rng.integers(-30, 30))
        radius = float(rng.integers(110, 200))
        theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        pts = [(cx_pos + radius * np.cos(t), cy_pos + radius * np.sin(t)) for t in theta]
        return clean_polygon(Polygon(pts), 15, axis_aligned=False), False

    if not parts:
        return None, True
    rects = [box(x1, y1, x2, y2) for x1, y1, x2, y2 in parts]
    union = unary_union(rects)
    if union.geom_type == "MultiPolygon":
        union = max(union.geoms, key=lambda p: p.area)
    return clean_polygon(union, 30, axis_aligned=True), True


# ---------------------------------------------------------------------------
# 3. Polygon → raw heatmap
# ---------------------------------------------------------------------------
def densify(pts: np.ndarray, spacing: float = 2.0) -> np.ndarray:
    out = []
    n = len(pts)
    for i in range(n):
        p1, p2 = pts[i], pts[(i + 1) % n]
        seg_len = np.linalg.norm(p2 - p1)
        k = max(2, int(seg_len / spacing))
        for t in np.linspace(0, 1, k, endpoint=False):
            out.append(p1 + (p2 - p1) * t)
    return np.array(out)


def compute_outward_normals(dense: np.ndarray) -> np.ndarray:
    n = len(dense)
    normals = np.zeros_like(dense)
    for i in range(n):
        tangent = dense[(i + 1) % n] - dense[(i - 1) % n]
        normal = np.array([-tangent[1], tangent[0]])
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal /= norm
        normals[i] = normal
    sa = sum(
        dense[i, 0] * dense[(i + 1) % n, 1] - dense[(i + 1) % n, 0] * dense[i, 1]
        for i in range(n)
    )
    if sa > 0:
        normals = -normals
    return normals


def perturb_outline(dense: np.ndarray, normals: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
    n = len(dense)
    base_amp = float(rng.uniform(2, 6))
    raw = rng.normal(0, 1, n)
    wobble = smooth_1d(raw, sigma=float(rng.uniform(2, 5))) * base_amp
    raw2 = rng.normal(0, 1, n)
    wobble += smooth_1d(raw2, sigma=1.0) * float(rng.uniform(0.3, 1.5))
    for _ in range(int(rng.integers(2, 8))):
        c = int(rng.integers(0, n))
        w = int(rng.integers(6, 25))
        d = float(rng.uniform(6, 18))
        for j in range(-w, w + 1):
            wobble[(c + j) % n] -= d * np.cos(np.pi * j / (2 * w)) ** 2
    for _ in range(int(rng.integers(0, 3))):
        c = int(rng.integers(0, n))
        w = int(rng.integers(4, 12))
        h = float(rng.uniform(3, 10))
        for j in range(-w, w + 1):
            wobble[(c + j) % n] += h * np.cos(np.pi * j / (2 * w)) ** 2
    return dense + normals * wobble[:, None]


def add_interior_smear(canvas: np.ndarray, mask: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
    layer = np.zeros((H, W, 3), dtype=np.float32)
    ys, xs = np.where(mask > 128)
    if len(ys) == 0:
        return canvas
    haze = np.zeros_like(layer)
    for _ in range(int(rng.integers(3, 8))):
        idx = int(rng.integers(0, len(ys)))
        cy, cx = int(ys[idx]), int(xs[idx])
        rw = int(rng.integers(40, 220))
        rh = int(rng.integers(20, 100))
        color = inferno_bgr(float(rng.uniform(0.10, 0.20)))
        x1 = max(0, cx - rw // 2)
        y1 = max(0, cy - rh // 2)
        x2 = min(W, cx + rw // 2)
        y2 = min(H, cy + rh // 2)
        cv2.rectangle(haze, (x1, y1), (x2, y2), color, -1)
    haze = cv2.GaussianBlur(haze, (51, 51), 16.0)
    smudge = np.zeros_like(layer)
    for _ in range(int(rng.integers(40, 130))):
        idx = int(rng.integers(0, len(ys)))
        cy, cx = int(ys[idx]), int(xs[idx])
        rw = int(rng.integers(4, 25))
        rh = int(rng.integers(3, 20))
        color = inferno_bgr(float(rng.uniform(0.10, 0.30)))
        x1 = max(0, cx - rw // 2)
        y1 = max(0, cy - rh // 2)
        x2 = min(W, cx + rw // 2)
        y2 = min(H, cy + rh // 2)
        cv2.rectangle(smudge, (x1, y1), (x2, y2), color, -1)
    smudge = cv2.GaussianBlur(smudge, (11, 11), 3.0)
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    haze = np.where(mask3 > 128, haze, 0).astype(np.uint8)
    smudge = np.where(mask3 > 128, smudge, 0).astype(np.uint8)
    canvas = np.maximum(canvas, haze)
    canvas = np.maximum(canvas, smudge)
    return canvas


def add_exterior_smear(canvas: np.ndarray, mask: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
    layer = np.zeros((H, W, 3), dtype=np.float32)
    band = (cv2.dilate(mask, np.ones((50, 50), np.uint8)) > 0) & (mask == 0)
    ys, xs = np.where(band)
    if len(ys) == 0:
        return canvas
    for _ in range(int(rng.integers(8, 25))):
        idx = int(rng.integers(0, len(ys)))
        cy, cx = int(ys[idx]), int(xs[idx])
        rw = int(rng.integers(15, 50))
        rh = int(rng.integers(10, 40))
        color = inferno_bgr(float(rng.uniform(0.10, 0.22)))
        x1 = max(0, cx - rw // 2)
        y1 = max(0, cy - rh // 2)
        x2 = min(W, cx + rw // 2)
        y2 = min(H, cy + rh // 2)
        cv2.rectangle(layer, (x1, y1), (x2, y2), color, -1)
    layer = cv2.GaussianBlur(layer, (21, 21), 6.0)
    return np.maximum(canvas, layer.astype(np.uint8))


def draw_inferno_stroke(canvas: np.ndarray, perturbed: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    n = len(perturbed)
    h_low = int(rng.integers(2, 4))
    h_high = int(rng.integers(5, 11))
    raw = rng.uniform(0, 1, n)
    sh = smooth_1d(raw, sigma=float(rng.uniform(3, 7)))
    sh = (sh - sh.min()) / (sh.max() - sh.min() + 1e-9)
    halo_t = (h_low + (h_high - h_low) * sh).astype(int)
    core_t = np.maximum(1, halo_t // 3)
    bm = np.ones(n, dtype=bool)
    for _ in range(int(rng.integers(2, 7))):
        c = int(rng.integers(0, n))
        w = int(rng.integers(3, 14))
        for j in range(-w, w + 1):
            bm[(c + j) % n] = False
    halo_b = smooth_1d(rng.uniform(0, 1, n), sigma=4.0)
    halo_b = (halo_b - halo_b.min()) / (halo_b.max() - halo_b.min() + 1e-9)
    halo_b = 0.10 + 0.45 * halo_b
    core_b = smooth_1d(rng.uniform(0, 1, n), sigma=2.5)
    core_b = (core_b - core_b.min()) / (core_b.max() - core_b.min() + 1e-9)
    core_b = 0.55 + 0.40 * core_b
    cv_mask = np.ones(n, dtype=bool)
    for _ in range(int(rng.integers(2, 5))):
        c = int(rng.integers(0, n))
        w = int(rng.integers(20, 70))
        for j in range(-w, w + 1):
            cv_mask[(c + j) % n] = False
    faint = np.ones(n)
    for _ in range(int(rng.integers(1, 4))):
        c = int(rng.integers(0, n))
        w = int(rng.integers(15, 60))
        for j in range(-w, w + 1):
            faint[(c + j) % n] *= 1 - 0.75 * np.cos(np.pi * j / (2 * w)) ** 2
    halo_b *= faint
    core_b *= faint
    for i in range(n):
        if not bm[i]:
            continue
        x, y = int(perturbed[i, 0]), int(perturbed[i, 1])
        cv2.circle(canvas, (x, y), int(halo_t[i]), inferno_bgr(halo_b[i]), -1)
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.8)
    for i in range(n):
        if not bm[i] or not cv_mask[i]:
            continue
        x, y = int(perturbed[i, 0]), int(perturbed[i, 1])
        cv2.circle(canvas, (x, y), int(core_t[i]), inferno_bgr(core_b[i]), -1)
    return canvas


def make_mask(pts: np.ndarray) -> np.ndarray:
    m = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(m, [pts.astype(np.int32)], 255)
    return m


def generate_pair(
    rng: np.random.Generator,
    style: str | None = None,
    disable_smear: bool = False,
):
    """Returns (canvas BGR uint8, mask uint8, polygon Nx2, style str) or None.

    Args:
        disable_smear: if True, skip interior + exterior smear so the canvas
            shows only the perturbed wall stroke. Useful as an augmentation:
            train mixes smear-on and smear-off samples so the model handles
            real RTABMap heatmaps where users may pre-filter low counts.
    """
    polygon = None
    final_style = style
    for attempt in range(20):
        attempt_rng = (
            rng if attempt == 0 else np.random.default_rng(rng.integers(0, 1 << 30) + attempt)
        )
        result = make_layout(attempt_rng, style=style)
        if result and result[0] is not None and len(result[0]) >= 4:
            polygon, _ = result
            break
    if polygon is None:
        return None
    dense = densify(polygon, 2.0)
    normals = compute_outward_normals(dense)
    perturbed = perturb_outline(dense, normals, rng)
    mask = make_mask(polygon)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    if not disable_smear:
        canvas = add_interior_smear(canvas, mask, rng)
        canvas = add_exterior_smear(canvas, mask, rng)
    canvas = draw_inferno_stroke(canvas, perturbed, rng)
    return canvas, mask, polygon, final_style
