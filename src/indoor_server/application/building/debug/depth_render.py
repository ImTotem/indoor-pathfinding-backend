"""depth_render — per-frame depth colormap PNG + depth-aware layout PNG 렌더."""
from __future__ import annotations

import logging

import numpy as np

from indoor_server.domain.building.models import DepthAwareLayout, DepthMap

logger = logging.getLogger(__name__)


def render_depth_colormap_png(
    depth_map: DepthMap,
    original_image: np.ndarray | None = None,
) -> np.ndarray:
    """depth_relative → turbo colormap PNG (좌측: 원본, 우측: depth colormap).

    반환: (H, W*2, 3) BGR numpy 배열.
    scale=None이면 "scale=N/A" 표기.
    """
    import cv2

    depth = depth_map.depth_relative
    h, w = depth.shape

    # 원본 이미지 (없으면 검정 배경)
    if original_image is not None:
        left: np.ndarray = original_image.copy()
        if left.shape[:2] != (h, w):
            left = cv2.resize(left, (w, h))
        if left.ndim == 2:
            left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    else:
        left = np.zeros((h, w, 3), dtype=np.uint8)

    # depth colormap (turbo: 가까움=밝음 / 큰 값=가까움)
    depth_8u = cv2.normalize(
        depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U  # type: ignore[call-overload]
    )
    right = cv2.applyColorMap(depth_8u, cv2.COLORMAP_TURBO)

    # 헤더 정보
    scale_str = f"{depth_map.scale:.4f}" if depth_map.scale is not None else "N/A"
    header_text = (
        f"seq={depth_map.seq:06d}  scale={scale_str}  valid={depth_map.valid_pixel_count}px"
    )
    _put_header(right, header_text, w)

    combined: np.ndarray = np.concatenate([left, right], axis=1)
    return combined


def _put_header(img: np.ndarray, text: str, w: int) -> None:
    """이미지 상단에 반투명 헤더 텍스트 overlay (in-place)."""
    import cv2

    overlay_h = 20
    header_blank = np.zeros((overlay_h, w, 3), dtype=np.uint8)
    img[0:overlay_h] = cv2.addWeighted(img[0:overlay_h], 0.4, header_blank, 0.6, 0)
    cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)


def render_depth_aware_layout_png(
    layout: DepthAwareLayout,
    trajectory_xy: list[tuple[float, float]],
    scan_id: str = "",
    job_id: str = "",
) -> np.ndarray:
    """depth-aware layout → top-down PNG.

    Sprint 18 render_floor_layout_png과 동일 렌더 로직 재사용.
    DepthAwareLayout → FloorLayout-like 인터페이스로 변환 후 전달.
    """
    from indoor_server.application.building.debug.floor_layout_render import (
        render_floor_layout_png,
    )
    from indoor_server.domain.building.models import FloorLayout

    # DepthAwareLayout을 FloorLayout으로 변환 (동일 필드)
    floor_layout = FloorLayout(
        multipolygon=layout.multipolygon,
        z0=layout.z0,
        sub_polygon_count=layout.sub_polygon_count,
        total_area_m2=layout.total_area_m2,
    )
    return render_floor_layout_png(
        layout=floor_layout,
        trajectory_xy=trajectory_xy,
        scan_id=scan_id,
        job_id=job_id,
    )
