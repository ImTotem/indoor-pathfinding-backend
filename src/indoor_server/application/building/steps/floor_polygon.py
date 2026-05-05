"""FloorPolygonSimplifyStep — floor mask → 단순화된 polygon list."""
from __future__ import annotations

import numpy as np

from indoor_server.application.building.steps.floor_segmentation import KeyframeMasks
from indoor_server.domain.building.models import FloorPolygon

# ASSUMPTION: min_area_px = 500 (작은 spec/노이즈 제거)
_DEFAULT_MIN_AREA_PX = 500
_DEFAULT_EPSILON_RATIO = 0.01  # 1% arcLength
_DEFAULT_HORIZON_TOP_FRACTION = 0.30  # 상위 30% horizon 마스킹


class FloorPolygonSimplifyStep:
    """
    per-frame floor mask → 단순화된 FloorPolygon 목록.

    알고리즘:
        1. horizon mask: floor_mask 상위 horizon_top_fraction 행을 False로 마스킹
           (horizon 근처 픽셀은 back-projection 시 far away spike 유발)
        2. cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_NONE) +
           cv2.approxPolyDP(epsilon = epsilon_ratio * arcLength, closed=True)
    """

    def __init__(
        self,
        epsilon_ratio: float = _DEFAULT_EPSILON_RATIO,
        min_area_px: int = _DEFAULT_MIN_AREA_PX,
        horizon_top_fraction: float = _DEFAULT_HORIZON_TOP_FRACTION,
    ) -> None:
        self._epsilon_ratio = epsilon_ratio
        self._min_area_px = min_area_px
        self._horizon_top_fraction = horizon_top_fraction

    def run(self, all_masks: list[KeyframeMasks]) -> list[FloorPolygon]:
        """
        all_masks: FloorSegmentationStep 출력 전량.
        반환: 모든 프레임의 FloorPolygon list (seq 오름차순).
        """
        import cv2

        result: list[FloorPolygon] = []
        for km in all_masks:
            frame_polys = self._process_frame(km, cv2)
            result.extend(frame_polys)
        return result

    def _process_frame(self, km: KeyframeMasks, cv2: object) -> list[FloorPolygon]:
        # Fix 1: horizon mask — 상위 fraction 행을 floor 아닌 것으로 간주
        floor_mask = km.floor_mask.copy()
        h, w = floor_mask.shape
        horizon_rows = int(h * self._horizon_top_fraction)
        if horizon_rows > 0:
            floor_mask[:horizon_rows, :] = False

        mask_u8 = floor_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(  # type: ignore[attr-defined]
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE  # type: ignore[attr-defined]
        )
        polys: list[FloorPolygon] = []
        for idx, cnt in enumerate(contours):
            area = float(cv2.contourArea(cnt))  # type: ignore[attr-defined]
            if area < self._min_area_px:
                continue
            arc = float(cv2.arcLength(cnt, True))  # type: ignore[attr-defined]
            epsilon = self._epsilon_ratio * arc
            approx = cv2.approxPolyDP(cnt, epsilon, True)  # type: ignore[attr-defined]
            vertices = [(int(p[0][0]), int(p[0][1])) for p in approx]
            if len(vertices) < 3:
                continue
            polys.append(
                FloorPolygon(
                    seq=km.seq,
                    polygon_index=idx,
                    vertices_px=vertices,
                    image_w=w,
                    image_h=h,
                    area_px=area,
                    pose_matrix=km.pose_matrix,
                    tx=km.tx,
                    ty=km.ty,
                    tz=km.tz,
                )
            )
        return polys
