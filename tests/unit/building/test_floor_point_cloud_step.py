"""Sprint 46 — FloorPointCloudStep unit tests.

Synthetic-only. 실 RTABMap db 의존하지 않는다.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from indoor_server.application.building.steps.floor_point_cloud import (
    FloorPointCloud,
    FloorPointCloudStep,
    FloorPointCloudStepParams,
)
from indoor_server.domain.building.rtabmap_models import RtabmapDataFrame, RtabmapNode

# ── helpers ─────────────────────────────────────────────────────────────────


def _encode_rtabmap_float_depth(depth: np.ndarray) -> bytes:
    import cv2

    packed = depth.astype(np.float32).copy().view(np.uint8).reshape(
        *depth.shape, 4
    )
    ok, encoded = cv2.imencode(".png", packed)
    assert ok
    return bytes(encoded)


def _calibration_bytes(
    *,
    width: int,
    height: int,
    fx: float = 4.0,
    fy: float = 4.0,
    cx: float | None = None,
    cy: float | None = None,
) -> bytes:
    cx_val = cx if cx is not None else (width - 1) / 2.0
    cy_val = cy if cy is not None else (height - 1) / 2.0
    header = struct.pack(
        "<11i", 0, 23, 5, 0, width, height, 9, 0, 0, 0, 12
    )
    k = struct.pack(
        "<9d",
        fx, 0.0, cx_val,
        0.0, fy, cy_val,
        0.0, 0.0, 1.0,
    )
    local_transform = struct.pack(
        "<12f",
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
    )
    return header + k + local_transform


def _node(node_id: int, *, tx: float, ty: float, tz: float) -> RtabmapNode:
    return RtabmapNode(
        node_id=node_id,
        map_id=0,
        stamp=0.0,
        pose=(
            (1.0, 0.0, 0.0, tx),
            (0.0, 1.0, 0.0, ty),
            (0.0, 0.0, 1.0, tz),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )


def _three_node_floor_frame() -> list[RtabmapNode]:
    """xy 분산 > z 분산 → vertical_axis = z (axis 2). _FloorFrame이 그렇게 잡는다."""
    return [
        _node(1, tx=0.0, ty=0.0, tz=0.0),
        _node(2, tx=2.0, ty=0.0, tz=0.0),
        _node(3, tx=2.0, ty=2.0, tz=0.0),
    ]


def _floor_mask_full(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


# ── tests ──────────────────────────────────────────────────────────────────


def test_floor_pointcloud_happy_path() -> None:
    """identity pose, depth=2 plane, 모든 floor mask=True → 점이 잘 들어가야 한다."""
    h, w = 4, 4
    depth = np.full((h, w), 2.0, dtype=np.float32)
    frames = [
        RtabmapDataFrame(
            node_id=1,
            image_bytes=None,
            depth_bytes=_encode_rtabmap_float_depth(depth),
            calibration_bytes=_calibration_bytes(width=w, height=h),
        )
    ]
    nodes = _three_node_floor_frame()
    masks = {1: _floor_mask_full(h, w)}

    cloud = FloorPointCloudStep(
        FloorPointCloudStepParams(
            pixel_stride=1,
            min_depth_m=0.1,
            max_depth_m=10.0,
            height_tolerance_m=5.0,  # 넓게
            z0_mode="depth_percentile",
        )
    ).run(
        nodes=nodes,
        frames=frames,
        floor_masks_by_node_id=masks,
    )

    assert isinstance(cloud, FloorPointCloud)
    assert cloud.points_xy.shape[0] > 0
    assert cloud.metadata["world_point_count"] > 0
    assert cloud.metadata["valid_depth_pixel_count"] > 0
    assert cloud.metadata["frames_with_floor_mask"] == 1


def test_floor_pointcloud_empty_mask_returns_empty_with_reason() -> None:
    nodes = _three_node_floor_frame()
    cloud = FloorPointCloudStep().run(
        nodes=nodes,
        frames=[],
        floor_masks_by_node_id={},
    )
    assert cloud.points_xy.shape == (0, 2)
    assert cloud.metadata["world_point_count"] == 0
    assert cloud.metadata["empty_reason"] == "no_valid_world_points"


def test_floor_pointcloud_no_nodes_returns_empty() -> None:
    cloud = FloorPointCloudStep().run(
        nodes=[],
        frames=[],
        floor_masks_by_node_id={},
    )
    assert cloud.points_xy.shape == (0, 2)
    assert cloud.metadata["empty_reason"] == "no_nodes"


def test_floor_pointcloud_height_filter_drops_out_of_band() -> None:
    """height_tolerance를 작게 잡고, 점 절반만 통과하도록."""
    h, w = 4, 4
    depth = np.full((h, w), 2.0, dtype=np.float32)
    frame = RtabmapDataFrame(
        node_id=1,
        image_bytes=None,
        depth_bytes=_encode_rtabmap_float_depth(depth),
        calibration_bytes=_calibration_bytes(width=w, height=h),
    )

    nodes = _three_node_floor_frame()
    masks = {1: _floor_mask_full(h, w)}

    # height_tolerance 충분히 작아서 모두 포함하는 케이스 vs 모두 제외하는 케이스 비교
    loose = FloorPointCloudStep(
        FloorPointCloudStepParams(
            pixel_stride=1,
            min_depth_m=0.1,
            max_depth_m=10.0,
            height_tolerance_m=10.0,
            z0_mode="depth_percentile",
        )
    ).run(nodes=nodes, frames=[frame], floor_masks_by_node_id=masks)

    tight = FloorPointCloudStep(
        FloorPointCloudStepParams(
            pixel_stride=1,
            min_depth_m=0.1,
            max_depth_m=10.0,
            height_tolerance_m=0.001,  # 매우 좁음
            z0_mode="depth_percentile",
        )
    ).run(nodes=nodes, frames=[frame], floor_masks_by_node_id=masks)

    assert loose.metadata["world_point_count"] >= tight.metadata["world_point_count"]


def test_floor_pointcloud_axis_convention_world_z_matches_camera_height() -> None:
    """identity pose tz=1.5, depth=2.0 plane, identity local_transform → world z 분포 확인.

    OpenCV camera frame: z is forward, y is down. local_transform = identity 이고
    node pose = identity translation (tx=0,ty=0,tz=1.5) 이면 world point은
    cam point + (0,0,1.5) translation. depth=2.0 plane이므로 world z = 2.0+1.5=3.5
    (실제로는 cam frame z가 world에서 어느 axis로 뻗을지는 pose의 rotation에 따라
    다르다 — identity 회전이면 cam +z = world +z).

    본 테스트는 "swap이 필요해보이지 않는지" 확인용. design §7.3.
    pose가 identity면 world_z 분포는 depth 값 (2.0) + tz (1.5) 근처에 모여야 한다.
    """
    h, w = 6, 6
    depth = np.full((h, w), 2.0, dtype=np.float32)
    frame = RtabmapDataFrame(
        node_id=1,
        image_bytes=None,
        depth_bytes=_encode_rtabmap_float_depth(depth),
        calibration_bytes=_calibration_bytes(width=w, height=h),
    )
    # tz를 명시적으로 다르게 줘서 axis 영향이 분리되도록.
    nodes = [
        _node(1, tx=0.0, ty=0.0, tz=1.5),
        _node(2, tx=2.0, ty=0.0, tz=1.5),
        _node(3, tx=2.0, ty=2.0, tz=1.5),
    ]
    masks = {1: _floor_mask_full(h, w)}

    cloud = FloorPointCloudStep(
        FloorPointCloudStepParams(
            pixel_stride=1,
            min_depth_m=0.1,
            max_depth_m=10.0,
            height_tolerance_m=10.0,
            z0_mode="depth_percentile",
        )
    ).run(nodes=nodes, frames=[frame], floor_masks_by_node_id=masks)

    assert cloud.points_xy.shape[0] > 0
    # node tz=1.5, depth=2.0 (cam +z = world +z) → world z 평균은 3.5 근처여야 한다.
    assert cloud.z_values.mean() == pytest.approx(3.5, abs=0.5)


def test_floor_pointcloud_z0_disagreement_warn_metadata() -> None:
    """node_pose median 기반 floor 추정과 depth percentile이 크게 차이 → warn=True."""
    h, w = 4, 4
    depth = np.full((h, w), 2.0, dtype=np.float32)
    frame = RtabmapDataFrame(
        node_id=1,
        image_bytes=None,
        depth_bytes=_encode_rtabmap_float_depth(depth),
        calibration_bytes=_calibration_bytes(width=w, height=h),
    )
    # node tz=10 (camera_height_offset=1.5라 expected_floor=8.5)
    # 실제 depth로 추정한 world z=12 (10+2) → disagreement = |8.5 - 12| = 3.5m, warn=True
    nodes = [
        _node(1, tx=0.0, ty=0.0, tz=10.0),
        _node(2, tx=2.0, ty=0.0, tz=10.0),
        _node(3, tx=2.0, ty=2.0, tz=10.0),
    ]
    masks = {1: _floor_mask_full(h, w)}

    cloud = FloorPointCloudStep(
        FloorPointCloudStepParams(
            pixel_stride=1,
            min_depth_m=0.1,
            max_depth_m=20.0,
            height_tolerance_m=20.0,
            z0_mode="blended",
            z0_warn_disagreement_m=0.5,
        )
    ).run(nodes=nodes, frames=[frame], floor_masks_by_node_id=masks)

    assert cloud.metadata["z0_disagreement_warn"] is True
    assert cloud.metadata["z0_disagreement_m"] > 0.5


def test_floor_pointcloud_invalid_params_raise() -> None:
    with pytest.raises(ValueError):
        FloorPointCloudStep(FloorPointCloudStepParams(pixel_stride=0))
    with pytest.raises(ValueError):
        FloorPointCloudStep(FloorPointCloudStepParams(min_depth_m=0.0))
    with pytest.raises(ValueError):
        FloorPointCloudStep(
            FloorPointCloudStepParams(min_depth_m=2.0, max_depth_m=1.0)
        )
    with pytest.raises(ValueError):
        FloorPointCloudStep(FloorPointCloudStepParams(height_tolerance_m=0.0))
    with pytest.raises(ValueError):
        FloorPointCloudStep(FloorPointCloudStepParams(z0_depth_percentile=0.0))


def test_floor_pointcloud_resamples_mask_when_shape_differs() -> None:
    """mask는 8x8, depth는 4x4 → nearest-neighbor resample되어야 한다."""
    h, w = 4, 4
    depth = np.full((h, w), 2.0, dtype=np.float32)
    frame = RtabmapDataFrame(
        node_id=1,
        image_bytes=None,
        depth_bytes=_encode_rtabmap_float_depth(depth),
        calibration_bytes=_calibration_bytes(width=w, height=h),
    )
    nodes = _three_node_floor_frame()
    high_res_mask = np.zeros((8, 8), dtype=bool)
    high_res_mask[4:, :] = True  # 하단 절반만 floor
    masks = {1: high_res_mask}

    cloud = FloorPointCloudStep(
        FloorPointCloudStepParams(
            pixel_stride=1,
            min_depth_m=0.1,
            max_depth_m=10.0,
            height_tolerance_m=10.0,
            z0_mode="depth_percentile",
        )
    ).run(nodes=nodes, frames=[frame], floor_masks_by_node_id=masks)

    # 4x4 grid에서 하단 2 row만 살아남아야 한다 → 약 8개의 valid pixel.
    assert cloud.metadata["valid_depth_pixel_count"] == 8
