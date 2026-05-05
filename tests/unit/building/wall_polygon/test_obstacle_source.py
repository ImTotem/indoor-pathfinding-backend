"""Sprint 51 (Sprint 52 C-2 metadata cases) — ObstacleSourceStep unit tests."""
from __future__ import annotations

import struct

import numpy as np

from indoor_server.application.building.steps.wall_polygon.obstacle_source import (
    ObstacleSourceStep,
    ObstacleSourceStepParams,
)
from indoor_server.domain.building.rtabmap_models import (
    RtabmapDataFrame,
    RtabmapNode,
)


def test_obstacle_source_empty_inputs_returns_empty_heatmap() -> None:
    step = ObstacleSourceStep()
    heatmap = step.run(
        nodes=[],
        frames=[],
        floor_masks_by_node_id={},
        z0=0.0,
    )
    assert heatmap.metadata.get("world_obstacle_point_count", -1) == 0
    assert heatmap.metadata.get("empty_reason") == "no_nodes_or_frames"
    assert heatmap.counts.shape == (1, 1)
    # Sprint 52 C-2: fields exist even on empty input.
    assert heatmap.metadata.get("frames_with_floor_mask") == 0
    assert heatmap.metadata.get("mask_coverage_ratio") == 0.0


def _identity_pose() -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _make_node(node_id: int) -> RtabmapNode:
    return RtabmapNode(
        node_id=node_id,
        map_id=0,
        stamp=float(node_id),
        pose=_identity_pose(),
    )


def _make_node_at(node_id: int, *, tx: float, ty: float, tz: float) -> RtabmapNode:
    return RtabmapNode(
        node_id=node_id,
        map_id=0,
        stamp=float(node_id),
        pose=(
            (1.0, 0.0, 0.0, tx),
            (0.0, 1.0, 0.0, ty),
            (0.0, 0.0, 1.0, tz),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )


def _three_node_floor_frame() -> list[RtabmapNode]:
    return [
        _make_node_at(1, tx=0.0, ty=0.0, tz=0.0),
        _make_node_at(2, tx=2.0, ty=0.0, tz=0.0),
        _make_node_at(3, tx=2.0, ty=2.0, tz=0.0),
    ]


def _make_frame_no_depth(node_id: int) -> RtabmapDataFrame:
    return RtabmapDataFrame(
        node_id=node_id,
        image_bytes=None,
        depth_bytes=None,
        calibration_bytes=None,
    )


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


def _make_frame_with_depth(node_id: int, depth: np.ndarray) -> RtabmapDataFrame:
    h, w = depth.shape
    return RtabmapDataFrame(
        node_id=node_id,
        image_bytes=None,
        depth_bytes=_encode_rtabmap_float_depth(depth),
        calibration_bytes=_calibration_bytes(width=w, height=h),
    )


def test_obstacle_source_no_depth_frames_yields_zero_mask_ratio() -> None:
    """Frames have no depth data → frames_with_depth=0, mask ratio=0."""
    step = ObstacleSourceStep()
    nodes = [_make_node(i) for i in range(3)]
    frames = [_make_frame_no_depth(i) for i in range(3)]
    heatmap = step.run(
        nodes=nodes,
        frames=frames,
        floor_masks_by_node_id={
            i: np.ones((4, 4), dtype=bool) for i in range(3)
        },
        z0=0.0,
    )
    # frames_with_depth=0 because depth_bytes is None for every frame; the
    # mask ratio collapses to 0 since the denominator is 0.
    assert heatmap.metadata["frames_with_depth"] == 0
    assert heatmap.metadata["frames_with_floor_mask"] == 0
    assert heatmap.metadata["mask_coverage_ratio"] == 0.0


def test_obstacle_source_metadata_keys_exist() -> None:
    """C-2: metadata always exposes the new keys regardless of branch taken."""
    step = ObstacleSourceStep()
    heatmap = step.run(
        nodes=[],
        frames=[],
        floor_masks_by_node_id={},
        z0=0.0,
    )
    for key in ("frames_with_floor_mask", "mask_coverage_ratio"):
        assert key in heatmap.metadata, f"missing metadata key: {key}"


def test_obstacle_source_direct_mask_uses_only_direct_obstacle_pixels() -> None:
    depth = np.ones((4, 4), dtype=np.float32)
    direct_mask = np.zeros((4, 4), dtype=bool)
    direct_mask[0, :] = True

    heatmap = ObstacleSourceStep(
        ObstacleSourceStepParams(
            pixel_stride=1,
            min_depth_m=0.1,
            max_depth_m=10.0,
            height_above_floor_min_m=0.0,
            height_above_floor_max_m=10.0,
            mask_mode="direct_mask",
        )
    ).run(
        nodes=_three_node_floor_frame(),
        frames=[_make_frame_with_depth(1, depth)],
        floor_masks_by_node_id={1: np.ones((4, 4), dtype=bool)},
        obstacle_masks_by_node_id={1: direct_mask},
        z0=0.0,
    )

    assert heatmap.metadata["mask_mode"] == "direct_mask"
    assert heatmap.metadata["mask_source"] == "direct_obstacle_mask"
    assert heatmap.metadata["frames_with_floor_mask"] == 0
    assert heatmap.metadata["frames_with_obstacle_mask"] == 1
    assert heatmap.metadata["mask_coverage_ratio"] == 1.0
    assert heatmap.metadata["world_obstacle_point_count"] == int(direct_mask.sum())


def test_obstacle_source_direct_mask_skips_frames_without_direct_mask() -> None:
    depth = np.ones((4, 4), dtype=np.float32)

    heatmap = ObstacleSourceStep(
        ObstacleSourceStepParams(
            pixel_stride=1,
            min_depth_m=0.1,
            max_depth_m=10.0,
            mask_mode="direct_mask",
        )
    ).run(
        nodes=_three_node_floor_frame(),
        frames=[_make_frame_with_depth(1, depth)],
        floor_masks_by_node_id={1: np.ones((4, 4), dtype=bool)},
        obstacle_masks_by_node_id={},
        z0=0.0,
    )

    assert heatmap.metadata["mask_mode"] == "direct_mask"
    assert heatmap.metadata["frames_with_depth"] == 1
    assert heatmap.metadata["frames_with_obstacle_mask"] == 0
    assert heatmap.metadata["world_obstacle_point_count"] == 0
    assert heatmap.metadata["empty_reason"] == "no_obstacle_points"
