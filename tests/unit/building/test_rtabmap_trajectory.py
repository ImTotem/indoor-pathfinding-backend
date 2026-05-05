from __future__ import annotations

import math
import struct
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest
from shapely.geometry import shape

from indoor_server.application.building.pipeline import BuildPipeline
from indoor_server.application.building.steps.rtabmap_trajectory import (
    RtabmapTrajectoryRoadStep,
)
from indoor_server.domain.building.enums import BuildStep
from indoor_server.domain.building.rtabmap_models import (
    RtabmapDataFrame,
    RtabmapFeaturePoint,
    RtabmapLink,
    RtabmapNode,
)
from indoor_server.infrastructure.ml.protocol import SegmentationOutput


def _pose(tx: float, ty: float, tz: float = 0.0) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    return (
        (1.0, 0.0, 0.0, tx),
        (0.0, 1.0, 0.0, ty),
        (0.0, 0.0, 1.0, tz),
        (0.0, 0.0, 0.0, 1.0),
    )


def _node(node_id: int, x: float, y: float, z: float = 0.0) -> RtabmapNode:
    return RtabmapNode(
        node_id=node_id,
        map_id=0,
        stamp=1000.0 + node_id,
        pose=_pose(x, y, z),
    )


def _link(a: int, b: int, link_type: int = 0) -> RtabmapLink:
    return RtabmapLink(from_id=a, to_id=b, link_type=link_type, transform=None)


def _rot(x: float, y: float, angle_deg: float) -> tuple[float, float]:
    rad = math.radians(angle_deg)
    return (
        x * math.cos(rad) - y * math.sin(rad),
        x * math.sin(rad) + y * math.cos(rad),
    )


def _feature(node_id: int, word_id: int, local_y: float) -> RtabmapFeaturePoint:
    return RtabmapFeaturePoint(
        node_id=node_id,
        word_id=word_id,
        pixel_x=0.0,
        pixel_y=0.0,
        local_xyz=(0.0, local_y, 0.0),
    )


def _rtabmap_depth_frame(node_id: int) -> RtabmapDataFrame:
    import cv2

    depth = np.full((4, 4), 1.0, dtype=np.float32)
    packed = depth.copy().view(np.uint8).reshape(4, 4, 4)
    ok, encoded = cv2.imencode(".png", packed)
    assert ok
    header = struct.pack("<11i", 0, 23, 5, 0, 4, 4, 9, 0, 0, 0, 12)
    k = struct.pack(
        "<9d",
        4.0,
        0.0,
        1.5,
        0.0,
        4.0,
        1.5,
        0.0,
        0.0,
        1.0,
    )
    local_transform = struct.pack(
        "<12f",
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    )
    return RtabmapDataFrame(
        node_id=node_id,
        image_bytes=None,
        depth_bytes=bytes(encoded),
        calibration_bytes=header + k + local_transform,
    )


def _rtabmap_image_depth_frame(node_id: int) -> RtabmapDataFrame:
    import cv2

    size = 16
    depth = np.full((size, size), 1.0, dtype=np.float32)
    packed = depth.copy().view(np.uint8).reshape(size, size, 4)
    ok, depth_encoded = cv2.imencode(".png", packed)
    assert ok
    header = struct.pack("<11i", 0, 23, 5, 0, size, size, 9, 0, 0, 0, 12)
    k = struct.pack(
        "<9d",
        16.0,
        0.0,
        7.5,
        0.0,
        16.0,
        7.5,
        0.0,
        0.0,
        1.0,
    )
    local_transform = struct.pack(
        "<12f",
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    )
    image = np.zeros((size, size, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return RtabmapDataFrame(
        node_id=node_id,
        image_bytes=bytes(encoded),
        depth_bytes=bytes(depth_encoded),
        calibration_bytes=header + k + local_transform,
    )


def test_rtabmap_trajectory_step_uses_unique_neighbor_links() -> None:
    nodes = [
        _node(1, 0.0, 0.0),
        _node(2, 2.0, 0.0),
        _node(3, 2.0, 2.0),
        _node(4, 4.0, 2.0),
    ]
    links = [
        _link(1, 2),
        _link(2, 1),
        _link(2, 3),
        _link(3, 2),
        _link(3, 4),
        _link(4, 3),
        _link(1, 1, link_type=9),
    ]

    result = RtabmapTrajectoryRoadStep(half_width_m=0.5).run(
        nodes=nodes,
        links=links,
    )

    assert result.footprint_geojson is not None
    assert result.metadata["edge_count"] == 3
    assert result.metadata["node_count"] == 4
    assert result.grid.mask.sum() > 0
    assert result.metadata["floor_frame"] == {
        "horizontal_axes": ["x", "y"],
        "vertical_axis": "z",
    }


def test_rtabmap_trajectory_step_uses_flat_caps_not_round_end_blobs() -> None:
    nodes = [_node(1, 0.0, 0.0), _node(2, 2.0, 0.0)]
    links = [_link(1, 2), _link(2, 1)]

    result = RtabmapTrajectoryRoadStep(half_width_m=0.5).run(
        nodes=nodes,
        links=links,
    )

    assert result.footprint_geojson is not None
    footprint = shape(result.footprint_geojson)
    min_x, min_y, max_x, max_y = footprint.bounds
    assert min_x == pytest.approx(0.0)
    assert max_x == pytest.approx(2.0)
    assert min_y == pytest.approx(-0.5)
    assert max_y == pytest.approx(0.5)


def test_rtabmap_trajectory_step_records_dominant_building_angle() -> None:
    angle = 17.0
    p1 = _rot(0.0, 0.0, angle)
    p2 = _rot(0.0, 5.0, angle)
    p3 = _rot(4.0, 5.0, angle)
    nodes = [
        _node(1, p1[0], p1[1]),
        _node(2, p2[0], p2[1]),
        _node(3, p3[0], p3[1]),
    ]
    links = [_link(1, 2), _link(2, 3)]

    result = RtabmapTrajectoryRoadStep(half_width_m=0.5).run(
        nodes=nodes,
        links=links,
    )

    assert result.metadata["dominant_angle_source"] == "rtabmap_link_segments"
    assert abs(result.metadata["dominant_angle_deg"]) == pytest.approx(angle, abs=1.0)


def test_rtabmap_trajectory_step_expands_width_from_feature_cloud() -> None:
    nodes = [
        _node(1, 0.0, 0.0),
        _node(2, 2.0, 0.0),
        _node(3, 4.0, 0.01),
        _node(4, 6.0, 0.01),
    ]
    links = [_link(1, 2), _link(2, 3), _link(3, 4)]
    features = [
        _feature(node_id=((idx % 4) + 1), word_id=1000 + idx, local_y=1.2)
        for idx in range(24)
    ]

    result = RtabmapTrajectoryRoadStep(
        half_width_m=0.5,
        use_feature_evidence=True,
        feature_min_count=4,
    ).run(
        nodes=nodes,
        links=links,
        features=features,
    )

    assert result.metadata["resolved_half_width_m"] == pytest.approx(1.2, abs=0.02)
    assert result.metadata["half_width_m"] == pytest.approx(1.2, abs=0.02)
    feature_evidence = result.metadata["feature_evidence"]
    assert isinstance(feature_evidence, dict)
    assert feature_evidence["accepted_count"] == 24
    assert feature_evidence["estimated_half_width_m"] == pytest.approx(1.2, abs=0.02)


def test_rtabmap_trajectory_step_keeps_abstract_width_by_default() -> None:
    nodes = [
        _node(1, 0.0, 0.0),
        _node(2, 2.0, 0.0),
        _node(3, 4.0, 0.01),
        _node(4, 6.0, 0.01),
    ]
    links = [_link(1, 2), _link(2, 3), _link(3, 4)]
    features = [
        _feature(node_id=((idx % 4) + 1), word_id=1000 + idx, local_y=1.2)
        for idx in range(24)
    ]

    result = RtabmapTrajectoryRoadStep(
        half_width_m=0.5,
        feature_min_count=4,
    ).run(
        nodes=nodes,
        links=links,
        features=features,
    )

    assert result.metadata["resolved_half_width_m"] == pytest.approx(0.5)
    assert result.metadata["feature_evidence_enabled"] is False
    assert "feature_evidence" not in result.metadata


@pytest.mark.asyncio
async def test_pipeline_rtabmap_trajectory_smoke_skips_segmentation(tmp_path: Path) -> None:
    class FailingSegmenter:
        async def segment(self, image: object) -> object:  # pragma: no cover
            raise AssertionError("rtabmap trajectory mode must not call segmenter")

    nodes = [
        _node(1, 0.0, 0.0),
        _node(2, 2.0, 0.0),
        _node(3, 2.0, 2.0),
        _node(4, 4.0, 2.0),
    ]
    links = [_link(1, 2), _link(2, 3), _link(3, 4)]

    pipeline = BuildPipeline(
        segmenter=FailingSegmenter(),  # type: ignore[arg-type]
        storage_root=tmp_path,
        use_rtabmap_trajectory=True,
        rtabmap_trajectory_half_width_m=0.7,
    )

    async def _progress(step: BuildStep, p: float) -> None:
        pass

    async def _cancel() -> bool:
        return False

    outcome = await pipeline.execute(
        scan_id=UUID("11111111-1111-1111-1111-111111111111"),
        build_job_id=UUID("22222222-2222-2222-2222-222222222222"),
        keyframes=[],
        pois=[],
        rtabmap_nodes=nodes,
        rtabmap_links=links,
        rtabmap_frames=[_rtabmap_depth_frame(1)],
        progress_sink=_progress,
        cancel_check=_cancel,
    )

    assert outcome.passed_quality_gate
    assert outcome.counts.build_source == "rtabmap_trajectory"
    assert outcome.counts.walkable_cells > 0
    assert outcome.counts.map_nodes > 0
    assert outcome.counts.rtabmap_trajectory is not None
    assert outcome.counts.rtabmap_trajectory["edge_count"] == 3
    assert "rectilinear_cover" in outcome.counts.rtabmap_trajectory
    cover = outcome.counts.rtabmap_trajectory["rectilinear_cover"]
    assert isinstance(cover, dict)
    assert cover["all_edges_axis_aligned"] is True
    assert "rotated_grid_used" in cover
    assert cover["all_edges_dominant_axis_aligned"] is True
    assert cover["confidence_used"] is True
    depth_evidence = outcome.counts.rtabmap_trajectory["depth_evidence"]
    assert isinstance(depth_evidence, dict)
    assert depth_evidence["decoded_count"] == 1
    assert depth_evidence["calibration_decoded_count"] == 1


@pytest.mark.asyncio
async def test_pipeline_rtabmap_trajectory_uses_image_masks_for_depth_and_cover(
    tmp_path: Path,
) -> None:
    class MockSegmenter:
        async def segment(self, image: np.ndarray) -> SegmentationOutput:
            mask = np.full((image.shape[0], image.shape[1]), 3, dtype=np.int32)
            mask[2:, 2:] = 0
            return SegmentationOutput(class_mask=mask)

    nodes = [
        _node(1, 0.0, 0.0),
        _node(2, 2.0, 0.0),
        _node(3, 2.0, 2.0),
        _node(4, 4.0, 2.0),
    ]
    links = [_link(1, 2), _link(2, 3), _link(3, 4)]

    pipeline = BuildPipeline(
        segmenter=MockSegmenter(),  # type: ignore[arg-type]
        storage_root=tmp_path,
        use_rtabmap_trajectory=True,
        rtabmap_trajectory_half_width_m=0.7,
        rtabmap_image_segmentation_enabled=True,
    )

    async def _progress(step: BuildStep, p: float) -> None:
        pass

    async def _cancel() -> bool:
        return False

    outcome = await pipeline.execute(
        scan_id=UUID("11111111-1111-1111-1111-111111111111"),
        build_job_id=UUID("22222222-2222-2222-2222-222222222222"),
        keyframes=[],
        pois=[],
        rtabmap_nodes=nodes,
        rtabmap_links=links,
        rtabmap_frames=[_rtabmap_image_depth_frame(1)],
        progress_sink=_progress,
        cancel_check=_cancel,
    )

    assert outcome.passed_quality_gate
    assert outcome.counts.rtabmap_trajectory is not None
    image_evidence = outcome.counts.rtabmap_trajectory["image_evidence"]
    assert isinstance(image_evidence, dict)
    assert image_evidence["segmented_count"] == 1
    assert image_evidence["floor_mask_node_count"] == 1
    assert image_evidence["nonwalkable_mask_node_count"] == 1
    depth_evidence = outcome.counts.rtabmap_trajectory["depth_evidence"]
    assert isinstance(depth_evidence, dict)
    assert depth_evidence["floor_mask_required"] is True
    cover = outcome.counts.rtabmap_trajectory["rectilinear_cover"]
    assert isinstance(cover, dict)
    assert cover["confidence_used"] is True
    assert cover["avoid_used"] is True


def test_pipeline_mutex_adaptive_and_rtabmap_trajectory_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        BuildPipeline(
            segmenter=object(),  # type: ignore[arg-type]
            storage_root=tmp_path,
            use_adaptive_buffer=True,
            use_rtabmap_trajectory=True,
        )
