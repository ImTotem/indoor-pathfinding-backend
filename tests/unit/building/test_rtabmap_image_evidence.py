from __future__ import annotations

import numpy as np
import pytest

from indoor_server.application.building.steps.rtabmap_image_evidence import (
    RtabmapImageEvidenceStep,
)
from indoor_server.domain.building.rtabmap_models import RtabmapDataFrame
from indoor_server.infrastructure.ml.protocol import SegmentationOutput


def _jpg_bytes(*, width: int = 4, height: int = 4) -> bytes:
    import cv2

    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return bytes(buf)


@pytest.mark.asyncio
async def test_rtabmap_image_evidence_decodes_data_image_by_node() -> None:
    evidence = await RtabmapImageEvidenceStep().run(
        [
            RtabmapDataFrame(
                node_id=1,
                image_bytes=_jpg_bytes(),
                depth_bytes=None,
                calibration_bytes=None,
            ),
            RtabmapDataFrame(
                node_id=2,
                image_bytes=None,
                depth_bytes=None,
                calibration_bytes=None,
            ),
        ],
        node_pose_ids={1},
    )

    assert evidence.frame_count == 2
    assert evidence.decoded_count == 1
    assert evidence.segmented_count == 0
    assert evidence.image_shapes == [(4, 4)]
    assert evidence.frames[0].node_id == 1
    assert evidence.frames[0].has_pose
    assert evidence.frames[1].issue == "node:2:image_missing"
    assert not evidence.frames[1].has_pose


@pytest.mark.asyncio
async def test_rtabmap_image_evidence_can_attach_floor_wall_candidates() -> None:
    class MockSegmenter:
        async def segment(self, image: np.ndarray) -> SegmentationOutput:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
            mask[:2, :] = 3
            mask[3, :] = 53
            return SegmentationOutput(class_mask=mask)

    evidence = await RtabmapImageEvidenceStep().run(
        [
            RtabmapDataFrame(
                node_id=7,
                image_bytes=_jpg_bytes(),
                depth_bytes=None,
                calibration_bytes=None,
            )
        ],
        segmenter=MockSegmenter(),
        node_pose_ids={7},
    )

    assert evidence.segmented_count == 1
    assert evidence.floor_ratio_mean == pytest.approx(0.5)
    assert evidence.wall_ratio_mean == pytest.approx(0.25)
    assert evidence.stair_ratio_mean == pytest.approx(0.25)
    assert set(evidence.floor_masks_by_node_id) == {7}
    assert set(evidence.wall_masks_by_node_id) == {7}
    assert set(evidence.stair_masks_by_node_id) == {7}
    assert evidence.floor_masks_by_node_id[7].sum() == 8
    assert evidence.wall_masks_by_node_id[7].sum() == 4
    assert evidence.stair_masks_by_node_id[7].sum() == 4
    assert evidence.metadata()["floor_mask_node_count"] == 1
    frame = evidence.frames[0]
    assert frame.node_id == 7
    assert frame.floor_ratio == pytest.approx(0.5)
    assert frame.wall_ratio == pytest.approx(0.25)
    assert frame.stair_ratio == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_rtabmap_image_evidence_marks_bed_and_furniture_as_nonwalkable() -> None:
    class MockSegmenter:
        async def segment(self, image: np.ndarray) -> SegmentationOutput:
            mask = np.full((image.shape[0], image.shape[1]), 3, dtype=np.int32)
            mask[0, :] = 0
            mask[1, :] = 7
            mask[2, :2] = 15
            mask[2, 2:] = 19
            mask[3, :] = 28
            return SegmentationOutput(class_mask=mask)

    evidence = await RtabmapImageEvidenceStep().run(
        [
            RtabmapDataFrame(
                node_id=11,
                image_bytes=_jpg_bytes(),
                depth_bytes=None,
                calibration_bytes=None,
            )
        ],
        segmenter=MockSegmenter(),
    )

    assert evidence.floor_ratio_mean == pytest.approx(0.25)
    assert evidence.object_ratio_mean == pytest.approx(0.5)
    assert evidence.nonwalkable_ratio_mean == pytest.approx(0.75)
    assert set(evidence.object_masks_by_node_id) == {11}
    assert set(evidence.nonwalkable_masks_by_node_id) == {11}
    assert evidence.object_masks_by_node_id[11].sum() == 8
    assert evidence.nonwalkable_masks_by_node_id[11].sum() == 12
    assert not evidence.object_masks_by_node_id[11][0, :].any()
    assert evidence.object_masks_by_node_id[11][1, :].all()
    assert evidence.object_masks_by_node_id[11][2, :].all()
    assert not evidence.nonwalkable_masks_by_node_id[11][3, :].any()
    assert evidence.frames[0].object_mask_used is True
    assert evidence.frames[0].nonwalkable_mask_used is True
    metadata = evidence.metadata()
    assert metadata["object_mask_node_count"] == 1
    assert metadata["nonwalkable_mask_node_count"] == 1
    assert metadata["used_object_ratio_mean"] == pytest.approx(0.5)
    assert metadata["used_nonwalkable_ratio_mean"] == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_rtabmap_image_evidence_restores_rotated_masks_to_depth_coordinates() -> None:
    class MockSegmenter:
        async def segment(self, image: np.ndarray) -> SegmentationOutput:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
            mask[:, 0] = 3
            return SegmentationOutput(class_mask=mask)

    evidence = await RtabmapImageEvidenceStep().run(
        [
            RtabmapDataFrame(
                node_id=9,
                image_bytes=_jpg_bytes(width=5, height=3),
                depth_bytes=None,
                calibration_bytes=None,
            )
        ],
        segmenter=MockSegmenter(),
        orientation_mode="rotate_cw_90",
    )

    floor = evidence.floor_masks_by_node_id[9]
    assert floor.shape == (3, 5)
    assert floor[2, :].all()
    assert not floor[:2, :].any()
    assert evidence.frames[0].selected_orientation == "rotate_cw_90"


@pytest.mark.asyncio
async def test_rtabmap_image_evidence_suppresses_untrusted_masks() -> None:
    class MockSegmenter:
        async def segment(self, image: np.ndarray) -> SegmentationOutput:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
            mask[0, 0] = 3
            return SegmentationOutput(class_mask=mask)

    evidence = await RtabmapImageEvidenceStep().run(
        [
            RtabmapDataFrame(
                node_id=10,
                image_bytes=_jpg_bytes(),
                depth_bytes=None,
                calibration_bytes=None,
            )
        ],
        segmenter=MockSegmenter(),
        floor_mask_min_ratio=0.5,
        wall_mask_max_ratio=0.5,
    )

    assert evidence.floor_ratio_mean == pytest.approx(1 / 16)
    assert evidence.wall_ratio_mean == pytest.approx(15 / 16)
    assert evidence.used_floor_ratio_mean == pytest.approx(0.0)
    assert evidence.used_wall_ratio_mean == pytest.approx(0.0)
    assert evidence.floor_masks_by_node_id == {}
    assert evidence.wall_masks_by_node_id == {}
    warnings = evidence.frames[0].mask_warnings or []
    assert any(w.startswith("floor_mask_suppressed_low_ratio") for w in warnings)
    assert any(w.startswith("wall_mask_suppressed_high_ratio") for w in warnings)
