from __future__ import annotations

import struct

import numpy as np
import pytest

from indoor_server.application.building.steps.rtabmap_depth_evidence import (
    RtabmapDepthEvidenceStep,
    _sample_floor_mask,
    decode_rtabmap_calibration,
    decode_rtabmap_depth,
)
from indoor_server.domain.building.models import GridOrigin, WalkableGrid
from indoor_server.domain.building.rtabmap_models import RtabmapDataFrame, RtabmapNode


def test_decode_rtabmap_float_depth_and_calibration() -> None:
    depth = np.array([[1.0, 2.5], [3.25, 4.0]], dtype=np.float32)
    encoded = _encode_rtabmap_float_depth(depth)

    decoded = decode_rtabmap_depth(encoded)
    calibration = decode_rtabmap_calibration(_calibration_bytes(width=960, height=720))

    assert decoded is not None
    assert decoded.dtype == np.float32
    np.testing.assert_allclose(decoded, depth)
    assert calibration.version == (0, 23, 5)
    assert calibration.image_size == (960, 720)
    assert calibration.fx == pytest.approx(670.0)
    assert calibration.cy == pytest.approx(360.0)
    assert calibration.local_transform_matrix().shape == (4, 4)


def test_rtabmap_depth_evidence_projects_confidence_grid() -> None:
    depth = np.full((4, 4), 1.0, dtype=np.float32)
    frame = RtabmapDataFrame(
        node_id=1,
        image_bytes=None,
        depth_bytes=_encode_rtabmap_float_depth(depth),
        calibration_bytes=_calibration_bytes(
            width=4,
            height=4,
            fx=4.0,
            fy=4.0,
            cx=1.5,
            cy=1.5,
        ),
    )
    grid = WalkableGrid(
        origin=GridOrigin(x0=-2.0, y0=-2.0, z0=0.0, cell_size=0.25, w=16, h=16),
        mask=np.ones((16, 16), dtype=bool),
        observation_count=np.ones((16, 16), dtype=np.uint16),
    )

    evidence = RtabmapDepthEvidenceStep(
        pixel_stride=1,
        lower_fraction=0.5,
        vertical_tolerance_m=None,
    ).run(
        frames=[frame],
        nodes=_nodes(),
        grid=grid,
    )

    assert evidence.decoded_count == 1
    assert evidence.calibration_decoded_count == 1
    assert evidence.projected_point_count > 0
    assert evidence.confidence is not None
    assert np.count_nonzero(evidence.confidence) > 0
    assert evidence.metadata()["confidence_max"] == pytest.approx(1.0)


def test_rtabmap_depth_evidence_uses_floor_and_avoid_masks() -> None:
    depth = np.full((4, 4), 1.0, dtype=np.float32)
    frame = RtabmapDataFrame(
        node_id=1,
        image_bytes=None,
        depth_bytes=_encode_rtabmap_float_depth(depth),
        calibration_bytes=_calibration_bytes(
            width=4,
            height=4,
            fx=4.0,
            fy=4.0,
            cx=1.5,
            cy=1.5,
        ),
    )
    grid = WalkableGrid(
        origin=GridOrigin(x0=-2.0, y0=-2.0, z0=0.0, cell_size=0.25, w=16, h=16),
        mask=np.ones((16, 16), dtype=bool),
        observation_count=np.ones((16, 16), dtype=np.uint16),
    )
    floor_mask = np.zeros((4, 4), dtype=bool)
    floor_mask[2:, :] = True
    avoid_mask = np.zeros((4, 4), dtype=bool)
    avoid_mask[:2, :] = True

    evidence = RtabmapDepthEvidenceStep(
        pixel_stride=1,
        lower_fraction=1.0,
        vertical_tolerance_m=None,
    ).run(
        frames=[frame],
        nodes=_nodes(),
        grid=grid,
        floor_masks_by_node_id={1: floor_mask},
        avoid_masks_by_node_id={1: avoid_mask},
    )

    assert evidence.projected_point_count > 0
    assert evidence.avoid_projected_point_count > 0
    assert evidence.confidence is not None
    assert evidence.avoid is not None
    assert np.count_nonzero(evidence.confidence) > 0
    assert np.count_nonzero(evidence.avoid) > 0
    metadata = evidence.metadata()
    assert metadata["avoid_nonzero_cells"] > 0
    assert metadata["avoid_max"] == pytest.approx(1.0)


def test_rtabmap_depth_evidence_requires_floor_mask_when_enabled() -> None:
    depth = np.full((4, 4), 1.0, dtype=np.float32)
    frame = RtabmapDataFrame(
        node_id=1,
        image_bytes=None,
        depth_bytes=_encode_rtabmap_float_depth(depth),
        calibration_bytes=_calibration_bytes(
            width=4,
            height=4,
            fx=4.0,
            fy=4.0,
            cx=1.5,
            cy=1.5,
        ),
    )
    grid = WalkableGrid(
        origin=GridOrigin(x0=-2.0, y0=-2.0, z0=0.0, cell_size=0.25, w=16, h=16),
        mask=np.ones((16, 16), dtype=bool),
        observation_count=np.ones((16, 16), dtype=np.uint16),
    )

    evidence = RtabmapDepthEvidenceStep(
        pixel_stride=1,
        lower_fraction=1.0,
        vertical_tolerance_m=None,
        require_floor_mask=True,
    ).run(
        frames=[frame],
        nodes=_nodes(),
        grid=grid,
        floor_masks_by_node_id={},
    )

    assert evidence.projected_point_count == 0
    assert evidence.confidence is not None
    assert np.count_nonzero(evidence.confidence) == 0
    assert evidence.frames[0].projection_skipped_reason == (
        "floor_mask_required_missing_or_suppressed"
    )
    assert evidence.metadata()["floor_mask_required"] is True


def test_floor_mask_sampling_aligns_high_res_image_mask_to_depth_grid() -> None:
    floor_mask = np.zeros((8, 8), dtype=bool)
    floor_mask[4:, :] = True
    yy, xx = np.meshgrid(
        np.arange(0, 4, dtype=np.int32),
        np.arange(0, 4, dtype=np.int32),
        indexing="ij",
    )

    sampled = _sample_floor_mask(floor_mask, (4, 4), yy, xx)

    assert sampled.shape == (4, 4)
    assert not sampled[:2, :].any()
    assert sampled[2:, :].all()


def _encode_rtabmap_float_depth(depth: np.ndarray) -> bytes:
    import cv2

    packed = depth.astype(np.float32).copy().view(np.uint8).reshape(*depth.shape, 4)
    ok, encoded = cv2.imencode(".png", packed)
    assert ok
    return bytes(encoded)


def _calibration_bytes(
    *,
    width: int,
    height: int,
    fx: float = 670.0,
    fy: float = 670.0,
    cx: float = 480.0,
    cy: float = 360.0,
) -> bytes:
    header = struct.pack(
        "<11i",
        0,
        23,
        5,
        0,
        width,
        height,
        9,
        0,
        0,
        0,
        12,
    )
    k = struct.pack(
        "<9d",
        fx,
        0.0,
        cx,
        0.0,
        fy,
        cy,
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
    return header + k + local_transform


def _nodes() -> list[RtabmapNode]:
    return [
        _node(1, tx=0.0, ty=0.0, tz=0.0),
        _node(2, tx=1.0, ty=0.0, tz=0.0),
        _node(3, tx=1.0, ty=1.0, tz=0.0),
    ]


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
