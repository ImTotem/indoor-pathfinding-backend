"""RTAB-Map Data.depth/calibration evidence decoding.

RTAB-Map stores CV_32FC1 depth maps as PNG-encoded 8UC4 bytes and stores
camera calibration with CameraModel::serialize(). This module mirrors the
small subset of that format needed by the server so the map builder can use
uploaded RTABMap data directly instead of falling back to raw ARKit keyframes.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from indoor_server.application.building.steps.rtabmap_trajectory import _FloorFrame
from indoor_server.domain.building.models import WalkableGrid
from indoor_server.domain.building.rtabmap_models import RtabmapDataFrame, RtabmapNode

# Sprint 52 C-1: shared depth projection helpers (avoid duplicated bodies).
# Imported lazily inside shims at end of file to avoid circular import — the
# helper module imports `RtabmapCameraCalibration` from this module.

_CAMERA_MODEL_HEADER_SIZE = 11
_MONO_CAMERA_TYPE = 0


@dataclass(frozen=True)
class RtabmapCameraCalibration:
    version: tuple[int, int, int]
    image_size: tuple[int, int]
    k: tuple[float, ...]
    local_transform: tuple[float, ...]
    bytes_read: int

    @property
    def fx(self) -> float:
        return float(self.k[0])

    @property
    def fy(self) -> float:
        return float(self.k[4])

    @property
    def cx(self) -> float:
        return float(self.k[2])

    @property
    def cy(self) -> float:
        return float(self.k[5])

    def scaled_intrinsics(self, *, width: int, height: int) -> tuple[float, float, float, float]:
        src_w, src_h = self.image_size
        sx = float(width) / float(src_w) if src_w > 0 else 1.0
        sy = float(height) / float(src_h) if src_h > 0 else 1.0
        return self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy

    def local_transform_matrix(self) -> np.ndarray:
        if len(self.local_transform) != 12:
            return np.eye(4, dtype=np.float64)
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :4] = np.asarray(self.local_transform, dtype=np.float64).reshape(3, 4)
        return mat

    def metadata(self) -> dict[str, object]:
        return {
            "version": list(self.version),
            "image_size": list(self.image_size),
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "has_local_transform": len(self.local_transform) == 12,
            "bytes_read": self.bytes_read,
        }


@dataclass(frozen=True)
class RtabmapDepthFrameEvidence:
    node_id: int
    depth_shape: tuple[int, int] | None
    depth_dtype: str | None
    calibration: RtabmapCameraCalibration | None = None
    valid_ratio: float | None = None
    depth_min_m: float | None = None
    depth_p50_m: float | None = None
    depth_p95_m: float | None = None
    depth_max_m: float | None = None
    sampled_point_count: int = 0
    projected_point_count: int = 0
    confidence_cell_count: int = 0
    avoid_projected_point_count: int = 0
    avoid_cell_count: int = 0
    projection_skipped_reason: str | None = None
    issue: str | None = None

    def metadata(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "depth_shape": self.depth_shape,
            "depth_dtype": self.depth_dtype,
            "calibration": (
                self.calibration.metadata() if self.calibration is not None else None
            ),
            "valid_ratio": self.valid_ratio,
            "depth_min_m": self.depth_min_m,
            "depth_p50_m": self.depth_p50_m,
            "depth_p95_m": self.depth_p95_m,
            "depth_max_m": self.depth_max_m,
            "sampled_point_count": self.sampled_point_count,
            "projected_point_count": self.projected_point_count,
            "confidence_cell_count": self.confidence_cell_count,
            "avoid_projected_point_count": self.avoid_projected_point_count,
            "avoid_cell_count": self.avoid_cell_count,
            "projection_skipped_reason": self.projection_skipped_reason,
            "issue": self.issue,
        }


@dataclass(frozen=True)
class RtabmapDepthEvidence:
    frame_count: int
    decoded_count: int
    calibration_decoded_count: int
    projected_point_count: int
    confidence: np.ndarray | None
    avoid_projected_point_count: int
    avoid: np.ndarray | None
    issues: list[str]
    frames: list[RtabmapDepthFrameEvidence]
    floor_mask_required: bool = False

    def metadata(self) -> dict[str, object]:
        confidence_nonzero = (
            int(np.count_nonzero(self.confidence))
            if self.confidence is not None
            else 0
        )
        confidence_max = (
            float(np.max(self.confidence))
            if self.confidence is not None and self.confidence.size
            else None
        )
        avoid_nonzero = (
            int(np.count_nonzero(self.avoid))
            if self.avoid is not None
            else 0
        )
        avoid_max = (
            float(np.max(self.avoid))
            if self.avoid is not None and self.avoid.size
            else None
        )
        return {
            "frame_count": self.frame_count,
            "decoded_count": self.decoded_count,
            "calibration_decoded_count": self.calibration_decoded_count,
            "projected_point_count": self.projected_point_count,
            "confidence_nonzero_cells": confidence_nonzero,
            "confidence_max": confidence_max,
            "avoid_projected_point_count": self.avoid_projected_point_count,
            "avoid_nonzero_cells": avoid_nonzero,
            "avoid_max": avoid_max,
            "issues": self.issues,
            "frames": [frame.metadata() for frame in self.frames],
            "floor_mask_required": self.floor_mask_required,
        }


class RtabmapDepthEvidenceStep:
    """Decode RTABMap metric depth and project a sparse confidence grid."""

    def __init__(
        self,
        *,
        pixel_stride: int = 8,
        lower_fraction: float = 0.60,
        min_depth_m: float = 0.20,
        max_depth_m: float = 8.0,
        vertical_tolerance_m: float | None = None,
        require_floor_mask: bool = False,
    ) -> None:
        if pixel_stride <= 0:
            raise ValueError("pixel_stride must be > 0")
        if not (0.0 < lower_fraction <= 1.0):
            raise ValueError("lower_fraction must be in (0, 1]")
        if min_depth_m <= 0 or max_depth_m <= min_depth_m:
            raise ValueError("depth range must satisfy 0 < min < max")
        self._pixel_stride = pixel_stride
        self._lower_fraction = lower_fraction
        self._min_depth_m = min_depth_m
        self._max_depth_m = max_depth_m
        self._vertical_tolerance_m = vertical_tolerance_m
        self._require_floor_mask = require_floor_mask

    def run(
        self,
        *,
        frames: list[RtabmapDataFrame],
        nodes: list[RtabmapNode],
        grid: WalkableGrid | None = None,
        floor_masks_by_node_id: dict[int, np.ndarray] | None = None,
        avoid_masks_by_node_id: dict[int, np.ndarray] | None = None,
        max_frames: int | None = None,
    ) -> RtabmapDepthEvidence:
        selected = frames[:max_frames] if max_frames is not None else frames
        node_pose = {
            node.node_id: np.asarray(node.pose, dtype=np.float64)
            for node in nodes
        }
        floor_frame = _FloorFrame.from_nodes(nodes) if nodes else None
        z_values = (
            [floor_frame.project_node_z(node) for node in nodes]
            if floor_frame is not None
            else []
        )
        z0 = float(np.median(z_values)) if z_values else 0.0
        confidence = (
            np.zeros(grid.mask.shape, dtype=np.float32)
            if grid is not None
            else None
        )
        avoid = (
            np.zeros(grid.mask.shape, dtype=np.float32)
            if grid is not None and avoid_masks_by_node_id is not None
            else None
        )

        issues: list[str] = []
        frame_evidence: list[RtabmapDepthFrameEvidence] = []
        decoded_count = 0
        calibration_count = 0
        total_projected = 0
        total_avoid_projected = 0

        for frame in selected:
            if frame.depth_bytes is None:
                issue = f"node:{frame.node_id}:depth_missing"
                issues.append(issue)
                frame_evidence.append(
                    RtabmapDepthFrameEvidence(
                        node_id=frame.node_id,
                        depth_shape=None,
                        depth_dtype=None,
                        issue=issue,
                    )
                )
                continue

            depth = decode_rtabmap_depth(frame.depth_bytes)
            if depth is None:
                issue = f"node:{frame.node_id}:depth_decode_failed"
                issues.append(issue)
                frame_evidence.append(
                    RtabmapDepthFrameEvidence(
                        node_id=frame.node_id,
                        depth_shape=None,
                        depth_dtype=None,
                        issue=issue,
                    )
                )
                continue
            decoded_count += 1
            calibration: RtabmapCameraCalibration | None = None
            if frame.calibration_bytes is not None:
                try:
                    calibration = decode_rtabmap_calibration(frame.calibration_bytes)
                    calibration_count += 1
                except ValueError as e:
                    issue = f"node:{frame.node_id}:calibration_decode_failed:{e}"
                    issues.append(issue)

            valid = np.isfinite(depth) & (
                (depth >= self._min_depth_m) & (depth <= self._max_depth_m)
            )
            valid_depth = depth[valid]
            stats = _depth_stats(valid_depth, total_pixels=depth.size)

            sampled = 0
            projected = 0
            confidence_cells = 0
            avoid_projected = 0
            avoid_cells = 0
            skipped_reason: str | None = None
            if (
                confidence is not None
                and grid is not None
                and floor_frame is not None
                and calibration is not None
                and frame.node_id in node_pose
            ):
                floor_mask = (
                    floor_masks_by_node_id.get(frame.node_id)
                    if floor_masks_by_node_id is not None
                    else None
                )
                if self._require_floor_mask and floor_mask is None:
                    skipped_reason = "floor_mask_required_missing_or_suppressed"
                else:
                    sampled, projected, confidence_cells = _project_depth_confidence(
                        confidence=confidence,
                        depth=depth,
                        calibration=calibration,
                        node_pose=node_pose[frame.node_id],
                        floor_frame=floor_frame,
                        grid=grid,
                        z0=z0,
                        pixel_stride=self._pixel_stride,
                        lower_fraction=self._lower_fraction,
                        min_depth_m=self._min_depth_m,
                        max_depth_m=self._max_depth_m,
                        vertical_tolerance_m=self._vertical_tolerance_m,
                        floor_mask=floor_mask,
                    )
                    total_projected += projected
                avoid_mask = (
                    avoid_masks_by_node_id.get(frame.node_id)
                    if avoid_masks_by_node_id is not None
                    else None
                )
                if avoid is not None and avoid_mask is not None:
                    _, avoid_projected, avoid_cells = _project_depth_confidence(
                        confidence=avoid,
                        depth=depth,
                        calibration=calibration,
                        node_pose=node_pose[frame.node_id],
                        floor_frame=floor_frame,
                        grid=grid,
                        z0=z0,
                        pixel_stride=self._pixel_stride,
                        lower_fraction=1.0,
                        min_depth_m=self._min_depth_m,
                        max_depth_m=self._max_depth_m,
                        vertical_tolerance_m=self._vertical_tolerance_m,
                        floor_mask=avoid_mask,
                    )
                    total_avoid_projected += avoid_projected

            frame_evidence.append(
                RtabmapDepthFrameEvidence(
                    node_id=frame.node_id,
                    depth_shape=(int(depth.shape[1]), int(depth.shape[0])),
                    depth_dtype=str(depth.dtype),
                    calibration=calibration,
                    valid_ratio=stats["valid_ratio"],
                    depth_min_m=stats["min"],
                    depth_p50_m=stats["p50"],
                    depth_p95_m=stats["p95"],
                    depth_max_m=stats["max"],
                    sampled_point_count=sampled,
                    projected_point_count=projected,
                    confidence_cell_count=confidence_cells,
                    avoid_projected_point_count=avoid_projected,
                    avoid_cell_count=avoid_cells,
                    projection_skipped_reason=skipped_reason,
                )
            )

        _normalize_positive_grid(confidence)
        _normalize_positive_grid(avoid)

        return RtabmapDepthEvidence(
            frame_count=len(selected),
            decoded_count=decoded_count,
            calibration_decoded_count=calibration_count,
            projected_point_count=total_projected,
            confidence=confidence,
            avoid_projected_point_count=total_avoid_projected,
            avoid=avoid,
            issues=issues,
            frames=frame_evidence,
            floor_mask_required=self._require_floor_mask,
        )


def decode_rtabmap_depth(data: bytes) -> np.ndarray | None:
    import cv2

    raw = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.dtype == np.uint8 and raw.ndim == 3 and raw.shape[2] == 4:
        return raw.copy().view("<f4").reshape(raw.shape[:2]).astype(np.float32)
    if raw.dtype == np.uint16:
        return (raw.astype(np.float32) / 1000.0).astype(np.float32)
    if raw.dtype == np.float32 and raw.ndim == 2:
        return raw.astype(np.float32, copy=False)
    if raw.ndim == 2:
        return raw.astype(np.float32)
    return None


def decode_rtabmap_calibration(data: bytes) -> RtabmapCameraCalibration:
    header_bytes = _CAMERA_MODEL_HEADER_SIZE * 4
    if len(data) < header_bytes:
        raise ValueError("header_too_short")
    header = struct.unpack("<11i", data[:header_bytes])
    if header[3] != _MONO_CAMERA_TYPE:
        raise ValueError(f"unsupported_camera_type:{header[3]}")

    k_count = header[6]
    d_count = header[7]
    r_count = header[8]
    p_count = header[9]
    local_count = header[10]
    if k_count not in (0, 9):
        raise ValueError(f"unexpected_k_count:{k_count}")
    if r_count not in (0, 9):
        raise ValueError(f"unexpected_r_count:{r_count}")
    if p_count not in (0, 12):
        raise ValueError(f"unexpected_p_count:{p_count}")
    if local_count not in (0, 12):
        raise ValueError(f"unexpected_local_transform_count:{local_count}")

    required = header_bytes + 8 * (k_count + d_count + r_count + p_count) + 4 * local_count
    if len(data) < required:
        raise ValueError(f"data_too_short:{len(data)}<{required}")

    idx = header_bytes
    k = _read_doubles(data, idx, k_count)
    idx += k_count * 8
    idx += d_count * 8
    idx += r_count * 8
    idx += p_count * 8
    local_transform = _read_floats(data, idx, local_count)
    idx += local_count * 4
    if not k:
        raise ValueError("missing_intrinsics_k")

    return RtabmapCameraCalibration(
        version=(int(header[0]), int(header[1]), int(header[2])),
        image_size=(int(header[4]), int(header[5])),
        k=tuple(float(v) for v in k),
        local_transform=tuple(float(v) for v in local_transform),
        bytes_read=idx,
    )


def _read_doubles(data: bytes, offset: int, count: int) -> tuple[float, ...]:
    if count == 0:
        return ()
    return struct.unpack("<" + "d" * count, data[offset:offset + count * 8])


def _read_floats(data: bytes, offset: int, count: int) -> tuple[float, ...]:
    if count == 0:
        return ()
    return struct.unpack("<" + "f" * count, data[offset:offset + count * 4])


def _depth_stats(values: np.ndarray, *, total_pixels: int) -> dict[str, float | None]:
    if values.size == 0:
        return {
            "valid_ratio": 0.0,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    return {
        "valid_ratio": float(values.size / float(total_pixels)),
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _normalize_positive_grid(grid: np.ndarray | None) -> None:
    if grid is None or not grid.size or np.max(grid) <= 0:
        return
    positive = grid[grid > 0]
    cap = float(np.percentile(positive, 95))
    if cap > 0:
        grid[:] = np.clip(grid / cap, 0.0, 1.0)


def _project_depth_confidence(
    *,
    confidence: np.ndarray,
    depth: np.ndarray,
    calibration: RtabmapCameraCalibration,
    node_pose: np.ndarray,
    floor_frame: _FloorFrame,
    grid: WalkableGrid,
    z0: float,
    pixel_stride: int,
    lower_fraction: float,
    min_depth_m: float,
    max_depth_m: float,
    vertical_tolerance_m: float | None,
    floor_mask: np.ndarray | None,
) -> tuple[int, int, int]:
    """Sprint 52 C-1: shim — body delegated to `depth_projection` helper.

    Inputs/outputs unchanged. The lower-half cropping (`lower_fraction`) is
    handled by trimming the depth array before delegating; the helper itself
    cannot lower-crop because callers like floor_point_cloud need the full
    depth grid.
    """
    from indoor_server.application.building.depth_projection import (
        project_depth_pixels_to_world,
    )
    h = depth.shape[0]
    y_start = max(0, int(round(h * (1.0 - lower_fraction))))
    if y_start >= h:
        return 0, 0, 0
    cropped_depth = depth[y_start:, :]
    cropped_mask = (
        floor_mask[y_start:, :] if floor_mask is not None and floor_mask.shape == depth.shape
        else floor_mask
    )
    # When mask shape differs we leave it to the helper's resampler.
    world, _sampled, valid_count = project_depth_pixels_to_world(
        depth=cropped_depth,
        floor_mask=cropped_mask,
        calibration=calibration,
        node_pose=node_pose,
        pixel_stride=pixel_stride,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        invert_mask=False,
    )
    if valid_count == 0 or world.size == 0:
        return 0, 0, 0

    sampled = int(valid_count)
    if vertical_tolerance_m is not None:
        vertical = world[:, floor_frame.vertical_axis]
        keep = np.abs(vertical - z0) <= vertical_tolerance_m
        world = world[keep]
    if world.size == 0:
        return sampled, 0, 0

    x = world[:, floor_frame.horizontal_axes[0]]
    y = world[:, floor_frame.horizontal_axes[1]]
    cols = np.floor((x - grid.origin.x0) / grid.origin.cell_size).astype(np.int32)
    rows = np.floor((y - grid.origin.y0) / grid.origin.cell_size).astype(np.int32)
    inside = (
        (rows >= 0)
        & (rows < confidence.shape[0])
        & (cols >= 0)
        & (cols < confidence.shape[1])
    )
    if not np.any(inside):
        return sampled, 0, 0

    rows = rows[inside]
    cols = cols[inside]
    before = int(np.count_nonzero(confidence))
    np.add.at(confidence, (rows, cols), 1.0)
    after = int(np.count_nonzero(confidence))
    return sampled, int(rows.size), after - before


def _sample_floor_mask(
    floor_mask: np.ndarray,
    depth_shape: tuple[int, int],
    yy: np.ndarray,
    xx: np.ndarray,
) -> np.ndarray:
    """Sprint 52 C-1: shim — `depth_projection.sample_floor_mask_to_depth_grid`."""
    from indoor_server.application.building.depth_projection import (
        sample_floor_mask_to_depth_grid,
    )
    return sample_floor_mask_to_depth_grid(floor_mask, depth_shape, yy, xx)
