"""RTAB-Map Data.image decode + optional floor segmentation evidence."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from indoor_server.domain.building.rtabmap_models import RtabmapDataFrame
from indoor_server.infrastructure.ml.protocol import SegmentationOutput
from indoor_server.infrastructure.ml.segformer_onnx import (
    extract_floor_stair_masks,
    extract_object_avoid_mask,
)

_WALL_CLASS_IDX = 0
OrientationMode = Literal[
    "sensor",
    "rotate_cw_90",
    "rotate_ccw_90",
    "rotate_180",
    "auto",
]
_FIXED_ORIENTATIONS: tuple[OrientationMode, ...] = (
    "sensor",
    "rotate_cw_90",
    "rotate_ccw_90",
    "rotate_180",
)


class _SegmenterLike(Protocol):
    async def segment(self, image: np.ndarray) -> SegmentationOutput:
        ...


@dataclass(frozen=True)
class RtabmapImageFrameEvidence:
    node_id: int
    image_shape: tuple[int, int] | None
    has_pose: bool
    floor_ratio: float | None = None
    wall_ratio: float | None = None
    stair_ratio: float | None = None
    object_ratio: float | None = None
    nonwalkable_ratio: float | None = None
    selected_orientation: str | None = None
    floor_mask_used: bool | None = None
    wall_mask_used: bool | None = None
    stair_mask_used: bool | None = None
    object_mask_used: bool | None = None
    nonwalkable_mask_used: bool | None = None
    mask_warnings: list[str] | None = None
    issue: str | None = None

    def metadata(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "image_shape": self.image_shape,
            "has_pose": self.has_pose,
            "floor_ratio": self.floor_ratio,
            "wall_ratio": self.wall_ratio,
            "stair_ratio": self.stair_ratio,
            "object_ratio": self.object_ratio,
            "nonwalkable_ratio": self.nonwalkable_ratio,
            "selected_orientation": self.selected_orientation,
            "floor_mask_used": self.floor_mask_used,
            "wall_mask_used": self.wall_mask_used,
            "stair_mask_used": self.stair_mask_used,
            "object_mask_used": self.object_mask_used,
            "nonwalkable_mask_used": self.nonwalkable_mask_used,
            "mask_warnings": self.mask_warnings or [],
            "issue": self.issue,
        }


@dataclass(frozen=True)
class RtabmapImageEvidence:
    frame_count: int
    decoded_count: int
    segmented_count: int
    image_shapes: list[tuple[int, int]]
    floor_ratio_mean: float | None
    wall_ratio_mean: float | None
    stair_ratio_mean: float | None
    object_ratio_mean: float | None
    nonwalkable_ratio_mean: float | None
    issues: list[str]
    frames: list[RtabmapImageFrameEvidence]
    floor_masks_by_node_id: dict[int, np.ndarray]
    wall_masks_by_node_id: dict[int, np.ndarray]
    stair_masks_by_node_id: dict[int, np.ndarray]
    object_masks_by_node_id: dict[int, np.ndarray]
    nonwalkable_masks_by_node_id: dict[int, np.ndarray]
    orientation_mode: str = "sensor"
    selected_orientation_counts: dict[str, int] | None = None
    floor_mask_min_ratio: float = 0.0
    wall_mask_max_ratio: float = 1.0
    used_floor_ratio_mean: float | None = None
    used_wall_ratio_mean: float | None = None
    used_stair_ratio_mean: float | None = None
    used_object_ratio_mean: float | None = None
    used_nonwalkable_ratio_mean: float | None = None

    def metadata(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "decoded_count": self.decoded_count,
            "segmented_count": self.segmented_count,
            "image_shapes": self.image_shapes,
            "floor_ratio_mean": self.floor_ratio_mean,
            "wall_ratio_mean": self.wall_ratio_mean,
            "stair_ratio_mean": self.stair_ratio_mean,
            "object_ratio_mean": self.object_ratio_mean,
            "nonwalkable_ratio_mean": self.nonwalkable_ratio_mean,
            "issues": self.issues,
            "frames": [frame.metadata() for frame in self.frames],
            "floor_mask_node_count": len(self.floor_masks_by_node_id),
            "wall_mask_node_count": len(self.wall_masks_by_node_id),
            "stair_mask_node_count": len(self.stair_masks_by_node_id),
            "object_mask_node_count": len(self.object_masks_by_node_id),
            "nonwalkable_mask_node_count": len(self.nonwalkable_masks_by_node_id),
            "orientation_mode": self.orientation_mode,
            "selected_orientation_counts": self.selected_orientation_counts or {},
            "floor_mask_min_ratio": self.floor_mask_min_ratio,
            "wall_mask_max_ratio": self.wall_mask_max_ratio,
            "used_floor_ratio_mean": self.used_floor_ratio_mean,
            "used_wall_ratio_mean": self.used_wall_ratio_mean,
            "used_stair_ratio_mean": self.used_stair_ratio_mean,
            "used_object_ratio_mean": self.used_object_ratio_mean,
            "used_nonwalkable_ratio_mean": self.used_nonwalkable_ratio_mean,
        }


class RtabmapImageEvidenceStep:
    """Decode RTAB-Map Data.image and optionally run the existing segmenter.

    This is a hook for the next geometry-fusion step. It deliberately does not
    alter road polygons yet.
    """

    async def run(
        self,
        frames: list[RtabmapDataFrame],
        *,
        segmenter: _SegmenterLike | None = None,
        node_pose_ids: set[int] | None = None,
        max_frames: int | None = None,
        orientation_mode: OrientationMode = "sensor",
        floor_mask_min_ratio: float = 0.0,
        wall_mask_max_ratio: float = 1.0,
    ) -> RtabmapImageEvidence:
        if orientation_mode not in (*_FIXED_ORIENTATIONS, "auto"):
            raise ValueError(f"unsupported orientation_mode: {orientation_mode}")
        if not (0.0 <= floor_mask_min_ratio <= 1.0):
            raise ValueError("floor_mask_min_ratio must be in [0, 1]")
        if not (0.0 <= wall_mask_max_ratio <= 1.0):
            raise ValueError("wall_mask_max_ratio must be in [0, 1]")

        selected = frames[:max_frames] if max_frames is not None else frames
        decoded: list[tuple[RtabmapDataFrame, np.ndarray]] = []
        frame_evidence: list[RtabmapImageFrameEvidence] = []
        issues: list[str] = []
        shapes: list[tuple[int, int]] = []

        for frame in selected:
            has_pose = node_pose_ids is None or frame.node_id in node_pose_ids
            if frame.image_bytes is None:
                issue = f"node:{frame.node_id}:image_missing"
                issues.append(issue)
                frame_evidence.append(
                    RtabmapImageFrameEvidence(
                        node_id=frame.node_id,
                        image_shape=None,
                        has_pose=has_pose,
                        issue=issue,
                    )
                )
                continue
            image = await asyncio.to_thread(_decode_image_bytes, frame.image_bytes)
            if image is None:
                issue = f"node:{frame.node_id}:image_decode_failed"
                issues.append(issue)
                frame_evidence.append(
                    RtabmapImageFrameEvidence(
                        node_id=frame.node_id,
                        image_shape=None,
                        has_pose=has_pose,
                        issue=issue,
                    )
                )
                continue
            shape = (int(image.shape[1]), int(image.shape[0]))
            decoded.append((frame, image))
            shapes.append(shape)
            if segmenter is None:
                frame_evidence.append(
                    RtabmapImageFrameEvidence(
                        node_id=frame.node_id,
                        image_shape=shape,
                        has_pose=has_pose,
                    )
                )

        if segmenter is None:
            return RtabmapImageEvidence(
                frame_count=len(selected),
                decoded_count=len(decoded),
                segmented_count=0,
                image_shapes=shapes,
                floor_ratio_mean=None,
                wall_ratio_mean=None,
                stair_ratio_mean=None,
                object_ratio_mean=None,
                nonwalkable_ratio_mean=None,
                issues=issues,
                frames=frame_evidence,
                floor_masks_by_node_id={},
                wall_masks_by_node_id={},
                stair_masks_by_node_id={},
                object_masks_by_node_id={},
                nonwalkable_masks_by_node_id={},
                orientation_mode=orientation_mode,
                selected_orientation_counts={},
                floor_mask_min_ratio=floor_mask_min_ratio,
                wall_mask_max_ratio=wall_mask_max_ratio,
                used_floor_ratio_mean=None,
                used_wall_ratio_mean=None,
                used_stair_ratio_mean=None,
                used_object_ratio_mean=None,
                used_nonwalkable_ratio_mean=None,
            )

        floor_ratios: list[float] = []
        wall_ratios: list[float] = []
        stair_ratios: list[float] = []
        object_ratios: list[float] = []
        nonwalkable_ratios: list[float] = []
        used_floor_ratios: list[float] = []
        used_wall_ratios: list[float] = []
        used_stair_ratios: list[float] = []
        used_object_ratios: list[float] = []
        used_nonwalkable_ratios: list[float] = []
        floor_masks_by_node_id: dict[int, np.ndarray] = {}
        wall_masks_by_node_id: dict[int, np.ndarray] = {}
        stair_masks_by_node_id: dict[int, np.ndarray] = {}
        object_masks_by_node_id: dict[int, np.ndarray] = {}
        nonwalkable_masks_by_node_id: dict[int, np.ndarray] = {}
        selected_orientation_counts: dict[str, int] = {}
        for frame, image in decoded:
            candidate = await _segment_with_orientation(
                image=image,
                segmenter=segmenter,
                orientation_mode=orientation_mode,
            )
            selected_orientation_counts[candidate.orientation] = (
                selected_orientation_counts.get(candidate.orientation, 0) + 1
            )
            floor_mask, stair_mask = extract_floor_stair_masks(candidate.class_mask)
            wall_mask = candidate.class_mask == _WALL_CLASS_IDX
            object_mask = extract_object_avoid_mask(candidate.class_mask)
            nonwalkable_mask = wall_mask | stair_mask | object_mask
            total = float(floor_mask.size)
            floor_ratio = float(floor_mask.sum()) / total
            wall_ratio = float(wall_mask.sum()) / total
            stair_ratio = float(stair_mask.sum()) / total
            object_ratio = float(object_mask.sum()) / total
            nonwalkable_ratio = float(nonwalkable_mask.sum()) / total
            floor_ratios.append(floor_ratio)
            wall_ratios.append(wall_ratio)
            stair_ratios.append(stair_ratio)
            object_ratios.append(object_ratio)
            nonwalkable_ratios.append(nonwalkable_ratio)
            mask_warnings: list[str] = []
            floor_used = floor_ratio >= floor_mask_min_ratio
            wall_used = wall_ratio <= wall_mask_max_ratio
            stair_used = stair_ratio > 0.0
            object_used = object_ratio > 0.0
            nonwalkable_used = nonwalkable_ratio > 0.0
            if floor_used:
                floor_masks_by_node_id[frame.node_id] = floor_mask.astype(bool, copy=False)
                used_floor_ratios.append(floor_ratio)
            else:
                mask_warnings.append(
                    f"floor_mask_suppressed_low_ratio:{floor_ratio:.4f}"
                )
                used_floor_ratios.append(0.0)
            if wall_used:
                wall_masks_by_node_id[frame.node_id] = wall_mask.astype(bool, copy=False)
                used_wall_ratios.append(wall_ratio)
            else:
                mask_warnings.append(
                    f"wall_mask_suppressed_high_ratio:{wall_ratio:.4f}"
                )
                used_wall_ratios.append(0.0)
            if stair_used:
                stair_masks_by_node_id[frame.node_id] = stair_mask.astype(bool, copy=False)
                used_stair_ratios.append(stair_ratio)
            else:
                used_stair_ratios.append(0.0)
            if object_used:
                object_masks_by_node_id[frame.node_id] = object_mask.astype(
                    bool, copy=False
                )
                used_object_ratios.append(object_ratio)
            else:
                used_object_ratios.append(0.0)
            if nonwalkable_used:
                nonwalkable_masks_by_node_id[frame.node_id] = nonwalkable_mask.astype(
                    bool, copy=False
                )
                used_nonwalkable_ratios.append(nonwalkable_ratio)
            else:
                used_nonwalkable_ratios.append(0.0)
            frame_evidence.append(
                RtabmapImageFrameEvidence(
                    node_id=frame.node_id,
                    image_shape=(int(image.shape[1]), int(image.shape[0])),
                    has_pose=node_pose_ids is None or frame.node_id in node_pose_ids,
                    floor_ratio=floor_ratio,
                    wall_ratio=wall_ratio,
                    stair_ratio=stair_ratio,
                    object_ratio=object_ratio,
                    nonwalkable_ratio=nonwalkable_ratio,
                    selected_orientation=candidate.orientation,
                    floor_mask_used=floor_used,
                    wall_mask_used=wall_used,
                    stair_mask_used=stair_used,
                    object_mask_used=object_used,
                    nonwalkable_mask_used=nonwalkable_used,
                    mask_warnings=mask_warnings,
                )
            )

        return RtabmapImageEvidence(
            frame_count=len(selected),
            decoded_count=len(decoded),
            segmented_count=len(floor_ratios),
            image_shapes=shapes,
            floor_ratio_mean=(
                float(np.mean(floor_ratios)) if floor_ratios else None
            ),
            wall_ratio_mean=(
                float(np.mean(wall_ratios)) if wall_ratios else None
            ),
            stair_ratio_mean=(
                float(np.mean(stair_ratios)) if stair_ratios else None
            ),
            object_ratio_mean=(
                float(np.mean(object_ratios)) if object_ratios else None
            ),
            nonwalkable_ratio_mean=(
                float(np.mean(nonwalkable_ratios)) if nonwalkable_ratios else None
            ),
            issues=issues,
            frames=frame_evidence,
            floor_masks_by_node_id=floor_masks_by_node_id,
            wall_masks_by_node_id=wall_masks_by_node_id,
            stair_masks_by_node_id=stair_masks_by_node_id,
            object_masks_by_node_id=object_masks_by_node_id,
            nonwalkable_masks_by_node_id=nonwalkable_masks_by_node_id,
            orientation_mode=orientation_mode,
            selected_orientation_counts=selected_orientation_counts,
            floor_mask_min_ratio=floor_mask_min_ratio,
            wall_mask_max_ratio=wall_mask_max_ratio,
            used_floor_ratio_mean=(
                float(np.mean(used_floor_ratios)) if used_floor_ratios else None
            ),
            used_wall_ratio_mean=(
                float(np.mean(used_wall_ratios)) if used_wall_ratios else None
            ),
            used_stair_ratio_mean=(
                float(np.mean(used_stair_ratios)) if used_stair_ratios else None
            ),
            used_object_ratio_mean=(
                float(np.mean(used_object_ratios)) if used_object_ratios else None
            ),
            used_nonwalkable_ratio_mean=(
                float(np.mean(used_nonwalkable_ratios))
                if used_nonwalkable_ratios
                else None
            ),
        )


def _decode_image_bytes(data: bytes) -> np.ndarray | None:
    import cv2

    buf = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


@dataclass(frozen=True)
class _SegmentationCandidate:
    orientation: str
    class_mask: np.ndarray
    score: float


async def _segment_with_orientation(
    *,
    image: np.ndarray,
    segmenter: _SegmenterLike,
    orientation_mode: OrientationMode,
) -> _SegmentationCandidate:
    orientations = (
        _FIXED_ORIENTATIONS if orientation_mode == "auto" else (orientation_mode,)
    )
    best: _SegmentationCandidate | None = None
    for orientation in orientations:
        oriented_image = _orient_image(image, orientation)
        output = await segmenter.segment(oriented_image)
        score = _segmentation_orientation_score(output.class_mask)
        class_mask = _restore_mask_orientation(output.class_mask, orientation)
        candidate = _SegmentationCandidate(
            orientation=orientation,
            class_mask=class_mask,
            score=score,
        )
        if best is None or candidate.score > best.score:
            best = candidate
    assert best is not None
    return best


def _orient_image(image: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "sensor":
        return image
    if orientation == "rotate_cw_90":
        return np.rot90(image, k=3).copy()
    if orientation == "rotate_ccw_90":
        return np.rot90(image, k=1).copy()
    if orientation == "rotate_180":
        return np.rot90(image, k=2).copy()
    raise ValueError(f"unsupported orientation: {orientation}")


def _restore_mask_orientation(mask: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "sensor":
        return mask
    if orientation == "rotate_cw_90":
        return np.rot90(mask, k=1).copy()
    if orientation == "rotate_ccw_90":
        return np.rot90(mask, k=3).copy()
    if orientation == "rotate_180":
        return np.rot90(mask, k=2).copy()
    raise ValueError(f"unsupported orientation: {orientation}")


def _segmentation_orientation_score(class_mask: np.ndarray) -> float:
    floor_mask, stair_mask = extract_floor_stair_masks(class_mask)
    wall_mask = class_mask == _WALL_CLASS_IDX
    object_mask = extract_object_avoid_mask(class_mask)
    total = max(float(class_mask.size), 1.0)
    floor_ratio = float(floor_mask.sum()) / total
    wall_ratio = float(wall_mask.sum()) / total
    stair_ratio = float(stair_mask.sum()) / total
    object_ratio = float(object_mask.sum()) / total

    h = class_mask.shape[0]
    lower_start = int(round(h * 0.55))
    upper_end = int(round(h * 0.45))
    floor_pixels = max(float(floor_mask.sum()), 1.0)
    lower_floor_share = float(floor_mask[lower_start:, :].sum()) / floor_pixels
    upper_floor_share = float(floor_mask[:upper_end, :].sum()) / floor_pixels

    score = (
        floor_ratio * 3.0
        + lower_floor_share * 0.4
        + stair_ratio * 0.3
        - wall_ratio * 0.8
        - object_ratio * 0.4
        - upper_floor_share * 0.2
    )
    if floor_ratio < 0.02:
        score -= 0.7
    if wall_ratio > 0.75:
        score -= 0.5
    return float(score)
