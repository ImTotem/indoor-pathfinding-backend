"""Map source RTAB-Map nodes to a merged multi-scan RTAB-Map database.

The merged RTAB-Map DB is treated as a pose spine. Polygon evidence should still
come from the original scan videos / frames; this module preserves provenance so
those original frames can be re-projected into the merged coordinate frame.
"""
from __future__ import annotations

import math
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs, quote, unquote, urlencode

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from indoor_server.application.rtabmap.reader import decode_pose_3x4_blob
from indoor_server.domain.building.rtabmap_models import Matrix4x4

PROVENANCE_LABEL_PREFIX = "__ipf_src__?"
MappingSource = Literal["label", "stamp", "ambiguous", "missing"]
FramePoseSource = Literal[
    "optimized_node_pose",
    "interpolated_optimized_pose",
    "raw_pose_transformed",
    "excluded",
]


@dataclass(frozen=True)
class SourceNodeRef:
    scan_id: str
    node_id: int
    stamp: float
    original_label: str | None = None

    @property
    def key(self) -> tuple[str, int]:
        return (self.scan_id, self.node_id)


@dataclass(frozen=True)
class RtabmapNodeRecord:
    node_id: int
    map_id: int
    stamp: float
    pose: Matrix4x4
    label: str | None = None


@dataclass(frozen=True)
class NodePoseMapping:
    source_scan_id: str
    source_node_id: int
    source_stamp: float
    mapping_source: MappingSource
    source_pose: Matrix4x4 | None = None
    merged_node_id: int | None = None
    merged_stamp: float | None = None
    optimized_pose: Matrix4x4 | None = None
    residual_s: float | None = None
    missing_reason: str | None = None

    @property
    def is_usable(self) -> bool:
        return self.mapping_source in {"label", "stamp"} and self.optimized_pose is not None


@dataclass(frozen=True)
class PoseMappingMetrics:
    source_scan_ids: list[str]
    source_node_count: int
    merged_node_count: int
    label_count: int
    stamp_count: int
    ambiguous_count: int
    missing_count: int
    usable_count: int
    per_scan_source_node_count: dict[str, int]
    per_scan_usable_count: dict[str, int]

    @property
    def usable_ratio(self) -> float:
        return self.usable_count / self.source_node_count if self.source_node_count else 0.0

    @property
    def missing_ratio(self) -> float:
        return self.missing_count / self.source_node_count if self.source_node_count else 0.0

    @property
    def ambiguous_ratio(self) -> float:
        return self.ambiguous_count / self.source_node_count if self.source_node_count else 0.0

    def per_scan_usable_ratio(self) -> dict[str, float]:
        return {
            scan_id: (
                self.per_scan_usable_count.get(scan_id, 0) / count
                if count > 0 else 0.0
            )
            for scan_id, count in self.per_scan_source_node_count.items()
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "source_scan_ids": self.source_scan_ids,
            "source_node_count": self.source_node_count,
            "merged_node_count": self.merged_node_count,
            "label_count": self.label_count,
            "stamp_count": self.stamp_count,
            "ambiguous_count": self.ambiguous_count,
            "missing_count": self.missing_count,
            "usable_count": self.usable_count,
            "usable_ratio": self.usable_ratio,
            "missing_ratio": self.missing_ratio,
            "ambiguous_ratio": self.ambiguous_ratio,
            "per_scan_source_node_count": self.per_scan_source_node_count,
            "per_scan_usable_count": self.per_scan_usable_count,
            "per_scan_usable_ratio": self.per_scan_usable_ratio(),
        }


@dataclass(frozen=True)
class NodePoseMappingResult:
    mappings: list[NodePoseMapping]
    metrics: PoseMappingMetrics

    def usable_mappings_by_scan(self) -> dict[str, list[NodePoseMapping]]:
        grouped: dict[str, list[NodePoseMapping]] = defaultdict(list)
        for mapping in self.mappings:
            if mapping.is_usable:
                grouped[mapping.source_scan_id].append(mapping)
        for values in grouped.values():
            values.sort(key=lambda item: item.source_stamp)
        return dict(grouped)


@dataclass(frozen=True)
class FramePoseAssignment:
    source_scan_id: str
    source_timestamp: float
    pose_source: FramePoseSource
    confidence: float
    before_node_id: int | None = None
    after_node_id: int | None = None
    reason: str | None = None


@dataclass(frozen=True)
class FramePoseInMerged:
    assignment: FramePoseAssignment
    pose: Matrix4x4 | None


def build_provenance_label(source: SourceNodeRef) -> str:
    values = {
        "scan": source.scan_id,
        "node": str(source.node_id),
        "stamp": repr(source.stamp),
    }
    if source.original_label:
        values["label"] = quote(source.original_label, safe="")
    return PROVENANCE_LABEL_PREFIX + urlencode(values)


def parse_provenance_label(label: str | None) -> SourceNodeRef | None:
    if not label or not label.startswith(PROVENANCE_LABEL_PREFIX):
        return None
    raw = label[len(PROVENANCE_LABEL_PREFIX):]
    parsed = parse_qs(raw, keep_blank_values=True)
    try:
        scan_id = _one(parsed, "scan")
        node_id = int(_one(parsed, "node"))
        stamp = float(_one(parsed, "stamp"))
    except (KeyError, TypeError, ValueError):
        return None
    original_label = parsed.get("label", [None])[0]
    return SourceNodeRef(
        scan_id=scan_id,
        node_id=node_id,
        stamp=stamp,
        original_label=unquote(original_label) if original_label else None,
    )


def load_rtabmap_node_records(db_path: Path) -> list[RtabmapNodeRecord]:
    uri = f"file:{db_path}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, map_id, stamp, pose, label FROM Node ORDER BY id"
        ).fetchall()
        records: list[RtabmapNodeRecord] = []
        for row in rows:
            pose = row["pose"]
            if pose is None:
                continue
            records.append(
                RtabmapNodeRecord(
                    node_id=int(row["id"]),
                    map_id=int(row["map_id"]),
                    stamp=float(row["stamp"] or 0.0),
                    pose=decode_pose_3x4_blob(pose),
                    label=row["label"],
                )
            )
        return records
    finally:
        conn.close()


def build_node_pose_mapping(
    *,
    source_dbs: Mapping[str, Path],
    merged_db: Path,
    stamp_tolerance_s: float = 0.033,
) -> NodePoseMappingResult:
    source_records = {
        scan_id: load_rtabmap_node_records(path)
        for scan_id, path in source_dbs.items()
    }
    source_scan_ids = list(source_dbs.keys())
    source_by_key = {
        (scan_id, node.node_id): node
        for scan_id, nodes in source_records.items()
        for node in nodes
    }
    merged_nodes = load_rtabmap_node_records(merged_db)
    merged_map_ids = {node.map_id for node in merged_nodes}

    mapped: dict[tuple[str, int], NodePoseMapping] = {}
    ambiguous: dict[tuple[str, int], NodePoseMapping] = {}

    for merged in merged_nodes:
        source_ref = parse_provenance_label(merged.label)
        if source_ref is not None and source_ref.key in source_by_key:
            source_node = source_by_key[source_ref.key]
            mapped[source_ref.key] = NodePoseMapping(
                source_scan_id=source_ref.scan_id,
                source_node_id=source_ref.node_id,
                source_stamp=source_ref.stamp,
                mapping_source="label",
                source_pose=source_node.pose,
                merged_node_id=merged.node_id,
                merged_stamp=merged.stamp,
                optimized_pose=merged.pose,
                residual_s=abs(merged.stamp - source_ref.stamp),
            )
            continue

        candidates = _source_candidates_by_stamp(
            source_records,
            merged_stamp=merged.stamp,
            tolerance_s=stamp_tolerance_s,
            expected_scan_id=_scan_id_for_merged_map_id(
                merged.map_id,
                source_scan_ids=source_scan_ids,
                merged_map_ids=merged_map_ids,
            ),
        )
        selected_ref = _unique_nearest_stamp_candidate(
            candidates,
            merged_stamp=merged.stamp,
        )
        if selected_ref is not None:
            ref = selected_ref
            key = ref.key
            source_node = source_by_key[key]
            mapped.setdefault(
                key,
                NodePoseMapping(
                    source_scan_id=ref.scan_id,
                    source_node_id=ref.node_id,
                    source_stamp=ref.stamp,
                    mapping_source="stamp",
                    source_pose=source_node.pose,
                    merged_node_id=merged.node_id,
                    merged_stamp=merged.stamp,
                    optimized_pose=merged.pose,
                    residual_s=abs(merged.stamp - ref.stamp),
                ),
            )
        elif candidates:
            for ref in candidates:
                ambiguous.setdefault(
                    ref.key,
                    NodePoseMapping(
                        source_scan_id=ref.scan_id,
                        source_node_id=ref.node_id,
                        source_stamp=ref.stamp,
                        mapping_source="ambiguous",
                        missing_reason="stamp_match_ambiguous",
                    ),
                )

    mappings: list[NodePoseMapping] = []
    for key, source in sorted(source_by_key.items()):
        if key in mapped:
            mappings.append(mapped[key])
        elif key in ambiguous:
            mappings.append(ambiguous[key])
        else:
            mappings.append(
                NodePoseMapping(
                    source_scan_id=key[0],
                    source_node_id=key[1],
                    source_stamp=source.stamp,
                    mapping_source="missing",
                    source_pose=source.pose,
                    missing_reason="no_merged_node_match",
                )
            )

    return NodePoseMappingResult(
        mappings=mappings,
        metrics=_metrics(
            source_scan_ids=list(source_dbs.keys()),
            source_records=source_records,
            merged_node_count=len(merged_nodes),
            mappings=mappings,
        ),
    )


def _scan_id_for_merged_map_id(
    map_id: int,
    *,
    source_scan_ids: list[str],
    merged_map_ids: set[int],
) -> str | None:
    """Narrow RTAB-Map append-mode stamp fallback when merged map IDs are useful.

    `rtabmap-reprocess -a` often rewrites labels for appended databases, but it
    still preserves sessions as `Node.map_id=0,1,...`. Use that only when the
    merged DB actually has more than one map id; otherwise single-map synthetic
    or legacy DBs should keep fail-closed timestamp ambiguity behavior.
    """
    if len(merged_map_ids) <= 1:
        return None
    if map_id < 0 or map_id >= len(source_scan_ids):
        return None
    return source_scan_ids[map_id]


def assign_frame_pose_source(
    *,
    source_scan_id: str,
    source_timestamp: float,
    mapping_result: NodePoseMappingResult,
    exact_tolerance_s: float = 0.033,
) -> FramePoseAssignment:
    by_scan = mapping_result.usable_mappings_by_scan()
    usable = by_scan.get(source_scan_id, [])
    if not usable:
        return FramePoseAssignment(
            source_scan_id=source_scan_id,
            source_timestamp=source_timestamp,
            pose_source="excluded",
            confidence=0.0,
            reason="no_usable_scan_pose_mapping",
        )

    nearest = min(usable, key=lambda item: abs(item.source_stamp - source_timestamp))
    if abs(nearest.source_stamp - source_timestamp) <= exact_tolerance_s:
        return FramePoseAssignment(
            source_scan_id=source_scan_id,
            source_timestamp=source_timestamp,
            pose_source="optimized_node_pose",
            confidence=1.0,
            before_node_id=nearest.source_node_id,
            after_node_id=nearest.source_node_id,
        )

    before = [item for item in usable if item.source_stamp < source_timestamp]
    after = [item for item in usable if item.source_stamp > source_timestamp]
    if before and after:
        left = before[-1]
        right = after[0]
        return FramePoseAssignment(
            source_scan_id=source_scan_id,
            source_timestamp=source_timestamp,
            pose_source="interpolated_optimized_pose",
            confidence=0.85,
            before_node_id=left.source_node_id,
            after_node_id=right.source_node_id,
        )

    return FramePoseAssignment(
        source_scan_id=source_scan_id,
        source_timestamp=source_timestamp,
        pose_source="raw_pose_transformed",
        confidence=0.55,
        before_node_id=nearest.source_node_id,
        after_node_id=nearest.source_node_id,
        reason="outside_optimized_node_stamp_range",
    )


def resolve_frame_pose_in_merged(
    *,
    source_scan_id: str,
    source_timestamp: float,
    raw_source_pose: np.ndarray,
    mapping_result: NodePoseMappingResult,
    exact_tolerance_s: float = 0.033,
) -> FramePoseInMerged:
    """Transform one source-frame pose into the merged RTAB-Map coordinate frame.

    `raw_source_pose` must be expressed in the same source coordinate frame as
    the source RTAB-Map node poses. The returned pose is suitable for dense
    frame evidence projection into the merged map frame.
    """
    assignment = assign_frame_pose_source(
        source_scan_id=source_scan_id,
        source_timestamp=source_timestamp,
        mapping_result=mapping_result,
        exact_tolerance_s=exact_tolerance_s,
    )
    if assignment.pose_source == "excluded":
        return FramePoseInMerged(assignment=assignment, pose=None)

    by_scan = mapping_result.usable_mappings_by_scan()
    usable = by_scan.get(source_scan_id, [])
    if not usable:
        return FramePoseInMerged(assignment=assignment, pose=None)

    if assignment.pose_source in {"optimized_node_pose", "raw_pose_transformed"}:
        nearest = _mapping_by_node_id(usable, assignment.before_node_id)
        if nearest is None:
            return FramePoseInMerged(assignment=assignment, pose=None)
        pose = _apply_mapping_correction(nearest, raw_source_pose)
        return FramePoseInMerged(assignment=assignment, pose=_numpy_to_matrix4(pose))

    if assignment.pose_source == "interpolated_optimized_pose":
        left = _mapping_by_node_id(usable, assignment.before_node_id)
        right = _mapping_by_node_id(usable, assignment.after_node_id)
        if left is None or right is None:
            return FramePoseInMerged(assignment=assignment, pose=None)
        left_pose = _apply_mapping_correction(left, raw_source_pose)
        right_pose = _apply_mapping_correction(right, raw_source_pose)
        denom = right.source_stamp - left.source_stamp
        alpha = 0.0 if denom <= 0 else (source_timestamp - left.source_stamp) / denom
        alpha = min(1.0, max(0.0, float(alpha)))
        pose = _interpolate_pose(left_pose, right_pose, alpha)
        return FramePoseInMerged(assignment=assignment, pose=_numpy_to_matrix4(pose))

    return FramePoseInMerged(assignment=assignment, pose=None)


def _one(parsed: Mapping[str, list[str]], key: str) -> str:
    values = parsed[key]
    if len(values) != 1:
        raise ValueError(key)
    return values[0]


def _source_candidates_by_stamp(
    source_records: Mapping[str, list[RtabmapNodeRecord]],
    *,
    merged_stamp: float,
    tolerance_s: float,
    expected_scan_id: str | None = None,
) -> list[SourceNodeRef]:
    candidates: list[SourceNodeRef] = []
    for scan_id, nodes in source_records.items():
        if expected_scan_id is not None and scan_id != expected_scan_id:
            continue
        for node in nodes:
            if math.isclose(node.stamp, merged_stamp, abs_tol=tolerance_s):
                candidates.append(
                    SourceNodeRef(
                        scan_id=scan_id,
                        node_id=node.node_id,
                        stamp=node.stamp,
                        original_label=node.label,
                    )
                )
    return candidates


def _unique_nearest_stamp_candidate(
    candidates: list[SourceNodeRef],
    *,
    merged_stamp: float,
    residual_gap_s: float = 1e-6,
) -> SourceNodeRef | None:
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda item: (abs(item.stamp - merged_stamp), item.scan_id, item.node_id),
    )
    if len(ranked) == 1:
        return ranked[0]
    best_residual = abs(ranked[0].stamp - merged_stamp)
    second_residual = abs(ranked[1].stamp - merged_stamp)
    if second_residual - best_residual > residual_gap_s:
        return ranked[0]
    return None


def _metrics(
    *,
    source_scan_ids: list[str],
    source_records: Mapping[str, list[RtabmapNodeRecord]],
    merged_node_count: int,
    mappings: list[NodePoseMapping],
) -> PoseMappingMetrics:
    counts = Counter(mapping.mapping_source for mapping in mappings)
    per_scan_source = {scan_id: len(nodes) for scan_id, nodes in source_records.items()}
    per_scan_usable: dict[str, int] = dict.fromkeys(source_scan_ids, 0)
    for mapping in mappings:
        if mapping.is_usable:
            per_scan_usable[mapping.source_scan_id] = (
                per_scan_usable.get(mapping.source_scan_id, 0) + 1
            )
    return PoseMappingMetrics(
        source_scan_ids=source_scan_ids,
        source_node_count=len(mappings),
        merged_node_count=merged_node_count,
        label_count=counts["label"],
        stamp_count=counts["stamp"],
        ambiguous_count=counts["ambiguous"],
        missing_count=counts["missing"],
        usable_count=counts["label"] + counts["stamp"],
        per_scan_source_node_count=per_scan_source,
        per_scan_usable_count=per_scan_usable,
    )


def _mapping_by_node_id(
    mappings: list[NodePoseMapping],
    node_id: int | None,
) -> NodePoseMapping | None:
    if node_id is None:
        return None
    for mapping in mappings:
        if mapping.source_node_id == node_id:
            return mapping
    return None


def _apply_mapping_correction(
    mapping: NodePoseMapping,
    raw_source_pose: np.ndarray,
) -> np.ndarray:
    if mapping.source_pose is None or mapping.optimized_pose is None:
        raise ValueError("mapping must include source_pose and optimized_pose")
    source_pose = _matrix4_to_numpy(mapping.source_pose)
    optimized_pose = _matrix4_to_numpy(mapping.optimized_pose)
    correction = optimized_pose @ np.linalg.inv(source_pose)
    return correction @ raw_source_pose.astype(np.float64)


def _interpolate_pose(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    rotations = Rotation.from_matrix(np.stack([left[:3, :3], right[:3, :3]], axis=0))
    out[:3, :3] = Slerp([0.0, 1.0], rotations)([alpha]).as_matrix()[0]
    out[:3, 3] = (1.0 - alpha) * left[:3, 3] + alpha * right[:3, 3]
    return out


def _matrix4_to_numpy(matrix: Matrix4x4) -> np.ndarray:
    return np.array(matrix, dtype=np.float64)


def _numpy_to_matrix4(matrix: np.ndarray) -> Matrix4x4:
    values = matrix.astype(float, copy=False)
    return (
        (float(values[0, 0]), float(values[0, 1]), float(values[0, 2]), float(values[0, 3])),
        (float(values[1, 0]), float(values[1, 1]), float(values[1, 2]), float(values[1, 3])),
        (float(values[2, 0]), float(values[2, 1]), float(values[2, 2]), float(values[2, 3])),
        (float(values[3, 0]), float(values[3, 1]), float(values[3, 2]), float(values[3, 3])),
    )
