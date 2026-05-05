"""Run RTAB-Map multi-database reprocess while preserving source provenance."""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from indoor_server.application.building.multiscan_pose_mapping import (
    SourceNodeRef,
    build_provenance_label,
)

logger = logging.getLogger(__name__)


class MultiScanRtabmapMergeError(Exception):
    """RTAB-Map multi-scan merge failed."""


@dataclass(frozen=True)
class SourceRtabmapScan:
    scan_id: str
    db_path: Path


@dataclass(frozen=True)
class PreparedRtabmapScan:
    scan_id: str
    source_db_path: Path
    prepared_db_path: Path
    labeled_node_count: int


@dataclass(frozen=True)
class MultiScanReprocessParams:
    append_mode: bool = True
    skip: int = 0
    feature_strategy: int = 1
    rehearsal_similarity: float = 1.0
    not_linked_nodes_kept: bool = True
    reduce_graph: bool = False
    memory_thr: int = 0
    time_thr: int = 0
    optimize_max_error: float = 3.0
    warn: bool = True
    extra_args: tuple[str, ...] = ()

    def to_args(self) -> list[str]:
        args: list[str] = []
        if self.append_mode:
            args.append("-a")
        args.extend(
            [
                "-skip",
                str(self.skip),
                f"--Kp/DetectorStrategy={self.feature_strategy}",
                f"--Vis/FeatureType={self.feature_strategy}",
                f"--Mem/RehearsalSimilarity={self.rehearsal_similarity}",
                f"--Mem/NotLinkedNodesKept={str(self.not_linked_nodes_kept).lower()}",
                f"--Mem/ReduceGraph={str(self.reduce_graph).lower()}",
                f"--Rtabmap/MemoryThr={self.memory_thr}",
                f"--Rtabmap/TimeThr={self.time_thr}",
                f"--RGBD/OptimizeMaxError={self.optimize_max_error}",
            ]
        )
        if self.warn:
            args.append("--uwarn")
        args.extend(self.extra_args)
        return args


@dataclass(frozen=True)
class MultiScanReprocessResult:
    output_db_path: Path
    duration_s: float
    source_scan_ids: list[str]
    prepared_scans: list[PreparedRtabmapScan]
    command: list[str]
    stdout_text: str
    stderr_text: str
    merged_node_count: int
    loop_closure_count: int
    extra: dict[str, object] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, object]:
        return {
            "output_db_path": str(self.output_db_path),
            "duration_s": self.duration_s,
            "source_scan_ids": self.source_scan_ids,
            "prepared_scans": [
                {
                    "scan_id": scan.scan_id,
                    "source_db_path": str(scan.source_db_path),
                    "prepared_db_path": str(scan.prepared_db_path),
                    "labeled_node_count": scan.labeled_node_count,
                }
                for scan in self.prepared_scans
            ],
            "command": self.command,
            "merged_node_count": self.merged_node_count,
            "loop_closure_count": self.loop_closure_count,
            "extra": self.extra,
        }


class MultiScanRtabmapReprocessRunner:
    """`rtabmap-reprocess -a "db1;db2"` wrapper for evidence and later production use."""

    def __init__(
        self,
        *,
        binary_path: str | None = None,
        default_timeout_s: float = 900.0,
    ) -> None:
        self._binary_path = binary_path or os.environ.get(
            "RTABMAP_REPROCESS_BIN"
        ) or shutil.which("rtabmap-reprocess")
        self._default_timeout_s = default_timeout_s

    @property
    def binary_path(self) -> str | None:
        return self._binary_path

    def is_available(self) -> bool:
        return self._binary_path is not None and (
            Path(self._binary_path).exists() or shutil.which(self._binary_path) is not None
        )

    async def run(
        self,
        *,
        sources: list[SourceRtabmapScan],
        output_db: Path,
        work_dir: Path,
        params: MultiScanReprocessParams | None = None,
        timeout_s: float | None = None,
    ) -> MultiScanReprocessResult:
        if self._binary_path is None:
            raise MultiScanRtabmapMergeError(
                "rtabmap-reprocess binary is not available in PATH or RTABMAP_REPROCESS_BIN."
            )
        if len(sources) < 2:
            raise MultiScanRtabmapMergeError("multi-scan merge requires at least 2 sources")

        for source in sources:
            if not source.db_path.exists():
                raise MultiScanRtabmapMergeError(f"source db missing: {source.db_path}")

        output_db.parent.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        if output_db.exists():
            output_db.unlink()

        prepared = prepare_rtabmap_sources(sources=sources, work_dir=work_dir)
        input_arg = ";".join(str(scan.prepared_db_path) for scan in prepared)
        effective_params = params or MultiScanReprocessParams()
        command = [
            self._binary_path,
            *effective_params.to_args(),
            input_arg,
            str(output_db),
        ]
        timeout = timeout_s if timeout_s is not None else self._default_timeout_s

        loop = asyncio.get_running_loop()
        start = loop.time()
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except TimeoutError as e:
            proc.kill()
            await proc.wait()
            raise MultiScanRtabmapMergeError(
                f"rtabmap multi-scan reprocess timeout > {timeout}s"
            ) from e
        duration = loop.time() - start

        stdout_text = stdout_b.decode(errors="replace") if stdout_b else ""
        stderr_text = stderr_b.decode(errors="replace") if stderr_b else ""
        if proc.returncode != 0:
            raise MultiScanRtabmapMergeError(
                f"rtabmap multi-scan reprocess exit code {proc.returncode}\n"
                f"stderr tail: {stderr_text[-1200:]}"
            )

        result = MultiScanReprocessResult(
            output_db_path=output_db,
            duration_s=duration,
            source_scan_ids=[source.scan_id for source in sources],
            prepared_scans=prepared,
            command=command,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            merged_node_count=count_nodes(output_db),
            loop_closure_count=count_loop_closures(output_db),
        )
        logger.info(
            "rtabmap multi-scan reprocess complete",
            extra=result.to_metadata(),
        )
        return result


def prepare_rtabmap_sources(
    *,
    sources: list[SourceRtabmapScan],
    work_dir: Path,
) -> list[PreparedRtabmapScan]:
    work_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    prepared: list[PreparedRtabmapScan] = []
    for index, source in enumerate(sources):
        if source.scan_id in seen:
            raise MultiScanRtabmapMergeError(f"duplicate source scan_id: {source.scan_id}")
        seen.add(source.scan_id)
        safe_scan_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in source.scan_id)
        target = work_dir / f"{index:02d}_{safe_scan_id}.db"
        if target.exists():
            target.unlink()
        shutil.copy2(source.db_path, target)
        labeled = inject_provenance_labels(target, scan_id=source.scan_id)
        prepared.append(
            PreparedRtabmapScan(
                scan_id=source.scan_id,
                source_db_path=source.db_path,
                prepared_db_path=target,
                labeled_node_count=labeled,
            )
        )
    return prepared


def inject_provenance_labels(db_path: Path, *, scan_id: str) -> int:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, stamp, label FROM Node ORDER BY id"
        ).fetchall()
        for row in rows:
            source = SourceNodeRef(
                scan_id=scan_id,
                node_id=int(row["id"]),
                stamp=float(row["stamp"] or 0.0),
                original_label=row["label"],
            )
            conn.execute(
                "UPDATE Node SET label = ? WHERE id = ?",
                (build_provenance_label(source), int(row["id"])),
            )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def count_nodes(db_path: Path) -> int:
    return _count_scalar(db_path, "SELECT COUNT(*) FROM Node")


def count_loop_closures(db_path: Path) -> int:
    return _count_scalar(db_path, "SELECT COUNT(*) FROM Link WHERE type IN (1, 2)")


def _count_scalar(db_path: Path, query: str) -> int:
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    try:
        row = conn.execute(query).fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0
    finally:
        conn.close()
