"""audit_graph.py — map_node/map_edge 품질 감사 스크립트 (Sprint 23 트랙 A).

사용:
    uv run python scripts/audit_graph.py --scan-id <uuid>
    uv run python scripts/audit_graph.py --build-job-id <uuid>

출력: JSON 메트릭 stdout. 임계 위반 시 exit code 1.

메트릭:
    edge_length_p50     중앙값 (≥3.0m → WARNING)
    edge_length_p95     95퍼센타일 (≥6.0m → WARNING)
    degree_1_ratio      endpoint 비율 (≥0.4 → WARNING)
    junction_count      degree≥3 노드 수 (0 → WARNING)
    nodes_per_walkable_m2  노드 밀도 (≤0.5/m² → WARNING)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indoor_server.infrastructure.db.engine import engine
from indoor_server.infrastructure.db import tables as t
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── 임계값 ────────────────────────────────────────────────────────────────────
_THRESH_P50_M = 3.0
_THRESH_P95_M = 6.0
_THRESH_DEGREE1_RATIO = 0.4
_THRESH_NODES_PER_M2 = 0.5
_CELL_SIZE_M = 0.10


# ── DB 조회 ───────────────────────────────────────────────────────────────────

async def _load_by_scan_id(
    session: AsyncSession,
    scan_id: str,
) -> tuple[list[dict], list[dict], float | None]:
    """scan_id의 최신 non-stale 노드/에지 + walkable_cells 반환."""
    # 최신 build_job 조회
    job_row = (
        await session.execute(
            sa.select(t.build_job)
            .where(
                t.build_job.c.scan_id == scan_id,
                t.build_job.c.state == "succeeded",
            )
            .order_by(t.build_job.c.enqueued_at.desc())
            .limit(1)
        )
    ).first()

    walkable_cells: float | None = None
    if job_row is not None and job_row.counts is not None:
        walkable_cells = job_row.counts.get("walkable_cells")

    node_rows = (
        await session.execute(
            sa.select(
                t.map_node.c.node_id,
                t.map_node.c.node_type,
                sa.func.ST_X(t.map_node.c.geom).label("x"),
                sa.func.ST_Y(t.map_node.c.geom).label("y"),
                sa.func.ST_Z(t.map_node.c.geom).label("z"),
            ).where(
                t.map_node.c.scan_id == scan_id,
                t.map_node.c.is_stale == sa.false(),
            )
        )
    ).fetchall()

    edge_rows = (
        await session.execute(
            sa.select(
                t.map_edge.c.edge_id,
                t.map_edge.c.from_node_id,
                t.map_edge.c.to_node_id,
                t.map_edge.c.length_m,
            ).where(
                t.map_edge.c.scan_id == scan_id,
                t.map_edge.c.is_stale == sa.false(),
            )
        )
    ).fetchall()

    nodes = [
        {"node_id": r.node_id, "node_type": r.node_type, "x": r.x, "y": r.y, "z": r.z}
        for r in node_rows
    ]
    edges = [
        {
            "edge_id": r.edge_id,
            "from": r.from_node_id,
            "to": r.to_node_id,
            "length_m": r.length_m,
        }
        for r in edge_rows
    ]
    return nodes, edges, walkable_cells


async def _load_by_build_job_id(
    session: AsyncSession,
    build_job_id: str,
) -> tuple[list[dict], list[dict], float | None]:
    """build_job_id 기준 non-stale 노드/에지 + walkable_cells 반환."""
    job_row = (
        await session.execute(
            sa.select(t.build_job).where(t.build_job.c.build_job_id == build_job_id)
        )
    ).first()

    walkable_cells: float | None = None
    if job_row is not None and job_row.counts is not None:
        walkable_cells = job_row.counts.get("walkable_cells")

    scan_id = job_row.scan_id if job_row is not None else None
    if scan_id is None:
        return [], [], None

    node_rows = (
        await session.execute(
            sa.select(
                t.map_node.c.node_id,
                t.map_node.c.node_type,
                sa.func.ST_X(t.map_node.c.geom).label("x"),
                sa.func.ST_Y(t.map_node.c.geom).label("y"),
                sa.func.ST_Z(t.map_node.c.geom).label("z"),
            ).where(
                t.map_node.c.build_job_id == build_job_id,
                t.map_node.c.is_stale == sa.false(),
            )
        )
    ).fetchall()

    edge_rows = (
        await session.execute(
            sa.select(
                t.map_edge.c.edge_id,
                t.map_edge.c.from_node_id,
                t.map_edge.c.to_node_id,
                t.map_edge.c.length_m,
            ).where(
                t.map_edge.c.build_job_id == build_job_id,
                t.map_edge.c.is_stale == sa.false(),
            )
        )
    ).fetchall()

    nodes = [
        {"node_id": r.node_id, "node_type": r.node_type, "x": r.x, "y": r.y, "z": r.z}
        for r in node_rows
    ]
    edges = [
        {
            "edge_id": r.edge_id,
            "from": r.from_node_id,
            "to": r.to_node_id,
            "length_m": r.length_m,
        }
        for r in edge_rows
    ]
    return nodes, edges, walkable_cells


# ── 메트릭 계산 ───────────────────────────────────────────────────────────────

def _compute_metrics(
    nodes: list[dict],
    edges: list[dict],
    walkable_cells: float | None,
) -> dict:
    """networkx 그래프 변환 → 메트릭 계산 → dict 반환."""
    g: nx.Graph = nx.Graph()
    for n in nodes:
        g.add_node(n["node_id"], node_type=n["node_type"], x=n["x"], y=n["y"], z=n["z"])
    for e in edges:
        g.add_edge(e["from"], e["to"], length_m=e["length_m"])

    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()

    # edge_length 분포
    lengths = [d["length_m"] for _, _, d in g.edges(data=True)]
    if lengths:
        arr = np.array(lengths, dtype=np.float64)
        edge_length_p50 = float(np.percentile(arr, 50))
        edge_length_p95 = float(np.percentile(arr, 95))
    else:
        edge_length_p50 = 0.0
        edge_length_p95 = 0.0

    # degree 분포
    degrees = dict(g.degree())
    degree_1_count = sum(1 for d in degrees.values() if d == 1)
    junction_count = sum(1 for d in degrees.values() if d >= 3)
    degree_1_ratio = degree_1_count / max(n_nodes, 1)

    # nodes_per_walkable_m2
    nodes_per_walkable_m2: float | None = None
    if walkable_cells is not None and walkable_cells > 0:
        walkable_m2 = float(walkable_cells) * (_CELL_SIZE_M ** 2)
        nodes_per_walkable_m2 = n_nodes / walkable_m2

    # 임계 위반 검사
    warnings: list[str] = []
    if edge_length_p50 >= _THRESH_P50_M:
        warnings.append(
            f"edge_length_p50={edge_length_p50:.2f}m ≥ {_THRESH_P50_M}m — densify 강화 필요"
        )
    if edge_length_p95 >= _THRESH_P95_M:
        warnings.append(
            f"edge_length_p95={edge_length_p95:.2f}m ≥ {_THRESH_P95_M}m — max_edge_length_m 누락 의심"
        )
    if degree_1_ratio >= _THRESH_DEGREE1_RATIO:
        warnings.append(
            f"degree_1_ratio={degree_1_ratio:.2%} ≥ {_THRESH_DEGREE1_RATIO:.0%} — skeleton end-prune 검토"
        )
    if junction_count == 0 and n_nodes > 0:
        warnings.append("junction_count=0 — grid 모폴로지 단조 의심")
    if nodes_per_walkable_m2 is not None and nodes_per_walkable_m2 <= _THRESH_NODES_PER_M2:
        warnings.append(
            f"nodes_per_walkable_m2={nodes_per_walkable_m2:.3f} ≤ {_THRESH_NODES_PER_M2} — A* waypoint sparse"
        )

    return {
        "node_count": n_nodes,
        "edge_count": n_edges,
        "walkable_cells": walkable_cells,
        "edge_length_p50": round(edge_length_p50, 4),
        "edge_length_p95": round(edge_length_p95, 4),
        "degree_1_ratio": round(degree_1_ratio, 4),
        "junction_count": junction_count,
        "nodes_per_walkable_m2": round(nodes_per_walkable_m2, 4) if nodes_per_walkable_m2 is not None else None,
        "warnings": warnings,
        "passed": len(warnings) == 0,
    }


# ── main ──────────────────────────────────────────────────────────────────────

async def _main(args: argparse.Namespace) -> int:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        if args.scan_id:
            nodes, edges, walkable_cells = await _load_by_scan_id(session, args.scan_id)
        else:
            nodes, edges, walkable_cells = await _load_by_build_job_id(session, args.build_job_id)

    metrics = _compute_metrics(nodes, edges, walkable_cells)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if metrics["warnings"]:
        for w in metrics["warnings"]:
            print(f"WARN: {w}", file=sys.stderr)
        return 1
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="map_node/map_edge 품질 감사")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scan-id", help="scan_id (UUID)")
    group.add_argument("--build-job-id", help="build_job_id (UUID)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    exit_code = asyncio.run(_main(args))
    sys.exit(exit_code)
