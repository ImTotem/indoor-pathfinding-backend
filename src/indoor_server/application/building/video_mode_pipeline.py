"""video_mode_pipeline — Sprint 82 pipeline thin wrapper.

production 코드가 scripts/ 를 직접 참조하지 않도록 한 단계 격리.
향후 production 승격 시 이 파일 내부만 수정하면 된다 (ADR 0005 thin adapter).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_sprint82(
    merged_db: Path,
    baseline_polygon: Path | None,
    out_dir: Path,
) -> dict[str, Any]:
    """Sprint 82 full pipeline 호출 wrapper.

    Parameters
    ----------
    merged_db:
        RTAB-Map merged DB 경로 (single-scan이면 zip 내 rtabmap.db).
    baseline_polygon:
        floor polygon GeoJSON 경로. None 이면 polygon 단계(step_refine_polygon) 를 skip 하고
        nav graph 만 생성한다.
    out_dir:
        산출물 디렉토리.

    Returns
    -------
    dict:
        {
            "nodes": [{"node_id": str, "x": float, "y": float}, ...],
            "edges": [{"from_id": str, "to_id": str, "weight": float}, ...],
            "polygon_geojson_path": Path | None,
            "metrics": {...},
        }

    Raises
    ------
    RuntimeError:
        pipeline 실패 시 propagate. BuildService 가 FAILED 로 기록.
    """
    import inspect as _inspect

    from scripts.sprint82.run_pipeline import (
        main as sprint82_main,
    )
    from scripts.sprint82.run_pipeline import (
        step_build_navgraph,
        step_preflight,
    )

    # 시그니처 검증 — 2026-05 기준 main(merged_db, baseline_polygon, out_dir)
    sig = _inspect.signature(sprint82_main)
    expected_params = {"merged_db", "baseline_polygon", "out_dir"}
    actual_params = set(sig.parameters)
    if not expected_params <= actual_params:
        raise RuntimeError(
            f"sprint82.run_pipeline.main 시그니처 변경 감지: "
            f"expected {expected_params}, got {actual_params}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    polygon_geojson_path: Path | None = None

    if baseline_polygon is not None and baseline_polygon.exists():
        # baseline 있으면 full pipeline (polygon + nav graph)
        logger.info(
            "video_mode_pipeline: full pipeline (with baseline polygon) merged_db=%s",
            merged_db,
        )
        metrics = sprint82_main(
            merged_db=merged_db,
            baseline_polygon=baseline_polygon,
            out_dir=out_dir,
        )
        candidate = out_dir / "polygon" / "floor_polygon_refined.geojson"
        if candidate.exists():
            polygon_geojson_path = candidate
    else:
        # baseline 없으면 preflight + nav graph only (polygon 단계 skip)
        logger.info(
            "video_mode_pipeline: nav-graph-only mode (no baseline polygon) merged_db=%s",
            merged_db,
        )
        preflight = step_preflight(merged_db, out_dir)

        # nav graph 를 위해 pose xy 를 직접 추출
        from scripts.sprint82.refine_polygon import load_pose_xy

        xy, map_id, pose_ids = load_pose_xy(merged_db)

        # polygon 없이 nav graph 생성 (polygon=None → convex hull 사용)
        from shapely.geometry import MultiPoint

        polygon = MultiPoint(xy).convex_hull if len(xy) >= 3 else None

        nav_nodes, nav_edges, nav_metrics = step_build_navgraph(
            merged_db=merged_db,
            polygon=polygon,
            xy=xy,
            map_id=map_id,
            pose_ids=pose_ids,
            out_dir=out_dir,
        )

        metrics = {
            "preflight": preflight,
            "polygon": None,
            "navgraph": nav_metrics,
        }

    # navgraph JSON 읽어서 nodes/edges 추출
    navgraph_json = out_dir / "navgraph" / "nav_graph.json"
    if not navgraph_json.exists():
        raise RuntimeError(
            f"video_mode_pipeline: nav_graph.json 없음 ({navgraph_json})"
        )

    import json

    raw = json.loads(navgraph_json.read_text())
    raw_nodes = raw.get("nodes", [])
    # networkx JSON 형식: edges는 "links" 키 사용, 각 항목은 source/target
    raw_edges = raw.get("links", raw.get("edges", []))

    # 노드/엣지를 공통 형식으로 변환
    nodes_out = [
        {
            "node_id": n.get("node_id", n.get("id", "")),
            "x": float(n.get("x", 0.0)),
            "y": float(n.get("y", 0.0)),
        }
        for n in raw_nodes
    ]
    edges_out = [
        {
            # networkx: source/target; 기존: from_id/to_id
            "from_id": e.get("from_id", e.get("from", e.get("source", ""))),
            "to_id": e.get("to_id", e.get("to", e.get("target", ""))),
            "weight": float(e.get("weight", e.get("dist_m", 1.0))),
        }
        for e in raw_edges
    ]

    logger.info(
        "video_mode_pipeline: navgraph nodes=%d edges=%d",
        len(nodes_out),
        len(edges_out),
    )

    return {
        "nodes": nodes_out,
        "edges": edges_out,
        "polygon_geojson_path": polygon_geojson_path,
        "metrics": metrics,
    }
