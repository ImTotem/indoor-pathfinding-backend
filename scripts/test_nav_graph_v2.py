"""nav_graph_v2 smoke test."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indoor_server.application.building.steps.nav_graph_v2 import (
    V2Node, V2Edge, build_nav_graph_v2,
)


def main() -> None:
    # corridor backbone: 4 노드 직선 chain
    corridor_nodes = [
        V2Node("c1", "corridor", 0.0, 0.0),
        V2Node("c2", "corridor", 5.0, 0.0),
        V2Node("c3", "corridor", 10.0, 0.0),
        V2Node("c4", "corridor", 15.0, 0.0),
    ]
    corridor_edges = [
        V2Edge("ec1", "c1", "c2", "corridor", 5.0),
        V2Edge("ec2", "c2", "c3", "corridor", 5.0),
        V2Edge("ec3", "c3", "c4", "corridor", 5.0),
    ]

    # POI/interfloor: 다양한 위치
    attach_targets = [
        V2Node("p1", "poi", 2.5, 2.0, label="301호"),       # ec1 의 중간 위 → split
        V2Node("p2", "poi", 7.5, -1.5, label="STAIRS"),    # ec2 의 중간 아래 → split
        V2Node("p3", "poi", 0.02, 1.0, label="ENTRANCE"), # c1 끝점 epsilon 내 → 끝점 reuse
        V2Node("p4", "poi", 14.95, 0.5, label="EXIT"),    # c4 끝점 epsilon 내 → 끝점 reuse
        V2Node("p5", "junction", 12.5, 3.0, label=None),  # ec3 의 중간 위 → split
    ]

    nodes, edges = build_nav_graph_v2(
        corridor_nodes=corridor_nodes,
        corridor_edges=corridor_edges,
        attach_targets=attach_targets,
    )

    print(f"Result: {len(nodes)} nodes, {len(edges)} edges")
    print()
    print("=== Nodes ===")
    for n in nodes:
        ref = n.source_ref or {}
        role = ref.get("role", "")
        print(f"  {n.node_id[:8]:8s} kind={n.kind:9s} pos=({n.x:5.2f},{n.y:5.2f},{n.z:4.1f}) "
              f"label={n.label!r:12s} role={role}")
    print()
    print("=== Edges ===")
    for e in edges:
        print(f"  {e.edge_id[:8]:8s} kind={e.kind:11s} {e.from_node_id[:8]:8s} → {e.to_node_id[:8]:8s} "
              f"len={e.length_m:.2f}")
    print()
    # 통계
    foot_count = sum(1 for n in nodes if n.kind == "attach")
    spur_count = sum(1 for e in edges if e.kind == "poi_spur")
    corridor_after = sum(1 for e in edges if e.kind == "corridor")
    print(f"Stats: foot_nodes={foot_count}, spur_edges={spur_count}, corridor_edges={corridor_after}")
    print(f"  expected: foot=3 (p1,p2,p5 split), spur=5 (모든 target), corridor=3+3=6 (3 split → 3*2=6)")


if __name__ == "__main__":
    main()
