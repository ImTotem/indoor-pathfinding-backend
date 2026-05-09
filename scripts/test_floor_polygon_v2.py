"""floor_polygon_v2 검증 — 1F 새 ZIP 의 noisy 데이터로 시뮬레이션."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indoor_server.application.building.steps.floor_polygon_v2 import (
    Node, Edge, build_floor_polygon,
)


def main() -> None:
    # 1F 새 ZIP 의 branch_mark + 가상 edge data
    nodes = [
        # corridor branch_mark — 5개 (id 1, 2, 7, 8, 9)
        Node("c1", "corridor", 0.46, -0.08, width_m=1.5),
        Node("c2", "corridor", 4.62,  2.45, width_m=1.5),
        Node("c7", "corridor", 3.51,  0.40, width_m=None),  # null → component 1.5 채움
        Node("c8", "corridor", 2.80,  0.17, width_m=None),
        Node("c9", "corridor", 4.09,  0.91, width_m=None),
        # corner branch_mark — 4개 (id 3, 4, 5, 6, 같은 mark_session_id)
        Node("k3", "corner", 6.44,  2.63, mark_session_id="sess1"),
        Node("k4", "corner", 5.69,  3.78, mark_session_id="sess1"),
        Node("k5", "corner", 4.71,  3.08, mark_session_id="sess1"),
        Node("k6", "corner", 5.32,  2.11, mark_session_id="sess1"),
    ]
    edges = [
        # corridor: c1 ↔ c2 직접 (둘 다 width=1.5 라 polygon 에 포함됨)
        Edge("e_main", "c1", "c2", "corridor"),
        # 그 외 width=null 노드 사이 edge — route 용 (polygon 제외)
        Edge("e1", "c1", "c8", "corridor"),
        Edge("e2", "c8", "c7", "corridor"),
        Edge("e3", "c7", "c9", "corridor"),
        Edge("e4", "c9", "c2", "corridor"),
        # corner 가 직사각형 cycle: k3 → k4 → k5 → k6 → k3
        Edge("ek1", "k3", "k4", "corner"),
        Edge("ek2", "k4", "k5", "corner"),
        Edge("ek3", "k5", "k6", "corner"),
        Edge("ek4", "k6", "k3", "corner"),
    ]
    fc = build_floor_polygon(nodes, edges, floor_id="1F-test")

    # 핵심 statistics
    rooms = [f for f in fc["features"] if f["properties"].get("kind") == "room"]
    corridors = [f for f in fc["features"] if f["properties"].get("kind") == "corridor"]
    unions = [f for f in fc["features"] if f["properties"].get("kind") == "floor_union"]

    print(f"FeatureCollection: {len(fc['features'])} features")
    print(f"  rooms      : {len(rooms)}")
    print(f"  corridors  : {len(corridors)}")
    print(f"  floor_union: {len(unions)}")
    print()
    if rooms:
        r = rooms[0]
        print(f"Room session={r['properties']['mark_session_id']}")
        print(f"  vertices : {r['properties']['vertex_count']}")
        print(f"  geometry type: {r['geometry']['type']}")
        coords = r['geometry']['coordinates'][0]
        print(f"  outline  : {len(coords)} pts (closed)")
    print()
    for c in corridors:
        p = c['properties']
        print(f"Corridor edge={p['edge_id']} comp={p['component_id']} "
              f"width={p['width_m']:.2f} m  {p['from_node_id']}→{p['to_node_id']}")
    print()
    if unions:
        u = unions[0]
        print(f"Floor union: {u['geometry']['type']} "
              f"rooms={u['properties']['rooms_count']} "
              f"corridors={u['properties']['corridors_count']}")
    print()
    out_path = Path(__file__).parent / "floor_polygon_v2_output.geojson"
    out_path.write_text(json.dumps(fc, ensure_ascii=False, indent=2))
    print(f"saved → {out_path}")


if __name__ == "__main__":
    main()
