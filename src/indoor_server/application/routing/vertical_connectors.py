"""Route-time vertical connector edge resolution.

Persisted map_edge only has skeleton/poi_spur values today. Multi-floor routing
therefore adds stair/elevator transitions as in-memory graph edges.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from uuid import NAMESPACE_URL, UUID, uuid5

import networkx as nx

from indoor_server.domain.routing.models import MapNodeRow


@dataclass(frozen=True)
class VerticalEdge:
    from_node_id: UUID
    to_node_id: UUID
    length_m: float
    source_ref: dict[str, object]


# 사용자 verticalPreference → 허용 connector_type set.
# escalator 는 connector_key_for_node 에서 미인식이라 graph 도달 안 하지만,
# 향후 supported 시 STAIRS 그룹에 함께 묶이도록 미리 포함.
_ALLOWED_BY_PREFERENCE: dict[str, set[str]] = {
    "ELEVATOR": {"elevator"},
    "STAIRS": {"stair", "stairs", "escalator"},
}


def _allowed_connector_types(vertical_preference: str | None) -> set[str] | None:
    if vertical_preference is None:
        return None
    return _ALLOWED_BY_PREFERENCE.get(vertical_preference.upper())


def _connector_type_of_edge(edge: VerticalEdge) -> str:
    return str(edge.source_ref.get("connector_type") or "").strip().lower()


def add_vertical_edges(
    graph: nx.Graph,
    node_lookup: dict[UUID, MapNodeRow],
    *,
    explicit_edges: list[VerticalEdge] | None = None,
    vertical_preference: str | None = None,
) -> int:
    """Add explicit and inferred stair/elevator transitions to graph.

    `vertical_preference`:
      - "ELEVATOR" → elevator only.
      - "STAIRS"   → stair/escalator only.
      - None       → no filter (legacy).
    """
    allowed = _allowed_connector_types(vertical_preference)

    added = 0
    for edge in explicit_edges or []:
        if edge.from_node_id not in node_lookup or edge.to_node_id not in node_lookup:
            continue
        if allowed is not None and _connector_type_of_edge(edge) not in allowed:
            continue
        _add_transition_edge(graph, edge)
        added += 1

    explicit_pairs = {
        frozenset((edge.from_node_id, edge.to_node_id))
        for edge in explicit_edges or []
    }
    for edge in infer_vertical_edges(node_lookup, vertical_preference=vertical_preference):
        if frozenset((edge.from_node_id, edge.to_node_id)) in explicit_pairs:
            continue
        _add_transition_edge(graph, edge)
        added += 1
    return added


def infer_vertical_edges(
    node_lookup: dict[UUID, MapNodeRow],
    *,
    vertical_preference: str | None = None,
) -> list[VerticalEdge]:
    """Infer cross-floor transitions from semantic stair/elevator nodes."""
    allowed = _allowed_connector_types(vertical_preference)
    groups: dict[str, list[MapNodeRow]] = defaultdict(list)
    for node in node_lookup.values():
        key = connector_key_for_node(node)
        if key is None:
            continue
        if allowed is not None:
            facility = key.split(":", 1)[0]
            if facility not in allowed:
                continue
        groups[key].append(node)

    edges: list[VerticalEdge] = []
    for key, nodes in groups.items():
        ordered = sorted(
            nodes,
            key=lambda n: (
                str(n.level_id or ""),
                str(n.scan_id or ""),
                str(n.node_id),
            ),
        )
        for index, first in enumerate(ordered):
            for second in ordered[index + 1:]:
                if _same_floor_stop(first, second):
                    continue
                connector_type = key.split(":", 1)[0]
                edges.append(
                    VerticalEdge(
                        from_node_id=first.node_id,
                        to_node_id=second.node_id,
                        length_m=_vertical_transition_cost_m(connector_type),
                        source_ref={
                            "edge_kind": "vertical_connector",
                            "source": "semantic_inference",
                            "connector_type": connector_type,
                            "connector_key": key,
                            "from_scan_id": str(first.scan_id)
                            if first.scan_id is not None
                            else None,
                            "to_scan_id": str(second.scan_id)
                            if second.scan_id is not None
                            else None,
                            "from_level_id": first.level_id,
                            "to_level_id": second.level_id,
                        },
                    )
                )
    return edges


def connector_key_for_node(node: MapNodeRow) -> str | None:
    """Return normalized connector key for stair/elevator semantic nodes."""
    source_ref = node.source_ref or {}
    facility_type = str(source_ref.get("facility_type") or "").strip().lower()
    if facility_type not in {"stair", "elevator"}:
        facility_type = _facility_type_from_label(node.label)
    if facility_type not in {"stair", "elevator"}:
        return None

    explicit_key = source_ref.get("connector_key")
    if explicit_key is not None and str(explicit_key).strip():
        return str(explicit_key).strip().lower()

    label_key = _connector_suffix_from_label(node.label, facility_type)
    return f"{facility_type}:{label_key}"


def make_connector_key(label: str | None, facility_type: str) -> str | None:
    """Build a stable connector key for display graph source_ref."""
    normalized_type = facility_type.strip().lower()
    if normalized_type not in {"stair", "elevator"}:
        return None
    return f"{normalized_type}:{_connector_suffix_from_label(label, normalized_type)}"


def _add_transition_edge(graph: nx.Graph, edge: VerticalEdge) -> None:
    graph.add_edge(
        edge.from_node_id,
        edge.to_node_id,
        length_m=edge.length_m,
        source_ref=edge.source_ref,
        edge_id=uuid5(
            NAMESPACE_URL,
            "indoor-route-vertical:"
            f"{edge.from_node_id}:{edge.to_node_id}:{edge.source_ref.get('connector_key')}",
        ),
    )


def _same_floor_stop(first: MapNodeRow, second: MapNodeRow) -> bool:
    if first.scan_id is not None and second.scan_id is not None and first.scan_id != second.scan_id:
        return False
    if first.level_id is not None and second.level_id is not None:
        return first.level_id == second.level_id
    return first.scan_id == second.scan_id


def _facility_type_from_label(label: str | None) -> str:
    if label is None:
        return ""
    value = label.strip().lower()
    if value.startswith("stair") or "계단" in value:
        return "stair"
    if value.startswith("elev") or "엘리베이터" in value or "엘베" in value:
        return "elevator"
    return ""


def _connector_suffix_from_label(label: str | None, facility_type: str) -> str:
    if label is None or not label.strip():
        return "default"
    value = label.strip().lower()
    prefixes = {
        "stair": ["stair_", "stair-", "stair "],
        "elevator": ["elev_", "elev-", "elev ", "elevator_", "elevator-"],
    }
    for prefix in prefixes.get(facility_type, []):
        if value.startswith(prefix):
            suffix = value[len(prefix):].split()[0]
            return _slug(suffix)
    first_token = value.split()[0]
    if first_token.startswith("stair"):
        return _slug(first_token.removeprefix("stair").strip("_- ") or "default")
    if first_token.startswith("elev"):
        return _slug(
            first_token.removeprefix("elevator").removeprefix("elev").strip("_- ")
            or "default"
        )
    return _slug(first_token)


def _slug(value: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")
    return out or "default"


def _vertical_transition_cost_m(connector_type: str) -> float:
    if connector_type == "elevator":
        return 8.0
    if connector_type in {"stair", "stairs", "escalator"}:
        return 12.0
    return 10.0
