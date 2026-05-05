"""Mock semantic analyzer.

외부 LLM/OCR 없이 관리자 label/class hint를 해석한다. 후속 Sprint에서
같은 인터페이스를 실제 Vision/LLM analyzer로 교체한다.
"""
from __future__ import annotations

import re

from indoor_server.domain.semantic.models import PlaceCategory, SemanticAnalysis

_STAIR_RE = re.compile(r"\b(?:STAIR|ST|계단)[_\-\s]*([A-Z0-9가-힣]+)?", re.IGNORECASE)
_ELEV_RE = re.compile(
    r"\b(?:ELEV|ELV|ELEVATOR|엘베|엘리베이터)[_\-\s]*([A-Z0-9가-힣]+)?",
    re.IGNORECASE,
)
_ROOM_RE = re.compile(r"(?:^|\b)([A-Z]?\d{2,4}[A-Z]?호?)(?:\b|$)", re.IGNORECASE)


class MockSemanticAnalyzer:
    """관리자 hint 기반 deterministic analyzer."""

    name = "mock"

    def analyze(
        self,
        *,
        label: str | None,
        class_name: str | None = None,
        source: str | None = None,
    ) -> SemanticAnalysis:
        text = " ".join(v for v in [label, class_name, source] if v).strip()
        lowered = text.lower()

        stair = _STAIR_RE.search(text)
        if stair is not None or "stairs" in lowered:
            key = self._connector_key("STAIR", stair.group(1) if stair else None)
            return SemanticAnalysis(
                category="stairs",
                name=label or "계단",
                confidence=0.8,
                analyzer=self.name,
                connector_type="stairs",
                connector_key=key,
                raw_text=text or None,
            )

        elev = _ELEV_RE.search(text)
        if elev is not None or "elevator" in lowered:
            key = self._connector_key("ELEV", elev.group(1) if elev else None)
            return SemanticAnalysis(
                category="elevator",
                name=label or "엘리베이터",
                confidence=0.8,
                analyzer=self.name,
                connector_type="elevator",
                connector_key=key,
                raw_text=text or None,
            )

        category = self._category_from_text(lowered)
        room_match = _ROOM_RE.search(text)
        name = label or (room_match.group(1) if room_match else None)
        confidence = 0.7 if name or category != "unknown" else 0.35
        return SemanticAnalysis(
            category=category,
            name=name,
            confidence=confidence,
            analyzer=self.name,
            raw_text=text or None,
            needs_review=confidence < 0.5,
        )

    def _category_from_text(self, text: str) -> PlaceCategory:
        if any(token in text for token in ["화장실", "restroom", "toilet", "bathroom"]):
            return "restroom"
        if any(token in text for token in ["입구", "entrance", "door", "doorway"]):
            return "entrance"
        if any(token in text for token in ["lab", "연구실", "실험실"]):
            return "lab"
        if any(token in text for token in ["office", "사무실", "교수실"]):
            return "office"
        if any(token in text for token in ["room", "강의실", "호"]):
            return "room"
        if any(token in text for token in ["vending", "printer", "service"]):
            return "service"
        return "destination" if text else "unknown"

    def _connector_key(self, prefix: str, suffix: str | None) -> str:
        clean = re.sub(r"[^A-Z0-9가-힣]+", "_", (suffix or "A").upper()).strip("_")
        return f"{prefix}_{clean or 'A'}"
