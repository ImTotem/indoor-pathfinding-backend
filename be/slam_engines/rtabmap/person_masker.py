from __future__ import annotations


class PersonMasker:
    """Fail-open person masker used by legacy debug/localize v2 path."""

    def detect_boxes(self, image_bytes: bytes) -> list[tuple[int, int, int, int]]:
        _ = image_bytes
        return []
