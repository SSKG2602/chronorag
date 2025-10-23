"""Toy entity matcher used in tests and light-mode runs."""

from __future__ import annotations

from typing import Dict


def match_entities(text: str) -> Dict[str, str]:
    """Return a minimal entity label with confidence based on keyword heuristics."""
    if "chronocorp" in text.lower():
        return {"entity": "ChronoCorp", "confidence": 0.9}
    return {"entity": "Unknown", "confidence": 0.3}
