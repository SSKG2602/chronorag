"""Lookup helpers for mapping source types to authority scores."""

from __future__ import annotations

from typing import Dict


AUTHORITY_SCORES: Dict[str, float] = {
    "filing": 1.0,
    "regulator": 0.9,
    "official_site": 0.8,
    "reliable_press": 0.6,
    "blog": 0.3,
    "unknown": 0.2,
}


def authority_from_uri(uri: str) -> float:
    """Return an authority score inferred from the URI string."""
    uri = uri or ""
    uri_lower = uri.lower()
    if "sec" in uri_lower or "filing" in uri_lower:
        return AUTHORITY_SCORES["filing"]
    if "regulator" in uri_lower:
        return AUTHORITY_SCORES["regulator"]
    if "official" in uri_lower or "company" in uri_lower:
        return AUTHORITY_SCORES["official_site"]
    if "press" in uri_lower or "news" in uri_lower:
        return AUTHORITY_SCORES["reliable_press"]
    if "blog" in uri_lower:
        return AUTHORITY_SCORES["blog"]
    return AUTHORITY_SCORES["unknown"]


def authority_from_label(label: str) -> float:
    """Return the authority score associated with a known label."""
    return AUTHORITY_SCORES.get(label, AUTHORITY_SCORES["unknown"])
