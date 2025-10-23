"""Simple authority/risk scoring derived from known source heuristics."""

from __future__ import annotations

from typing import Dict

from app.utils.authority import authority_from_uri


def score_source(uri: str) -> Dict[str, float]:
    """Return authority ∈ [0,1] and a complementary risk metric for the URI."""
    authority = authority_from_uri(uri)
    risk = 1.0 - authority
    return {"authority": authority, "risk": risk}
