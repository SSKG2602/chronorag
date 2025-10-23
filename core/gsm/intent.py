"""Minimal heuristic intent detection for routing domain-specific logic."""

from __future__ import annotations

import re
from typing import Dict


def detect_intent(query: str) -> Dict[str, str]:
    """Return a coarse intent dict (domain + target) inferred from the query."""
    q = query.lower()
    if any(keyword in q for keyword in ["gdp", "per capita", "historical statistics", "maddison", "19th century", "18th century", "world economy", "industrial revolution", "population 1820"]):
        return {"domain": "world-economy", "target": "macro_history"}
    if re.search(r"\b1[0-9]{3}\b", q) and any(term in q for term in ["gdp", "population", "growth", "economy"]):
        return {"domain": "world-economy", "target": "macro_history"}
    if any(token in q for token in ["ceo", "chief executive", "leadership"]):
        return {"domain": "roles", "target": "leadership"}
    if "revenue" in q or "q2" in q:
        return {"domain": "finance", "target": "financials"}
    return {"domain": "generic", "target": "general"}
