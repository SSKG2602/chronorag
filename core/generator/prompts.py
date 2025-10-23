"""Prompt builders for ChronoRAG answer generation.

The generator receives structured passages and timing metadata; this module turns
that into stable system/user messages that every LLM backend understands.  We keep
the formatting expressive but deterministic so we can diff prompts across runs.
"""

from __future__ import annotations

from typing import List

from app.utils.chrono_reducer import ChronoPassage
from app.utils.time_windows import TimeWindow

BASE_SYSTEM_PROMPT = (
    "You are ChronoRAG Answerer (PRO). Use ONLY the supplied evidence. "
    "Respect time filters and axis. Prefer filings/regulators. If conflicting "
    "claims fall in disjoint windows, output a short timeline first. Always return "
    "an Attribution Card (window, sources, quotes). Be concise, then give 2–3 "
    "bullets of rationale."
)

DOMAIN_NOTES = {
    "world-economy": (
        "For domain=world-economy, keep answers time-scoped. If the user is vague, "
        "provide a compact timeline instead of a single number. Always cite benchmark "
        "years and units (e.g., '1990 international dollars'). First line: concise "
        "result with year & unit, followed by 2–3 bullets covering region coverage and comparability caveats."
    )
}


def build_user_prompt(
    query: str,
    mode: str,
    axis: str,
    window: TimeWindow,
    evidence: List[ChronoPassage],
    domain: str,
    window_kind: str,
    snippet_chars: int = 180,
) -> str:
    """Format a detailed user message capturing mode, axis, window, and evidence."""
    lines = [
        f"[MODE] {mode} • [AXIS] {axis} • [WINDOW] {window.start.date()} → {window.end.date()} • [DOMAIN] {domain} • [WINDOW_KIND] {window_kind}"
    ]
    lines.append(f"Question: {query}")
    lines.append("Evidence (ranked):")
    for passage in evidence:
        units = ", ".join(passage.units) if passage.units else "n/a"
        entities = ", ".join(passage.entities) if passage.entities else "n/a"
        region = passage.region or passage.facets.get("region") or passage.facets.get("domain", "n/a")
        snippet = passage.text[:snippet_chars]
        lines.append(
            "- [{score:.2f}] {window}: {text} — {uri}".format(
                score=passage.score,
                window=f"{passage.valid_window.start.date()} → {passage.valid_window.end.date()}",
                text=snippet,
                uri=passage.uri,
            )
        )
        lines.append(f"  Units: {units} • Entities: {entities} • Region: {region}")
    return "\n".join(lines)


def build_messages(
    query: str,
    mode: str,
    axis: str,
    window: TimeWindow,
    evidence: List[ChronoPassage],
    domain: str,
    window_kind: str,
    snippet_chars: int = 180,
) -> List[dict]:
    """Return the system/user message pair consumed by downstream LLM backends."""
    system_prompt = BASE_SYSTEM_PROMPT
    if domain in DOMAIN_NOTES:
        system_prompt = f"{system_prompt} {DOMAIN_NOTES[domain]}"
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": build_user_prompt(
                query,
                mode,
                axis,
                window,
                evidence,
                domain,
                window_kind,
                snippet_chars=snippet_chars,
            ),
        },
    ]
