"""Answer generation orchestration for ChronoRAG.

This module turns retrieval output into final natural-language answers.  It
constructs role-tagged chat messages, selects the most appropriate language
model backend (local HuggingFace, OpenAI-compatible, llama.cpp, Ollama, …),
and executes the generation call with domain-aware overrides.  When the LLM
fails, we degrade gracefully by returning a short evidence-only message.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from app.utils.chrono_reducer import ChronoPassage
from app.utils.time_windows import TimeWindow
from core.generator.llm_loader import load_backend
from core.generator.prompts import build_messages

# Separator injected into the prompt; we cut any text that follows this marker.
STOP_MARKER = "<|ATTR_CARD|>"
logger = logging.getLogger(__name__)


def _format_passage_line(idx: int, passage: ChronoPassage) -> str:
    text = passage.text.strip().replace("\n", " ")
    if len(text) > 220:
        text = text[:220].rsplit(" ", 1)[0] + "…"
    window = f"{passage.valid_window.start.date()} → {passage.valid_window.end.date()}"
    source = passage.uri
    return f"{idx}. {text} (Window: {window}; Source: {source})"


def _fallback_response(query: str, evidence: List[ChronoPassage]) -> str:
    """Return a deterministic message when no LLM backend is reachable."""
    top = evidence[0] if evidence else None
    if not top:
        return f"No direct in-window evidence found for: {query}\n\n{STOP_MARKER}"
    lines = [
        "ChronoGuard fallback mode — unable to reach the language model, supplying evidence digest.",
        f"Query: {query}",
        "Key evidence:",
    ]
    for idx, passage in enumerate(evidence[:5], 1):
        lines.append(_format_passage_line(idx, passage))
    lines.append("Use the cited passages to construct the final narrative.")
    return "\n".join(lines) + "\n\n" + STOP_MARKER


def generate_answer(
    query: str,
    mode: str,
    axis: str,
    window: TimeWindow,
    evidence: List[ChronoPassage],
    models_cfg: Dict,
    domain: str,
    window_kind: str,
) -> Tuple[str, int]:
    """Generate an answer string and a token estimate from supplied evidence."""
    llm_cfg = models_cfg.get("llm", {})
    prompt_limits = llm_cfg.get("prompt_limits", {})
    max_passages = prompt_limits.get("max_passages")
    snippet_chars = prompt_limits.get("snippet_chars", 180)
    if not isinstance(snippet_chars, int) or snippet_chars <= 0:
        snippet_chars = 180
    prompt_evidence = evidence
    if isinstance(max_passages, int) and max_passages > 0:
        prompt_evidence = evidence[:max_passages]
    messages = build_messages(
        query,
        mode,
        axis,
        window,
        prompt_evidence,
        domain,
        window_kind,
        snippet_chars=snippet_chars,
    )
    stop_list = [STOP_MARKER]
    max_tokens = 512
    temperature = 0.15
    try:
        backend, backend_name = load_backend(llm_cfg)
        # Each backend exposes different knobs.  We read overrides lazily so that
        # configuration changes require no code updates.
        if backend_name == "local_hf":
            entry = llm_cfg.get("local_hf", {})
            stop_list = entry.get("stop", stop_list)
            max_tokens = entry.get("max_new_tokens", max_tokens)
            temperature = entry.get("temperature", temperature)
        elif backend_name == "openai_compat":
            entry = llm_cfg.get("openai_compat", {})
            stop_list = entry.get("stop", stop_list)
            max_tokens = entry.get("max_tokens", max_tokens)
            temperature = entry.get("temperature", temperature)
        elif backend_name == "llama_cpp":
            entry = llm_cfg.get("llama_cpp", {})
            stop_list = entry.get("stop", stop_list)
            max_tokens = entry.get("max_new_tokens", max_tokens)
            temperature = entry.get("temperature", temperature)
        elif backend_name == "ollama":
            entry = llm_cfg.get("ollama", {})
            stop_list = entry.get("stop", stop_list)
            max_tokens = entry.get("max_tokens", max_tokens)
            temperature = entry.get("temperature", temperature)
        raw = backend.generate(messages, max_tokens=max_tokens, temperature=temperature, stop=stop_list)
        if not raw or not raw.strip():
            logger.warning("LLM returned empty output; falling back to evidence digest")
            raw = _fallback_response(query, evidence)
    except Exception:
        logger.exception("LLM generation failed; returning evidence digest")
        raw = _fallback_response(query, evidence)
    clipped = raw.split(STOP_MARKER)[0].strip()
    if evidence and (
        not clipped
        or (len(clipped) < 40 and not any(ch in ".?!" for ch in clipped))
    ):
        logger.warning("LLM response too short to trust; emitting evidence digest")
        clipped = _fallback_response(query, evidence).split(STOP_MARKER)[0].strip()
    token_estimate = max(1, len(clipped.split()))
    return clipped, token_estimate
