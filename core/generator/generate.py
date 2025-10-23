"""Answer generation orchestration for ChronoRAG.

This module turns retrieval output into final natural-language answers.  It
constructs role-tagged chat messages, selects the most appropriate language
model backend (local HuggingFace, OpenAI-compatible, llama.cpp, Ollama, …),
and executes the generation call with domain-aware overrides.  When the LLM
fails, we degrade gracefully by returning a short evidence-only message.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from app.utils.chrono_reducer import ChronoPassage
from app.utils.time_windows import TimeWindow
from core.generator.llm_loader import load_backend
from core.generator.prompts import build_messages

# Separator injected into the prompt; we cut any text that follows this marker.
STOP_MARKER = "<|ATTR_CARD|>"


def _fallback_response(query: str, evidence: List[ChronoPassage]) -> str:
    """Return a deterministic message when no LLM backend is reachable."""
    top = evidence[0] if evidence else None
    if not top:
        return f"No direct in-window evidence found for: {query}" + STOP_MARKER
    return (
        f"Based on authoritative evidence, {top.text.strip()}"[:500]
        + "\n\n<|ATTR_CARD|>"
    )


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
    messages = build_messages(query, mode, axis, window, evidence, domain, window_kind)
    llm_cfg = models_cfg.get("llm", {})
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
    except Exception:
        raw = _fallback_response(query, evidence)
    clipped = raw.split(STOP_MARKER)[0].strip()
    token_estimate = max(1, len(clipped.split()))
    return clipped, token_estimate
