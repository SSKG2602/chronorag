"""LLM-based reranking that judges passages using the answer prompt context."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from app.light_mode import light_mode_enabled
from core.generator.llm_loader import load_backend


class LLMJudgeReranker:
    """Uses an LLM to rescore passages when judge mode is enabled."""

    def __init__(self, models_cfg: Dict):
        self.cfg = models_cfg
        self.enabled = models_cfg.get("llm", {}).get("judge", {}).get("enabled", False)
        self._backend = None
        self._backend_name: str | None = None

    def _ensure_backend(self) -> None:
        """Initialise the judge backend only once (skipping when light mode is active)."""
        if self._backend or light_mode_enabled():
            return
        backend, name = load_backend(self.cfg["llm"])
        self._backend = backend
        self._backend_name = name

    def rerank(
        self,
        query: str,
        axis: str,
        window: Dict[str, str],
        passages: Sequence[Tuple[str, str, float, float, float]],
    ) -> List[Tuple[str, float]]:
        if not self.enabled or not passages:
            return []
        if light_mode_enabled():
            scored: List[Tuple[str, float]] = []
            for chunk_id, text, base, time_weight, authority in passages:
                score = min(1.0, base * 0.5 + authority * 0.3 + time_weight * 0.2)
                scored.append((chunk_id, score))
            return scored

        self._ensure_backend()
        if not self._backend:
            return []
        prompt = _format_prompt(query, axis, window, passages)
        judge_payload = [
            {
                "role": "system",
                "content": (
                    "Score passages 0.0-1.0 using JSON array. Fields: id, score, reason. "
                    "Focus on temporal fit, authority, contradiction penalty."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        raw = self._backend.generate(
            judge_payload,
            max_tokens=self.cfg["llm"]["judge"].get("max_tokens", 100),
            temperature=self.cfg["llm"]["judge"].get("temperature", 0.1),
            stop=["]"],
        )
        parsed = _parse_scores(raw)
        return [(chunk_id, parsed.get(chunk_id, base)) for chunk_id, _, base, _, _ in passages]


def _format_prompt(
    query: str,
    axis: str,
    window: Dict[str, str],
    passages: Sequence[Tuple[str, str, float, float, float]],
) -> str:
    """Format passages and metadata into a compact prompt for the judge LLM."""
    header = f"Query: {query}\nAxis: {axis}\nWindow: {window['from']} → {window['to']}\nPassages:"
    lines: List[str] = []
    for idx, (chunk_id, text, base, time_weight, authority) in enumerate(passages, start=1):
        lines.append(
            f"{idx}. id={chunk_id} base={base:.3f} time={time_weight:.2f} auth={authority:.2f} :: {text[:220]}"
        )
    return "\n".join([header] + lines)


def _parse_scores(raw: str) -> Dict[str, float]:
    """Parse the judge JSON array into a chunk_id → score mapping."""
    import json

    raw = raw.strip()
    if not raw.startswith("["):
        raw = "[" + raw
    if not raw.endswith("]"):
        raw = raw + "]"
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    scores: Dict[str, float] = {}
    for item in data:
        try:
            chunk_id = str(item.get("id"))
            score = float(item.get("score", 0.0))
            scores[chunk_id] = max(0.0, min(1.0, score))
        except Exception:
            continue
    return scores
