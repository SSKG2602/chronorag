"""Cross-encoder reranker built on top of sentence-transformers."""

from __future__ import annotations

from typing import List, Tuple

from app.light_mode import light_mode_enabled


class CEReranker:
    """Wrapper around cross-encoder reranking with a deterministic light-mode stub."""

    def __init__(
        self,
        model_name: str = "bge-reranker-base",
        device: str | None = "auto",
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _ensure_model(self) -> None:
        """Load the cross-encoder lazily; install a stub when running in light mode."""
        if self._model is not None:
            return
        if light_mode_enabled():
            self._model = "_stub_"
            return
        from sentence_transformers import CrossEncoder  # noqa: WPS433

        device = None if self.device == "auto" else self.device
        self._model = CrossEncoder(self.model_name, max_length=512, device=device)

    def rerank(self, query: str, passages: List[str]) -> List[Tuple[int, float]]:
        """Return cross-encoder scores normalised to [0, 1] for the given passages."""
        self._ensure_model()
        if self._model == "_stub_":
            query_terms = set(query.lower().split())
            scores: List[Tuple[int, float]] = []
            for idx, passage in enumerate(passages):
                overlap = sum(1 for token in passage.lower().split() if token in query_terms)
                scores.append((idx, min(1.0, 0.1 * overlap)))
            scores.sort(key=lambda item: item[1], reverse=True)
            return scores

        import math

        pairs = [[query, passage] for passage in passages]
        scores: List[Tuple[int, float]] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            batch_scores = self._model.predict(batch, convert_to_numpy=True).tolist()
            for offset, score in enumerate(batch_scores):
                logistic = float(1 / (1 + math.exp(-score)))
                scores.append((start + offset, logistic))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores
