"""Lightweight BM25 wrapper with a deterministic fallback for tests/light mode."""

from __future__ import annotations

from typing import List, Sequence, Tuple

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:  # pragma: no cover
    BM25Okapi = None


def tokenize(text: str) -> List[str]:
    """Lowercase whitespace tokeniser shared by the BM25 and fallback paths."""
    return [tok.lower() for tok in text.split()]


def bm25_search(query: str, documents: Sequence[Tuple[str, str]], top_k: int = 5) -> List[Tuple[str, float]]:
    """Return top-k lexical matches using BM25 when available, else a simple overlap proxy."""
    if not documents:
        return []
    corpus_tokens = [tokenize(text) for _, text in documents]
    if BM25Okapi is not None:
        bm25 = BM25Okapi(corpus_tokens)
        scores = bm25.get_scores(tokenize(query))
    else:
        query_tokens = set(tokenize(query))
        scores = [len(query_tokens.intersection(set(tokens))) for tokens in corpus_tokens]
    paired = [(doc_id, float(score)) for (doc_id, _), score in zip(documents, scores)]
    ranked = sorted(paired, key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
