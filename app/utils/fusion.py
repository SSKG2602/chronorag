"""Scoring utilities that combine rank, time, and authority signals."""

from __future__ import annotations

from typing import Dict


def monotone_temporal_fusion(
    r: float,
    t: float,
    a: float,
    tx_mismatch: float,
    age_penalty: float,
    weights: Dict[str, float],
) -> float:
    """Blend retrieval signals into a single monotonic score in [0, 1]."""
    alpha = max(0.0, min(1.0, weights.get("alpha", 0.55)))
    beta_time = max(0.0, min(1.0, weights.get("beta_time", 0.25)))
    gamma_auth = max(0.0, min(1.0, weights.get("gamma_authority", 0.15)))
    delta_age = max(0.0, weights.get("delta_age", 0.05))
    tx_gamma = max(0.0, weights.get("tx_gamma", 0.40))

    base = (alpha * max(0.0, min(1.0, r))) + (beta_time * max(0.0, min(1.0, t))) + (gamma_auth * max(0.0, min(1.0, a)))
    penalty = (delta_age * max(0.0, min(1.0, age_penalty))) + (tx_gamma * max(0.0, min(1.0, tx_mismatch)))
    combined = max(0.0, base - penalty)
    return min(1.0, combined)
