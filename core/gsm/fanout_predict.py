from __future__ import annotations

from typing import Dict


def predict_fanout(signals: Dict[str, float]) -> Dict[str, int]:
    base = signals.get("coverage", 0.0)
    n_hops = 1 if base > 0.6 else 2
    return {"predicted_hops": n_hops, "max_candidates": 24}
