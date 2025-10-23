"""Domain Heuristic Query Controller (DHQC) planning logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from core.dhqc.signals import RetrievalSignals


@dataclass
class DHQCPlan:
    """Simple data container describing hop count and candidate budget."""

    hops: int
    max_candidates: int
    reason: str


class DHQCController:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.tau = cfg.get("tau", 0.8)
        self.delta = cfg.get("delta", 0.2)
        self.n_max = cfg.get("n_max", 3)
        self.n_hard = cfg.get("n_hard", 6)
        self.fanout_cap_total = cfg.get("fanout_cap_total", 250000)

    def _marginal_gain(self, prev: float, current: float) -> float:
        """Return the marginal gain so we can shrink hops when coverage stalls."""
        if prev <= 0:
            return current
        return max(0.0, (current - prev) / prev)

    def plan(self, mode: str, signals: RetrievalSignals) -> DHQCPlan:
        """Decide how many retrieval hops to run given coverage and mode."""
        coverage = signals.coverage
        hops = 1
        reason = "baseline"
        if coverage < self.tau:
            hops = min(self.n_hard if mode == "HARD" else self.n_max, 3)
            reason = "low_coverage"
        if self._marginal_gain(self.tau, coverage) < self.delta:
            hops = max(1, hops - 1)
            reason = "marginal_gain_low"
        max_candidates = min(self.fanout_cap_total, 24 if hops > 1 else 12)
        return DHQCPlan(hops=hops, max_candidates=max_candidates, reason=reason)
