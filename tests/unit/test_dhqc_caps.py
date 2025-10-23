from __future__ import annotations

from core.dhqc.controller import DHQCController
from core.dhqc.signals import RetrievalSignals


def test_dhqc_caps_and_marginal_gain():
    controller = DHQCController({"tau": 0.8, "delta": 0.2, "n_max": 3, "n_hard": 6, "fanout_cap_total": 250000})
    signals = RetrievalSignals(coverage=0.3)
    plan = controller.plan("INTELLIGENT", signals)
    assert plan.hops >= 1
    assert plan.max_candidates <= 250000


def test_dhqc_low_gain_reduces_hops():
    controller = DHQCController({"tau": 0.8, "delta": 0.2, "n_max": 3, "n_hard": 6, "fanout_cap_total": 250000})
    signals = RetrievalSignals(coverage=0.79)
    plan = controller.plan("INTELLIGENT", signals)
    assert plan.hops <= 2
