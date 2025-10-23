from __future__ import annotations

from app.utils.fusion import monotone_temporal_fusion

DEFAULT_WEIGHTS = {
    "alpha": 0.55,
    "beta_time": 0.25,
    "gamma_authority": 0.15,
    "delta_age": 0.05,
}


def test_monotone_in_time_weight():
    r = 0.6
    a = 0.8
    mismatch = 0.0
    low = monotone_temporal_fusion(r, t=0.2, a=a, tx_mismatch=mismatch, age_penalty=0.0, weights=DEFAULT_WEIGHTS)
    high = monotone_temporal_fusion(r, t=0.8, a=a, tx_mismatch=mismatch, age_penalty=0.0, weights=DEFAULT_WEIGHTS)
    assert high >= low


def test_penalizes_mismatch_and_age():
    base = monotone_temporal_fusion(0.5, 1.0, 0.5, 0.0, age_penalty=0.0, weights=DEFAULT_WEIGHTS)
    penalized = monotone_temporal_fusion(0.5, 1.0, 0.5, 1.0, age_penalty=0.5, weights=DEFAULT_WEIGHTS)
    assert penalized < base
