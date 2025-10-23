from __future__ import annotations

from core.router.temporal_router import TemporalRouter


def build_router():
    policy = {
        "policy_sets": {
            "generic": {
                "time_axis_default": "valid",
                "time_mode_default": "INTELLIGENT",
                "hard_mode_for": [],
                "retrieval_weights": {"alpha": 0.55, "beta_time": 0.25, "gamma_authority": 0.15, "delta_age": 0.05},
            },
            "roles": {
                "time_axis_default": "valid",
                "time_mode_default": "INTELLIGENT",
                "hard_mode_for": ["explicit_year"],
                "retrieval_weights": {"alpha": 0.45, "beta_time": 0.35, "gamma_authority": 0.20, "delta_age": 0.10},
            },
            "world-economy": {
                "time_axis_default": "valid",
                "time_mode_default": "INTELLIGENT",
                "hard_mode_for": ["explicit_year", "explicit_period"],
                "retrieval_weights": {"alpha": 0.50, "beta_time": 0.35, "gamma_authority": 0.15, "delta_age": 0.00},
            },
        },
    }
    axis = {
        "snap_rules": {"INTELLIGENT_to_HARD": {"contradiction": 0.5, "low_confidence": 0.35}},
        "time_window_defaults": {"decade_padding_years": 5, "century_padding_years": 50},
        "fuzzy_period_map": {"post-war": {"from": "1945-01-01", "to": "1960-12-31"}},
    }
    tenant = {"fy_start": "APRIL", "locale": "IN"}
    return TemporalRouter(policy, axis, tenant)


def test_router_defaults_to_intelligent():
    router = build_router()
    decision = router.route("What is the revenue guidance?", None, signals=None)
    assert decision.mode == "INTELLIGENT"
    assert decision.axis in {"valid", "transaction"}


def test_router_snaps_to_hard_on_contradiction_signal():
    router = build_router()
    decision = router.route("Who is the CEO today?", None, signals={"contradiction": 0.6})
    assert decision.mode == "HARD"


def test_router_handles_world_economy_year():
    router = build_router()
    decision = router.route("GDP 1870 Europe", None, signals=None)
    assert decision.domain == "world-economy"
    assert decision.mode == "HARD"
    assert decision.window_kind in {"decade", "year"}
