from __future__ import annotations

from app.deps import get_app_state
from app.services.answer_service import answer
from app.services.ingest_service import ingest

SAMPLES = [
    "data/sample/docs/aihistory1.txt",
    "data/sample/docs/aihistory2.txt",
    "data/sample/docs/aihistory3.txt",
]


def setup_module(_module):
    state = get_app_state()
    state.pvdb.clear()
    ingest(SAMPLES, [], provenance=None)


def test_answer_returns_world_economy_timeline():
    response = answer("What was Europe's GDP per capita in 1870?", None, "", "")
    assert response["answer"] is not None
    card = response["attribution_card"]
    assert card["sources"], "Card should include sources"
    assert "temporal_confidence" in card
    stats = response["controller_stats"]
    assert stats["hops_used"] >= 1
    assert stats["rerank_method"] == "ce"
    assert stats["time_window_kind"] in {"decade", "year"}
    weights = stats["retrieval_weights"]
    assert weights["delta_age"] == 0.0
    router_metrics = stats.get("router_metrics", {})
    assert router_metrics.get("time_window_kind") in {"decade", "year"}
