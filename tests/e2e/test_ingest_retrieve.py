from __future__ import annotations

from app.deps import get_app_state
from app.services.ingest_service import ingest
from app.services.retrieve_service import retrieve
from app.utils.time_windows import make_window, parse_date

SAMPLES = [
    "data/sample/docs/aihistory1.txt",
    "data/sample/docs/aihistory2.txt",
    "data/sample/docs/aihistory3.txt",
]


def setup_module(_module):
    state = get_app_state()
    state.pvdb.clear()
    ingest(SAMPLES, [], provenance=None)


def test_retrieve_world_economy_units_and_facets():
    window = make_window(parse_date("1865-01-01"), parse_date("1875-01-01"))
    data = retrieve(
        "GDP 1870 Europe",
        window,
        mode="HARD",
        top_k=10,
        axis="valid",
        domain="world-economy",
    )
    assert data["results"], "Expected at least one result for historical GDP query"
    assert data["domain"] == "world-economy"
    for item in data["results"]:
        facets = item["facets"]
        assert facets["domain"] == "world-economy"
        assert facets["tenant"] == "lab"
        assert item["time_granularity"] == "year"
        assert item["time_sigma_days"] == 90
        assert "units_detected" in item and item["units_detected"], "Units should be detected"
    assert any(unit != "n/a" for item in data["results"] for unit in item["units_detected"])
