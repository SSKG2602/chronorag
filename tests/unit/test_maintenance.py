from __future__ import annotations

from app.deps import get_cache, get_pvdb
from app.services.ingest_service import ingest
from app.services.maintenance_service import purge_system


def test_purge_system_clears_pvdb_and_cache(tmp_path):
    pvdb = get_pvdb()
    pvdb.clear()
    cache = get_cache()
    cache.clear()

    ingest(["data/sample/docs/aihistory1.txt"], [], provenance=None)
    assert pvdb.list_chunks(), "Expected chunks to exist after ingest"
    cache.set("freshness:test", {"epoch": 1})
    assert cache.get("freshness:test") is not None

    status = purge_system()
    assert status["pvdb"] == "cleared"
    assert status["cache"] == "cleared"
    assert pvdb.list_chunks() == []
    assert cache.get("freshness:test") is None
