"""Maintenance helpers for clearing caches and persistent PVDB state."""

from __future__ import annotations

from typing import Dict

from app.deps import get_app_state, get_cache, get_pvdb


def purge_system() -> Dict[str, str]:
    """Clear PVDB chunks, ANN entries, cache contents, and policy idempotency state."""
    state = get_app_state()
    pvdb = get_pvdb()
    pvdb.clear()

    cache = get_cache()
    cache.clear()

    # Reset policy state to avoid stale idempotency keys interfering with new runs.
    state.policy_applied_keys.clear()

    return {"status": "ok", "pvdb": "cleared", "cache": "cleared"}
