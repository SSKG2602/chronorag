from __future__ import annotations

import os


def light_mode_enabled() -> bool:
    """
    Return True when ChronoRAG should avoid heavy model initialization.

    Defaults to True whenever tests are running (detected via PYTEST_CURRENT_TEST)
    unless explicitly disabled via CHRONORAG_LIGHT=0/false/no.
    """
    val = os.getenv("CHRONORAG_LIGHT", "").strip().lower()
    if val in ("0", "false", "no"):
        return False
    if val in ("1", "true", "yes"):
        return True
    return os.getenv("PYTEST_CURRENT_TEST") is not None
