"""Normalise optional time hints into canonical TimeWindow objects."""

from __future__ import annotations

import datetime as dt
from typing import Dict

from app.utils.time_windows import TimeWindow, make_window, parse_date


def normalize_time_hint(time_hint: Dict | None) -> TimeWindow:
    """Convert API-style time hints into a concrete TimeWindow."""
    if not time_hint:
        return make_window(parse_date("1970-01-01"), parse_date("9999-12-31"))
    operator = time_hint.get("operator", "AS_OF").upper()
    if operator == "AS_OF":
        at = parse_date(time_hint.get("at", "1970-01-01"))
        return make_window(at, at + dt.timedelta(days=1))
    frm = parse_date(time_hint.get("from", "1970-01-01"))
    to = parse_date(time_hint.get("to", "9999-12-31"))
    return make_window(frm, to)
