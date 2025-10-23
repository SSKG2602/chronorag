"""Utility for collapsing metadata into a canonical valid TimeWindow."""

from __future__ import annotations

from typing import Dict

from app.utils.time_windows import TimeWindow, make_window, parse_date


def reduce_timeline(metadata: Dict[str, str]) -> TimeWindow:
    """Return a TimeWindow derived from optional valid_from/valid_to fields."""
    start = parse_date(metadata.get("valid_from", "1970-01-01"))
    end = parse_date(metadata.get("valid_to", "9999-12-31"))
    return make_window(start, end)
