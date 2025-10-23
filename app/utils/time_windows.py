"""Chronological utilities for parsing dates and working with time windows.

The ingestion and retrieval layers frequently receive partially specified or noisy
temporal hints.  This module provides a small set of helpers that translate that
input into deterministic, timezone-aware datetime objects and rich TimeWindow
instances.  Every function is defensive: instead of raising when a value cannot be
interpreted, it falls back to reasonable defaults so that the pipeline maintains a
stable chronology.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Optional, Tuple

# Many historical sources lack explicit timestamps.  When we cannot recover any
# signal we position events at the Unix epoch (1970-01-01 UTC).  This keeps
# downstream comparisons predictable while still flagging "unknown" dates.
ISO_FALLBACK = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)


def _ensure_timezone(value: dt.datetime) -> dt.datetime:
    """Normalise a datetime object so that it is explicitly expressed in UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _parse_month_name(text: str) -> Optional[dt.datetime]:
    """Return a datetime parsed from strings like 'Jan 2024' or 'January 17, 2021'."""
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %Y", "%b %Y"):
        try:
            parsed = dt.datetime.strptime(text, fmt)
            if "%d" not in fmt:
                parsed = parsed.replace(day=1)
            return parsed.replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
    return None


def parse_date(text: str) -> dt.datetime:
    """Extract a timezone-aware datetime from messy text, defaulting to epoch."""
    text = text.strip()
    iso_match = re.search(r"(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}:\d{2}(?::\d{2})?)Z?)?", text)
    if iso_match:
        raw = iso_match.group(1)
        time_part = iso_match.group(2)
        dt_str = raw if not time_part else f"{raw}T{time_part}"
        try:
            parsed = dt.datetime.fromisoformat(dt_str)
            return _ensure_timezone(parsed)
        except ValueError:
            pass

    month = _parse_month_name(text)
    if month:
        return _ensure_timezone(month)

    return ISO_FALLBACK


@dataclass(frozen=True)
class TimeWindow:
    """Closed-open interval representing a valid period for a snippet of evidence."""

    start: dt.datetime
    end: dt.datetime

    @property
    def duration(self) -> float:
        """Return the window length in seconds, clamped at zero for inverted ranges."""
        return max(0.0, (self.end - self.start).total_seconds())

    def intersects(self, other: "TimeWindow") -> bool:
        """Return True when two windows overlap in time."""
        return not (self.end <= other.start or other.end <= self.start)

    def intersection(self, other: "TimeWindow") -> Optional["TimeWindow"]:
        """Return the overlapping segment between windows, or None when disjoint."""
        if not self.intersects(other):
            return None
        return TimeWindow(start=max(self.start, other.start), end=min(self.end, other.end))


def make_window(start: dt.datetime, end: Optional[dt.datetime] = None) -> TimeWindow:
    """Build a well-ordered, UTC-aligned time window from potentially messy inputs."""
    end = end or dt.datetime(9999, 12, 31, tzinfo=dt.timezone.utc)
    start = _ensure_timezone(start)
    end = _ensure_timezone(end)
    if start > end:
        start, end = end, start
    return TimeWindow(start=start, end=end)


def window_iou(left: TimeWindow, right: TimeWindow) -> float:
    """Compute an IoU-style overlap score in [0, 1] for two windows."""
    inter = left.intersection(right)
    if inter is None:
        return 0.0
    union = left.duration + right.duration - inter.duration
    if union <= 0:
        return 0.0
    return inter.duration / union


def tx_mismatch_penalty(valid_window: TimeWindow, tx_window: Optional[TimeWindow]) -> float:
    """Return a 0/1 penalty when the transactional window falls outside the valid window."""
    if tx_window is None:
        return 0.0
    return 0.0 if valid_window.intersects(tx_window) else 1.0


def expand_window(window: TimeWindow, seconds: int) -> TimeWindow:
    """Return a window padded symmetrically by the supplied number of seconds."""
    return TimeWindow(
        start=window.start - dt.timedelta(seconds=seconds),
        end=window.end + dt.timedelta(seconds=seconds),
    )


def hard_mode_pre_mask(candidate_window: TimeWindow, query_window: TimeWindow) -> bool:
    """Return True when the candidate survives the HARD mode temporal mask."""
    return candidate_window.intersects(query_window)


def intelligent_decay(candidate_window: TimeWindow, query_window: TimeWindow) -> float:
    """Return a decay weight in [0, 1] based on the gap between candidate and query."""
    if candidate_window.intersects(query_window):
        return 1.0
    gap = min(
        abs((candidate_window.start - query_window.end).total_seconds()),
        abs((query_window.start - candidate_window.end).total_seconds()),
    )
    days = gap / 86400.0
    return max(0.0, 1.0 / (1.0 + days))
