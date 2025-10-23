"""Temporal routing logic that selects axis, mode, and window for each query."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.utils.time_windows import TimeWindow, make_window, parse_date
from core.gsm.intent import detect_intent
from core.gsm.tnormalize import normalize_time_hint
from core.router.rules import ROUTE_KEYWORDS

YEAR_PATTERN = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
CENTURY_PATTERN = re.compile(r"\b([0-9]{1,2})(st|nd|rd|th)\s+century\b", re.IGNORECASE)


@dataclass
class RouteDecision:
    """Describes the temporal routing outcome consumed by downstream services."""

    axis: str
    mode: str
    window: TimeWindow
    intent: Dict[str, str]
    domain: str
    window_kind: str


class TemporalRouter:
    """Resolve temporal axis/mode/window using lexical signals and policy defaults."""

    def __init__(self, policy_cfg: Dict, axis_cfg: Dict, tenant_cfg: Dict):
        self.policy_cfg = policy_cfg
        self.axis_cfg = axis_cfg
        self.tenant_cfg = tenant_cfg
        self.policy_sets = policy_cfg.get("policy_sets", {})
        self.default_policy = self.policy_sets.get("generic", {})
        self.snap_rules = axis_cfg.get("snap_rules", {}).get("INTELLIGENT_to_HARD", {})
        self.period_windows = self._build_period_windows(axis_cfg.get("fuzzy_period_map", {}))
        self.window_defaults = axis_cfg.get("time_window_defaults", {})
        self.observability: Dict[str, str] = {}

    def _build_period_windows(self, raw: Dict[str, Dict[str, str]]) -> Dict[str, TimeWindow]:
        """Compile maps like 'post-war' → 1945–1960 into ready-made windows."""
        windows: Dict[str, TimeWindow] = {}
        for token, bounds in raw.items():
            start = parse_date(bounds["from"])
            end = parse_date(bounds["to"])
            windows[token] = make_window(start, end)
        return windows

    def _policy_for(self, domain: str) -> Dict:
        """Return the policy set for the resolved domain, defaulting to generic."""
        return self.policy_sets.get(domain, self.default_policy)

    def _pick_axis(self, query: str, intent: Dict[str, str], policy: Dict) -> str:
        """Choose between valid/transaction axes using heuristics and policy."""
        q = query.lower()
        if any(token in q for token in ROUTE_KEYWORDS["transaction"]):
            return "transaction"
        if intent.get("domain") == "finance":
            return "transaction"
        return policy.get("time_axis_default", self.default_policy.get("time_axis_default", "valid"))

    def _pick_mode(
        self,
        intent: Dict[str, str],
        signals: Optional[Dict[str, float]],
        policy: Dict,
        window_kind: str,
    ) -> str:
        """Select HARD vs INTELLIGENT mode based on policies and live signals."""
        default_mode = policy.get("time_mode_default", self.default_policy.get("time_mode_default", "INTELLIGENT"))
        hard_modes = set(policy.get("hard_mode_for", []))
        if window_kind in {"year", "decade"} and "explicit_year" in hard_modes:
            return "HARD"
        if window_kind == "century" and "explicit_century" in hard_modes:
            return "HARD"
        if window_kind == "period" and "explicit_period" in hard_modes:
            return "HARD"

        if not signals:
            return default_mode
        contradiction = signals.get("contradiction", 0.0)
        low_conf = signals.get("low_confidence", 0.0)
        threshold_contra = self.snap_rules.get("contradiction", 0.5)
        threshold_low = self.snap_rules.get("low_confidence", 0.35)
        if contradiction >= threshold_contra or low_conf >= threshold_low:
            return "HARD"
        return default_mode

    def _detect_time_signals(self, query: str) -> Dict[str, List]:
        """Extract explicit year/century/period tokens from the raw query string."""
        years = sorted({int(match.group(1)) for match in YEAR_PATTERN.finditer(query)})
        centuries = sorted({int(match.group(1)) for match in CENTURY_PATTERN.finditer(query)})
        query_lower = query.lower()
        periods = [token for token in self.period_windows if token in query_lower]
        return {"years": years, "centuries": centuries, "periods": periods}

    def _build_year_window(self, years: List[int]) -> Optional[Tuple[TimeWindow, str]]:
        """Derive a window around explicit years, adding configurable padding."""
        if not years:
            return None
        pad = int(self.window_defaults.get("decade_padding_years", 5))
        if len(years) == 1:
            year = years[0]
            start = dt.datetime(max(1, year - pad), 1, 1, tzinfo=dt.timezone.utc)
            end = dt.datetime(year + pad + 1, 1, 1, tzinfo=dt.timezone.utc)
            return make_window(start, end), "decade"
        start = dt.datetime(years[0], 1, 1, tzinfo=dt.timezone.utc)
        end = dt.datetime(years[-1] + 1, 1, 1, tzinfo=dt.timezone.utc)
        return make_window(start, end), "year_range"

    def _build_century_window(self, centuries: List[int]) -> Optional[Tuple[TimeWindow, str]]:
        """Return a window covering the requested century with optional padding."""
        if not centuries:
            return None
        century = centuries[0]
        start_year = (century - 1) * 100 + 1
        end_year = century * 100
        pad = int(self.window_defaults.get("century_padding_years", 50))
        start = dt.datetime(max(1, start_year - pad), 1, 1, tzinfo=dt.timezone.utc)
        end = dt.datetime(end_year + pad + 1, 1, 1, tzinfo=dt.timezone.utc)
        return make_window(start, end), "century"

    def _build_period_window(self, periods: List[str]) -> Optional[Tuple[TimeWindow, str]]:
        """Look up fuzzy period labels such as 'post-war' or 'industrial revolution'."""
        for token in periods:
            window = self.period_windows.get(token)
            if window:
                return window, "period"
        return None

    def _infer_window(self, query: str, time_hint: Optional[Dict], policy: Dict) -> Tuple[TimeWindow, str]:
        """Combine user hints, query signals, and fallbacks to produce a window."""
        if time_hint:
            return normalize_time_hint(time_hint), "hint"

        signals = self._detect_time_signals(query)
        if signals["periods"]:
            period = self._build_period_window(signals["periods"])
            if period:
                return period
        if signals["centuries"]:
            century = self._build_century_window(signals["centuries"])
            if century:
                return century
        if signals["years"]:
            year_window = self._build_year_window(signals["years"])
            if year_window:
                return year_window

        start = parse_date("0001-01-01")
        end = parse_date("2100-01-01")
        return make_window(start, end), "broad"

    def route(self, query: str, time_hint: Optional[Dict], signals: Optional[Dict[str, float]] = None) -> RouteDecision:
        """Resolve the full RouteDecision used by retrieval and answer generation."""
        intent = detect_intent(query)
        domain = intent.get("domain", "generic")
        policy = self._policy_for(domain)
        window, kind = self._infer_window(query, time_hint, policy)
        axis = self._pick_axis(query, intent, policy)
        mode = self._pick_mode(intent, signals, policy, kind)
        self.observability = {"time_window_kind": kind, "domain": domain}
        return RouteDecision(axis=axis, mode=mode, window=window, intent=intent, domain=domain, window_kind=kind)
