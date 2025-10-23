from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.common import TimeAxis, TimeMode


class SourcePayload(BaseModel):
    uri: str
    quote: str
    interval: Dict[str, str]
    score: float


class TemporalConfidence(BaseModel):
    level: str
    reasons: List[str]
    alternative_windows: List[str]


class AttributionCard(BaseModel):
    mode: TimeMode
    time_axis: TimeAxis
    window: Dict[str, str]
    sources: List[SourcePayload]
    temporal_confidence: TemporalConfidence
    counterfactuals: List[str]


class ControllerStats(BaseModel):
    hops_used: int
    signals: Dict[str, float]
    latency_ms: int
    cost_usd: float
    tokens_in: int
    tokens_out: int
    degraded: Optional[str]
    rerank_method: str


class AnswerRequest(BaseModel):
    query: str
    time_mode: TimeMode = TimeMode.INTELLIGENT
    time_axis: TimeAxis = TimeAxis.valid
    time_hint: Optional[dict] = None
    constraints: Dict[str, int] = Field(default_factory=dict)
    retrieval: Dict[str, float] = Field(default_factory=dict)
    audit_mode: bool = False


class AnswerResponse(BaseModel):
    answer: str
    attribution_card: AttributionCard
    controller_stats: ControllerStats
    audit_trail: List[Dict]
    evidence_only: bool = False
    reason: Optional[str] = None
