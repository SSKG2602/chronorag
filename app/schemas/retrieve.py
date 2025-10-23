from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.common import TimeAxis, TimeMode


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    time_axis: TimeAxis = TimeAxis.valid
    time_mode: TimeMode = TimeMode.INTELLIGENT
    time_hint: Optional[dict] = None


class RetrieveResponse(BaseModel):
    query: str
    results: list
