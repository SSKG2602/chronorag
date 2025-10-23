from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class PolicyApplyRequest(BaseModel):
    policy_version: str = Field(..., description="Semantic version for this policy bundle")
    changes: Dict[str, Optional[dict]] = Field(default_factory=dict)
    idempotency_key: Optional[str] = Field(None, description="Optional key to deduplicate PATCH requests")


class PolicyApplyResponse(BaseModel):
    policy_version: str
    accepted: bool
    previous_version: str
