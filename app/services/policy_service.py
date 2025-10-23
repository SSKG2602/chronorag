from __future__ import annotations

from __future__ import annotations

import time
from typing import Optional

from fastapi import HTTPException, status

from app.deps import get_app_state, get_policy_cfg, set_policy_cfg
from app.schemas.policy import PolicyApplyRequest, PolicyApplyResponse

ADMIN_TOKEN = "Bearer chronorag-admin"


def get_policy() -> dict:
    state = get_app_state()
    payload = get_policy_cfg().copy()
    payload["policy_version"] = state.policy_version
    return payload


def apply_policy(payload: PolicyApplyRequest, authorization: Optional[str]) -> PolicyApplyResponse:
    if authorization != ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="POLICY_APPLY_FORBIDDEN",
        )
    state = get_app_state()
    previous_version = state.policy_version
    if payload.policy_version == previous_version and payload.idempotency_key:
        return PolicyApplyResponse(
            policy_version=state.policy_version,
            previous_version=previous_version,
            accepted=False,
        )

    set_policy_cfg(payload.changes, payload.policy_version, payload.idempotency_key)
    get_policy_cfg()["last_applied_at"] = time.time()
    return PolicyApplyResponse(
        policy_version=payload.policy_version,
        previous_version=previous_version,
        accepted=True,
    )
