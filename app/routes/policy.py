from __future__ import annotations

from fastapi import APIRouter, Header

from app.services.policy_service import get_policy
from app.services.policy_service import apply_policy
from app.schemas.policy import PolicyApplyRequest, PolicyApplyResponse

router = APIRouter()


@router.get("/policy")
def policy_endpoint() -> dict:
    return get_policy()


@router.post("/policy/apply", response_model=PolicyApplyResponse)
def policy_apply(
    payload: PolicyApplyRequest,
    authorization: str | None = Header(default=None),
) -> PolicyApplyResponse:
    return apply_policy(payload, authorization=authorization)
