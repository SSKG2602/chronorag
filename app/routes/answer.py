from __future__ import annotations

from fastapi import APIRouter

from app.schemas.answer import AnswerRequest, AnswerResponse
from app.services.answer_service import answer

router = APIRouter()


@router.post("/answer", response_model=AnswerResponse)
def answer_endpoint(payload: AnswerRequest) -> AnswerResponse:
    response = answer(
        query=payload.query,
        time_hint=payload.time_hint,
        requested_mode=payload.time_mode.value,
        requested_axis=payload.time_axis.value,
    )
    return AnswerResponse(**response)
