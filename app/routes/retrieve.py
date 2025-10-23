from __future__ import annotations

from fastapi import APIRouter

from app.deps import get_router
from app.schemas.retrieve import RetrieveRequest, RetrieveResponse
from app.services.retrieve_service import retrieve

router = APIRouter()


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(payload: RetrieveRequest) -> RetrieveResponse:
    decision = get_router().route(payload.query, payload.time_hint, signals=None)
    window = decision.window
    data = retrieve(
        payload.query,
        window,
        payload.time_mode.value,
        payload.top_k,
        axis=payload.time_axis.value,
    )
    return RetrieveResponse(query=payload.query, results=data["results"])
