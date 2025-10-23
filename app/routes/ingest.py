from __future__ import annotations

from fastapi import APIRouter

from app.schemas.ingest import IngestRequest, IngestResponse
from app.services.ingest_service import ingest

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(payload: IngestRequest) -> IngestResponse:
    ingested = ingest(payload.paths, payload.text_blobs, payload.provenance)
    return IngestResponse(ingested=len(ingested), documents=ingested)
