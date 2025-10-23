from __future__ import annotations

from fastapi import APIRouter

from app.services.incident_service import report_incident

router = APIRouter()


@router.post("/incident")
def incident_endpoint(payload: dict) -> dict:
    return report_incident(payload)
