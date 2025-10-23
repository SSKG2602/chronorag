from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class IngestRequest(BaseModel):
    paths: List[str] = Field(default_factory=list)
    text_blobs: List[str] = Field(default_factory=list)
    provenance: Optional[str] = None

    @model_validator(mode="after")  # type: ignore
    def ensure_payload(cls, values: "IngestRequest") -> "IngestRequest":
        if not values.paths and not values.text_blobs:
            raise ValueError(
                "Provide at least one file path or text blob to ingest")
        return values


class IngestResponse(BaseModel):
    ingested: int
    documents: List[str]
