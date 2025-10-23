"""Persistent vector database used for ChronoRAG ingestion and retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import datetime as dt

from app.utils.time_windows import (
    TimeWindow,
    hard_mode_pre_mask,
    intelligent_decay,
    make_window,
)
from core.retrieval.vector_ann import InMemoryANNIndex


@dataclass
class DocumentRecord:
    """Lightweight container for document-level metadata."""

    doc_id: str
    metadata: Dict = field(default_factory=dict)
    chunk_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "metadata": self.metadata,
            "chunk_ids": list(self.chunk_ids),
        }

    @staticmethod
    def from_dict(payload: Dict) -> "DocumentRecord":
        return DocumentRecord(
            doc_id=payload.get("doc_id", ""),
            metadata=payload.get("metadata", {}) or {},
            chunk_ids=list(payload.get("chunk_ids", []) or []),
        )


@dataclass
class ChunkRecord:
    """Materialised chunk entry persisted to disk and served at query time."""

    chunk_id: str
    doc_id: str
    text: str
    uri: str
    valid_window: TimeWindow
    tx_window: Optional[TimeWindow]
    authority: float
    metadata: Dict
    facets: Dict[str, str]
    entities: List[str]
    tags: List[str]
    units: List[str]
    time_granularity: Optional[str]
    time_sigma_days: Optional[int]
    external_id: Optional[str]
    version_id: Optional[str]

    def to_dict(self) -> Dict:
        payload = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "uri": self.uri,
            "valid_window": {
                "from": self.valid_window.start.isoformat(),
                "to": self.valid_window.end.isoformat(),
            },
            "tx_window": None,
            "authority": self.authority,
            "metadata": self.metadata,
            "facets": self.facets,
            "entities": self.entities,
            "tags": self.tags,
            "units": self.units,
            "time_granularity": self.time_granularity,
            "time_sigma_days": self.time_sigma_days,
            "external_id": self.external_id,
            "version_id": self.version_id,
        }
        if self.tx_window is not None:
            payload["tx_window"] = {
                "from": self.tx_window.start.isoformat(),
                "to": self.tx_window.end.isoformat(),
            }
        return payload

    @staticmethod
    def from_dict(payload: Dict) -> "ChunkRecord":
        valid = payload.get("valid_window", {}) or {}
        start_iso = valid.get("from") or dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc).isoformat()
        end_iso = valid.get("to") or dt.datetime(9999, 12, 31, tzinfo=dt.timezone.utc).isoformat()
        tx_payload = payload.get("tx_window")
        valid_window = make_window(
            dt.datetime.fromisoformat(start_iso),
            dt.datetime.fromisoformat(end_iso),
        )
        tx_window = None
        if isinstance(tx_payload, dict) and tx_payload.get("from") and tx_payload.get("to"):
            tx_window = make_window(
                dt.datetime.fromisoformat(tx_payload.get("from")),
                dt.datetime.fromisoformat(tx_payload.get("to")),
            )
        return ChunkRecord(
            chunk_id=payload.get("chunk_id"),
            doc_id=payload.get("doc_id"),
            text=payload.get("text", ""),
            uri=payload.get("uri", ""),
            valid_window=valid_window,
            tx_window=tx_window,
            authority=float(payload.get("authority", 0.0)),
            metadata=payload.get("metadata", {}) or {},
            facets=payload.get("facets", {}) or {},
            entities=list(payload.get("entities", []) or []),
            tags=list(payload.get("tags", []) or []),
            units=list(payload.get("units", []) or []),
            time_granularity=payload.get("time_granularity"),
            time_sigma_days=payload.get("time_sigma_days"),
            external_id=payload.get("external_id"),
            version_id=payload.get("version_id"),
        )


class PVDB:
    """Small persistent vector database backed by JSON state on disk."""

    def __init__(self, models_cfg: Dict, persist_path: Path):
        self.models_cfg = models_cfg or {}
        self.persist_path = Path(persist_path) if persist_path else None
        embeddings_cfg = self.models_cfg.get("embeddings", {}) or {}
        model_name = embeddings_cfg.get("name", "bge-base-en-v1.5")
        self.ann_index = InMemoryANNIndex(model_name)
        self.chunks: Dict[str, ChunkRecord] = {}
        self.documents: Dict[str, DocumentRecord] = {}
        self._dirty = False
        self._counter = 1
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.persist_path or not self.persist_path.exists():
            return
        data = json.loads(self.persist_path.read_text(encoding="utf-8"))
        for chunk_payload in data.get("chunks", []):
            chunk = ChunkRecord.from_dict(chunk_payload)
            self.chunks[chunk.chunk_id] = chunk
            self._bump_counter(chunk.chunk_id)
        for doc_payload in data.get("documents", []):
            document = DocumentRecord.from_dict(doc_payload)
            if document.doc_id:
                self.documents[document.doc_id] = document
        if self.chunks:
            self.ann_index.bulk_add(
                (
                    chunk.chunk_id,
                    chunk.text,
                    {"doc_id": chunk.doc_id, "authority": chunk.authority},
                )
                for chunk in self.chunks.values()
            )
        self._dirty = False

    def flush(self) -> None:
        if not self.persist_path or not self._dirty:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [chunk.to_dict() for chunk in sorted(self.chunks.values(), key=lambda item: item.chunk_id)],
            "documents": [doc.to_dict() for doc in sorted(self.documents.values(), key=lambda item: item.doc_id)],
        }
        self.persist_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._dirty = False

    def clear(self) -> None:
        self.chunks.clear()
        self.documents.clear()
        self.ann_index = InMemoryANNIndex(
            self.models_cfg.get("embeddings", {}).get("name", "bge-base-en-v1.5")
        )
        self._counter = 1
        self._dirty = False
        if self.persist_path and self.persist_path.exists():
            self.persist_path.unlink()

    # ------------------------------------------------------------------
    # Chunk/document operations
    # ------------------------------------------------------------------
    def ingest_document(
        self,
        *,
        text: str,
        uri: str,
        valid_window: TimeWindow,
        tx_window: Optional[TimeWindow],
        authority: float,
        metadata: Dict,
        doc_id: Optional[str],
        external_id: Optional[str],
        version_id: Optional[str],
        facets: Dict[str, str],
        entities: Iterable[str],
        tags: Optional[Iterable[str]],
        units: Iterable[str],
        time_granularity: Optional[str],
        time_sigma_days: Optional[int],
    ) -> ChunkRecord:
        doc_id = doc_id or self._default_doc_id(uri)
        document = self.documents.setdefault(doc_id, DocumentRecord(doc_id=doc_id))
        chunk_id = self._next_chunk_id(doc_id)
        chunk = ChunkRecord(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            uri=uri,
            valid_window=valid_window,
            tx_window=tx_window,
            authority=float(authority),
            metadata=metadata or {},
            facets=facets or {},
            entities=sorted(set(entities or [])),
            tags=sorted(set(tags or [])),
            units=list(units or []),
            time_granularity=time_granularity,
            time_sigma_days=time_sigma_days,
            external_id=external_id,
            version_id=version_id,
        )
        self.chunks[chunk_id] = chunk
        document.chunk_ids.append(chunk_id)
        self.ann_index.add(
            chunk_id,
            text,
            {"doc_id": doc_id, "authority": chunk.authority},
        )
        self._dirty = True
        return chunk

    def upsert_document_metadata(self, doc_id: str, updates: Dict) -> None:
        if not doc_id:
            return
        record = self.documents.setdefault(doc_id, DocumentRecord(doc_id=doc_id))
        record.metadata.update(updates or {})
        self._dirty = True

    def list_chunks(self) -> List[ChunkRecord]:
        return sorted(self.chunks.values(), key=lambda chunk: chunk.valid_window.start)

    def ann_search(self, query: str, top_k: int = 5) -> List[Tuple[ChunkRecord, float]]:
        results: List[Tuple[ChunkRecord, float]] = []
        for chunk_id, score, _meta in self.ann_index.search(query, top_k=top_k):
            chunk = self.chunks.get(chunk_id)
            if chunk is not None:
                results.append((chunk, float(score)))
        return results

    def temporal_filter(
        self,
        candidates: Iterable[ChunkRecord],
        window: TimeWindow,
        *,
        mode: str,
    ) -> List[Tuple[ChunkRecord, float]]:
        filtered: List[Tuple[ChunkRecord, float]] = []
        for chunk in candidates:
            candidate_window = chunk.valid_window
            if mode == "HARD":
                if not hard_mode_pre_mask(candidate_window, window):
                    continue
                weight = 1.0
            else:
                weight = intelligent_decay(candidate_window, window)
                if weight <= 0.0:
                    continue
            filtered.append((chunk, weight))
        filtered.sort(key=lambda item: item[1], reverse=True)
        return filtered

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _default_doc_id(self, uri: str) -> str:
        return f"doc:{abs(hash(uri))}"

    def _next_chunk_id(self, doc_id: str) -> str:
        chunk_id = f"{doc_id}:chunk:{self._counter:05d}"
        self._counter += 1
        return chunk_id

    def _bump_counter(self, chunk_id: str) -> None:
        if not chunk_id:
            return
        try:
            suffix = int(chunk_id.split(":")[-1])
        except (ValueError, AttributeError):
            return
        self._counter = max(self._counter, suffix + 1)
