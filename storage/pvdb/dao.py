"""Persistence layer for ChronoRAG's in-memory + disk-backed PVDB."""

from __future__ import annotations

import uuid
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.utils.time_windows import TimeWindow, hard_mode_pre_mask, intelligent_decay
from core.retrieval.vector_ann import InMemoryANNIndex
from .models import ChunkRecord, DocumentRecord


class PVDB:
    """Simplified document + chunk store with optional JSON persistence."""

    def __init__(self, model_cfg: Dict, persist_path: Optional[Path] = None):
        self.documents: Dict[str, DocumentRecord] = {}
        self.chunks: Dict[str, ChunkRecord] = {}
        self.ann = InMemoryANNIndex(model_cfg["embeddings"]["name"])
        self.external_index: Dict[str, str] = {}
        self._dirty: bool = False
        self.persist_path = persist_path
        if self.persist_path:
            self._load_from_disk()

    def clear(self) -> None:
        """Destroy all documents/chunks and reset the ANN index."""
        self.documents.clear()
        self.chunks.clear()
        self.ann.entries.clear()
        self.external_index.clear()
        self._dirty = True
        self._persist(force=True)

    def upsert_document_metadata(self, doc_id: str, updates: Dict[str, Any]) -> None:
        """Attach metadata to DocumentRecord objects without touching chunks."""
        document = self.documents.get(doc_id)
        if document is None:
            document = DocumentRecord(doc_id=doc_id, source_path=None, text="", metadata={})
            self.documents[doc_id] = document
        document.metadata.update(updates)
        self._dirty = True

    # ingest
    def ingest_document(
        self,
        text: str,
        uri: str,
        valid_window: TimeWindow,
        tx_window: Optional[TimeWindow],
        authority: float,
        metadata: Optional[Dict] = None,
        *,
        doc_id: Optional[str] = None,
        external_id: Optional[str] = None,
        version_id: Optional[str] = None,
        facets: Optional[Dict[str, str]] = None,
        entities: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        units: Optional[List[str]] = None,
        time_granularity: Optional[str] = None,
        time_sigma_days: Optional[int] = None,
    ) -> ChunkRecord:
        """Store a new chunk, assign ANN embedding, and maintain external-id lineage."""
        doc_key = doc_id or uuid.uuid4().hex
        document = self.documents.get(doc_key)
        if document is None:
            document = DocumentRecord(doc_id=doc_key, source_path=uri, text=text, metadata={})
            self.documents[doc_key] = document
        if metadata:
            document.metadata.update(metadata)
        self._dirty = True

        chunk_id = uuid.uuid4().hex
        vector = self.ann.add(
            chunk_id,
            text,
            {
                "doc_id": doc_key,
                "uri": uri,
                "domain": (facets or {}).get("domain"),
                "external_id": external_id,
            },
        )

        if external_id:
            previous_chunk_id = self.external_index.get(external_id)
            if previous_chunk_id:
                previous_chunk = self.chunks.get(previous_chunk_id)
                if previous_chunk:
                    if tx_window:
                        prev_tx_start = previous_chunk.tx_window.start if previous_chunk.tx_window else previous_chunk.valid_window.start
                        tx_end = tx_window.start
                        if tx_end <= prev_tx_start:
                            tx_end = prev_tx_start
                        previous_chunk.tx_window = TimeWindow(
                            start=prev_tx_start,
                            end=tx_end,
                        )
                    previous_chunk.version_id = previous_chunk.version_id or version_id

        payload = ChunkRecord(
            chunk_id=chunk_id,
            doc_id=doc_key,
            text=text,
            uri=uri,
            authority=authority,
            valid_window=valid_window,
            tx_window=tx_window,
            external_id=external_id,
            version_id=version_id,
            facets=dict(facets or {}),
            entities=list(entities or []),
            tags=list(tags or []),
            units=list(units or []),
            time_granularity=time_granularity,
            time_sigma_days=time_sigma_days,
            embedding=vector.tolist(),
            extra=metadata or {},
        )
        self.chunks[chunk_id] = payload
        if external_id:
            self.external_index[external_id] = chunk_id
        self._dirty = True
        return payload

    def list_chunks(self) -> List[ChunkRecord]:
        """Return the full set of chunk records (used sparingly in tests/CLI)."""
        return list(self.chunks.values())

    def chunks_by_ids(self, chunk_ids: Iterable[str]) -> List[ChunkRecord]:
        """Return chunk objects for a subset of ids, ignoring missing ones."""
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]

    def ann_search(self, query: str, top_k: int) -> List[Tuple[ChunkRecord, float]]:
        """Perform ANN search and convert results back into chunk instances."""
        results = self.ann.search(query, top_k=top_k)
        output = []
        for chunk_id, score, metadata in results:
            chunk = self.chunks.get(chunk_id)
            if chunk:
                output.append((chunk, score))
        return output

    def temporal_filter(
        self,
        chunks: Iterable[ChunkRecord],
        query_window: TimeWindow,
        mode: str = "INTELLIGENT",
    ) -> List[Tuple[ChunkRecord, float]]:
        """Filter candidate chunks by temporal overlap (HARD) or decay weighting."""
        filtered = []
        for chunk in chunks:
            if mode == "HARD":
                if hard_mode_pre_mask(chunk.valid_window, query_window):
                    filtered.append((chunk, 1.0))
            else:
                weight = intelligent_decay(chunk.valid_window, query_window)
                if weight > 0:
                    filtered.append((chunk, weight))
        return filtered

    # persistence helpers
    def flush(self, force: bool = False) -> None:
        """Persist current state to disk when dirty or when force=True."""
        self._persist(force=force)

    def _persist(self, force: bool = False) -> None:
        """Write a JSON snapshot of documents/chunks/external index."""
        if not self.persist_path:
            return
        if not self._dirty and not force:
            return
        snapshot = {
            "documents": [doc.to_dict() for doc in self.documents.values()],
            "chunks": [chunk.to_dict() for chunk in self.chunks.values()],
            "external_index": self.external_index,
        }
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.persist_path.write_text(json.dumps(snapshot), encoding="utf-8")
        self._dirty = False

    def _load_from_disk(self) -> None:
        """Initialise PVDB from an on-disk snapshot, rebuilding ANN vectors."""
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            payload = json.loads(self.persist_path.read_text(encoding="utf-8"))
        except Exception:
            return
        docs = {item["doc_id"]: DocumentRecord.from_dict(item) for item in payload.get("documents", [])}
        chunks = {}
        for item in payload.get("chunks", []):
            chunk = ChunkRecord.from_dict(item)
            chunks[chunk.chunk_id] = chunk
            vector = self.ann.add(
                chunk.chunk_id,
                chunk.text,
                {
                    "doc_id": chunk.doc_id,
                    "uri": chunk.uri,
                    "domain": chunk.facets.get("domain"),
                    "external_id": chunk.external_id,
                },
            )
            chunk.embedding = vector.tolist()
        self.documents = docs
        self.chunks = chunks
        self.external_index = payload.get("external_index", {})
        self._dirty = False
