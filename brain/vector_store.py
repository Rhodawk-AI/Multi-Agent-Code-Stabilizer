"""
brain/vector_store.py — VectorBrain with Qdrant + ChromaDB backends.
Exposes is_available, index_chunk(), find_similar_to_issue().
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    chunk_id:   str   = ""
    file_path:  str   = ""
    line_start: int   = 0
    line_end:   int   = 0
    language:   str   = ""
    summary:    str   = ""
    distance:   float = 1.0


class VectorBrain:
    def __init__(
        self,
        store_path: str = "",
        qdrant_url: str = "http://localhost:6333",
        collection: str = "rhodawk_chunks",
    ) -> None:
        self.store_path = store_path
        self.qdrant_url = qdrant_url
        self.collection = collection
        self._client: Any = None
        self._backend = "none"

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def initialise(self) -> None:
        # Try Qdrant first
        try:
            from qdrant_client import QdrantClient  # type: ignore
            self._client  = QdrantClient(url=self.qdrant_url, timeout=5)
            self._client.get_collections()
            self._backend = "qdrant"
            log.info(f"VectorBrain: Qdrant connected at {self.qdrant_url}")
            return
        except Exception:
            pass
        # Try ChromaDB
        try:
            import chromadb  # type: ignore
            path = self.store_path or "/tmp/rhodawk_vectors"
            self._client  = chromadb.PersistentClient(path=path)
            self._backend = "chroma"
            log.info(f"VectorBrain: ChromaDB at {path}")
            return
        except Exception:
            pass
        log.info("VectorBrain: no vector backend available — semantic search disabled")

    def index_chunk(
        self, chunk_id: str, file_path: str, line_start: int, line_end: int,
        language: str, summary: str, observations: list[str],
    ) -> None:
        if not self._client:
            return
        text = f"{file_path} {summary} {' '.join(observations[:5])}"
        try:
            if self._backend == "qdrant":
                self._qdrant_upsert(chunk_id, file_path, line_start, line_end,
                                    language, summary, text)
            elif self._backend == "chroma":
                self._chroma_upsert(chunk_id, file_path, line_start, line_end,
                                    language, summary, text)
        except Exception as exc:
            log.debug(f"VectorBrain.index_chunk failed: {exc}")

    def find_similar_to_issue(
        self, query: str, n: int = 8
    ) -> list[VectorSearchResult]:
        if not self._client:
            return []
        try:
            if self._backend == "qdrant":
                return self._qdrant_search(query, n)
            elif self._backend == "chroma":
                return self._chroma_search(query, n)
        except Exception as exc:
            log.debug(f"VectorBrain.find_similar_to_issue failed: {exc}")
        return []

    def _embed(self, text: str) -> list[float]:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            if not hasattr(self, "_model"):
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            return self._model.encode(text).tolist()
        except Exception:
            # Fallback: deterministic hash embedding (very poor quality but won't crash)
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            return [b / 255.0 for b in h[:16]] * 24  # 384 dims

    def _qdrant_upsert(self, chunk_id, file_path, line_start, line_end,
                       language, summary, text) -> None:
        from qdrant_client.models import PointStruct  # type: ignore
        vec = self._embed(text)
        self._client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=abs(hash(chunk_id)) % (10**9),
                vector=vec,
                payload={
                    "chunk_id": chunk_id, "file_path": file_path,
                    "line_start": line_start, "line_end": line_end,
                    "language": language, "summary": summary,
                },
            )],
        )

    def _qdrant_search(self, query: str, n: int) -> list[VectorSearchResult]:
        vec = self._embed(query)
        hits = self._client.search(
            collection_name=self.collection, query_vector=vec, limit=n
        )
        return [
            VectorSearchResult(
                chunk_id=str(h.payload.get("chunk_id","")),
                file_path=h.payload.get("file_path",""),
                line_start=h.payload.get("line_start",0),
                line_end=h.payload.get("line_end",0),
                language=h.payload.get("language",""),
                summary=h.payload.get("summary",""),
                distance=1.0 - h.score,
            )
            for h in hits
        ]

    def _chroma_upsert(self, chunk_id, file_path, line_start, line_end,
                       language, summary, text) -> None:
        coll = self._client.get_or_create_collection(self.collection)
        coll.upsert(
            ids=[chunk_id],
            documents=[text],
            metadatas=[{"file_path": file_path, "line_start": line_start,
                        "line_end": line_end, "language": language,
                        "summary": summary}],
        )

    def _chroma_search(self, query: str, n: int) -> list[VectorSearchResult]:
        coll = self._client.get_or_create_collection(self.collection)
        results = coll.query(query_texts=[query], n_results=n)
        out = []
        for i, (doc_id, meta, dist) in enumerate(zip(
            results["ids"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            out.append(VectorSearchResult(
                chunk_id=doc_id,
                file_path=meta.get("file_path",""),
                line_start=meta.get("line_start",0),
                line_end=meta.get("line_end",0),
                language=meta.get("language",""),
                summary=meta.get("summary",""),
                distance=float(dist),
            ))
        return out

    def close(self) -> None:
        self._client = None
