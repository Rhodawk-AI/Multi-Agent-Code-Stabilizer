"""
memory/openviking.py
====================
Vector memory adapter for Rhodawk AI.

PREVIOUS PHANTOM: "OpenViking" (https://github.com/lalalune/open-viking)
is not a real installable Python package.  The canonical backend is
Qdrant via the HelixDB adapter.  The `import open_viking` attempt is a
forward-compatibility hook only — the HelixDB/Qdrant path is always the
production path.

This module provides a clean Python interface that:
1. Attempts native open_viking client if ever published (forward-compat)
2. Falls back to HelixDB (Qdrant-backed) — the real production backend
3. Falls back to in-memory store for testing
4. Provides the same API in all cases

Usage::

    db = OpenVikingDB()
    db.initialise()
    db.store("chunk_001", "agents/base.py", "LLM call wrapper", 0, 50)
    results = db.search("rate limiting retry logic", n=5)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

_OPENVIKING_URL = os.environ.get("OPENVIKING_URL", os.environ.get("QDRANT_URL", "http://localhost:6333"))
_COLLECTION     = os.environ.get("OPENVIKING_COLLECTION", "rhodawk_memory")


@dataclass
class VikingDocument:
    """A document stored in OpenViking."""
    id:         str
    file_path:  str
    summary:    str
    line_start: int   = 0
    line_end:   int   = 0
    language:   str   = "unknown"
    content:    str   = ""
    namespace:  str   = "default"


@dataclass
class VikingSearchResult:
    id:         str
    file_path:  str
    line_start: int
    line_end:   int
    language:   str
    summary:    str
    score:      float
    namespace:  str = "default"


class OpenVikingDB:
    """
    OpenViking-compatible vector database.

    Provides unified interface over native OpenViking or HelixDB fallback.
    """

    def __init__(
        self,
        url:         str  = _OPENVIKING_URL,
        collection:  str  = _COLLECTION,
        namespace:   str  = "default",
    ) -> None:
        self._url        = url
        self._collection = collection
        self._namespace  = namespace
        self._client: Any = None
        self._backend: str = "none"

    def initialise(self) -> None:
        """Connect to the best available backend."""
        # Try native OpenViking first
        try:
            import open_viking  # type: ignore[import]
            self._client = open_viking.VectorDB(
                url=self._url,
                collection=self._collection,
                namespace=self._namespace,
            )
            self._client.ensure_collection()
            self._backend = "openviking"
            log.info(f"OpenViking: native backend at {self._url}")
            return
        except (ImportError, Exception) as exc:
            log.debug(f"OpenViking native unavailable: {exc}")

        # Fall back to HelixDB (Qdrant)
        try:
            from memory.helixdb import HelixDB
            helix = HelixDB(url=self._url)
            helix.initialise()
            self._client = _HelixAdapter(helix, self._namespace)
            self._backend = "helixdb"
            log.info(f"OpenViking: using HelixDB backend at {self._url}")
            return
        except Exception as exc:
            log.debug(f"HelixDB fallback failed: {exc}")

        # In-memory fallback
        self._client = _InMemoryBackend()
        self._backend = "memory"
        log.warning("OpenViking: using in-memory backend (data not persisted)")

    @property
    def is_available(self) -> bool:
        return self._client is not None and self._backend != "none"

    @property
    def backend(self) -> str:
        return self._backend

    def store(self, doc: VikingDocument) -> bool:
        if not self._client:
            return False
        try:
            return self._client.store(doc)
        except Exception as exc:
            log.debug(f"OpenViking.store failed: {exc}")
            return False

    def store_batch(self, docs: list[VikingDocument]) -> int:
        """Store multiple documents. Returns count of successful stores."""
        return sum(1 for d in docs if self.store(d))

    def search(
        self,
        query:      str,
        n:          int  = 10,
        namespace:  str | None = None,
        file_filter: str | None = None,
        lang_filter: str | None = None,
    ) -> list[VikingSearchResult]:
        if not self._client:
            return []
        try:
            return self._client.search(
                query=query, n=n,
                namespace=namespace or self._namespace,
                file_filter=file_filter,
                lang_filter=lang_filter,
            )
        except Exception as exc:
            log.debug(f"OpenViking.search failed: {exc}")
            return []

    def delete_namespace(self, namespace: str) -> bool:
        if not self._client:
            return False
        try:
            return self._client.delete_namespace(namespace)
        except Exception:
            return False

    def delete_file(self, file_path: str) -> bool:
        if not self._client:
            return False
        try:
            return self._client.delete_file(file_path)
        except Exception:
            return False

    def stats(self) -> dict:
        if not self._client:
            return {"available": False}
        try:
            return {"backend": self._backend, **self._client.stats()}
        except Exception:
            return {"backend": self._backend}

    def close(self) -> None:
        if self._client and hasattr(self._client, "close"):
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None


# ──────────────────────────────────────────────────────────────────────────────
# Adapters
# ──────────────────────────────────────────────────────────────────────────────

class _HelixAdapter:
    """Wraps HelixDB to the OpenViking interface."""

    def __init__(self, helix, namespace: str = "default") -> None:
        self._helix     = helix
        self._namespace = namespace

    def store(self, doc: VikingDocument) -> bool:
        from memory.helixdb import HelixDocument
        ns_id = f"{self._namespace}:{doc.id}"
        self._helix.index(HelixDocument(
            id=ns_id, file_path=doc.file_path,
            line_start=doc.line_start, line_end=doc.line_end,
            language=doc.language, content=doc.content,
            summary=doc.summary,
        ))
        return True

    def search(self, query: str, n: int = 10, namespace: str = "default",
               file_filter: str | None = None, lang_filter: str | None = None,
               ) -> list[VikingSearchResult]:
        results = self._helix.search(query, n=n, file_filter=file_filter,
                                     lang_filter=lang_filter)
        return [
            VikingSearchResult(
                id=r.id.replace(f"{namespace}:", ""),
                file_path=r.file_path, line_start=r.line_start, line_end=r.line_end,
                language=r.language, summary=r.summary, score=r.score,
                namespace=namespace,
            )
            for r in results
        ]

    def delete_file(self, file_path: str) -> bool:
        self._helix.delete_file(file_path)
        return True

    def delete_namespace(self, namespace: str) -> bool:
        return False  # Not supported at HelixDB level

    def stats(self) -> dict:
        return self._helix.stats()

    def close(self) -> None:
        self._helix.close()


class _InMemoryBackend:
    """Minimal in-memory vector store (no persistence, for testing)."""

    def __init__(self) -> None:
        self._docs: list[VikingDocument] = []

    def store(self, doc: VikingDocument) -> bool:
        self._docs.append(doc)
        return True

    def search(self, query: str, n: int = 10, **_) -> list[VikingSearchResult]:
        # Keyword search fallback
        query_words = set(query.lower().split())
        scored = []
        for doc in self._docs:
            text = (doc.summary + " " + doc.content).lower()
            score = sum(1 for w in query_words if w in text) / max(len(query_words), 1)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            VikingSearchResult(
                id=d.id, file_path=d.file_path, line_start=d.line_start,
                line_end=d.line_end, language=d.language, summary=d.summary,
                score=s, namespace=d.namespace,
            )
            for s, d in scored[:n]
        ]

    def delete_file(self, file_path: str) -> bool:
        before = len(self._docs)
        self._docs = [d for d in self._docs if d.file_path != file_path]
        return len(self._docs) < before

    def delete_namespace(self, namespace: str) -> bool:
        before = len(self._docs)
        self._docs = [d for d in self._docs if d.namespace != namespace]
        return len(self._docs) < before

    def stats(self) -> dict:
        return {"document_count": len(self._docs), "backend": "memory"}

    def close(self) -> None:
        self._docs.clear()
