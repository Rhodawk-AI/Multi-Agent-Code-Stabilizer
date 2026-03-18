"""
brain/vector_store.py
=====================
Semantic vector store for MACS using ChromaDB.

Closes GAP-8: the ``vector_store_enabled`` flag existed in config/default.toml
and BrainConfig, but the implementation was absent.

Architecture
────────────
• ``VectorBrain`` wraps a persistent ChromaDB collection.
• The ``ReaderAgent`` calls ``index_chunk()`` after each ``append_chunk()``
  when ``vector_store_enabled=True``.
• Any agent or controller can call ``find_similar()`` to find semantically
  related code chunks — e.g. "all places that handle balance arithmetic" for
  finance domain analysis, or "all safety gate patterns" for architecture audit.
• Falls back to a no-op stub when chromadb is not installed so the system
  degrades gracefully.

Dependencies
────────────
    pip install chromadb sentence-transformers

Notes
─────
• We use ``all-MiniLM-L6-v2`` (22 MB) as the default embedding model.
  For air-gapped deployments, supply ``embedding_function=None`` to use
  ChromaDB's built-in hash-based embedder (lower quality but zero deps).
• The ChromaDB collection is created with cosine distance — appropriate for
  code summaries where magnitude is less meaningful than direction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Optional imports — graceful degradation
# ──────────────────────────────────────────────────────────────────────────────

try:
    import chromadb                                          # type: ignore[import]
    from chromadb.config import Settings as _ChromaSettings  # type: ignore[import]
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    log.info(
        "chromadb not installed — VectorBrain operating in stub mode. "
        "Run: pip install chromadb sentence-transformers"
    )

try:
    from chromadb.utils.embedding_functions import (         # type: ignore[import]
        SentenceTransformerEmbeddingFunction as _STEF,
    )
    _STEF_AVAILABLE = True
except (ImportError, Exception):
    _STEF_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────────────

class VectorSearchResult:
    """Single result from ``VectorBrain.find_similar()``."""

    __slots__ = ("chunk_id", "file_path", "line_start", "line_end",
                 "language", "summary", "distance")

    def __init__(
        self,
        chunk_id:   str,
        file_path:  str,
        line_start: int,
        line_end:   int,
        language:   str,
        summary:    str,
        distance:   float,
    ) -> None:
        self.chunk_id   = chunk_id
        self.file_path  = file_path
        self.line_start = line_start
        self.line_end   = line_end
        self.language   = language
        self.summary    = summary
        self.distance   = distance

    def __repr__(self) -> str:
        return (
            f"VectorSearchResult(file={self.file_path!r}, "
            f"L{self.line_start}-{self.line_end}, dist={self.distance:.3f})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# VectorBrain
# ──────────────────────────────────────────────────────────────────────────────

class VectorBrain:
    """
    Persistent vector index over code chunks.

    Parameters
    ----------
    store_path:
        Directory for ChromaDB's on-disk storage.
    embedding_model:
        HuggingFace sentence-transformers model name.
        Pass ``None`` to use ChromaDB's built-in hash embedder.
    collection_name:
        ChromaDB collection name.
    """

    COLLECTION_NAME = "macs_code_chunks"

    def __init__(
        self,
        store_path:      str | Path = ".stabilizer/vectors",
        embedding_model: str | None = "all-MiniLM-L6-v2",
        collection_name: str        = COLLECTION_NAME,
    ) -> None:
        self._store_path   = Path(store_path)
        self._model_name   = embedding_model
        self._collection_name = collection_name
        self._client: Any       = None
        self._collection: Any   = None
        self._available: bool   = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def initialise(self) -> None:
        """Open (or create) the ChromaDB collection.  Idempotent."""
        if not _CHROMA_AVAILABLE:
            log.info("VectorBrain: chromadb unavailable — running as stub")
            return

        self._store_path.mkdir(parents=True, exist_ok=True)

        try:
            self._client = chromadb.PersistentClient(
                path=str(self._store_path),
                settings=_ChromaSettings(anonymized_telemetry=False),
            )
        except TypeError:
            # Older chromadb versions don't accept Settings in PersistentClient
            self._client = chromadb.PersistentClient(path=str(self._store_path))

        ef = self._make_embedding_function()

        try:
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
            self._available = True
            log.info(
                f"VectorBrain: collection '{self._collection_name}' ready "
                f"at {self._store_path} "
                f"({self._collection.count()} existing documents)"
            )
        except Exception as exc:
            log.error(f"VectorBrain: failed to open collection: {exc}")

    def _make_embedding_function(self) -> Any:
        """Return best available embedding function, falling back gracefully."""
        if self._model_name and _STEF_AVAILABLE:
            try:
                ef = _STEF(model_name=self._model_name)
                log.info(f"VectorBrain: using SentenceTransformer '{self._model_name}'")
                return ef
            except Exception as exc:
                log.warning(f"VectorBrain: SentenceTransformer failed ({exc}) — using default")
        log.info("VectorBrain: using ChromaDB default embedding function")
        return None   # ChromaDB will use its built-in

    def close(self) -> None:
        self._client    = None
        self._collection = None
        self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    # ── Write ────────────────────────────────────────────────────────────────

    def index_chunk(
        self,
        chunk_id:   str,
        file_path:  str,
        line_start: int,
        line_end:   int,
        language:   str,
        summary:    str,
        observations: list[str],
    ) -> None:
        """
        Upsert a single code chunk into the vector index.

        The document text is ``summary + " " + " ".join(observations)`` which
        gives the model both high-level context and concrete low-level signals.
        """
        if not self._available:
            return

        document = (summary + " " + " ".join(observations)).strip()
        if not document:
            return

        metadata: dict[str, Any] = {
            "file_path":  file_path,
            "line_start": line_start,
            "line_end":   line_end,
            "language":   language,
            "summary":    summary[:500],   # ChromaDB metadata values must be str/int/float
        }

        try:
            self._collection.upsert(
                ids=[chunk_id],
                documents=[document],
                metadatas=[metadata],
            )
        except Exception as exc:
            log.debug(f"VectorBrain.index_chunk failed for {chunk_id}: {exc}")

    def index_chunks_batch(self, chunks: list[dict]) -> None:
        """
        Bulk upsert.  Each dict must have the same keys as the parameters of
        ``index_chunk()``.
        """
        if not self._available or not chunks:
            return

        ids       = []
        documents = []
        metadatas = []

        for c in chunks:
            doc = (c.get("summary", "") + " " + " ".join(c.get("observations", []))).strip()
            if not doc:
                continue
            ids.append(c["chunk_id"])
            documents.append(doc)
            metadatas.append({
                "file_path":  c["file_path"],
                "line_start": c.get("line_start", 0),
                "line_end":   c.get("line_end", 0),
                "language":   c.get("language", "unknown"),
                "summary":    c.get("summary", "")[:500],
            })

        if not ids:
            return

        try:
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            log.debug(f"VectorBrain: indexed {len(ids)} chunks")
        except Exception as exc:
            log.warning(f"VectorBrain.index_chunks_batch failed: {exc}")

    # ── Read ─────────────────────────────────────────────────────────────────

    def find_similar(
        self,
        query:          str,
        n:              int           = 10,
        language_filter: str | None   = None,
        file_filter:    str | None    = None,
    ) -> list[VectorSearchResult]:
        """
        Semantic nearest-neighbour search.

        Parameters
        ----------
        query:
            Natural language or code snippet to search for.
        n:
            Maximum results to return.
        language_filter:
            If set, restrict results to chunks from files of this language.
        file_filter:
            If set, restrict results to this specific file path.

        Returns
        -------
        List of ``VectorSearchResult`` sorted by ascending cosine distance
        (lower = more similar).
        """
        if not self._available:
            return []

        where: dict[str, Any] | None = None
        filters: list[dict] = []
        if language_filter:
            filters.append({"language": {"$eq": language_filter}})
        if file_filter:
            filters.append({"file_path": {"$eq": file_filter}})

        if len(filters) == 1:
            where = filters[0]
        elif len(filters) > 1:
            where = {"$and": filters}

        try:
            raw = self._collection.query(
                query_texts=[query],
                n_results=min(n, max(1, self._collection.count())),
                where=where,
            )
        except Exception as exc:
            log.warning(f"VectorBrain.find_similar failed: {exc}")
            return []

        results: list[VectorSearchResult] = []
        ids       = raw.get("ids",       [[]])[0]
        metas     = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for chunk_id, meta, dist in zip(ids, metas, distances):
            results.append(VectorSearchResult(
                chunk_id   = chunk_id,
                file_path  = meta.get("file_path",  ""),
                line_start = int(meta.get("line_start", 0)),
                line_end   = int(meta.get("line_end",   0)),
                language   = meta.get("language",   "unknown"),
                summary    = meta.get("summary",    ""),
                distance   = float(dist),
            ))

        return results

    def find_similar_to_issue(
        self,
        issue_description: str,
        n: int = 8,
    ) -> list[VectorSearchResult]:
        """
        Convenience wrapper: find code chunks semantically related to a given
        issue description.  Used by the fixer to gather cross-file context.
        """
        return self.find_similar(
            query=f"code pattern: {issue_description}",
            n=n,
        )

    def collection_size(self) -> int:
        if not self._available:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    def delete_file_chunks(self, file_path: str) -> None:
        """Remove all vectors for *file_path* — called when a file is re-indexed."""
        if not self._available:
            return
        try:
            self._collection.delete(where={"file_path": {"$eq": file_path}})
        except Exception as exc:
            log.debug(f"VectorBrain.delete_file_chunks({file_path}): {exc}")

    def summary(self) -> dict:
        return {
            "available":       self._available,
            "collection":      self._collection_name,
            "store_path":      str(self._store_path),
            "document_count":  self.collection_size(),
            "embedding_model": self._model_name or "chromadb-default",
        }
