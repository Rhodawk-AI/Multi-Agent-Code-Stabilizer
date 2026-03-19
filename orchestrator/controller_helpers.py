"""
orchestrator/controller_helpers.py
====================================
Auxiliary helpers for StabilizerController.

Extracted from controller.py to resolve the latent ImportError where
_init_vector_store() imported HelixBrainShim from this module but the
module did not exist.  All helpers here are pure adapters — they carry
no business logic; that stays in controller.py.
"""
from __future__ import annotations

from typing import Any


class HelixBrainShim:
    """
    Adapts a HelixDB instance to the VectorBrain interface expected by
    StabilizerController and all agent subclasses.

    VectorBrain interface required by agents:
        .is_available       -> bool
        .initialise()       -> None
        .close()            -> None
        .index_chunk(...)   -> None
        .find_similar_to_issue(query, n) -> list[VectorSearchResult]
    """

    def __init__(self, helix: Any) -> None:
        self._helix = helix

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        return getattr(self._helix, "is_available", False)

    def initialise(self) -> None:
        # HelixDB.initialise() is called by the caller before constructing
        # this shim.  Nothing to do here.
        pass

    def close(self) -> None:
        try:
            self._helix.close()
        except Exception:
            pass

    # ── Write ──────────────────────────────────────────────────────────────────

    def index_chunk(
        self,
        chunk_id:     str,
        file_path:    str,
        line_start:   int,
        line_end:     int,
        language:     str,
        summary:      str,
        observations: list[str],
    ) -> None:
        from memory.helixdb import HelixDocument
        self._helix.index(
            HelixDocument(
                id=chunk_id,
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                language=language,
                content=" ".join(observations),
                summary=summary,
            )
        )

    # ── Read ───────────────────────────────────────────────────────────────────

    def find_similar_to_issue(self, query: str, n: int = 8) -> list[Any]:
        from brain.vector_store import VectorSearchResult

        results = self._helix.search(query, n=n)
        return [
            VectorSearchResult(
                chunk_id=r.id,
                file_path=r.file_path,
                line_start=r.line_start,
                line_end=r.line_end,
                language=r.language,
                summary=r.summary,
                distance=1.0 - r.score,
            )
            for r in results
        ]

    def find_similar(self, query: str, n: int = 8) -> list[Any]:
        """Alias used by some agent subclasses."""
        return self.find_similar_to_issue(query, n=n)
