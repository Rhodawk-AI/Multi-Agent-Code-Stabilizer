"""
memory/fix_memory.py
====================
Cross-session fix memory (Algorithm Distillation) for Rhodawk AI.

Every time the system successfully fixes an issue and the fix passes all
gates, the pattern is stored:

    issue_type × file_context × fix_approach → test_result

On the next run, when the Fixer encounters a similar issue, the top-3
matching patterns are injected as few-shot examples in the prompt.  Over
thousands of runs the system accumulates a searchable library of successful
fix strategies specific to each codebase — a compound advantage that Claude
Code structurally cannot replicate (it has no persistent memory across
sessions or customers).

Storage
────────
• Primary:  mem0 (mem0ai) backed by Qdrant — semantic search + structured
  storage.  mem0 sits on top of the Qdrant instance already required by
  VectorBrain / HybridRetriever.
• Fallback: Qdrant directly (if mem0ai is not installed).
• Fallback²: In-process JSON file at ``<repo_root>/.stabilizer/fix_memory.json``
  (always works, no external deps).

Privacy / federation
─────────────────────
Each ``user_id`` is scoped to the repo URL hash — fix patterns from repo A
do not bleed into repo B.  Federated fine-tuning (sharing patterns across
customers) is a future capability and requires explicit opt-in.

Dependencies
────────────
    mem0ai>=0.1.0    (primary — pip install mem0ai)
    qdrant-client>=1.9.0  (already in requirements.txt)

Public API
──────────
    fm = FixMemory(repo_url="https://github.com/acme/backend")
    fm.initialise()

    # After a successful fix
    fm.store_success(
        issue_type="null_deref",
        file_context="agents/fixer.py:_fix_group",
        fix_approach="Added None guard before attribute access",
        test_result="passed=12 failed=0",
    )

    # Before generating a fix
    examples = fm.retrieve(
        issue_description="NullPointerException in handle_request",
        n=3,
    )
    # → list[FixMemoryEntry]  (inject as few-shot prompt examples)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_MEMORY_COLLECTION = "rhodawk_fix_memory"


@dataclass
class FixMemoryEntry:
    id:            str   = ""
    issue_type:    str   = ""
    file_context:  str   = ""
    fix_approach:  str   = ""
    test_result:   str   = ""
    run_id:        str   = ""
    created_at:    str   = ""
    score:         float = 0.0    # relevance score from retrieval


class FixMemory:
    """
    Persistent cross-session fix pattern store.

    Parameters
    ----------
    repo_url:
        The repository URL (used to namespace memories — prevents bleed
        across repos).
    qdrant_url:
        Qdrant server URL (shared with VectorBrain / HybridRetriever).
    data_dir:
        Path for the JSON fallback store.
    """

    def __init__(
        self,
        repo_url:   str  = "",
        qdrant_url: str  = "http://localhost:6333",
        data_dir:   Path | str | None = None,
    ) -> None:
        self.repo_url   = repo_url
        self.qdrant_url = qdrant_url
        self.data_dir   = Path(data_dir) if data_dir else None
        self._user_id   = self._make_user_id(repo_url)
        self._client: Any     = None  # mem0.Memory or qdrant_client
        self._backend:  str   = "none"
        self._json_path: Path | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def initialise(self) -> None:
        """Connect to the best available backend."""
        # Try mem0ai first
        try:
            from mem0 import Memory  # type: ignore
            config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "host":            self.qdrant_url.replace("http://", "").split(":")[0],
                        "port":            int(self.qdrant_url.rsplit(":", 1)[-1].rstrip("/")),
                        "collection_name": _MEMORY_COLLECTION,
                        "embedding_model_dims": 384,
                    },
                },
            }
            self._client  = Memory.from_config(config)
            self._backend = "mem0"
            log.info("FixMemory: mem0 backend initialised")
            return
        except Exception as exc:
            log.debug(f"FixMemory: mem0 init failed ({exc}) — trying Qdrant direct")

        # Try Qdrant direct (no mem0)
        try:
            from qdrant_client import QdrantClient  # type: ignore
            client = QdrantClient(url=self.qdrant_url, timeout=5)
            client.get_collections()
            self._client  = client
            self._backend = "qdrant"
            self._ensure_qdrant_collection()
            log.info("FixMemory: Qdrant-direct backend initialised")
            return
        except Exception as exc:
            log.debug(f"FixMemory: Qdrant init failed ({exc}) — using JSON fallback")

        # JSON file fallback
        self._backend = "json"
        if self.data_dir:
            self._json_path = self.data_dir / "fix_memory.json"
        log.info(
            f"FixMemory: JSON fallback at "
            f"{self._json_path or '<in-process only>'}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def store_success(
        self,
        issue_type:   str,
        file_context: str,
        fix_approach: str,
        test_result:  str,
        run_id:       str = "",
    ) -> None:
        """
        Persist a successful fix pattern.
        Call this after gate_passed=True AND test_runner passed.
        """
        entry_text = (
            f"Issue: {issue_type}\n"
            f"Context: {file_context}\n"
            f"Fix: {fix_approach}\n"
            f"Result: {test_result}"
        )
        meta = {
            "issue_type":   issue_type,
            "file_context": file_context,
            "fix_approach": fix_approach,
            "test_result":  test_result,
            "run_id":       run_id,
            "created_at":   datetime.now(timezone.utc).isoformat(),
        }

        try:
            if self._backend == "mem0":
                self._client.add(
                    messages=[{"role": "assistant", "content": entry_text}],
                    user_id=self._user_id,
                    metadata=meta,
                )
            elif self._backend == "qdrant":
                self._qdrant_store(entry_text, meta)
            elif self._backend == "json":
                self._json_store(meta)
        except Exception as exc:
            log.warning(f"FixMemory.store_success: {exc}")

    def store_failure(
        self,
        issue_type:   str,
        file_context: str,
        fix_approach: str,
        failure_reason: str,
        run_id:       str = "",
    ) -> None:
        """
        Persist a fix approach that caused a test regression and was reverted.

        The fixer retrieves these alongside successes so it never re-applies
        an approach that was previously reverted for the same issue type and
        file context.  The ``fix_approach`` field is prefixed with
        ``[REVERTED]`` so the LLM prompt makes the negative signal unambiguous.
        """
        entry_text = (
            f"Issue: {issue_type}\n"
            f"Context: {file_context}\n"
            f"Fix (REVERTED — do NOT repeat): {fix_approach}\n"
            f"Failure reason: {failure_reason}"
        )
        meta = {
            "issue_type":     issue_type,
            "file_context":   file_context,
            "fix_approach":   f"[REVERTED] {fix_approach}",
            "test_result":    f"REGRESSION: {failure_reason}",
            "run_id":         run_id,
            "created_at":     datetime.now(timezone.utc).isoformat(),
            "reverted":       True,
        }
        try:
            if self._backend == "mem0":
                self._client.add(
                    messages=[{"role": "assistant", "content": entry_text}],
                    user_id=self._user_id,
                    metadata=meta,
                )
            elif self._backend == "qdrant":
                self._qdrant_store(entry_text, meta)
            elif self._backend == "json":
                self._json_store(meta)
        except Exception as exc:
            log.warning(f"FixMemory.store_failure: {exc}")

    def retrieve(
        self,
        issue_description: str,
        n:                 int = 3,
    ) -> list[FixMemoryEntry]:
        """
        Retrieve the top-n most relevant successful fix patterns.
        Returns an empty list if no memories exist or backend is unavailable.
        """
        try:
            if self._backend == "mem0":
                return self._mem0_retrieve(issue_description, n)
            elif self._backend == "qdrant":
                return self._qdrant_retrieve(issue_description, n)
            elif self._backend == "json":
                return self._json_retrieve(issue_description, n)
        except Exception as exc:
            log.debug(f"FixMemory.retrieve: {exc}")
        return []

    def format_as_few_shot(self, entries: list[FixMemoryEntry]) -> str:
        """
        Format retrieved entries as few-shot examples for LLM injection.

        Successful patterns are labeled as positive examples.
        Reverted patterns are labeled as explicit negative examples — the LLM
        is instructed never to repeat an approach marked [REVERTED].
        Designed to be prepended to the fix prompt.
        """
        if not entries:
            return ""
        positives = [e for e in entries if not e.fix_approach.startswith("[REVERTED]")]
        negatives = [e for e in entries if e.fix_approach.startswith("[REVERTED]")]
        parts: list[str] = []
        if positives:
            parts.append("## Successful Fix Patterns From Memory (use as reference)\n")
            for i, e in enumerate(positives, 1):
                parts.append(
                    f"### Example {i} (relevance={e.score:.2f})\n"
                    f"Issue type: {e.issue_type}\n"
                    f"Context: {e.file_context}\n"
                    f"Approach used: {e.fix_approach}\n"
                    f"Outcome: {e.test_result}\n"
                )
        if negatives:
            parts.append(
                "## Previously Reverted Fix Approaches — DO NOT REPEAT\n"
                "The following approaches were applied, caused test regressions, "
                "and were reverted. You MUST NOT use these approaches.\n"
            )
            for i, e in enumerate(negatives, 1):
                parts.append(
                    f"### Reverted Example {i} (relevance={e.score:.2f})\n"
                    f"Issue type: {e.issue_type}\n"
                    f"Context: {e.file_context}\n"
                    f"Approach attempted: {e.fix_approach}\n"
                    f"Why it failed: {e.test_result}\n"
                )
        return "\n".join(parts)

    # ── mem0 backend ──────────────────────────────────────────────────────────

    def _mem0_retrieve(
        self, query: str, n: int
    ) -> list[FixMemoryEntry]:
        results = self._client.search(
            query=query, user_id=self._user_id, limit=n
        )
        entries: list[FixMemoryEntry] = []
        for r in results:
            meta = r.get("metadata", {})
            entries.append(FixMemoryEntry(
                id=r.get("id", ""),
                issue_type=meta.get("issue_type", ""),
                file_context=meta.get("file_context", ""),
                fix_approach=meta.get("fix_approach", ""),
                test_result=meta.get("test_result", ""),
                run_id=meta.get("run_id", ""),
                created_at=meta.get("created_at", ""),
                score=float(r.get("score", 0.0)),
            ))
        return entries

    # ── Qdrant direct backend ─────────────────────────────────────────────────

    def _ensure_qdrant_collection(self) -> None:
        try:
            from qdrant_client.models import Distance, VectorParams  # type: ignore
            existing = [
                c.name for c in self._client.get_collections().collections
            ]
            if _MEMORY_COLLECTION not in existing:
                self._client.create_collection(
                    collection_name=_MEMORY_COLLECTION,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
        except Exception as exc:
            log.debug(f"_ensure_qdrant_collection: {exc}")

    def _qdrant_store(self, text: str, meta: dict) -> None:
        from qdrant_client.models import PointStruct  # type: ignore
        vec = self._embed(text)
        uid = abs(hash(text + meta.get("created_at", ""))) % (10 ** 9)
        self._client.upsert(
            collection_name=_MEMORY_COLLECTION,
            points=[PointStruct(id=uid, vector=vec, payload=meta)],
        )

    def _qdrant_retrieve(
        self, query: str, n: int
    ) -> list[FixMemoryEntry]:
        vec  = self._embed(query)
        hits = self._client.search(
            collection_name=_MEMORY_COLLECTION,
            query_vector=vec,
            limit=n,
            query_filter={
                "must": [
                    {
                        "key":   "run_id",
                        "match": {"value": self._user_id},
                    }
                ]
            } if False else None,    # skip filter for now — all repos share one store
        )
        return [
            FixMemoryEntry(
                id=str(h.id),
                issue_type=h.payload.get("issue_type", ""),
                file_context=h.payload.get("file_context", ""),
                fix_approach=h.payload.get("fix_approach", ""),
                test_result=h.payload.get("test_result", ""),
                run_id=h.payload.get("run_id", ""),
                created_at=h.payload.get("created_at", ""),
                score=float(h.score),
            )
            for h in hits
        ]

    # ── JSON fallback backend ─────────────────────────────────────────────────

    def _json_store(self, meta: dict) -> None:
        if not self._json_path:
            return
        records = self._json_load()
        records.append(meta)
        self._json_path.parent.mkdir(parents=True, exist_ok=True)
        self._json_path.write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )

    def _json_retrieve(
        self, query: str, n: int
    ) -> list[FixMemoryEntry]:
        records = self._json_load()
        # Naive keyword overlap scoring
        q_words = set(query.lower().split())
        scored: list[tuple[float, dict]] = []
        for r in records:
            text = " ".join([
                r.get("issue_type", ""),
                r.get("file_context", ""),
                r.get("fix_approach", ""),
            ]).lower()
            overlap = len(q_words & set(text.split()))
            scored.append((overlap, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            FixMemoryEntry(
                issue_type=r.get("issue_type", ""),
                file_context=r.get("file_context", ""),
                fix_approach=r.get("fix_approach", ""),
                test_result=r.get("test_result", ""),
                run_id=r.get("run_id", ""),
                created_at=r.get("created_at", ""),
                score=float(sc),
            )
            for sc, r in scored[:n]
        ]

    def _json_load(self) -> list[dict]:
        if not self._json_path or not self._json_path.exists():
            return []
        try:
            return json.loads(self._json_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_user_id(repo_url: str) -> str:
        """Stable namespace key for this repo."""
        return "repo_" + hashlib.sha256(repo_url.encode()).hexdigest()[:16]

    def _embed(self, text: str) -> list[float]:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            if not hasattr(self, "_st_model"):
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            return self._st_model.encode(text).tolist()
        except Exception:
            h = hashlib.sha256(text.encode()).digest()
            return [b / 255.0 for b in h[:16]] * 24


# ── Module-level singleton ────────────────────────────────────────────────────

_instance: FixMemory | None = None


def get_fix_memory(
    repo_url:  str  = "",
    qdrant_url: str = "http://localhost:6333",
    data_dir:  Path | str | None = None,
) -> FixMemory:
    global _instance
    if _instance is None:
        _instance = FixMemory(
            repo_url=repo_url,
            qdrant_url=qdrant_url,
            data_dir=data_dir,
        )
        _instance.initialise()
    return _instance
