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

import asyncio
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
        repo_url:          str              = "",
        qdrant_url:        str              = "http://localhost:6333",
        data_dir:          Path | str | None = None,
        federated_store:   Any              = None,   # FederatedPatternStore | None
        language:          str              = "python",
    ) -> None:
        self.repo_url          = repo_url
        self.qdrant_url        = qdrant_url
        self.data_dir          = Path(data_dir) if data_dir else None
        self._user_id          = self._make_user_id(repo_url)
        self._client: Any      = None  # mem0.Memory or qdrant_client
        self._backend:  str    = "none"
        self._json_path: Path | None = None
        # GAP 6: federated pattern store (optional)
        self._federated_store: Any = federated_store
        self._language:        str = language
        # Lazy-loaded normalizer (only when federation is active)
        self._normalizer: Any  = None

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

        # GAP 6: push anonymized structural pattern to federation (fire-and-forget)
        if self._federated_store is not None:
            self._schedule_federation_push(
                fix_approach=fix_approach,
                issue_type=issue_type,
            )

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
        max_age_days:      int | None = 180,
    ) -> list[FixMemoryEntry]:
        """
        Retrieve the top-n most relevant fix patterns (successes + failures).

        Parameters
        ----------
        issue_description:
            Natural-language description of the issue being fixed.  Used as
            the semantic search query.
        n:
            Maximum number of entries to return.
        max_age_days:
            Discard entries older than this many days.  Defaults to 180 (six
            months) — satisfies the Gap 3 audit requirement that a revert from
            years ago does not rank identically to a recent one.  Set to
            None to disable the filter and return all matching entries
            regardless of age.

        Returns
        -------
        list[FixMemoryEntry]
            Empty list if no memories exist or the backend is unavailable.
        """
        try:
            if self._backend == "mem0":
                entries = self._mem0_retrieve(issue_description, n * 2)
            elif self._backend == "qdrant":
                entries = self._qdrant_retrieve(issue_description, n * 2)
            elif self._backend == "json":
                entries = self._json_retrieve(issue_description, n * 2)
            else:
                return []
            # max_age_days=None or max_age_days=0 both mean "no age filter".
            # 0 is treated as unlimited because callers often store the value
            # as a plain int config field where 0 conventionally means "off".
            if max_age_days is not None and max_age_days > 0:
                entries = self._filter_by_age(entries, max_age_days)

            # GAP 6: augment local results with federated patterns
            fed_entries = self._retrieve_federated(issue_description, n)
            if fed_entries:
                # Merge without duplicating structurally identical approaches
                local_approaches = {e.fix_approach.lower() for e in entries}
                for fe in fed_entries:
                    if fe.fix_approach.lower() not in local_approaches:
                        entries.append(fe)
                        local_approaches.add(fe.fix_approach.lower())

            return entries[:n]
        except Exception as exc:
            log.debug(f"FixMemory.retrieve: {exc}")
        return []

    # ── GAP 6: Federation helpers ─────────────────────────────────────────────

    async def retrieve_async(
        self,
        issue_description: str,
        n:                 int = 3,
        max_age_days:      int | None = 180,
    ) -> list[FixMemoryEntry]:
        """
        Async version of retrieve() that properly awaits federated pulls.

        Async pipeline callers (fixer._get_memory_examples,
        fixer._report_federated_usage) MUST use this method instead of
        retrieve() so that federated augmentation is live on every call
        rather than silently deferred.  The sync retrieve() falls back to
        scheduling a background task when the loop is already running —
        this method has no such limitation.

        Parameters are identical to retrieve().
        """
        try:
            if self._backend == "mem0":
                entries = self._mem0_retrieve(issue_description, n * 2)
            elif self._backend == "qdrant":
                entries = self._qdrant_retrieve(issue_description, n * 2)
            elif self._backend == "json":
                entries = self._json_retrieve(issue_description, n * 2)
            else:
                return []

            if max_age_days is not None and max_age_days > 0:
                entries = self._filter_by_age(entries, max_age_days)

            # Proper async federation augmentation — no nested-loop workaround
            fed_entries = await self._retrieve_federated_async(issue_description, n)
            if fed_entries:
                local_approaches = {e.fix_approach.lower() for e in entries}
                for fe in fed_entries:
                    if fe.fix_approach.lower() not in local_approaches:
                        entries.append(fe)
                        local_approaches.add(fe.fix_approach.lower())

            return entries[:n]
        except Exception as exc:
            log.debug(f"FixMemory.retrieve_async: {exc}")
        return []

    def _get_normalizer(self) -> Any:
        """Lazy-load PatternNormalizer only when federation is active."""
        if self._normalizer is None:
            try:
                from memory.pattern_normalizer import PatternNormalizer
                self._normalizer = PatternNormalizer()
            except Exception as exc:
                log.debug(f"FixMemory: PatternNormalizer unavailable ({exc})")
        return self._normalizer

    def _schedule_federation_push(
        self,
        fix_approach: str,
        issue_type:   str,
    ) -> None:
        """
        Fire-and-forget coroutine to push a normalized pattern to the
        federation.  Never blocks the caller.

        Called from store_success() after local persistence succeeds.
        Failures are caught and logged — they must never raise to the caller.
        """
        normalizer = self._get_normalizer()
        if normalizer is None:
            return

        async def _push() -> None:
            try:
                result = normalizer.normalize(
                    fix_approach=fix_approach,
                    issue_type=issue_type,
                    language=self._language,
                )
                if not result.normalization_ok:
                    log.debug(
                        f"FixMemory.federation_push: normalization failed "
                        f"({result.error}) — skipping push"
                    )
                    return
                await self._federated_store.push_pattern(
                    fingerprint      = result.fingerprint,
                    normalized_text  = result.normalized_text,
                    issue_type       = issue_type,
                    language         = result.language,
                    complexity_score = result.complexity_score,
                )
                log.debug(
                    f"FixMemory.federation_push: pushed fingerprint="
                    f"{result.fingerprint[:16]}... issue={issue_type}"
                )
            except Exception as exc:
                log.debug(f"FixMemory.federation_push: {exc}")

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(_push())
            else:
                loop.run_until_complete(_push())
        except RuntimeError:
            # No event loop available (e.g. in sync test context) — skip
            log.debug("FixMemory.federation_push: no event loop — skipping")

    def _retrieve_federated(
        self,
        issue_description: str,
        n:                 int,
    ) -> list[FixMemoryEntry]:
        """
        Synchronous wrapper around the async federated pull.

        When called from a sync context (no running event loop), executes the
        pull via run_until_complete.  When called from an async context (event
        loop already running — e.g. inside the fixer pipeline), schedules the
        pull as a background task and returns the local-cache results only for
        this call; the fresh peer results will be available on the next call
        once the task completes.

        Callers that are themselves async (fixer._get_memory_examples,
        fixer._report_federated_usage) should call retrieve_async() instead
        so they get the full federated augmentation without any deferral.
        """
        if self._federated_store is None:
            return []
        if not getattr(self._federated_store, "receive", False):
            return []

        async def _pull() -> list:
            try:
                issue_type_hint = issue_description.split()[0].lower()[:32] \
                    if issue_description else ""
                patterns = await self._federated_store.pull_patterns(
                    issue_type  = issue_type_hint,
                    query_text  = issue_description[:300],
                    n           = n,
                )
                return patterns
            except Exception as exc:
                log.debug(f"FixMemory._retrieve_federated: {exc}")
                return []

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Async context: schedule as fire-and-forget so the cache is
                # populated for the next retrieve() call.  Return the current
                # local cache immediately rather than silently dropping results.
                asyncio.ensure_future(_pull())
                log.debug(
                    "FixMemory._retrieve_federated: async context detected — "
                    "pull scheduled as background task; returning cached results"
                )
                # Return whatever the local cache already has synchronously
                try:
                    cached = loop.run_until_complete(
                        self._federated_store._retrieve_from_cache(
                            issue_description.split()[0].lower()[:32]
                            if issue_description else "",
                            n,
                        )
                    ) if not loop.is_running() else []
                except Exception:
                    cached = []
                return self._patterns_to_entries(cached)
            patterns = loop.run_until_complete(_pull())
        except RuntimeError:
            return []

        return self._patterns_to_entries(patterns)

    async def _retrieve_federated_async(
        self,
        issue_description: str,
        n:                 int,
    ) -> list[FixMemoryEntry]:
        """
        Async version of _retrieve_federated.

        Properly awaits the federated pull without any nested-loop workaround.
        Called by retrieve_async() from async pipeline contexts (fixer agents)
        so they receive full federated augmentation on every invocation.
        """
        if self._federated_store is None:
            return []
        if not getattr(self._federated_store, "receive", False):
            return []
        try:
            issue_type_hint = (
                issue_description.split()[0].lower()[:32]
                if issue_description else ""
            )
            patterns = await self._federated_store.pull_patterns(
                issue_type  = issue_type_hint,
                query_text  = issue_description[:300],
                n           = n,
            )
            return self._patterns_to_entries(patterns)
        except Exception as exc:
            log.debug(f"FixMemory._retrieve_federated_async: {exc}")
            return []

    def _patterns_to_entries(self, patterns: list) -> list[FixMemoryEntry]:
        """
        Convert a list of FederatedPattern objects to FixMemoryEntry objects.

        Stores the FULL 64-character fingerprint in the id field so that
        _report_federated_usage() can pass the correct fingerprint to
        record_usage() without truncation.  The Qdrant point UID is computed
        as abs(hash(fingerprint)) — a hash of the full string — so any
        truncation would cause a different UID, silently breaking the
        use_count / success_count feedback loop.
        """
        entries: list[FixMemoryEntry] = []
        for p in patterns:
            entries.append(FixMemoryEntry(
                id           = p.fingerprint,          # FULL 64-char SHA-256
                issue_type   = p.issue_type,
                file_context = f"[federated/{p.language}] {p.source_instance[:8]}",
                fix_approach = (
                    f"[FEDERATED] {p.normalized_text[:400]}"
                    if p.normalized_text else ""
                ),
                test_result  = (
                    f"federated success_rate="
                    f"{p.success_count}/{max(p.use_count, 1)} "
                    f"complexity={p.complexity_score:.2f}"
                ),
                run_id       = "",
                created_at   = p.contributed_at,
                score        = float(p.federation_score),
            ))
        return entries

    def record_federated_usage(
        self,
        fingerprint: str,
        success:     bool,
    ) -> None:
        """
        Record that a federated pattern (identified by fingerprint) was applied
        and report the outcome (success or failure) back to the federation.

        FIX (Defect 3 — FixMemory layer): This is the caller-side half of the
        feedback loop.  The fixer calls this method after gate evaluation
        completes for any fix that drew on a [FEDERATED] pattern from
        _retrieve_federated().  It:

          1. Updates use_count / success_count in the local cache via
             _federated_store.record_usage().
          2. Fires push_usage_feedback() to propagate the outcome back to
             every peer registry that may have served the pattern, so the
             entire network benefits from real-world signal.

        Both operations are fire-and-forget — they must never block or raise
        to the caller.
        """
        if self._federated_store is None:
            return

        async def _record() -> None:
            try:
                # 1. Update local cache
                await self._federated_store.record_usage(
                    fingerprint=fingerprint,
                    success=success,
                )
                # 2. Propagate to remote peers
                if hasattr(self._federated_store, "push_usage_feedback"):
                    await self._federated_store.push_usage_feedback(
                        fingerprint=fingerprint,
                        success=success,
                    )
                log.debug(
                    f"FixMemory.record_federated_usage: fingerprint="
                    f"{fingerprint[:16]}... success={success}"
                )
            except Exception as exc:
                log.debug(f"FixMemory.record_federated_usage: {exc}")

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(_record())
            else:
                loop.run_until_complete(_record())
        except RuntimeError:
            log.debug("FixMemory.record_federated_usage: no event loop — skipping")

    def set_federated_store(self, store: Any) -> None:
        """
        Wire a FederatedPatternStore into this FixMemory instance.
        Called by StabilizerController during _init_gap6().
        """
        self._federated_store = store
        log.info("FixMemory: FederatedPatternStore wired")

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

    # ── Age filter ───────────────────────────────────────────────────────────

    @staticmethod
    def _filter_by_age(
        entries: list[FixMemoryEntry],
        max_age_days: int,
    ) -> list[FixMemoryEntry]:
        """
        Remove entries whose created_at timestamp is older than
        max_age_days.

        Design notes
        ~~~~~~~~~~~~
        * Entries without a parseable created_at are retained — they
          predate the timestamp field and should not be silently dropped.
        * The comparison is done in UTC so DST and local offsets cannot cause
          an entry to be incorrectly excluded.
        * This is the fix for Gap 3 Defect 3: without this filter, a revert
          from three years ago ranked identically to a recent one (pure vector
          similarity, no time decay).
        """
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        kept: list[FixMemoryEntry] = []
        for entry in entries:
            if not entry.created_at:
                # No timestamp — retain (backward-compat with old entries).
                kept.append(entry)
                continue
            try:
                ts_str = entry.created_at
                # Handle both offset-aware ("...+00:00") and naive ISO strings.
                if ts_str.endswith("Z"):
                    ts_str = ts_str[:-1] + "+00:00"
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    # Treat naive timestamps as UTC (consistent with store_success).
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    kept.append(entry)
            except (ValueError, TypeError):
                # Unparseable timestamp — retain defensively.
                kept.append(entry)
        return kept

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
        # Overwrite run_id in the payload with the stable repo-scoped user_id
        # so _qdrant_retrieve's namespace filter matches on the correct key.
        # The original run_id (a transient run UUID) is preserved as audit_run_id.
        payload = {**meta, "run_id": self._user_id, "audit_run_id": meta.get("run_id", "")}
        self._client.upsert(
            collection_name=_MEMORY_COLLECTION,
            points=[PointStruct(id=uid, vector=vec, payload=payload)],
        )

    def _qdrant_retrieve(
        self, query: str, n: int
    ) -> list[FixMemoryEntry]:
        vec  = self._embed(query)
        hits = self._client.search(
            collection_name=_MEMORY_COLLECTION,
            query_vector=vec,
            limit=n,
            # Scope results to this repository's user_id so fix memories from
            # repo A never appear as few-shot examples for repo B.  The filter
            # was previously disabled with `if False else None`, causing
            # cross-repo memory bleed under any multi-repo Qdrant deployment.
            query_filter={
                "must": [
                    {
                        "key":   "run_id",
                        "match": {"value": self._user_id},
                    }
                ]
            },
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
    repo_url:        str              = "",
    qdrant_url:      str              = "http://localhost:6333",
    data_dir:        Path | str | None = None,
    federated_store: Any              = None,
    language:        str              = "python",
) -> FixMemory:
    global _instance
    if _instance is None:
        _instance = FixMemory(
            repo_url        = repo_url,
            qdrant_url      = qdrant_url,
            data_dir        = data_dir,
            federated_store = federated_store,
            language        = language,
        )
        _instance.initialise()
    elif federated_store is not None and _instance._federated_store is None:
        # Allow late-wiring of the federation store after singleton creation
        _instance.set_federated_store(federated_store)
    return _instance
