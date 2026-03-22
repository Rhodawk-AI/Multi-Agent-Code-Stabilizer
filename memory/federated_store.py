"""
memory/federated_store.py
==========================
Federated anonymized fix-pattern store for Rhodawk AI (GAP 6).

Architecture
────────────
Each Rhodawk deployment can optionally participate in a federated pattern
network.  Participation is entirely opt-in and controlled by two independent
flags:

  gap6_federation_enabled   = true    # master switch
  gap6_contribute_patterns  = true    # allow this deployment to push patterns
  gap6_receive_patterns     = true    # allow this deployment to pull patterns

When a fix is committed and ``store_success()`` fires, the pattern normalizer
strips all identifiers and the anonymized structural pattern is pushed to every
configured peer registry (including a central hub if configured).

When the fixer asks for few-shot examples via ``retrieve()``, the local memory
is augmented with patterns pulled from the federation.  The federation results
are ranked by structural similarity and interleaved with local results.

Privacy model
─────────────
Only normalized patterns leave the deployment.  The normalization pipeline
(PatternNormalizer) guarantees:
• Zero variable/function/class names from source code
• Zero string or numeric literals
• Only structural shapes: control flow, type annotations, operator sequences

The structural fingerprint (SHA-256 of normalized text) enables registry-level
deduplication without the registry ever seeing raw source.

Federation topology
────────────────────
The registry is a peer-to-peer HTTP REST network.  There is no required central
server — any Rhodawk instance can act as a registry for others.  For hub-and-
spoke deployments, configure ``gap6_registry_url`` to point to a shared hub.

Registry endpoints (served by ``api/routes/federation.py``):
  POST /api/federation/patterns          — receive a pattern from a peer
  GET  /api/federation/patterns          — serve patterns to peers
  GET  /api/federation/status            — registry health + stats
  POST /api/federation/peers             — register a peer
  DELETE /api/federation/peers/{peer_id} — deregister a peer

Local pattern cache
────────────────────
Received federated patterns are stored in a dedicated Qdrant collection
(``rhodawk_fed_patterns``) keyed by their structural fingerprint.  This
prevents re-download of patterns already seen and enables offline operation
when peers are unavailable.

Deduplication
──────────────
Before pushing, the fingerprint is checked against the registry.  If an
identical fingerprint already exists (pushed by another deployment that fixed
the same structural bug), the push is skipped.  This is the mechanism that
converts N independent deployments fixing the same pattern into a single
shared entry — without any deployment seeing another's code.

Public API
──────────
    from memory.federated_store import FederatedPatternStore

    store = FederatedPatternStore(
        instance_id="deploy-abc123",       # stable ID for this deployment
        registry_url="https://hub.example.com",
        qdrant_url="http://localhost:6333",
        contribute=True,
        receive=True,
    )
    await store.initialise()

    # Push a new normalized pattern (called from FixMemory.store_success)
    await store.push_pattern(
        fingerprint="sha256hex...",
        normalized_text="func_0 arg_0: if var_0 is None: raise var_str_0",
        issue_type="null_deref",
        language="python",
        complexity_score=0.72,
    )

    # Pull matching patterns (called from FixMemory.retrieve augmentation)
    patterns = await store.pull_patterns(
        issue_type="null_deref",
        query_text="NullPointerException in request handler",
        n=5,
    )
    # → list[FederatedPattern]
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

log = logging.getLogger(__name__)

_FED_COLLECTION = "rhodawk_fed_patterns"
_LOCAL_REGISTRY = "rhodawk_fed_registry"   # stores peer records
_PUSH_TIMEOUT_S = 10.0
_PULL_TIMEOUT_S = 15.0
_MAX_PUSH_RETRIES = 2


@dataclass
class FederatedPattern:
    """A single anonymized fix pattern in the federation."""
    fingerprint:      str   = ""         # SHA-256 of normalized_text
    normalized_text:  str   = ""
    issue_type:       str   = ""
    language:         str   = "python"
    complexity_score: float = 0.0
    source_instance:  str   = ""         # opaque deployment ID (not reversible)
    contributed_at:   str   = ""
    federation_score: float = 0.0        # similarity score from retrieval
    use_count:        int   = 0          # times this pattern was applied
    success_count:    int   = 0          # times application led to passing tests


@dataclass
class FederationPeer:
    """A known peer registry deployment."""
    id:          str  = ""
    url:         str  = ""
    name:        str  = ""               # human-readable label (optional)
    last_seen:   str  = ""
    pattern_count: int = 0
    trust_level: str  = "peer"           # "peer" | "hub" | "client"
    active:      bool = True


@dataclass
class FederationStats:
    """Runtime statistics for federation monitoring."""
    patterns_contributed: int   = 0
    patterns_received:    int   = 0
    patterns_applied:     int   = 0
    push_failures:        int   = 0
    pull_failures:        int   = 0
    active_peers:         int   = 0
    cache_size:           int   = 0


class FederatedPatternStore:
    """
    Manages federated pattern sharing across Rhodawk deployments.

    Parameters
    ----------
    instance_id : str
        Stable opaque ID for this deployment.  Hashed before transmission
        so peers cannot identify the origin deployment.
    registry_url : str
        URL of the hub or primary peer registry.  Leave empty for isolated
        (local-only) operation.
    qdrant_url : str
        Qdrant server URL for the local federated-pattern cache.
    contribute : bool
        Push locally learned patterns to the federation.
    receive : bool
        Pull patterns from the federation to augment local retrieval.
    min_complexity : float
        Only push patterns with complexity_score >= min_complexity.
        Low-complexity patterns are structurally trivial and add noise.
    data_dir : Path | None
        JSON fallback directory when Qdrant is unavailable.
    """

    def __init__(
        self,
        instance_id:     str         = "",
        registry_url:    str         = "",
        qdrant_url:      str         = "http://localhost:6333",
        contribute:      bool        = True,
        receive:         bool        = True,
        min_complexity:  float       = 0.15,
        data_dir:        Path | None = None,
        extra_peer_urls: list[str]   = None,
    ) -> None:
        self.instance_id    = instance_id or str(uuid.uuid4())[:16]
        # Opaque sender ID — SHA-256 of instance_id so peers cannot map it back
        self._sender_hash   = hashlib.sha256(
            self.instance_id.encode()
        ).hexdigest()[:24]
        self.registry_url   = registry_url.rstrip("/") if registry_url else ""
        self.qdrant_url     = qdrant_url
        self.contribute     = contribute
        self.receive        = receive
        self.min_complexity = min_complexity
        self.data_dir       = data_dir
        self.extra_peer_urls: list[str] = [
            u.rstrip("/") for u in (extra_peer_urls or []) if u
        ]

        self._qdrant_client: Any     = None
        self._backend:       str     = "none"
        self._json_path:     Path | None = None
        self._peers:         list[FederationPeer] = []
        self._stats          = FederationStats()
        self._session:       Any | None = None   # aiohttp.ClientSession

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialise(self) -> None:
        """Connect to Qdrant cache and discover peers."""
        await self._init_cache()
        await self._load_peers()
        log.info(
            f"FederatedPatternStore: backend={self._backend} "
            f"contribute={self.contribute} receive={self.receive} "
            f"peers={len(self._peers)} registry={self.registry_url or 'none'}"
        )

    async def close(self) -> None:
        """Release HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ── Push (contribute) ─────────────────────────────────────────────────────

    async def push_pattern(
        self,
        fingerprint:      str,
        normalized_text:  str,
        issue_type:       str,
        language:         str   = "python",
        complexity_score: float = 0.0,
    ) -> bool:
        """
        Push an anonymized fix pattern to the federation.

        Returns True if at least one peer accepted the pattern.
        Failures are logged but never propagated — federation is always
        best-effort and must not block the main fix pipeline.
        """
        if not self.contribute:
            return False
        if complexity_score < self.min_complexity:
            log.debug(
                f"FederatedStore.push: complexity {complexity_score:.3f} < "
                f"min {self.min_complexity} — skipping push"
            )
            return False

        # Store in local cache first (enables offline replay)
        await self._cache_pattern(FederatedPattern(
            fingerprint      = fingerprint,
            normalized_text  = normalized_text,
            issue_type       = issue_type,
            language         = language,
            complexity_score = complexity_score,
            source_instance  = self._sender_hash,
            contributed_at   = datetime.now(timezone.utc).isoformat(),
        ))

        payload = {
            "fingerprint":      fingerprint,
            "normalized_text":  normalized_text,
            "issue_type":       issue_type,
            "language":         language,
            "complexity_score": complexity_score,
            "sender_hash":      self._sender_hash,
            "contributed_at":   datetime.now(timezone.utc).isoformat(),
        }

        # Push to registry hub (if configured) + extra peers
        targets: list[str] = []
        if self.registry_url:
            targets.append(self.registry_url)
        targets.extend(self.extra_peer_urls)
        for peer in self._peers:
            if peer.active and peer.url and peer.url not in targets:
                targets.append(peer.url)

        if not targets:
            return True  # local-only — cache is the federation

        accepted = 0
        tasks = [
            self._push_to_peer(url, payload) for url in targets
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if r is True:
                accepted += 1
            elif isinstance(r, Exception):
                self._stats.push_failures += 1
                log.debug(f"FederatedStore.push: peer error — {r}")

        if accepted > 0:
            self._stats.patterns_contributed += 1
        return accepted > 0

    # ── Pull (receive) ────────────────────────────────────────────────────────

    async def pull_patterns(
        self,
        issue_type:  str,
        query_text:  str = "",
        n:           int = 5,
    ) -> list[FederatedPattern]:
        """
        Retrieve federated patterns relevant to the given issue.

        First checks local cache, then fetches from peers if cache is sparse.
        Results are sorted by (success_count / use_count) × 0.6 + complexity_score × 0.4 descending.
        This matches the architecture spec and rewards patterns with high empirical
        success rates, not just high retrieval similarity.
        """
        if not self.receive:
            return []

        # 1. Check local cache
        cached = await self._retrieve_from_cache(issue_type, n * 2)

        # 2. Supplement from peers if cache has fewer than n patterns
        if len(cached) < n:
            fresh = await self._pull_from_peers(issue_type, query_text, n)
            # Merge + deduplicate by fingerprint
            seen: set[str] = {p.fingerprint for p in cached}
            for p in fresh:
                if p.fingerprint not in seen:
                    cached.append(p)
                    seen.add(p.fingerprint)
                    await self._cache_pattern(p)

        # 3. Sort by quality signal
        cached.sort(
            key=lambda p: (p.success_count / max(p.use_count, 1)) * 0.6 + p.complexity_score * 0.4,
            reverse=True,
        )
        if cached:
            self._stats.patterns_received += len(cached)
        return cached[:n]

    # ── Peer management ───────────────────────────────────────────────────────

    async def register_peer(self, url: str, name: str = "", trust: str = "peer") -> FederationPeer:
        """Add a new peer to the local registry."""
        peer = FederationPeer(
            id          = hashlib.sha256(url.encode()).hexdigest()[:16],
            url         = url.rstrip("/"),
            name        = name or url,
            last_seen   = datetime.now(timezone.utc).isoformat(),
            trust_level = trust,
            active      = True,
        )
        # Avoid duplicates
        existing_ids = {p.id for p in self._peers}
        if peer.id not in existing_ids:
            self._peers.append(peer)
            await self._persist_peers()
        return peer

    async def deregister_peer(self, peer_id: str) -> bool:
        """Deactivate a peer (does not delete from registry — preserves audit trail)."""
        for peer in self._peers:
            if peer.id == peer_id:
                peer.active = False
                await self._persist_peers()
                return True
        return False

    def get_peers(self) -> list[FederationPeer]:
        return list(self._peers)

    def get_stats(self) -> FederationStats:
        self._stats.active_peers = sum(1 for p in self._peers if p.active)
        return self._stats

    async def record_usage(self, fingerprint: str, success: bool) -> bool:
        """
        Increment use_count (always) and success_count (when success=True)
        for the pattern identified by fingerprint.

        FIX (Defect 3): This method is the missing feedback loop for GAP 6.
        use_count and success_count were defined on FederatedPattern,
        propagated through pull/push, and displayed in formatted output —
        but nothing ever wrote back to them.  Without this method being
        called, quality-based ranking in serve_patterns was permanently
        frozen at 0/0 for every pattern.

        Returns True if the pattern was found and updated, False if the
        fingerprint does not exist in the local cache (caller should 404).
        """
        if self._backend == "qdrant" and self._qdrant_client:
            return await self._record_usage_qdrant(fingerprint, success)
        elif self._backend == "json" and self._json_path:
            return self._record_usage_json(fingerprint, success)
        return False

    async def fingerprint_exists(self, fingerprint: str) -> bool:
        """
        O(1) check whether a fingerprint is already in the local cache.

        Uses a direct Qdrant point lookup by UID (computed identically to
        _cache_pattern) rather than scanning all stored patterns.  This
        replaces the O(N) _retrieve_from_cache("", n=10_000) scan that
        receive_pattern previously used for deduplication — critical at
        scale where thousands of patterns may be stored.

        Falls back to a linear JSON scan when Qdrant is unavailable.
        Always returns False on any error rather than propagating exceptions,
        so the caller can treat any error as a cache miss and proceed.
        """
        if self._backend == "qdrant" and self._qdrant_client:
            try:
                uid = abs(hash(fingerprint)) % (10 ** 9)
                results = self._qdrant_client.retrieve(
                    collection_name=_FED_COLLECTION,
                    ids=[uid],
                    with_payload=True,
                )
                if not results:
                    return False
                # Verify payload fingerprint to guard against hash collisions
                stored_fp = (results[0].payload or {}).get("fingerprint", "")
                return stored_fp == fingerprint
            except Exception as exc:
                log.debug(f"FederatedStore.fingerprint_exists (qdrant): {exc}")
                return False

        if self._backend == "json":
            records = self._json_load_patterns()
            return any(r.get("fingerprint") == fingerprint for r in records)

        return False

    async def _record_usage_qdrant(self, fingerprint: str, success: bool) -> bool:
        """Update use_count / success_count for a pattern stored in Qdrant."""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore

            # Locate the point by fingerprint payload field
            uid = abs(hash(fingerprint)) % (10 ** 9)
            results = self._qdrant_client.retrieve(
                collection_name=_FED_COLLECTION,
                ids=[uid],
                with_payload=True,
            )
            if not results:
                return False

            point   = results[0]
            payload = dict(point.payload or {})

            # Guard: verify fingerprint matches (hash collision defence)
            if payload.get("fingerprint") != fingerprint:
                return False

            payload["use_count"]     = int(payload.get("use_count", 0)) + 1
            if success:
                payload["success_count"] = int(payload.get("success_count", 0)) + 1

            from qdrant_client.models import PointStruct  # type: ignore
            vec = self._embed(payload.get("normalized_text", fingerprint))
            self._qdrant_client.upsert(
                collection_name=_FED_COLLECTION,
                points=[PointStruct(id=uid, vector=vec, payload=payload)],
            )
            self._stats.patterns_applied += 1
            log.debug(
                f"FederatedStore.record_usage (qdrant): fingerprint="
                f"{fingerprint[:16]}... use={payload['use_count']} "
                f"success={payload.get('success_count', 0)}"
            )
            return True
        except Exception as exc:
            log.debug(f"FederatedStore._record_usage_qdrant: {exc}")
            return False

    def _record_usage_json(self, fingerprint: str, success: bool) -> bool:
        """Update use_count / success_count for a pattern stored in the JSON fallback."""
        if not self._json_path or not self._json_path.exists():
            return False
        try:
            records = self._json_load_patterns()
            found   = False
            for record in records:
                if record.get("fingerprint") == fingerprint:
                    record["use_count"]     = int(record.get("use_count", 0)) + 1
                    if success:
                        record["success_count"] = int(record.get("success_count", 0)) + 1
                    found = True
                    break
            if not found:
                return False
            self._json_path.write_text(
                json.dumps(records, indent=2), encoding="utf-8"
            )
            self._stats.patterns_applied += 1
            log.debug(
                f"FederatedStore.record_usage (json): fingerprint="
                f"{fingerprint[:16]}... updated"
            )
            return True
        except Exception as exc:
            log.debug(f"FederatedStore._record_usage_json: {exc}")
            return False

    # ── Internal: peer HTTP ───────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=_PUSH_TIMEOUT_S)
            )
        return self._session

    async def _push_to_peer(self, base_url: str, payload: dict) -> bool:
        """POST a normalized pattern to a peer registry endpoint."""
        url = f"{base_url}/api/federation/patterns"
        for attempt in range(_MAX_PUSH_RETRIES):
            try:
                session = await self._get_session()
                async with session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=_PUSH_TIMEOUT_S),
                ) as resp:
                    if resp.status in (200, 201, 409):
                        # 409 = Conflict = fingerprint already known (OK)
                        return True
                    body = await resp.text()
                    log.debug(
                        f"FederatedStore: push to {url} status={resp.status} body={body[:200]}"
                    )
                    return False
            except asyncio.TimeoutError:
                log.debug(f"FederatedStore: push timeout to {url} (attempt {attempt+1})")
            except Exception as exc:
                log.debug(f"FederatedStore: push error to {url}: {exc}")
                if attempt >= _MAX_PUSH_RETRIES - 1:
                    raise
                await asyncio.sleep(0.5)
        return False

    async def push_usage_feedback(
        self,
        fingerprint: str,
        success:     bool,
    ) -> None:
        """
        Report usage feedback for a federated pattern back to the peer
        registries that originally served it.

        FIX (Defect 3 - network layer): Closes the feedback loop at the
        network level.  When this deployment applies a pattern received from
        a remote registry and the test result is known, this method POSTs
        back to every peer so their use_count / success_count are updated —
        not just our local cache.

        Fire-and-forget: failures are logged but never propagated.
        """
        if not self.contribute:
            return

        targets: list[str] = []
        if self.registry_url:
            targets.append(self.registry_url)
        targets.extend(self.extra_peer_urls)
        for peer in self._peers:
            if peer.active and peer.url and peer.url not in targets:
                targets.append(peer.url)

        if not targets:
            return

        payload = {"success": success}
        tasks = [
            self._post_usage_to_peer(url, fingerprint, payload)
            for url in targets
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for url, r in zip(targets, results):
            if isinstance(r, Exception):
                log.debug(
                    f"FederatedStore.push_usage_feedback: peer {url} — {r}"
                )

    async def _post_usage_to_peer(
        self, base_url: str, fingerprint: str, payload: dict
    ) -> None:
        """POST usage feedback to a single peer registry endpoint."""
        url = f"{base_url}/api/federation/patterns/{fingerprint}/usage"
        try:
            session = await self._get_session()
            async with session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=_PUSH_TIMEOUT_S),
            ) as resp:
                if resp.status not in (200, 404):
                    body = await resp.text()
                    log.debug(
                        f"FederatedStore._post_usage_to_peer: {url} "
                        f"status={resp.status} body={body[:200]}"
                    )
        except Exception as exc:
            log.debug(f"FederatedStore._post_usage_to_peer {url}: {exc}")

    async def _pull_from_peers(
        self, issue_type: str, query_text: str, n: int
    ) -> list[FederatedPattern]:
        """GET patterns from all active peers."""
        if not self.receive:
            return []

        targets: list[str] = []
        if self.registry_url:
            targets.append(self.registry_url)
        targets.extend(self.extra_peer_urls)
        for peer in self._peers:
            if peer.active and peer.url and peer.url not in targets:
                targets.append(peer.url)

        if not targets:
            return []

        tasks = [
            self._pull_from_peer(url, issue_type, query_text, n)
            for url in targets
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        patterns: list[FederatedPattern] = []
        seen_fps: set[str] = set()
        for r in results:
            if isinstance(r, Exception):
                self._stats.pull_failures += 1
                log.debug(f"FederatedStore.pull error: {r}")
                continue
            for p in r:
                if p.fingerprint not in seen_fps:
                    patterns.append(p)
                    seen_fps.add(p.fingerprint)

        return patterns

    async def _pull_from_peer(
        self, base_url: str, issue_type: str, query_text: str, n: int
    ) -> list[FederatedPattern]:
        url = f"{base_url}/api/federation/patterns"
        params: dict[str, Any] = {"issue_type": issue_type, "n": n}
        if query_text:
            params["q"] = query_text[:200]

        try:
            session = await self._get_session()
            async with session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=_PULL_TIMEOUT_S),
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                patterns: list[FederatedPattern] = []
                for item in data.get("patterns", []):
                    patterns.append(FederatedPattern(
                        fingerprint      = item.get("fingerprint", ""),
                        normalized_text  = item.get("normalized_text", ""),
                        issue_type       = item.get("issue_type", ""),
                        language         = item.get("language", "python"),
                        complexity_score = float(item.get("complexity_score", 0)),
                        source_instance  = item.get("sender_hash", ""),
                        contributed_at   = item.get("contributed_at", ""),
                        federation_score = float(item.get("relevance_score", 0)),
                        use_count        = int(item.get("use_count", 0)),
                        success_count    = int(item.get("success_count", 0)),
                    ))

                # Update pattern_count on the matching peer so list_peers()
                # returns a meaningful number rather than the permanent 0 it
                # had before (the field was declared but never incremented).
                registry_info = data.get("registry", {})
                total_reported = int(registry_info.get("total_patterns", 0))
                if total_reported > 0:
                    normalised_base = base_url.rstrip("/")
                    for peer in self._peers:
                        if peer.url.rstrip("/") == normalised_base:
                            peer.pattern_count = total_reported
                            peer.last_seen = datetime.now(timezone.utc).isoformat()
                            break

                return patterns
        except asyncio.TimeoutError:
            log.debug(f"FederatedStore: pull timeout from {url}")
            return []
        except Exception as exc:
            log.debug(f"FederatedStore: pull error from {url}: {exc}")
            return []

    # ── Internal: Qdrant cache ────────────────────────────────────────────────

    async def _init_cache(self) -> None:
        """Initialise the local federated-pattern cache."""
        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client.models import Distance, VectorParams  # type: ignore

            client = QdrantClient(url=self.qdrant_url, timeout=5)
            client.get_collections()

            existing = [c.name for c in client.get_collections().collections]
            if _FED_COLLECTION not in existing:
                client.create_collection(
                    collection_name=_FED_COLLECTION,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
            self._qdrant_client = client
            self._backend = "qdrant"
        except Exception as exc:
            log.debug(f"FederatedStore: Qdrant unavailable ({exc}) — using JSON fallback")
            self._backend = "json"
            if self.data_dir:
                self._json_path = self.data_dir / "federated_patterns.json"

    async def _cache_pattern(self, pattern: FederatedPattern) -> None:
        """Store a received pattern in the local cache for offline access."""
        if self._backend == "qdrant" and self._qdrant_client:
            try:
                from qdrant_client.models import PointStruct  # type: ignore
                vec   = self._embed(pattern.normalized_text)
                uid   = abs(hash(pattern.fingerprint)) % (10 ** 9)
                self._qdrant_client.upsert(
                    collection_name=_FED_COLLECTION,
                    points=[PointStruct(
                        id      = uid,
                        vector  = vec,
                        payload = asdict(pattern),
                    )],
                )
            except Exception as exc:
                log.debug(f"FederatedStore._cache_pattern Qdrant: {exc}")
        elif self._backend == "json" and self._json_path:
            try:
                records = self._json_load_patterns()
                fp_set  = {r.get("fingerprint") for r in records}
                if pattern.fingerprint not in fp_set:
                    records.append(asdict(pattern))
                    self._json_path.parent.mkdir(parents=True, exist_ok=True)
                    self._json_path.write_text(
                        json.dumps(records, indent=2), encoding="utf-8"
                    )
            except Exception as exc:
                log.debug(f"FederatedStore._cache_pattern JSON: {exc}")

    async def _retrieve_from_cache(
        self, issue_type: str, n: int
    ) -> list[FederatedPattern]:
        """Retrieve patterns from local cache filtered by issue_type."""
        if self._backend == "qdrant" and self._qdrant_client:
            try:
                from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore

                # Query with issue_type filter
                query_vec  = self._embed(issue_type)
                hits = self._qdrant_client.search(
                    collection_name=_FED_COLLECTION,
                    query_vector=query_vec,
                    limit=n,
                    query_filter=Filter(
                        must=[FieldCondition(
                            key="issue_type",
                            match=MatchValue(value=issue_type),
                        )]
                    ) if issue_type else None,
                )
                patterns: list[FederatedPattern] = []
                for hit in hits:
                    p = hit.payload or {}
                    patterns.append(FederatedPattern(
                        fingerprint      = p.get("fingerprint", ""),
                        normalized_text  = p.get("normalized_text", ""),
                        issue_type       = p.get("issue_type", ""),
                        language         = p.get("language", "python"),
                        complexity_score = float(p.get("complexity_score", 0)),
                        source_instance  = p.get("source_instance", ""),
                        contributed_at   = p.get("contributed_at", ""),
                        federation_score = float(hit.score),
                        use_count        = int(p.get("use_count", 0)),
                        success_count    = int(p.get("success_count", 0)),
                    ))
                return patterns
            except Exception as exc:
                log.debug(f"FederatedStore._retrieve_from_cache Qdrant: {exc}")

        # JSON fallback
        records = self._json_load_patterns()
        matches = [
            r for r in records
            if not issue_type or r.get("issue_type") == issue_type
        ]
        return [
            FederatedPattern(
                fingerprint      = r.get("fingerprint", ""),
                normalized_text  = r.get("normalized_text", ""),
                issue_type       = r.get("issue_type", ""),
                language         = r.get("language", "python"),
                complexity_score = float(r.get("complexity_score", 0)),
                source_instance  = r.get("source_instance", ""),
                contributed_at   = r.get("contributed_at", ""),
                use_count        = int(r.get("use_count", 0)),
                success_count    = int(r.get("success_count", 0)),
            )
            for r in matches[:n]
        ]

    # ── Internal: peer persistence ────────────────────────────────────────────

    async def _load_peers(self) -> None:
        """Load peer registry from JSON file on disk."""
        if not self.data_dir:
            return
        peer_path = self.data_dir / "federation_peers.json"
        if peer_path.exists():
            try:
                raw = json.loads(peer_path.read_text(encoding="utf-8"))
                self._peers = [FederationPeer(**p) for p in raw]
            except Exception as exc:
                log.debug(f"FederatedStore._load_peers: {exc}")

        # Auto-register configured registry as hub
        if self.registry_url:
            hub_id = hashlib.sha256(self.registry_url.encode()).hexdigest()[:16]
            if not any(p.id == hub_id for p in self._peers):
                self._peers.append(FederationPeer(
                    id          = hub_id,
                    url         = self.registry_url,
                    name        = "hub",
                    last_seen   = datetime.now(timezone.utc).isoformat(),
                    trust_level = "hub",
                    active      = True,
                ))

    async def _persist_peers(self) -> None:
        """Save peer registry to JSON file on disk."""
        if not self.data_dir:
            return
        peer_path = self.data_dir / "federation_peers.json"
        try:
            peer_path.parent.mkdir(parents=True, exist_ok=True)
            peer_path.write_text(
                json.dumps([asdict(p) for p in self._peers], indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            log.debug(f"FederatedStore._persist_peers: {exc}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _json_load_patterns(self) -> list[dict]:
        if not self._json_path or not self._json_path.exists():
            return []
        try:
            return json.loads(self._json_path.read_text(encoding="utf-8"))
        except Exception:
            return []

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

_instance: FederatedPatternStore | None = None


def get_federated_store(
    instance_id:     str         = "",
    registry_url:    str         = "",
    qdrant_url:      str         = "http://localhost:6333",
    contribute:      bool        = True,
    receive:         bool        = True,
    min_complexity:  float       = 0.15,
    data_dir:        Path | None = None,
    extra_peer_urls: list[str]   = None,
) -> FederatedPatternStore:
    """
    Module-level singleton factory.

    First call creates and configures the store.  Subsequent calls return the
    existing instance (allowing the initialise() coroutine to be awaited only
    once from the controller).
    """
    global _instance
    if _instance is None:
        _instance = FederatedPatternStore(
            instance_id     = instance_id,
            registry_url    = registry_url,
            qdrant_url      = qdrant_url,
            contribute      = contribute,
            receive         = receive,
            min_complexity  = min_complexity,
            data_dir        = data_dir,
            extra_peer_urls = extra_peer_urls or [],
        )
    return _instance
