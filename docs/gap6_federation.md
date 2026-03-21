# GAP 6: Federated Anonymized Fix-Pattern Store

> **Referenced by** `config/default.toml` under `[gap6]`.  
> Read this before setting `federation_enabled = true`.

---

## What It Is

Every time Rhodawk fixes a bug and the fix passes all gates, it stores the structural
pattern of that fix — not the source code, just the *shape* of the solution.  By default
this pattern store is local to the deployment (per-repo).

Federation allows multiple Rhodawk deployments to share those structural patterns
anonymously.  A fix learned from 10,000 repositories in the wild becomes available as a
few-shot example for your local fixer — without any of those repositories' source code
ever leaving their networks.

This is the open-source counter to CodeRabbit/Greptile's closed-source advantage of
having indexed thousands of codebases.

---

## Privacy Model

**Nothing identifiable leaves the deployment.**

Before any pattern is transmitted, it passes through `PatternNormalizer`:

| Input | Output |
|---|---|
| Variable names (`user_id`, `payment_obj`) | `var_str_0`, `var_0` |
| Function names (`process_payment`) | `func_0` |
| Class names (`PaymentService`) | `cls_0` |
| String literals (`"acme_corp"`) | `<str_literal>` |
| Numeric literals (`42`) | `<num_literal>` |
| Comments | stripped |
| Type annotations (`Optional[str]`) | preserved verbatim |
| Control flow keywords (`if`, `raise`, `return`) | preserved verbatim |

The structural fingerprint (SHA-256 of normalized text) enables cross-deployment
deduplication.  Two deployments that fix structurally identical bugs produce the
same fingerprint — the registry stores the pattern once.

The sender identity is hashed (`SHA-256(instance_id)[:24]`) before transmission.
Peers cannot reverse this to identify the originating deployment.

---

## Topology

Federation is peer-to-peer HTTP.  There is no required central server.

```
 Deployment A ←──────────────────────→ Deployment B
      │                                      │
      │    POST /api/federation/patterns     │
      │    GET  /api/federation/patterns     │
      │    POST /api/federation/patterns/    │
      │         {fp}/usage                   │
      │                                      │
      └──────────────── Hub ─────────────────┘
             (optional shared registry)
```

For **hub-and-spoke** deployments (recommended for enterprises):

1. Designate one Rhodawk instance as the hub (can be a dedicated lightweight instance).
2. Set `registry_url` on all other deployments to point at the hub.
3. The hub aggregates patterns from all spokes and serves them back.

For **pure peer-to-peer** (no hub), add peer URLs to `extra_peer_urls`.

For **local-only** (no network sharing), leave `registry_url` empty.  The local
Qdrant/JSON cache still accumulates patterns and they are available for
`pull_patterns()` from the local store.

---

## Enabling Federation

Edit `config/default.toml` (or set environment variables):

```toml
[gap6]
federation_enabled   = true       # master switch
contribute_patterns  = true       # push local patterns to peers
receive_patterns     = true       # pull peer patterns into local retrieval
registry_url         = "https://hub.your-company.internal"  # or "" for local-only
extra_peer_urls      = ""         # comma-separated additional peer URLs
instance_id          = ""         # auto-generated UUID if empty
min_complexity       = 0.15       # skip trivially simple patterns (0–1)
```

**Opt-in flags are independent:**

- `contribute_patterns = false, receive_patterns = true` — receive-only (consume without contributing)
- `contribute_patterns = true, receive_patterns = false` — contribute-only (donate without consuming)

---

## Security

### Authentication

By default the federation endpoints are open (suitable for private networks).

To require bearer-token auth on all inbound federation requests, set the environment
variable on the **receiving** deployment:

```bash
export RHODAWK_FED_TOKEN="your-secret-token"
```

Sending deployments do not need any token configuration — they just POST normally.
Token checking is server-side only.

For production deployments, place the federation API behind a reverse proxy (nginx,
Caddy) with TLS termination and optionally mTLS.

### Input Validation

Every inbound pattern is validated before storage:

- `fingerprint` must be a 64-char lowercase hex SHA-256.
- `normalized_text` is spot-checked for leaked identifiers (second line of defense
  after the normalizer).
- `complexity_score` must be in `[0.0, 1.0]`.
- Payload size is capped at 16 KB.

If a pattern arrives that appears to contain un-normalized identifiers (more than 10
suspect tokens matching the camelCase/snake_case heuristic), it is rejected with 422.

---

## API Reference

All endpoints are under `/api/federation/` and require
`Authorization: Bearer <token>` when `RHODAWK_FED_TOKEN` is set.

### `POST /api/federation/patterns`

Receive a normalized pattern from a peer.

**Request body:**

```json
{
  "fingerprint":      "<64-char sha256 hex>",
  "normalized_text":  "if var_0 is None: raise var_str_0",
  "issue_type":       "null_deref",
  "language":         "python",
  "complexity_score": 0.42,
  "sender_hash":      "<opaque 24-char hex>",
  "contributed_at":   "2026-03-21T15:00:00+00:00"
}
```

**Responses:**

| Status | Meaning |
|---|---|
| `201 Created` | Pattern accepted and stored |
| `409 Conflict` | Fingerprint already known — safe to ignore |
| `422 Unprocessable` | Validation failed (bad fingerprint, leaked identifiers, etc.) |
| `503 Service Unavailable` | Federation not enabled on this deployment |

---

### `GET /api/federation/patterns`

Retrieve patterns for peer consumption.

**Query parameters:**

| Parameter | Default | Description |
|---|---|---|
| `issue_type` | `""` | Filter by bug category |
| `q` | `""` | Free-text relevance query |
| `n` | `10` | Max results (1–50) |
| `lang` | `""` | Filter by language |

**Response:** JSON with `count`, `patterns` array, and `registry` metadata.

Results are ranked by quality: `(success_count / use_count) × 0.6 + complexity_score × 0.4`.

---

### `POST /api/federation/patterns/{fingerprint}/usage`

Report the outcome of applying a federated pattern.  This is the **feedback loop**
that drives quality-based ranking.  Call this after every fix attempt that used a
`[FEDERATED]` pattern from `FixMemory.retrieve()`.

**Request body:**

```json
{"success": true}
```

- `success: true` — the fix that used this pattern passed all tests and gates.
- `success: false` — the fix was attempted but failed validation/tests.

Both increment `use_count`.  Only `true` increments `success_count`.

**Responses:**

| Status | Meaning |
|---|---|
| `200 OK` | Counters updated |
| `404 Not Found` | Fingerprint not in local cache |
| `422 Unprocessable` | Malformed fingerprint |

---

### `GET /api/federation/status`

Returns federation health, stats, and peer list.

---

### `POST /api/federation/peers`

Register a new peer.  Body: `{"url": "https://...", "name": "optional-label"}`.

---

### `DELETE /api/federation/peers/{peer_id}`

Deactivate a peer (sets `active=False`; does not delete from registry).

---

### `GET /api/federation/peers`

List all known peers (active and inactive).

---

## How It Integrates With the Fix Pipeline

```
store_success() called (fix passed all gates)
    │
    ├── Local FixMemory persists the raw pattern
    │
    └── _schedule_federation_push()
            │
            ├── PatternNormalizer strips all identifiers
            ├── FederatedPatternStore.push_pattern() caches locally
            └── POSTs to all registered peers / hub

retrieve() called (fixer needs few-shot examples)
    │
    ├── Local FixMemory results (top n)
    │
    └── _retrieve_federated()
            │
            └── FederatedPatternStore.pull_patterns()
                    ├── Checks local Qdrant cache
                    └── Fetches from peers if cache is sparse
                    → merged into results as [FEDERATED] entries

Fix succeeds / fails → gate evaluation complete
    │
    └── _report_federated_usage()
            │
            ├── FixMemory.record_federated_usage(fingerprint, success)
            │       ├── FederatedPatternStore.record_usage() — updates local cache
            │       └── FederatedPatternStore.push_usage_feedback() — notifies peers
            └── Peers update their use_count / success_count for this pattern
```

---

## Quality Ranking

Patterns in the federation accumulate a success rate over time:

```
quality_score = (success_count / max(use_count, 1)) × 0.6
              + complexity_score × 0.4
```

New patterns start with `use_count=0, success_count=0` and are ranked by
complexity alone until they accumulate real-world signal.  Over thousands of
fix attempts across the network, high-quality patterns rise to the top and
are preferentially returned to the fixer.

This is the compounding advantage: each deployment's successes and failures
improve the pattern store for every other deployment.

---

## Local-Only Operation (No Federation)

If `registry_url` is empty and `extra_peer_urls` is empty, the system operates
in local-only mode.  `push_pattern()` still normalizes and stores patterns in
the local Qdrant/JSON cache, and `pull_patterns()` retrieves from that local
cache.  The federation feedback loop (`/usage` endpoint) still works locally.

This is the default configuration and requires no network access.

---

## Troubleshooting

**Patterns not appearing in retrieval**

Check that `receive_patterns = true` and that the Qdrant collection
`rhodawk_fed_patterns` is accessible.  Run `GET /api/federation/status` on the
peer to verify pattern count.

**Push failures**

Federation failures are always logged at DEBUG level and never block the fix
pipeline.  Check logs for `FederatedStore: push error` messages.  Verify the
peer URL is reachable and the auth token (if any) is correct.

**Privacy audit**

To audit what leaves the deployment, set log level to DEBUG and watch for
`FixMemory.federation_push: pushed fingerprint=...` lines.  The `normalized_text`
in those log lines is exactly what is transmitted to peers.

**Disabling federation without restart**

Set `gap6_federation_enabled = false` in the config and restart.  No data is
deleted from the local cache — re-enabling resumes contribution and consumption
from where it left off.
