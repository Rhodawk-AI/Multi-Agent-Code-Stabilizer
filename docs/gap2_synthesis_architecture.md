# Gap 2: Multi-Auditor Synthesis Architecture

## Problem Statement

Before Gap 2, three `AuditorAgent` instances ran in parallel —
`SECURITY`, `ARCHITECTURE`, `STANDARDS` — all calling the same LLM with
different system prompts. On large codebases this produced three structural
defects:

| Defect | Effect |
|---|---|
| **Duplicate findings** | Same bug reported 3 times in different language |
| **Missed findings** | Each auditor stays in its lane; never combines signals |
| **No cross-auditor reasoning** | Security auditor never learns what architecture auditor found |

### Why this matters competitively

CodeRabbit and Greptile use a single-pass LLM scan. Their fundamental
limitation is they **cannot reason about a bug that requires combining
information across auditor domains**.

Example:

```
auth_bypass.py:47     — SECURITY: missing privilege check on admin endpoint
async_handler.py:112  — ARCHITECTURE: shared state accessed without lock
```

Neither of these findings is CRITICAL in isolation. But the combination
means: **race-condition-enabled privilege escalation** — an attacker
can win the race to bypass the auth check before the lock is acquired.

A single-domain auditor scanning `auth_bypass.py` sees a medium-severity
auth issue. A single-domain auditor scanning `async_handler.py` sees a
medium-severity concurrency issue. Only a synthesis pass seeing **both
outputs simultaneously** can identify the compound as a CRITICAL
cross-domain vulnerability.

---

## Architecture

```
AuditorAgent(SECURITY)     ─┐
                             │  raw findings (may have duplicates)
AuditorAgent(ARCHITECTURE) ─┼──────────────────────────────────────┐
                             │                                      │
AuditorAgent(STANDARDS)    ─┘                                      ▼
                                                          SynthesisAgent
                                                                   │
                                                  ┌────────────────┴──────────────────┐
                                                  │                                   │
                                           Step 1: Dedup                    Step 2: Compound
                                                  │                                   │
                                    ┌─────────────┴──────────┐        ┌──────────────┴──────────┐
                                    │                        │        │                         │
                              Fingerprint              Semantic       Domains        LLM compound
                              exact match              LLM pass       grouped        finding prompt
                                    │                        │        │                         │
                                    └──────────┬─────────────┘        └──────────┬──────────────┘
                                               │                                 │
                                         deduplicated                    CompoundFinding[]
                                           issues                         (materialised as
                                                                         Issue[SYNTHESIS])
                                               │                                 │
                                               └──────────────┬──────────────────┘
                                                              │
                                                    ConsensusEngine
                                                              │
                                                         FixerAgent
```

---

## What SynthesisAgent Does

### Step 1 — Fingerprint Deduplication (deterministic)

Every issue already has a fingerprint (`file_path:line_start:line_end:desc[:120]`
SHA-256). Issues with identical fingerprints are clustered; the best
representative is kept (higher severity > higher confidence > SECURITY domain).

This catches exact duplicates produced when all three auditors flag the
same obvious bug.

### Step 2 — Semantic Deduplication (LLM, temperature=0.0)

After fingerprint dedup, the remaining issues are batched (≤60 per call)
and sent to the synthesis LLM. The LLM identifies clusters of issues that
describe the **same root defect** from different angles — e.g. "missing
bounds check" (STANDARDS) and "buffer overflow" (SECURITY) on the same
line in the same function.

This catches semantic duplicates that fingerprinting cannot.

### Step 3 — Cross-Domain Compound Finding Detection (LLM, temperature=0.1)

The synthesis LLM receives all findings **grouped by auditor domain** and
is asked: "which combinations of findings from different domains create a
vulnerability more severe than any single finding?"

The synthesis model uses `critical_fix_model` by default — a **different
model family** from the auditors. Using a different family ensures
genuinely fresh cross-domain reasoning rather than the same model
repeating its own earlier conclusions.

Compound findings are classified by category:

| Category | Description |
|---|---|
| `SECURITY_ARCHITECTURE` | Race condition + auth bypass; SSRF through architectural path |
| `SECURITY_STANDARDS` | CWE violation made exploitable by missing CERT-C protection |
| `ARCHITECTURE_STANDARDS` | Rule violation breaks architectural invariant → system failure |
| `ALL_DOMAINS` | Rarest, most severe — defect chain spanning all three domains |

Each `CompoundFinding` includes:
- `amplification_factor` — how much more severe than worst individual finding (1.0–10.0)
- `fix_complexity` — `LOW | MEDIUM | HIGH | ARCHITECTURAL`
- `contributing_issue_ids` — the issues that combine to form it
- `domains_involved` — which auditor domains contributed
- `rationale` — LLM explanation of why the combination is uniquely dangerous

### Step 4 — Materialise as Issues

Every `CompoundFinding` is also persisted as a regular `Issue` with
`executor_type=SYNTHESIS`. This means it flows through the normal
consensus → fix → review pipeline with no changes to downstream agents.
The `CompoundFinding` record holds the cross-domain metadata; the `Issue`
record is what the `FixerAgent` and `ReviewerAgent` consume.

---

## Configuration

### config/default.toml

```toml
[synthesis]
enabled                = true   # master switch
dedup_enabled          = true   # fingerprint + LLM semantic dedup
compound_enabled       = true   # cross-domain compound detection
synthesis_model        = ""     # defaults to critical_fix_model
max_compound_findings  = 20     # cap per synthesis pass
```

### StabilizerConfig fields

| Field | Type | Default | Description |
|---|---|---|---|
| `synthesis_enabled` | bool | `True` | Master switch |
| `synthesis_dedup_enabled` | bool | `True` | Run deduplication |
| `synthesis_compound_enabled` | bool | `True` | Run compound detection |
| `synthesis_model` | str | `""` | Model for synthesis (defaults to `critical_fix_model`) |
| `synthesis_max_compound` | int | `20` | Max compound findings per pass |

### Environment variables

| Variable | Field |
|---|---|
| `RHODAWK_SYNTHESIS_MODEL` | `synthesis_model` |
| `RHODAWK_SYNTHESIS_ENABLED` | `synthesis_enabled` |
| `RHODAWK_COMPOUND_ENABLED` | `synthesis_compound_enabled` |
| `RHODAWK_SYNTHESIS_MAX` | `synthesis_max_compound` |

---

## Model Selection

The synthesis model **should be a different family** from the primary
auditors. If the auditors use Qwen2.5-Coder (Alibaba family), use:

```toml
# config/default.toml
[synthesis]
synthesis_model = "openrouter/deepseek/deepseek-coder-v2-0724"
# or for cloud deployment:
# synthesis_model = "claude-opus-4-20250514"
```

Using the same model family as the auditors reduces compound finding
quality — the model will tend to reproduce the same framing it used when
generating the individual findings rather than reasoning freshly across
domains.

---

## Failure Modes and Mitigations

| Failure | Mitigation |
|---|---|
| Synthesis LLM call fails | `fail-open`: original undeduped issues returned unchanged; no findings lost |
| Compound finding has < 2 valid contributing indices | Dropped silently with debug log |
| Synthesis crashes mid-pass | Exception caught at `_phase_audit` level; `_last_compound_findings = []` |
| Semantic dedup over-aggressively removes real issues | LLM prompt explicitly instructs to only cluster issues with the *same root defect*; false-positives become conservative (cluster not applied) |

---

## Output

After `run_audit_phase()`, the controller exposes:

```python
controller._last_audit_issues       # deduplicated issues + compound issues
controller._last_compound_findings  # CompoundFinding[] from this pass
```

The `SynthesisReport` (persisted to storage) records per-cycle:

```
raw_issue_count           = 47  # before any dedup
fingerprint_dedup_count   = 12  # removed by fingerprint pass
semantic_dedup_count      =  8  # removed by LLM semantic pass
final_issue_count         = 27  # after all dedup
compound_finding_count    =  4  # new cross-domain findings
compound_critical_count   =  3  # CRITICAL severity
```

---

## What This Beats

| Tool | Limitation | This system |
|---|---|---|
| CodeRabbit | Single-pass LLM scan | Multi-domain + synthesis |
| Greptile | Single-pass LLM scan | Multi-domain + synthesis |
| SonarQube | Single-domain rule engine | Cross-domain compound detection |
| Semgrep | Pattern matching, no compound reasoning | LLM compound synthesis |

The commercial tools cannot find compound cross-domain findings because
their architecture is a single-pass scan. This system's multi-agent
structure makes compound detection structurally possible; the
`SynthesisAgent` is the component that exploits that structure.
