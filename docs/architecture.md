# MACS Architecture
## Multi-Agent Code Stabilizer — General Interactive Intelligence

---

## 1. The Core Loop

```
READ → GRAPH BUILD → AUDIT → CONSENSUS → FIX → REVIEW → GATE → COMMIT → TEST → RE-READ → repeat
```

Every phase is a separate agent with a single responsibility.
All agents share one thing: the **brain** (SQLite + optional vector store + dependency graph).

---

## 2. The Brain

The brain is a SQLite database (`.stabilizer/brain.db`) inside the target repo.

It stores:

- Every file's metadata, hash, read status, and load-bearing flag
- Every file chunk's extracted facts: symbols, dependencies, observations
- Every discovered issue with full lifecycle tracking and consensus metadata
- Every fix attempt with complete file content, gate result, and formal proof IDs
- Every review decision with per-decision verdicts and reasoning
- Every LLM session with cost and model used
- Every patrol event (stall detection, cost warnings, regression alerts)
- Every graph edge (dependency graph snapshot)
- Every formal verification result (Z3 proofs)
- Every test run result (post-fix test suite execution)
- Every audit trail entry (HMAC-signed, tamper-proof)

The brain is the system's memory across sessions:

- **Persistent** — survives crashes and restarts, resumes mid-run
- **Queryable** — full REST API exposes all data
- **Auditable** — every decision has a cryptographic HMAC chain
- **Incremental** — only re-reads files whose content hash changed
- **Auto-migrating** — new columns added via `ALTER TABLE … ADD COLUMN IF NOT EXISTS`

---

## 3. Hybrid Chunking

Key to handling million-line codebases without exhausting context windows:

```
< 200 lines     → FULL         (one shot, full source)
200–1K lines    → HALF         (two halves, 20-line overlap)
1K–5K lines     → AST_NODES    (split at class/function boundaries)
5K–20K lines    → SKELETON     (skeleton + full content chunks)
> 20K lines     → SKELETON_ONLY (skeleton + targeted reads on demand)
```

Chunks within a file are now processed **in parallel** (bounded by `chunk_concurrency`, default 4). A 25,000-line file with 32 chunks that previously took 32 sequential LLM calls now completes in ~8 passes.

The skeleton pass extracts imports, class/function signatures, and decorators. Deep reads are triggered only when an issue is suspected in a specific region.

---

## 4. Dependency Graph Engine

Built after every read phase from the `dependencies` arrays already stored in SQLite — no additional source parsing required.

**What it provides:**

- **Topological fix order**: files are fixed leaf-first, hub-last — so you never fix a caller before its callee
- **Impact radius**: after a commit, every file that transitively imports the changed file is re-queued for re-read and re-audit automatically (replaces the shallow `fix_requires_files` heuristic)
- **Betweenness centrality + PageRank**: hub files with high centrality get a raised consensus confidence floor, prioritised audit order, and stricter review
- **Parallel fix batch partitioning**: `non_overlapping_fix_batches()` finds groups of files with no shared membership so fixes can run in parallel safely

**Backends**: `networkx` (default, zero infrastructure) or `neo4j` (for very large repos).

---

## 5. Direct-Source Auditing

Previous versions audited from pre-computed file summaries. Logic errors the reader missed were invisible to the auditor.

Files **under 5,000 lines** are now audited from their actual source code. The auditor sees the real text, line by line. Files above the threshold use the summary + observations approach (still the only practical option at Binance scale), with an explicit note to the LLM.

---

## 6. Finding Validation (Hallucination Filter)

After each auditor runs, a lightweight second pass asks the LLM: "Is this finding actually present in the code?" Findings the validator rejects are **downgraded to INFO** (not discarded) so they remain visible for review. This eliminates phantom findings that would waste fix cycles.

---

## 7. Consensus Engine

All three auditors (Security, Architecture, Standards) now participate in a weighted voting mechanism before any finding proceeds to the fix phase.

**Default rules:**

| Severity | Min agents | Required domains | Confidence floor |
|----------|-----------|-----------------|-----------------|
| CRITICAL | 2 | SECURITY (required) | 0.85 |
| MAJOR    | 2 | any | 0.70 |
| MINOR    | 1 | any | 0.50 |
| INFO     | 1 | any | 0.00 |

- SECURITY votes are weighted **2×** for CRITICAL findings.
- ARCHITECTURE votes are weighted **1.5×** for CRITICAL and MAJOR findings.
- Files with betweenness centrality > 0.70 get their confidence floor multiplied by 1.20.

---

## 8. Multi-Language Static Analysis Gate

Every LLM-generated fix passes through the gate before touching the repo:

1. **Dangerous pattern check** — Universal: `eval()`, `exec()`, `os.system()`, `pickle.loads()`. Domain-specific: finance (float on price, MD5), medical (float on dose, alarm disable), military (malloc, goto, stdio).
2. **Syntax check** — Python: `ast.parse()`. All other languages: `tree-sitter-language-pack` covering Java, Go, Rust, C, C++, TypeScript, Kotlin, Ruby, PHP, Swift, and 10+ more. Falls back to heuristic bracket balance if tree-sitter is unavailable.
3. **Python tools** — ruff (errors + security), mypy (type errors), bandit (security vulnerabilities).
4. **Semgrep** — language-aware rules for Python, JavaScript, Java, Go, C/C++.
5. **Invariants** — bare `except:`, empty files, silent exception swallowing, TODO/FIXME in safety-critical paths.

If the gate rejects a fix, it is never written to disk and the issue is re-opened for the next cycle.

---

## 9. Formal Verification

For CRITICAL fixes in `finance`, `medical`, or `military` domain mode, a Z3 SMT solver attempts to mathematically prove invariants before committing:

- **Finance**: balance non-negative after every mutation, no float on monetary arithmetic, no MD5/SHA-1.
- **Medical**: dosage always positive, no null patient_id, alarm flags not set to False.
- **Military**: no malloc outside init, no goto (MISRA Rule 15.1), no stdio in handlers.

The LLM extracts Z3 constraints from the fixed code; Z3 checks them. If Z3 returns SAT (a counterexample exists), the fix is blocked and the counterexample is stored in the brain and displayed in the PR description.

Falls back to static pattern matching when Z3 is unavailable or the LLM cannot express the property as machine-checkable constraints.

---

## 10. Semantic Vector Store

When `vector_store_enabled = true`, every code chunk is indexed into a ChromaDB collection using `sentence-transformers/all-MiniLM-L6-v2` embeddings as it is read.

The fixer uses this to gather cross-file context: before generating a fix, it retrieves the top-N semantically similar code snippets from OTHER files. This lets the model fix a balance calculation in `billing.py` with awareness of the validation logic in `payment.py` without uploading the entire repo into the context window.

---

## 11. Post-Fix Test Runner

After each committed fix, the target repo's own test suite is run automatically. Supported frameworks are auto-detected:

| Framework | Detection indicator |
|-----------|-------------------|
| pytest | `pytest.ini`, `pyproject.toml [tool.pytest]`, `conftest.py`, `test_*.py` |
| unittest | Python fallback |
| jest | `jest.config.*` |
| mocha | `.mocharc.*` |
| go test | `go.mod` |
| cargo test | `Cargo.toml` |
| Maven | `pom.xml` |
| Gradle | `build.gradle` |

If tests fail after a fix is committed, a patrol event is logged and a warning is displayed on the dashboard. The controller does not automatically revert — it flags for human review.

All sensitive credentials are scrubbed from the subprocess environment before tests run.

---

## 12. Domain Mode Switching

MACS has five operating modes selectable via `MACS_DOMAIN_MODE` or `--domain-mode`:

| Mode | Standards enforced |
|------|--------------------|
| `general` | General best practices (default) |
| `finance` | PCI-DSS, Decimal-only arithmetic, atomic balance ops, no MD5/SHA-1 |
| `medical` | IEC 62304, HIPAA, dosage safety, alarm gate, patient_id never null |
| `military` | MISRA-C:2012, DO-178C, no malloc, no goto, no stdio, bounded loops |
| `embedded` | RTOS rules: no dynamic alloc after init, bounded loops, no stdio in ISRs |

Domain mode affects: auditor system prompts, static gate hard-deny patterns, formal verification property selection, and consensus required-domain rules.

---

## 13. Multi-Model Strategy

| Phase | Model | Rationale |
|-------|-------|-----------|
| File triage / validation | Haiku | Cheap; just needs binary signal |
| Audit synthesis | Sonnet | Depth + cost balance |
| CRITICAL fixes | Opus | Highest quality; only for CRITICAL |
| Cross-validation | Second provider | Independence catches model-specific blindspots |
| Formal constraint extraction | Haiku | Cheap; structured output, not creative |

---

## 14. PR Batching (Module-Grouped)

Fixes are grouped by their top-level module directory before PR creation. A repo with 150 issues in `agents/`, `brain/`, and `utils/` produces three PRs — not 150.

Each PR description includes: issues fixed, verification checklist (gate / planner / review / formal proof), and cycle number.

---

## 15. Convergence Guarantee

The loop terminates because at least one of these conditions always triggers:

1. **Fingerprint dedup** — same finding seen 3+ times → escalated, never re-queued
2. **Consensus filter** — findings without multi-agent agreement are blocked before fixing
3. **Stall detection** — no score improvement for 2 cycles → halt
4. **Regression detection** — score increased by >10% → halt
5. **Max cycles** — hard ceiling at 50 (configurable)
6. **Cost ceiling** — hard stop at $50 (configurable)
7. **Patrol agent** — background watchdog enforces all of the above asynchronously

---

## 16. Trust Architecture

What makes MACS safe to run on production code:

1. **Never executes generated code** on the host machine — static analysis only
2. **Never pushes to main** — PRs only, human merge required
3. **Load-bearing file escalation** — safety-critical files require human approval always
4. **Full HMAC-signed audit trail** — every decision logged with cryptographic chain
5. **Multi-layer static gate** — syntax + security + invariants before any file is written
6. **Formal verification** — mathematical proofs for critical invariants in mission-critical modes
7. **Consequence simulation** — PlannerAgent reasons about reversibility before every commit
8. **Credential separation** — GitHub token and LLM API keys never coexist in the same subprocess
9. **Path traversal protection** — every file path validated against repo root before read or write
10. **Test regression detection** — post-fix test suite run catches regressions immediately

---

## 17. File Structure

```
├── agents/
│   ├── base.py              BaseAgent: LLM calls, cost tracking, rate limiter
│   ├── auditor.py           3-domain auditor with direct source + hallucination filter
│   ├── fixer.py             Parallel fix engine with vector search context
│   ├── reviewer.py          Adversarial reviewer with tiered diff caps
│   ├── planner.py           Consequence simulation + reversibility classification
│   ├── patrol.py            Background watchdog
│   ├── reader.py            Parallel chunk reader + VectorBrain indexing
│   ├── formal_verifier.py   Z3 formal verification (NEW)
│   └── test_runner.py       Post-fix test suite runner (NEW)
├── brain/
│   ├── schemas.py           All Pydantic models including new GII schemas
│   ├── storage.py           Abstract storage interface
│   ├── sqlite_storage.py    Full SQLite implementation
│   ├── graph.py             Dependency graph engine — networkx (NEW)
│   └── vector_store.py      ChromaDB semantic search (NEW)
├── orchestrator/
│   ├── controller.py        Main loop: all agents wired end-to-end
│   ├── convergence.py       Stall/regression/stabilized detector
│   └── consensus.py         Weighted N-agent voting engine (NEW)
├── sandbox/
│   └── executor.py          Multi-language static analysis gate
├── config/
│   ├── loader.py            TOML + env config (class name fixed)
│   ├── default.toml         Full default configuration
│   └── prompts/
│       ├── base.md          Master audit specification
│       ├── adversarial.md   Adversarial robustness prompt
│       └── os_safety.md     OS safety rules
├── plugins/
│   ├── base.py              Plugin ABC + manager
│   └── builtins/
│       └── no_secrets.py    Credential pattern scanner
├── github_integration/
│   └── pr_manager.py        Batched PR creation
├── mcp_clients/
│   └── manager.py           MCP + direct I/O fallback
├── api/
│   ├── app.py               FastAPI + WebSocket dashboard
│   └── routes/              REST endpoints for runs, issues, fixes, files
├── scripts/
│   ├── cli.py               typer CLI (stabilize, audit, bootstrap, status, serve)
│   └── run_one_cycle.py     Single-cycle runner
└── tests/
    └── unit/
        ├── test_agents.py       Agent schema + gate unit tests
        ├── test_brain.py        SQLite CRUD + concurrency tests
        ├── test_chunking.py     Chunking strategy tests
        ├── test_convergence.py  Convergence detector tests
        ├── test_graph.py        Dependency graph engine tests (NEW)
        ├── test_consensus.py    Consensus engine tests (NEW)
        └── test_executor.py     Multi-language gate tests (extended)
```
