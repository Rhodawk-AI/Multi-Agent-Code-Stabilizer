# RHODAWK AI CODE STABILIZER Architecture

## The Core Loop

```
READ → AUDIT → FIX → REVIEW → GATE → COMMIT → RE-AUDIT → repeat
```

Every phase is a separate agent with a single responsibility.
All agents share one thing: the brain.

## The Brain

The brain is a SQLite database (`.stabilizer/brain.db`) inside the target repo.

It stores:
- Every file's metadata, hash, and read status
- Every file chunk's extracted facts (symbols, deps, observations)
- Every discovered issue with full lifecycle tracking
- Every fix attempt with complete file content
- Every review decision with reasoning
- Every LLM session with cost
- Every patrol event

The brain is the system's memory across sessions. It is:
- **Persistent** — survives crashes and restarts
- **Queryable** — REST API exposes everything
- **Auditable** — complete causal chain for every fix
- **Incremental** — only re-reads files whose hash changed

## Hybrid Chunking

The key to handling million-line codebases without exhausting context windows.

```
< 200 lines     → FULL       (one shot)
200-1K lines    → HALF       (two halves, 20-line overlap)
1K-5K lines     → AST_NODES  (split at class/function boundaries)
5K-20K lines    → SKELETON   (skeleton + full content chunks)
> 20K lines     → SKELETON_ONLY (skeleton + targeted reads on demand)
```

The skeleton pass extracts imports, class/function signatures, and decorators.
This gives the auditor a structural map without reading every line.
Deep reads are triggered only when an issue is suspected in a specific region.

## Static Analysis Gate

Every LLM-generated fix passes through the gate before touching the repo:

1. **Syntax check** — ast.parse() or language parser
2. **ruff** — Python errors and security patterns
3. **bandit** — Python security vulnerabilities
4. **Invariant checks** — bare except, empty files, silent exception swallows

If the gate rejects a fix, it is never written to disk.
The issue goes back to the fixer with the gate's rejection reason.

## Multi-Model Strategy

Different phases use different models:

| Phase | Model | Why |
|---|---|---|
| File triage | Haiku / fast model | Cheap, just needs yes/no |
| Audit synthesis | Sonnet | Balance of depth and cost |
| Critical fixes | Opus / best model | Most expensive, only for CRITICAL |
| Cross-validation | Second provider | Independence — catches model-specific blindspots |

## Convergence Guarantee

The loop terminates because:

1. **Fingerprint dedup** — same issue seen 3+ times → escalated, never re-queued
2. **Stall detection** — no score improvement for 2 cycles → halt
3. **Regression detection** — score got worse → revert + halt
4. **Max cycles** — hard ceiling at 50 (configurable)
5. **Cost ceiling** — hard stop at $50 (configurable)

At least one condition always triggers.

## Trust Architecture

What makes RHODAWK AI CODE STABILIZER safe to run on production code:

1. **Never executes generated code** on the host machine
2. **Never pushes to main** — PRs only
3. **Architectural lock** — safety-critical files require human approval
4. **Full audit trail** — every decision logged with reasoning
5. **Static analysis gate** — syntax and security check before every commit
6. **Revert command** — undo all stabilizer commits since any timestamp
7. **Credential separation** — GitHub token and LLM tokens never in same env
