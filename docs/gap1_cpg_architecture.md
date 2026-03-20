# Gap 1: CPG-Based Causal Context Architecture

## Overview

Gap 1 is the most important architectural change in Rhodawk AI Code Stabilizer.
Every other gap is downstream of this one.

At 10M lines, no model sees the whole codebase. The question is: **which 2,000 lines
out of 10,000,000 are relevant to this bug?**

Before Gap 1: **vector similarity** — finds code that *looks like* the bug.  
After Gap 1: **CPG backward slicing** — finds code that *caused* the bug.

---

## The Causality Problem

Vector similarity finds semantically similar code. The PDG finds causally related code.
These two sets diverge completely at cross-domain boundaries.

**Example:**
```
null dereference  →  payment_service.py:47   (crash location)
caused by            auth_middleware.py:112  (returns None instead of User)
introduced by        user_model.py:89        (refactor 3 commits ago)
```

None of these files are semantically similar. An embedding model will never find
`auth_middleware.py` by searching for "code related to payment_service.py".
The CPG data flow graph connects them in milliseconds.

---

## Architecture

```
Bug / Issue Location
       │
       ▼
CPGContextSelector.select_context_for_issues()      ← NEW (Gap 1)
       │
       ├─── Joern available? ────────────────────────────────────────────┐
       │    YES                                                          │
       │    ProgramSlicer.compute_backward_slice()                       │
       │      │                                                          │
       │      ├── CPGEngine.compute_context_slice()                      │
       │      │     │                                                    │
       │      │     ├── JoernClient.get_callers(depth=3)     ← call graph│
       │      │     ├── JoernClient.compute_backward_slice() ← PDG slice │
       │      │     └── JoernClient.get_data_flows_to()      ← data flow │
       │      │                                                          │
       │      └── SliceResult: {files_in_slice, line_ranges, header}     │
       │                                                                  │
       ├─── Joern unavailable? ──────────────────────────────────────────┘
       │    YES: fall back to networkx DependencyGraphEngine
       │    (file-level impact_radius — lower quality but always works)
       │
       └─── Neither available?
            Fall back to HybridRetriever (BM25 + dense vectors)
            (semantic similarity — same as before Gap 1)

       │
       ▼
ContextSlice: {cpg_header, file_excerpts, context_text}
       │
       ▼
FixerAgent._generate_full_fix() / _generate_patch_fix()
  Injects CPG context BEFORE file content in the LLM prompt:
    1. Repo map (global symbol layout)
    2. Fix memory few-shot examples
    3. **CPG causal context** ← NEW (Gap 1) — injected here
    4. File contents
    5. Vector similarity context (still used for few-shot)
```

---

## Components

### `cpg/joern_client.py` — Joern HTTP Client

Pure protocol adapter for the Joern server REST API.
All public methods return empty results when Joern is unavailable — never raise.

Key queries:
- `get_callers(fn, depth)` — call graph traversal
- `get_callees(fn, depth)` — downstream dependency detection
- `compute_backward_slice(fn, var, line)` — PDG backward slice
- `get_data_flows_to_function(sink, source)` — cross-domain taint tracking
- `compute_impact_set(fns, depth)` — Gap 4 audit target computation

### `cpg/cpg_engine.py` — CPG Engine

Business logic layer above JoernClient. Owns:
- In-memory cache (TTL=1h, invalidated per commit)
- Graceful fallback to networkx when Joern unavailable
- `compute_context_slice()` — the Gap 1 core operation
- `compute_blast_radius()` — Gap 3 fix: replaces heuristic risk_score in PlannerAgent

### `cpg/program_slicer.py` — Program Slicer

Converts CPGEngine results into `SliceResult` objects with:
- LLM-injectable context headers (`as_context_header()`)
- Line ranges per file (loads only relevant code, not full files)
- Support for backward, forward, and chop slices

### `cpg/context_selector.py` — Context Selector

The Gap 1 fix applied to FixerAgent. Replaces `_get_vector_context()` for
context selection with `_get_cpg_context()`.

**Key design decision**: vector search is NOT removed — it is moved to a
secondary role for pattern matching and few-shot examples. CPG handles
the primary context selection.

### `cpg/incremental_updater.py` — Incremental Updater

Gap 4 integration point. After every commit:
1. Parses the diff to find changed functions
2. Invalidates CPG cache for those functions
3. Triggers Joern incremental file update
4. Computes CPG impact set (50–200 functions)
5. Creates `FunctionStalenessMark` records for the audit targets

The output is the Gap 4 audit scheduler's input: instead of auditing 10M
lines, audit only the 50–200 functions in the impact set.

### `tools/servers/joern_server.py` — MCP Server

Exposes all CPG operations as MCP tools for agent use:
- `cpg_backward_slice` — root cause analysis
- `cpg_impact_set` — commit impact (Gap 4)
- `cpg_blast_radius` — fix risk assessment (Gap 3)
- `cpg_data_flows` — cross-domain taint tracking
- `cpg_vulnerability_scan` — CPG-based vuln detection

---

## Joern Setup

### Docker (recommended)

```bash
# Add JOERN_REPO_PATH to .env
echo "JOERN_REPO_PATH=/path/to/your/repo" >> .env

# Start Joern
docker-compose up joern

# Verify
curl http://localhost:8080/api/v1/projects
```

### First-time CPG build

On first run, Rhodawk automatically imports the codebase into Joern.
The build time depends on codebase size:

| Codebase size | Build time | RAM needed |
|---------------|-----------|------------|
| < 100K lines  | 30-60s    | 2GB        |
| 1M lines      | 2-5 min   | 4GB        |
| 10M lines     | 15-40 min | 8-10GB     |

Subsequent runs use incremental updates (seconds per commit).

### Memory configuration

The Joern container is configured with `mem_limit: 10g` in docker-compose.yml.
Adjust based on your codebase size. The `--max-heap-size 8g` JVM flag controls
how much of that the JVM can use.

---

## Integration Points in Existing Code

### FixerAgent (`agents/fixer.py`)

New constructor parameters:
- `cpg_engine` — CPGEngine instance
- `cpg_context_selector` — CPGContextSelector instance
- `program_slicer` — ProgramSlicer instance

New method:
- `_get_cpg_context(issues)` — returns CPG context string for prompt injection

Modified methods:
- `_generate_full_fix()` — accepts `cpg_context` param, injects before file content
- `_generate_patch_fix()` — same

### PlannerAgent (`agents/planner.py`)

New constructor parameter:
- `cpg_engine` — CPGEngine instance

Modified method:
- `evaluate()` — blends CPG blast radius (40%) into LLM risk score (60%)
  If blast radius > 50 functions → requires_human_review = True

### ReaderAgent (`agents/reader.py`)

New constructor parameter:
- `cpg_engine` — CPGEngine instance

Modified method:
- `run()` — triggers Joern CPG initialise/invalidate after read pass

### StabilizerController (`orchestrator/controller.py`)

New config fields (StabilizerConfig):
- `cpg_enabled`, `joern_url`, `joern_repo_path`, `joern_project_name`
- `cpg_blast_radius_threshold`, `cpg_max_slice_nodes`, `cpg_max_files_in_slice`

New instance fields:
- `_cpg_engine`, `_program_slicer`, `_cpg_context_selector`, `_incremental_updater`

New methods:
- `_init_cpg()` — initialises all CPG subsystems

Modified methods:
- `run_read_phase()` — passes cpg_engine to ReaderAgent
- `_phase_fix()` — passes CPG subsystems to FixerAgent
- `_phase_gate()` — passes cpg_engine to PlannerAgent
- `_phase_commit()` — runs IncrementalCPGUpdater after commit
- `_cleanup()` — closes Joern connection

---

## Fallback Hierarchy

The system ALWAYS works without Joern. The fallback hierarchy is:

```
1. Joern CPG (Joern server running)
   → backward slice, call graph, data flow
   → most accurate, cross-domain causality

2. NetworkX DependencyGraphEngine (always available)
   → file-level impact_radius via import graph
   → file-level accuracy only (no function-level)

3. HybridRetriever (BM25 + dense, Qdrant running)
   → semantic similarity + keyword matching
   → same quality as before Gap 1

4. VectorBrain (dense only)
   → semantic similarity only
   → baseline quality

5. Empty context
   → LLM sees only the file being fixed
   → worst case, still functional
```

---

## Testing

```bash
# Run CPG unit tests (no Joern server needed)
pytest tests/unit/test_cpg.py -v

# Run with live Joern (requires JOERN_URL set)
JOERN_URL=http://localhost:8080 pytest tests/unit/test_cpg.py -v -k "live"
```

---

## Operational Notes

### Why Joern specifically

Joern is the only open-source tool (Apache 2.0) that:
1. Builds a full CPG (AST + CFG + PDG) — not just an import graph
2. Supports all relevant languages in one tool
3. Runs incremental updates (critical for commit-level auditing)
4. Has a queryable server mode with a mature query language
5. Has been battle-tested on 10M+ line codebases

Alternatives considered:
- **codeql** — GitHub-only, closed source for production use
- **semgrep** — pattern matching only, no inter-procedural analysis
- **pyan3** — import graph only, no data flow
- **networkx** (existing) — same as pyan3

### Resource scaling

For production 10M line codebases:
- Joern JVM heap: 8GB minimum (`--max-heap-size 8g`)
- Container memory: 10GB (`mem_limit: 10g`)
- CPUs: 4+ recommended for CPG build speed
- Storage: ~5GB for the workspace (CPG is persisted between restarts)

### Security

The Joern container mounts the target repo **read-only** (`ro` volume flag).
It has no write access to the repo and no network access to the internet.
The Joern HTTP API is exposed only on the internal Docker network.
