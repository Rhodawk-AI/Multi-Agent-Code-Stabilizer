# Rhodawk AI Code Stabilizer

> **Point it at any repository. It reads the code, finds the bugs, writes the fixes, tests them, and commits.**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/your-org/rhodawk)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-blue)](pyproject.toml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![SWE-bench Target](https://img.shields.io/badge/SWE--bench%20target-85%25%2B-purple)](swe_bench/)

---

## What It Does

Most code quality tools stop at finding problems. They produce a report, hand it to a developer, and the developer decides what to do. For small teams and large codebases this does not scale — the report grows faster than the team can work through it.

Rhodawk closes that loop. It is a multi-agent system that reads your codebase end-to-end, identifies bugs, writes patches, runs your test suite against them, and commits the ones that pass — continuously, without a human in the loop. It does not care what kind of project it is. A SaaS backend, a CLI tool, a Linux kernel module, an internal monorepo with 10 million lines — the pipeline is the same. Point it at a repo URL or drop in a zip file, and it gets to work.

The core loop runs up to 200 cycles. Each cycle finds new issues, proposes fixes, tests them in a sandbox, and closes the ones that hold up. It stops when there is nothing left to fix or when you tell it to stop. Anything it cannot fix automatically gets escalated for human review, with enough context attached that a developer can resolve it quickly.

The fix quality bar is deliberately high. Every patch has to pass static analysis, survive an adversarial agent that actively tries to break it, get ranked against competing fix candidates from different models, and clear a formal verification step before it ever touches your codebase. The goal is not to generate a lot of patches — it is to generate patches that are actually correct.

---

## System Architecture

Rhodawk is organized as a pipeline of specialized components. Each layer has one job.

**Orchestration (`orchestrator/`)** — `StabilizerController` runs the main loop. It manages the lifecycle of a run from initialization through convergence: reading files, collecting findings, coordinating fixes, running gates, and deciding when to stop. It handles graceful shutdown on SIGTERM (no mid-write file corruption on restarts), run resumption by ID, and cost ceiling enforcement.

**Agents (`agents/`)** — Thirteen agents, each owning one phase of the pipeline. The `ReaderAgent` ingests files using a four-tier chunking strategy that adapts to file size: full content for small files, AST skeleton for large ones, so the LLM never gets flooded with irrelevant context. The `AuditorAgent` finds bugs. The `FixerAgent` writes patches — either a full file rewrite for small files or a surgical unified diff for large ones, using libCST for syntactically safe AST-level Python edits. The `ReviewerAgent` checks the patch independently. The `FormalVerifierAgent` runs Z3 SMT and CBMC model checking. The `AdversarialCriticAgent` actively attacks each patch before it is accepted. `TestGeneratorAgent` and `MutationVerifierAgent` generate a test suite for the fix and verify it catches real regressions.

**Code Property Graph (`cpg/`)** — Joern integration gives the system a compiler-quality view of the codebase: call graphs, data-flow graphs, type-flow graphs. When the auditor finds a bug, the `ProgramSlicer` extracts a precise causal slice — only the code that actually flows into the bug site. The fixer sees exactly the right context, not a random chunk of the file. The CPG is updated incrementally at commit granularity, so re-runs are fast.

**Best-of-N Ensemble (`swe_bench/`)** — When a critical fix is needed, the `BoBNSampler` generates multiple candidate patches from two different model families concurrently (Qwen2.5-Coder-32B and DeepSeek-Coder-V2), runs each through the test loop, attacks all candidates with the adversarial critic, scores them on test pass rate / robustness / minimality, and then has a third model (different family again) either pick the best one or merge elements from multiple candidates. The statistical effect: if one model solves a hard bug 40% of the time, running five attempts across two model families pushes that to over 90%.

**Memory (`memory/`)** — Every committed fix is stored as a normalized structural pattern. On future runs, similar bug sites get those patterns as few-shot examples, improving fix quality over time. An optional federation layer lets multiple deployments share anonymized patterns — all identifiers and literals stripped before anything leaves your instance.

**Storage (`brain/`)** — SQLite for development, PostgreSQL for production. Qdrant and ChromaDB for vector retrieval. NetworkX (or Neo4j) for the dependency graph, which tracks inter-module relationships and prioritizes which files to fix first based on how central they are to the codebase.

**API and Workers (`api/`, `workers/`)** — FastAPI serves all orchestrator capabilities over HTTP. Celery with Redis handles long-running tasks asynchronously. WebSockets stream run progress in real time. GitHub webhook integration triggers runs automatically on push events.

---

## Core Features

- **Fully autonomous pipeline** — reads, audits, fixes, tests, reviews, and commits without human intervention across repositories of any size.

- **Code Property Graph** — Joern-backed call/data/type-flow analysis with incremental updates. The fixer sees a precise causal slice of the code, not a noisy file dump.

- **Best-of-N adversarial ensemble** — two model families generate competing patches; an adversarial critic attacks them; a synthesis model picks the winner or merges the best parts. Statistically far more reliable than a single model attempt.

- **Formal verification on every patch** — structural diff sanity check, safety pattern scan, CBMC bounded model checking (C/C++), and Z3 SMT constraints. A patch that fails the gate is discarded; the next best candidate is promoted.

- **Sandboxed test execution** — LLM-generated test code runs inside Docker containers, isolated from the host filesystem and network. The host never executes untrusted code directly.

- **Mutation-verified test suites** — `MutationVerifierAgent` uses `mutmut` to confirm that the generated tests actually catch real regressions, not just pass trivially.

- **Fix memory that improves over time** — every committed fix is stored and retrieved as few-shot context on future similar bugs. The system gets better the longer it runs against a codebase.

- **Multi-tier chunking** — FULL → HALF → AST → SKELETON. Adapts to file size automatically. A 200-line utility and a 20,000-line module both get handled correctly without manual tuning.

- **Static analysis gate** — Ruff, MyPy, Semgrep, and Bandit run on every proposed fix. Any failure blocks the commit.

- **Consensus engine** — configurable multi-agent quorum requirements per severity level. Critical bugs require agreement from multiple agents above a confidence floor before a fix is attempted.

- **Zip upload for offline use** — streaming extraction with zip bomb protection and zip slip prevention. Works for codebases that cannot be pointed at a URL.

- **GitHub PR integration** — opens pull requests for committed fixes with structured descriptions and links to the originating findings.

- **Full observability** — Prometheus metrics, LangSmith trace capture on every LLM call, WebSocket-based real-time progress streaming.

- **Cryptographic audit trail** — every state transition is HMAC-signed, producing a tamper-evident log of what the system did and why.

- **Plugin system** — extend the auditor with custom rule plugins. A built-in `no_secrets` plugin is included as a reference.

---

## Tech Stack

**LLM / Inference**
- LiteLLM (unified routing across all providers), Instructor (structured outputs), Anthropic, OpenAI, OpenRouter
- vLLM for local GPU inference (Qwen2.5-Coder-32B, DeepSeek-Coder-V2)
- Ollama for local CPU fallback (Granite-Code-3B, Qwen2.5-Coder-7B)

**Agent Orchestration**
- LangGraph (typed state machine), CrewAI (role definitions), AutoGen (agent personas)

**Static Analysis and Verification**
- Joern (Code Property Graph), Z3 SMT solver, CBMC (bounded model checking)
- Ruff, MyPy, Semgrep, Bandit
- tree-sitter (multi-language AST parsing), libCST (safe Python AST rewriting)
- Pynguin (automated test generation), Hypothesis (property-based testing), mutmut (mutation testing)

**Storage**
- PostgreSQL + asyncpg (production), SQLite + aiosqlite (development)
- SQLAlchemy 2.0 async, Alembic
- Qdrant, ChromaDB (vector stores), NetworkX / Neo4j (dependency graph)
- mem0ai (cross-session fix memory)

**API and Infrastructure**
- FastAPI, Uvicorn, WebSockets
- Celery 5 + Redis (task queue)
- Docker, Docker Compose
- Rust (MCP server for high-throughput tool dispatch)
- Prometheus, OpenTelemetry, LangSmith

**Auth and Security**
- JWT (python-jose + bcrypt), scope-enforced route authorization
- HMAC-SHA256 audit trail signing

**Developer Tooling**
- PDM (dependency management), pytest + pytest-asyncio, Typer + Rich (CLI)

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- PDM: `pip install pdm`
- Ollama with at least one local model: `ollama pull granite-code:3b`
- *(Optional)* A GPU with 24 GB+ VRAM for the Best-of-N ensemble via vLLM
- *(Optional)* Joern for code property graph analysis: `bash scripts/setup_joern.sh`

### Installation

```bash
# Clone
git clone https://github.com/your-org/rhodawk-ai-code-stabilizer
cd rhodawk-ai-code-stabilizer

# Install dependencies
pdm install

# Configure
cp .env.example .env
# Edit .env — see Environment Variables section below

# Generate the required secrets
python -c "import secrets; print(secrets.token_hex(32))"
# Use the output for RHODAWK_JWT_SECRET, RHODAWK_AUDIT_SECRET,
# RHODAWK_WEBHOOK_SECRET, and RHODAWK_FED_TOKEN
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `RHODAWK_JWT_SECRET` | **Yes** | 32-byte hex secret for JWT signing |
| `RHODAWK_AUDIT_SECRET` | **Yes** | 32-byte hex secret for audit trail signing |
| `ANTHROPIC_API_KEY` | **Yes** | Anthropic API key |
| `OPENROUTER_API_KEY` | **Yes** | OpenRouter key (cloud model fallback) |
| `GITHUB_TOKEN` | If using PR integration | GitHub personal access token |
| `RHODAWK_WEBHOOK_SECRET` | In production | HMAC secret for GitHub webhook validation |
| `RHODAWK_FED_TOKEN` | If federation enabled | Auth token for federated pattern sharing |
| `OLLAMA_BASE_URL` | No | Default: `http://localhost:11434` |
| `QDRANT_URL` | No | Default: `http://localhost:6333` |
| `DATABASE_URL` | No | PostgreSQL connection string. Falls back to SQLite if unset |
| `REDIS_URL` | No | Default: `redis://localhost:6379/0` |
| `JOERN_URL` | No | Default: `http://localhost:8080` |
| `JOERN_REPO_PATH` | If CPG enabled | Absolute path to the repository on the host |
| `CPG_ENABLED` | No | Set to `1` to enable code property graph analysis |
| `RHODAWK_GAP5_ENABLED` | No | Set to `true` to enable the Best-of-N ensemble |
| `VLLM_PRIMARY_BASE_URL` | If BoBN enabled | Local vLLM endpoint for Qwen2.5-Coder-32B |
| `VLLM_SECONDARY_BASE_URL` | If BoBN enabled | Local vLLM endpoint for DeepSeek-Coder-V2 |
| `RHODAWK_ENV` | No | Set to `development` for dev mode |
| `RHODAWK_DEV_AUTH` | No | `1` disables all auth. **Dev only — startup fails if env is not `development`** |

### Running

**Docker Compose — recommended**

```bash
# Start everything: API, worker, PostgreSQL, Qdrant, Redis
docker compose up -d

# Check health
curl http://localhost:8000/health
```

**Demo mode — zero config, no external services required**

```bash
docker compose -f docker-compose.demo.yml up -d
```

**CLI — for single runs**

```bash
# Point at a GitHub repo
python run.py --repo-url https://github.com/your-org/your-repo

# Point at a local directory
python run.py --repo-url . --repo-root /path/to/repo

# Resume an interrupted run
python run.py --repo-url . --resume <run-id>

# Dev mode with SQLite (no database setup needed)
python run.py --repo-url . --sqlite
```

**Named commands**

```bash
pdm run rhodawk audit     https://github.com/your-org/repo --output report.md
pdm run rhodawk stabilize https://github.com/your-org/repo
pdm run rhodawk status    /path/to/repo
```

**Tests**

```bash
pdm run pytest tests/ -v --tb=short
# or
make test
```

---

## API Overview

All endpoints except `/health` and `/auth/token` require a Bearer JWT.

**Get a token**

```bash
POST /auth/token
{"username": "admin", "password": "your-password"}
# → {"access_token": "eyJ...", "token_type": "bearer"}
```

**Start a run**

```bash
POST /api/runs
Authorization: Bearer <token>

{
  "repo_url": "https://github.com/your-org/your-repo",
  "repo_root": "/workspace/your-repo",
  "max_cycles": 200,
  "cost_ceiling_usd": 50.0
}
# → {"run_id": "a3f8c1d2-...", "status": "INITIALIZING"}
```

**Upload a zip (no git required)**

```bash
POST /api/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data
# fields: file=<repo.zip>, max_cycles=200

# → {"run_id": "...", "status": "READING"}
```

**Poll status**

```bash
GET /api/runs/{run_id}
# → {"status": "FIXING", "cycles_completed": 12,
#    "issues_found": 47, "issues_closed": 31, "cost_usd": 3.42}
```

**Get findings**

```bash
GET /api/issues?run_id={run_id}&severity=CRITICAL
# Structured findings with file, line, function, description,
# confidence score, and fix attempt history
```

**Get cross-file compound findings**

```bash
GET /api/compound_findings?run_id={run_id}
# Bugs that span multiple files or require coordinated multi-file fixes
```

**Promote to baseline**

```bash
POST /api/baselines
{"run_id": "a3f8c1d2-..."}
```

**GitHub webhook — auto-trigger on push**

```bash
POST /api/github/webhook
X-Hub-Signature-256: sha256=<hmac>
# HMAC validated against RHODAWK_WEBHOOK_SECRET
```

**System capabilities**

```bash
GET /api/capabilities
# Which features are active, which models are routing,
# CPG status, federation status, cost tracking
```

---

## Project Structure

```
rhodawk-ai-code-stabilizer/
│
├── run.py                          # CLI entry point; SIGTERM-safe shutdown
├── pyproject.toml                  # PDM manifest; defines `rhodawk` CLI command
│
├── config/
│   ├── default.toml                # All runtime defaults (models, loop, chunking)
│   ├── loader.py                   # Config loader with env var overrides
│   └── prompts/                    # Externalized LLM prompt templates
│
├── agents/                         # One agent per pipeline phase
│   ├── base.py                     # BaseAgent: LiteLLM call, retry, structured output
│   ├── reader.py                   # Four-tier file chunking (FULL/HALF/AST/SKELETON)
│   ├── auditor.py                  # Bug finder; structured finding schema
│   ├── planner.py                  # Fix prioritization; centrality-weighted ordering
│   ├── fixer.py                    # Patch generation (full rewrite vs. unified diff)
│   ├── reviewer.py                 # Independent patch review
│   ├── test_runner.py              # Sandboxed test execution
│   ├── test_runner_universal.py    # Multi-language runner (Python/JS/Go/Rust/C)
│   ├── test_generator.py           # Automated test generation (Pynguin + Hypothesis)
│   ├── mutation_verifier.py        # Mutation kill-rate gate
│   ├── formal_verifier.py          # Z3 SMT + CBMC four-layer formal gate
│   ├── adversarial_critic.py       # Actively attacks fix candidates before acceptance
│   ├── synthesis_agent.py          # Cross-file compound finding synthesis
│   ├── localization_agent.py       # Causal context slice prep for BoBN
│   ├── patch_synthesis_agent.py    # Picks best or merges candidates from BoBN
│   └── patrol.py                   # Background patrol; cost/stall/rejection monitoring
│
├── orchestrator/
│   ├── controller.py               # StabilizerController: main run loop
│   ├── consensus.py                # Multi-agent quorum engine
│   ├── convergence.py              # Convergence and stall detection
│   └── commit_audit_scheduler.py   # Incremental commit-granularity scheduling
│
├── cpg/                            # Code Property Graph layer
│   ├── joern_client.py             # Joern HTTP client and query builder
│   ├── cpg_engine.py               # CPG construction and orchestration
│   ├── program_slicer.py           # Causal forward/backward slice extraction
│   ├── context_selector.py         # CPG-guided context assembly for LLM prompts
│   ├── incremental_updater.py      # Commit-granularity CPG diff and patch
│   ├── shard_manager.py            # CPG sharding for very large repos
│   ├── jni_bridge_tracker.py       # Java/C++ cross-language call tracking
│   ├── idl_preprocessor.py         # Protobuf / IDL preprocessing
│   ├── service_boundary_tracker.py # Microservice boundary tracking
│   └── generated_code_filter.py    # Excludes generated/vendor code from scope
│
├── swe_bench/                      # Best-of-N ensemble and benchmark harness
│   ├── bobn_sampler.py             # Generate → execute → attack → rank → synthesize → gate
│   ├── evaluator.py                # SWE-bench Verified evaluation harness
│   ├── execution_loop.py           # Test → observe → revise feedback loop per candidate
│   ├── localization.py             # Bug localization
│   └── trajectory_collector.py    # Fix trajectory recording for fine-tuning
│
├── memory/
│   ├── fix_memory.py               # Cross-session fix pattern store (mem0ai)
│   ├── federated_store.py          # Optional federated pattern sharing
│   └── pattern_normalizer.py       # Strips all identifiers before federation
│
├── brain/                          # Persistent state and retrieval
│   ├── schemas.py                  # All Pydantic models (Issue, Fix, Run, ...)
│   ├── storage.py                  # Abstract storage interface
│   ├── sqlite_storage.py           # SQLite backend (dev)
│   ├── postgres_storage.py         # PostgreSQL backend (production)
│   ├── graph.py                    # Dependency graph; centrality scoring
│   ├── vector_store.py             # Qdrant + ChromaDB embedding store
│   └── hybrid_retriever.py         # BM25 + dense hybrid retrieval
│
├── api/
│   ├── app.py                      # FastAPI app; CORS; startup security checks
│   ├── routes/                     # One file per resource
│   │   ├── runs.py                 # POST/GET /api/runs
│   │   ├── upload.py               # POST /api/upload (zip; streaming; bomb-safe)
│   │   ├── issues.py               # GET /api/issues
│   │   ├── fixes.py                # GET /api/fixes
│   │   ├── compound_findings.py    # GET /api/compound_findings
│   │   ├── federation.py           # Federation registry endpoints
│   │   ├── escalations.py          # Human escalation approval
│   │   ├── github_webhook.py       # POST /api/github/webhook
│   │   └── auth.py                 # POST /auth/token
│   └── websocket/manager.py        # Real-time run progress streaming
│
├── swarm/
│   ├── langgraph_state.py          # LangGraph typed state machine
│   ├── crew_roles.py               # CrewAI role definitions
│   ├── autogen_agents.py           # AutoGen persona definitions
│   └── deerflow_orchestrator.py    # Ensemble branching orchestration
│
├── verification/
│   ├── independence_enforcer.py    # Enforces fixer ≠ reviewer model family
│   └── model_registry.yaml         # Model family classification
│
├── sandbox/
│   ├── executor.py                 # StaticAnalysisGate + Docker-sandboxed execution
│   └── ast_rewrite.py              # libCST-based syntactically safe rewrites
│
├── tools/servers/                  # MCP tool adapters
│   ├── joern_server.py             # CPG queries over MCP
│   ├── semgrep_server.py           # Pattern-based analysis
│   ├── mariana_trench_server.py    # Taint analysis
│   └── promptfoo_server.py         # LLM red-teaming
│
├── workers/
│   ├── celery_app.py               # Celery app; queue definitions
│   └── tasks.py                    # Async task definitions
│
├── metrics/
│   ├── prometheus_exporter.py      # Prometheus counters, gauges, histograms
│   └── langsmith_tracer.py         # LangSmith trace capture
│
├── security/aegis.py               # Runtime anomaly detection
├── escalation/human_escalation.py  # Escalation routing and notification
├── github_integration/pr_manager.py # PR creation for committed fixes
├── plugins/                        # Custom auditor rule plugins
├── rust/mcp_server/src/main.rs     # High-throughput Rust MCP tool server
├── ui/index.html                   # Standalone web dashboard (no build step)
├── scripts/
│   ├── cli.py                      # Typer CLI (`rhodawk` command)
│   ├── arpo_trainer.py             # Online RL fine-tuning from fix trajectories
│   ├── benchmark.py                # SWE-bench benchmark runner
│   └── setup_joern.sh              # Joern installation script
├── tests/
│   ├── unit/                       # Per-component unit tests
│   └── integration/                # End-to-end pipeline tests
├── Dockerfile                      # Multi-stage production image
├── docker-compose.yml              # Full production stack
├── docker-compose.demo.yml         # Zero-config demo stack
└── .env.example                    # Complete environment variable reference
```

---

## Roadmap

**SWE-bench Verified score** — The Best-of-N ensemble and CPG-guided localization are built specifically to push the benchmark score above 85%. Running the full evaluation via `swe_bench/evaluator.py` with the ensemble enabled is the next concrete milestone.

**Online fine-tuning from fix trajectories** — `scripts/arpo_trainer.py` and `swe_bench/trajectory_collector.py` already collect training data from every run. Closing the loop — fine-tuning the local model on those trajectories after each benchmark cycle — produces a system that gets measurably better at fixing the specific codebase it runs against.

**Managed federation hub** — The federated pattern store currently requires each deployment to operate its own registry node. A hosted central hub creates a compounding effect: every deployment that contributes makes every other deployment's few-shot context better.

**Browser-based run dashboard** — `ui/index.html` is a standalone file with no build step. Expanding it into a full real-time dashboard with run history, issue heat maps, a fix diff viewer, and cost tracking would make the system accessible to engineers who do not want to use the CLI or raw API.

---

## License

MIT — see [LICENSE](LICENSE) for full terms.

---

## Security

See [SECURITY.md](SECURITY.md) for the vulnerability disclosure policy.
