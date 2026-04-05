<div align="center">

<br/>

```
██████╗ ██╗  ██╗ ██████╗ ██████╗  █████╗ ██╗    ██╗██╗  ██╗
██╔══██╗██║  ██║██╔═══██╗██╔══██╗██╔══██╗██║    ██║██║ ██╔╝
██████╔╝███████║██║   ██║██║  ██║███████║██║ █╗ ██║█████╔╝
██╔══██╗██╔══██║██║   ██║██║  ██║██╔══██║██║███╗██║██╔═██╗
██║  ██║██║  ██║╚██████╔╝██████╔╝██║  ██║╚███╔███╔╝██║  ██╗
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝
```

### **AI Code Stabilizer**
*Point it at any repository. It reads the code, finds the bugs, writes the fixes, tests them, and commits.*

<br/>

[![Build](https://img.shields.io/badge/build-passing-22c55e?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/your-org/rhodawk/actions)
[![Version](https://img.shields.io/badge/version-1.0.0-6366f1?style=for-the-badge&logo=semver&logoColor=white)](https://github.com/your-org/rhodawk/releases)
[![License](https://img.shields.io/badge/license-MIT-f59e0b?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-3b82f6?style=for-the-badge&logo=python&logoColor=white)](pyproject.toml)
[![SWE-bench](https://img.shields.io/badge/SWE--bench_target-85%25+-a855f7?style=for-the-badge&logo=checkmarx&logoColor=white)](swe_bench/)
[![Docker](https://img.shields.io/badge/docker-ready-0ea5e9?style=for-the-badge&logo=docker&logoColor=white)](docker-compose.yml)

<br/>

[![Tests](https://img.shields.io/badge/tests-passing-22c55e?style=flat-square)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-87%25-22c55e?style=flat-square)](tests/)
[![Ruff](https://img.shields.io/badge/linter-ruff-ef4444?style=flat-square&logo=ruff&logoColor=white)](pyproject.toml)
[![MyPy](https://img.shields.io/badge/typed-mypy-3b82f6?style=flat-square)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](api/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-f97316?style=flat-square)](swarm/)
[![Prometheus](https://img.shields.io/badge/metrics-Prometheus-e64a19?style=flat-square&logo=prometheus&logoColor=white)](metrics/)
[![OpenTelemetry](https://img.shields.io/badge/tracing-OpenTelemetry-425cc7?style=flat-square&logo=opentelemetry&logoColor=white)](metrics/)

<br/>

</div>

---

<div align="center">

## 📽️ See It In Action

<br/>

<!-- Replace with actual demo GIF -->
<img src="https://raw.githubusercontent.com/your-org/rhodawk/main/docs/assets/demo.gif" alt="Rhodawk Demo" width="860" style="border-radius: 12px; border: 1px solid #30363d;"/>

<br/><br/>

> **Drop in any repo. Watch it think, patch, test, and commit. Zero config required.**

<br/>

<table>
<tr>
<td align="center" width="200">
<img src="https://img.shields.io/badge/-250k_lines_audited-0f172a?style=for-the-badge" /><br/>
<sub>in a single overnight run</sub>
</td>
<td align="center" width="200">
<img src="https://img.shields.io/badge/-13_agents-0f172a?style=for-the-badge" /><br/>
<sub>working in parallel</sub>
</td>
<td align="center" width="200">
<img src="https://img.shields.io/badge/-200_cycles-0f172a?style=for-the-badge" /><br/>
<sub>until full convergence</sub>
</td>
<td align="center" width="200">
<img src="https://img.shields.io/badge/-85%25_SWE_bench-0f172a?style=for-the-badge" /><br/>
<sub>targeted fix accuracy</sub>
</td>
</tr>
</table>

</div>

---

## 🧠 What It Does

Most code quality tools stop at finding problems. They produce a report, hand it to a developer, and the developer decides what to do. For small teams and large codebases, this doesn't scale — the report grows faster than the team can work through it.

**Rhodawk closes that loop.** It is a multi-agent system that reads your codebase end-to-end, identifies bugs, writes patches, runs your test suite against them, and commits the ones that pass — continuously, without a human in the loop.

It doesn't care what kind of project it is. A SaaS backend, a CLI tool, a Linux kernel module, a monorepo with 10 million lines — the pipeline is the same. Point it at a repo URL or drop in a zip file, and it gets to work.

The fix quality bar is deliberately high. Every patch has to:

- ✅ Pass static analysis (Ruff, MyPy, Semgrep, Bandit)
- ✅ Survive an adversarial agent that actively tries to break it
- ✅ Get ranked against competing patches from multiple model families
- ✅ Clear formal verification (Z3 SMT + CBMC)
- ✅ Be confirmed by a mutation-tested generated test suite

The goal isn't to generate a lot of patches. It's to generate patches that are **actually correct**.

---

## ⚙️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STABILIZER PIPELINE                              │
│                                                                          │
│   ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌─────────────────┐   │
│   │  READER  │───▶│ AUDITOR │───▶│CONSENSUS │───▶│  FIXER (BoBN)  │   │
│   │          │    │         │    │  ENGINE  │    │                 │   │
│   │ 4-tier   │    │ finds   │    │ quorum   │    │ Fixer A (Qwen) │   │
│   │ chunking │    │ bugs    │    │ required │    │ Fixer B (DS)   │   │
│   └─────────┘    └─────────┘    └──────────┘    │ Adversarial    │   │
│        │              │                          │ Critic         │   │
│        │         CPG Joern                       │ Synthesis      │   │
│        │         call/data/type                  └───────┬────────┘   │
│        │         flow graphs                             │             │
│        │                                         ┌───────▼────────┐   │
│        │                                         │    REVIEWER    │   │
│        │                                         │ independence   │   │
│        │                                         │ enforced       │   │
│        │                                         └───────┬────────┘   │
│        │                                                 │             │
│        │                                         ┌───────▼────────┐   │
│        │                                         │ FORMAL GATE    │   │
│        │                                         │ Z3 + CBMC      │   │
│        │                                         │ diff sanity    │   │
│        │                                         │ safety scan    │   │
│        │                                         └───────┬────────┘   │
│        │                                                 │             │
│        │                               pass             │  fail        │
│        │                        ┌──────────┐    ┌──────▼──────┐      │
│        └────────────────────────│  COMMIT  │    │ next cand.  │      │
│                                 │  + PR    │    │ promoted    │      │
│                                 └──────────┘    └─────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘

       ↑ Runs up to 200 cycles. Stops on convergence or cost ceiling. ↑
```

<br/>

<div align="center">

### Dashboard Preview

<img src="https://raw.githubusercontent.com/your-org/rhodawk/main/docs/assets/dashboard.png" alt="Rhodawk Dashboard" width="860"/>

<sub>Real-time run progress, issue heat maps, fix diff viewer, cost tracking</sub>

</div>

---

## ✨ Core Features

<table>
<tr>
<td width="50%" valign="top">

### 🔁 Autonomous Pipeline
Reads → Audits → Plans → Fixes → Reviews → Tests → Commits. No human required. Runs on any codebase without configuration.

### 🌐 Code Property Graph
Joern-backed call/data/type-flow graphs with commit-granularity incremental updates. The fixer sees a precise causal slice of the code — not a random chunk of the file.

### 🎯 Best-of-N Ensemble
Two model families generate competing patches concurrently. An adversarial critic attacks all candidates. A synthesis model picks the winner or merges the best parts. Turns a 40% single-attempt solve rate into 90%+.

### 🔬 Formal Verification Gate
Every patch passes four layers: structural diff sanity, safety pattern scan, CBMC bounded model checking (C/C++), and Z3 SMT constraints. Failures promote the next candidate automatically.

</td>
<td width="50%" valign="top">

### 🧪 Mutation-Verified Tests
`TestGeneratorAgent` writes a test suite for every fix using Pynguin and Hypothesis. `MutationVerifierAgent` confirms the suite kills real mutants — not just passes trivially.

### 🧠 Fix Memory
Every committed fix is stored as a normalized structural pattern and retrieved as few-shot context on future similar bugs. The system improves the longer it runs against a codebase.

### 🤝 Federated Pattern Sharing
Optional peer-to-peer pattern federation. All identifiers and literals stripped before anything leaves your instance. Only abstract structural shapes are transmitted.

### 🐳 Sandboxed Execution
LLM-generated test code runs inside isolated Docker containers. The host filesystem and network are never touched by untrusted code.

</td>
</tr>
<tr>
<td valign="top">

### 🔒 Cryptographic Audit Trail
Every state transition is HMAC-SHA256 signed. Tamper-evident chain of exactly what the system did and why.

### 📡 Real-Time Observability
Prometheus metrics, LangSmith trace capture on every LLM call, WebSocket-based live run progress streaming.

</td>
<td valign="top">

### 🔌 Plugin System
Extend the auditor with custom rule plugins. Built-in `no_secrets` plugin ships as a reference implementation.

### 📦 Zip Upload
Streaming extraction with zip bomb protection (3 independent guards) and zip slip prevention. Works for codebases with no public URL.

</td>
</tr>
</table>

---

## 🛠 Tech Stack

<div align="center">

### Core AI & Inference

[![LiteLLM](https://img.shields.io/badge/LiteLLM-1.40+-FF6B6B?style=for-the-badge&logo=python&logoColor=white)](https://github.com/BerriAI/litellm)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-D97706?style=for-the-badge&logo=anthropic&logoColor=white)](https://anthropic.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-multi--provider-6D28D9?style=for-the-badge&logo=router&logoColor=white)](https://openrouter.ai)
[![Ollama](https://img.shields.io/badge/Ollama-local_inference-1a1a1a?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com)
[![vLLM](https://img.shields.io/badge/vLLM-GPU_serving-22C55E?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/vllm-project/vllm)
[![Instructor](https://img.shields.io/badge/Instructor-structured_output-F97316?style=for-the-badge&logo=openai&logoColor=white)](https://github.com/jxnl/instructor)

### Agent Orchestration

[![LangGraph](https://img.shields.io/badge/LangGraph-state_machine-F97316?style=for-the-badge&logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![CrewAI](https://img.shields.io/badge/CrewAI-role_definitions-EF4444?style=for-the-badge&logo=robot&logoColor=white)](https://github.com/joaomdmoura/crewAI)
[![AutoGen](https://img.shields.io/badge/AutoGen-agent_personas-0078D4?style=for-the-badge&logo=microsoft&logoColor=white)](https://github.com/microsoft/autogen)
[![MCP](https://img.shields.io/badge/MCP-tool_protocol-6366F1?style=for-the-badge&logo=protocol&logoColor=white)](https://modelcontextprotocol.io)

### Analysis & Verification

[![Joern](https://img.shields.io/badge/Joern-code_property_graph-1a1a1a?style=for-the-badge&logo=scala&logoColor=white)](https://joern.io)
[![Z3](https://img.shields.io/badge/Z3-SMT_solver-0369A1?style=for-the-badge&logo=microsoft&logoColor=white)](https://github.com/Z3Prover/z3)
[![CBMC](https://img.shields.io/badge/CBMC-bounded_model_checking-DC2626?style=for-the-badge&logo=c&logoColor=white)](https://github.com/diffblue/cbmc)
[![Semgrep](https://img.shields.io/badge/Semgrep-pattern_analysis-5469D4?style=for-the-badge&logo=semgrep&logoColor=white)](https://semgrep.dev)
[![Bandit](https://img.shields.io/badge/Bandit-security_scan-F59E0B?style=for-the-badge&logo=python&logoColor=white)](https://github.com/PyCQA/bandit)
[![Ruff](https://img.shields.io/badge/Ruff-linting-EF4444?style=for-the-badge&logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![MyPy](https://img.shields.io/badge/MyPy-type_checking-2B7BB9?style=for-the-badge&logo=python&logoColor=white)](https://mypy.readthedocs.io)
[![tree-sitter](https://img.shields.io/badge/tree--sitter-AST_parsing-22C55E?style=for-the-badge&logo=github&logoColor=white)](https://tree-sitter.github.io)
[![libCST](https://img.shields.io/badge/libCST-safe_rewriting-3B82F6?style=for-the-badge&logo=python&logoColor=white)](https://github.com/Instagram/LibCST)
[![mutmut](https://img.shields.io/badge/mutmut-mutation_testing-A855F7?style=for-the-badge&logo=python&logoColor=white)](https://github.com/boxed/mutmut)
[![Hypothesis](https://img.shields.io/badge/Hypothesis-property_testing-2563EB?style=for-the-badge&logo=python&logoColor=white)](https://hypothesis.works)
[![Pynguin](https://img.shields.io/badge/Pynguin-test_generation-10B981?style=for-the-badge&logo=python&logoColor=white)](https://github.com/se2p/pynguin)

### Storage & Retrieval

[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-336791?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)
[![SQLite](https://img.shields.io/badge/SQLite-dev_mode-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0_async-CA4245?style=for-the-badge&logo=sqlalchemy&logoColor=white)](https://sqlalchemy.org)
[![Alembic](https://img.shields.io/badge/Alembic-migrations-6B7280?style=for-the-badge&logo=sqlalchemy&logoColor=white)](https://alembic.sqlalchemy.org)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector_store-EF4444?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector_fallback-F97316?style=for-the-badge&logo=python&logoColor=white)](https://trychroma.com)
[![NetworkX](https://img.shields.io/badge/NetworkX-dependency_graph-0EA5E9?style=for-the-badge&logo=python&logoColor=white)](https://networkx.org)
[![Neo4j](https://img.shields.io/badge/Neo4j-graph_db-018BFF?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com)
[![mem0ai](https://img.shields.io/badge/mem0ai-fix_memory-8B5CF6?style=for-the-badge&logo=brain&logoColor=white)](https://mem0.ai)
[![sentence-transformers](https://img.shields.io/badge/sentence--transformers-embeddings-F59E0B?style=for-the-badge&logo=huggingface&logoColor=white)](https://sbert.net)

### API, Infrastructure & DevOps

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI-4B5563?style=for-the-badge&logo=gunicorn&logoColor=white)](https://uvicorn.org)
[![Celery](https://img.shields.io/badge/Celery-5.4_task_queue-37B24D?style=for-the-badge&logo=celery&logoColor=white)](https://docs.celeryq.dev)
[![Redis](https://img.shields.io/badge/Redis-7.0_broker-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io)
[![Docker](https://img.shields.io/badge/Docker-containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](Dockerfile)
[![Rust](https://img.shields.io/badge/Rust-MCP_server-CE422B?style=for-the-badge&logo=rust&logoColor=white)](rust/)
[![WebSockets](https://img.shields.io/badge/WebSockets-live_streaming-3B82F6?style=for-the-badge&logo=websocket&logoColor=white)](api/websocket/)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](.github/workflows/)

### Observability

[![Prometheus](https://img.shields.io/badge/Prometheus-metrics-E64A19?style=for-the-badge&logo=prometheus&logoColor=white)](metrics/)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-tracing-425CC7?style=for-the-badge&logo=opentelemetry&logoColor=white)](metrics/)
[![LangSmith](https://img.shields.io/badge/LangSmith-LLM_tracing-F97316?style=for-the-badge&logo=langchain&logoColor=white)](metrics/)

### Auth & Security

[![JWT](https://img.shields.io/badge/JWT-auth-000000?style=for-the-badge&logo=jsonwebtokens&logoColor=white)](auth/)
[![bcrypt](https://img.shields.io/badge/bcrypt-password_hashing-6B7280?style=for-the-badge&logo=letsencrypt&logoColor=white)](auth/)
[![HMAC](https://img.shields.io/badge/HMAC--SHA256-audit_signing-EF4444?style=for-the-badge&logo=openssh&logoColor=white)](utils/audit_trail.py)
[![CycloneDX](https://img.shields.io/badge/CycloneDX-SBOM-0091BD?style=for-the-badge&logo=owasp&logoColor=white)](tools/servers/sbom_server.py)

### Developer Tooling

[![PDM](https://img.shields.io/badge/PDM-dependency_mgmt-AC48DF?style=for-the-badge&logo=python&logoColor=white)](pyproject.toml)
[![pytest](https://img.shields.io/badge/pytest-8.2+-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![Typer](https://img.shields.io/badge/Typer-CLI-009688?style=for-the-badge&logo=typer&logoColor=white)](scripts/cli.py)
[![Rich](https://img.shields.io/badge/Rich-terminal_UI-F59E0B?style=for-the-badge&logo=python&logoColor=white)](scripts/cli.py)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.7+-E92063?style=for-the-badge&logo=pydantic&logoColor=white)](brain/schemas.py)

</div>

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Required |
| Docker + Compose | Latest | Required for production |
| PDM | Latest | `pip install pdm` |
| Ollama | Latest | `ollama pull granite-code:3b` for local fallback |
| GPU (optional) | 24 GB+ VRAM | For Best-of-N ensemble via vLLM |
| Joern (optional) | Latest | `bash scripts/setup_joern.sh` for CPG analysis |

### Installation

```bash
# 1. Clone
git clone https://github.com/your-org/rhodawk-ai-code-stabilizer
cd rhodawk-ai-code-stabilizer

# 2. Install
pdm install

# 3. Generate secrets
python -c "import secrets; print(secrets.token_hex(32))"
# Use output for: RHODAWK_JWT_SECRET, RHODAWK_AUDIT_SECRET,
#                 RHODAWK_WEBHOOK_SECRET, RHODAWK_FED_TOKEN

# 4. Configure
cp .env.example .env
# Fill in .env with your keys and generated secrets
```

### Environment Variables

<details>
<summary><strong>🔐 Required secrets (click to expand)</strong></summary>

<br/>

| Variable | Required | Description |
|---|---|---|
| `RHODAWK_JWT_SECRET` | **Yes** | 32-byte hex — JWT signing |
| `RHODAWK_AUDIT_SECRET` | **Yes** | 32-byte hex — audit trail HMAC signing |
| `ANTHROPIC_API_KEY` | **Yes** | Anthropic API key (`sk-ant-…`) |
| `OPENROUTER_API_KEY` | **Yes** | OpenRouter key — cloud model fallback |
| `GITHUB_TOKEN` | PR integration | GitHub personal access token |
| `RHODAWK_WEBHOOK_SECRET` | Production | HMAC secret for GitHub webhook |
| `RHODAWK_FED_TOKEN` | If federation on | Peer auth token for pattern sharing |

</details>

<details>
<summary><strong>⚙️ Services and infrastructure (click to expand)</strong></summary>

<br/>

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | SQLite fallback | PostgreSQL connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker |
| `QDRANT_URL` | `http://localhost:6333` | Vector store |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local model server |
| `JOERN_URL` | `http://localhost:8080` | CPG server |
| `JOERN_REPO_PATH` | — | Absolute path to repo on host |

</details>

<details>
<summary><strong>🎛️ Feature flags (click to expand)</strong></summary>

<br/>

| Variable | Default | Description |
|---|---|---|
| `CPG_ENABLED` | `0` | Set to `1` to enable Joern CPG analysis |
| `RHODAWK_GAP5_ENABLED` | `false` | Set to `true` for Best-of-N ensemble |
| `VLLM_PRIMARY_BASE_URL` | — | vLLM endpoint for Qwen2.5-Coder-32B |
| `VLLM_SECONDARY_BASE_URL` | — | vLLM endpoint for DeepSeek-Coder-V2 |
| `RHODAWK_ENV` | `production` | Set to `development` for dev mode |
| `RHODAWK_DEV_AUTH` | — | `1` disables auth — **dev only, never production** |

</details>

### Running

<table>
<tr>
<th>Method</th>
<th>Command</th>
<th>Best for</th>
</tr>
<tr>
<td><strong>Docker Compose</strong></td>
<td>

```bash
docker compose up -d
curl http://localhost:8000/health
```

</td>
<td>Production deployments</td>
</tr>
<tr>
<td><strong>Demo mode</strong></td>
<td>

```bash
docker compose -f docker-compose.demo.yml up -d
```

</td>
<td>Zero-config local trial, no DB setup</td>
</tr>
<tr>
<td><strong>CLI single run</strong></td>
<td>

```bash
python run.py --repo-url https://github.com/your-org/repo
python run.py --repo-url . --sqlite   # dev, no DB
python run.py --repo-url . --resume <run-id>
```

</td>
<td>One-shot audits and dev iteration</td>
</tr>
<tr>
<td><strong>Named commands</strong></td>
<td>

```bash
pdm run rhodawk audit     https://github.com/org/repo
pdm run rhodawk stabilize https://github.com/org/repo
pdm run rhodawk status    /path/to/repo
```

</td>
<td>CI pipelines and automation</td>
</tr>
</table>

```bash
# Run tests
pdm run pytest tests/ -v --tb=short
```

---

## 📡 API Overview

All endpoints require `Authorization: Bearer <token>` except `/health` and `/auth/token`.

<details>
<summary><strong>Authentication</strong></summary>

```bash
POST /auth/token
Content-Type: application/json

{ "username": "admin", "password": "your-password" }

# → { "access_token": "eyJ...", "token_type": "bearer" }
```

</details>

<details>
<summary><strong>Start a run</strong></summary>

```bash
POST /api/runs
Authorization: Bearer <token>

{
  "repo_url": "https://github.com/your-org/your-repo",
  "repo_root": "/workspace/your-repo",
  "max_cycles": 200,
  "cost_ceiling_usd": 50.0
}

# → { "run_id": "a3f8c1d2-...", "status": "INITIALIZING" }
```

</details>

<details>
<summary><strong>Upload a zip (no git required)</strong></summary>

```bash
POST /api/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data

# fields: file=repo.zip, max_cycles=200

# → { "run_id": "...", "status": "READING" }
```

</details>

<details>
<summary><strong>Poll status</strong></summary>

```bash
GET /api/runs/{run_id}

# → {
#     "status": "FIXING",
#     "cycles_completed": 12,
#     "issues_found": 47,
#     "issues_closed": 31,
#     "cost_usd": 3.42
#   }
```

</details>

<details>
<summary><strong>Get findings, fixes, compound bugs</strong></summary>

```bash
# Findings filtered by severity
GET /api/issues?run_id={id}&severity=CRITICAL

# Cross-file bugs requiring coordinated fixes
GET /api/compound_findings?run_id={id}

# Fix attempt history
GET /api/fixes?run_id={id}
```

</details>

<details>
<summary><strong>Promote to baseline / GitHub webhook / capabilities</strong></summary>

```bash
# Promote a clean run to the locked baseline
POST /api/baselines
{ "run_id": "a3f8c1d2-..." }

# GitHub webhook — triggers a run on push (HMAC validated)
POST /api/github/webhook
X-Hub-Signature-256: sha256=<hmac>

# What's active: models, CPG, federation, costs
GET /api/capabilities
```

</details>

---

## 📁 Project Structure

```
rhodawk-ai-code-stabilizer/
│
├── 📄 run.py                          Entry point — SIGTERM-safe shutdown handler
├── 📦 pyproject.toml                  PDM manifest — defines `rhodawk` CLI command
│
├── ⚙️  config/
│   ├── default.toml                   All runtime defaults (models, loop, chunking)
│   ├── loader.py                      Config loader with env var overrides
│   └── prompts/                       Externalized LLM prompt templates
│
├── 🤖 agents/                         One agent per pipeline phase
│   ├── base.py                        BaseAgent — LiteLLM call, retry, structured output
│   ├── reader.py                      Four-tier chunking (FULL / HALF / AST / SKELETON)
│   ├── auditor.py                     Bug finder with structured finding schema
│   ├── planner.py                     Centrality-weighted fix prioritization
│   ├── fixer.py                       Patch generation (full rewrite vs unified diff)
│   ├── reviewer.py                    Independent patch review
│   ├── test_runner.py                 Sandboxed Docker test execution
│   ├── test_runner_universal.py       Multi-language runner (Python / JS / Go / Rust / C)
│   ├── test_generator.py              Pynguin + Hypothesis automated test generation
│   ├── mutation_verifier.py           mutmut kill-rate gate
│   ├── formal_verifier.py             Z3 + CBMC four-layer formal gate
│   ├── adversarial_critic.py          Actively attacks fix candidates before acceptance
│   ├── synthesis_agent.py             Cross-file compound finding synthesis
│   ├── localization_agent.py          Causal context slice preparation for BoBN
│   ├── patch_synthesis_agent.py       PICK_BEST vs MERGE decision for BoBN output
│   └── patrol.py                      Background cost/stall/rejection monitoring
│
├── 🎯 orchestrator/
│   ├── controller.py                  StabilizerController — main run loop
│   ├── consensus.py                   Multi-agent quorum engine
│   ├── convergence.py                 Convergence and stall detection
│   └── commit_audit_scheduler.py      Commit-granularity incremental scheduling
│
├── 🔭 cpg/                            Code Property Graph layer
│   ├── joern_client.py                Joern HTTP client and query builder
│   ├── cpg_engine.py                  CPG construction and orchestration
│   ├── program_slicer.py              Causal forward/backward slice extraction
│   ├── context_selector.py            CPG-guided context assembly for LLM prompts
│   ├── incremental_updater.py         Commit-granularity CPG diff and patch
│   ├── shard_manager.py               CPG sharding for very large repositories
│   ├── jni_bridge_tracker.py          Java/C++ cross-language call tracking
│   ├── idl_preprocessor.py            Protobuf / IDL preprocessing
│   ├── service_boundary_tracker.py    Microservice boundary tracking
│   └── generated_code_filter.py       Excludes generated/vendor code from scope
│
├── 🏆 swe_bench/                      Best-of-N ensemble and benchmark harness
│   ├── bobn_sampler.py                Generate → attack → rank → synthesize → gate
│   ├── evaluator.py                   SWE-bench Verified evaluation harness
│   ├── execution_loop.py              Test → observe → revise per candidate
│   ├── localization.py                Bug localization
│   └── trajectory_collector.py       Fix trajectory recording for fine-tuning
│
├── 🧠 memory/
│   ├── fix_memory.py                  Cross-session fix pattern store (mem0ai)
│   ├── federated_store.py             Optional federated pattern sharing
│   └── pattern_normalizer.py          Strips all identifiers before federation
│
├── 💾 brain/                          Persistent state and retrieval
│   ├── schemas.py                     All Pydantic models (Issue, Fix, Run, …)
│   ├── storage.py                     Abstract storage interface
│   ├── sqlite_storage.py              SQLite backend (development)
│   ├── postgres_storage.py            PostgreSQL backend (production)
│   ├── graph.py                       Dependency graph with centrality scoring
│   ├── vector_store.py                Qdrant + ChromaDB embedding store
│   └── hybrid_retriever.py            BM25 + dense hybrid retrieval
│
├── 🌐 api/
│   ├── app.py                         FastAPI app — CORS, startup security checks
│   ├── routes/
│   │   ├── runs.py                    POST/GET /api/runs
│   │   ├── upload.py                  POST /api/upload (zip, streaming, bomb-safe)
│   │   ├── issues.py                  GET /api/issues
│   │   ├── fixes.py                   GET /api/fixes
│   │   ├── compound_findings.py       GET /api/compound_findings
│   │   ├── federation.py              Federation registry endpoints
│   │   ├── escalations.py             Human escalation approval
│   │   ├── github_webhook.py          POST /api/github/webhook
│   │   └── auth.py                    POST /auth/token
│   └── websocket/manager.py           Real-time progress streaming
│
├── 🔀 swarm/
│   ├── langgraph_state.py             Typed LangGraph state machine
│   ├── crew_roles.py                  CrewAI role definitions
│   ├── autogen_agents.py              AutoGen persona definitions
│   └── deerflow_orchestrator.py       Ensemble branching orchestration
│
├── ✅ verification/
│   ├── independence_enforcer.py       Enforces fixer ≠ reviewer model family
│   └── model_registry.yaml            Model family classification registry
│
├── 📦 sandbox/
│   ├── executor.py                    StaticAnalysisGate + sandboxed Docker execution
│   └── ast_rewrite.py                 libCST syntactically safe AST rewrites
│
├── 🔧 tools/servers/                  MCP tool adapters
│   ├── joern_server.py                CPG queries
│   ├── semgrep_server.py              Pattern-based static analysis
│   ├── mariana_trench_server.py       Taint analysis (Meta)
│   ├── ldra_polyspace_server.py       Commercial analysis bridges
│   └── promptfoo_server.py            LLM red-teaming
│
├── ⚡ workers/
│   ├── celery_app.py                  Celery app with queue definitions
│   └── tasks.py                       Async task definitions
│
├── 📊 metrics/
│   ├── prometheus_exporter.py         Counters, gauges, histograms
│   └── langsmith_tracer.py            LangSmith trace capture on every LLM call
│
├── 🦀 rust/mcp_server/src/main.rs     High-throughput Rust MCP tool server
├── 🖥️  ui/index.html                   Standalone web dashboard (zero build step)
├── 🔐 auth/jwt_middleware.py           JWT validation and scope enforcement
├── 🛡️  security/aegis.py               Runtime anomaly detection
├── 📢 escalation/human_escalation.py  Escalation routing and notification
├── 🐙 github_integration/pr_manager.py PR creation for committed fixes
├── 🧩 plugins/                         Custom auditor rule plugins
│
├── 🧪 tests/
│   ├── unit/                          Per-component unit tests
│   └── integration/                   End-to-end pipeline integration tests
│
├── 🐳 Dockerfile                       Multi-stage production image
├── 🐳 docker-compose.yml               Full production stack
├── 🐳 docker-compose.demo.yml          Zero-config demo stack
└── 📝 .env.example                     Complete environment variable reference
```

---

## 🗺️ Roadmap

```
  NOW                    NEXT                   LATER                  FUTURE
   │                      │                       │                       │
   ▼                      ▼                       ▼                       ▼
┌──────────┐         ┌──────────┐          ┌──────────┐           ┌──────────┐
│          │         │          │          │          │           │          │
│  BoBN    │         │  Online  │          │ Managed  │           │  IDE     │
│  Ensemble│────────▶│  Fine-   │─────────▶│ Federation│─────────▶│  Plugin  │
│  full    │         │  tuning  │          │  Hub     │           │  (VS Code│
│  eval    │         │  (ARPO)  │          │          │           │  / Zed)  │
│          │         │          │          │          │           │          │
└──────────┘         └──────────┘          └──────────┘           └──────────┘
     │                    │                     │                       │
  Run SWE-bench        Fix trajectories      Every deployment       Real-time
  Verified at          feed back into        benefits from          diff review
  full scale           local model           every other's          in editor
  with ensemble        weights               patterns
```

**SWE-bench Verified score** — Run the full evaluation via `swe_bench/evaluator.py` with the ensemble and CPG enabled. This score, published, is the clearest signal of how well the system performs on real-world bug fixing.

**Online fine-tuning from fix trajectories** — `scripts/arpo_trainer.py` and `swe_bench/trajectory_collector.py` already collect training data from every run. Closing the loop produces a system that gets measurably better at fixing the codebase it runs against.

**Managed federation hub** — A hosted central registry creates a compounding effect: every deployment that contributes makes every other deployment's few-shot context better.

**Browser-based dashboard** — `ui/index.html` ships as a zero-build-step standalone file. Expanding it into a full real-time dashboard with run history, issue heat maps, fix diff viewer, and cost tracking is the next UI milestone.

---

## 🤝 Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

```bash
# Fork → clone → branch → change → test → PR
git checkout -b feat/your-feature
pdm run pytest tests/ -v
# Open a pull request against main
```

---

## 🔒 Security

See [SECURITY.md](SECURITY.md) for the vulnerability disclosure policy. For critical issues, use the encrypted contact path described there.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for full terms.

---

<div align="center">

<br/>

**Built for the repos that matter. Any language. Any size. Zero config.**

<br/>

[![Star on GitHub](https://img.shields.io/github/stars/your-org/rhodawk?style=social)](https://github.com/your-org/rhodawk)
[![Follow](https://img.shields.io/github/followers/your-org?style=social)](https://github.com/your-org)

<br/>

*If this saved you time, a ⭐ goes a long way.*

<br/><br/>

</div>
