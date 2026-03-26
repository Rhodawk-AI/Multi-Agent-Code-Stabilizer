# Rhodawk AI Code Stabilizer

## Overview

Rhodawk AI is a production-grade, swarm-based autonomous AI software engineer designed to identify, audit, and fix vulnerabilities and logic errors in large-scale codebases. It targets high-reliability and safety-critical domains (military, aerospace, medical) and aims for ≥85% on SWE-bench Verified.

## Architecture

- **`api/`** — FastAPI REST API (main entry point), runs on port 8000
- **`agents/`** — Specialized AI agents (auditor, fixer, reviewer, adversarial critic, test generator, formal verifier)
- **`orchestrator/`** — DeerFlow async DAG orchestration; coordinates agents
- **`brain/`** — Storage layer (SQLite dev / PostgreSQL prod + vector store)
- **`cpg/`** — Code Property Graph integration (Joern-based)
- **`swarm/`** — Multi-agent framework integrations (CrewAI, AutoGen, LangGraph)
- **`tools/`** — Model Context Protocol (MCP) tool implementations
- **`auth/`** — JWT authentication middleware
- **`compliance/`** — RTM/SAS compliance exporters (DO-178C)
- **`workers/`** — Celery task queue workers

## Running the App

The application starts automatically via the "Start application" workflow, which runs:
```
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

## Key Endpoints

- `GET /health` — Health check with version and CPG status
- `GET /docs` — Interactive API documentation (Swagger UI)
- `GET /api/capabilities` — Feature matrix report
- `POST /runs` — Start a new stabilization run (requires JWT auth)
- `GET /api/issues` — List found issues
- `GET /api/fixes` — List generated fixes
- `GET /api/files` — Browse analyzed files

## Environment Configuration

All secrets are managed via Replit's Secrets panel. Key environment variables:

| Variable | Description |
|---|---|
| `RHODAWK_ENV` | `development` or `production` |
| `RHODAWK_JWT_ALGORITHM` | `HS256` (dev) or `RS256` (prod) |
| `RHODAWK_JWT_SECRET` | High-entropy secret for HS256 mode |
| `ANTHROPIC_API_KEY` | Claude API key for cloud LLM routing |
| `OPENROUTER_API_KEY` | OpenRouter API key for Llama4/Devstral |
| `GITHUB_TOKEN` | GitHub token for repo access |
| `RHODAWK_PG_DSN` | PostgreSQL DSN (uses SQLite if not set) |

## Storage

- **Development**: SQLite (auto-created at `.stabilizer/brain.db` per repo)
- **Production**: PostgreSQL via `RHODAWK_PG_DSN` environment variable
- **Vector Store**: Qdrant (local) or in-memory fallback

## Tech Stack

- **Python 3.11**
- **FastAPI + Uvicorn** — REST API server
- **LiteLLM** — LLM abstraction layer (Claude, OpenAI, Ollama, vLLM)
- **LangGraph, CrewAI, AutoGen** — Multi-agent orchestration
- **SQLAlchemy + aiosqlite/asyncpg** — Database ORM
- **python-jose** — JWT authentication
- **z3-solver** — Formal verification
- **Semgrep, Ruff, Bandit, Mypy** — Static analysis tools

## Stabilization Status (March 2026)

Full adversarial review (46 findings across 130 files) was completed and ALL bugs fixed end-to-end:

### Batch 1 (27 findings)
- **BLOCK-01–08**: Phantom model tags, vLLM port collision, `use_sqlite=True` default, `gap5_enabled=False` safe default, synthesis model default, `/auth/token` endpoint created
- **BUG-01/04/07/08**: AuditScore medium_count, rate_limiter in BaseAgent, duplicate import, adversarial critic timeout, Celery error handling
- **ARCH-05/07**: PostgresBrainStorage error messaging improved
- **SEC-05**: MAX_CONCURRENT_RUNS=50 cap
- **MISSING-04/05**: SAS dynamic objectives, refactor proposals

### Batch 2 (19 findings)
- **BUG-02/03**: completion_tokens reads actual usage; instructor client FD leak fixed
- **BUG-05/06**: `solver_used=` field mismatch; `max_cycles` default 50→200
- **SEC-01/02/03/04**: WebSocket subprotocol auth; webhook HMAC required in prod; `os._exit` → `sys.exit`; AegisEDR sanitization on all source before LLM prompts
- **ARCH-01/02/03/04/08**: Honest SWE-bench estimates; DO-178C advisory disclaimer on SAS; `cpg_enabled=False` default; model registry YAML; 60s controller init timeout
- **ARCH-06**: PostgreSQL DDL synced with SQLite (13 missing tables added)
- **DEMO-02/03/04**: Subprocess sandbox fallback (no Docker); Lean4 advisory language; Prometheus metrics pre-initialized
- **MISSING-01/02/03**: Leanstral wired into formal_verifier; federation peer warning; escalation notification fallback

### Code Review Fixes
- **AegisEDR regex**: Variable-width lookbehind replaced with simple pattern
- **Formal verifier**: `NOT_APPLICABLE` → `SKIPPED` (enum member exists); stale `solver` → `solver_used`
- **Subprocess sandbox**: Shell injection via `bash -c` replaced with direct argv + test ID validation

All 15 unit test files pass at 100% (agents, brain, chunking, consensus, convergence, cpg, execution_feedback, executor, gap3, gap4, gap5, gap6, graph, synthesis).

## Review Documents

- **`ADVERSARIAL_REVIEW.md`** — Hostile technical review of the entire codebase (46 findings, 130 files analyzed). Covers BLOCK bugs, security issues, SWE-bench ceiling analysis, and DO-178C compliance gaps.
- **`BILLION_DOLLAR_ROADMAP.md`** — Strategic roadmap covering: path to 90% SWE-bench via open-source integrations (Moatless Tools, OpenHands, ARPO), billion-dollar product architecture, and 14-week sprint plan.

## Security Notes

- JWT authentication is enforced on all protected endpoints
- CORS origins are configured via `RHODAWK_CORS_ORIGINS`
- `RHODAWK_DEV_AUTH=1` must never be set in production
- Dev mode (`RHODAWK_ENV=development`) relaxes startup security checks for local development

## Route Structure

After migration, routes are registered with these prefixes:
- `/runs` — Stabilization runs (via runs router)
- `/api/issues` — Issue listings
- `/api/fixes` — Fix listings
- `/api/files` — File browsing
- `/api/escalations/` — Escalation management
- `/api/compound-findings/` — Compound findings
- `/api/synthesis-reports/` — Synthesis reports
- `/api/refactor-proposals/` — Refactor proposals
- `/commits/` — Commit audit
