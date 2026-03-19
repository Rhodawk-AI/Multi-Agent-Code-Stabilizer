# Rhodawk AI Code Stabilizer v1.0

**Swarm-based autonomous AI engineer** targeting ≥85% on SWE-bench Verified, beating Claude Code (80.9%).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Rhodawk AI Swarm                          │
├──────────────────┬──────────────────┬───────────────────────┤
│  LangGraph       │  CrewAI Crews    │  AutoGen Agents       │
│  State Machine   │  (Security,      │  (Conversational      │
│                  │   SWE-bench)     │   coordination)       │
└──────────────────┴──────────────────┴───────────────────────┘
           ↕ DeerFlow Workflow Orchestration
┌─────────────────────────────────────────────────────────────┐
│                Tiered Model Router                           │
│  Tier1: Granite 4.0-H-Tiny (7B/1B)  — local, $0.00        │
│  Tier2: Granite 4.0-H-Small (32B/9B) — local, $0.00       │
│  Tier3: Llama4 / Devstral2 via OpenRouter — cloud           │
│  Tier4: Claude Sonnet/Opus — critical fallback              │
└─────────────────────────────────────────────────────────────┘
           ↕                    ↕
┌──────────────────┐  ┌──────────────────────────────────────┐
│  HelixDB         │  │  ToolHive MCP Layer                  │
│  (Qdrant shards) │  │  MiroFish | Semgrep | CVE | SBOM     │
│  10M+ lines      │  │  Jujutsu  | Aurite  | Promptfoo      │
└──────────────────┘  └──────────────────────────────────────┘
           ↕
┌─────────────────────────────────────────────────────────────┐
│  Aegis EDR + Leanstral (Lean4) Formal Verification          │
│  Prometheus Metrics | JWT Auth | HMAC Audit Trail           │
└─────────────────────────────────────────────────────────────┘
```

## Benchmark Targets

| Benchmark          | Target    | Baseline (Claude Code) |
|--------------------|-----------|------------------------|
| SWE-bench Verified | **≥85%**  | 80.9%                  |
| Terminal-Bench 2.0 | **≥75%**  | 65.4%                  |
| FLTEval (formal)   | **26.3%** | N/A                    |
| Cost per issue     | **<$0.30**| ~$2.00                 |

## Quick Start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Set required secrets (never skip this)
export RHODAWK_JWT_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
export RHODAWK_AUDIT_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
export RHODAWK_DEV_AUTH=1

# 3. Pull local models (Ollama)
ollama pull granite4-small
ollama pull granite4-tiny

# 4. Start API server
uvicorn api.app:app --port 8000

# 5. Run stabilization
rhodawk run --repo-url https://github.com/org/repo \
            --repo-root /path/to/cloned/repo \
            --max-cycles 10

# 6. Run SWE-bench evaluation
rhodawk-bench run --limit 50
```

## Bug Fixes (B1–B12)

| ID  | Component              | Fix |
|-----|------------------------|-----|
| B1  | audit_trail.py         | HMAC secret from env, fail-fast |
| B2  | api/routes             | JWT Bearer auth on all endpoints |
| B3  | config/loader.py       | Required secrets validated at startup |
| B4  | api/app.py             | Prometheus /metrics endpoint |
| B5  | plugins/base.py        | Subprocess env scrubbed of credentials |
| B6  | plugins/base.py        | Plugin path validation |
| B7  | utils/rate_limiter.py  | Returns key, never sets os.environ |
| B8  | security/aegis.py      | ExfiltrationGuard covers terminal ops |
| B9  | api/websocket          | WebSocket JWT via ?token= |
| B10 | config/loader.py       | Config fails if secrets missing |
| B11 | orchestrator/controller| Aegis EDR scans every fix pre-commit |
| B12 | memory/helixdb.py      | Qdrant-backed 10M+ line scale |

## New Modules

```
auth/            JWT middleware + token factory
metrics/         Prometheus instrumentation
models/          Tiered model router (local→cloud)
swarm/           LangGraph + CrewAI + DeerFlow
verification/    Leanstral (Lean4) + Z3 formal verification
memory/          HelixDB (Qdrant graph+vector)
security/        Aegis EDR (exploit + injection detection)
tools/           ToolHive MCP layer + server stubs
swe_bench/       SWE-bench Verified evaluation harness
workers/         Celery distributed workers
rust/mcp_server/ High-performance Rust MCP static analysis
```
