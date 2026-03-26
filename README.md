# Rhodawk AI Code Stabilizer v1.0

**Swarm-based autonomous AI engineer** for safety-critical code stabilization.

> **Benchmark status (March 2026):** SWE-bench Verified has not yet been
> measured on this system. The 85% target below is an *architectural design
> goal*, not a measured result. No independent evaluation has been run.
> Claims in the comparison table are projected targets, not demonstrated scores.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Rhodawk AI Swarm                          │
├──────────────────┬──────────────────┬───────────────────────┤
│  LangGraph       │  CrewAI Crews    │  AutoGen Agents       │
│  State Machine   │  (Security,      │  (Conversational      │
│                  │   SWE-bench)     │   coordination)       │
└──────────────────┴──────────────────┴───────────────────────┘
           ↕ DeerFlow Workflow Orchestration (bespoke async DAG — not Prefect)
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
│  (Qdrant-backed) │  │  cppcheck | Semgrep | CVE | SBOM     │
│  10M+ lines      │  │  Jujutsu  | Swarm Health | Promptfoo  │
└──────────────────┘  └──────────────────────────────────────┘
           ↕
┌─────────────────────────────────────────────────────────────┐
│  Aegis EDR + Z3/CBMC Formal Verification                    │
│  Prometheus Metrics | JWT Auth | HMAC Audit Trail           │
└─────────────────────────────────────────────────────────────┘
```

## Benchmark Targets

> **Model Note:** Granite 4.0-H-Tiny/Small referenced in earlier documentation
> are unreleased as of this writing. The router uses `granite-code:3b` and
> `granite-code:8b` (real, available on Ollama hub). Update `models/router.py`
> when Granite 4.0 ships.


| Benchmark          | Target | Realistic Estimate | Baseline (Claude Code) |
|--------------------|--------|--------------------|-----------------------|
| SWE-bench Verified | ≥85% | **60–73%** | 80.9% |
| Terminal-Bench 2.0 | ≥75% | **55–65%** | 65.4% |
| FLTEval (formal)   | 26.3% | **~20%** | N/A |
| Cost per issue     | <$0.30 | **$0.15–$0.50** | ~$2.00 |

**Status: NO EVALUATION HAS BEEN RUN.** All numbers in this table are
engineering estimates based on component ablation studies (Agent S3 BoBN
paper, Qwen2.5-Coder benchmarks, CBMC coverage literature). The "Target"
column is aspirational; the "Realistic Estimate" column reflects what we
expect without ARPO fine-tuning or CPG integration. To obtain an actual
measured score, run: `rhodawk-bench run --limit 50`

## Positioning

Rhodawk targets **regulated-industry codebases** (aerospace, defense, nuclear,
automotive) where DO-178C / IEC 61508 compliance evidence is mandatory and
human review bottlenecks are the dominant cost driver.

The BoBN (Best-of-Best-of-N) ensemble uses N=10 candidate generations per fix
by default, which requires 8–10× the GPU compute of single-model solutions.
This is configurable via environment variables to match your budget:

| Profile | Env Vars | GPU Cost | Expected Lift |
|---------|----------|----------|---------------|
| **Minimal** (N=2) | `RHODAWK_BOBN_FIXER_A=1 RHODAWK_BOBN_FIXER_B=1` | ~2× baseline | +5-8pp |
| **Balanced** (N=5) | `RHODAWK_BOBN_FIXER_A=3 RHODAWK_BOBN_FIXER_B=2` | ~5× baseline | +10-15pp |
| **Full** (N=10, default) | `RHODAWK_BOBN_FIXER_A=6 RHODAWK_BOBN_FIXER_B=4` | ~10× baseline | +12-18pp |

In safety-critical domains, the cost of a missed defect (FAA airworthiness
directive, nuclear safety shutdown) far exceeds the compute premium. For
general-purpose coding tasks where cost-per-token matters more than correctness
guarantees, single-model solutions or N=2 are more appropriate.

### ARPO Fine-Tuning Compute Options

Full ARPO training (OpenRLHF, 32B model) requires 4×A100 80GB with ZeRO-3.
For smaller deployments, use the built-in TRL GRPO single-GPU fallback:

```bash
python scripts/arpo_trainer.py --trl    # Single GPU, 7B-14B models
python scripts/arpo_trainer.py --run    # Multi-GPU, 32B (requires 4×A100)
```

### Integration Options

Rhodawk exposes a REST API (FastAPI) and CLI (`rhodawk`, `rhodawk-bench`).
For CI/CD integration, use the webhook endpoint (`POST /api/webhook/ci`)
with HMAC verification. GitHub App and VS Code extension are on the roadmap
but not yet implemented.

## Configuration: Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RHODAWK_JWT_SECRET` | Yes | 256-bit hex secret for JWT signing |
| `RHODAWK_AUDIT_SECRET` | Yes | HMAC secret for audit trail integrity |
| `RHODAWK_DEV_AUTH` | Dev only | Set to `1` to bypass auth in development |
| `RHODAWK_WEBHOOK_SECRET` | Prod | HMAC secret for CI webhook verification |
| `RHODAWK_CORS_ORIGINS` | No | Comma-separated allowed CORS origins |
| `RHODAWK_SLACK_WEBHOOK_URL` | No | Slack webhook for escalation notifications |
| `RHODAWK_ESCALATION_WEBHOOKS` | No | Additional webhook URLs for escalation alerts |
| `RHODAWK_ENV` | No | Set to `development` for dev mode (default: `production`) |
| `DATABASE_URL` | Prod | PostgreSQL connection string (SQLite used if absent) |
| `gap6_federation_peers` | No | Comma-separated peer URLs for federated pattern sharing |

> **Note:** Without `RHODAWK_SLACK_WEBHOOK_URL` or `RHODAWK_ESCALATION_WEBHOOKS`,
> escalation notifications are logged but not delivered externally. Configure at
> least one notification channel for DO-178C DAL-A deployments where human-in-the-loop
> approval is mandatory.

## Quick Start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Set required secrets (never skip this)
export RHODAWK_JWT_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
export RHODAWK_AUDIT_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
export RHODAWK_DEV_AUTH=1

# 3. Pull local models (Ollama)
ollama pull granite-code:8b
ollama pull granite-code:3b
ollama pull qwen2.5-coder:32b

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
swarm/           DeerFlow (bespoke DAG) + CrewAI + AutoGen
verification/    Advisory property reasoning + Z3 formal verification
memory/          HelixDB (Qdrant graph+vector)
security/        Aegis EDR (exploit + injection detection)
tools/           ToolHive MCP layer + server stubs
swe_bench/       SWE-bench Verified evaluation harness
workers/         Celery distributed workers
rust/mcp_server/ High-performance Rust MCP static analysis
```
