<div align="center">

<img src="https://img.shields.io/badge/Rhodawk%20AI-Code%20Stabilizer%20v1.0-0a0a0a?style=for-the-badge&labelColor=0a0a0a&color=00e5ff" />

# Rhodawk AI Code Stabilizer

### Autonomous, swarm-based AI engineering for safety-critical software

**The first open-source multi-agent system purpose-built to audit, stabilize, and formally verify codebases at 10M+ line scale — targeting aerospace, defense, nuclear, and automotive compliance.**

[![Website](https://img.shields.io/badge/Website-rhodawkai.com-00e5ff?style=flat-square)](https://rhodawkai.com)
[![License](https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Demo%20Ready-brightgreen?style=flat-square)]()
[![SWE--bench](https://img.shields.io/badge/SWE--bench-Evaluation%20Pending-orange?style=flat-square)]()
[![Contact](https://img.shields.io/badge/Contact-founder%40rhodawk.com-blue?style=flat-square)](mailto:founder@rhodawk.com)

</div>

---

## The Problem

> **Software defects in safety-critical industries cost lives and billions of dollars — yet the tools to prevent them haven't changed in a decade.**

- **$2.08 trillion** — annual global cost of software failures (Consortium for IT Software Quality)
- **DO-178C, IEC 61508, MISRA-C, CERT-C** compliance audits take months of senior engineering time and cost millions per program
- **Aerospace and defense codebases routinely exceed 10 million lines** — far beyond what any human review team can audit thoroughly at speed
- **No existing AI coding tool** is designed for the regulated-industry threat model: formal verification, auditability, HMAC-signed audit trails, and adversarial ensemble checking

Human review is the bottleneck. It is expensive, slow, and non-scalable. Rhodawk replaces it with an autonomous AI swarm.

---

## The Solution

**Rhodawk AI Code Stabilizer** is a multi-agent autonomous system that:

1. **Ingests** any codebase at 10M+ line scale via a Qdrant-backed vector-graph memory layer (HelixDB)
2. **Audits** using an adversarial multi-model ensemble (BoBN — Best-of-Best-of-N, N=10) across four LLM families simultaneously
3. **Fixes** bugs with blast-radius awareness, formal verification via Z3/CBMC, and Aegis EDR pre-commit scanning
4. **Proves** correctness with HMAC-signed audit trails, Prometheus metrics, and machine-readable compliance artifacts for DO-178C / IEC 61508
5. **Learns** through federated anonymized fix-pattern sharing across deployments (Gap 6 architecture)

All of this runs **locally at near-zero variable cost** via a tiered model router that starts with local Ollama models before escalating to cloud APIs only when needed.

---

## Why Now

| Shift | Impact |
|-------|--------|
| LLMs can now generate and reason about code at scale | The audit bottleneck is now solvable in software |
| Granite, Qwen2.5-Coder, and Llama 4 run locally on commodity GPUs | Regulated industries can deploy without cloud data exposure |
| DO-178C and IEC 61508 audit costs are accelerating | $50K–$500K per audit cycle creates immediate ROI for automation |
| No competitor addresses this regulated-industry niche end-to-end | First-mover advantage in a defensible vertical |

---

## Market Opportunity

| Segment | Size |
|---------|------|
| Global software quality assurance market | **$60B** (2025), growing at 12% CAGR |
| Aerospace & defense software verification | **$8.2B** addressable |
| AI-assisted DevSecOps tooling | **$12B** by 2028 |
| **Rhodawk initial beachhead** (DO-178C compliance automation, Tier-1 aerospace/defense) | **~$1.2B SAM** |

Rhodawk's compliance-first positioning creates a moat that general-purpose AI coding tools (GitHub Copilot, Cursor, Claude Code) cannot enter without a complete architectural rebuild.

---

## Traction & Status

| Milestone | Status |
|-----------|--------|
| Core multi-agent swarm (LangGraph + CrewAI + AutoGen) | ✅ Complete |
| 12 critical security & stability bug fixes (B1–B12) | ✅ Complete |
| HelixDB: Qdrant-backed memory at 10M+ line scale | ✅ Complete |
| Aegis EDR + Z3/CBMC formal verification pipeline | ✅ Complete |
| HMAC audit trail + JWT auth + Prometheus metrics | ✅ Complete |
| BoBN adversarial ensemble (N=10 generations per fix) | ✅ Complete |
| REST API (FastAPI) + CLI (`rhodawk`, `rhodawk-bench`) | ✅ Complete |
| All internal pytests passing — demo ready | ✅ Complete |
| SWE-bench Verified external evaluation | 🔄 In progress |
| GitHub App + VS Code extension | 🗓 Roadmap Q3 2026 |
| Enterprise compliance dashboard (DO-178C artifacts) | 🗓 Roadmap Q4 2026 |
| Federated fix-pattern network (Gap 6 production) | 🗓 Roadmap Q1 2027 |

> **Honest benchmark note:** SWE-bench Verified has not yet been externally measured on this system. The 85% target is an architectural design goal. Realistic engineering estimates are 60–73% pre-ARPO fine-tuning. Run `rhodawk-bench run --limit 50` to produce a measured score.

---

## Key Differentiators

| Capability | Rhodawk | GitHub Copilot | SonarQube | Snyk |
|------------|---------|----------------|-----------|------|
| Autonomous multi-agent repair | ✅ | ❌ | ❌ | ❌ |
| Formal verification (Z3/CBMC) | ✅ | ❌ | ❌ | ❌ |
| DO-178C / IEC 61508 audit artifacts | ✅ | ❌ | Partial | ❌ |
| HMAC-signed tamper-proof audit trail | ✅ | ❌ | ❌ | ❌ |
| 10M+ line codebase scale | ✅ | ❌ | ✅ | ✅ |
| Adversarial BoBN ensemble | ✅ | ❌ | ❌ | ❌ |
| Local-first / air-gap capable | ✅ | ❌ | Partial | ❌ |
| Open source | ✅ | ❌ | Community | ❌ |
| Cost per issue (target) | **$0.15–$0.50** | N/A | ~$2.00 | ~$3.00+ |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Rhodawk AI Swarm                          │
├──────────────────┬──────────────────┬───────────────────────┤
│  LangGraph       │  CrewAI Crews    │  AutoGen Agents       │
│  State Machine   │  (Security,      │  (Conversational      │
│                  │   SWE-bench)     │   coordination)       │
└──────────────────┴──────────────────┴───────────────────────┘
           ↕ DeerFlow Workflow Orchestration (bespoke async DAG)
┌─────────────────────────────────────────────────────────────┐
│                Tiered Model Router                           │
│  Tier 1: granite-code:3b  (local, Ollama)  — $0.00         │
│  Tier 2: granite-code:8b  (local, Ollama)  — $0.00         │
│  Tier 3: Llama 4 / Devstral via OpenRouter — cloud          │
│  Tier 4: Claude Sonnet/Opus — critical fallback             │
└─────────────────────────────────────────────────────────────┘
           ↕                    ↕
┌──────────────────┐  ┌──────────────────────────────────────┐
│  HelixDB         │  │  ToolHive MCP Layer                  │
│  (Qdrant-backed) │  │  cppcheck | Semgrep | CVE | SBOM     │
│  10M+ lines      │  │  Jujutsu  | Swarm Health | Promptfoo │
└──────────────────┘  └──────────────────────────────────────┘
           ↕
┌─────────────────────────────────────────────────────────────┐
│  Aegis EDR + Z3/CBMC Formal Verification                    │
│  Prometheus Metrics | JWT Auth | HMAC Audit Trail           │
└─────────────────────────────────────────────────────────────┘
```

### BoBN Ensemble Configuration

The Best-of-Best-of-N ensemble runs N=10 candidate fix generations per defect by default, using adversarial cross-checking across four LLM families. This is configurable by budget:

| Profile | Env Vars | GPU Cost | Expected Lift |
|---------|----------|----------|---------------|
| **Minimal** (N=2) | `RHODAWK_BOBN_FIXER_A=1 RHODAWK_BOBN_FIXER_B=1` | ~2× baseline | +5–8pp |
| **Balanced** (N=5) | `RHODAWK_BOBN_FIXER_A=3 RHODAWK_BOBN_FIXER_B=2` | ~5× baseline | +10–15pp |
| **Full** (N=10, default) | `RHODAWK_BOBN_FIXER_A=6 RHODAWK_BOBN_FIXER_B=4` | ~10× baseline | +12–18pp |

> In safety-critical domains, the cost of a missed defect (FAA airworthiness directive, nuclear safety shutdown) far exceeds the compute premium. For general-purpose use where cost-per-token matters more than correctness guarantees, N=2 is appropriate.

---

## Benchmark Targets

| Benchmark | Target | Realistic Estimate | Baseline (Claude Code) |
|-----------|--------|--------------------|------------------------|
| SWE-bench Verified | ≥85% | **60–73%** | 80.9% |
| Terminal-Bench 2.0 | ≥75% | **55–65%** | 65.4% |
| FLTEval (formal verification) | 26.3% | **~20%** | N/A |
| Cost per issue | <$0.30 | **$0.15–$0.50** | ~$2.00 |

> All estimates are engineering projections based on component ablation studies (Agent S3 BoBN paper, Qwen2.5-Coder benchmarks, CBMC coverage literature). Run `rhodawk-bench run --limit 50` to produce a measured score. ARPO fine-tuning (32B model, 4×A100) is expected to push SWE-bench scores 10–15pp above baseline estimates.

---

## Quick Start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Set required secrets (never skip this step)
export RHODAWK_JWT_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
export RHODAWK_AUDIT_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
export RHODAWK_DEV_AUTH=1

# 3. Pull local models via Ollama (zero variable cost)
ollama pull granite-code:8b
ollama pull granite-code:3b
ollama pull qwen2.5-coder:32b

# 4. Start API server
uvicorn api.app:app --port 8000

# 5. Run stabilization on a repo
rhodawk run --repo-url https://github.com/org/repo \
            --repo-root /path/to/cloned/repo \
            --max-cycles 10

# 6. Run SWE-bench evaluation
rhodawk-bench run --limit 50
```

### ARPO Fine-Tuning (Optional — Boosts Accuracy)

```bash
python scripts/arpo_trainer.py --trl    # Single GPU, 7B–14B models
python scripts/arpo_trainer.py --run    # Multi-GPU, 32B (requires 4×A100 80GB)
```

---

## Configuration: Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RHODAWK_JWT_SECRET` | **Yes** | 256-bit hex secret for JWT signing |
| `RHODAWK_AUDIT_SECRET` | **Yes** | HMAC secret for audit trail integrity |
| `RHODAWK_DEV_AUTH` | Dev only | Set to `1` to bypass auth in development |
| `RHODAWK_WEBHOOK_SECRET` | Prod | HMAC secret for CI webhook verification |
| `RHODAWK_CORS_ORIGINS` | No | Comma-separated allowed CORS origins |
| `RHODAWK_SLACK_WEBHOOK_URL` | No | Slack webhook for escalation notifications |
| `RHODAWK_ESCALATION_WEBHOOKS` | No | Additional webhook URLs for escalation alerts |
| `RHODAWK_ENV` | No | Set to `development` for dev mode (default: `production`) |
| `DATABASE_URL` | Prod | PostgreSQL connection string (SQLite used if absent) |
| `gap6_federation_peers` | No | Comma-separated peer URLs for federated pattern sharing |

> **DO-178C DAL-A deployments:** Configure at least one notification channel (`RHODAWK_SLACK_WEBHOOK_URL` or `RHODAWK_ESCALATION_WEBHOOKS`) — human-in-the-loop approval is mandatory for this assurance level.

### Integration

Rhodawk exposes a **REST API (FastAPI)** and **CLI** (`rhodawk`, `rhodawk-bench`). For CI/CD, use the webhook endpoint (`POST /api/webhook/ci`) with HMAC verification. GitHub App and VS Code extension are on the roadmap.

---

## Codebase Modules

```
auth/            JWT middleware + token factory
metrics/         Prometheus instrumentation
models/          Tiered model router (local → cloud)
swarm/           DeerFlow async DAG + CrewAI + AutoGen
verification/    Z3 formal verification + advisory property reasoning
memory/          HelixDB (Qdrant graph + vector, 10M+ line scale)
security/        Aegis EDR (exploit + injection detection)
tools/           ToolHive MCP layer + static analysis stubs
swe_bench/       SWE-bench Verified evaluation harness
workers/         Celery distributed workers
rust/mcp_server/ High-performance Rust MCP static analysis server
```

## Security & Stability Fixes (B1–B12)

| ID | Component | Fix Applied |
|----|-----------|-------------|
| B1 | `audit_trail.py` | HMAC secret sourced from env; fail-fast on missing key |
| B2 | `api/routes` | JWT Bearer auth enforced on all endpoints |
| B3 | `config/loader.py` | Required secrets validated at startup, not at runtime |
| B4 | `api/app.py` | Prometheus `/metrics` endpoint added |
| B5 | `plugins/base.py` | Subprocess env scrubbed of credentials before exec |
| B6 | `plugins/base.py` | Plugin path traversal validation |
| B7 | `utils/rate_limiter.py` | Returns key value; never writes to `os.environ` |
| B8 | `security/aegis.py` | ExfiltrationGuard covers all terminal operations |
| B9 | `api/websocket` | WebSocket JWT auth via `?token=` query param |
| B10 | `config/loader.py` | Config fails hard if secrets are absent or malformed |
| B11 | `orchestrator/controller` | Aegis EDR scans every fix candidate pre-commit |
| B12 | `memory/helixdb.py` | Qdrant-backed storage verified at 10M+ line scale |

---

## Roadmap

```
Q2 2026  → SWE-bench Verified external evaluation + published results
Q2 2026  → ARPO fine-tuning pipeline (open weights released)
Q3 2026  → GitHub App + VS Code extension
Q3 2026  → Enterprise compliance dashboard (DO-178C artifact export)
Q4 2026  → Joern CPG (call/data/type flow graph) integration
Q4 2026  → SaaS beta — managed Rhodawk for mid-market aerospace teams
Q1 2027  → Federated fix-pattern network (Gap 6 production launch)
Q2 2027  → MISRA-C / CERT-C / ISO 26262 (automotive) expansion
```

---

## Why Open Source

We believe the tooling for safety-critical software verification should be **transparent, auditable, and community-owned**. Closed-source black boxes cannot be trusted in DO-178C DAL-A or nuclear-safety contexts. Rhodawk is open source so that the aerospace and defense community can inspect, extend, and trust every layer of the stack.

The open-source core creates a moat through trust, community contribution, and a federated fix-pattern network that improves with every deployment. Commercial offerings will build on top of this foundation (SaaS, compliance dashboards, managed deployments) — not replace it.

---

## For Investors & Accelerator Partners

We are actively seeking accelerator programs, strategic investors, and design partners in aerospace, defense, and regulated software sectors.

**What we're looking for:**
- Access to enterprise design partners with DO-178C or IEC 61508 obligations
- GPU compute credits for ARPO fine-tuning and SWE-bench evaluation
- Introductions to aerospace/defense procurement channels
- Mentorship on enterprise sales motion and compliance certification strategy

**What we offer:**
- A working, demo-ready system with 200+ files and 60K+ lines of production code
- A defensible technical moat in a vertical that general AI coding tools cannot easily enter
- A solo founder with deep hands-on expertise across multi-agent systems, formal verification, and safety-critical software architecture
- A clear path to $1M+ ARR through per-seat enterprise licensing and compliance audit automation

---

## Team

| Role | Contact |
|------|---------|
| Founder & Lead Engineer | [founder@rhodawk.com](mailto:founder@rhodawk.com) |
| Operations & Partnerships | [manager@rhodawk.com](mailto:manager@rhodawk.com) |

🌐 **Website:** [rhodawkai.com](https://rhodawkai.com)

---

## Contributing

We welcome contributions from the aerospace software safety, formal verification, and AI agent communities.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Ensure all tests pass (`pytest`)
4. Submit a pull request with a clear description of the change and its safety implications

For security disclosures, contact [founder@rhodawk.com](mailto:founder@rhodawk.com) directly. Do not open public issues for security vulnerabilities.

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Rhodawk AI Code Stabilizer**
*Autonomous verification for the software that can't fail.*

[rhodawkai.com](https://rhodawkai.com) · [founder@rhodawk.com](mailto:founder@rhodawk.com) · [manager@rhodawk.com](mailto:manager@rhodawk.com)

</div>
