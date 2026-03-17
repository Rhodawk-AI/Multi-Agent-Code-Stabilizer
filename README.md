<div align="center">

```
  ██████╗ ██████╗ ███████╗███╗   ██╗███╗   ███╗ ██████╗ ███████╗███████╗
 ██╔═══██╗██╔══██╗██╔════╝████╗  ██║████╗ ████║██╔═══██╗██╔════╝██╔════╝
 ██║   ██║██████╔╝█████╗  ██╔██╗ ██║██╔████╔██║██║   ██║███████╗███████╗
 ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║╚██╔╝██║██║   ██║╚════██║╚════██║
 ╚██████╔╝██║     ███████╗██║ ╚████║██║ ╚═╝ ██║╚██████╔╝███████║███████║
  ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚══════╝
```

**Autonomous Multi-Agent Code Stabilizer**  
*The AI that doesn't stop until your codebase is perfect.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![CI](https://github.com/rhodawk-ai-code-stabilizer/rhodawk-ai-code-stabilizer/actions/workflows/ci.yml/badge.svg)](https://github.com/rhodawk-ai-code-stabilizer/rhodawk-ai-code-stabilizer/actions)
[![Coverage](https://img.shields.io/badge/coverage-85%25-green.svg)](tests/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)
[![Discord](https://img.shields.io/discord/000000000?label=Discord&logo=discord)](https://discord.gg/rhodawk-ai-code-stabilizer)

</div>

---

> **Give RHODAWK AI CODE STABILIZER a GitHub repo. It reads every single line, finds every bug, fixes them all, commits the fixes, and loops — until your codebase meets your exact quality bar. Then it stops.**

---

## What This Is

RHODAWK AI CODE STABILIZER is an open-source autonomous agent system that:

1. **Reads your entire codebase** — 100 files or 100,000 files. Every line. Across multiple LLM sessions with a persistent brain that accumulates everything.
2. **Audits against your master prompt** — you define what "correct" means. Security? Architecture? DO-178C? MISRA? You write the spec, RHODAWK AI CODE STABILIZER enforces it.
3. **Fixes everything** — returns complete, production-ready fixed files. Not diffs. Not suggestions. The actual fixed files.
4. **Commits to GitHub** — creates branches and PRs with full audit trails.
5. **Loops until done** — re-audits after every fix. Never stops until zero critical issues remain.

## Why It Beats Code Review Tools

| | CodeRabbit / Copilot | RHODAWK AI CODE STABILIZER |
|---|---|---|
| **Mode** | Reactive (reviews your PRs) | Autonomous (creates its own) |
| **Memory** | Per-PR only | Full codebase brain |
| **Cross-file reasoning** | ❌ | ✅ Dependency graph aware |
| **Fixes things** | ❌ Comments only | ✅ Complete fixed files |
| **Loops until done** | ❌ | ✅ Self-correcting |
| **Custom standards** | Limited | ✅ Any spec you can write |
| **Privacy** | Your code on their servers | ✅ Fully self-hosted |
| **Million-line support** | ❌ | ✅ Hybrid chunking engine |

## Quick Start

```bash
# 1. Install
pip install rhodawk-ai-code-stabilizer

# 2. Set keys
export ANTHROPIC_API_KEY=your_key
export GITHUB_TOKEN=your_token

# 3. Stabilize any repo
rhodawk-ai-code-stabilizer stabilize https://github.com/your/repo --path /path/to/local/clone
```

That's it. Watch it work.

## Docker (Zero Setup)

```bash
git clone https://github.com/rhodawk-ai-code-stabilizer/rhodawk-ai-code-stabilizer
cd rhodawk-ai-code-stabilizer
cp .env.example .env   # add your API keys
docker compose up

# Open http://localhost:8000 for the live dashboard
```

## Web Dashboard

RHODAWK AI CODE STABILIZER ships with a real-time dashboard. Watch agents work live:

```
┌─────────────────────────────────────────────────────┐
│  RHODAWK AI CODE STABILIZER Dashboard              Run: abc123        │
├──────────────┬──────────────────────────────────────┤
│ CYCLE 3/50   │  ████████████░░░░  Score: 47 → 12   │
├──────────────┴──────────────────────────────────────┤
│ 🔵 Reader    reading core/gii/gii_loop.py (chunk 2) │
│ 🔴 Security  CRITICAL: credential in line 287        │
│ 🟡 Fixer     fixing policy/engine.py (847 lines)    │
│ 🟢 Reviewer  APPROVED: fix abc (confidence: 0.94)   │
│ 🔵 Patrol    all systems nominal                     │
└─────────────────────────────────────────────────────┘
```

## How It Handles Million-Line Codebases

The secret is the **hybrid chunking engine** + **persistent brain**:

```
File size          →  Strategy
──────────────────────────────────────────────────
< 200 lines        →  Single session (whole file)
200–1,000 lines    →  Two halves with 20-line overlap
1,000–5,000 lines  →  AST-boundary splits (class/function)
5,000–20,000 lines →  Skeleton first, then targeted deep reads
> 20,000 lines     →  Structural skeleton + on-demand sections
```

Between sessions, everything is stored in a SQLite brain: every symbol, every dependency, every observation. The audit synthesizes the brain — not the raw files.

## Plugin System

Extend RHODAWK AI CODE STABILIZER with custom audit rules:

```python
from rhodawk-ai-code-stabilizer.plugins import AuditPlugin, Issue, Severity

class NASACodingStandards(AuditPlugin):
    name = "nasa_standards"
    description = "JPL Institutional Coding Standard for C"

    async def audit_file(self, path: str, content: str) -> list[Issue]:
        issues = []
        if "goto" in content:
            issues.append(Issue(
                severity=Severity.CRITICAL,
                description="Rule 1: No goto statements (JPL Rule 1)",
                ...
            ))
        return issues
```

```bash
rhodawk-ai-code-stabilizer stabilize --plugin ./nasa_standards.py https://github.com/your/repo
```

## Multi-Model Support

Use any LLM — or all of them simultaneously for cross-validation:

```toml
# config/default.toml
[models]
primary = "claude-sonnet-4-20250514"
fallbacks = ["gpt-4o-mini", "deepseek-chat"]
local = "ollama/qwen2.5-coder:32b"   # free, private, runs on your machine

[strategy]
# Use cheap model for triage, expensive for critical fixes only
triage_model = "claude-haiku-4-5-20251001"
critical_fix_model = "claude-opus-4-20250514"
cross_validate_critical = true
```

## Master Prompt — You Define "Correct"

The master prompt is the spec. It lives in your repo. It's versioned. It's yours.

```markdown
# My Project's Quality Spec

## Rule 1 — No Unhandled Exceptions (CRITICAL)
Every external call must have explicit error handling.

## Rule 2 — Safety Gate Required (CRITICAL)  
All state-modifying operations must pass ConsequenceReasoner.

## Rule 3 — MISRA-C Compliance (MAJOR)
...
```

The system is stabilized when your spec passes. Nothing more, nothing less.

## Real-World Results

| Codebase | Size | Issues Found | Cycles to Stabilize |
|---|---|---|---|
| ProjectZeo (GII agent) | 80K lines, 240 files | 47 critical | 6 cycles |
| Node.js REST API | 12K lines | 8 critical | 2 cycles |
| Python ML pipeline | 35K lines | 23 critical | 4 cycles |

## Architecture

```
rhodawk-ai-code-stabilizer stabilize repo
        │
        ▼
┌───────────────────────────────────────────────┐
│              PERSISTENT BRAIN (SQLite)         │
│  file_map · issues · fixes · sessions · cost  │
└───────────────────┬───────────────────────────┘
                    │  all agents read/write here
        ┌───────────┴───────────┐
        ▼                       ▼
  ┌──────────┐           ┌─────────────┐
  │  READER  │           │   PATROL    │
  │ (Phase 1)│           │  (watchdog) │
  └────┬─────┘           └─────────────┘
       │
       ▼
  ┌──────────────────────────────────┐
  │    AUDITORS (parallel × 3)       │
  │  Security · Architecture · Stds  │
  └──────────────┬───────────────────┘
                 │
                 ▼
           ┌──────────┐
           │  FIXER   │  ← complete files returned
           └────┬─────┘
                │
                ▼
          ┌──────────┐
          │ REVIEWER │  ← APPROVED / REJECTED / ESCALATE
          └────┬─────┘
               │
               ▼
         ┌──────────┐
         │  GITHUB  │  ← branch + PR + audit trail
         └────┬─────┘
              │
         LOOP UNTIL
         STABILIZED
```

## CLI Reference

```bash
# Full stabilization loop
rhodawk-ai-code-stabilizer stabilize <repo_url> --path <local_path>

# Audit only (no commits)
rhodawk-ai-code-stabilizer audit <repo_url> --path <local_path> --output report.md

# Pre-index a large repo cheaply before auditing
rhodawk-ai-code-stabilizer bootstrap <local_path>

# Check current run status
rhodawk-ai-code-stabilizer status <local_path>

# View live dashboard
rhodawk-ai-code-stabilizer dashboard <local_path>
```

## Safety

RHODAWK AI CODE STABILIZER is designed to be safe to run on production codebases:

- **Never executes LLM-generated code** on your host
- **Architectural lock** — safety-critical files require human approval before any fix is committed
- **PR-only commits** — never pushes directly to main
- **Full audit trail** — every decision logged with reasoning
- **Cost ceiling** — hard stop before you overspend
- **Revert command** — undo all stabilizer commits since any timestamp

## Contributing

RHODAWK AI CODE STABILIZER is fully open source. We welcome:
- New audit plugins (security standards, language-specific rules)
- New LLM adapters
- Dashboard improvements
- Documentation

See [CONTRIBUTING.md](docs/CONTRIBUTING.md).

## License

MIT — use it on anything, including commercial and military-grade systems.

---

<div align="center">
Built with conviction. Runs until done.
</div>
