# Repository Analysis Report — Rhodawk-AI/Multi-Agent-Code-Stabilizer

**Reviewer role:** Principal Software Architect & Lead Security Researcher
**Repo HEAD analysed:** `29e8c5b` (branch `main`)
**Methodology:** All non-`vendor/` Python files traversed for top-level `class` / `def` / `import` declarations via ripgrep over the working tree. Counts and citations are line-grounded; nothing in this report is asserted unless it appears in the repo at the cited path. Where the user's brief named a file or symbol that does **not** exist in the tree, the absence is recorded verbatim as a **Gap** with the negative grep result that proved it.

---

## 0. Methodology Compliance Notice (must read first)

Several artefacts named in the analysis brief do not exist in the cloned tree at `29e8c5b`. Per the user's "no hallucinations" instruction, they are recorded here as **Gaps**, not as analysed components:

| Brief reference | Status in repo | Evidence |
|---|---|---|
| `vendor/openclaude/` (the "OpenClaude engine", 2,500+ files) | **ABSENT** | `find . -type d -name vendor` → empty. `rg -i openclaude` → 0 hits. |
| `hermes_orchestrator.py` | **ABSENT** | `find . -name 'hermes*'` → empty. Only token: `verification/independence_enforcer.py:180:    "hermes": "nous"` (model-family map entry). |
| `ToolSearchTool.ts` and any TypeScript MCP tooling | **ABSENT** | `find . -name '*.ts'` → 0 results. Metadata snapshot below shows zero `.ts`/`.tsx` files. |
| Zod validation / "String vs. Number" coercion fix | **ABSENT** | No `.ts` files exist; `rg -i "zod"` → 0 hits. |
| gRPC port `50051` / `OpenClaudeAgent` references | **ABSENT** | `rg "50051"` → 0 hits. `rg OpenClaudeAgent` → 0 hits. The 16 `grpc` hits all live inside `cpg/` and refer to **inbound** Joern/IDL handling (parsing `.proto`, filtering generated `_pb2_grpc.py`), not outbound calls to an OpenClaude engine. |
| Z3 SMT checks in the Red-Team gate | **ABSENT** | `rg -i "z3|z3-solver|smtlib"` → 0 hits in source. The repo's formal layer is **CBMC** (bounded model checking, C/C++) and **Lean 4** (advisory), not Z3. See §V. |
| Agent IDs `python-debug`, `security-researcher`, `adversarial-reviewer` | **PARTIAL** | None registered as agent IDs. `Adversarial_Reviewer` exists only as an AutoGen `AssistantAgent` *name* in `swarm/autogen_agents.py:114`. `security researchers` appears as prose in `tools/servers/codeql_server.py:19`. `python-debug` → 0 hits. See §II.B. |
| Maintainer-email / OSINT discovery | **ABSENT** | `rg -i "maintainer.email\|osint\|find_maintainer"` → 0 hits. See §III.A. |

The body that follows analyses what **does** exist.

---

## I. Metadata Snapshot

Counts produced by `find … -type f -name '*.<ext>'` then `wc -l` and `du -bc`. Only extensions with non-zero counts are listed.

| Extension | Files | Lines | Bytes |
|---|---:|---:|---:|
| `.py` | 210 | 79,481 | 3,310,664 |
| `.md` | 13 | 2,443 | 104,696 |
| `.yml` | 7 | 501 | 19,350 |
| `.toml` | 4 | 356 | 11,290 |
| `.yaml` | 1 | 80 | 2,295 |
| `.sh` | 1 | 165 | 6,354 |
| `.rs` | 1 | 280 | 10,472 |
| **`.ts` / `.tsx` / `.js` / `.jsx`** | **0** | **0** | **0** |

Total tracked files (excluding `.git/`): **249**.

**Top-level directories:**
`agents/  api/  auth/  brain/  compliance/  config/  context/  cpg/  disclosure/  discovery/  docs/  escalation/  github_integration/  intelligence/  mcp_clients/  memory/  metrics/  models/  orchestrator/  plugins/  rust/  sandbox/  scripts/  security/  startup/  swarm/  swe_bench/  tests/  tools/  ui/  utils/  verification/  workers/`

Heuristic AST tally over all `.py`: 551 `class` declarations and 579 `def` / `async def` declarations (`rg "^class "` and `rg "^(async )?def "`).

---

## II. Analysis Dimensions

### II.A — The Orchestrator Brain

The brief asks about `hermes_orchestrator.py` and `app.py`. **`hermes_orchestrator.py` does not exist** (see §0). The two files actually performing those roles are:

| Brief role | Actual file in repo | Lines |
|---|---|---:|
| Orchestrator | `orchestrator/controller.py` | 3,911 |
| App entry | `api/app.py` | 484 |
| Single-pass workflow runner | `swarm/deerflow_orchestrator.py` | 305 |

#### `orchestrator/controller.py` — `StabilizerController`

- Imports (lines 1–73, verbatim): `AuditorAgent, FixerAgent, FormalVerifierAgent, PatrolAgent, PlannerAgent, ReaderAgent, ReviewerAgent, TestRunnerAgent` (worker pool); `DependencyGraphEngine, SQLiteBrainStorage, PostgresBrainStorage, VectorBrain` (memory); `EscalationManager, PRManager, MCPManager`; `ConsensusEngine, ConvergenceDetector`; `StaticAnalysisGate`; `AegisEDR`; `IndependenceEnforcer`; `TestGeneratorAgent, MutationVerifierAgent, HybridRetriever`; `SynthesisAgent, LocalizationAgent`; and the optional `AdversarialCriticAgent` + `BoBNSampler` block guarded by `_GAP5_AVAILABLE` (lines 70–76).
- `class StabilizerConfig(BaseModel)` (line 84) is the runtime config and pins model defaults: `primary_model="openai/Qwen/Qwen2.5-Coder-32B-Instruct"`, `critical_fix_model="openrouter/meta-llama/llama-4-scout"`, `triage_model="ollama/granite-code:3b"`, `reviewer_model="ollama/qwen2.5-coder:32b"`, `max_cycles=200`, `cost_ceiling_usd=50.0`, `concurrency=4`, `autonomy_level=AutonomyLevel.AUTO_FIX`, `auto_commit=True`, `base_branch="main"`, `branch_prefix="stabilizer"`.
- `SURGICAL_PATCH_THRESHOLD = 2_000` (line 81): files larger than 2 000 lines switch to `UNIFIED_DIFF` patching mode.
- The CPG block (lines 56–63) imports `CPGEngine, get_cpg_engine, ProgramSlicer, CPGContextSelector, IncrementalCPGUpdater` inside a `try/except ImportError` and exposes `_CPG_AVAILABLE` for downstream branching.

#### `api/app.py` — FastAPI surface

- Lines 32–37 derive an explicit CORS allow-list from `RHODAWK_CORS_ORIGINS` (defaults to `http://localhost:3000,http://localhost:8000`); the file's own header comment (lines 1–17) labels this a fix to a previous wildcard `["*"]`.
- `_enforce_production_security()` (line 64) hard-exits via `sys.exit(msg)` if `RHODAWK_DEV_AUTH=1` outside of `RHODAWK_ENV=development` (lines 88–98), and similarly exits if `RHODAWK_WEBHOOK_SECRET` is unset in production (lines 100–112) or if a `HS256` JWT secret is missing/`CHANGE_ME*` (lines 114+).
- `_AppState` (line 43) tracks `storage` and `cpg_available`. `get_storage()` (line 56) raises `HTTPException(503, "Storage not initialised. Call /api/runs/start first.")` when accessed before init.
- Routes registered under `api/routes/`: `auth.py, commits.py, compound_findings.py, escalations.py, federation.py, files.py, fixes.py, github_webhook.py, issues.py, refactor_proposals.py, runs.py, upload.py`. Realtime channel under `api/websocket/manager.py`.

#### "Lead Researcher" vs. worker agents — what the code actually says

The brief asks how a "Lead Researcher" personality differs from worker agents. **There is no `LeadResearcher` class, role, or agent_id in the codebase** (`rg -i "lead.researcher\|LeadResearch"` → 0 hits). What exists instead, that *plays a coordinating role* over single-purpose worker agents, is split across three places:

1. **`orchestrator/controller.py:StabilizerController`** is the only object that owns every worker, the storage, the consensus engine, the convergence detector, the escalation manager, and the PR manager. All workers (`FixerAgent`, `ReviewerAgent`, `AuditorAgent`, etc. — see §II.B) are constructed and called from this controller. There is no per-worker LLM persona prompt that names them "researcher".
2. **`swarm/deerflow_orchestrator.py`** (`class DeerFlowOrchestrator`, header comment lines 1–28) declares: *"DeerFlow is the ONLY orchestration path. Classic and LangGraph paths removed."* It executes one DAG pass per cycle (`READ → AUDIT → CONSENSUS → FIX → REVIEW → GATE → COMMIT → REINDEX`); the controller's outer convergence loop calls `DeerFlowOrchestrator.run()` per cycle. It uses `WorkflowStep` dataclasses with `depends_on`, `max_retry=3`, and `timeout_s=600.0` (lines 51+).
3. **CrewAI roles in `swarm/crew_roles.py`** are the closest thing to named "personalities". They are *role strings passed to `crewai.Agent(role=…, goal=…, backstory=…)`*, not orchestrator personalities. `build_security_crew()` (line 70) instantiates three roles: `"Senior Threat Modeler"` (line 79), `"Vulnerability Analyst"` (line 91), `"Remediation Engineer"` (line 102). None of them is labelled "Lead Researcher".

**Verdict:** the difference between an "orchestrator personality" and a "worker personality" in this repo is structural (the controller owns state and dispatches; workers are stateless `BaseAgent` subclasses), **not** prompt-driven. There is no specific "Lead Researcher" prompt asset.

---

### II.B — The Swarm Roster

`agents/` files (line counts from `wc -l`):

| File | Lines | Class observed |
|---|---:|---|
| `agents/base.py` | 581 | `BaseAgent`, `AgentConfig` |
| `agents/auditor.py` | 642 | `AuditorAgent` |
| `agents/fixer.py` | 1,499 | `FixerAgent` |
| `agents/reviewer.py` | 189 | `ReviewerAgent` |
| `agents/planner.py` | 190 | `PlannerAgent` (`agent_type = ExecutorType.PLANNER`, line 47) |
| `agents/reader.py` | 532 | `ReaderAgent` |
| `agents/patrol.py` | 89 | `PatrolAgent` |
| `agents/test_generator.py` | 463 | `TestGeneratorAgent` |
| `agents/test_runner.py` | 673 | `TestRunnerAgent` |
| `agents/test_runner_universal.py` | 690 | universal runner |
| `agents/mutation_verifier.py` | 315 | `MutationVerifierAgent` |
| `agents/formal_verifier.py` | 1,482 | `FormalVerifierAgent` |
| `agents/synthesis_agent.py` | 1,268 | `SynthesisAgent` |
| `agents/patch_synthesis_agent.py` | 409 | `PatchSynthesisAgent` |
| `agents/localization_agent.py` | 457 | `LocalizationAgent` |
| `agents/adversarial_critic.py` | 581 | `AdversarialCriticAgent` |

#### Brief-named IDs vs. reality

| Brief ID | Status | Evidence |
|---|---|---|
| `python-debug` | **GAP** — not registered | `rg -i "python.debug"` → 0 hits |
| `security-researcher` | **GAP** — not registered | The phrase appears once as prose in `tools/servers/codeql_server.py:19` ("queries are written by security researchers"). No agent class, no registry entry. |
| `adversarial-reviewer` | **PARTIAL** — exists only inside an AutoGen conversation, not as a controller-level agent | `swarm/autogen_agents.py:114-116`: `autogen.AssistantAgent(name="Adversarial_Reviewer", system_message="You are a hostile code reviewer…")`. This is a transient role created inside `run_code_review_conversation()` when `_AUTOGEN_AVAILABLE` (line 38) is True; it is not constructed by `StabilizerController` and has no `ExecutorType` enum value. |

The repo's *de facto* "adversarial reviewer" is the controller-owned `AdversarialCriticAgent` in `agents/adversarial_critic.py`. Its file header (lines 1–62) names a "VIB-01 FIX (Glasswing Red-Team Audit, 2026-04-13)" and pins the architecture to: `Fixer A: Qwen2.5-Coder-32B`, `Fixer B: DeepSeek-Coder-V2-16B`, `Critic: Llama-3.3-70B-Instruct`. Determinism is enforced by `_CRITIC_TEMPERATURE: float = 0.0` (line 78, comment "IMMUTABLE — do not add env-override") and integer-millipoint scoring weights `_W_TEST_MILLIS=600`, `_W_ROBUSTNESS_MILLIS=300`, `_W_MINIMALITY_MILLIS=100` with an `assert sum == 1000` at line 73.

#### Where agents are wired in

There is no central `AGENT_REGISTRY` dict. Agents are instantiated **directly** in `orchestrator/controller.py` (the constructor block driven by the import list at lines 13–30 and 49–76). Optional ensembles are gated by:

- `_CPG_AVAILABLE` (controller.py line 63)
- `_GAP5_AVAILABLE` (controller.py line 76) → enables `AdversarialCriticAgent` + `BoBNSampler`

CrewAI roles in `swarm/crew_roles.py` and AutoGen roles in `swarm/autogen_agents.py` are **conditionally available**: both files set `_CREWAI_AVAILABLE` / `_AUTOGEN_AVAILABLE` from a `try: import` (crew_roles.py lines 26–35; autogen_agents.py lines 36–45) and degrade to no-ops if the dep is absent.

---

### II.C — The Tooling (MCP)

The brief says "23+ servers". The actual count in the tree is **18 Python MCP servers + 1 Rust MCP server = 19**.

```
$ ls tools/servers/*.py | grep -v __init__ | wc -l
18
```

#### Python MCP servers (`tools/servers/`)

`afl_server.py, antigravity_server.py, aurite_server.py, codeql_server.py, cve_server.py, huggingface_skills_server.py, infer_server.py, joern_server.py, jujutsu_server.py, ldra_polyspace_server.py, mariana_trench_server.py, mirofish_server.py, nuclei_server.py, openviking_server.py, promptfoo_server.py, sbom_server.py, semgrep_server.py, trufflehog_server.py`

#### Rust MCP server (`rust/mcp_server/src/`, single `.rs` file, 280 lines)

Header (lines 1–10): *"rhodawk-mcp — High-performance MCP server for Rhodawk AI static analysis. Provides: lint_file, complexity, dependencies, dangerous_patterns. Transport: stdio (MCP JSON-RPC 2.0)"*. Defines `JsonRpcRequest`, `JsonRpcResponse`, `JsonRpcError`, `LintIssue`, `DependencyInfo`, and a `DANGEROUS_PATTERNS: &[(&str, &str, &str)]` table (`eval(`, `exec(`, `__import__(`, `pickle.loads(`, `os.system(`, `subprocess.call(`, `md5(`, `sha1(` — all CRITICAL/HIGH/MEDIUM).

#### Container brokerage — `tools/toolhive.py`

`class ToolHive` (line 18) shells out to `docker run --rm` with `_RESOURCE_FLAGS = ["--memory","2g","--cpus","2","--pids-limit","100"]` (line 11) and `_NETWORK_NONE = ["--network","none"]` (line 13). The `_NETWORK_ONLY_TOOLS` set (line 14) — `{"semgrep","cve_lookup","sbom","github","joern"}` — is the **only** group that gets network access; everything else runs network-isolated. Joern gets a higher resource ceiling: `_JOERN_RESOURCE_FLAGS = ["--memory","10g","--cpus","4","--pids-limit","200"]` (line 12).

#### "Zod validation schemas" / "String vs. Number coercion fix"

**GAP — not in repo.** There are 0 `.ts` files (see §I) and `rg -i "zod"` returns 0 hits. The Python equivalent for input validation is **Pydantic** (`pydantic.BaseModel`, used pervasively in `brain/schemas.py`, `agents/planner.py`, `orchestrator/controller.py`). Determinism is enforced not by Zod coercion but by the integer-only ranking pipeline in `agents/adversarial_critic.py` (the VIB-01 FIX block at the top of that file replaces float `attack_confidence` with `attack_severity_ordinal: int`) and by the same change mirrored in `swe_bench/bobn_sampler.py` (header lines 1–32: `composite_score_int: int`, `_patch_tiebreaker: str`, `ranking_key()` returns `(-composite_score_int, _patch_tiebreaker)`).

#### Pinned ruleset hardening — `tools/servers/semgrep_server.py`

Lines 1–32 document and implement a deliberate refusal of `semgrep --config=auto`. The resolution order, taken verbatim from the file header: `RHODAWK_SEMGREP_RULES env var → <repo>/.semgrep/ → <repo>/.semgrep.yml → bundled tools/rules/semgrep/ → fallback p/owasp-top-ten only when RHODAWK_ALLOW_SEMGREP_REGISTRY=1`. The same file states: *"It NEVER silently falls back to --config=auto."*

---

### II.D — The Security Pipeline (Failed Test → Red Team gate)

The brief asks where a failed test gets intercepted by `adversarial_reviewer.py` and Z3 math checks before reaching "the Boss (Max)". Mapping to the actual code:

- **`adversarial_reviewer.py`** — does not exist. The intercepting agent is `agents/adversarial_critic.py:AdversarialCriticAgent`.
- **Z3** — does not exist. The "math check" layer is **CBMC** (referenced as a hard layer in `swe_bench/bobn_sampler.py` header lines 79–87) plus advisory **Lean 4** via `verification/leanstral.py`.
- **"Max" / "the Boss"** — does not exist as an identifier in the code. The actual blocking authority is the human approver wired through `escalation/human_escalation.py:EscalationManager` and the `BASELINE_PENDING` `RunStatus` returned by `orchestrator/convergence.py:ConvergenceDetector.suggest_status()` (lines 137+: *"Require human promotion — not auto-approved"*).

#### Actual flow when tests fail

The pipeline that *does* exist is documented in `swe_bench/bobn_sampler.py` (header lines 60–92) and implemented across `orchestrator/controller.py`, `agents/adversarial_critic.py`, `agents/patch_synthesis_agent.py`, and `agents/formal_verifier.py`:

1. **GENERATE** — Fixer A (Qwen-32B) and Fixer B (DeepSeek-Coder-V2-16B) emit candidates at temperatures `0.2/0.4/0.6` and `0.3/0.7` (header lines 65–69).
2. **EXECUTE** — `swe_bench/execution_loop.py` (file present, 1 file in `swe_bench/`) drives `MAX_ROUNDS test→observe→revise` iterations; pass-rate is measured against `FAIL_TO_PASS` tests.
3. **ATTACK** — `AdversarialCriticAgent` runs concurrently against every candidate. It returns `CriticAttackReport` per candidate. `compute_deterministic_composite(test_component, robustness_component, minimality_component) → int millipoints` is the only ranking signal (no float arithmetic).
4. **RANK** — `BoBNCandidate.ranking_key()` returns `(-composite_score_int, _patch_tiebreaker)` where `_patch_tiebreaker = sha256(patch)[:16]`. (Bobn header lines 21–28.)
5. **SYNTHESIZE** — `agents/patch_synthesis_agent.py:PatchSynthesisAgent` either picks the top candidate (`PICK_BEST`) or merges hunks across candidates (`MERGE`). Per the bobn header (lines 73–78), this agent is constrained to a *third* model family (CLOUD_OSS Devstral/Llama-4) to preserve independence from both fixers and the critic.
6. **FORMAL GATE** — bobn header lines 79–87 list four layers: (L1) structural diff sanity, (L2) safety-pattern scan on `+` lines (unbounded loops, `shell=True`, …), (L3) **CBMC** bounded model checking on C/C++ patches (hard gate, non-blocking when CBMC is absent or times out), (L4) — the header truncates here; remainder is in `agents/formal_verifier.py` (1,482 lines) which the brief did not name.
7. **CONSENSUS GATE** — `orchestrator/consensus.py:ConsensusEngine` (508 lines). `_BASE_CONFIDENCE_FLOORS` (line 47) defines per-domain floors: `MILITARY/AEROSPACE 0.90`, `NUCLEAR 0.92`, `MEDICAL 0.85`, `FINANCE 0.80`, `EMBEDDED 0.82`, `GENERAL 0.70`. `DEFAULT_RULES` (line 91) escalates **CRITICAL** issues with `disagreement_action=DisagreementAction.ESCALATE_HUMAN`, `minimum_agents=2`, `confidence_floor=0.75`, `required_domains=[ExecutorType.SECURITY]`.
8. **HUMAN ESCALATION** — `escalation/human_escalation.py` (449 lines). Header (lines 11–22): *"AUDIT FIX: ConsensusEngine raised DisagreementAction.ESCALATE_HUMAN but StabilizerController had no implementation."* `EscalationManager`, `NotificationDispatcher`, `ApprovalPoller` (header line 16: *"async loop that blocks the pipeline until resolved"*), `verify_approval_signature()` (line 431, HMAC-SHA256, requires `RHODAWK_AUDIT_SECRET`). Channels (lines 63–66): `RHODAWK_SLACK_WEBHOOK_URL`, `RHODAWK_EMAIL_WEBHOOK_URL`, `RHODAWK_ESCALATION_WEBHOOKS`, `RHODAWK_PAGERDUTY_ROUTING_KEY`. **Refusal-to-proceed clause (lines 95–105):** if no channels are configured, `notify()` raises `RuntimeError("No escalation notification channels configured. … Refusing to proceed silently past a human-in-the-loop safety gate.")`. Timeout default (line 47): `RHODAWK_ESCALATION_TIMEOUT_H=24` hours; *timeouts do not auto-approve* (header lines 19–20) — they move to `EscalationStatus.TIMEOUT` and `IssueStatus.DEFERRED`.

#### Reviewer independence

`verification/independence_enforcer.py` (361 lines) is the layer that prevents the Reviewer from being from the same model family as the Fixer. Header (lines 1–22): *"DO-178C 6.3.4 states: 'The software verification process activities shall be performed by a person or tool that is independent of the developer of the software being verified.' For LLM-based systems, independence means the reviewer model must come from a different organization's training pipeline."* Family lookup is loaded from `verification/model_registry.yaml` (entries observed: `claude→anthropic`, `llama→meta`, `mistral/devstral/mixtral→mistral`, `granite→ibm`, `qwen→alibaba`, `gemini/palm→google`, `gpt/o1/o3→openai`, `deepseek→deepseek`, `command→cohere`, `phi→microsoft`, `grok→xai`, `falcon→tii`, `starcoder→bigcode`, `stable→stability`, `titan→amazon`, `yi→…`). `IndependenceViolationError` (line 60) is raised in strict mode (`military/aerospace/nuclear`).

---

## III. The "Employee" Evaluation

The brief asks three yes/no questions. Each is answered strictly from grep evidence.

### III.A — Autonomous Disclosure: is there logic to find maintainer emails (OSINT)?

**No.** Negative grep:

```
$ rg -in "maintainer\.email|find_maintainer|osint|whois|harvester|theHarvester" --type py
(no output)
```

What *does* exist for disclosure is **`disclosure/security_advisory.py`** (529 lines, `class SecurityAdvisory`). It assumes the maintainer surface is GitHub-native:

- `create_private_advisory()` (line ~110) POSTs a private GHSA via the GitHub REST API (`/repos/{owner}/{name}/security-advisories`) using a token passed at construction (`github_token` arg or `GITHUB_TOKEN` env). Required scopes documented at lines 84–85: `security_events:write, pull_requests:write, contents:write, repository_advisories:write`.
- The disclosure deadline counter is implemented (line 56): `_DISCLOSURE_DEADLINE_DAYS = int(os.environ.get("RHODAWK_DISCLOSURE_DEADLINE_DAYS","90"))` — Project-Zero-style 90-day clock.
- Notification channels for human-in-the-loop are in `escalation/human_escalation.py:NotificationDispatcher` (Slack/email-webhook/PagerDuty/generic webhook) but `_email_url` is read as `RHODAWK_EMAIL_WEBHOOK_URL` (line 63) — a webhook URL, not an SMTP target. There is no SMTP client, no `whois` lookup, no GitHub-org member crawl, no commit-history email harvesting.

### III.B — Approval Hooks: is there a "Pause for Boss" before `git push` or PR submission?

**Yes for PR submission, partial for `git push`.** Citations:

- **PR submission gate.** `escalation/human_escalation.py:ApprovalPoller` (header line 16) explicitly *"blocks the pipeline until resolved"*. `EscalationManager.approve()` (line 306) and `EscalationManager.reject()` (line 349) are the unblock points; `approve()` sets `EscalationStatus.APPROVED`, `approved_by`, `approved_at`, and `approval_rationale`. Approval webhook bodies are HMAC-verified by `verify_approval_signature()` (line 431).
- **Consensus → escalate path.** `orchestrator/consensus.py` `DEFAULT_RULES[Severity.CRITICAL]` (line 91) sets `disagreement_action=DisagreementAction.ESCALATE_HUMAN` — every CRITICAL with disagreement is forced through the approval gate before the controller can commit.
- **Convergence → human promotion.** `orchestrator/convergence.py:ConvergenceDetector.suggest_status()` (lines 134+) returns `RunStatus.BASELINE_PENDING` for both `score_stable` and `zero_critical_issues` halts, with the in-file comment *"Require human promotion — not auto-approved"*. Promotion to baseline is an explicit human action exposed via `api/routes/runs.py`.
- **`git push` itself.** `github_integration/pr_manager.py:PRManager._push_branch()` (lines 38–76) calls `git push origin HEAD:{branch_name} --force-with-lease` *unconditionally* before opening the PR. It is gated only by the `--force-with-lease` safety, not by a human-approval check at the push line. The human gate is upstream — if the run hits a `CRITICAL` escalation, the controller never reaches `_create_pr_sync()`. The push is also subject to the `auto_commit: bool = True` flag in `StabilizerConfig` (controller.py line 117) and to `AutonomyLevel`. The `AutonomyLevel` enum (`brain/schemas.py:99`) defines five rungs:
  ```python
  READ_ONLY, SUGGEST, AUTO_FIX, AUTO_FIX_PR, FULL_AUTONOMOUS
  ```
  `READ_ONLY` is honoured at `orchestrator/controller.py:3182` (`or self.config.autonomy_level == AutonomyLevel.READ_ONLY`). There is no per-call `await human_confirm_push(branch)` shim — i.e. the model is "fail to escalation upstream", not "always pause at the push line".

### III.C — Briefing Logic: are "Executive Summaries" generated, vs. raw test logs?

**Yes, in three places — but none of them is named "Executive Summary".**

1. **`disclosure/security_advisory.py`** (header lines 19–32): the security PR body is enriched with *"CVE/CWE/GHSA cross-references; CVSS v3.1 base score and vector; attack scenario; impact statement; PoC indicator; fix rationale; testing evidence; reviewer checklist (DO-178C / MISRA compliance items if applicable)"*. Implemented by `_build_advisory_description(issue, fix, cve_match)` (called from `create_private_advisory`, line ~134). Field `cwes` is filtered to entries that `startswith("CWE-")` (line ~146). CVSS only attached if a real `cve_match.cvss_v3_vector` exists (line ~150) — no fabricated scores.
2. **`compliance/rtm_exporter.py` and `compliance/sas_generator.py`** — the two compliance-export modules (`compliance/__init__.py, rtm_exporter.py, sas_generator.py`). Their existence is wired into `api/app.py` per the file's docstring (lines 14–15): *"RTM and SAS export endpoints wired to RTMExporter / SASGenerator."* RTM = Requirements Traceability Matrix; SAS = Software Accomplishment Summary. These are formal compliance artifacts, not free-form executive briefs, but they are higher-order summaries built from raw findings.
3. **`/api/capabilities`** (referenced in `api/app.py` header line 16): *"`/api/capabilities` returns structured feature matrix report."* Implemented via `startup/feature_matrix.py:verify_startup()` (called from `run.py` and `api/app.py`).

What does **not** exist: an `executive_summary()` function, a `brief_for_human()` function, or any module that converts test-runner logs into a paragraph-form summary. Search:

```
$ rg -in "executive.summary|brief_for_human|summary_for_human" --type py
(no output)
```

---

## IV. Output Format — Per-directory map and execution flow

### Per-directory map (line counts where the file was inspected for this report)

#### `agents/` — 16 files, all `BaseAgent`-shaped
See table in §II.B.

#### `orchestrator/` — 5 files
| File | Lines | Notes |
|---|---:|---|
| `controller.py` | 3,911 | `StabilizerController`, `StabilizerConfig` |
| `commit_audit_scheduler.py` | 983 | scheduled re-audit of commits |
| `consensus.py` | 508 | `ConsensusEngine`, `DEFAULT_RULES`, `_BASE_CONFIDENCE_FLOORS` |
| `convergence.py` | 149 | `ConvergenceDetector`, regression / baseline-pending logic |
| `controller_helpers.py` | 95 | helpers |
| `patch_transaction.py` | 473 | atomic patch apply/rollback |

#### `swarm/` — 5 files
| File | Lines | Notes |
|---|---:|---|
| `crew_roles.py` | 295 | CrewAI roles (`build_security_crew`, others), gated by `_CREWAI_AVAILABLE` |
| `autogen_agents.py` | 347 | AutoGen `Adversarial_Reviewer`/`Fix_Author` conversation; `_MAX_TURNS=15` |
| `langgraph_state.py` | 314 | LangGraph state graph (gated by `_LANGGRAPH_AVAILABLE`) |
| `deerflow_orchestrator.py` | 305 | bespoke async DAG executor — "the ONLY orchestration path" |
| `hf_skills.py` | 198 | HuggingFace skills shim |

#### `brain/` — 9 files
| File | Lines | Notes |
|---|---:|---|
| `schemas.py` | 1,029 | All Pydantic models incl. `AutonomyLevel`, `RunStatus`, `Issue`, `EscalationRecord`, `ConsensusResult` |
| `sqlite_storage.py` | 2,580 | dev/test backend |
| `postgres_storage.py` | 983 | prod backend |
| `storage.py` | 402 | abstract `BrainStorage` |
| `hybrid_retriever.py` | 448 | dense+sparse retrieval |
| `vector_store.py` | 175 | Qdrant/HelixDB shim |
| `graph.py` | 367 | `DependencyGraphEngine` |
| `migrations.py` | 88 | DB migrations |

#### `verification/` — 3 source files + 1 yaml
| File | Lines | Notes |
|---|---:|---|
| `independence_enforcer.py` | 361 | `IndependenceViolationError`, `verify_model_identity` |
| `leanstral.py` | 117 | `llm_reason_property()` (advisory). Header (lines 1–20): *"Previous version claimed to produce 'Lean 4 formal proofs' via a fictional tool 'Leanstral.' … This version is honest."* `lean_prove()` retained as deprecated alias. |
| `model_registry.yaml` | 80 | model→family map + checksums |

#### `escalation/`, `disclosure/`, `github_integration/` — see §II.D and §III

#### `swe_bench/` — 6 files
`bobn_sampler.py, execution_loop.py, evaluator.py, localization.py, terminal_bench.py, trajectory_collector.py`. Header of `bobn_sampler.py` (lines 39–48) cites *"Agent S3 paper (arxiv 2410.02052)"* and the empirical lift estimate (12–18 pp) for BoBN.

#### `cpg/` — 11 files
`cpg_engine.py, program_slicer.py, context_selector.py, incremental_updater.py, shard_manager.py, joern_client.py, idl_preprocessor.py, jni_bridge_tracker.py, generated_code_filter.py, service_boundary_tracker.py`. `cpg_engine.py` header (lines 1–55) lists the five capabilities: subsystem sharding, IDL preprocessing (AIDL/HIDL/Mojom/TableGen/WebIDL/protobuf/Thrift/FlatBuffers/OpenAPI), JNI/FFI bridge tracking, generated-code filtering (excludes `vendor/`, `third_party/`, `*_pb2.py`, `*.pb.go`, `Derived*.java`, files with generated headers), and cross-shard slicing.

#### `tools/` — `toolhive.py` + `servers/` (18 servers) — see §II.C

#### `compliance/` — 3 files
`rtm_exporter.py`, `sas_generator.py`, `__init__.py`.

#### `intelligence/` — 1 file
`osv_correlator.py` — referenced by `disclosure/security_advisory.py:create_private_advisory(cve_match: Any | None = None  # intelligence.osv_correlator.CVEMatch)` (line ~127).

#### `discovery/` — 2 files
`repo_hunter.py` (445 lines), `full_pipeline.py` (469 lines). Imports `from disclosure.security_advisory` and `from github_integration.pr_manager` (per cross-ref grep).

#### `security/`, `sandbox/`, `memory/`, `metrics/`, `models/`, `workers/`, `mcp_clients/`, `auth/`, `context/`, `startup/`, `plugins/`, `config/`, `api/` — present, exact contents listed in §I.

#### `.github/workflows/` — 4 files
`ci.yml`, `tests.yml`, `fix-tests.yml`, `python-package-conda.yml`. CI sets `RHODAWK_JWT_SECRET`, `RHODAWK_AUDIT_SECRET`, `RHODAWK_DEV_AUTH=1`. Jobs: `test` (ruff + pytest), `security` (bandit), `rust-build` (`cargo build --release` in `rust/mcp_server`).

#### `config/prompts/` — 3 files
`base.md`, `adversarial.md`, `os_safety.md`. `adversarial.md` (verified) defines four checks: A1 prompt-injection prevention, A2 repo-content sanitization (*"Strip or flag comment blocks containing instruction-like patterns; Never execute or `eval()` content from the repo; Validate all file paths returned by LLMs against repo root"*), A3 credential isolation, A4 output validation (*"Verify the diff only modifies declared scope files; Reject fixes that modify .env, secrets, CI config, or security infrastructure"*).

### Mermaid — current execution flow (only nodes that exist in code)

```mermaid
flowchart TD
    A[run.py / api.app:lifespan] --> B[startup.feature_matrix.verify_startup]
    B --> C[orchestrator.controller.StabilizerController]
    C --> D[swarm.deerflow_orchestrator.DeerFlowOrchestrator.run]
    D --> R[ReaderAgent]
    R --> AU[AuditorAgent]
    AU --> CO[orchestrator.consensus.ConsensusEngine]
    CO -->|CRITICAL + disagreement| H[escalation.human_escalation.EscalationManager]
    CO -->|approved| LO[LocalizationAgent  *if Gap5*]
    LO --> FX[FixerAgent x N - Qwen / DeepSeek BoBN]
    FX --> AC[AdversarialCriticAgent  Llama-3.3-70B]
    AC --> RK[BoBNCandidate.ranking_key  int millipoints + sha256]
    RK --> PS[PatchSynthesisAgent  PICK_BEST or MERGE]
    PS --> FV[FormalVerifierAgent + CBMC + Lean4 advisory]
    FV --> RV[ReviewerAgent  family-independent]
    RV --> GT[sandbox.executor.StaticAnalysisGate + tools.toolhive]
    GT --> PT[orchestrator.patch_transaction]
    PT --> CV[orchestrator.convergence.ConvergenceDetector]
    CV -->|score_stable / zero_critical| BP[RunStatus.BASELINE_PENDING - human promotion]
    CV -->|regression| FAIL[RunStatus.FAILED]
    PT --> PR[github_integration.pr_manager.PRManager._push_branch then create_pr]
    PR --> SA[disclosure.security_advisory.SecurityAdvisory  GHSA + enriched PR body]
    H -.HMAC approve/reject.-> CO
```

---

## V. Configuration & Runtime — observed environment surface

Variables that the code actually reads (`os.environ.get` / `os.environ[…]`):

| Variable | Read in | Effect |
|---|---|---|
| `RHODAWK_ENV` | `api/app.py:38` | `production` (default) vs `development`; gates startup hard-exits |
| `RHODAWK_DEV_AUTH` | `api/app.py:80,89` | `1` outside `development` → `sys.exit` |
| `RHODAWK_WEBHOOK_SECRET` | `api/app.py:100` | required in prod |
| `RHODAWK_JWT_SECRET` | `api/app.py:114` | rejected if `CHANGE_ME*` under HS256 |
| `RHODAWK_JWT_ALGORITHM` | `api/app.py:113` | `RS256` default |
| `RHODAWK_CORS_ORIGINS` | `api/app.py:32` | explicit allow-list |
| `RHODAWK_AUDIT_SECRET` | `escalation/human_escalation.py:441` | HMAC-SHA256 key for approval webhooks |
| `RHODAWK_SLACK_WEBHOOK_URL` / `RHODAWK_EMAIL_WEBHOOK_URL` / `RHODAWK_ESCALATION_WEBHOOKS` / `RHODAWK_PAGERDUTY_ROUTING_KEY` | `escalation/human_escalation.py:63-67` | escalation channels |
| `RHODAWK_ESCALATION_TIMEOUT_H` | `escalation/human_escalation.py:47` | default 24 h |
| `RHODAWK_ESCALATION_POLL_S` | `escalation/human_escalation.py:48` | default 30 s |
| `RHODAWK_DISCLOSURE_DEADLINE_DAYS` | `disclosure/security_advisory.py:56` | default 90 |
| `RHODAWK_PRIMARY_MODEL` | (per controller.py comment line 90) | overrides `primary_model` |
| `RHODAWK_TOOLHIVE_REGISTRY` | `tools/toolhive.py:21` | docker image prefix; defaults to `ghcr.io/rhodawk` |
| `RHODAWK_SEMGREP_RULES` | `tools/servers/semgrep_server.py` (resolution order block) | pinned ruleset path |
| `RHODAWK_ALLOW_SEMGREP_REGISTRY` | same | `1` to allow `p/owasp-top-ten` fallback |
| `JOERN_URL` | `tools/toolhive.py:cpg_health_check` | default `http://localhost:8080` |
| `OLLAMA_BASE_URL`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | `swarm/autogen_agents.py:_llm_config` | model routing |
| `AUTOGEN_CACHE_DIR`, `AUTOGEN_MAX_TURNS` | `swarm/autogen_agents.py:46-47` | defaults `.stabilizer/autogen_cache`, `15` |
| `GITHUB_TOKEN` | `disclosure/security_advisory.py:99` | fallback when `github_token` arg empty |

A `.env` file and an `.env.example` are present at the repo root; their contents are not reproduced here.

---

## VI. Conclusion — 24-Hour Priority Action List

This list is bound to evidence above. Each item closes a *named gap from the brief* by adding code that today does not exist.

| # | Action | Why (evidence) |
|---|---|---|
| 1 | **Decide on the `vendor/openclaude/` story.** Either (a) add the submodule/copy and document it, or (b) delete the brief's claim. Today the directory is absent and there are zero `OpenClaudeAgent`/`50051`/`grpc` outbound references. | §0, §II.A |
| 2 | **Rename or alias `agents/adversarial_critic.py` → `adversarial_reviewer`** *and* register `python-debug` and `security-researcher` as real `ExecutorType` members in `brain/schemas.py`, with classes under `agents/`. Right now those IDs only exist as prose. | §II.B |
| 3 | **Replace the AutoGen-only `Adversarial_Reviewer` role** with a controller-instantiated agent so adversarial review is mandatory rather than `_AUTOGEN_AVAILABLE`-gated. | `swarm/autogen_agents.py:38-45,114` |
| 4 | **Add a TypeScript MCP layer with Zod schemas** if the brief's "ToolSearchTool.ts / Zod / String-vs-Number coercion" requirement is real. There are 0 `.ts` files today; the Python equivalent is Pydantic in `brain/schemas.py`. | §I, §II.C |
| 5 | **Decide whether Z3 belongs in the gate.** Today the formal layer is CBMC + advisory Lean 4. If Z3 SMT checks are required for the Red-Team gate, add a `verification/z3_gate.py` and wire it into `agents/formal_verifier.py`. | §II.D |
| 6 | **Add maintainer-discovery (OSINT).** Today `disclosure/security_advisory.py` only knows the GitHub REST API. Add a `discovery/maintainer_lookup.py` that resolves emails from `git log`, `pyproject.toml`/`package.json` `authors`, and GitHub `/users/:login`. | §III.A |
| 7 | **Add an explicit `await human_confirm_push()` shim** in `github_integration/pr_manager.py:_push_branch` gated by `AutonomyLevel < FULL_AUTONOMOUS`. Today the human gate is upstream-only (consensus → escalation); the push line itself is unconditional once reached. | §III.B |
| 8 | **Add an `executive_summary(run_id)` generator** that consumes `AuditRun`, `EscalationRecord`s, `ConvergenceRecord`, and `BoBNAuditRecord` and emits a Markdown brief. Today no such function exists; consumers receive RTM/SAS exports or raw findings. | §III.C |
| 9 | **Reconcile the "23+ MCP servers" claim with the actual count of 19** (18 Python + 1 Rust). Either ship the missing 4 or correct the brief. | §II.C |
| 10 | **Document `AutonomyLevel.AUTO_FIX_PR` and `FULL_AUTONOMOUS` semantics in code.** They exist in the enum but only `READ_ONLY` is observed being branched on (`orchestrator/controller.py:3182`). | §III.B |
| 11 | **Make `_CRITIC_TEMPERATURE` and the integer-millipoint weights part of a qualification test** in `tests/unit/test_adversarial_critic.py` (file already present), so future PRs can't reintroduce float scoring. | `agents/adversarial_critic.py:78,67-73` |
| 12 | **Replace the silent `_GAP5_AVAILABLE` / `_CPG_AVAILABLE` / `_AUTOGEN_AVAILABLE` / `_LANGGRAPH_AVAILABLE` / `_CREWAI_AVAILABLE` `try/except ImportError` no-ops** with explicit `verify_startup()` checks in `startup/feature_matrix.py`, so a production deployment fails fast on a missing optional dep instead of silently degrading the swarm. | `orchestrator/controller.py:56-76`; `swarm/{crew_roles,autogen_agents,langgraph_state}.py` |

---

*End of report. All claims above are line-grounded against the working tree at commit `29e8c5b`. Where the brief named an artefact that does not exist in the repo, the absence is recorded as a Gap with the negative grep that proved it; no behaviour has been inferred or invented.*
