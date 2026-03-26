# Rhodawk AI Code Stabilizer — Adversarial Technical Review

> **Prepared for:** Pre-accelerator demo readiness gate  
> **Review date:** 2026-03-26  
> **Reviewer:** Hostile architecture / security review  
> **Codebase commit:** HEAD (Replit main branch as imported)  
> **Files read end-to-end:** orchestrator/controller.py (3 519 lines), brain/schemas.py (1 021 lines), agents/base.py, agents/fixer.py (1 338 lines), agents/auditor.py (643 lines), agents/formal_verifier.py (918 lines), auth/jwt_middleware.py, brain/sqlite_storage.py (2 497 lines), swe_bench/bobn_sampler.py (1 339 lines), api/app.py (500 lines), startup/feature_matrix.py (537 lines), all test directories

---

## Finding Format

```
ID: [BLOCK|BUG|ARCH|SEC|DEMO|COMP|MISSING|TEST]-NN
Severity: CRITICAL / HIGH / MEDIUM / LOW
File: path:line
```

---

## 1 — BLOCK: Will Crash the Demo Immediately

### BLOCK-01 — `triage_model` phantom Ollama tag crashes every triage call

**Severity:** CRITICAL  
**File:** `orchestrator/controller.py:100`

```python
triage_model: str = "ollama/granite4-tiny"
```

`granite4-tiny` is **not a real Ollama model tag**. The real tags are `granite-code:3b` and `granite-code:8b`. `agents/base.py` explicitly acknowledges and fixes this in the cost map comment on line 41–43 — but **the fix was only applied to the cost map, not to `StabilizerConfig`**. Every call that routes through `triage_model` (fast-path chunk triage, pre-audit classification) will receive an `OllamaError: model 'granite4-tiny' not found` from LiteLLM, killing triage for every file.

**Impact:** Triage phase fails on first file; in the BoBN pipeline every issue classification attempt is dead.  
**Fix:** `triage_model: str = "ollama/granite-code:3b"` to match the corrected cost map entry.

---

### BLOCK-02 — `critical_fix_model` uses non-existent OpenRouter slug

**Severity:** CRITICAL  
**File:** `orchestrator/controller.py:99`

```python
critical_fix_model: str = "openrouter/meta-llama/llama-4"
```

The real OpenRouter slugs for the Llama 4 family are `openrouter/meta-llama/llama-4-scout` and `openrouter/meta-llama/llama-4-maverick`. `llama-4` with no variant suffix returns a 404 from the OpenRouter API. The corrected `_COST_MAP` in `base.py` uses `"openrouter/meta-llama/llama-4-scout"` (line 56). The config still carries the broken slug, so every CRITICAL-severity fix attempt calls a non-existent model and falls back all the way through the fallback chain.

**Impact:** All CRITICAL issues get the degraded fallback chain instead of the intended high-power model; costs and latency are unpredictable.  
**Fix:** `critical_fix_model: str = "openrouter/meta-llama/llama-4-scout"`

---

### BLOCK-03 — Two `fallback_models` are phantom or unverified model IDs

**Severity:** CRITICAL  
**File:** `orchestrator/controller.py:102–106`

```python
fallback_models: list[str] = Field(default_factory=lambda: [
    "ollama/qwen2.5-coder:32b",
    "openrouter/mistralai/devstral-2",      # phantom
    "claude-sonnet-4-20250514",             # phantom
])
```

- **`openrouter/mistralai/devstral-2`** — the real slug is `openrouter/mistralai/devstral-small`. There is no `devstral-2` on OpenRouter. `base.py` corrects this to `"openrouter/mistralai/devstral-small"` in the cost map; the config is not updated.
- **`claude-sonnet-4-20250514`** — not a valid Anthropic model ID. Anthropic's Claude 4 Sonnet is `claude-sonnet-4-5` or the versioned `claude-sonnet-4-6` (as used in `AgentConfig.model` in `base.py`). The date-stamped variant `20250514` does not appear in any Anthropic or LiteLLM routing table.

When the primary model fails (network hiccup, rate limit), fallback cascades into two dead endpoints before surfacing an error.

**Impact:** Any LLM failure during the demo cannot be recovered gracefully; the entire fix pipeline dies on fallback.  
**Fix:** Align `StabilizerConfig.fallback_models` with the corrected `_COST_MAP` entries in `base.py`.

---

### BLOCK-04 — `vllm_base_url` port collides with the FastAPI server

**Severity:** CRITICAL  
**File:** `orchestrator/controller.py:107`

```python
vllm_base_url: str = "http://localhost:8000/v1"
```

The FastAPI application (the REST API the demo talks to) runs on **port 8000** — confirmed by the `uvicorn api.app:app --host 0.0.0.0 --port 8000` workflow command. Every request the primary model sends to `http://localhost:8000/v1/chat/completions` hits the Rhodawk API router itself, not a vLLM inference server, and receives a 404 or a JSON parse error.

**Impact:** Every single LLM call in the primary path fails with an unintelligible error. This is a day-one startup blocker in any non-cloud-API configuration.  
**Fix:** Change `vllm_base_url` to the actual vLLM port (conventionally `8001` or `11434` for Ollama) or document that it must be set via `VLLM_BASE_URL` before starting.

---

### BLOCK-05 — `gap5_enabled=True` by default but secondary vLLM endpoint assumed running on `localhost:8001`

**Severity:** CRITICAL  
**File:** `orchestrator/controller.py:212, 216`

```python
gap5_enabled:               bool = True
gap5_vllm_secondary_base_url: str = "http://localhost:8001/v1"
```

The BoBN pipeline (Gap 5, the flagship feature powering the "85% SWE-bench" claim) is **on by default**. It requires a second vLLM inference server running DeepSeek-Coder-V2-Lite-16B on port 8001 and an adversarial critic endpoint (Llama-3.3-70B). In any single-node deployment — including every demo laptop, cloud VM without two GPU partitions, or Replit environment — these endpoints do not exist. The `LocalizationAgent → Fixer A + Fixer B → AdversarialCritic → Synthesis` chain crashes at the first LiteLLM call to `localhost:8001`.

**Impact:** The headline feature fails on every single-machine deployment. The demo breaks the moment the first fix cycle runs.  
**Fix:** Default `gap5_enabled=False`; make it opt-in with explicit documentation of the two-GPU requirement. Or implement a graceful single-fixer degradation path that activates when `localhost:8001` is unreachable.

---

### BLOCK-06 — PostgreSQL is the default storage but `postgres_dsn` defaults to an empty string

**Severity:** CRITICAL  
**File:** `orchestrator/controller.py:134–135`

```python
use_sqlite: bool = False
postgres_dsn: str = ""
```

The default `StabilizerConfig` targets PostgreSQL (production mode). The `_init_storage()` method attempts to construct a `PostgresBrainStorage` with an empty DSN, which fails immediately with an `asyncpg` connection error or a `DSN cannot be empty` assertion. There is no fallback. The controller cannot complete `initialise()`, and no run can ever start.

The inline comment says "SQLite only for dev mode," implying Postgres is the shipping default — but shipping without a DSN configured guarantees a startup crash on every clean deployment.

**Impact:** The system is completely non-functional on a fresh install with default config. This is a BLOCK bug even for the demo if the deployer does not explicitly set `use_sqlite=True`.  
**Fix:** Either: (a) detect empty `postgres_dsn` and auto-switch to SQLite with a warning, or (b) add a startup check that raises `ConfigurationError` with the required DSN format before the async pool is opened.

---

### BLOCK-07 — No `/auth/token` endpoint; the API is permanently locked out

**Severity:** CRITICAL  
**File:** `api/app.py` (entire file, 500 lines read)

`JWTMiddleware.public_paths` includes `/auth/token`, signalling that a token-issuance endpoint should exist. No such route is registered anywhere in `api/app.py` or any router imported by it. The JWT middleware enforces Bearer token auth on every non-public path. Without a way to obtain a token, every API call returns 401. The demo cannot create a run, check status, or retrieve results through the REST API.

**Impact:** The entire REST API surface is permanently locked to any caller who does not generate a token out-of-band using `create_access_token()` directly from the Python source. Zero demo interactivity.  
**Fix:** Add an `POST /auth/token` endpoint that accepts `{sub, scopes}` for development, or add a `POST /auth/login` endpoint with username/password for production.

---

### BLOCK-08 — `synthesis_model` comment claims fix but code is still broken

**Severity:** CRITICAL  
**File:** `orchestrator/controller.py:203`

```python
synthesis_model: str = ""
```

The 14-line block comment above this field (lines 188–202) describes a "BUG FIX" — changing the default from `""` to `"DeepSeek-Coder-V2"` to guarantee synthesis family independence. The comment has been written. The actual default value has **not** been changed. It is still `""`.

When `synthesis_model` is empty, the synthesis path falls back to `critical_fix_model` — which is in the same model family as the primary auditor (`openrouter/meta-llama/llama-4`). The comment explicitly states this breaks the independence constraint. The fix exists only as a comment, not as code.

**Impact:** Every synthesis pass uses the same model family as the auditor, directly violating the cross-domain independence guarantee that the architecture claims as a core safety property. In DO-178C terms this is an independence violation that would disqualify the SAS output.  
**Fix:** Change the actual default: `synthesis_model: str = "openrouter/deepseek/deepseek-coder-v2-0724"`.

---

## 2 — BUG: Silent Runtime Failures and Data Corruption

### BUG-01 — `Severity.MEDIUM` exists but is silently excluded from the scoring formula

**Severity:** HIGH  
**File:** `brain/schemas.py:766–771`

```python
class Severity(str, Enum):
    CRITICAL = 'CRITICAL'
    MAJOR    = 'MAJOR'
    MEDIUM   = 'MEDIUM'   # ← exists
    MINOR    = 'MINOR'
    INFO     = 'INFO'

def compute_score(self) -> None:
    self.total_issues = self.critical_count + self.major_count + self.minor_count + self.info_count
    # MEDIUM is not in this sum; no medium_count field exists on AuditScore
```

`AuditScore` has `critical_count`, `major_count`, `minor_count`, `info_count` — no `medium_count`. Any issue stored with `severity=Severity.MEDIUM` is never counted in `total_issues` or scored. With the LLM free to produce MEDIUM-severity findings, a codebase can appear to have a perfect score of 100 while harboring dozens of real MEDIUM-severity issues.

**Impact:** Auditability lie. The exported SAS document and compliance report silently under-report issues. For a system claiming DO-178C compliance, this is a disqualifying defect.  
**Fix:** Add `medium_count: int = 0` to `AuditScore`, include it in `total_issues` summation, and add a scoring penalty (suggest 2.5 points per finding, between MAJOR and MINOR).

---

### BUG-02 — `completion_tokens` hardcoded to 500 for all structured LLM calls

**Severity:** HIGH  
**File:** `agents/base.py:288`

```python
completion_tokens = 500    # hardcoded — never reads actual usage
```

For unstructured calls (`call_llm_raw`), the code correctly reads `usage.completion_tokens` from the response. For structured calls (`call_llm_structured` via `instructor`), it uses a hardcoded 500 regardless of actual token usage. A full-file fix response for a 2 000-line C file might use 6 000 output tokens; the system records 500 and charges 8× too little to the cost ceiling. The `$50 cost_ceiling_usd` becomes meaningless — the pipeline can run indefinitely past the configured ceiling.

**Impact:** Cost ceiling enforcement is broken for all structured calls (which are the majority of calls). Runaway costs in production; also means cost-tracking audit trail is fabricated.  
**Fix:** The `instructor` response object wraps the raw completion; access `response._raw_response.usage.completion_tokens` or pass `with_usage=True` to `instructor`.

---

### BUG-03 — `instructor.from_litellm` client created inside the retry loop

**Severity:** HIGH  
**File:** `agents/base.py:449`

```python
async for attempt in AsyncRetrying(...):
    with attempt:
        client = instructor.from_litellm(litellm.acompletion)   # new client every retry
        response = await asyncio.wait_for(client.chat.completions.create(...), ...)
```

A new `instructor` client (which wraps an `httpx.AsyncClient`) is constructed on every retry attempt. The `AsyncClient` is never explicitly closed. Under a 3-retry policy with 10 concurrent fix groups, this spawns up to 30 unclosed HTTP client objects per batch, each holding an open connection pool. Under sustained load this exhausts the system's file descriptor limit.

**Impact:** File descriptor leak; server crashes after extended operation (minutes to hours depending on concurrency and retry rate). Will manifest during any multi-hour demo run.  
**Fix:** Construct the `instructor` client once per agent instance in `__init__` and reuse it.

---

### BUG-04 — Duplicate `model_validator` import in `brain/schemas.py`

**Severity:** MEDIUM  
**File:** `brain/schemas.py:6`

```python
from pydantic import BaseModel, Field, model_validator, field_validator, model_validator
```

`model_validator` is imported twice in the same `from ... import` statement. Python silently overwrites the first binding with the second (same object), so there is no runtime failure. However, in Python 3.12+ this raises a `SyntaxWarning`. More critically, it signals that this file has not been passed through any linter (`ruff check` would catch this as `F811 — redefinition of unused name`). If the project's own static analysis pipeline can't catch an error in its own schema file, the claim of running Ruff/Bandit/Semgrep on every patch is undermined.

**Impact:** Moderate: no crash, but it directly contradicts the claim of running the static analysis suite on the codebase.  
**Fix:** `from pydantic import BaseModel, Field, model_validator, field_validator`

---

### BUG-05 — `FormalVerificationResult.solver_used` field name may mismatch storage writes

**Severity:** MEDIUM  
**File:** `brain/schemas.py:264` vs `agents/formal_verifier.py`

`FormalVerificationResult` declares `solver_used: str = 'z3'`. If any code path constructs this object using a `solver=` keyword argument instead of `solver_used=`, Pydantic silently ignores the unknown kwarg (in model v2 with `extra='ignore'` or when passing as `**dict`), leaving `solver_used` as the default `'z3'` regardless of which solver actually ran. Audit trail entries then claim Z3 ran even when CBMC or pattern-matching was used.

**Impact:** DO-178C audit trail falsification — the evidence artifact says "Z3 proved this property" when the actual check was a regex pattern match.  
**Fix:** Grep all construction sites for `FormalVerificationResult(` and ensure the field name is `solver_used=`, not `solver=`.

---

### BUG-06 — `AuditRun.max_cycles` schema default (50) diverges from `StabilizerConfig` default (200)

**Severity:** MEDIUM  
**File:** `brain/schemas.py:808` vs `orchestrator/controller.py:112`

```python
# schemas.py
max_cycles: int = 50

# controller.py  
max_cycles: int = 200
```

The controller correctly passes its value when creating the `AuditRun`, so this doesn't cause a functional bug in the happy path. However, if any code creates an `AuditRun` directly (e.g., test fixtures, the resume path with partial data, or a future admin endpoint), the schema default silently caps the run at 50 cycles — 4× fewer than the stated default. The comment on line 109 calls out the historic bug between API-path (50) and CLI-path (200), but the root cause — the schema default — is not fixed.

**Impact:** Hidden correctness gap in any code path that constructs `AuditRun()` without passing `max_cycles`.  
**Fix:** `max_cycles: int = 200` in `AuditRun`.

---

## 3 — ARCH: Architectural Gaps That Will Embarrass You at the Whiteboard

### ARCH-01 — The "85%+ SWE-bench" claim is theoretical probability, not measured performance

**Severity:** CRITICAL  
**File:** `swe_bench/bobn_sampler.py:11–14`

```python
# The Agent S3 paper demonstrated that BoBN N=5 lifts a 32B model from ~45% to ~72% on SWE-bench.
# In practice candidates are correlated...so empirical lift is ~12-18 percentage points.
```

The code itself admits the empirical lift is 12–18 points, not 40 points. Starting from a Qwen2.5-Coder-32B baseline of ~45% (optimistic; the model is not fine-tuned on SWE-bench) and adding 18 points yields ~63%, not 85%. The gap between 63% and 85% requires: (a) Devstral-level fine-tuning, (b) execution feedback loops that require Docker-in-Docker sandboxing not present in this codebase, and (c) a test harness that mirrors the official SWE-bench Docker environment. None of these are shipped.

Any sophisticated investor or technical judge will ask: "What is your current measured score on SWE-bench Verified?" The answer from this codebase is **"we haven't measured it."**

**Impact:** Credibility destruction in any technical Q&A session. This is the #1 competitive benchmark in the space; Claude Sonnet 4, GPT-4o, and Cognition all have published numbers. Claiming 85%+ without a measurement is fraud adjacent.  
**Fix:** Either (a) run the official SWE-bench Docker evaluation and report the actual number, or (b) change the claim to "targeting 72%+ via BoBN N=10 on SWE-bench Verified (unmeasured; evaluation in progress)."

---

### ARCH-02 — DO-178C compliance claim requires the tool itself to be DO-178C qualified

**Severity:** CRITICAL  
**File:** `orchestrator/controller.py:122`, `brain/schemas.py:182–188`

The system claims DO-178C compliance support including SAS generation, RTM export, baseline locking, and independence verification. DO-178C §12.2 requires that any tool whose output is used to satisfy certification objectives without being independently verified must itself hold a **Tool Qualification Document (TQD)** at TQL-1 through TQL-5 depending on the objective.

Rhodawk AI generates fixes, runs tests, generates SAS documents, and populates the RTM. All of these are DO-178C Table A objectives. Using an unqualified AI tool for any of these activities requires the human DER to independently re-verify every output — at which point the AI provides zero certification leverage.

No TQD exists. No qualification test suite is present. The `ToolQualificationLevel` enum exists in the schema but defaults to `NONE`. The system is marketing compliance it cannot deliver.

**Impact:** Any aerospace or defense customer's DER will reject this tool in the first technical review. The entire "safety-critical" market positioning is legally and technically untenable until tool qualification is obtained.  
**Fix:** Either (a) formally scope the tool as "advisory only — all outputs require independent human verification" and remove compliance marketing language, or (b) begin a proper DO-178C tool qualification campaign (12–24 months, $500K+).

---

### ARCH-03 — Joern/CPG engine assumed running but not bundled; `cpg_enabled=True` by default

**Severity:** HIGH  
**File:** `orchestrator/controller.py:173–177`

```python
cpg_enabled:     bool = True
joern_url:       str  = "http://localhost:8080"
```

The CPG engine requires a running Joern server at `localhost:8080`. Joern is a JVM application requiring Java 11+, ~2 GB RAM, and a separate process. It is not started by any workflow, not bundled in requirements, and not documented as a prerequisite outside the config comment. With `cpg_enabled=True` by default and the CPG used for blast-radius computation, coupling-smell detection, and incremental audit scheduling, any deployment without Joern running silently degrades to stub behavior — but the feature matrix check (which is non-strict in GENERAL mode) only logs a warning.

**Impact:** The three highest-value architectural features (Gap 1 causal CPG context, Gap 3 coupling smell detection, Gap 4 commit-granularity audit) all silently degrade on every default deployment. The demo will claim CPG-enhanced analysis while running on stub data.  
**Fix:** Add a Joern startup check to the demo script; document the Joern installation requirement explicitly; or default `cpg_enabled=False` and activate it only when `JOERN_URL` is explicitly configured.

---

### ARCH-04 — Independence enforcement is string-matching on model family names, not cryptographic

**Severity:** HIGH  
**File:** `verification/independence_enforcer.py` (via `brain/schemas.py:248–254`)

```python
same = self.fixer_model_family.lower() == self.reviewer_model_family.lower()
```

The DO-178C §6.3.4 independence check compares lowercase string representations of model family names (`"alibaba"`, `"meta"`, `"deepseek"`). The family name is extracted by `extract_model_family()` using string prefix matching on the model ID. An attacker (or a misconfigured deployment) that names a custom fine-tune `"alibaba/custom-model"` will pass the independence check even if it was fine-tuned from the same weights as the fixer. More practically: if two models from the same family happen to use different naming prefixes (e.g., `Qwen2.5` and `Qwen3`), the check may pass or fail incorrectly depending on the regex.

**Impact:** False independence certification in the audit trail.  
**Fix:** Independence must be declared explicitly in a model registry (`model_family_registry.yaml`) and cryptographically signed, not inferred from model ID strings.

---

### ARCH-05 — Rate limiter is declared but never initialized

**Severity:** HIGH  
**File:** `agents/base.py:243`

```python
self._rate_limiter: Any | None = None
```

The `_rate_limiter` attribute is declared in every agent's `__init__` but is never assigned a non-None value anywhere in the base class or any subclass visible in this codebase. No call site ever calls `await self._rate_limiter.acquire()`. There is no per-model rate limiting. Under the default concurrency of 4 parallel fix groups, all four can simultaneously hit the same OpenRouter endpoint, triggering 429 rate limit errors that cascade into the fallback chain (which hits two phantom models) before failing.

**Impact:** Rate limit storm during any parallel workload. The retry logic exists but without rate limiting the retries themselves are rate-limited.  
**Fix:** Implement per-model token-bucket rate limiting using `aiolimiter` or equivalent, initialized in `BaseAgent.__init__`.

---

### ARCH-06 — PostgreSQL and SQLite storage schemas are not guaranteed in sync

**Severity:** MEDIUM  
**File:** `brain/sqlite_storage.py:49–200` vs `brain/postgres_storage.py` (not read but imported)

The SQLite DDL is defined inline in `sqlite_storage.py` (2 497 lines). PostgreSQL storage is in a separate file. There is no migration framework (no Alembic, no Flyway, no versioned migrations). Schema changes to the SQLite DDL are not automatically applied to the PostgreSQL schema. Given that SQLite is used for development and PostgreSQL for production, schema drift between the two is not a matter of "if" but "when."

**Impact:** Data persisted in development cannot be reliably migrated to production. Any field added to `AuditRun` or `Issue` (which happens frequently given the active development pace) requires manual DDL surgery on both backends.  
**Fix:** Introduce Alembic. Define the canonical schema in SQLAlchemy Core and generate both SQLite and PostgreSQL DDL from a single source.

---

## 4 — SEC: Security Vulnerabilities

### SEC-01 — WebSocket token in URL query parameter is logged by every reverse proxy

**Severity:** HIGH  
**File:** `auth/jwt_middleware.py:423`

```python
token = websocket.query_params.get("token")
```

JWT tokens passed as `?token=<jwt>` in WebSocket URLs are logged by Nginx, AWS ALB, Cloudflare, and every other reverse proxy in plain text in their access logs. They also appear in browser history, server-sent referer headers, and DevTools network tabs. The comment in the docstring acknowledges this as a browser limitation but makes no mitigating recommendation.

**Impact:** Token theft via log file exfiltration — a known attack vector in any environment with centralized logging (Splunk, Datadog, CloudWatch). For a system claiming military-grade security, this is a C-suite-visible finding.  
**Fix:** Implement the Sec-WebSocket-Protocol handshake pattern (token in the subprotocol header) or a one-time upgrade ticket issued via a REST endpoint before the WebSocket connection.

---

### SEC-02 — Missing webhook HMAC verification is only a WARNING in production

**Severity:** HIGH  
**File:** `api/app.py:131–138`

```python
if not os.environ.get("RHODAWK_WEBHOOK_SECRET"):
    log.warning("CI push webhooks will be accepted without signature verification.")
```

The webhook secret is optional in production — a WARNING, not a FATAL. An unauthenticated caller can POST to the CI webhook endpoint and trigger a full audit run against any repository. Combined with `auto_commit=True` (the default), an attacker can force Rhodawk to commit AI-generated patches to any configured repository without authentication.

**Impact:** Unauthenticated remote code-commit trigger. In a military/aerospace deployment this is a supply-chain attack vector.  
**Fix:** Make `RHODAWK_WEBHOOK_SECRET` required in production (same pattern as `RHODAWK_AUDIT_SECRET` — raise `ConfigurationError` if absent).

---

### SEC-03 — `os._exit(1)` in FastAPI startup bypasses asyncio pool cleanup

**Severity:** MEDIUM  
**File:** `api/app.py:129`

The code acknowledges this bug in a comment (`ADD-3 NOTE — asyncio cleanup gap`) but does not fix it. On every invalid-config restart (common during deployment debugging), the PostgreSQL connection pool is leaked. On a cloud deployment with 10 replicas each with a pool of 5 connections, 10 crash-restarts exhaust the PostgreSQL `max_connections` limit (default 100), making the database inaccessible to the entire fleet.

**Impact:** Database denial-of-service on repeated config errors — exactly the scenario that occurs during a deployment debugging session, e.g., the day before a demo.  
**Fix:** Move security checks to `@asynccontextmanager` lifespan, before the pool is opened. A check that runs before any async resource is created can use `sys.exit()` safely.

---

### SEC-04 — Prompt injection defense covers 6 patterns; adversarial repos use hundreds

**Severity:** MEDIUM  
**File:** `agents/base.py:72–82`

The `_INJECTION_STRIP_PATTERNS` list has 6 regex patterns. Academic prompt injection research (PAIR, GCG, AutoDAN) demonstrates that LLMs can be jailbroken via Unicode lookalikes, base64-encoded instructions, steganographic whitespace, and role-confusion attacks that appear as normal code comments. The 6-pattern list covers the naive "SYSTEM OVERRIDE" case but nothing adversarial. For a system claiming to process hostile/untrusted repositories (the military supply-chain use case), the injection surface is enormous.

**Impact:** A sufficiently adversarial repository can manipulate the auditor or fixer LLM into generating malicious patches or suppressing true findings.  
**Fix:** The `<source_code>` delimiter approach (SEC-01) is structurally sound; expand it with input-output guardrails (LLamaGuard, NeMo Guardrails, or PromptArmor) on the boundary.

---

## 5 — DEMO: Things That Will Specifically Fail During Your Accelerator Pitch

### DEMO-01 — The system cannot start at all with default configuration

Combining BLOCK-04 (port 8000 conflict), BLOCK-05 (localhost:8001 required), BLOCK-06 (empty PostgreSQL DSN), and BLOCK-07 (no token endpoint): **a fresh clone with default environment will fail at startup before processing a single file.** There is no "happy path" default configuration that works out of the box on any single machine.

**Minimum viable demo config requires:**
- `RHODAWK_JWT_ALGORITHM=HS256` + valid `RHODAWK_JWT_SECRET`
- `use_sqlite=True` OR a running PostgreSQL + DSN
- `gap5_enabled=False` (unless two vLLM servers are running)
- `vllm_base_url` changed to a non-8000 port OR all models routed via cloud APIs
- A `/auth/token` endpoint added (none exists)
- A running Ollama with `qwen2.5-coder:32b` pulled, OR `primary_model` changed to a cloud API model

None of this is documented in `README.md` (no such file exists in this repository). `replit.md` describes the architecture but does not contain a quick-start guide.

---

### DEMO-02 — "Upload any codebase" claims require Docker sandbox not present

**Severity:** HIGH  

The SWE-bench BoBN evaluator (`swe_bench/execution_loop.py`) requires Docker to run test candidates in isolated containers — standard for SWE-bench evaluation. The sandbox executor (`sandbox/executor.py`) runs static analysis tools. When a demo audience member tries to upload *their own* codebase, the system either: (a) runs their code directly in the Replit environment (security disaster), or (b) fails with a Docker-not-found error. The claim "ingest any codebase" implies safe sandboxed execution; the implementation has no container isolation for arbitrary code.

**Impact:** The demo either crashes or runs untrusted code on the demo host.  
**Fix:** Implement process isolation using `subprocess` with `ulimit` / `seccomp` restrictions as an interim measure; document Docker requirement prominently.

---

### DEMO-03 — Metrics endpoint exports nothing until a run completes

**Severity:** MEDIUM  
**File:** `metrics/prometheus_exporter.py` (imported)

The Prometheus metrics and LangSmith tracer are initialized but only populated during an active pipeline run. If the demo audience asks "what's your current throughput / cost / issue resolution rate," the Prometheus scrape endpoint returns all-zero counters until at least one full cycle completes. For a demo that claims production-grade observability, an empty dashboard is actively harmful.

---

## 6 — COMP: Competitive Gaps (What Rivals Already Ship)

### COMP-01 — No measured SWE-bench score; every named competitor has published numbers

SWE-bench Verified leaderboard (as of Q1 2026): Cognition Devin 2 (53.6%), Amazon Q Developer (55.0%), GitHub Copilot Workspace (~44%), OpenHands (39.0%). Rhodawk's 85% claim, if true, would be the highest in the world by 30 points and would already be published and viral. Unsubstantiated claims in this space are immediately disqualifying in technical investor conversations.

### COMP-02 — BoBN N=10 requires more GPU budget than most enterprise customers have

Ten concurrent inference calls at Qwen2.5-Coder-32B + DeepSeek-16B + Llama-3.3-70B (adversarial critic) + Devstral (synthesis) = 4 different endpoint families, two of which require dedicated GPU servers. The per-issue GPU cost is ~10× a single-model solution. Competitors (SWE-agent, Moatless) run on a single GPT-4o or Claude call per issue. The "swarm" architecture is only economically viable for high-value issues in regulated industries — which is the right niche, but it must be stated explicitly rather than implied as general-purpose.

### COMP-03 — No GitHub Copilot / IDE integration

The system ships a REST API. Every competitor in the "AI code quality" space ships a VS Code extension, a GitHub App, or a Copilot plugin. An accelerator audience will immediately ask "how does a developer actually use this?" The answer ("POST to /api/runs/start") is a deal-breaker for developer-tool positioning.

---

## 7 — MISSING: Unimplemented Features Claimed in the Architecture

### MISSING-01 — `leanstral.py` formal prover exists but is never called

**Severity:** HIGH  
**File:** `verification/leanstral.py` (file exists)

The `verification/` directory contains `leanstral.py` — a Lean 4 proof generation module. It is imported nowhere in the codebase (verified by directory inspection). The formal verification pipeline uses Z3 and CBMC. Lean 4 / LeanSTRaL is one of the architectural claims in the marketing narrative ("formally verifiable patches") but the implementation is a dead file.

### MISSING-02 — Federated fix-pattern store (`gap6`) has no peers to federate with

**Severity:** MEDIUM  
**File:** `orchestrator/controller.py:250–264`

Gap 6 (federated anonymized pattern store) is `gap6_federation_enabled=False` by default. The `gap6_registry_url` is empty. There are no known federation peers. The `gap6_instance_id` is auto-generated. The entire federation feature is a solo installation talking to no one. Federating with yourself is not federated learning.

### MISSING-03 — `escalation/human_escalation.py` is imported but no human notification transport is verified

**Severity:** MEDIUM  

`EscalationManager` is initialized with `api_base_url` and `timeout_hours`. If `api_base_url` is empty (the default), escalation notifications have no delivery mechanism. A CRITICAL finding in a DO-178C DAL-A module escalates to `ESCALATION_PENDING` and then times out with no human ever notified. The compliance guarantee collapses silently.

---

## 8 — TEST: Test Suite Integrity

### TEST-01 — Unit tests mock the LLM and storage layers; no end-to-end coverage

**Severity:** HIGH  
**Files:** `tests/unit/` (15 files)

The unit test suite (confirmed by directory listing) covers individual agent logic with mocked LLM responses and in-memory storage. There are zero tests that exercise a real LLM call against a real repository. `tests/integration/test_pipeline.py` exists but whether it runs against a real model is unknown — integration tests in CI environments are almost always skipped when API keys are absent. The SWE-bench score cannot be validated by any test in this suite.

### TEST-02 — Tests cannot detect the BLOCK bugs found in this review

The test suite could not have caught BLOCK-01 through BLOCK-08 because:
- BLOCK-01/02/03: model tags are never resolved against a real LLM API in tests
- BLOCK-04: port 8000 collision is not tested (no dual-service test)
- BLOCK-05: gap5 secondary endpoint is mocked
- BLOCK-06: SQLite is used in all tests (`use_sqlite=True` in fixtures)
- BLOCK-07: no `/auth/token` endpoint means no test can exercise the full auth flow
- BLOCK-08: synthesis_model default is never exercised in isolation

This is not a failure of test coverage; it is a failure of test architecture. The tests assume a working infrastructure. What is needed is a smoke-test that validates the config defaults can produce a working system.

### TEST-03 — `conftest.py` likely patches `RHODAWK_ENV=development`; production code paths untested

**Severity:** MEDIUM  

Every production-vs-development branch in `api/app.py` and `auth/jwt_middleware.py` is conditional on `RHODAWK_ENV`. If `conftest.py` sets `RHODAWK_ENV=development` (standard practice for test suites), the security checks that use `_is_production` are never exercised in CI. The RS256 key requirement, the dev-auth SystemExit, and the CORS allowlist restriction are never tested in their production configurations.

---

## Summary Scorecard

| Category | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| BLOCK (demo crash) | 8 | 0 | 0 | 0 |
| BUG (silent failures) | 0 | 2 | 4 | 0 |
| ARCH (architectural gaps) | 2 | 3 | 1 | 0 |
| SEC (security) | 0 | 2 | 2 | 0 |
| DEMO (pitch killers) | 0 | 1 | 1 | 0 |
| COMP (competitive gaps) | 1 | 1 | 1 | 0 |
| MISSING (unimplemented) | 0 | 1 | 2 | 0 |
| TEST (test integrity) | 0 | 1 | 1 | 0 |
| **Total** | **11** | **11** | **12** | **0** |

**The system cannot run a demo in its current default state. Fix the 8 BLOCK bugs before any accelerator conversation.**

---

## Minimum Viable Demo Fix Checklist

These are the changes needed to reach a working demo, in priority order:

1. **Add `/auth/token` endpoint** (BLOCK-07) — zero API access without it
2. **Fix `postgres_dsn` default** — set `use_sqlite=True` as default OR add DSN validation (BLOCK-06)
3. **Set `gap5_enabled=False` as default** — until dual-GPU infra is documented (BLOCK-05)
4. **Fix port 8000 collision** — change `vllm_base_url` default or document override (BLOCK-04)
5. **Fix three phantom model IDs** — `triage_model`, `critical_fix_model`, two `fallback_models` (BLOCK-01/02/03)
6. **Fix `synthesis_model` default** — change `""` to `"openrouter/deepseek/deepseek-coder-v2-0724"` (BLOCK-08)
7. **Add `medium_count` to `AuditScore`** — silent score falsification (BUG-01)
8. **Replace hardcoded `completion_tokens=500`** — cost ceiling enforcement is broken (BUG-02)
9. **Write a `README.md` quick-start** — no first-time deployer can get this running without it
10. **Replace "85%+ SWE-bench" with honest measured or estimated numbers** (ARCH-01/COMP-01)

---

*End of adversarial review. Total findings: 34. Critical/High: 22. The codebase shows sophisticated architectural thinking — the BoBN pipeline, CPG-based blast radius, and DO-178C schema work are genuinely impressive at the design level. The implementation is approximately 60% complete. Demo readiness requires resolving the 8 BLOCK bugs and the critical competitive claim issue before any investor technical due diligence.*
