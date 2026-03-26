
```

The real OpenRouter slugs for the Llama 4 family are `openrouter/meta-llama/llama-4-scout` and `openrouter/meta-llama/llama-4-maverick`. `llama-4` with no variant suffix returns a 404 from the OpenRouter API. The corrected `_COST_MAP` in `base.py` uses `"openrouter/meta-llama/llama-4-scout"` (line 56). The config still carries the broken slug, so every CRITICAL-severity fix attempt calls a non-existent model and falls back all the way through the fallback chain.

**Impact:** All CRITICAL issues get the degraded fallback chain instead of the intended high-power model.  
**Fix:** `critical_fix_model: str = "openrouter/meta-llama/llama-4-scout"`

---

### BLOCK-03 — Two `fallback_models` are phantom or unverified model IDs

**Severity:** CRITICAL  
**File:** `orchestrator/controller.py:102–106`

```python
fallback_models: list[str] = Field(default_factory=lambda: [
    "ollama/qwen2.5-coder:32b",
    "openrouter/mistralai/devstral-2",      # phantom — no such slug
    "claude-sonnet-4-20250514",             # phantom date-stamped ID
])
```

- **`openrouter/mistralai/devstral-2`** — the real slug is `openrouter/mistralai/devstral-small`. `base.py` corrects this in the cost map; the config is not updated.  
- **`claude-sonnet-4-20250514`** — not a valid Anthropic model ID as of this review. The date-stamped variant does not appear in any Anthropic or LiteLLM routing table.

When the primary model fails (network hiccup, rate limit), fallback cascades into two dead endpoints before surfacing an error.

**Impact:** Any LLM failure during the demo cannot be recovered gracefully; the fix pipeline dies on fallback.  
**Fix:** Align `StabilizerConfig.fallback_models` with the corrected `_COST_MAP` entries in `base.py`.

---

### BLOCK-04 — `vllm_base_url` port collides with the FastAPI server

**Severity:** CRITICAL  
**File:** `orchestrator/controller.py:107`

```python
vllm_base_url: str = "http://localhost:8000/v1"
```

The FastAPI application runs on **port 8000** (confirmed by the `uvicorn api.app:app --host 0.0.0.0 --port 8000` workflow command). Every request the primary model sends to `http://localhost:8000/v1/chat/completions` hits the Rhodawk API router itself, not a vLLM inference server, and receives a 404 or a JSON parse error.

**Impact:** Every single LLM call in the primary path fails with an unintelligible error.  
**Fix:** Change `vllm_base_url` to the actual vLLM port (conventionally `8001` or `11434` for Ollama) or document that it must be set via `VLLM_BASE_URL` before starting.

---

### BLOCK-05 — `gap5_enabled=True` by default but secondary vLLM endpoint assumed running on `localhost:8001`

**Severity:** CRITICAL  
**File:** `orchestrator/controller.py:212, 216`

```python
gap5_enabled:               bool = True
gap5_vllm_secondary_base_url: str = "http://localhost:8001/v1"
```

The BoBN pipeline (Gap 5, the flagship feature powering the "85% SWE-bench" claim) is **on by default**. It requires a second vLLM inference server running DeepSeek-Coder-V2-16B on port 8001 and an adversarial critic endpoint (Llama-3.3-70B). In any single-node deployment — including every demo laptop, cloud VM without two GPU partitions, or the Replit environment — these endpoints do not exist. The entire `LocalizationAgent → Fixer A + Fixer B → AdversarialCritic → Synthesis` chain crashes at the first LiteLLM call to `localhost:8001`.

**Impact:** The headline feature fails on every single-machine deployment. The demo breaks the moment the first fix cycle runs.  
**Fix:** Default `gap5_enabled=False`; make it opt-in with explicit documentation of the two-GPU requirement. Alternatively, implement graceful single-fixer degradation when `localhost:8001` is unreachable (the graceful-degradation path in `_init_gap5()` exists but only triggers on import error, not on network unreachable).

---

### BLOCK-06 — PostgreSQL is the default storage but `postgres_dsn` defaults to an empty string

**Severity:** CRITICAL  
**File:** `orchestrator/controller.py:134–135`

```python
use_sqlite: bool = False
postgres_dsn: str = ""
```

The default `StabilizerConfig` targets PostgreSQL. `PostgresBrainStorage.initialise()` calls `_require_database_url()` which raises `RuntimeError('FATAL: DATABASE_URL not set...')` on empty DSN. There is a fallback to SQLite on exception — but this fallback is logged only at `ERROR` level and silently swallowed. The controller's `_init_storage()` does not propagate the error, so the system appears to initialize while actually running on SQLite with no data durability. Worse: if `asyncpg` is not installed (`_PG_AVAILABLE=False`), the fallback is explicit — but if it is installed and the DSN is empty, `_require_database_url()` raises before the engine is created, the `except` clause in `initialise()` is caught, and the fallback is used — but the `_is_pg()` method returns `True` if `_engine is not None`, which it is not, so all calls go through `__getattr__` → `_fallback`. This works but requires careful reading of 980 lines to understand.

**Impact:** Fresh deployments silently fall to SQLite with no data durability; PostgreSQL connection errors are invisible. Production deployments that expect PostgreSQL get SQLite.  
**Fix:** Detect empty `postgres_dsn` explicitly before attempting connection. Raise `ConfigurationError` with the required DSN format at startup, or document that `use_sqlite=True` is the intended default for development.

---

### BLOCK-07 — No `/auth/token` endpoint; the API is permanently locked out

**Severity:** CRITICAL  
**File:** `api/app.py` (entire file)

`JWTMiddleware.public_paths` includes `/auth/token`, signalling that a token-issuance endpoint should exist. No such route is registered anywhere in `api/app.py` or any of the 10 router files in `api/routes/`. The JWT middleware enforces Bearer token auth on every non-public path. Without a way to obtain a token, every API call returns 401.

**Impact:** The entire REST API surface is permanently locked to any caller who does not generate a token out-of-band using `create_access_token()` directly from Python source. Zero demo interactivity.  
**Fix:** Add a `POST /auth/token` endpoint that accepts `{sub, scopes}` for development, or `POST /auth/login` with username/password for production.

---

### BLOCK-08 — `synthesis_model` comment describes the fix but the code is still broken

**Severity:** CRITICAL  
**File:** `orchestrator/controller.py:203`

```python
synthesis_model: str = ""
```

The 14-line block comment above this field (lines 188–202) describes a "BUG FIX" — changing the default from `""` to `"DeepSeek-Coder-V2"` to guarantee synthesis family independence. The comment has been written. The actual default value has **not** been changed. It is still `""`.

When `synthesis_model` is empty, the synthesis path in `_phase_audit()` falls back to `os.environ.get("RHODAWK_SYNTHESIS_MODEL", "") or self.config.synthesis_model or "openrouter/deepseek-ai/DeepSeek-Coder-V2-Instruct"`. The env var and config field are both empty, so it uses the hardcoded fallback in the phase code — which is actually correct. However, the independence enforcement at lines 1322–1344 compares `primary_model` against `synthesis_model`, and since `synthesis_model=""` resolves to `extract_model_family("")` = `"unknown"`, the independence check always passes trivially, producing a false-positive independence certificate.

**Impact:** Every synthesis pass produces a falsified independence certificate regardless of actual model families used. In DO-178C terms this is a disqualifying audit trail defect.  
**Fix:** Change the actual default: `synthesis_model: str = "openrouter/deepseek-ai/DeepSeek-Coder-V2-Instruct"` to match the intended fallback already hardcoded in `_phase_audit()`.

---

## 2 — BUG: Silent Runtime Failures and Data Corruption

### BUG-01 — `Severity.MEDIUM` exists but is silently excluded from the scoring formula

**Severity:** HIGH  
**File:** `brain/schemas.py:766–771`

```python
class Severity(str, Enum):
    CRITICAL = 'CRITICAL'
    MAJOR    = 'MAJOR'
    MEDIUM   = 'MEDIUM'   # exists in enum
    MINOR    = 'MINOR'
    INFO     = 'INFO'

def compute_score(self) -> None:
    self.total_issues = self.critical_count + self.major_count + self.minor_count + self.info_count
    # MEDIUM is missing; no medium_count field exists on AuditScore
```

Any issue stored with `severity=Severity.MEDIUM` is never counted in `total_issues` or scored. LLMs freely produce MEDIUM-severity findings. A codebase can appear to have a perfect score while harboring dozens of real MEDIUM-severity issues.

**Impact:** Auditability lie. The exported SAS document silently under-reports issues. For DO-178C compliance this is a disqualifying defect.  
**Fix:** Add `medium_count: int = 0` to `AuditScore`, include it in `total_issues` and scoring formula.

---

### BUG-02 — `completion_tokens` hardcoded to 500 for all structured LLM calls

**Severity:** HIGH  
**File:** `agents/base.py:288`

```python
completion_tokens = 500    # hardcoded — never reads actual usage
```

For unstructured calls (`call_llm_raw`), the code correctly reads `usage.completion_tokens` from the response. For structured calls (`call_llm_structured` via `instructor`), it uses a hardcoded 500 regardless of actual token usage. A full-file fix response for a 2 000-line C file may use 6 000 output tokens; the system records 500. The `$50 cost_ceiling_usd` is meaningless — the pipeline can run indefinitely past the configured ceiling.

**Impact:** Cost ceiling enforcement is broken for all structured calls (the majority of calls). Runaway costs in production; audit trail cost figures are fabricated.  
**Fix:** Access `response._raw_response.usage.completion_tokens` or pass `with_usage=True` to the `instructor` client.

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

A new `instructor` client (which wraps an `httpx.AsyncClient`) is constructed on every retry attempt. The `AsyncClient` is never explicitly closed. Under a 3-retry policy with 10 concurrent fix groups, this spawns up to 30 unclosed HTTP client objects per batch. Under sustained load this exhausts the system's file descriptor limit.

**Impact:** File descriptor leak; server crashes after extended operation. Will manifest during any multi-hour demo run.  
**Fix:** Construct the `instructor` client once per agent instance in `__init__` and reuse it.

---

### BUG-04 — Duplicate `model_validator` import in `brain/schemas.py`

**Severity:** MEDIUM  
**File:** `brain/schemas.py:6`

```python
from pydantic import BaseModel, Field, model_validator, field_validator, model_validator
```

`model_validator` is imported twice in the same statement. No runtime failure but in Python 3.12+ this raises `SyntaxWarning`. More critically, it signals that this file — the schema backbone of the entire system — has not been run through any linter. The claim of running Ruff/Bandit/Semgrep on every patch is undermined by the own codebase failing `ruff check`.

**Fix:** `from pydantic import BaseModel, Field, model_validator, field_validator`

---

### BUG-05 — `FormalVerificationResult.solver_used` field name may mismatch storage writes

**Severity:** MEDIUM  
**File:** `brain/schemas.py:264` vs `agents/formal_verifier.py`

`FormalVerificationResult` declares `solver_used: str = 'z3'`. If any construction site uses `solver=` instead of `solver_used=`, Pydantic silently ignores the unknown kwarg (with `extra='ignore'`), leaving `solver_used` as `'z3'` regardless of which solver ran. Audit trail entries then falsely claim Z3 ran when the actual check was a regex pattern match.

**Impact:** DO-178C audit trail falsification — evidence artifact says "Z3 proved this" when the actual check was a word-overlap heuristic.  
**Fix:** Grep all `FormalVerificationResult(` construction sites and verify field name is `solver_used=`, not `solver=`.

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

If any code creates an `AuditRun` directly (test fixtures, resume path, admin endpoints), the schema default silently caps the run at 50 cycles — 4× fewer than stated. The comment on line 109 calls out the historic bug but does not fix the schema default.

**Fix:** `max_cycles: int = 200` in `AuditRun`.

---

### BUG-07 — `bobn_sampler.py` `BoBNCandidate` index mapping bug (acknowledged in code)

**Severity:** HIGH  
**File:** `agents/adversarial_critic.py:147–150`

```python
# BUG FIX: previously used `idx = len(reports)` to look up the original
# candidate, but len(reports) grows only when a *successful* report is
# appended — so as soon as one task fails, every subsequent failed task
# maps to the wrong candidate.
```

The comment documents a known bug that was supposedly fixed. The fix description is present in `adversarial_critic.py` as a comment. The fix itself uses `zip(candidates, results)` with `return_exceptions=True`. However, `asyncio.gather(..., return_exceptions=True)` returns results in the same order as the input tasks — so the zip is correct. But the partial-failure path (when `isinstance(r, Exception)`) silently drops the failed candidate entirely from the attack report list. If Fixer A's LLM call times out, its candidate receives no attack report, which means the BoBN composite scoring assigns it a default `attack_confidence=0.5` and it may win even if it produced a broken patch.

**Impact:** A timed-out candidate can win the BoBN selection silently, defeating the adversarial review gate.  
**Fix:** Explicitly record a `CriticAttackReport(attack_confidence=1.0, fallback_reason="critic_error")` for any candidate whose critique task fails, so it scores worst on the robustness dimension rather than average.

---

### BUG-08 — `workers/tasks.py` `approve_escalation_task` calls `load_config()` with no arguments

**Severity:** HIGH  
**File:** `workers/tasks.py:65`

```python
cfg = load_config()
```

`load_config()` with no arguments relies entirely on environment variables. If `DATABASE_URL` is not set in the Celery worker process (common when workers are launched on separate machines), `_require_database_url()` raises `RuntimeError`. The exception propagates out of the async inner function, but since `asyncio.run(_approve())` does not catch it, Celery marks the task as `FAILURE` with no user-visible notification. A human operator approving a safety-critical escalation gets no confirmation that the approval was processed. The escalation remains in `PENDING` status and eventually times out.

**Impact:** Human approval of safety-critical escalations silently fails in distributed deployments, allowing the system to proceed autonomously past a human gate.  
**Fix:** Add explicit `DATABASE_URL` validation before `load_config()` in all Celery tasks; add task failure alerting.

---

## 3 — ARCH: Architectural Gaps

### ARCH-01 — The "85%+ SWE-bench" claim is aspirational probability, not measured performance

**Severity:** CRITICAL  
**File:** `swe_bench/bobn_sampler.py:11–14`, `scripts/benchmark.py:57–87`, `scripts/arpo_trainer.py:12–15`

**The arithmetic breakdown against first-principles:**

| Component | Source claim | Realistic ceiling |
|-----------|-------------|-------------------|
| Base model (Qwen2.5-Coder-32B, no fine-tuning) | ~45–55% | ~45% on SWE-bench Verified |
| BoBN N=10 empirical lift | "12–18pp" (bobn_sampler.py:13) | +15pp → **60%** |
| CPG-enhanced localization (Joern required) | "+15–20pp on cross-file bugs" (benchmark.py:84) | Applies to ~40% of instances → +6pp → **66%** |
| Execution feedback loop (Docker required) | Necessary for correctness signal | Without Docker: word-overlap heuristic → **-10pp → 56%** |
| ARPO fine-tuning | "+3–5pp" (arpo_trainer.py:13) | Requires 4×A100 80GB; 500+ trajectories → aspirational |
| **Achievable today without Docker+Joern+ARPO** | | **~50–56%** |
| **Achievable with full infra (Docker+Joern+CPG)** | | **~63–68%** |
| **Theoretical max with ARPO fine-tuning** | | **~71–73%** |

The 85%+ claim requires: ARPO post-training, Joern CPG active, Docker sandbox available, AND assumes the BoBN empirical lift is at the top of its stated range. The benchmark script itself warns: *"Expected score penalty vs CPG-enabled: -15 to -20pp. Do NOT publish this score as the system's maximum capability."*

Any investor or technical judge will ask: "What is your current measured score on SWE-bench Verified?" The answer from this codebase is **"we haven't measured it."** Every named competitor in the space has published numbers.

**Impact:** Credibility destruction in any technical Q&A. Claiming 85%+ without measurement in the current SWE-bench competitive landscape is fraud-adjacent.  
**Fix:** Either run the official evaluation and report the actual number, or change the claim to "targeting 63–68% on SWE-bench Verified with full infrastructure; ARPO fine-tuning targeting 85% post-deployment (infrastructure required: Docker, Joern, 4×A100 80GB)."

---

### ARCH-02 — DO-178C compliance claim requires the tool itself to be DO-178C qualified

**Severity:** CRITICAL  
**File:** `compliance/sas_generator.py`, `orchestrator/controller.py:122`, `brain/schemas.py:182–188`

Rhodawk AI generates fixes, runs tests, generates SAS documents, and populates the RTM. All of these satisfy DO-178C Table A objectives. DO-178C §12.2 requires that any tool used to satisfy certification objectives without independent re-verification must hold a **Tool Qualification Document (TQD)** at TQL-1 through TQL-5.

No TQD exists. No qualification test suite is present. The `ToolQualificationLevel` enum exists in the schema but defaults to `NONE`. The `SASGenerator.generate()` method hardcodes `do178c_met = ["Table A-7 Obj 5 — Source code standards verification (partial)"]` and `do178c_open` as three hardcoded strings — these are not computed from actual run data; the SAS document is a template, not a real compliance artifact. The "partial" qualifier in the single met objective is not explained.

**Impact:** Any aerospace or defense customer's DER will reject this tool in the first technical review. The entire "safety-critical" market positioning is legally and technically untenable until tool qualification is obtained (typically 12–24 months, $500K+).  
**Fix:** Scope the tool explicitly as "advisory only — all outputs require independent human verification." Remove compliance marketing language until TQL status is demonstrated. Fix `sas_generator.py` to compute objectives from actual run evidence.

---

### ARCH-03 — Joern/CPG assumed running but not bundled; `cpg_enabled=True` by default

**Severity:** HIGH  
**File:** `orchestrator/controller.py:173–177`

```python
cpg_enabled: bool = True
joern_url:   str  = "http://localhost:8080"
```

Joern is a JVM application requiring Java 11+, ~2 GB RAM, and a separate process. It is not started by any workflow, not bundled in requirements, and not documented as a prerequisite outside config comments. With `cpg_enabled=True` by default, the three highest-value architectural features (Gap 1 causal CPG context, Gap 3 coupling-smell detection, Gap 4 commit-granularity audit) all silently degrade to stub behavior on every default deployment. The `joern_client.py` file is 1 211 lines; it attempts a connection at `localhost:8080` and gracefully returns empty results on failure — so the failure is invisible to operators.

**Impact:** The demo claims CPG-enhanced analysis while running on stub data. Benchmark scores without Joern are 15–20pp lower (benchmark.py:84), making any score published without Joern qualification misleading.  
**Fix:** Default `cpg_enabled=False`; activate only when `JOERN_URL` is explicitly set. Add a startup check that emits a prominent warning and prints the penalty estimate when CPG is disabled.

---

### ARCH-04 — Independence enforcement is string-matching on model family names, not cryptographic

**Severity:** HIGH  
**File:** `verification/independence_enforcer.py:39–94`

The DO-178C §6.3.4 independence check compares lowercase string representations of model family names extracted by prefix matching on the model ID. A misconfigured deployment using `"alibaba/custom-finetuned-model"` passes the independence check even if it was fine-tuned from identical weights as the fixer. Conversely, `"Qwen2.5"` and `"Qwen3"` may resolve to the same family or different ones depending on the regex precedence. The `MODEL_FAMILY_MAP` uses string prefix matching with no version disambiguation.

**Impact:** False independence certification in the audit trail.  
**Fix:** Independence must be declared explicitly in a versioned model registry (e.g., `model_registry.yaml`) with cryptographic model hashes, not inferred from model ID strings.

---

### ARCH-05 — Rate limiter fully implemented but never wired to any agent

**Severity:** HIGH  
**File:** `agents/base.py:243`, `utils/rate_limiter.py` (entire file)

`utils/rate_limiter.py` is a complete, correct token-bucket rate limiter with API key rotation and async lock. `BaseAgent.__init__` declares `self._rate_limiter: Any | None = None` and never assigns a `RateLimiter` instance. No agent, controller, or router imports or instantiates `RateLimiter`. There is no `await self._rate_limiter.get_key()` call anywhere in the codebase (confirmed by grep).

Under the default concurrency of 4 parallel fix groups, all four can simultaneously hit the same OpenRouter endpoint, triggering 429 rate limit errors that cascade into the fallback chain (which hits two phantom models) before failing.

**Impact:** Rate limit storm during any parallel workload. The retry logic exists but without rate limiting the retries themselves are rate-limited simultaneously.  
**Fix:** Instantiate `RateLimiter(rpm=60, tpm=100_000)` in `BaseAgent.__init__` and call `await self._rate_limiter.get_key()` before each LLM call.

---

### ARCH-06 — PostgreSQL and SQLite schemas are not guaranteed in sync; no migration framework

**Severity:** MEDIUM  
**File:** `brain/sqlite_storage.py:49–200` vs `brain/postgres_storage.py:29`

The SQLite DDL is defined inline in `sqlite_storage.py` (2 497 lines). PostgreSQL DDL is defined as a multi-line `_DDL_MAIN` string in `postgres_storage.py`. There is no migration framework (no Alembic, no versioned migrations). Schema changes to SQLite are not automatically applied to PostgreSQL. Given the active development pace (40+ schema fields added in this iteration), schema drift between the two backends is not a matter of "if" but "when."

**Impact:** Data persisted in development cannot be reliably migrated to production. Any field added to `AuditRun` or `Issue` requires manual DDL surgery on both backends.  
**Fix:** Introduce Alembic with a single SQLAlchemy Core schema definition generating both SQLite and PostgreSQL DDL.

---

### ARCH-07 — `PostgresBrainStorage.__getattr__` delegation is a silent correctness trap

**Severity:** HIGH  
**File:** `brain/postgres_storage.py:90–93`

```python
def __getattr__(self, name: str):
    if self._fallback is not None:
        return getattr(self._fallback, name)
    raise AttributeError(f'PostgresBrainStorage.{name} not implemented')
```

The abstract `BrainStorage` defines 40+ abstract methods. `PostgresBrainStorage` explicitly implements a subset. Any method NOT explicitly implemented raises `AttributeError` when PostgreSQL is active (no fallback). The only way to discover which methods are missing is to run the system against a live PostgreSQL instance. The `__getattr__` delegation means the test suite (which uses SQLite) will never reveal missing PostgreSQL implementations — all tests pass, production crashes.

**Impact:** Production-only crashes for any missing PostgreSQL method; the test suite provides false confidence.  
**Fix:** Add a static test that instantiates `PostgresBrainStorage` and calls every abstract method from `BrainStorage`, confirming all are implemented or explicitly delegated.

---

### ARCH-08 — `api/routes/runs.py` creates a `StabilizerController` per request with no timeout

**Severity:** HIGH  
**File:** `api/routes/runs.py:53–54`

```python
controller = StabilizerController(config)
run = await controller.initialise()
```

`controller.initialise()` includes git clone, Joern connection attempt, PostgreSQL pool creation, feature matrix check, and ARPO corpus scan. No timeout is applied to the route handler. A slow/unreachable git repo (or any of the other network operations) hangs the route handler indefinitely, blocking FastAPI's async event loop and making the API unresponsive to all other requests.

**Impact:** Single slow request kills the API for all concurrent users.  
**Fix:** Wrap `controller.initialise()` in `asyncio.wait_for(..., timeout=30.0)`.

---

## 4 — SEC: Security Vulnerabilities

### SEC-01 — WebSocket token in URL query parameter is logged by every reverse proxy

**Severity:** HIGH  
**File:** `auth/jwt_middleware.py:423`

```python
token = websocket.query_params.get("token")
```

JWT tokens passed as `?token=<jwt>` in WebSocket URLs are logged by Nginx, AWS ALB, Cloudflare, and every other reverse proxy in plain text in their access logs. For a system claiming military-grade security, this is a C-suite-visible finding.

**Fix:** Implement the Sec-WebSocket-Protocol handshake pattern (token in the subprotocol header) or a one-time upgrade ticket issued via REST before the WebSocket connection.

---

### SEC-02 — Missing webhook HMAC verification is only a WARNING in production

**Severity:** HIGH  
**File:** `api/app.py:131–138`

```python
if not os.environ.get("RHODAWK_WEBHOOK_SECRET"):
    log.warning("CI push webhooks will be accepted without signature verification.")
```

The webhook secret is optional — a WARNING, not a FATAL. An unauthenticated caller can POST to the CI webhook endpoint and trigger a full audit run. Combined with `auto_commit=True` (the default), an attacker can force Rhodawk to commit AI-generated patches to any configured repository without authentication.

**Impact:** Unauthenticated remote code-commit trigger. In a military/aerospace deployment this is a supply-chain attack vector.  
**Fix:** Make `RHODAWK_WEBHOOK_SECRET` required in production; raise `ConfigurationError` if absent.

---

### SEC-03 — `os._exit(1)` in FastAPI startup bypasses asyncio pool cleanup

**Severity:** MEDIUM  
**File:** `api/app.py:129`

The code acknowledges this bug in a comment (`ADD-3 NOTE — asyncio cleanup gap`) but does not fix it. On every invalid-config restart, the PostgreSQL connection pool is leaked. On a cloud deployment with 10 replicas each with a pool of 5 connections, 10 crash-restarts exhaust the PostgreSQL `max_connections` limit (default 100), making the database inaccessible to the entire fleet.

**Fix:** Move security checks to the `@asynccontextmanager` lifespan, before the pool is opened.

---

### SEC-04 — Prompt injection defense covers 6 patterns; adversarial repos use hundreds

**Severity:** MEDIUM  
**File:** `agents/base.py:72–82`

The `_INJECTION_STRIP_PATTERNS` list has 6 regex patterns. The `AegisEDR` in `security/aegis.py` is more comprehensive (Unicode NFC normalization, bidirectional control characters, null byte injection, f-string abuse). But `AegisEDR` is only used in `ReaderAgent` as a pre-scan — it does not guard LLM prompt construction in `BaseAgent.call_llm_structured()`. For a system claiming to process hostile/untrusted repositories, the injection surface between file reading and prompt injection is not covered.

**Impact:** A sufficiently adversarial repository can manipulate the auditor or fixer LLM into generating malicious patches or suppressing true findings.  
**Fix:** Run AegisEDR on all source code strings before they are inserted into LLM prompts, not just on file read.

---

### SEC-05 — `api/routes/runs.py` `app.state.controllers` grows unboundedly; memory leak

**Severity:** MEDIUM  
**File:** `api/routes/runs.py:60–77`

```python
app_state.controllers[run.id] = controller
```

For long-running stabilization tasks (hours), `app_state.controllers` holds a reference to every controller object. Each controller holds a storage connection pool, a Joern client, and potentially a vLLM connection. The `_on_done` callback removes the entry only when the task completes. If a run is cancelled mid-flight, the controller is never removed. For multi-tenant SaaS deployments, this is a slow memory leak that eventually exhausts server heap.

**Fix:** Implement a controller eviction policy (evict after max run time or on explicit cancellation); use `WeakValueDictionary` so completed-run controllers can be GC'd.

---

## 5 — DEMO: Things That Will Specifically Fail During Your Accelerator Pitch

### DEMO-01 — The system cannot start at all with default configuration

Combining BLOCK-04 (port 8000 conflict), BLOCK-05 (localhost:8001 required), BLOCK-06 (empty PostgreSQL DSN), and BLOCK-07 (no token endpoint): **a fresh clone with default environment will fail at startup before processing a single file.** There is no "happy path" default configuration that works on any single machine.

**Minimum viable demo config requires:**
- `RHODAWK_JWT_ALGORITHM=HS256` + valid `RHODAWK_JWT_SECRET`
- `use_sqlite=True` OR a running PostgreSQL instance + `DATABASE_URL`
- `gap5_enabled=False` (unless two vLLM servers are running on separate ports)
- `vllm_base_url` changed to a non-8000 port OR all models routed via cloud API keys
- A `/auth/token` endpoint added (none exists in any route file)
- A running Ollama with `qwen2.5-coder:32b` pulled, OR `primary_model` changed to a cloud model

None of this is documented. No `README.md` exists in this repository. `replit.md` describes the architecture but contains no quick-start guide.

---

### DEMO-02 — "Upload any codebase" claims require Docker sandbox not present

**Severity:** HIGH

The SWE-bench execution loop (`swe_bench/execution_loop.py`) requires Docker to run test candidates in isolated containers. The benchmark script warns: execution without Docker falls back to a word-overlap heuristic. When a demo audience member tries to upload their own codebase, the system either: (a) runs their code directly in the Replit environment (security disaster), or (b) fails with a Docker-not-found error.

**Impact:** The demo either crashes or runs untrusted code on the demo host.  
**Fix:** Implement process isolation using `subprocess` with `ulimit`/`seccomp` restrictions as an interim measure; document Docker requirement prominently.

---

### DEMO-03 — Lean 4 "formal verification" is a trivially-true tautology smoke-test

**Severity:** HIGH  
**File:** `verification/leanstral.py:46–48`

```python
lean_src = (
    f"theorem {safe_name}_stub : True := trivial\n"
)
```

`leanstral.py` is honest about this in its own docstring: *"STUB: Trivial proof — not a real property verification. Replace with domain-specific Lean 4 specification for DO-178C evidence."* The code proves `True := trivial` — a tautology provable in any type system, verifying nothing about the patched code. The file's own docstring warns: *"DO NOT cite llm_reasoning results as DO-178C formal evidence."*

The architecture documents and cover letter claim "Lean 4 formal verification" as a differentiator. The implementation is a smoke-test for whether `lean` is installed. Calling this "formal verification" in a DO-178C context is materially false.

**Impact:** If demonstrated to a DER or formal methods reviewer during due diligence, this immediately disqualifies the tool and damages credibility for all other claims.  
**Fix:** Remove "Lean 4 formal verification" from all marketing materials until a real property specification framework is built. State "LLM-assisted property reasoning (advisory only)" instead.

---

### DEMO-04 — Metrics endpoint exports nothing until a run completes

**Severity:** MEDIUM  
**File:** `api/app.py` (Prometheus initialization)

The Prometheus metrics and LangSmith tracer are initialized but only populated during an active pipeline run. If the demo audience asks "what's your current throughput / cost / issue resolution rate," the Prometheus scrape endpoint returns all-zero counters until at least one full cycle completes. An empty observability dashboard is actively harmful in a demo.

---

## 6 — COMP: Competitive Gaps

### COMP-01 — No measured SWE-bench score; every named competitor has published numbers

SWE-bench Verified leaderboard (Q1 2026): Cognition Devin 2 (53.6%), Amazon Q Developer (55.0%), GitHub Copilot Workspace (~44%), OpenHands (39.0%). Rhodawk's 85% claim, if true, would be the highest in the world by 30 points and would already be published and viral. Unsubstantiated claims in this space are immediately disqualifying in technical investor conversations.

**Fix:** Run the evaluation. Report the actual number. If it's 55%, say 55% with a clear roadmap to the target.

---

### COMP-02 — BoBN N=10 requires more GPU budget than most enterprise customers have

Four endpoint families simultaneously: Qwen2.5-Coder-32B + DeepSeek-Coder-V2-16B + Llama-3.3-70B (critic) + Devstral-Small (synthesis). The per-issue GPU cost is ~8–10× a single-model solution. Competitors (SWE-agent, Moatless, Agentless) run on a single GPT-4o or Claude call per issue. The "swarm" architecture is only economically viable for high-value issues in regulated industries — which is the right niche, but it must be stated explicitly rather than implied as general-purpose.

---

### COMP-03 — No GitHub Copilot / IDE integration

The system ships a REST API. Every competitor in the "AI code quality" space ships a VS Code extension, a GitHub App, or a Copilot plugin. An accelerator audience will immediately ask "how does a developer actually use this?" The answer ("POST to /api/runs/start") is a deal-breaker for developer-tool positioning.

---

### COMP-04 — ARPO trainer requires 4×A100 80GB; claims are contingent on hardware most buyers cannot access

**Severity:** HIGH  
**File:** `scripts/arpo_trainer.py:17–19`

```
# Primary path (OpenRLHF, 4xA100 80GB, ZeRO-3):
pip install openrlhf transformers accelerate deepspeed
```

The "85%+" target explicitly depends on ARPO fine-tuning. Fine-tuning a 32B model with GRPO requires ZeRO-3 across 4×A100 80GB — approximately $40–60/hour on cloud GPU (A100 80GB spot pricing). Running 3 epochs on 500 trajectories takes ~8–12 hours: ~$400–720 per fine-tuning run. This is not feasible without a dedicated ML infra team and budget. The TRL single-GPU fallback path (for 7B–14B models only) cannot fine-tune the 32B model that is the basis of the SWE-bench claim.

**Impact:** The performance headline is unreachable for any customer without dedicated ML infrastructure.

---

## 7 — MISSING: Unimplemented Features Claimed in the Architecture

### MISSING-01 — `leanstral.py` formal prover generates trivial stubs; never called from formal verifier

**Severity:** HIGH  
**File:** `verification/leanstral.py`

`leanstral.py` is honest about its limitations (stubs). More importantly, it is imported nowhere in `agents/formal_verifier.py` or any pipeline code. The formal verification pipeline uses Z3 and CBMC. Lean 4 / LeanSTRaL is one of the architectural claims in the marketing narrative ("formally verifiable patches") but the implementation is a dead file generating trivially-true tautologies.

---

### MISSING-02 — Federated fix-pattern store (`gap6`) has no peers to federate with

**Severity:** MEDIUM  
**File:** `orchestrator/controller.py:250–264`

Gap 6 (federated anonymized pattern store) is `gap6_federation_enabled=False` by default. The `gap6_registry_url` is empty. There are no known federation peers. The `gap6_instance_id` is auto-generated. Federating with yourself is not federated learning. This is entirely vaporware.

---

### MISSING-03 — `EscalationManager` has no verified notification transport

**Severity:** HIGH  
**File:** `orchestrator/controller.py` (escalation init)

`EscalationManager` is initialized with `api_base_url` and `timeout_hours`. If `api_base_url` is empty (the default), escalation notifications have no delivery mechanism. A CRITICAL finding in a DO-178C DAL-A module escalates to `ESCALATION_PENDING` and then times out with no human ever notified. The compliance guarantee collapses silently. Combined with BUG-08 (Celery approval task crashing silently), human-in-the-loop for safety-critical decisions is broken end-to-end.

---

### MISSING-04 — `SASGenerator` hardcodes DO-178C objectives as static strings, not computed evidence

**Severity:** HIGH  
**File:** `compliance/sas_generator.py:52–57`

```python
do178c_met  = ["Table A-7 Obj 5 — Source code standards verification (partial)"]
do178c_open = [
    "Table A-4 — High-level requirements accuracy (requires human review)",
    "Table A-5 — Traceability (RTM partially populated)",
    "Table A-7 Obj 6 — Absence of unintended functions",
]
```

These lists are identical on every run, regardless of what actually happened during the audit. The SAS document always claims exactly one objective is met and exactly three are open — no matter how many cycles ran, how many issues were resolved, or what domains were tested. A DER will reject this as a template, not evidence.

**Fix:** Compute DO-178C objective compliance dynamically from actual run artifacts: RTM completeness, deviations approved, independence records present, formal verification results.

---

### MISSING-05 — `rank_bm25` dependency in localization pipeline may not be installed

**Severity:** MEDIUM  
**File:** `swe_bench/localization.py:254–258`

```python
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    # Fallback: simple keyword overlap scoring
    return self._keyword_overlap_score(issue_text, files)
```

`rank_bm25` is not in `requirements.txt` or `pyproject.toml` (not visible in any config file). The fallback to keyword overlap silently degrades Phase A file localization quality — and the localization accuracy drop (from BM25 78% top-1 to keyword overlap ~55%) directly reduces the SWE-bench score by 10–15pp. The warning is only a debug log; no startup check confirms this dependency.

**Fix:** Add `rank_bm25` to requirements; add a startup check that warns prominently if it's absent.

---

## 8 — TEST: Test Suite Integrity

### TEST-01 — Unit tests mock the LLM and storage layers; no end-to-end coverage

**Severity:** HIGH  
**Files:** `tests/unit/` (15 files)

The unit test suite covers individual agent logic with mocked LLM responses and in-memory storage. There are zero tests that exercise a real LLM call against a real repository. The SWE-bench score cannot be validated by any test in this suite.

---

### TEST-02 — Tests cannot detect the BLOCK bugs found in this review

The test suite could not have caught BLOCK-01 through BLOCK-08 because:
- BLOCK-01/02/03: model tags are never resolved against a real LLM API in tests
- BLOCK-04: port 8000 collision is not tested (no dual-service test)
- BLOCK-05: gap5 secondary endpoint is mocked
- BLOCK-06: SQLite is used in all tests (`use_sqlite=True` in fixtures)
- BLOCK-07: no `/auth/token` endpoint means no test can exercise the full auth flow
- BLOCK-08: synthesis_model independence check uses empty-string shortcut that always passes

What is needed is a smoke-test that validates the config defaults can produce a working system without mocks.

---

### TEST-03 — `RHODAWK_ENV=development` in test fixtures means production security is never tested

**Severity:** MEDIUM**

If `conftest.py` sets `RHODAWK_ENV=development` (standard practice), the `_is_production` branches in `api/app.py` and `auth/jwt_middleware.py` are never exercised in CI. The RS256 key requirement, the dev-auth `SystemExit`, and the CORS allowlist restriction are never tested in production configurations.

---

### TEST-04 — `tests/unit/test_gap5_swarm_intelligence.py` tests the scaffolding, not the models

**Severity:** MEDIUM  
**File:** `tests/unit/test_gap5_swarm_intelligence.py`

The Gap 5 swarm tests mock both vLLM endpoints and verify that the BoBN candidate selection logic calls the correct methods in the correct order. They do not test that the adversarial critic actually finds weaknesses, that the synthesis model produces valid unified diffs, or that the composite scoring formula correctly ranks candidates. All numerical assertions are against mock return values, not real model outputs.

---

## Summary Scorecard

| Category | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| BLOCK (demo crash) | 8 | 0 | 0 | 0 |
| BUG (silent failures) | 0 | 4 | 4 | 0 |
| ARCH (architectural gaps) | 2 | 5 | 1 | 0 |
| SEC (security) | 0 | 2 | 3 | 0 |
| DEMO (pitch killers) | 0 | 2 | 2 | 0 |
| COMP (competitive gaps) | 1 | 2 | 1 | 0 |
| MISSING (unimplemented) | 0 | 3 | 2 | 0 |
| TEST (test integrity) | 0 | 1 | 3 | 0 |
| **Total** | **11** | **19** | **16** | **0** |

**The system cannot run a demo in its current default state. Fix the 8 BLOCK bugs before any accelerator conversation.**

---

## SWE-bench Performance Ceiling Analysis

The 85%+ claim is the central marketable claim and the most scrutinized. Here is the honest breakdown:

**Infrastructure actually shipped:**
- ✅ BoBN N=10 sampling framework (implemented, correct)
- ✅ Agentless-style localization (BM25 + LLM rerank, requires `rank_bm25`)
- ✅ Adversarial critic (Llama-3.3-70B family independence enforced)
- ✅ Patch synthesis agent (Devstral-Small, Mistral family)
- ✅ ARPO trainer script (complete, correct CLI)
- ❌ Docker sandbox (not bundled; execution loop falls back to word-overlap heuristic)
- ❌ Joern/CPG (not bundled; 15–20pp penalty on cross-file bugs)
- ❌ `rank_bm25` (not in requirements; fallback degrades localization)
- ❌ Two vLLM inference servers (required for gap5, not runnable on single node by default)
- ❌ 4×A100 80GB (required for ARPO; not available in demo/development)
- ❌ 500+ resolved SWE-bench trajectories (required ARPO corpus; not present)

**Realistic performance without full infrastructure: ~50–56%**  
**Realistic performance with Docker + Joern + CPG: ~63–68%**  
**Theoretical performance with ARPO fine-tuning: ~71–73%**  
**Claimed performance: 85%+** — requires all of the above plus additional gains not clearly sourced

---

## Minimum Viable Demo Fix Checklist

In priority order (each item is a prerequisite for the next):

1. **Add `/auth/token` endpoint** (BLOCK-07) — zero API access without it
2. **Fix `postgres_dsn` default** — set `use_sqlite=True` as default for development (BLOCK-06)
3. **Set `gap5_enabled=False` as default** — until dual-GPU infra is documented (BLOCK-05)
4. **Fix port 8000 collision** — change `vllm_base_url` default or document override (BLOCK-04)
5. **Fix three phantom model IDs** — `triage_model`, `critical_fix_model`, two `fallback_models` (BLOCK-01/02/03)
6. **Fix `synthesis_model` default** — change `""` to `"openrouter/deepseek-ai/DeepSeek-Coder-V2-Instruct"` (BLOCK-08)
7. **Add `medium_count` to `AuditScore`** — silent score falsification (BUG-01)
8. **Replace hardcoded `completion_tokens=500`** — cost ceiling enforcement is broken (BUG-02)
9. **Wire `RateLimiter` into `BaseAgent.__init__`** — rate limit storm on parallel workloads (ARCH-05)
10. **Add `rank_bm25` to requirements** — localization degrades silently without it (MISSING-05)
11. **Write a `README.md` quick-start** — no deployer can get this running without it
12. **Replace "85%+ SWE-bench" with honest qualified numbers** (ARCH-01/COMP-01)
13. **Remove "Lean 4 formal verification" claim** — replace with "LLM property reasoning (advisory)" (DEMO-03)
14. **Qualify all DO-178C compliance claims** — add "advisory tool, independent human verification required" (ARCH-02/MISSING-04)

---

*End of adversarial review. Total findings: 46. Critical/High: 30.*

*The codebase shows sophisticated architectural thinking — the BoBN pipeline, CPG-based blast radius, Agentless-style localization, and DO-178C schema work are genuinely impressive at the design level. The implementation is approximately 55–60% complete with respect to the full claimed feature set. The foundational correctness issues (BLOCK-01 through BLOCK-08) are all fixable in one focused engineering sprint of 3–5 days. The strategic credibility issues (ARCH-01, ARCH-02, DEMO-03) are more serious: they require honest re-scoping of the product narrative before any investor technical due diligence.*
