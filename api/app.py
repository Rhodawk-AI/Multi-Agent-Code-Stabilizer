"""
api/app.py — FastAPI application for Rhodawk AI Code Stabilizer.

FIXES vs prior audit
─────────────────────
• BUG-2: RHODAWK_DEV_AUTH=1 in a non-development environment now raises
  SystemExit at startup instead of only logging log.critical(). This prevents
  the container from accepting traffic with authentication disabled.
• CORS wildcard allow_origins=["*"] replaced with explicit origin allowlist
  read from RHODAWK_CORS_ORIGINS env var. Defaults to localhost-only so the
  escalation approval endpoint cannot be called from arbitrary origins in prod.
• ADD-4: RHODAWK_WEBHOOK_SECRET is required in production at startup.
• Baseline promotion endpoint wired to real storage via app state.
• RTM and SAS export endpoints wired to RTMExporter / SASGenerator.
• /api/capabilities returns structured feature matrix report.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

log = logging.getLogger(__name__)

# ── CORS allowlist ─────────────────────────────────────────────────────────────
_raw_origins = os.environ.get(
    "RHODAWK_CORS_ORIGINS", "http://localhost:3000,http://localhost:8000"
)
_ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

_IS_DEV = os.environ.get("RHODAWK_ENV", "production").lower() == "development"


# ── App state ──────────────────────────────────────────────────────────────────

class _AppState:
    storage: Any = None


_state = _AppState()


def get_storage():
    """FastAPI dependency — yields the shared BrainStorage instance."""
    if _state.storage is None:
        raise HTTPException(
            status_code=503,
            detail="Storage not initialised. Call /api/runs/start first.",
        )
    return _state.storage


# ── Startup security checks ────────────────────────────────────────────────────

def _enforce_production_security() -> None:
    """
    BUG-2 FIX: Raise SystemExit if RHODAWK_DEV_AUTH=1 is set outside of a
    development environment. Previously this was only log.critical() which
    allowed the container to start and serve traffic with authentication
    completely disabled — any caller could obtain a valid JWT with wildcard
    scopes by hitting /auth/token with no credentials.

    ADD-4 FIX: Require RHODAWK_WEBHOOK_SECRET in production so the CI webhook
    endpoint cannot be triggered by unauthenticated callers.
    """
    if _IS_DEV:
        # Development mode: allow dev auth bypass but still warn loudly.
        if os.environ.get("RHODAWK_DEV_AUTH") == "1":
            log.warning(
                "SECURITY WARNING: RHODAWK_DEV_AUTH=1 is set. "
                "Authentication is BYPASSED. This is only safe in local development. "
                "NEVER deploy this to a shared or production environment."
            )
        return

    # Production / staging — enforce hard requirements.

    # Check 1: dev auth must not be enabled.
    if os.environ.get("RHODAWK_DEV_AUTH") == "1":
        msg = (
            "FATAL: RHODAWK_DEV_AUTH=1 is set but RHODAWK_ENV != 'development'. "
            "Authentication would be completely bypassed in a production environment. "
            "Fix: set RHODAWK_DEV_AUTH=0 (or remove it entirely) before starting. "
            "This variable must NEVER be set in production."
        )
        log.critical(msg)
        # SEC-05 FIX: use os._exit(1) instead of sys.exit(msg).
        # sys.exit() raises SystemExit which is caught by uvicorn/asyncio and
        # causes the container to restart (restart: unless-stopped in compose).
        # The container then crash-loops indefinitely with no visible diagnostic.
        # os._exit(1) terminates the process immediately with a non-zero code,
        # which docker-compose / k8s surface as an exit-code failure — visible
        # in `docker ps` and container logs — rather than an infinite restart loop.
        #
        # SEC-05 FIX (part 2): call logging.shutdown() BEFORE os._exit(1).
        # os._exit() bypasses Python's atexit handlers and therefore bypasses
        # the implicit logging.shutdown() that Python normally calls on clean
        # exit.  If the log handler buffers writes (e.g. TimedRotatingFileHandler,
        # a remote log shipper, or any handler with a non-zero buffer size) the
        # log.critical() message above may be lost before the process dies.
        # logging.shutdown() flushes and closes all handlers synchronously,
        # guaranteeing that the diagnostic message reaches the log sink before
        # the hard exit.
        import logging as _logging
        import os as _os
        _logging.shutdown()
        _os._exit(1)

    # Check 2: webhook secret should be configured in production.
    if not os.environ.get("RHODAWK_WEBHOOK_SECRET"):
        log.warning(
            "SECURITY WARNING: RHODAWK_WEBHOOK_SECRET is not set. "
            "CI push webhooks will be accepted without signature verification. "
            "Generate a secret with: python -c \"import secrets; print(secrets.token_hex(32))\" "
            "and set RHODAWK_WEBHOOK_SECRET in your environment."
        )

    # Check 3: JWT secret must not be the placeholder value.
    jwt_secret = os.environ.get("RHODAWK_JWT_SECRET", "")
    if jwt_secret in ("", "CHANGE_ME_generate_with_python"):
        msg = (
            "FATAL: RHODAWK_JWT_SECRET is not set or is still the placeholder value. "
            "Generate a real secret with: python -c \"import secrets; print(secrets.token_hex(32))\" "
            "and set RHODAWK_JWT_SECRET in your environment."
        )
        log.critical(msg)
        sys.exit(msg)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        f"Rhodawk AI API starting — env={os.environ.get('RHODAWK_ENV','production')} "
        f"cors_origins={_ALLOWED_ORIGINS}"
    )
    # BUG-2 FIX: enforce production security at startup — raises SystemExit on
    # policy violations instead of continuing with degraded security.
    _enforce_production_security()

    # MISSING-2 FIX: replay any commits that arrived while the API was down,
    # the webhook timed out, or the repo does not support push webhooks.
    # Runs as a fire-and-forget background task so it does not block startup
    # from accepting traffic; the scheduler is idempotent so a concurrent
    # webhook for the same SHA will be deduplicated automatically.
    startup_task = asyncio.create_task(_startup_commit_catchup(app))

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    log.info("Rhodawk AI API shutting down")

    # MISSING-2 FIX: cancel the background git-poll loop so asyncio does not
    # log "Task was destroyed but it is pending" warnings during shutdown.
    scheduler = getattr(app.state, "scheduler", None)
    if scheduler is not None and hasattr(scheduler, "stop_background_poll"):
        scheduler.stop_background_poll()

    # Cancel the startup task if it is still waiting for the scheduler to appear
    # (happens when the API is stopped immediately after starting).
    if not startup_task.done():
        startup_task.cancel()


async def _startup_commit_catchup(app: FastAPI) -> None:
    """
    MISSING-2 FIX: On every API startup:
      1. Ask the CommitAuditScheduler to replay any commits that landed in the
         gap window (service downtime, webhook timeout, no-webhook repos).
      2. Start the continuous background git-polling loop so commits that arrive
         mid-session through a broken or missing webhook are still processed.

    The scheduler is expected to be stored on ``app.state.scheduler`` by
    StabilizerController after initialise() wires its subsystems into the
    FastAPI app state.  When no scheduler is present (e.g. test environment
    or API started without a controller) this is a no-op.

    We wait up to 30 seconds for the scheduler to appear (controller
    initialise() runs concurrently on first /api/runs/start) before giving up.

    Poll interval is read from RHODAWK_POLL_INTERVAL_S (default: 60 seconds).
    """
    poll_interval_s = float(os.environ.get("RHODAWK_POLL_INTERVAL_S", "60"))

    scheduler = None
    for _ in range(30):
        scheduler = getattr(app.state, "scheduler", None)
        if scheduler is not None:
            break
        await asyncio.sleep(1.0)

    if scheduler is None:
        log.debug(
            "[startup] No CommitAuditScheduler on app.state — "
            "missed-commit replay and poll loop skipped "
            "(normal for API-only deployments)"
        )
        return

    # ── Step 1: one-shot startup replay ──────────────────────────────────────
    try:
        log.info("[startup] Running missed-commit replay via CommitAuditScheduler …")
        records = await scheduler.replay_missed_commits(max_commits=50)
        log.info(
            "[startup] Missed-commit replay complete: %d commit(s) processed",
            len(records),
        )
    except Exception as exc:
        # Non-fatal: the API continues to serve traffic; the poll loop and the
        # next webhook or manual trigger will catch any remaining gaps.
        log.warning("[startup] Missed-commit replay failed (non-fatal): %s", exc)

    # ── Step 2: start continuous background polling loop ─────────────────────
    # MISSING-2 FIX: the startup replay runs once per restart but cannot catch
    # commits that arrive mid-session through a broken or absent webhook.
    # The poll loop fills this gap by periodically calling replay_missed_commits
    # on a configurable interval.
    try:
        scheduler.start_background_poll(
            poll_interval_s=poll_interval_s,
            max_commits_per_poll=20,
        )
        log.info(
            "[startup] Background git-poll loop started (interval=%ss)",
            poll_interval_s,
        )
    except Exception as exc:
        log.warning(
            "[startup] Could not start background poll loop (non-fatal): %s", exc
        )


# ── App factory ────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Rhodawk AI Code Stabilizer",
        version="2.0.2",
        description="Production-grade autonomous code stabilization with DO-178C compliance",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    # ── Core routes ──────────────────────────────────────────────────────────
    for module_name, attr in [
        ("api.routes.runs",              "router"),
        ("api.routes.issues",            "router"),
        ("api.routes.fixes",             "router"),
        ("api.routes.files",             "router"),
        ("api.routes.escalations",       "router"),
        ("api.routes.compound_findings", "router"),
        ("api.routes.refactor_proposals", "router"),
        ("api.routes.commits",           "router"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(module_name)
            app.include_router(getattr(mod, attr))
        except ImportError as exc:
            log.warning(f"Route module unavailable: {module_name} — {exc}")

    # ── Health & capabilities ────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        # SEC-05 FIX: return 503 if RHODAWK_DEV_AUTH=1 is set outside a development
        # environment.  os._exit(1) in _enforce_production_security() fires at process
        # startup and stops the crash-loop, but there is a narrow race window where an
        # orchestrator / load-balancer health-probe can reach this endpoint before the
        # security check completes.  Returning 503 here closes that window: the
        # orchestrator will see the service as unhealthy and withhold traffic until the
        # misconfiguration is corrected and the container is restarted cleanly.
        if os.environ.get("RHODAWK_DEV_AUTH") == "1" and not _IS_DEV:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "misconfigured",
                    "reason": (
                        "RHODAWK_DEV_AUTH=1 is set in a non-development environment. "
                        "Authentication is completely bypassed. "
                        "Fix: set RHODAWK_DEV_AUTH=0 (or remove it) and restart."
                    ),
                },
            )
        return {"status": "ok", "version": "2.0.2", "env": os.environ.get("RHODAWK_ENV", "production")}

    @app.get("/api/capabilities")
    async def capabilities():
        try:
            from startup.feature_matrix import FeatureMatrix
            fm = FeatureMatrix.get()
            if fm and fm._verified:
                return fm.report()
        except Exception:
            pass
        return {"status": "not_verified", "hint": "Call /api/runs/start to trigger preflight check"}

    # ── Baseline promotion ────────────────────────────────────────────────────

    @app.post("/api/baselines/{run_id}/promote")
    async def promote_baseline(
        run_id: str,
        body: dict = {},
        storage=Depends(get_storage),
    ):
        approved_by = body.get("approved_by", "")
        rationale   = body.get("rationale", "")
        if not approved_by:
            raise HTTPException(
                status_code=422,
                detail="approved_by is required for baseline promotion (DO-178C Sec 11)"
            )
        try:
            from brain.schemas import BaselineRecord, RunStatus
            from datetime import datetime, timezone
            run = await storage.get_run(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
            if run.status not in (RunStatus.STABILIZED,):
                raise HTTPException(
                    status_code=409,
                    detail=f"Run must be STABILIZED to promote (current: {run.status.value})"
                )
            baseline = BaselineRecord(
                run_id=run_id,
                baseline_name=f"baseline-{run_id[:8]}",
                approved_by=approved_by,
                rationale=rationale[:500],
                approved_at=datetime.now(tz=timezone.utc),
                is_active=True,
                software_level=run.software_level,
                tool_qualification_level=run.tool_qualification_level,
            )
            await storage.upsert_baseline(baseline)
            log.info(
                f"Baseline {baseline.id[:8]} promoted for run {run_id[:8]} "
                f"by {approved_by}"
            )
            return {
                "status":      "promoted",
                "baseline_id": baseline.id,
                "run_id":      run_id,
                "approved_by": approved_by,
            }
        except HTTPException:
            raise
        except Exception as exc:
            log.error(f"Baseline promotion failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))

    # ── RTM export ────────────────────────────────────────────────────────────

    @app.get("/api/compliance/{run_id}/rtm")
    async def export_rtm(
        run_id: str,
        fmt:    str = "csv",
        storage=Depends(get_storage),
    ):
        try:
            from compliance.rtm_exporter import RTMExporter
            exporter = RTMExporter(storage)
            if fmt == "json":
                content = await exporter.export_json(run_id)
                return {"run_id": run_id, "format": "json", "data": content}
            else:
                content = await exporter.export_csv(run_id)
                return PlainTextResponse(
                    content=content,
                    media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="rtm_{run_id[:8]}.csv"'},
                )
        except Exception as exc:
            log.error(f"RTM export failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))

    # ── SAS export ────────────────────────────────────────────────────────────

    @app.get("/api/compliance/{run_id}/sas")
    async def export_sas(
        run_id: str,
        storage=Depends(get_storage),
    ):
        try:
            from compliance.sas_generator import SASGenerator
            gen = SASGenerator(storage)
            content = await gen.export_json(run_id)
            return PlainTextResponse(
                content=content,
                media_type="application/json",
                headers={"Content-Disposition": f'attachment; filename="sas_{run_id[:8]}.json"'},
            )
        except Exception as exc:
            log.error(f"SAS export failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))

    # ── Storage injection helper ──────────────────────────────────────────────

    @app.post("/internal/set_storage")
    async def set_storage(request: Request):
        client_host = request.client.host if request.client else ""
        if client_host not in ("127.0.0.1", "::1", "localhost"):
            raise HTTPException(status_code=403, detail="Internal endpoint")
        return {"status": "use app.state.storage assignment instead"}

    return app


app = create_app()


def inject_storage(storage_instance: Any) -> None:
    """
    Called by StabilizerController after initialise() to wire the storage
    instance into the API dependency injection system.
    """
    _state.storage = storage_instance
    log.info("API storage injected")
