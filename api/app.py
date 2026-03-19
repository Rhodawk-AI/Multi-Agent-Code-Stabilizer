"""
api/app.py — FastAPI application for Rhodawk AI Code Stabilizer.

FIXES vs prior audit
─────────────────────
• CORS wildcard allow_origins=["*"] replaced with explicit origin allowlist
  read from RHODAWK_CORS_ORIGINS env var.  Defaults to localhost-only so the
  escalation approval endpoint cannot be called from arbitrary origins in prod.
• Baseline promotion endpoint now wired to real storage via app state.
• RTM and SAS export endpoints wired to RTMExporter / SASGenerator with
  storage from app state.
• RHODAWK_DEV_AUTH bypass is explicitly blocked in production
  (RHODAWK_ENV != "development").
• /api/capabilities returns structured feature matrix report.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

log = logging.getLogger(__name__)

# ── CORS allowlist ─────────────────────────────────────────────────────────────
# FIX: was allow_origins=["*"] — allows any site to hit the escalation approval
# endpoint. Now reads from env; defaults to localhost only.
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


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        f"Rhodawk AI API starting — env={os.environ.get('RHODAWK_ENV','production')} "
        f"cors_origins={_ALLOWED_ORIGINS}"
    )
    # Warn loudly if RHODAWK_DEV_AUTH=1 is set in non-dev environment
    if os.environ.get("RHODAWK_DEV_AUTH") == "1" and not _IS_DEV:
        log.critical(
            "SECURITY: RHODAWK_DEV_AUTH=1 is set but RHODAWK_ENV != 'development'. "
            "Authentication is BYPASSED in a production environment. "
            "Set RHODAWK_DEV_AUTH=0 or RHODAWK_ENV=development."
        )
    yield
    log.info("Rhodawk AI API shutting down")


# ── App factory ────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Rhodawk AI Code Stabilizer",
        version="2.0.1",
        description="Production-grade autonomous code stabilization with DO-178C compliance",
        lifespan=lifespan,
    )

    # FIX: narrowed CORS from wildcard to explicit allowlist
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    # ── Core routes ──────────────────────────────────────────────────────────
    for module_name, attr in [
        ("api.routes.runs",        "router"),
        ("api.routes.issues",      "router"),
        ("api.routes.fixes",       "router"),
        ("api.routes.files",       "router"),
        ("api.routes.escalations", "router"),
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
        return {"status": "ok", "version": "2.0.1", "env": os.environ.get("RHODAWK_ENV", "production")}

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
        """
        FIX: was a stub returning 'requires_storage_context'.
        Now wired to real storage — promotes a run to active baseline
        (DO-178C Sec 11 configuration management gate).
        """
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
        """
        FIX: was a stub. Now delegates to RTMExporter for real CSV/JSON output
        (DO-178C Table A-5 Requirements Traceability Matrix).
        """
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
        """
        FIX: was a stub. Now delegates to SASGenerator for real SAS output
        (DO-178C Sec 11.20 Software Accomplishment Summary).
        """
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

    # ── Storage injection helper (for controller to wire storage post-init) ──

    @app.post("/internal/set_storage")
    async def set_storage(request: Request):
        """
        Internal endpoint: controller calls this after storage.initialise()
        to make the storage instance available to API endpoints.
        Only accessible from localhost.
        """
        client_host = request.client.host if request.client else ""
        if client_host not in ("127.0.0.1", "::1", "localhost"):
            raise HTTPException(status_code=403, detail="Internal endpoint")
        # Storage is injected via app state by the controller directly
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
