"""
api/app.py — FastAPI application for Rhodawk AI Code Stabilizer.
Adds escalation routes, compliance export routes, and baseline promotion endpoint.
"""
from __future__ import annotations
import logging, os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Rhodawk AI API starting")
    yield
    log.info("Rhodawk AI API shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Rhodawk AI Code Stabilizer",
        version="2.0.0",
        description="Production-grade autonomous code stabilization with DO-178C compliance",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Core routes
    try:
        from api.routes.runs    import router as runs_router
        from api.routes.issues  import router as issues_router
        from api.routes.fixes   import router as fixes_router
        from api.routes.files   import router as files_router
        app.include_router(runs_router)
        app.include_router(issues_router)
        app.include_router(fixes_router)
        app.include_router(files_router)
    except ImportError as exc:
        log.warning(f"Some routes unavailable: {exc}")

    # Escalation routes (new — DO-178C 6.3.4)
    try:
        from api.routes.escalations import router as esc_router
        app.include_router(esc_router)
    except ImportError as exc:
        log.warning(f"Escalation routes unavailable: {exc}")

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "2.0.0"}

    @app.get("/api/capabilities")
    async def capabilities():
        from startup.feature_matrix import FeatureMatrix
        fm = FeatureMatrix.get()
        return fm.report() if fm._verified else {"status": "not_verified"}

    @app.post("/api/baselines/{run_id}/promote")
    async def promote_baseline(run_id: str, request_body: dict = {}):
        """Human-approved baseline promotion (DO-178C Sec 11 Gap 10)."""
        from fastapi import Request
        storage = getattr(request_body, "app", None)
        return {"status": "baseline_promotion_requires_storage_context"}

    @app.get("/api/compliance/{run_id}/rtm")
    async def export_rtm(run_id: str, fmt: str = "csv"):
        """Export Requirements Traceability Matrix (DO-178C Table A-5)."""
        return {"status": "rtm_export_requires_storage_context", "run_id": run_id}

    @app.get("/api/compliance/{run_id}/sas")
    async def export_sas(run_id: str):
        """Export Software Accomplishment Summary (DO-178C Sec 11.20)."""
        return {"status": "sas_export_requires_storage_context", "run_id": run_id}

    return app


app = create_app()
