"""
api/app.py — B2/B4/B9 fixes: JWT auth, metrics endpoint, WS auth.
"""
from __future__ import annotations
import logging, os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from auth.jwt_middleware import create_access_token, get_current_user
from metrics.prometheus_exporter import make_metrics_app, set_build_info

log = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    set_build_info(version="1.0.0", commit=os.environ.get("GIT_COMMIT","unknown"))
    log.info("Rhodawk AI API starting")
    yield

app = FastAPI(
    title="Rhodawk AI Code Stabilizer", version="1.0.0",
    description="Swarm-based autonomous code engineer — targets 85%+ SWE-bench",
    docs_url="/docs", redoc_url="/redoc", lifespan=lifespan,
)
app.add_middleware(CORSMiddleware,
    allow_origins=os.environ.get("RHODAWK_CORS_ORIGINS","*").split(","),
    allow_methods=["GET","POST","PUT","DELETE"],
    allow_headers=["Authorization","Content-Type"])

app.mount("/metrics", make_metrics_app())

@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok", "service": "rhodawk-ai"}

@app.post("/auth/token", tags=["auth"])
async def get_token(username: str, password: str):
    if os.environ.get("RHODAWK_DEV_AUTH","0") != "1":
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED,
            "Set RHODAWK_DEV_AUTH=1 for dev or wire identity provider")
    token = create_access_token(sub=username, scopes=["runs:read","runs:write","admin"])
    return {"access_token": token, "token_type": "bearer"}

def _register_routes():
    try:
        from api.routes.runs    import router as runs_router
        from api.routes.issues  import router as issues_router
        from api.routes.fixes   import router as fixes_router
        from api.routes.files   import router as files_router
        from api.websocket.manager import router as ws_router
        app.include_router(runs_router,   prefix="/api/v1")
        app.include_router(issues_router, prefix="/api/v1")
        app.include_router(fixes_router,  prefix="/api/v1")
        app.include_router(files_router,  prefix="/api/v1")
        app.include_router(ws_router)
    except ImportError as e:
        log.warning(f"Route registration partial: {e}")

_register_routes()
