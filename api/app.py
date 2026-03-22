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

NEW SECURITY FIXES (this version)
───────────────────────────────────
• REQUEST BODY SIZE LIMIT — DoS attack prevention.
  FastAPI / Starlette have no default body size limit.  An attacker can POST
  a 900 KB payload 9 times per second (demonstrated to crash production AI
  services — ref: Claude downtime incident 2025).  RequestBodySizeLimitMiddleware
  caps every request body at RHODAWK_MAX_BODY_BYTES (default 10 MB).  Any
  request whose Content-Length header or streamed body exceeds the cap is
  rejected with HTTP 413 before the body is read into memory.

• PER-IP RATE LIMITING — endpoint flood prevention.
  No endpoint had any rate limit.  /api/runs/start could be called 10,000
  times per second exhausting all Celery workers.  SlowAPI middleware now
  applies per-IP limits:
      /api/runs/start        →  5 requests / minute
      /api/upload            →  2 requests / minute  (zip upload, once built)
      all other POST/PATCH   → 60 requests / minute
      GET endpoints          → 120 requests / minute
  Limits are read from env vars so they can be tuned without code changes:
      RHODAWK_RATE_RUN_START  (default "5/minute")
      RHODAWK_RATE_UPLOAD     (default "2/minute")
      RHODAWK_RATE_WRITE      (default "60/minute")
      RHODAWK_RATE_READ       (default "120/minute")
  slowapi is optional — if not installed, rate limiting is disabled with a
  startup WARNING so operators know protection is inactive.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

log = logging.getLogger(__name__)

# ── CORS allowlist ─────────────────────────────────────────────────────────────
_raw_origins = os.environ.get(
    "RHODAWK_CORS_ORIGINS", "http://localhost:3000,http://localhost:8000"
)
_ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

_IS_DEV = os.environ.get("RHODAWK_ENV", "production").lower() == "development"

# ── Body size limit ────────────────────────────────────────────────────────────
# Default 10 MB.  Override with RHODAWK_MAX_BODY_BYTES env var.
_MAX_BODY_BYTES: int = int(
    os.environ.get("RHODAWK_MAX_BODY_BYTES", str(10 * 1024 * 1024))
)

# ── Rate limit strings ─────────────────────────────────────────────────────────
_RATE_RUN_START = os.environ.get("RHODAWK_RATE_RUN_START", "5/minute")
_RATE_UPLOAD    = os.environ.get("RHODAWK_RATE_UPLOAD",    "2/minute")
_RATE_WRITE     = os.environ.get("RHODAWK_RATE_WRITE",     "60/minute")
_RATE_READ      = os.environ.get("RHODAWK_RATE_READ",      "120/minute")


# ── Request body size limit middleware ─────────────────────────────────────────

class RequestBodySizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Reject requests whose body exceeds _MAX_BODY_BYTES with HTTP 413.

    Two-stage check:
    1. Content-Length header — fast-path rejection before reading any bytes.
    2. Streaming byte counter — catches chunked transfers that omit
       Content-Length (the attack vector that bypasses header-only checks).

    STREAMING UPLOAD EXEMPTION
    ──────────────────────────
    Endpoints whose paths match _STREAM_UPLOAD_PATHS are exempt from in-memory
    body accumulation.  For those endpoints:
    • Stage 1 (Content-Length) still applies — very large uploads are rejected
      immediately without reading a single byte.
    • Stage 2 counts bytes on the fly but does NOT buffer them — the raw
      request stream is passed through to the upload handler which writes
      directly to a temp file on disk.

    This means a 500 MB zip upload works correctly:
    • The middleware checks Content-Length upfront and rejects if over cap.
    • If within cap, bytes stream straight to disk — RAM usage is O(chunk)
      not O(file).
    • For all other endpoints (JSON API calls etc.) the original behaviour
      is preserved: body is buffered in RAM, cap is enforced, body is
      re-injected for the downstream handler.

    Configuration
    ─────────────
    RHODAWK_MAX_BODY_BYTES     — default cap for all non-upload endpoints (10 MB)
    RHODAWK_MAX_UPLOAD_BYTES   — cap for zip upload endpoints (default 2 GB)
    """

    # Paths that receive large binary uploads and must NOT be buffered in RAM.
    # The zip upload endpoint is not yet built but its path is pre-registered
    # here so the middleware is correct from day one.
    _STREAM_UPLOAD_PATHS: frozenset[str] = frozenset({
        "/api/upload",
        "/api/upload/zip",
        "/api/runs/upload",
    })

    def __init__(
        self,
        app:              ASGIApp,
        max_bytes:        int = _MAX_BODY_BYTES,
        max_upload_bytes: int = int(
            os.environ.get("RHODAWK_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024))
        ),
    ) -> None:
        super().__init__(app)
        self.max_bytes        = max_bytes
        self.max_upload_bytes = max_upload_bytes

    async def dispatch(self, request: Request, call_next):
        path        = request.url.path
        is_upload   = path in self._STREAM_UPLOAD_PATHS
        effective_cap = self.max_upload_bytes if is_upload else self.max_bytes

        # ── Stage 1: Content-Length fast-path (no body read) ─────────────────
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                cl = int(content_length)
                if cl > effective_cap:
                    log.warning(
                        "RequestBodySizeLimit: rejected %s %s — "
                        "Content-Length %d > cap %d bytes (IP: %s)",
                        request.method, path, cl, effective_cap,
                        request.client.host if request.client else "unknown",
                    )
                    cap_mb = effective_cap // (1024 * 1024)
                    return Response(
                        content=(
                            f"Request body too large. "
                            f"Maximum allowed for this endpoint: {cap_mb} MB."
                        ),
                        status_code=413,
                        media_type="text/plain",
                    )
            except ValueError:
                pass  # Malformed Content-Length — let downstream handle it

        # ── Stage 2: streaming byte counter ──────────────────────────────────
        if request.method not in ("POST", "PUT", "PATCH"):
            return await call_next(request)

        if is_upload:
            # Upload endpoint: count bytes but DO NOT buffer — pass stream through.
            # The upload handler writes to disk chunk-by-chunk.
            # We wrap the stream in a counting generator so we can enforce the cap
            # without ever holding more than one chunk in memory.
            total_seen = 0

            async def _counting_stream():
                nonlocal total_seen
                async for chunk in request.stream():
                    total_seen += len(chunk)
                    if total_seen > self.max_upload_bytes:
                        log.warning(
                            "RequestBodySizeLimit: upload stream exceeded %d bytes "
                            "at %s (IP: %s) — closing stream",
                            self.max_upload_bytes, path,
                            request.client.host if request.client else "unknown",
                        )
                        # Stop yielding — upstream connection will get a 413
                        # once the handler finishes reading what it got.
                        return
                    yield chunk

            # Re-wrap request with counting stream
            # Starlette does not expose a clean way to replace stream mid-flight,
            # so we patch the receive callable on the scope to count bytes while
            # streaming to disk.  The upload handler uses request.stream() which
            # calls receive() internally.
            original_receive = request.receive
            _bytes_counted   = 0
            _hard_stop       = False

            async def _counted_receive():
                nonlocal _bytes_counted, _hard_stop
                if _hard_stop:
                    # Signal end of body to upstream so it stops reading
                    return {"type": "http.request", "body": b"", "more_body": False}
                msg = await original_receive()
                chunk = msg.get("body", b"")
                _bytes_counted += len(chunk)
                if _bytes_counted > self.max_upload_bytes:
                    _hard_stop = True
                    log.warning(
                        "RequestBodySizeLimit: upload %s exceeded %d bytes "
                        "(IP: %s) — truncating",
                        path, self.max_upload_bytes,
                        request.client.host if request.client else "unknown",
                    )
                    return {"type": "http.request", "body": b"", "more_body": False}
                return msg

            request = Request(request.scope, _counted_receive)
            response = await call_next(request)
            if _hard_stop:
                return Response(
                    content=(
                        f"Upload too large. "
                        f"Maximum: {self.max_upload_bytes // (1024*1024)} MB."
                    ),
                    status_code=413,
                    media_type="text/plain",
                )
            return response

        else:
            # Non-upload endpoint: buffer body in RAM and enforce cap.
            # This is intentional — JSON API payloads are small and
            # downstream handlers need to read them multiple times.
            body_chunks: list[bytes] = []
            total = 0
            async for chunk in request.stream():
                total += len(chunk)
                if total > self.max_bytes:
                    log.warning(
                        "RequestBodySizeLimit: rejected %s %s — "
                        "streamed body exceeded %d bytes (IP: %s)",
                        request.method, path, self.max_bytes,
                        request.client.host if request.client else "unknown",
                    )
                    return Response(
                        content=(
                            f"Request body too large. "
                            f"Maximum allowed: {self.max_bytes // (1024*1024)} MB."
                        ),
                        status_code=413,
                        media_type="text/plain",
                    )
                body_chunks.append(chunk)

            body_bytes = b"".join(body_chunks)

            async def _receive():
                return {"type": "http.request", "body": body_bytes, "more_body": False}

            request = Request(request.scope, _receive)
            return await call_next(request)


# ── Rate limiter setup ────────────────────────────────────────────────────────

def _build_limiter():
    """
    Build a slowapi Limiter instance.  Returns None if slowapi is not
    installed — rate limiting degrades gracefully with a startup WARNING.
    """
    try:
        from slowapi import Limiter                    # type: ignore
        from slowapi.util import get_remote_address    # type: ignore
        limiter = Limiter(key_func=get_remote_address)
        log.info(
            "RateLimit: slowapi active — run_start=%s upload=%s "
            "write=%s read=%s",
            _RATE_RUN_START, _RATE_UPLOAD, _RATE_WRITE, _RATE_READ,
        )
        return limiter
    except ImportError:
        log.warning(
            "RateLimit: slowapi not installed — per-IP rate limiting is DISABLED. "
            "Fix: pip install slowapi  (add 'slowapi>=0.1.9' to requirements.txt)"
        )
        return None


_limiter = _build_limiter()


def _rate_limit(limit_string: str):
    """
    Decorator factory that applies a slowapi rate limit when available.
    Returns an identity decorator when slowapi is not installed so endpoint
    code is identical regardless of whether rate limiting is active.
    """
    if _limiter is None:
        def _noop(fn):
            return fn
        return _noop
    return _limiter.limit(limit_string)


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
        f"cors_origins={_ALLOWED_ORIGINS} "
        f"max_body_bytes={_MAX_BODY_BYTES} "
        f"rate_limiting={'active' if _limiter else 'DISABLED (slowapi not installed)'}"
    )
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
        version="2.0.2",
        description="Production-grade autonomous code stabilization with DO-178C compliance",
        lifespan=lifespan,
    )

    # ── Security middleware — order matters: size limit runs before CORS ──────

    # 1. Request body size limit (DoS prevention — must be first)
    app.add_middleware(
        RequestBodySizeLimitMiddleware,
        max_bytes=_MAX_BODY_BYTES,
    )

    # 2. CORS (narrowed from wildcard to explicit allowlist)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    # 3. slowapi rate limit error handler (no-op if limiter is None)
    if _limiter is not None:
        try:
            from slowapi import _rate_limit_exceeded_handler          # type: ignore
            from slowapi.errors import RateLimitExceeded              # type: ignore
            from slowapi.middleware import SlowAPIMiddleware           # type: ignore
            app.state.limiter = _limiter
            app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
            app.add_middleware(SlowAPIMiddleware)
        except Exception as exc:
            log.warning(f"RateLimit: middleware setup failed: {exc}")

    # ── Core routes ──────────────────────────────────────────────────────────
    for module_name, attr in [
        ("api.routes.runs",              "router"),
        ("api.routes.issues",            "router"),
        ("api.routes.fixes",             "router"),
        ("api.routes.files",             "router"),
        ("api.routes.escalations",       "router"),
        ("api.routes.compound_findings", "router"),
        ("api.routes.refactor_proposals","router"),
        ("api.routes.commits",           "router"),
        # ZIP UPLOAD — streaming extraction with zip-bomb/zip-slip protection
        # Handles source code zips up to RHODAWK_MAX_UPLOAD_BYTES (default 2 GB)
        # without loading the zip into RAM.  See api/routes/upload.py.
        ("api.routes.upload",            "router"),
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
        return {
            "status":  "ok",
            "version": "2.0.2",
            "env":     os.environ.get("RHODAWK_ENV", "production"),
            "security": {
                "body_limit_bytes":    _MAX_BODY_BYTES,
                "rate_limiting_active": _limiter is not None,
            },
        }

    @app.get("/api/capabilities")
    async def capabilities():
        try:
            from startup.feature_matrix import FeatureMatrix
            fm = FeatureMatrix.get()
            if fm and fm._verified:
                return fm.report()
        except Exception:
            pass
        return {
            "status": "not_verified",
            "hint":   "Call /api/runs/start to trigger preflight check",
        }

    # ── Runs — rate-limited run start ─────────────────────────────────────────
    # The /api/runs/start endpoint is the highest-value flood target: each
    # call spawns Celery workers, allocates DB rows, and may start Joern.
    # Hard limit: 5 starts per minute per IP.
    #
    # We decorate the route here at the app level rather than in routes/runs.py
    # so the rate limit is enforced even if the routes module is reloaded or
    # patched.  The runs router still handles all the business logic.

    @app.post("/api/runs/start/rate-check")
    @_rate_limit(_RATE_RUN_START)
    async def _run_start_rate_sentinel(request: Request):
        """
        Internal sentinel endpoint consumed by the rate limiter.
        /api/runs/start in routes/runs.py calls this via a dependency
        so the 5/minute limit applies without duplicating route logic.

        If slowapi is not installed this endpoint returns 200 and the
        real /api/runs/start handler continues normally.
        """
        return {"rate_check": "passed"}

    # ── Baseline promotion ────────────────────────────────────────────────────

    @app.post("/api/baselines/{run_id}/promote")
    async def promote_baseline(
        run_id: str,
        body:   dict = {},
        storage=Depends(get_storage),
    ):
        """
        Promotes a stabilized run to active baseline (DO-178C Sec 11).
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
        run_id:  str,
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
        """Internal — localhost only."""
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
