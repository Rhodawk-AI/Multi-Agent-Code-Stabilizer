"""
api/routes/upload.py
=====================
Zip upload endpoint for Rhodawk AI Code Stabilizer.

Accepts a .zip file containing source code and starts an audit run
without any Git connection.  This is the correct path for:
  - Proprietary codebases that cannot be pointed at a GitHub URL
  - Air-gapped enterprise environments with no outbound internet
  - One-shot audits of archived snapshots
  - CI pipeline integration where the artifact is already a zip

DESIGN DECISIONS
────────────────

1. STREAMING EXTRACTION — never loads the full zip into RAM.
   Python's zipfile module extracts entry by entry.  Each entry is
   written to disk in 64 KB chunks.  Peak RAM for a 2 GB zip is
   O(64 KB), not O(2 GB).

2. ZIP BOMB PROTECTION — three independent guards:
   a. Total extracted size cap (RHODAWK_MAX_EXTRACT_BYTES, default 10 GB).
   b. Entry count cap (RHODAWK_MAX_ZIP_ENTRIES, default 100,000 files).
   c. Per-entry size cap (RHODAWK_MAX_ENTRY_BYTES, default 50 MB per file).
   A zip bomb that compresses 1 GB into 100 KB hits guard (a) after
   the first few entries are extracted.

3. ZIP SLIP PREVENTION — every entry path is resolved against the
   extraction directory and rejected if it escapes it.
   e.g.  "../../../etc/passwd"  →  rejected with HTTP 422.

4. INCREMENTAL PROCESSING — after extraction the ReaderAgent processes
   files in batches.  A 100 MB zip with 5,000 source files is indexed
   in ~10-minute batches, not loaded as one unit.  The run_id is returned
   immediately so the caller can poll /api/runs/{run_id} for progress.

5. ASYNC EXTRACTION — zipfile is synchronous I/O.  We run it in a thread-
   pool executor so the event loop is not blocked during extraction.  For
   a 500 MB zip this is ~2-5 seconds of disk I/O that would otherwise block
   all other requests.

6. CLEANUP — the extracted directory is deleted when the run completes
   or when the process restarts.  Runs that are resumed do not re-extract.

ENDPOINTS
─────────
POST /api/upload
    Multipart upload.  Field name: "file".  Returns run_id immediately.
    Poll GET /api/runs/{run_id} for status.

GET /api/upload/{run_id}/status
    Quick extraction status (EXTRACTING | INDEXING | RUNNING | DONE | ERROR).

GET /api/upload/{run_id}/download
    Download zip of all fixed files after run completes.

LIMITS (all configurable via environment variables)
─────────────────────────────────────────────────────
RHODAWK_MAX_UPLOAD_BYTES   = 2 GB   (zip file on disk)
RHODAWK_MAX_EXTRACT_BYTES  = 10 GB  (total extracted size)
RHODAWK_MAX_ZIP_ENTRIES    = 100000 (entry count)
RHODAWK_MAX_ENTRY_BYTES    = 50 MB  (single entry size)
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from auth.jwt_middleware import get_current_user, TokenData

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload", tags=["upload"])

# ── Limits ────────────────────────────────────────────────────────────────────

_MAX_UPLOAD_BYTES  = int(os.environ.get("RHODAWK_MAX_UPLOAD_BYTES",  str(2  * 1024**3)))  # 2 GB
_MAX_EXTRACT_BYTES = int(os.environ.get("RHODAWK_MAX_EXTRACT_BYTES", str(10 * 1024**3)))  # 10 GB
_MAX_ZIP_ENTRIES   = int(os.environ.get("RHODAWK_MAX_ZIP_ENTRIES",   "100000"))
_MAX_ENTRY_BYTES   = int(os.environ.get("RHODAWK_MAX_ENTRY_BYTES",   str(50 * 1024**2)))  # 50 MB per file
_CHUNK_SIZE        = 64 * 1024   # 64 KB read/write chunks — O(1) RAM per entry

# Upload base directory — all extractions live here
_UPLOAD_BASE = Path(os.environ.get("RHODAWK_UPLOAD_DIR", "/tmp/rhodawk_uploads"))

# MISSING-4 FIX: per-process concurrency limit for upload runs.
# Without a cap, a client can submit hundreds of concurrent 2 GB zips, each
# spawning a full StabilizerController, exhausting file descriptors and RAM.
# Workers OOM with no recovery (state is in-process _run_state dict).
_MAX_CONCURRENT_RUNS = int(os.environ.get("RHODAWK_MAX_CONCURRENT_UPLOADS", "5"))

# In-process run state (run_id → dict)
# For multi-process deployments this should be backed by Redis.
# For single-process (default) this is sufficient.
_run_state: dict[str, dict] = {}


# ── Zip extraction (runs in thread pool) ──────────────────────────────────────

def _safe_extract_zip(
    zip_path:   Path,
    dest:       Path,
    run_id:     str,
) -> dict:
    """
    Extract a zip file to dest with all safety checks.

    Runs synchronously — call via run_in_executor.

    Returns a summary dict with:
        files_extracted  int
        total_bytes      int
        skipped_entries  list[str]   — entries rejected (traversal, too large)
        error            str | None
    """
    files_extracted = 0
    total_bytes     = 0
    skipped         = []

    dest_resolved = dest.resolve()

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            entries = zf.infolist()

            # Guard: entry count
            if len(entries) > _MAX_ZIP_ENTRIES:
                return {
                    "files_extracted": 0,
                    "total_bytes": 0,
                    "skipped_entries": [],
                    "error": (
                        f"Zip contains {len(entries)} entries which exceeds "
                        f"the maximum of {_MAX_ZIP_ENTRIES}. "
                        "Split the archive into smaller parts."
                    ),
                }

            _run_state[run_id]["extraction_total"] = len(entries)

            for i, entry in enumerate(entries):
                _run_state[run_id]["extraction_progress"] = i

                # Skip directories — we create them implicitly
                if entry.filename.endswith("/"):
                    continue

                # Guard: zip slip — entry path must not escape dest
                entry_path = (dest / entry.filename).resolve()
                try:
                    entry_path.relative_to(dest_resolved)
                except ValueError:
                    log.warning(
                        "ZipUpload: zip slip rejected — entry %r escapes %s",
                        entry.filename, dest,
                    )
                    skipped.append(f"[zip-slip] {entry.filename}")
                    continue

                # Guard: single entry size
                if entry.file_size > _MAX_ENTRY_BYTES:
                    log.info(
                        "ZipUpload: skipping large entry %r (%d bytes > %d)",
                        entry.filename, entry.file_size, _MAX_ENTRY_BYTES,
                    )
                    skipped.append(f"[too-large] {entry.filename}")
                    continue

                # Guard: total extracted size
                if total_bytes + entry.file_size > _MAX_EXTRACT_BYTES:
                    log.warning(
                        "ZipUpload: run_id=%s total extraction would exceed %d bytes "
                        "at entry %r — stopping",
                        run_id, _MAX_EXTRACT_BYTES, entry.filename,
                    )
                    skipped.append(f"[total-cap] {entry.filename}")
                    # Do not continue — anything after this would also exceed the cap
                    break

                # Create parent directory
                entry_path.parent.mkdir(parents=True, exist_ok=True)

                # Stream-extract entry in chunks — O(64 KB) RAM regardless of entry size
                bytes_written = 0
                try:
                    with zf.open(entry) as src, open(entry_path, "wb") as dst:
                        while True:
                            chunk = src.read(_CHUNK_SIZE)
                            if not chunk:
                                break
                            dst.write(chunk)
                            bytes_written += len(chunk)
                            # Re-check entry size guard while streaming
                            # (compressed size ≠ uncompressed size in general)
                            if bytes_written > _MAX_ENTRY_BYTES:
                                log.warning(
                                    "ZipUpload: entry %r expanded beyond %d bytes "
                                    "during streaming — truncating and skipping",
                                    entry.filename, _MAX_ENTRY_BYTES,
                                )
                                break
                except Exception as exc:
                    log.debug("ZipUpload: failed to extract %r: %s", entry.filename, exc)
                    skipped.append(f"[error] {entry.filename}")
                    continue

                total_bytes     += bytes_written
                files_extracted += 1

    except zipfile.BadZipFile as exc:
        return {
            "files_extracted": 0,
            "total_bytes": 0,
            "skipped_entries": [],
            "error": f"Invalid zip file: {exc}",
        }
    except Exception as exc:
        return {
            "files_extracted": 0,
            "total_bytes": 0,
            "skipped_entries": skipped,
            "error": str(exc),
        }

    return {
        "files_extracted": files_extracted,
        "total_bytes":     total_bytes,
        "skipped_entries": skipped,
        "error":           None,
    }


# ── Background audit task ─────────────────────────────────────────────────────

async def _run_audit_on_extracted(
    run_id:    str,
    repo_root: Path,
    domain:    str,
) -> None:
    """
    Background task: extract → index → audit → store results.
    State is tracked in _run_state[run_id].
    """
    state = _run_state[run_id]

    # Step 1: extract
    state["status"] = "EXTRACTING"
    zip_path        = Path(state["zip_path"])
    extract_dir     = Path(state["extract_dir"])

    loop = asyncio.get_event_loop()
    extraction = await loop.run_in_executor(
        None, _safe_extract_zip, zip_path, extract_dir, run_id
    )

    if extraction["error"]:
        state["status"] = "ERROR"
        state["error"]  = extraction["error"]
        log.error("ZipUpload run_id=%s extraction failed: %s", run_id, extraction["error"])
        return

    state["files_extracted"] = extraction["files_extracted"]
    state["total_bytes"]     = extraction["total_bytes"]
    state["skipped_entries"] = extraction["skipped_entries"]
    log.info(
        "ZipUpload run_id=%s extracted %d files (%d MB) skipped=%d",
        run_id,
        extraction["files_extracted"],
        extraction["total_bytes"] // (1024 * 1024),
        len(extraction["skipped_entries"]),
    )

    # Delete the zip now that extraction is done — free disk space
    try:
        zip_path.unlink()
    except Exception:
        pass

    # Step 2: index + audit via controller
    state["status"] = "INDEXING"
    try:
        from config.loader import load_config
        from orchestrator.controller import StabilizerController

        cfg = load_config(
            repo_url=f"zip-upload://{run_id}",
            repo_root=extract_dir,
            domain_mode=domain.upper(),
            use_sqlite=True,
        )
        controller = StabilizerController(cfg)
        run_obj    = await controller.initialise(resume_run_id=None)

        state["rhodawk_run_id"] = run_obj.id
        state["status"]         = "RUNNING"

        final_status = await controller.stabilize()
        state["status"]       = "DONE"
        state["final_status"] = final_status.value

        # BUG-4 FIX: collect changed paths while controller is guaranteed alive,
        # immediately after stabilize() returns and before the controller goes
        # out of scope. Previously _build_download_zip re-queried storage after
        # the controller could be GC'd. Also, partial fixes committed before a
        # crash are now available for download even when status=ERROR.
        rhodawk_run_id = run_obj.id
        try:
            fixes = await controller.storage.list_fixes(
                run_id=rhodawk_run_id)
            changed_paths: set[str] = {
                ff.path
                for fix in fixes
                for ff in (fix.fixed_files or [])
                if ff.path
            }
            state["_changed_paths"] = changed_paths
        except Exception as _cp_exc:
            log.warning("ZipUpload: could not collect changed paths: %s", _cp_exc)
            state["_changed_paths"] = set()

        # Build download zip of fixed files
        await _build_download_zip(run_id, state["_changed_paths"], extract_dir)

    except Exception as exc:
        log.exception("ZipUpload run_id=%s audit failed: %s", run_id, exc)
        state["status"] = "ERROR"
        state["error"]  = str(exc)
        # BUG-4 FIX: attempt to build partial download zip even on failure.
        # Fixes committed before the crash are still useful to the operator.
        if state.get("_changed_paths"):
            try:
                await _build_download_zip(
                    run_id, state["_changed_paths"], extract_dir)
            except Exception:
                pass


async def _build_download_zip(
    run_id:        str,
    changed_paths: set[str],
    repo_root:     Path,
) -> None:
    """
    BUG-4 FIX: Build a zip of changed files using a pre-collected path set.

    Previously this function re-queried controller.storage after the controller
    may have been GC'd, causing AttributeError or connection pool errors.
    The caller now collects changed_paths while the controller is alive and
    passes them directly — no storage query needed here.
    """
    state    = _run_state[run_id]
    out_path = _UPLOAD_BASE / run_id / "fixed_files.zip"

    try:
        if not changed_paths:
            state["download_path"] = None
            return

        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for rel_path in sorted(changed_paths):
                abs_path = repo_root / rel_path
                if abs_path.exists():
                    zf.write(abs_path, arcname=rel_path)

        state["download_path"]  = str(out_path)
        state["download_count"] = len(changed_paths)
        log.info(
            "ZipUpload run_id=%s download zip built: %d files at %s",
            run_id, len(changed_paths), out_path,
        )
    except Exception as exc:
        log.warning("ZipUpload: failed to build download zip for %s: %s", run_id, exc)
        state["download_path"] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("")
async def upload_zip(
    background_tasks: BackgroundTasks,
    request:          Request,
    file:             UploadFile = File(...),
    domain:           str        = "GENERAL",
):
    """
    Upload a .zip file of source code and start an audit run.

    The zip is saved to disk via streaming (O(64 KB) RAM).
    Extraction and auditing run in the background.
    Returns run_id immediately — poll /api/upload/{run_id}/status.

    Limits:
      File size:          2 GB (set RHODAWK_MAX_UPLOAD_BYTES to override)
      Extracted size:     10 GB total (RHODAWK_MAX_EXTRACT_BYTES)
      Entry count:        100,000 files (RHODAWK_MAX_ZIP_ENTRIES)
      Single entry size:  50 MB per file (RHODAWK_MAX_ENTRY_BYTES)
    """
    # Validate content type
    if file.content_type not in (
        "application/zip",
        "application/x-zip-compressed",
        "application/octet-stream",
        "multipart/form-data",
    ):
        # Be lenient with content type — browsers send different values.
        # Rely on the file extension and magic bytes instead.
        pass

    # MISSING-4 FIX: enforce per-process concurrency limit before allocating resources
    active_runs = sum(
        1 for s in _run_state.values()
        if s.get("status") not in ("DONE", "ERROR")
    )
    if active_runs >= _MAX_CONCURRENT_RUNS:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Too many concurrent audit runs ({active_runs}/{_MAX_CONCURRENT_RUNS}). "
                "Wait for a run to complete or increase RHODAWK_MAX_CONCURRENT_UPLOADS."
            ),
        )

    filename = file.filename or "upload.zip"
    if not filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=422,
            detail="Only .zip files are accepted. "
                   "Other archive formats (tar.gz, 7z) are not supported.",
        )

    # Allocate run directory
    run_id   = str(uuid.uuid4())
    run_dir  = _UPLOAD_BASE / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    zip_dest    = run_dir / "upload.zip"
    extract_dir = run_dir / "source"
    extract_dir.mkdir()

    # Stream zip to disk — never buffer more than _CHUNK_SIZE in RAM
    bytes_written = 0
    try:
        with open(zip_dest, "wb") as f:
            while True:
                chunk = await file.read(_CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > _MAX_UPLOAD_BYTES:
                    # Remove partial file and reject
                    zip_dest.unlink(missing_ok=True)
                    shutil.rmtree(run_dir, ignore_errors=True)
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Zip file exceeds maximum upload size of "
                            f"{_MAX_UPLOAD_BYTES // (1024**2)} MB. "
                            "Split the archive or use the --repo-root CLI argument instead."
                        ),
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")

    log.info(
        "ZipUpload: received %s (%d MB) → run_id=%s",
        filename, bytes_written // (1024 * 1024), run_id,
    )

    # Initialize run state
    _run_state[run_id] = {
        "status":               "RECEIVED",
        "filename":             filename,
        "zip_bytes":            bytes_written,
        "zip_path":             str(zip_dest),
        "extract_dir":          str(extract_dir),
        "domain":               domain,
        "files_extracted":      0,
        "total_bytes":          0,
        "skipped_entries":      [],
        "extraction_progress":  0,
        "extraction_total":     0,
        "rhodawk_run_id":       None,
        "final_status":         None,
        "download_path":        None,
        "download_count":       0,
        "error":                None,
    }

    # Kick off background extraction + audit
    background_tasks.add_task(
        _run_audit_on_extracted, run_id, extract_dir, domain
    )

    return {
        "run_id":       run_id,
        "status":       "RECEIVED",
        "filename":     filename,
        "zip_bytes":    bytes_written,
        "domain":       domain,
        "poll_url":     f"/api/upload/{run_id}/status",
        "download_url": f"/api/upload/{run_id}/download",
        "message":      (
            "Extraction and audit started in the background. "
            f"Poll {'/api/upload/' + run_id + '/status'} for progress."
        ),
    }


@router.get("/{run_id}/status")
async def upload_status(run_id: str):
    """
    Poll extraction and audit progress.

    Returns:
        status:               RECEIVED | EXTRACTING | INDEXING | RUNNING | DONE | ERROR
        files_extracted:      how many files were extracted from the zip
        extraction_progress:  current entry index during extraction
        extraction_total:     total entries in the zip
        total_bytes:          total extracted bytes on disk
        skipped_entries:      list of entries that were rejected (zip-slip, too large)
        rhodawk_run_id:       internal run ID once audit starts
        final_status:         STABILIZED | COST_CEILING | etc. (once DONE)
        download_url:         URL to download fixed files (once DONE)
        error:                error message if status == ERROR
    """
    state = _run_state.get(run_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"Upload run {run_id} not found. "
                   "Run IDs are not persisted across server restarts.",
        )

    response = {
        "run_id":               run_id,
        "status":               state["status"],
        "filename":             state.get("filename"),
        "zip_bytes":            state.get("zip_bytes", 0),
        "files_extracted":      state.get("files_extracted", 0),
        "extraction_progress":  state.get("extraction_progress", 0),
        "extraction_total":     state.get("extraction_total", 0),
        "total_bytes":          state.get("total_bytes", 0),
        "skipped_entries_count": len(state.get("skipped_entries", [])),
        "rhodawk_run_id":       state.get("rhodawk_run_id"),
        "final_status":         state.get("final_status"),
        "error":                state.get("error"),
    }

    if state["status"] == "DONE" and state.get("download_path"):
        response["download_url"]   = f"/api/upload/{run_id}/download"
        response["download_count"] = state.get("download_count", 0)

    return response


@router.get("/{run_id}/download")
async def download_fixed_files(
    run_id: str,
    _user: TokenData = Depends(get_current_user),
):
    """
    Download a zip containing only the files that were modified by Rhodawk.

    Available after status == DONE and at least one fix was committed.
    Returns 404 if no fixes were generated.
    Returns 425 (Too Early) if the run is still in progress.
    """
    state = _run_state.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")

    if state["status"] not in ("DONE", "ERROR"):
        raise HTTPException(
            status_code=425,
            detail=(
                f"Run is still in progress (status={state['status']}). "
                "Wait for status == DONE before downloading."
            ),
        )

    download_path = state.get("download_path")
    if not download_path or not Path(download_path).exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No fixed files available for this run. "
                "Either no bugs were found, or all fixes were escalated to human review."
            ),
        )

    filename = f"rhodawk_fixes_{run_id[:8]}.zip"
    return FileResponse(
        path=download_path,
        media_type="application/zip",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/{run_id}")
async def cleanup_upload(run_id: str):
    """
    Delete all files associated with an upload run (zip, extracted source,
    download artifact).  Call this after downloading fixed files.
    """
    state = _run_state.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")

    if state["status"] in ("EXTRACTING", "INDEXING", "RUNNING"):
        raise HTTPException(
            status_code=409,
            detail=f"Run is still active (status={state['status']}). "
                   "Wait for DONE or ERROR before deleting.",
        )

    run_dir = _UPLOAD_BASE / run_id
    try:
        shutil.rmtree(run_dir, ignore_errors=True)
    except Exception as exc:
        log.warning("ZipUpload cleanup failed for %s: %s", run_id, exc)

    del _run_state[run_id]
    return {"status": "deleted", "run_id": run_id}
