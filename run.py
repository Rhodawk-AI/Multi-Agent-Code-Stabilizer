#!/usr/bin/env python3
"""
run.py — Rhodawk AI Code Stabilizer entry point.

Usage:
    python run.py --repo-url <URL> [--config <path>] [--domain military]
    python run.py --help
"""
from __future__ import annotations
import argparse, asyncio, logging, os, signal, sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("rhodawk")

# ── ADD-3 FIX: Graceful SIGTERM handler ───────────────────────────────────────
# docker-compose restart: unless-stopped sends SIGTERM with a 10-second grace
# window before SIGKILL. Without a handler, a fix mid-write receives SIGTERM,
# leaves a partially written source file on disk, never sets committed_at, and
# causes an infinite corrupt-patch loop on restart.
#
# _SHUTDOWN_REQUESTED is a module-level flag checked between pipeline phases
# inside stabilize(). Setting it on SIGTERM allows the current phase to finish
# cleanly before the process exits, preventing mid-write corruption.
_SHUTDOWN_REQUESTED: bool = False
# NOTE: this flag is per-process. When running via uvicorn --workers N,
# each worker has its own copy. Use docker stop (SIGTERM to PID 1) or
# uvicorn --graceful-timeout to coordinate shutdown across workers.


def _graceful_sigterm_handler(signum: int, frame: object) -> None:
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    log.info(
        "SIGTERM received — completing current pipeline phase before exit. "
        "Send SIGKILL if immediate shutdown is required."
    )


signal.signal(signal.SIGTERM, _graceful_sigterm_handler)


async def main(args: argparse.Namespace) -> int:
    # Startup feature matrix (before any imports that use features)
    from startup.feature_matrix import verify_startup, ConfigurationError
    domain = args.domain.upper() if args.domain else "GENERAL"
    try:
        fm = verify_startup(
            domain_mode=domain,
            strict=(domain in {"MILITARY", "AEROSPACE", "NUCLEAR"}),
        )
    except ConfigurationError as exc:
        log.critical(f"Startup failed:\n{exc}")
        return 1

    from config.loader import load_config
    from orchestrator.controller import StabilizerController

    try:
        cfg = load_config(
            config_path=args.config,
            repo_url=args.repo_url or os.getcwd(),
            repo_root=Path(args.repo_root or "."),
            domain_mode=domain,
            max_cycles=int(args.max_cycles),
            use_sqlite=args.sqlite,
        )
    except Exception as exc:
        log.critical(f"Config load failed: {exc}")
        return 1

    controller = StabilizerController(cfg)

    # Expose the shutdown flag to the controller so stabilize() can poll it
    # between phases and exit cleanly without aborting mid-write.
    controller._shutdown_requested = lambda: _SHUTDOWN_REQUESTED

    try:
        run = await controller.initialise(resume_run_id=args.resume)
        log.info(f"Run {run.id[:8]} started [domain={domain}]")

        status = await controller.stabilize()
        log.info(f"Run {run.id[:8]} finished: {status.value}")

        if status.value in ("STABILIZED", "BASELINE_PENDING"):
            log.info(
                "✅ Stabilization complete. "
                f"Promote to baseline: POST /api/baselines with run_id={run.id}"
            )
            return 0
        # ADD-3: distinguish SIGTERM-initiated clean exit from a real failure
        if _SHUTDOWN_REQUESTED:
            log.info("Run terminated cleanly on SIGTERM after phase boundary.")
            return 0
        return 1

    except KeyboardInterrupt:
        log.info("Interrupted by user")
        return 130
    except Exception as exc:
        log.exception(f"Fatal error: {exc}")
        return 1


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rhodawk AI Code Stabilizer — production-grade autonomous code auditing"
    )
    p.add_argument("--repo-url",   default="",   help="Repository URL or local path")
    p.add_argument("--repo-root",  default=".",  help="Local repository root (default: .)")
    p.add_argument("--config",     default=None, help="YAML/TOML config file path")
    p.add_argument("--domain",     default="GENERAL",
                   choices=["GENERAL","MILITARY","AEROSPACE","MEDICAL","FINANCE",
                             "EMBEDDED","NUCLEAR"],
                   help="Domain mode")
    # ADD-2 FIX: Default raised from 50 to 200. Repos with interdependent issues
    # routinely require >50 cycles to converge. At 50 the run exits with
    # MAX_CYCLES_REACHED, leaving in-progress fixes uncommitted. 200 accommodates
    # the Linux kernel and other large multi-subsystem codebases without artificially
    # truncating a run. Operators who need a hard cap can set --max-cycles explicitly.
    p.add_argument("--max-cycles", default=200,  type=int,
                   help="Maximum stabilization cycles (default: 200)")
    p.add_argument("--resume",     default=None, help="Resume from run ID")
    p.add_argument("--sqlite",     action="store_true",
                   help="Use SQLite storage (dev mode only)")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(_parse())))
