#!/usr/bin/env python3
"""
run.py — Rhodawk AI Code Stabilizer entry point.

Usage:
    python run.py --repo-url <URL> [--config <path>] [--domain military]
    python run.py --help
"""
from __future__ import annotations
import argparse, asyncio, logging, os, sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("rhodawk")


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
    p.add_argument("--max-cycles", default=50,   type=int, help="Maximum cycles")
    p.add_argument("--resume",     default=None, help="Resume from run ID")
    p.add_argument("--sqlite",     action="store_true",
                   help="Use SQLite storage (dev mode only)")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(_parse())))
