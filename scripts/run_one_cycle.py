"""
scripts/run_one_cycle.py
========================
Smoke-runs a single stabilisation cycle for CI / local validation.

Usage
-----
    python scripts/run_one_cycle.py --repo-root /path/to/repo [--sqlite]

Exit codes
----------
    0   cycle completed (STABILIZED or BASELINE_PENDING)
    1   cycle failed or regressed
    2   configuration error (missing capability)
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")


@app.command()
def main(
    repo_root: Path = typer.Option(..., help="Absolute path to the repository root"),
    repo_url:  str  = typer.Option("file://local", help="Repository URL (for PR creation)"),
    sqlite:    bool = typer.Option(False, "--sqlite", help="Use SQLite (dev only)"),
    max_cycles: int = typer.Option(1, help="Number of stabilisation cycles to run"),
    domain:    str  = typer.Option("general", help="Domain mode: general|military|aerospace|medical"),
    dry_run:   bool = typer.Option(False, "--dry-run", help="Read + audit only, no fixes or commits"),
) -> None:
    """Single-cycle smoke runner for Rhodawk AI Code Stabilizer."""
    from orchestrator.controller import StabilizerConfig, StabilizerController
    from brain.schemas import AutonomyLevel, DomainMode

    domain_map = {
        "general":   DomainMode.GENERAL,
        "military":  DomainMode.MILITARY,
        "aerospace": DomainMode.AEROSPACE,
        "medical":   DomainMode.MEDICAL,
        "nuclear":   DomainMode.NUCLEAR,
    }
    domain_mode = domain_map.get(domain.lower(), DomainMode.GENERAL)

    cfg = StabilizerConfig(
        repo_url=repo_url,
        repo_root=repo_root,
        use_sqlite=sqlite,
        max_cycles=max_cycles,
        auto_commit=not dry_run,
        autonomy_level=AutonomyLevel.READ_ONLY if dry_run else AutonomyLevel.AUTO_FIX,
        domain_mode=domain_mode,
    )
    ctrl = StabilizerController(cfg)

    async def _run() -> int:
        from brain.schemas import RunStatus
        try:
            await ctrl.initialise()
            status = await ctrl.stabilize()
            if status in (RunStatus.STABILIZED, RunStatus.BASELINE_PENDING):
                typer.echo(f"✅  {status.value}")
                return 0
            typer.echo(f"❌  {status.value}", err=True)
            return 1
        except Exception as exc:
            typer.echo(f"💥  {exc}", err=True)
            return 2

    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    app()
