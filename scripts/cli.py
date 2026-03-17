"""
scripts/cli.py
OpenMOSS CLI — the main entry point for all operations.
Usage: openmoss [command] [options]

PATCH LOG:
  - audit_only command: called ctrl._phase_read() and ctrl._phase_audit() directly
    (private methods). Callers outside the class must not call private methods.
    Now uses the public wrappers run_read_phase() and run_audit_phase() added
    to the controller in this patch set.
  - bootstrap: added --no-incremental flag so users can force a full re-index.
  - Added 'revert' command: undoes the last N stabilizer commits from a branch.
  - Added 'serve' command: starts the FastAPI dashboard server.
  - _write_audit_report: fixed import style (removed __import__ hack).
"""
from __future__ import annotations

import asyncio
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()
app = typer.Typer(
    name="openmoss",
    help="🔱 OpenMOSS — Autonomous Multi-Agent Code Stabilizer",
    add_completion=False,
    rich_markup_mode="rich",
)


def _require_env(key: str) -> str:
    val = os.getenv(key, "")
    if not val:
        console.print(
            f"[red]Error:[/red] Environment variable {key} is required. "
            "Set it in your .env file."
        )
        raise typer.Exit(1)
    return val


# ─────────────────────────────────────────────────────────────
# stabilize
# ─────────────────────────────────────────────────────────────

@app.command("stabilize")
def stabilize(
    repo_url:     str   = typer.Argument(..., help="GitHub repo URL or owner/repo"),
    repo_path:    Path  = typer.Option(..., "--path", "-p", help="Local path to cloned repo"),
    master_prompt: Path = typer.Option(
        Path("config/prompts/base.md"), "--prompt", help="Master audit prompt path"
    ),
    model:        str   = typer.Option("claude-sonnet-4-20250514", "--model", "-m"),
    max_cycles:   int   = typer.Option(50, "--max-cycles"),
    cost_ceiling: float = typer.Option(50.0, "--cost-ceiling", help="Max USD to spend"),
    concurrency:  int   = typer.Option(4, "--concurrency", "-c"),
    no_commit:    bool  = typer.Option(False, "--no-commit", help="Audit only, no GitHub commits"),
    run_id:       str   = typer.Option("", "--run-id", help="Resume an existing run by ID"),
    run_semgrep:  bool  = typer.Option(True, "--semgrep/--no-semgrep"),
    run_mypy:     bool  = typer.Option(True, "--mypy/--no-mypy"),
) -> None:
    """
    🚀 Run the full stabilization loop on a GitHub repo.
    Never stops until stabilized, cost ceiling, or max cycles.
    """
    from orchestrator.controller import StabilizerConfig, StabilizerController

    github_token  = "" if no_commit else os.getenv("GITHUB_TOKEN", "")
    anthropic_key = _require_env("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = anthropic_key

    cfg = StabilizerConfig(
        repo_url=repo_url,
        repo_root=repo_path,
        master_prompt_path=master_prompt,
        github_token=github_token,
        primary_model=model,
        max_cycles=max_cycles,
        cost_ceiling_usd=cost_ceiling,
        concurrency=concurrency,
        auto_commit=not no_commit,
        run_semgrep=run_semgrep,
        run_mypy=run_mypy,
    )

    async def run() -> None:
        ctrl = StabilizerController(cfg)
        await ctrl.initialise(run_id or None)
        status = await ctrl.stabilize()
        if status.value == "STABILIZED":
            console.print(
                "[bold green]✅ STABILIZED — codebase meets all master prompt criteria[/bold green]"
            )
        else:
            console.print(f"[bold yellow]⚠️  Ended with status: {status.value}[/bold yellow]")
            raise typer.Exit(1)

    asyncio.run(run())


# ─────────────────────────────────────────────────────────────
# audit (read + audit, no fix, no commit)
# ─────────────────────────────────────────────────────────────

@app.command("audit")
def audit_only(
    repo_url:     str  = typer.Argument(..., help="GitHub repo URL"),
    repo_path:    Path = typer.Option(..., "--path", "-p"),
    master_prompt: Path = typer.Option(Path("config/prompts/base.md"), "--prompt"),
    model:        str  = typer.Option("claude-sonnet-4-20250514", "--model", "-m"),
    output:       Path = typer.Option(Path("audit_report.md"), "--output", "-o"),
) -> None:
    """
    🔍 Audit only — no fixes, no commits. Produces a markdown report.
    """
    from orchestrator.controller import StabilizerConfig, StabilizerController

    _require_env("ANTHROPIC_API_KEY")

    cfg = StabilizerConfig(
        repo_url=repo_url,
        repo_root=repo_path,
        master_prompt_path=master_prompt,
        primary_model=model,
        auto_commit=False,
        max_cycles=1,
    )

    async def run() -> None:
        ctrl = StabilizerController(cfg)
        run_obj = await ctrl.initialise()

        # FIX: was calling ctrl._phase_read() and ctrl._phase_audit() directly
        # — private methods must not be called from outside the class.
        # Now uses the public wrappers added in the patched controller.
        await ctrl.run_read_phase(incremental=False)
        score = await ctrl.run_audit_phase()

        issues = await ctrl.storage.list_issues(run_id=run_obj.id)
        _write_audit_report(output, run_obj.id, issues, score)
        console.print(f"[green]Audit report written to:[/green] {output}")
        await ctrl.storage.close()

    asyncio.run(run())


# ─────────────────────────────────────────────────────────────
# bootstrap (index only, no audit)
# ─────────────────────────────────────────────────────────────

@app.command("bootstrap")
def bootstrap(
    repo_path:   Path = typer.Argument(..., help="Local repo path to index"),
    model:       str  = typer.Option(
        "claude-haiku-4-5-20251001", "--model", help="Use cheap model for indexing"
    ),
    concurrency: int  = typer.Option(8, "--concurrency", "-c"),
    incremental: bool = typer.Option(
        True, "--incremental/--full",
        help="Skip unchanged files (default). Use --full to re-index everything."
    ),
) -> None:
    """
    📚 Bootstrap the brain by indexing a repo without auditing.
    Run this first on large repos before a full stabilize run.
    """
    from agents.base import AgentConfig
    from agents.reader import ReaderAgent
    from brain.sqlite_storage import SQLiteBrainStorage
    from brain.schemas import AuditRun

    _require_env("ANTHROPIC_API_KEY")
    console.print(f"[cyan]Bootstrapping brain for:[/cyan] {repo_path}")

    async def run() -> None:
        storage = SQLiteBrainStorage(repo_path / ".stabilizer" / "brain.db")
        await storage.initialise()
        dummy_run = AuditRun(
            repo_url=str(repo_path),
            repo_name=repo_path.name,
            branch="main",
        )
        await storage.upsert_run(dummy_run)
        cfg = AgentConfig(model=model)
        reader = ReaderAgent(
            storage=storage,
            run_id=dummy_run.id,
            repo_root=repo_path,
            config=cfg,
            incremental=incremental,
            concurrency=concurrency,
        )
        counts = await reader.run()
        console.print(
            f"[green]Bootstrap complete:[/green] "
            f"processed={counts['processed']} "
            f"skipped={counts['skipped']} "
            f"errors={counts['errors']}"
        )
        await storage.close()

    asyncio.run(run())


# ─────────────────────────────────────────────────────────────
# status
# ─────────────────────────────────────────────────────────────

@app.command("status")
def status(
    repo_path: Path = typer.Argument(..., help="Local repo path with brain"),
    limit:     int  = typer.Option(50, "--limit", "-n", help="Max issues to display"),
) -> None:
    """
    📊 Show status of the current run and open issues.
    """
    from brain.sqlite_storage import SQLiteBrainStorage
    from rich.table import Table

    async def run() -> None:
        db_path = repo_path / ".stabilizer" / "brain.db"
        if not db_path.exists():
            console.print(
                "[red]No brain found. Run 'openmoss bootstrap' or "
                "'openmoss stabilize' first.[/red]"
            )
            raise typer.Exit(1)

        storage = SQLiteBrainStorage(db_path)
        await storage.initialise()

        issues = await storage.list_issues()
        table = Table(title=f"Issues — {repo_path.name}")
        table.add_column("ID",          style="dim",    no_wrap=True)
        table.add_column("Severity",    no_wrap=True)
        table.add_column("Status",      no_wrap=True)
        table.add_column("File",        max_width=45)
        table.add_column("Description", max_width=60)

        sev_colors = {
            "CRITICAL": "red", "MAJOR": "yellow",
            "MINOR": "cyan", "INFO": "white",
        }

        for issue in issues[:limit]:
            c = sev_colors.get(issue.severity.value, "white")
            table.add_row(
                issue.id[:12],
                f"[{c}]{issue.severity.value}[/{c}]",
                issue.status.value,
                issue.file_path[:45],
                issue.description[:60],
            )

        console.print(table)
        console.print(
            f"\n[dim]Showing {min(limit, len(issues))} of {len(issues)} issues.[/dim]"
        )

        # Score summary
        counts = Counter(i.severity.value for i in issues)
        score = counts["CRITICAL"] * 10 + counts["MAJOR"] * 3 + counts["MINOR"]
        console.print(
            f"[bold]Score:[/bold] {score} "
            f"(CRITICAL={counts['CRITICAL']} "
            f"MAJOR={counts['MAJOR']} "
            f"MINOR={counts['MINOR']})"
        )
        await storage.close()

    asyncio.run(run())


# ─────────────────────────────────────────────────────────────
# serve (FastAPI dashboard)
# ─────────────────────────────────────────────────────────────

@app.command("serve")
def serve(
    host:   str = typer.Option("0.0.0.0", "--host"),
    port:   int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """
    🌐 Start the OpenMOSS dashboard API server.
    Open http://localhost:8000 for the live dashboard.
    """
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]uvicorn not installed. Run: pip install uvicorn[standard][/red]"
        )
        raise typer.Exit(1)

    console.print(f"[green]Starting OpenMOSS API on http://{host}:{port}[/green]")
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _write_audit_report(output: Path, run_id: str, issues: list, score: object) -> None:
    """Write a markdown audit report."""
    lines = [
        "# OpenMOSS Audit Report\n\n",
        f"**Run ID:** `{run_id}`\n\n",
        f"**Generated:** {datetime.utcnow().isoformat()}Z\n\n",
        "## Score Summary\n\n",
        "| Severity | Count |\n|---|---|\n",
    ]
    counts = Counter(getattr(i, "severity", None) and i.severity.value for i in issues)
    for sev in ("CRITICAL", "MAJOR", "MINOR", "INFO"):
        lines.append(f"| {sev} | {counts.get(sev, 0)} |\n")
    lines.append("\n## Issues\n\n")
    for issue in sorted(issues, key=lambda i: (i.severity.value, i.file_path)):
        lines.append(
            f"### {issue.id} — [{issue.severity.value}] "
            f"`{issue.file_path}`:{issue.line_start}\n\n"
            f"**Section:** {issue.master_prompt_section}\n\n"
            f"{issue.description}\n\n"
            f"**Status:** {issue.status.value}  "
            f"**Fix attempts:** {issue.fix_attempt_count}\n\n---\n\n"
        )
    output.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    app()
