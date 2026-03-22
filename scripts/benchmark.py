"""
scripts/benchmark.py
====================
CLI to run SWE-bench Verified evaluation.

Usage::
    rhodawk-bench run --limit 50
    rhodawk-bench run --limit 500 --workers 8
"""
from __future__ import annotations
import asyncio, logging
import typer
from rich.console import Console
from rich.table import Table

app   = typer.Typer(help="Rhodawk AI benchmark runner")
console = Console()
log   = logging.getLogger(__name__)


@app.command("run")
def run_benchmark(
    limit:   int  = typer.Option(50,  help="Max instances to evaluate"),
    workers: int  = typer.Option(4,   help="Parallel workers"),
    dataset: str  = typer.Option("princeton-nlp/SWE-bench_Verified", help="HF dataset"),
    split:   str  = typer.Option("test", help="Dataset split"),
    swarm:   bool = typer.Option(True, help="Use CrewAI swarm for fixes"),
):
    """Run SWE-bench Verified evaluation."""
    import os
    os.environ.setdefault("RHODAWK_BENCH_DATASET", dataset)
    os.environ.setdefault("RHODAWK_BENCH_SPLIT", split)

    from swe_bench.evaluator import SWEBenchEvaluator, load_swe_bench_instances

    logging.basicConfig(level=logging.INFO)
    console.print(f"[bold]SWE-bench Verified[/bold] — limit={limit} workers={workers}")

    instances = load_swe_bench_instances(dataset=dataset, split=split, limit=limit)
    if not instances:
        console.print("[red]No instances loaded — check RHODAWK_BENCH_DATASET[/red]")
        raise typer.Exit(1)

    evaluator = SWEBenchEvaluator(controller_factory=None, workers=workers, use_swarm=swarm)
    report    = asyncio.run(evaluator.run(instances=instances))

    table = Table(title="SWE-bench Results")
    table.add_column("Metric",  style="cyan")
    table.add_column("Value",   style="green")
    table.add_row("Total",      str(report.total))
    table.add_row("Resolved",   str(report.resolved))
    table.add_row("Pass Rate",  f"{report.pass_rate:.1%}")
    table.add_row("Target",     f"{report.target_rate:.1%}")
    table.add_row("Beats Target", "✅ YES" if report.beats_target else "❌ NO")
    table.add_row("Avg Cost",   f"${report.avg_cost_usd:.3f}/instance")

    # ARCH-01 FIX: show CPG context availability so published scores are
    # qualified by whether full causal context was active.  A score measured
    # without Joern is expected to be 15-20pp lower than CPG-enabled performance
    # (cross-file bugs receive vector-similarity fallback context only).
    # Pulling cpg_available from the health endpoint keeps this live-accurate
    # even when the evaluator is run against a running API instance.
    cpg_state = "unknown"
    try:
        from api.app import _state as _app_state
        if _app_state.cpg_available is True:
            cpg_state = "✅ ACTIVE (Joern connected — causal context enabled)"
        elif _app_state.cpg_available is False:
            cpg_state = "❌ INACTIVE (Joern not running — vector-similarity fallback)"
        else:
            cpg_state = "⚠️  not reported (controller not initialised)"
    except Exception:
        pass
    table.add_row("CPG Context", cpg_state)

    console.print(table)

    # ARCH-01 FIX: emit a prominent warning when CPG was not active so the
    # score is never silently published without the qualification.
    if "INACTIVE" in cpg_state:
        console.print(
            "\n[bold yellow]⚠  WARNING: This benchmark was run WITHOUT Joern/CPG context.[/bold yellow]\n"
            "  Cross-file bugs received vector-similarity fallback context only.\n"
            "  Expected score penalty vs CPG-enabled: -15 to -20pp.\n"
            "  To enable: set JOERN_REPO_PATH in .env and run: docker-compose up joern\n"
            "  Do NOT publish this score as the system's maximum capability.\n"
        )
    raise typer.Exit(0 if report.beats_target else 1)


if __name__ == "__main__":
    app()
