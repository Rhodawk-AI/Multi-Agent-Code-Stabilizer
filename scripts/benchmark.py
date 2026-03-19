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
    console.print(table)
    raise typer.Exit(0 if report.beats_target else 1)


if __name__ == "__main__":
    app()
