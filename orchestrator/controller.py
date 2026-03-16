"""
orchestrator/controller.py
The core stabilization loop. This is the heart of OpenMOSS.
State machine: READ → AUDIT → FIX → REVIEW → COMMIT → RE-AUDIT → repeat.
Never stops until stabilized, cost ceiling, or manual halt.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from agents.auditor import AuditorAgent
from agents.base import AgentConfig
from agents.fixer import FixerAgent
from agents.reader import ReaderAgent
from agents.reviewer import ReviewerAgent
from brain.schemas import (
    AuditRun,
    AuditScore,
    ExecutorType,
    FixAttempt,
    IssueStatus,
    RunStatus,
    Severity,
)
from brain.sqlite_storage import SQLiteBrainStorage
from brain.storage import BrainStorage
from github_integration.pr_manager import PRManager
from orchestrator.convergence import ConvergenceDetector

log = logging.getLogger(__name__)
console = Console()


class StabilizerConfig:
    def __init__(
        self,
        repo_url:           str,
        repo_root:          Path,
        master_prompt_path: Path,
        github_token:       str = "",
        primary_model:      str = "claude-sonnet-4-20250514",
        fallback_models:    list[str] | None = None,
        max_cycles:         int = 50,
        cost_ceiling_usd:   float = 50.0,
        concurrency:        int = 4,
        auto_commit:        bool = True,
        branch_prefix:      str = "stabilizer",
        incremental:        bool = True,
    ) -> None:
        self.repo_url           = repo_url
        self.repo_root          = repo_root
        self.master_prompt_path = master_prompt_path
        self.github_token       = github_token
        self.primary_model      = primary_model
        self.fallback_models    = fallback_models or ["gpt-4o-mini"]
        self.max_cycles         = max_cycles
        self.cost_ceiling_usd   = cost_ceiling_usd
        self.concurrency        = concurrency
        self.auto_commit        = auto_commit
        self.branch_prefix      = branch_prefix
        self.incremental        = incremental


class StabilizerController:
    """
    Orchestrates the full READ → AUDIT → FIX → REVIEW → COMMIT → REPEAT loop.
    This loop never stops until:
      - Zero CRITICAL + MAJOR issues (STABILIZED)
      - Cost ceiling exceeded (HALTED)
      - Max cycles reached (HALTED)
      - All remaining issues escalated (ESCALATED)
    """

    def __init__(self, cfg: StabilizerConfig) -> None:
        self.cfg        = cfg
        self.storage: BrainStorage = SQLiteBrainStorage(
            cfg.repo_root / ".stabilizer" / "brain.db"
        )
        self.convergence = ConvergenceDetector(stall_threshold=2, max_cycles=cfg.max_cycles)
        self.pr_manager: PRManager | None = None
        self.run: AuditRun | None = None
        self._agent_cfg = AgentConfig(
            model=cfg.primary_model,
            fallback_models=cfg.fallback_models,
            cost_ceiling_usd=cfg.cost_ceiling_usd,
        )

    async def initialise(self, run_id: str | None = None) -> AuditRun:
        """Set up storage, run record, GitHub integration."""
        await self.storage.initialise()

        repo_name = Path(self.cfg.repo_url).stem if "/" not in self.cfg.repo_url else \
            self.cfg.repo_url.rstrip("/").split("/")[-1]

        self.run = AuditRun(
            repo_url=self.cfg.repo_url,
            repo_name=repo_name,
            branch="main",
            master_prompt_path=str(self.cfg.master_prompt_path),
            max_cycles=self.cfg.max_cycles,
        )
        if run_id:
            existing = await self.storage.get_run(run_id)
            if existing:
                self.run = existing
                log.info(f"Resuming existing run {run_id}")
            else:
                self.run.id = run_id

        await self.storage.upsert_run(self.run)

        if self.cfg.github_token:
            self.pr_manager = PRManager(
                token=self.cfg.github_token,
                repo_url=self.cfg.repo_url,
                branch_prefix=self.cfg.branch_prefix,
            )

        return self.run

    # ─────────────────────────────────────────────────────────
    # Main stabilization loop — never stops until done
    # ─────────────────────────────────────────────────────────

    async def stabilize(self) -> RunStatus:
        """
        The primary loop. Runs until stabilized or terminated.
        Returns final run status.
        """
        assert self.run is not None, "Call initialise() first"

        console.rule(f"[bold blue]OpenMOSS Stabilizer — {self.run.repo_name}")
        console.print(f"Run ID: {self.run.id}")
        console.print(f"Model:  {self.cfg.primary_model}")
        console.print(f"Max cycles: {self.cfg.max_cycles} | Cost ceiling: ${self.cfg.cost_ceiling_usd}")
        console.print()

        try:
            # ── Phase 1: Read entire repo (once, then incrementally) ──
            await self._phase_read()

            # ── Main stabilization loop ────────────────────────────────
            while True:
                self.run.cycle_count += 1
                await self.storage.upsert_run(self.run)

                console.rule(f"Cycle {self.run.cycle_count}/{self.cfg.max_cycles}")

                # Phase 2: Audit
                score = await self._phase_audit()
                console.print(
                    f"[bold]Audit score:[/bold] {score.score:.1f} | "
                    f"CRITICAL={score.critical_count} MAJOR={score.major_count} "
                    f"MINOR={score.minor_count}"
                )

                # Check stabilized
                if score.critical_count == 0 and score.major_count == 0:
                    return await self._finish(RunStatus.STABILIZED)

                # Phase 3: Fix
                fix_attempts = await self._phase_fix()
                if not fix_attempts:
                    if await self._all_escalated():
                        return await self._finish(RunStatus.ESCALATED)

                # Phase 4: Review
                approved = await self._phase_review(fix_attempts)
                if not approved:
                    log.warning("No fixes approved in this cycle")

                # Phase 5: Commit approved fixes
                if approved:
                    await self._phase_commit(approved)
                    # Incremental re-read of changed files
                    await self._phase_read(incremental=True)

                # Convergence checks
                status = self.convergence.check(score)
                if status == RunStatus.HALTED:
                    return await self._finish(RunStatus.HALTED)

                # Cost check
                total_cost = await self.storage.get_total_cost(self.run.id)
                if total_cost >= self.cfg.cost_ceiling_usd:
                    console.print(
                        f"[bold red]Cost ceiling ${self.cfg.cost_ceiling_usd:.2f} reached "
                        f"(${total_cost:.4f} spent). Halting.[/bold red]"
                    )
                    return await self._finish(RunStatus.HALTED)

                # Max cycles
                if self.run.cycle_count >= self.cfg.max_cycles:
                    return await self._finish(RunStatus.HALTED)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
            return await self._finish(RunStatus.HALTED)
        except Exception as exc:
            log.error(f"Stabilizer fatal error: {exc}", exc_info=True)
            return await self._finish(RunStatus.FAILED)
        finally:
            await self.storage.close()

    # ─────────────────────────────────────────────────────────
    # Phases
    # ─────────────────────────────────────────────────────────

    async def _phase_read(self, incremental: bool = False) -> None:
        console.print("[cyan]Phase 1: Reading repository...[/cyan]")
        reader = ReaderAgent(
            storage=self.storage,
            run_id=self.run.id,
            repo_root=self.cfg.repo_root,
            config=self._agent_cfg,
            incremental=incremental or self.cfg.incremental,
            concurrency=self.cfg.concurrency,
        )
        counts = await reader.run()
        console.print(
            f"  Read: {counts['processed']} files | "
            f"Skipped (unchanged): {counts['skipped']} | "
            f"Errors: {counts['errors']}"
        )

    async def _phase_audit(self) -> AuditScore:
        console.print("[cyan]Phase 2: Auditing...[/cyan]")
        domains = [
            ExecutorType.SECURITY,
            ExecutorType.ARCHITECTURE,
            ExecutorType.STANDARDS,
        ]
        # Run all audit domains in parallel
        auditors = [
            AuditorAgent(
                storage=self.storage,
                run_id=self.run.id,
                executor_type=domain,
                master_prompt_path=self.cfg.master_prompt_path,
                config=self._agent_cfg,
            )
            for domain in domains
        ]
        results = await asyncio.gather(
            *[a.run() for a in auditors], return_exceptions=True
        )
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                log.warning(f"Auditor {domains[i].value} failed: {res}")

        # Compute score
        all_open = await self.storage.list_issues(
            run_id=self.run.id, status=IssueStatus.OPEN.value
        )
        score = AuditScore(run_id=self.run.id)
        for issue in all_open:
            if issue.severity == Severity.CRITICAL:
                score.critical_count += 1
            elif issue.severity == Severity.MAJOR:
                score.major_count += 1
            elif issue.severity == Severity.MINOR:
                score.minor_count += 1
            else:
                score.info_count += 1
        score.compute_score()
        await self.storage.append_score(score)
        self.run.scores.append(score)
        return score

    async def _phase_fix(self) -> list[FixAttempt]:
        console.print("[cyan]Phase 3: Generating fixes...[/cyan]")
        fixer = FixerAgent(
            storage=self.storage,
            run_id=self.run.id,
            repo_root=self.cfg.repo_root,
            master_prompt_path=self.cfg.master_prompt_path,
            config=self._agent_cfg,
        )
        attempts = await fixer.run()
        console.print(f"  Generated {len(attempts)} fix attempt(s)")
        return attempts

    async def _phase_review(self, attempts: list[FixAttempt]) -> list[FixAttempt]:
        console.print("[cyan]Phase 4: Reviewing fixes...[/cyan]")
        reviewer = ReviewerAgent(
            storage=self.storage,
            run_id=self.run.id,
            config=self._agent_cfg,
        )
        results = await reviewer.run()
        approved = []
        for attempt in attempts:
            review = await self.storage.get_review(attempt.id)
            if review and review.approve_for_commit:
                approved.append(attempt)
                console.print(f"  [green]APPROVED[/green]: fix {attempt.id[:8]}")
            else:
                console.print(f"  [red]REJECTED[/red]: fix {attempt.id[:8]}")
        return approved

    async def _phase_commit(self, approved: list[FixAttempt]) -> None:
        console.print("[cyan]Phase 5: Committing approved fixes...[/cyan]")
        fixer = FixerAgent(
            storage=self.storage,
            run_id=self.run.id,
            repo_root=self.cfg.repo_root,
            master_prompt_path=self.cfg.master_prompt_path,
            config=self._agent_cfg,
        )
        for attempt in approved:
            written = await fixer.write_fixed_files_to_disk(attempt)
            console.print(f"  Written {len(written)} file(s) to disk")

            if self.pr_manager and self.cfg.auto_commit:
                try:
                    pr_url = await self.pr_manager.create_pr_for_fix(
                        attempt=attempt,
                        run_id=self.run.id,
                        cycle=self.run.cycle_count,
                        repo_root=self.cfg.repo_root,
                    )
                    attempt.pr_url = pr_url
                    attempt.committed_at = datetime.utcnow()
                    await self.storage.upsert_fix(attempt)
                    console.print(f"  [green]PR created:[/green] {pr_url}")
                except Exception as exc:
                    log.error(f"Failed to create PR: {exc}")

            # Mark issues closed
            for iid in attempt.issue_ids:
                await self.storage.update_issue_status(iid, IssueStatus.CLOSED.value)

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    async def _all_escalated(self) -> bool:
        """True if all remaining open issues are escalated."""
        open_issues = await self.storage.list_issues(
            run_id=self.run.id, status=IssueStatus.OPEN.value
        )
        return len(open_issues) == 0

    async def _finish(self, status: RunStatus) -> RunStatus:
        assert self.run is not None
        self.run.completed_at = datetime.utcnow()
        await self.storage.update_run_status(self.run.id, status)

        total_cost = await self.storage.get_total_cost(self.run.id)
        self._print_final_report(status, total_cost)
        return status

    def _print_final_report(self, status: RunStatus, total_cost: float) -> None:
        assert self.run is not None
        table = Table(title=f"OpenMOSS Run Complete — {status.value}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Status", f"[bold {'green' if status == RunStatus.STABILIZED else 'red'}]{status.value}[/]")
        table.add_row("Cycles", str(self.run.cycle_count))
        table.add_row("Total Cost", f"${total_cost:.4f}")
        if self.run.scores:
            s = self.run.scores[-1]
            table.add_row("Final Score", str(s.score))
            table.add_row("CRITICAL remaining", str(s.critical_count))
            table.add_row("MAJOR remaining", str(s.major_count))
        console.print(table)
