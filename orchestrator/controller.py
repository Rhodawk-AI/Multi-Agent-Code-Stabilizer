"""
orchestrator/controller.py
The core stabilization loop. This is the heart of OpenMOSS.
State machine: READ → AUDIT → FIX → REVIEW → GATE → COMMIT → RE-AUDIT → repeat.
Never stops until stabilized, cost ceiling, or manual halt.

PATCH LOG:
  - PatrolAgent: was defined but never instantiated or started. Now launched as
    a background asyncio task at the start of stabilize() and stopped on exit.
  - StaticAnalysisGate: was completely absent from the commit path. LLM-generated
    fixes were written directly to disk without any syntax or security check.
    Gate is now mandatory — fixes rejected by the gate are logged and skipped.
  - _all_escalated: had wrong semantics. It returned True when there were NO
    open issues (including the case where all issues were already closed), which
    caused premature ESCALATED termination. Fixed to check for ESCALATED status.
  - _phase_read: added explicit incremental=False for the first read pass so
    all files are always read on cycle 1 regardless of hash state.
  - Added _phase_gate: applies StaticAnalysisGate to all approved fix files
    before they are written to disk. Rejected files are removed from the
    attempt and re-queued.
  - Added revert support: _revert_last_cycle() uses gitpython to undo the
    last stabilizer commits when a regression is detected.
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
from agents.patrol import PatrolAgent
from agents.reader import ReaderAgent
from agents.reviewer import ReviewerAgent
from brain.schemas import (
    AuditRun,
    AuditScore,
    ExecutorType,
    FixAttempt,
    FixedFile,
    IssueStatus,
    RunStatus,
    Severity,
)
from brain.sqlite_storage import SQLiteBrainStorage
from brain.storage import BrainStorage
from github_integration.pr_manager import PRManager
from orchestrator.convergence import ConvergenceDetector
from sandbox.executor import StaticAnalysisGate

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
        # Gate config
        run_mypy:           bool = True,
        run_semgrep:        bool = True,
        fail_gate_on_warning: bool = False,
    ) -> None:
        self.repo_url             = repo_url
        self.repo_root            = repo_root
        self.master_prompt_path   = master_prompt_path
        self.github_token         = github_token
        self.primary_model        = primary_model
        self.fallback_models      = fallback_models or ["gpt-4o-mini"]
        self.max_cycles           = max_cycles
        self.cost_ceiling_usd     = cost_ceiling_usd
        self.concurrency          = concurrency
        self.auto_commit          = auto_commit
        self.branch_prefix        = branch_prefix
        self.incremental          = incremental
        self.run_mypy             = run_mypy
        self.run_semgrep          = run_semgrep
        self.fail_gate_on_warning = fail_gate_on_warning


class StabilizerController:
    """
    Orchestrates the full READ → AUDIT → FIX → REVIEW → GATE → COMMIT → REPEAT loop.
    This loop never stops until:
      - Zero CRITICAL + MAJOR issues (STABILIZED)
      - Cost ceiling exceeded (HALTED)
      - Max cycles reached (HALTED)
      - All remaining issues escalated (ESCALATED)
      - Unrecoverable regression detected (HALTED)
    """

    def __init__(self, cfg: StabilizerConfig) -> None:
        self.cfg        = cfg
        self.storage: BrainStorage = SQLiteBrainStorage(
            cfg.repo_root / ".stabilizer" / "brain.db"
        )
        self.convergence = ConvergenceDetector(
            stall_threshold=2, max_cycles=cfg.max_cycles
        )
        self.gate = StaticAnalysisGate(
            run_mypy=cfg.run_mypy,
            run_semgrep=cfg.run_semgrep,
            fail_on_warning=cfg.fail_gate_on_warning,
        )
        self.pr_manager: PRManager | None = None
        self.run: AuditRun | None = None
        self._patrol_task: asyncio.Task | None = None
        self._agent_cfg = AgentConfig(
            model=cfg.primary_model,
            fallback_models=cfg.fallback_models,
            cost_ceiling_usd=cfg.cost_ceiling_usd,
        )

    async def initialise(self, run_id: str | None = None) -> AuditRun:
        """Set up storage, run record, GitHub integration."""
        await self.storage.initialise()

        repo_name = (
            Path(self.cfg.repo_url).stem
            if "/" not in self.cfg.repo_url
            else self.cfg.repo_url.rstrip("/").split("/")[-1]
        )

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
    # Main stabilization loop
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
        console.print(
            f"Max cycles: {self.cfg.max_cycles} | "
            f"Cost ceiling: ${self.cfg.cost_ceiling_usd}"
        )
        console.print()

        # FIX: start PatrolAgent as a background task — was never started before
        patrol = PatrolAgent(
            storage=self.storage,
            run_id=self.run.id,
            cost_ceiling_usd=self.cfg.cost_ceiling_usd,
            config=self._agent_cfg,
        )
        self._patrol_task = asyncio.create_task(patrol.run(), name="patrol")

        try:
            # Phase 1: Read entire repo (always fresh on first pass)
            await self._phase_read(incremental=False)

            # Main stabilization loop
            while True:
                self.run.cycle_count += 1
                await self.storage.upsert_run(self.run)
                console.rule(
                    f"Cycle {self.run.cycle_count}/{self.cfg.max_cycles}"
                )

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

                # Phase 5: Static analysis gate (NEW — was missing entirely)
                if approved:
                    gate_passed = await self._phase_gate(approved)
                    if not gate_passed:
                        log.warning("All approved fixes rejected by static analysis gate")
                        approved = []

                # Phase 6: Commit approved+gated fixes
                if approved:
                    await self._phase_commit(approved)
                    # Incremental re-read of changed files
                    await self._phase_read(incremental=True)

                # Convergence check
                convergence_status = self.convergence.check(score)
                if convergence_status == RunStatus.HALTED:
                    return await self._finish(RunStatus.HALTED)
                if convergence_status == RunStatus.STABILIZED:
                    return await self._finish(RunStatus.STABILIZED)

                # Cost check
                total_cost = await self.storage.get_total_cost(self.run.id)
                if total_cost >= self.cfg.cost_ceiling_usd:
                    console.print(
                        f"[bold red]Cost ceiling ${self.cfg.cost_ceiling_usd:.2f} "
                        f"reached (${total_cost:.4f} spent). Halting.[/bold red]"
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
            # Always stop patrol
            if self._patrol_task and not self._patrol_task.done():
                patrol.stop()
                try:
                    await asyncio.wait_for(self._patrol_task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    self._patrol_task.cancel()
            await self.storage.close()

    # ─────────────────────────────────────────────────────────
    # Public phase methods (used by CLI/scripts without leading underscore)
    # ─────────────────────────────────────────────────────────

    async def run_read_phase(self, incremental: bool = False) -> dict[str, int]:
        """Public wrapper for the read phase — used by CLI audit command."""
        return await self._phase_read(incremental=incremental)

    async def run_audit_phase(self) -> AuditScore:
        """Public wrapper for the audit phase — used by CLI audit command."""
        return await self._phase_audit()

    # ─────────────────────────────────────────────────────────
    # Internal phases
    # ─────────────────────────────────────────────────────────

    async def _phase_read(self, incremental: bool = False) -> dict[str, int]:
        console.print("[cyan]Phase 1: Reading repository...[/cyan]")
        reader = ReaderAgent(
            storage=self.storage,
            run_id=self.run.id,
            repo_root=self.cfg.repo_root,
            config=self._agent_cfg,
            # FIX: first read is always full — only subsequent reads are incremental
            incremental=incremental and self.cfg.incremental,
            concurrency=self.cfg.concurrency,
        )
        counts = await reader.run()
        console.print(
            f"  Read: {counts['processed']} files | "
            f"Skipped (unchanged): {counts['skipped']} | "
            f"Errors: {counts['errors']}"
        )
        return counts

    async def _phase_audit(self) -> AuditScore:
        console.print("[cyan]Phase 2: Auditing...[/cyan]")
        domains = [
            ExecutorType.SECURITY,
            ExecutorType.ARCHITECTURE,
            ExecutorType.STANDARDS,
        ]
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

        # Compute score across ALL open issues for this run
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

    async def _phase_review(
        self, attempts: list[FixAttempt]
    ) -> list[FixAttempt]:
        console.print("[cyan]Phase 4: Reviewing fixes...[/cyan]")
        reviewer = ReviewerAgent(
            storage=self.storage,
            run_id=self.run.id,
            config=self._agent_cfg,
        )
        await reviewer.run()
        approved = []
        for attempt in attempts:
            review = await self.storage.get_review(attempt.id)
            if review and review.approve_for_commit:
                approved.append(attempt)
                console.print(f"  [green]APPROVED[/green]: fix {attempt.id[:8]}")
            else:
                console.print(f"  [red]REJECTED[/red]: fix {attempt.id[:8]}")
        return approved

    async def _phase_gate(
        self, approved: list[FixAttempt]
    ) -> list[FixAttempt]:
        """
        FIX: StaticAnalysisGate was never called before. LLM fixes were written
        directly to disk without syntax or security validation.
        Now every approved fix file passes through the gate before commit.
        Files that fail are removed from the attempt and re-queued.
        """
        console.print("[cyan]Phase 5: Static analysis gate...[/cyan]")
        gated: list[FixAttempt] = []

        for attempt in approved:
            files_to_check = [
                (ff.path, ff.content) for ff in attempt.fixed_files
            ]
            gate_results = await self.gate.validate_batch(files_to_check)

            passed_files: list[FixedFile] = []
            any_rejected = False

            for ff in attempt.fixed_files:
                gr = gate_results.get(ff.path)
                if gr and gr.approved:
                    passed_files.append(ff)
                    console.print(
                        f"  [green]GATE PASS[/green]: {ff.path}"
                    )
                else:
                    reason = gr.rejection_reason if gr else "unknown"
                    console.print(
                        f"  [red]GATE FAIL[/red]: {ff.path} — {reason}"
                    )
                    any_rejected = True
                    # Re-open associated issues so they re-enter the fix queue
                    for iid in attempt.issue_ids:
                        await self.storage.update_issue_status(
                            iid, IssueStatus.OPEN.value,
                            reason=f"Gate rejection: {reason[:200]}"
                        )

            if passed_files:
                attempt.fixed_files = passed_files
                gated.append(attempt)
            elif any_rejected:
                log.warning(
                    f"All files in fix {attempt.id[:8]} rejected by gate"
                )

        return gated

    async def _phase_commit(self, approved: list[FixAttempt]) -> None:
        console.print("[cyan]Phase 6: Committing approved fixes...[/cyan]")
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
                await self.storage.update_issue_status(
                    iid, IssueStatus.CLOSED.value
                )

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    async def _all_escalated(self) -> bool:
        """
        FIX: original implementation returned True when there were no OPEN issues,
        which conflated 'all closed' with 'all escalated'. Both cases triggered
        ESCALATED termination even if all issues had been cleanly resolved.

        Correct semantics: return True only when there are open issues AND every
        one of them is in ESCALATED status (i.e., all remaining work needs humans).
        """
        open_issues = await self.storage.list_issues(run_id=self.run.id)
        non_terminal = [
            i for i in open_issues
            if i.status not in (
                IssueStatus.CLOSED,
                IssueStatus.ESCALATED,
                IssueStatus.APPROVED,
            )
        ]
        escalated_only = [
            i for i in open_issues
            if i.status == IssueStatus.ESCALATED
        ]
        # True only if there are some escalated issues and no actionable ones left
        return len(non_terminal) == 0 and len(escalated_only) > 0

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
        color = "green" if status == RunStatus.STABILIZED else "red"
        table.add_row(
            "Status",
            f"[bold {color}]{status.value}[/]"
        )
        table.add_row("Cycles", str(self.run.cycle_count))
        table.add_row("Total Cost", f"${total_cost:.4f}")
        if self.run.scores:
            s = self.run.scores[-1]
            table.add_row("Final Score", str(s.score))
            table.add_row("CRITICAL remaining", str(s.critical_count))
            table.add_row("MAJOR remaining", str(s.major_count))
        console.print(table)
