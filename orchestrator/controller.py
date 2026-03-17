from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from agents.auditor import AuditorAgent
from agents.base import AgentConfig
from agents.fixer import FixerAgent
from agents.patrol import PatrolAgent
from agents.planner import PlannerAgent
from agents.reader import ReaderAgent
from agents.reviewer import ReviewerAgent
from config.loader import load_config
from plugins.base import PluginManager
from utils.rate_limiter import RateLimiter
from brain.schemas import (
    AuditRun,
    AuditScore,
    AuditTrailEntry,
    AutonomyLevel,
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
from utils.audit_trail import AuditTrailSigner

log = logging.getLogger(__name__)
console = Console()


class StabilizerConfig:
    def __init__(
        self,
        repo_url: str,
        repo_root: Path,
        master_prompt_path: Path,
        github_token: str = "",
        primary_model: str = "claude-sonnet-4-20250514",
        fallback_models: list[str] | None = None,
        max_cycles: int = 50,
        cost_ceiling_usd: float = 50.0,
        concurrency: int = 4,
        auto_commit: bool = True,
        branch_prefix: str = "stabilizer",
        incremental: bool = True,
        run_mypy: bool = True,
        run_semgrep: bool = True,
        fail_gate_on_warning: bool = False,
        autonomy_level: AutonomyLevel = AutonomyLevel.AUTO_FIX,
        load_bearing_paths: list[str] | None = None,
        hmac_secret: str = "",
        notification_hooks: list[Any] | None = None,
    ) -> None:
        self.repo_url = repo_url
        self.repo_root = repo_root
        self.master_prompt_path = master_prompt_path
        self.github_token = github_token
        self.primary_model = primary_model
        self.fallback_models = fallback_models or ["gpt-4o-mini"]
        self.max_cycles = max_cycles
        self.cost_ceiling_usd = cost_ceiling_usd
        self.concurrency = concurrency
        self.auto_commit = auto_commit
        self.branch_prefix = branch_prefix
        self.incremental = incremental
        self.run_mypy = run_mypy
        self.run_semgrep = run_semgrep
        self.fail_gate_on_warning = fail_gate_on_warning
        self.autonomy_level = autonomy_level
        self.load_bearing_paths = load_bearing_paths or []
        self.hmac_secret = hmac_secret
        self.notification_hooks = notification_hooks or []


class StabilizerController:

    def __init__(self, cfg: StabilizerConfig, toml_config_path: Path | None = None) -> None:
        self.cfg = cfg
        self._toml = load_config(toml_config_path or cfg.repo_root / "config" / "default.toml")
        self.storage: BrainStorage = SQLiteBrainStorage(
            cfg.repo_root / ".stabilizer" / "brain.db"
        )
        self.convergence = ConvergenceDetector(
            stall_threshold=self._toml.loop.stall_threshold,
            regression_threshold=self._toml.loop.regression_threshold,
            max_cycles=cfg.max_cycles,
        )
        self.gate = StaticAnalysisGate(
            run_mypy=cfg.run_mypy,
            run_semgrep=cfg.run_semgrep,
            fail_on_warning=cfg.fail_gate_on_warning,
            repo_root=cfg.repo_root,
        )
        self.trail_signer = AuditTrailSigner(secret=cfg.hmac_secret)
        self.pr_manager: PRManager | None = None
        self.run: AuditRun | None = None
        self._patrol_task: asyncio.Task | None = None
        self._plugin_manager = PluginManager()
        self._rate_limiter = RateLimiter()
        self._agent_cfg = AgentConfig(
            model=cfg.primary_model,
            fallback_models=cfg.fallback_models,
            cost_ceiling_usd=cfg.cost_ceiling_usd,
        )

    async def initialise(self, run_id: str | None = None) -> AuditRun:
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
            autonomy_level=self.cfg.autonomy_level,
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

    async def stabilize(self) -> RunStatus:
        assert self.run is not None, "Call initialise() first"

        console.rule(f"[bold blue]RHODAWK AI CODE STABILIZER Stabilizer — {self.run.repo_name}")
        console.print(f"Run ID: {self.run.id}")
        console.print(f"Model:  {self.cfg.primary_model}")
        console.print(f"Autonomy: {self.cfg.autonomy_level.value}")
        console.print(
            f"Max cycles: {self.cfg.max_cycles} | "
            f"Cost ceiling: ${self.cfg.cost_ceiling_usd}"
        )

        patrol = PatrolAgent(
            storage=self.storage,
            run_id=self.run.id,
            cost_ceiling_usd=self.cfg.cost_ceiling_usd,
            config=self._agent_cfg,
            notification_hooks=self.cfg.notification_hooks,
        )
        self._patrol_task = asyncio.create_task(patrol.run(), name="patrol")

        try:
            await self._phase_read(incremental=False)

            while True:
                self.run.cycle_count += 1
                await self.storage.upsert_run(self.run)
                console.rule(f"Cycle {self.run.cycle_count}/{self.cfg.max_cycles}")

                score = await self._phase_audit()
                console.print(
                    f"[bold]Audit score:[/bold] {score.score:.1f} | "
                    f"CRITICAL={score.critical_count} MAJOR={score.major_count} "
                    f"MINOR={score.minor_count} ESCALATED={score.escalated_count}"
                )

                if score.critical_count == 0 and score.major_count == 0:
                    if score.escalated_count > 0:
                        console.print(
                            f"[yellow]Warning: {score.escalated_count} escalated issues "
                            "require human review — returning ESCALATED, not STABILIZED.[/yellow]"
                        )
                        return await self._finish(RunStatus.ESCALATED)
                    return await self._finish(RunStatus.STABILIZED)

                fix_attempts = await self._phase_fix()
                if not fix_attempts:
                    if await self._all_escalated():
                        return await self._finish(RunStatus.ESCALATED)

                if self.cfg.autonomy_level == AutonomyLevel.READ_ONLY:
                    console.print("[yellow]Autonomy=read_only: fixes generated but not committed.[/yellow]")
                    return await self._finish(RunStatus.HALTED)

                reviewed = await self._phase_review(fix_attempts)

                if self.cfg.autonomy_level == AutonomyLevel.PROPOSE_ONLY:
                    if reviewed and self.pr_manager and self.cfg.auto_commit:
                        await self._phase_commit(reviewed, create_pr_only=True)
                    console.print("[yellow]Autonomy=propose_only: PRs created, not auto-merged.[/yellow]")
                else:
                    if reviewed:
                        planner_passed = await self._phase_plan(reviewed)
                        if not planner_passed:
                            log.warning("Planner blocked all fixes in this cycle")
                            reviewed = []

                    if reviewed:
                        gate_passed = await self._phase_gate(reviewed)
                        if not gate_passed:
                            log.warning("Static gate rejected all fixes in this cycle")
                            reviewed = []

                    if reviewed:
                        await self._phase_commit(reviewed)
                        await self._phase_read(incremental=True)

                convergence_status = self.convergence.check(score)
                if convergence_status == RunStatus.HALTED:
                    return await self._finish(RunStatus.HALTED)
                if convergence_status == RunStatus.STABILIZED:
                    return await self._finish(RunStatus.STABILIZED)

                total_cost = await self.storage.get_total_cost(self.run.id)
                if total_cost >= self.cfg.cost_ceiling_usd:
                    console.print(
                        f"[bold red]Cost ceiling ${self.cfg.cost_ceiling_usd:.2f} "
                        f"reached (${total_cost:.4f} spent). Halting.[/bold red]"
                    )
                    return await self._finish(RunStatus.HALTED)

                if self.run.cycle_count >= self.cfg.max_cycles:
                    return await self._finish(RunStatus.HALTED)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
            return await self._finish(RunStatus.HALTED)
        except Exception as exc:
            log.error(f"Stabilizer fatal error: {exc}", exc_info=True)
            return await self._finish(RunStatus.FAILED)
        finally:
            if self._patrol_task and not self._patrol_task.done():
                patrol.stop()
                try:
                    await asyncio.wait_for(self._patrol_task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    self._patrol_task.cancel()
            await self.storage.close()

    async def run_read_phase(self, incremental: bool = False) -> dict[str, int]:
        return await self._phase_read(incremental=incremental)

    async def run_audit_phase(self) -> AuditScore:
        return await self._phase_audit()

    async def _phase_read(self, incremental: bool = False) -> dict[str, int]:
        console.print("[cyan]Phase 1: Reading repository...[/cyan]")
        reader = ReaderAgent(
            storage=self.storage,
            run_id=self.run.id,
            repo_root=self.cfg.repo_root,
            config=self._agent_cfg,
            incremental=incremental and self.cfg.incremental,
            concurrency=self.cfg.concurrency,
            load_bearing_paths=self.cfg.load_bearing_paths,
        )
        counts = await reader.run()
        console.print(
            f"  Read: {counts['processed']} | "
            f"Skipped: {counts['skipped']} | "
            f"Errors: {counts['errors']}"
        )
        return counts

    async def _phase_audit(self) -> AuditScore:
        console.print("[cyan]Phase 2: Auditing...[/cyan]")
        self._plugin_manager.load_builtin_plugins()
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

        files = await self.storage.list_files()
        plugin_issue_count = 0
        for file_record in files:
            abs_path = self.cfg.repo_root / file_record.path
            try:
                content_text = abs_path.read_text(encoding="utf-8", errors="replace")
                plugin_issues = await self._plugin_manager.run_all(
                    file_path=file_record.path,
                    content=content_text,
                    language=file_record.language,
                    run_id=self.run.id,
                )
                for pi in plugin_issues:
                    await self.storage.upsert_issue(pi)
                plugin_issue_count += len(plugin_issues)
            except OSError:
                pass
        if plugin_issue_count:
            console.print(f"  Plugins found {plugin_issue_count} additional issue(s)")

        all_open = await self.storage.list_issues(
            run_id=self.run.id, status=IssueStatus.OPEN.value
        )
        escalated = await self.storage.list_issues(
            run_id=self.run.id, status=IssueStatus.ESCALATED.value
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
        score.escalated_count = len(escalated)
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
            repo_root=self.cfg.repo_root,
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

    async def _phase_plan(
        self, approved: list[FixAttempt]
    ) -> list[FixAttempt]:
        console.print("[cyan]Phase 5: Consequence reasoning (Planner)...[/cyan]")
        planner = PlannerAgent(
            storage=self.storage,
            run_id=self.run.id,
            repo_root=self.cfg.repo_root,
            config=self._agent_cfg,
        )

        for attempt in approved:
            attempt.planner_approved = None

        await planner.run()

        planner_passed: list[FixAttempt] = []
        for attempt in approved:
            refreshed = await self.storage.get_fix(attempt.id)
            if refreshed and refreshed.planner_approved is not False:
                planner_passed.append(attempt)
                console.print(f"  [green]PLANNER OK[/green]: fix {attempt.id[:8]}")
            else:
                reason = refreshed.planner_reason if refreshed else "unknown"
                console.print(f"  [red]PLANNER BLOCKED[/red]: fix {attempt.id[:8]} — {reason[:60]}")
        return planner_passed

    async def _phase_gate(
        self, approved: list[FixAttempt]
    ) -> list[FixAttempt]:
        console.print("[cyan]Phase 6: Static analysis gate...[/cyan]")
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
                    console.print(f"  [green]GATE PASS[/green]: {ff.path}")
                else:
                    reason = gr.rejection_reason if gr else "unknown"
                    console.print(f"  [red]GATE FAIL[/red]: {ff.path} — {reason}")
                    any_rejected = True
                    for iid in attempt.issue_ids:
                        await self.storage.update_issue_status(
                            iid, IssueStatus.OPEN.value,
                            reason=f"Gate rejection: {reason[:200]}",
                        )

            attempt.gate_passed = len(passed_files) > 0
            attempt.gate_reason = "All files passed" if not any_rejected else "Some files rejected"
            await self.storage.upsert_fix(attempt)

            if passed_files:
                attempt.fixed_files = passed_files
                gated.append(attempt)
            elif any_rejected:
                log.warning(f"All files in fix {attempt.id[:8]} rejected by gate")

        return gated

    async def _phase_commit(
        self,
        approved: list[FixAttempt],
        create_pr_only: bool = False,
    ) -> None:
        console.print("[cyan]Phase 7: Committing approved fixes...[/cyan]")
        fixer = FixerAgent(
            storage=self.storage,
            run_id=self.run.id,
            repo_root=self.cfg.repo_root,
            master_prompt_path=self.cfg.master_prompt_path,
            config=self._agent_cfg,
        )
        for attempt in approved:
            if not create_pr_only:
                written = await fixer.write_fixed_files_to_disk(attempt)
                console.print(f"  Written {len(written)} file(s) to disk")

                entry = AuditTrailEntry(
                    run_id=self.run.id,
                    event_type="FILES_COMMITTED",
                    entity_id=attempt.id,
                    entity_type="FixAttempt",
                    after_state=str([f.path for f in attempt.fixed_files]),
                    actor="stabilizer",
                )
                entry.hmac_signature = self.trail_signer.sign(entry)
                await self.storage.append_audit_trail(entry)

            if self.pr_manager and self.cfg.auto_commit:
                try:
                    pr_url = await self.pr_manager.create_pr_for_fix(
                        attempt=attempt,
                        run_id=self.run.id,
                        cycle=self.run.cycle_count,
                        repo_root=self.cfg.repo_root,
                    )
                    attempt.pr_url = pr_url
                    attempt.committed_at = datetime.now(tz=timezone.utc)
                    await self.storage.upsert_fix(attempt)
                    console.print(f"  [green]PR created:[/green] {pr_url}")
                except Exception as exc:
                    log.error(f"Failed to create PR: {exc}")

            for iid in attempt.issue_ids:
                await self.storage.update_issue_status(iid, IssueStatus.CLOSED.value)

            if not create_pr_only:
                await self._requeue_transitive_dependents(attempt)

    async def _requeue_transitive_dependents(self, attempt: FixAttempt) -> None:
        changed_paths = {ff.path for ff in attempt.fixed_files}
        if not changed_paths:
            return
        all_issues = await self.storage.list_issues(run_id=self.run.id)
        requeued = 0
        for issue in all_issues:
            if issue.status == IssueStatus.CLOSED:
                continue
            deps = set(issue.fix_requires_files or [issue.file_path])
            if deps & changed_paths:
                await self.storage.update_issue_status(
                    issue.id, IssueStatus.OPEN.value,
                    reason=f"Re-opened: dependent file changed ({', '.join(changed_paths & deps)})",
                )
                requeued += 1
        if requeued:
            console.print(f"  [dim]Transitive re-audit: {requeued} dependent issue(s) re-opened[/dim]")

    async def revert_last_cycle(self) -> bool:
        assert self.run is not None
        try:
            import git
            repo = git.Repo(self.cfg.repo_root)
            branch_prefix = self.cfg.branch_prefix
            log.info(f"Reverting last stabilizer cycle on {self.cfg.repo_root}")
            commits = list(repo.iter_commits(max_count=20))
            for commit in commits:
                if f"fix(rhodawk-ai-code-stabilizer)" in commit.message or branch_prefix in commit.message:
                    repo.git.revert(commit.hexsha, no_edit=True)
                    log.info(f"Reverted commit: {commit.hexsha[:8]} — {commit.message[:60]}")
                    entry = AuditTrailEntry(
                        run_id=self.run.id,
                        event_type="CYCLE_REVERTED",
                        entity_id=commit.hexsha,
                        entity_type="GitCommit",
                        before_state=commit.hexsha,
                        after_state="reverted",
                        actor="stabilizer",
                    )
                    entry.hmac_signature = self.trail_signer.sign(entry)
                    await self.storage.append_audit_trail(entry)
                    return True
            log.warning("No stabilizer commit found to revert")
            return False
        except Exception as exc:
            log.error(f"Revert failed: {exc}")
            return False

    async def _all_escalated(self) -> bool:
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
            i for i in open_issues if i.status == IssueStatus.ESCALATED
        ]
        return len(non_terminal) == 0 and len(escalated_only) > 0

    async def _finish(self, status: RunStatus) -> RunStatus:
        assert self.run is not None
        self.run.completed_at = datetime.now(tz=timezone.utc)
        await self.storage.update_run_status(self.run.id, status)

        total_cost = await self.storage.get_total_cost(self.run.id)
        self._print_final_report(status, total_cost)
        return status

    def _print_final_report(self, status: RunStatus, total_cost: float) -> None:
        assert self.run is not None
        table = Table(title=f"RHODAWK AI CODE STABILIZER Run Complete — {status.value}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        color = "green" if status == RunStatus.STABILIZED else "red"
        table.add_row("Status", f"[bold {color}]{status.value}[/]")
        table.add_row("Cycles", str(self.run.cycle_count))
        table.add_row("Total Cost", f"${total_cost:.4f}")
        if self.run.scores:
            s = self.run.scores[-1]
            table.add_row("Final Score", str(s.score))
            table.add_row("CRITICAL remaining", str(s.critical_count))
            table.add_row("MAJOR remaining", str(s.major_count))
            table.add_row("ESCALATED", str(s.escalated_count))
        console.print(table)
