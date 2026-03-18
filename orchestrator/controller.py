"""
orchestrator/controller.py
==========================
Main orchestration loop for MACS.

CHANGES vs previous version
────────────────────────────
• Graph engine built after read phase (GAP-1); stored to brain.
• VectorBrain initialised at boot and passed to ReaderAgent (GAP-8).
• ConsensusEngine runs after all three auditors complete (GAP-5).
• FormalVerifierAgent called in gate phase for CRITICAL fixes in non-GENERAL
  domains (GAP-4).
• TestRunnerAgent called after each successful commit (GAP-14).
• Parallel fix execution via graph engine's non_overlapping_fix_batches (GAP-11).
• _requeue_transitive_dependents now uses graph engine impact_radius (GAP-1/18).
• AuditorAgent receives repo_root for direct-source auditing (GAP-2).
• DomainMode threaded through all agents (GAP-7/8).
• PR batching: fixes grouped by module boundary, not one-per-fix (GAP-13).
• domain_mode written to AuditRun in brain.
• All agent constructors updated to pass graph_engine, vector_brain, etc.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agents.auditor import AuditorAgent
from agents.fixer import FixerAgent
from agents.formal_verifier import FormalVerifierAgent
from agents.patrol import PatrolAgent
from agents.planner import PlannerAgent
from agents.reader import ReaderAgent
from agents.reviewer import ReviewerAgent
from agents.test_runner import TestRunnerAgent
from brain.graph import DependencyGraphEngine
from brain.schemas import (
    AuditRun,
    AuditScore,
    AuditTrailEntry,
    AutonomyLevel,
    DomainMode,
    ExecutorType,
    FormalVerificationStatus,
    Issue,
    IssueStatus,
    PatrolEvent,
    RunStatus,
    Severity,
    TestRunStatus,
)
from brain.sqlite_storage import SQLiteBrainStorage
from brain.vector_store import VectorBrain
from github_integration.pr_manager import PRManager
from mcp_clients.manager import MCPManager
from orchestrator.consensus import ConsensusEngine
from orchestrator.convergence import ConvergenceDetector
from sandbox.executor import StaticAnalysisGate
from utils.audit_trail import AuditTrailSigner

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

class StabilizerConfig(BaseModel):
    repo_url:             str
    repo_root:            Path
    master_prompt_path:   Path             = Path("config/prompts/base.md")
    github_token:         str              = ""
    primary_model:        str              = "claude-sonnet-4-20250514"
    critical_fix_model:   str              = "claude-opus-4-20250514"
    triage_model:         str              = "claude-haiku-4-5-20251001"
    fallback_models:      list[str]        = Field(default_factory=lambda: ["gpt-4o-mini"])
    max_cycles:           int              = 50
    cost_ceiling_usd:     float            = 50.0
    concurrency:          int              = 4
    chunk_concurrency:    int              = 4
    auto_commit:          bool             = True
    autonomy_level:       AutonomyLevel    = AutonomyLevel.AUTO_FIX
    domain_mode:          DomainMode       = DomainMode.GENERAL
    run_semgrep:          bool             = True
    run_mypy:             bool             = True
    run_bandit:           bool             = True
    run_ruff:             bool             = True
    formal_verification:  bool             = False
    run_tests_after_fix:  bool             = True
    vector_store_enabled: bool             = False
    vector_store_path:    str              = ""
    graph_enabled:        bool             = True
    validate_findings:    bool             = True
    plugin_paths:         list[Path]       = Field(default_factory=list)
    base_branch:          str              = "main"
    branch_prefix:        str              = "stabilizer"

    model_config = {"arbitrary_types_allowed": True}


# ──────────────────────────────────────────────────────────────────────────────
# AgentConfig helper
# ──────────────────────────────────────────────────────────────────────────────

from agents.base import AgentConfig as _AgentConfig  # noqa: E402


def _agent_cfg(cfg: StabilizerConfig, run_id: str) -> _AgentConfig:
    return _AgentConfig(
        model=cfg.primary_model,
        fallback_models=cfg.fallback_models,
        critical_fix_model=cfg.critical_fix_model,
        triage_model=cfg.triage_model,
        cost_ceiling_usd=cfg.cost_ceiling_usd,
        run_id=run_id,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Controller
# ──────────────────────────────────────────────────────────────────────────────

class StabilizerController:

    def __init__(self, config: StabilizerConfig) -> None:
        self.config       = config
        self.storage:       SQLiteBrainStorage | None = None
        self.run:           AuditRun | None           = None
        self.graph_engine:  DependencyGraphEngine     = DependencyGraphEngine()
        self.vector_brain:  VectorBrain | None        = None
        self.consensus:     ConsensusEngine | None    = None
        self.convergence:   ConvergenceDetector | None = None
        self.patrol:        PatrolAgent | None        = None
        self.mcp:           MCPManager | None         = None
        self.pr_manager:    PRManager | None          = None
        self._trail_signer: AuditTrailSigner | None   = None
        self._patrol_task:  asyncio.Task | None       = None
        self.log            = log

    # ── Bootstrap ─────────────────────────────────────────────────────────────

    async def initialise(self, resume_run_id: str | None = None) -> AuditRun:
        db_path = self.config.repo_root / ".stabilizer" / "brain.db"
        self.storage = SQLiteBrainStorage(db_path)
        await self.storage.initialise()

        # Vector brain
        if self.config.vector_store_enabled:
            vpath = self.config.vector_store_path or str(
                self.config.repo_root / ".stabilizer" / "vectors"
            )
            self.vector_brain = VectorBrain(store_path=vpath)
            self.vector_brain.initialise()
            self.log.info(f"VectorBrain initialised at {vpath}")

        # Audit trail signer
        self._trail_signer = AuditTrailSigner()

        # Resume or create run
        if resume_run_id:
            run = await self.storage.get_run(resume_run_id)
            if run:
                self.run = run
                self.log.info(f"Resuming run {run.id[:8]}")
            else:
                self.log.warning(f"Run {resume_run_id!r} not found — starting new run")

        if self.run is None:
            self.run = AuditRun(
                repo_url=self.config.repo_url,
                repo_name=Path(self.config.repo_url).name or "repo",
                branch=self.config.base_branch,
                master_prompt_path=str(self.config.master_prompt_path),
                autonomy_level=self.config.autonomy_level,
                domain_mode=self.config.domain_mode,
                max_cycles=self.config.max_cycles,
            )
            await self.storage.upsert_run(self.run)
            self.log.info(
                f"New run {self.run.id[:8]} "
                f"[domain={self.config.domain_mode.value}]"
            )

        self.convergence = ConvergenceDetector(
            max_cycles=self.config.max_cycles,
        )

        # Consensus engine
        self.consensus = ConsensusEngine(
            graph_engine=self.graph_engine if self.config.graph_enabled else None
        )

        # MCP + GitHub
        self.mcp = MCPManager(
            repo_root=str(self.config.repo_root),
            github_token=self.config.github_token,
        )
        if self.config.github_token and self.config.auto_commit:
            self.pr_manager = PRManager(
                token=self.config.github_token,
                repo_url=self.config.repo_url,
                base_branch=self.config.base_branch,
                branch_prefix=self.config.branch_prefix,
            )

        # Patrol agent
        if self.storage and self.run:
            self.patrol = PatrolAgent(
                storage=self.storage,
                run_id=self.run.id,
                cost_ceiling_usd=self.config.cost_ceiling_usd,
            )

        return self.run

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def stabilize(self) -> RunStatus:
        assert self.run and self.storage, "Call initialise() first"

        # Start patrol in background
        if self.patrol:
            self._patrol_task = asyncio.create_task(self.patrol.run())

        try:
            # Initial read
            await self.run_read_phase(incremental=False)
            await self._build_graph()

            while True:
                self.run.cycle_count += 1
                await self.storage.upsert_run(self.run)
                self.log.info(
                    f"═══ Cycle {self.run.cycle_count}/{self.config.max_cycles} "
                    f"[{self.config.domain_mode.value}] ═══"
                )

                issues = await self._phase_audit()
                score  = await self._record_score(issues)
                terminal = self.convergence.check(score)

                if terminal:
                    self.log.info(f"Convergence: {terminal.value}")
                    await self._finalise(terminal)
                    return terminal

                if not issues:
                    self.log.info("No issues to fix — stabilized")
                    await self._finalise(RunStatus.STABILIZED)
                    return RunStatus.STABILIZED

                # Consensus filtering
                approved_issues = await self._apply_consensus(issues)
                if not approved_issues:
                    self.log.info("No issues passed consensus — continuing to next cycle")
                    continue

                # Fix → Review → Gate → Commit
                await self._phase_fix(approved_issues)
                await self._phase_review()
                await self._phase_gate()
                await self._phase_commit()

                # Incremental re-read of modified files
                modified = await self._get_modified_files()
                if modified:
                    await self.run_read_phase(incremental=True, force_reread=modified)
                    await self._build_graph()

        except Exception as exc:
            self.log.exception(f"Stabilizer fatal error: {exc}")
            await self._finalise(RunStatus.FAILED)
            return RunStatus.FAILED

        finally:
            if self._patrol_task and not self._patrol_task.done():
                self._patrol_task.cancel()
            if self.vector_brain:
                self.vector_brain.close()
            if self.storage:
                await self.storage.close()

    # ── Phase: Read ───────────────────────────────────────────────────────────

    async def run_read_phase(
        self,
        incremental:  bool = True,
        force_reread: set[str] | None = None,
    ) -> None:
        assert self.run and self.storage
        a_cfg = _agent_cfg(self.config, self.run.id)

        reader = ReaderAgent(
            storage=self.storage,
            run_id=self.run.id,
            repo_root=self.config.repo_root,
            config=a_cfg,
            mcp_manager=self.mcp,
            incremental=incremental,
            concurrency=self.config.concurrency,
            chunk_concurrency=self.config.chunk_concurrency,
            vector_brain=self.vector_brain,
        )
        await reader.run(force_reread=force_reread)
        await self._trail("READ_PHASE_COMPLETE", self.run.id, "run")

    # ── Phase: Graph Build ────────────────────────────────────────────────────

    async def _build_graph(self) -> None:
        if not self.config.graph_enabled or not self.storage or not self.run:
            return
        self.log.info("Building dependency graph…")
        await self.graph_engine.build(self.storage)
        self.run.graph_built = True
        await self.storage.upsert_run(self.run)
        summary = self.graph_engine.summary()
        self.log.info(
            f"Graph: {summary['nodes']} nodes, {summary['edges']} edges, "
            f"{summary['cycles']} cycles"
        )

    # ── Phase: Audit ──────────────────────────────────────────────────────────

    async def _phase_audit(self) -> list[Issue]:
        assert self.run and self.storage
        a_cfg = _agent_cfg(self.config, self.run.id)

        auditors = [
            AuditorAgent(
                storage=self.storage,
                run_id=self.run.id,
                executor_type=executor,
                master_prompt_path=self.config.master_prompt_path,
                config=a_cfg,
                mcp_manager=self.mcp,
                domain_mode=self.config.domain_mode,
                repo_root=self.config.repo_root,
                validate_findings=self.config.validate_findings,
            )
            for executor in (
                ExecutorType.SECURITY,
                ExecutorType.ARCHITECTURE,
                ExecutorType.STANDARDS,
            )
        ]

        results = await asyncio.gather(*[a.run() for a in auditors], return_exceptions=True)
        all_issues: list[Issue] = []
        for r in results:
            if isinstance(r, list):
                all_issues.extend(r)
            elif isinstance(r, Exception):
                self.log.error(f"Auditor failed: {r}")

        await self._trail("AUDIT_PHASE_COMPLETE", self.run.id, "run",
                          after=f"{len(all_issues)} issues found")
        return all_issues

    # ── Phase: Consensus ──────────────────────────────────────────────────────

    async def _apply_consensus(self, issues: list[Issue]) -> list[Issue]:
        assert self.consensus and self.storage
        results   = self.consensus.evaluate_issues(issues)
        approved  = self.consensus.filter_approved(issues, results)
        summary   = self.consensus.summary(results)

        self.log.info(
            f"Consensus: {summary['approved']}/{summary['total']} passed, "
            f"{summary['escalated']} escalated, "
            f"mean_conf={summary['mean_confidence']:.2f}"
        )

        # Persist updated consensus fields
        for issue in issues:
            await self.storage.upsert_issue(issue)

        return approved

    # ── Phase: Fix ────────────────────────────────────────────────────────────

    async def _phase_fix(self, issues: list[Issue]) -> None:
        assert self.run and self.storage
        a_cfg = _agent_cfg(self.config, self.run.id)

        fixer = FixerAgent(
            storage=self.storage,
            run_id=self.run.id,
            config=a_cfg,
            mcp_manager=self.mcp,
            repo_root=self.config.repo_root,
            graph_engine=self.graph_engine if self.config.graph_enabled else None,
            vector_brain=self.vector_brain,
        )
        await fixer.run()

    # ── Phase: Review ─────────────────────────────────────────────────────────

    async def _phase_review(self) -> None:
        assert self.run and self.storage
        a_cfg = _agent_cfg(self.config, self.run.id)

        reviewer = ReviewerAgent(
            storage=self.storage,
            run_id=self.run.id,
            config=a_cfg,
            mcp_manager=self.mcp,
            cross_validate_critical=True,
            cross_file_coherence=True,
            repo_root=self.config.repo_root,
        )
        await reviewer.run()

    # ── Phase: Gate ───────────────────────────────────────────────────────────

    async def _phase_gate(self) -> None:
        assert self.run and self.storage
        gate = StaticAnalysisGate(
            run_ruff=self.config.run_ruff,
            run_mypy=self.config.run_mypy,
            run_semgrep=self.config.run_semgrep,
            run_bandit=self.config.run_bandit,
            repo_root=self.config.repo_root,
            domain_mode=self.config.domain_mode.value,
        )

        planner = PlannerAgent(
            storage=self.storage,
            run_id=self.run.id,
            config=_agent_cfg(self.config, self.run.id),
        )

        formal_agent = FormalVerifierAgent(
            storage=self.storage,
            run_id=self.run.id,
            domain_mode=self.config.domain_mode,
            config=_agent_cfg(self.config, self.run.id),
            repo_root=self.config.repo_root,
        ) if (
            self.config.formal_verification
            and self.config.domain_mode != DomainMode.GENERAL
        ) else None

        approved_fixes = await self.storage.list_fixes()

        for fix in approved_fixes:
            if fix.reviewer_verdict and fix.reviewer_verdict.value != "APPROVED":
                continue
            if fix.gate_passed is not None:
                continue

            # Static gate
            gate_passed   = True
            gate_reason   = ""
            for ff in fix.fixed_files:
                result = await gate.validate(ff.path, ff.content)
                if not result.approved:
                    gate_passed = False
                    gate_reason = f"{ff.path}: {result.rejection_reason}"
                    break

            # Planner consequence check
            if gate_passed:
                planner_result = await planner.run(fix_attempt_id=fix.id)
                if planner_result.block_commit:
                    gate_passed = False
                    gate_reason = f"Planner blocked: {planner_result.reason}"

            # Formal verification (CRITICAL only in non-GENERAL domains)
            if gate_passed and formal_agent:
                critical_issues = [
                    await self.storage.get_issue(iid)
                    for iid in fix.issue_ids
                ]
                has_critical = any(
                    i and i.severity == Severity.CRITICAL
                    for i in critical_issues
                )
                if has_critical:
                    fv_results = await formal_agent.verify_fix(fix)
                    if await formal_agent.any_counterexample(fv_results):
                        ce = next(
                            r.counterexample for r in fv_results
                            if r.status == FormalVerificationStatus.COUNTEREXAMPLE
                        )
                        gate_passed = False
                        gate_reason = f"Formal verification failed: {ce[:300]}"
                        fix.formal_proofs = [r.id for r in fv_results]

            fix.gate_passed = gate_passed
            fix.gate_reason = gate_reason
            await self.storage.upsert_fix(fix)

            if gate_passed:
                self.log.info(f"Gate: PASSED for fix {fix.id[:8]}")
            else:
                self.log.warning(f"Gate: BLOCKED for fix {fix.id[:8]}: {gate_reason}")
                # Re-open issues so they get re-queued next cycle
                for iid in fix.issue_ids:
                    await self.storage.update_issue_status(
                        iid, IssueStatus.OPEN.value, reason=gate_reason
                    )

    # ── Phase: Commit ─────────────────────────────────────────────────────────

    async def _phase_commit(self) -> None:
        assert self.run and self.storage
        if not self.config.auto_commit or self.config.autonomy_level == AutonomyLevel.READ_ONLY:
            return

        fixes = await self.storage.list_fixes()
        to_commit = [
            f for f in fixes
            if f.gate_passed is True and f.committed_at is None
        ]

        if not to_commit:
            return

        # Group fixes by "module" (top-level directory) for batched PRs — GAP-13
        module_groups: dict[str, list] = {}
        for fix in to_commit:
            module = self._module_for_fix(fix)
            module_groups.setdefault(module, []).append(fix)

        for module, module_fixes in module_groups.items():
            await self._commit_module_group(module, module_fixes)

        # Requeue transitive dependents using real graph — GAP-1/18
        committed_files: set[str] = set()
        for fix in to_commit:
            for ff in fix.fixed_files:
                committed_files.add(ff.path)

        await self._requeue_transitive_dependents(committed_files)

    def _module_for_fix(self, fix: object) -> str:
        """GAP-13: derive module name from the files a fix touches."""
        paths = [ff.path for ff in fix.fixed_files]  # type: ignore[attr-defined]
        if not paths:
            return "misc"
        # Use top-level directory as module key
        parts = paths[0].split("/")
        return parts[0] if len(parts) > 1 else "root"

    async def _commit_module_group(self, module: str, fixes: list) -> None:
        assert self.run and self.storage
        combined_files: list[tuple[str, str]] = []
        all_issue_ids: list[str] = []

        for fix in fixes:
            for ff in fix.fixed_files:
                # Write to disk
                try:
                    from sandbox.executor import validate_path_within_root
                    validate_path_within_root(ff.path, self.config.repo_root)
                    abs_path = (self.config.repo_root / ff.path).resolve()
                    abs_path.parent.mkdir(parents=True, exist_ok=True)
                    abs_path.write_text(ff.content, encoding="utf-8")
                    combined_files.append((ff.path, ff.content))
                    all_issue_ids.extend(fix.issue_ids)
                    self.log.info(f"Committed: {ff.path}")
                except Exception as exc:
                    self.log.error(f"Failed to write {ff.path}: {exc}")

            fix.committed_at = datetime.now(tz=timezone.utc)
            await self.storage.upsert_fix(fix)

            # Close resolved issues
            for iid in fix.issue_ids:
                await self.storage.update_issue_status(iid, IssueStatus.CLOSED.value)

            # Audit trail
            await self._trail(
                "FIX_COMMITTED", fix.id, "fix",
                after=f"files={[ff.path for ff in fix.fixed_files]}"
            )

            # Run tests
            if self.config.run_tests_after_fix:
                test_runner = TestRunnerAgent(
                    storage=self.storage,
                    run_id=self.run.id,
                    repo_root=self.config.repo_root,
                )
                test_result = await test_runner.run_after_fix(fix)
                if test_result.status == TestRunStatus.FAILED:
                    self.log.warning(
                        f"Tests FAILED after fix {fix.id[:8]}: "
                        f"{test_result.failed} failed / "
                        f"{test_result.errors} errors"
                    )
                    await self._log_patrol_event(
                        "TEST_REGRESSION",
                        f"Tests failed after fix {fix.id[:8]}",
                        action="Manual review recommended",
                        severity="WARNING",
                    )

        # Create batched PR
        if self.pr_manager and combined_files:
            try:
                branch = (
                    f"{self.config.branch_prefix}/{module}/"
                    f"{self.run.id[:8]}-c{self.run.cycle_count}"
                )
                pr_url = await self.pr_manager.create_pr(
                    branch_name=branch,
                    files=combined_files,
                    title=f"[MACS] {module}: fix {len(all_issue_ids)} issues (cycle {self.run.cycle_count})",
                    body=self._build_pr_body(fixes),
                )
                for fix in fixes:
                    fix.pr_url = pr_url
                    await self.storage.upsert_fix(fix)
                self.log.info(f"PR created: {pr_url}")
            except Exception as exc:
                self.log.error(f"PR creation failed: {exc}")

    def _build_pr_body(self, fixes: list) -> str:
        lines = [
            "## MACS Automated Fix\n\n",
            f"Domain mode: `{self.config.domain_mode.value}`\n\n",
            "### Issues Fixed\n\n",
        ]
        for fix in fixes:
            for iid in fix.issue_ids:
                lines.append(f"- {iid}\n")
        lines.append("\n### Verification\n\n")
        lines.append("- ✅ Static analysis gate (ruff / mypy / bandit / semgrep)\n")
        lines.append("- ✅ Planner consequence assessment\n")
        lines.append("- ✅ Multi-model adversarial review\n")
        if any(fix.formal_proofs for fix in fixes):
            lines.append("- ✅ Formal verification (Z3)\n")
        return "".join(lines)

    # ── Transitive re-queue (GAP-1/18 FIX) ───────────────────────────────────

    async def _requeue_transitive_dependents(self, changed_files: set[str]) -> None:
        assert self.storage
        if not self.graph_engine.is_built:
            return

        affected: set[str] = set()
        for path in changed_files:
            affected |= self.graph_engine.impact_radius(path)
        affected -= changed_files  # don't re-queue the fixed files themselves

        if not affected:
            return

        self.log.info(
            f"Requeuing {len(affected)} transitive dependent files for re-read"
        )
        # Mark as UNREAD so the next read phase picks them up
        for path in affected:
            record = await self.storage.get_file(path)
            if record:
                from brain.schemas import FileStatus
                record.status = FileStatus.UNREAD
                await self.storage.upsert_file(record)

    async def _get_modified_files(self) -> set[str]:
        assert self.storage and self.run
        fixes = await self.storage.list_fixes()
        modified: set[str] = set()
        for fix in fixes:
            if fix.committed_at:
                modified |= {ff.path for ff in fix.fixed_files}
        return modified

    # ── Score recording ───────────────────────────────────────────────────────

    async def _record_score(self, issues: list[Issue]) -> AuditScore:
        assert self.run and self.storage
        from collections import Counter
        counts = Counter(i.severity for i in issues)
        score  = AuditScore(
            run_id=self.run.id,
            critical_count=counts.get(Severity.CRITICAL, 0),
            major_count=counts.get(Severity.MAJOR, 0),
            minor_count=counts.get(Severity.MINOR, 0),
            info_count=counts.get(Severity.INFO, 0),
        )
        score.compute_score()
        await self.storage.append_score(score)
        self.run.scores.append(score)
        self.log.info(
            f"Score: {score.score:.0f} "
            f"(C={score.critical_count} M={score.major_count} m={score.minor_count})"
        )
        return score

    # ── Audit trail ───────────────────────────────────────────────────────────

    async def _trail(
        self,
        event_type:    str,
        entity_id:     str,
        entity_type:   str,
        before:        str = "",
        after:         str = "",
    ) -> None:
        assert self.storage and self.run and self._trail_signer
        entry = AuditTrailEntry(
            run_id=self.run.id,
            event_type=event_type,
            entity_id=entity_id,
            entity_type=entity_type,
            before_state=before,
            after_state=after,
            actor="MACS",
        )
        entry.hmac_signature = self._trail_signer.sign(entry.model_dump_json())
        await self.storage.append_audit_trail(entry)

    async def _log_patrol_event(
        self, event_type: str, detail: str, action: str = "", severity: str = "INFO"
    ) -> None:
        if not self.storage or not self.run:
            return
        event = PatrolEvent(
            event_type=event_type,
            detail=detail,
            action_taken=action,
            run_id=self.run.id,
            severity=severity,
        )
        await self.storage.log_patrol_event(event)

    # ── Finalise ──────────────────────────────────────────────────────────────

    async def _finalise(self, status: RunStatus) -> None:
        if self.run and self.storage:
            await self.storage.update_run_status(self.run.id, status)
            await self._trail("RUN_FINALISED", self.run.id, "run", after=status.value)
            self.log.info(f"Run {self.run.id[:8]} finalised: {status.value}")

    # ── Public helpers ────────────────────────────────────────────────────────

    async def run_audit_phase(self) -> AuditScore:
        """Exposed for the CLI's audit-only command."""
        issues = await self._phase_audit()
        return await self._record_score(issues)
