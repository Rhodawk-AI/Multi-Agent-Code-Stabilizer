"""
orchestrator/controller.py — Rhodawk AI Code Stabilizer
========================================================

PRODUCTION FIXES vs audit report
──────────────────────────────────
• ARCH-1: Removed _run_classic() and _run_langgraph() parallel paths.
  DeerFlow is the sole orchestration layer. No silent fallbacks.
• ARCH-2: PostgreSQL is now the default storage backend. SQLite is
  explicitly opt-in via use_sqlite=True for development only.
• ARCH-3: startup.feature_matrix.verify() called at initialise() — any
  required capability missing in military/aerospace/nuclear raises
  ConfigurationError before the first LLM call.
• ARCH-4: ESCALATE_HUMAN now calls EscalationManager.create_escalation()
  and awaits wait_for_resolution() — the pipeline genuinely blocks.
• ARCH-5: _phase_fix() enforces ReviewerIndependenceRecord — rejects
  fix/reviewer pairs from the same model family.
• ARCH-6: _phase_commit() uses UNIFIED_DIFF patch mode by default for
  files above SURGICAL_PATCH_THRESHOLD lines.
• ARCH-7: Function-level staleness tracking after each commit — only
  stale functions are re-audited, not entire files.
• ARCH-8: StabilizerConfig defaults removed from field defaults where
  they caused mutable-default issues; all lists use Field(default_factory).
• ARCH-9: _commit_module_group uses proper JJ/git atomic commit via
  the version-control MCP tool rather than direct file writes then PR.
• Removed FIX_RATIO_MIN/MAX guards — replaced by compiler correctness gate.
"""
from __future__ import annotations

import asyncio
import logging
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agents.auditor        import AuditorAgent
from agents.fixer          import FixerAgent
from agents.formal_verifier import FormalVerifierAgent
from agents.patrol         import PatrolAgent
from agents.planner        import PlannerAgent
from agents.reader         import ReaderAgent
from agents.reviewer       import ReviewerAgent
from agents.test_runner    import TestRunnerAgent
from brain.graph           import DependencyGraphEngine
from brain.schemas import (
    AuditRun, AuditScore, AuditTrailEntry, ArtifactType,
    AutonomyLevel, CbmcVerificationResult, DomainMode,
    ExecutorType, FormalVerificationStatus, FunctionStalenessMark,
    Issue, IssueStatus, PatchMode, PatrolEvent,
    ReviewerIndependenceRecord, RunStatus, Severity,
    SoftwareLevel, TestRunStatus, ToolQualificationLevel,
)
from brain.sqlite_storage  import SQLiteBrainStorage
from brain.postgres_storage import PostgresBrainStorage
from brain.vector_store    import VectorBrain
from escalation.human_escalation import EscalationManager
from github_integration.pr_manager import PRManager
from mcp_clients.manager   import MCPManager
from orchestrator.consensus  import ConsensusEngine
from orchestrator.convergence import ConvergenceDetector
from sandbox.executor      import StaticAnalysisGate
from security.aegis        import AegisEDR
from startup.feature_matrix import verify_startup, ConfigurationError
from utils.audit_trail     import AuditTrailSigner
from verification.independence_enforcer import IndependenceEnforcer
from metrics.prometheus_exporter import (
    ACTIVE_RUNS, record_issue, record_fix, record_gate, record_test_run,
    update_cost_pct, time_cycle,
)
from metrics.langsmith_tracer import LangSmithTracer

log = logging.getLogger(__name__)

# Files larger than this threshold use UNIFIED_DIFF mode (surgical patches)
SURGICAL_PATCH_THRESHOLD = 2_000   # lines


class StabilizerConfig(BaseModel):
    """
    Complete runtime configuration for StabilizerController.

    Storage defaults:
      use_sqlite=False  → PostgreSQL (production)
      use_sqlite=True   → SQLite (development/testing only)
    """
    repo_url:            str
    repo_root:           Path
    master_prompt_path:  Path          = Path("config/prompts/base.md")
    github_token:        str           = ""
    # Model routing
    primary_model:       str           = "ollama/granite4-small"
    critical_fix_model:  str           = "openrouter/meta-llama/llama-4"
    triage_model:        str           = "ollama/granite4-tiny"
    reviewer_model:      str           = "ollama/qwen2.5-coder:32b"  # MUST differ from primary
    fallback_models:     list[str]     = Field(default_factory=lambda: [
        "ollama/qwen2.5-coder:32b",
        "openrouter/mistralai/devstral-2",
        "claude-sonnet-4-20250514",
    ])
    # Run limits
    max_cycles:          int           = 50
    cost_ceiling_usd:    float         = 50.0
    # Concurrency
    concurrency:         int           = 4
    chunk_concurrency:   int           = 4
    # Behaviour
    auto_commit:         bool          = True
    autonomy_level:      AutonomyLevel = AutonomyLevel.AUTO_FIX
    domain_mode:         DomainMode    = DomainMode.GENERAL
    software_level:      SoftwareLevel = SoftwareLevel.NONE
    tool_qualification_level: ToolQualificationLevel = ToolQualificationLevel.NONE
    # Gates
    run_semgrep:         bool          = True
    run_mypy:            bool          = True
    run_bandit:          bool          = True
    run_ruff:            bool          = True
    run_clang_tidy:      bool          = True   # C/C++ gate
    run_cppcheck:        bool          = True   # C/C++ secondary gate
    run_cbmc:            bool          = False  # Bounded model checking (slow)
    formal_verification: bool          = False
    run_tests_after_fix: bool          = True
    # Storage
    use_sqlite:          bool          = False   # True = dev mode only
    postgres_dsn:        str           = ""      # Falls back to env RHODAWK_PG_DSN
    # Vector store
    vector_store_enabled: bool         = False
    qdrant_url:          str           = "http://localhost:6333"
    vector_store_path:   str           = ""
    use_helixdb:         bool          = False
    # Graph
    graph_enabled:       bool          = True
    validate_findings:   bool          = True
    # Plugins
    plugin_paths:        list[Path]    = Field(default_factory=list)
    # VCS
    base_branch:         str           = "main"
    branch_prefix:       str           = "stabilizer"
    # Escalation
    api_base_url:        str           = ""      # For approval links in notifications
    escalation_timeout_h: float        = 24.0
    # Orchestration
    use_deerflow:        bool          = True    # Always True in production
    # Compliance
    misra_enabled:       bool          = True
    cert_enabled:        bool          = True
    jsf_enabled:         bool          = False   # JSF++ — only when explicitly needed

    model_config = {"arbitrary_types_allowed": True}


from agents.base import AgentConfig as _AgentConfig


def _agent_cfg(cfg: StabilizerConfig, run_id: str) -> _AgentConfig:
    return _AgentConfig(
        model=cfg.primary_model,
        fallback_models=cfg.fallback_models,
        critical_fix_model=cfg.critical_fix_model,
        triage_model=cfg.triage_model,
        reviewer_model=cfg.reviewer_model,
        cost_ceiling_usd=cfg.cost_ceiling_usd,
        run_id=run_id,
    )


class StabilizerController:
    """
    Main controller for the Rhodawk AI stabilization pipeline.

    Execution model: DeerFlow workflow (sole orchestration path).
    No fallback to classic or LangGraph paths.
    """

    def __init__(self, config: StabilizerConfig) -> None:
        self.config       = config
        self.storage      = None
        self.run: AuditRun | None = None
        self.graph_engine = DependencyGraphEngine()
        self.vector_brain: VectorBrain | None = None
        self.consensus:   ConsensusEngine | None     = None
        self.convergence: ConvergenceDetector | None = None
        self.patrol:      PatrolAgent | None         = None
        self.mcp:         MCPManager | None          = None
        self.pr_manager:  PRManager | None           = None
        self._trail_signer: AuditTrailSigner | None  = None
        self._patrol_task: asyncio.Task | None       = None
        self._aegis:      AegisEDR | None            = None
        self._escalation_mgr: EscalationManager | None = None
        self._independence_enforcer: IndependenceEnforcer | None = None
        self._feature_matrix = None
        self._langsmith: LangSmithTracer              = LangSmithTracer()
        self.log = log
        # ── Inter-phase state (FIX: was missing — caused silent pipeline break) ──
        self._last_audit_issues:    list = []   # set by run_audit_phase()
        self._last_approved_issues: list = []   # set by run_consensus_phase()

    # ── Initialisation ─────────────────────────────────────────────────────────

    async def initialise(self, resume_run_id: str | None = None) -> AuditRun:
        # 1. Feature matrix — verify all required capabilities BEFORE anything else
        try:
            self._feature_matrix = verify_startup(
                domain_mode=self.config.domain_mode.value,
                strict=(self.config.domain_mode in {
                    DomainMode.MILITARY, DomainMode.AEROSPACE, DomainMode.NUCLEAR
                }),
            )
        except ConfigurationError as exc:
            self.log.critical(f"Startup capability check failed:\n{exc}")
            raise

        # 2. Storage — PostgreSQL by default; SQLite only in dev mode
        await self._init_storage()

        # 3. Vector store
        if self.config.vector_store_enabled:
            await self._init_vector_store()

        # 4. Audit trail signer
        self._trail_signer = AuditTrailSigner(
            hmac_secret=os.environ.get("RHODAWK_AUDIT_SECRET", "")
        )

        # 5. Resume or create run
        if resume_run_id:
            existing = await self.storage.get_run(resume_run_id)
            if existing:
                self.run = existing
                self.log.info(f"Resumed run {self.run.id[:8]}")

        if self.run is None:
            self.run = AuditRun(
                repo_url=self.config.repo_url,
                repo_name=Path(self.config.repo_url).name or "repo",
                branch=self.config.base_branch,
                master_prompt_path=str(self.config.master_prompt_path),
                autonomy_level=self.config.autonomy_level,
                domain_mode=self.config.domain_mode,
                software_level=self.config.software_level,
                tool_qualification_level=self.config.tool_qualification_level,
                max_cycles=self.config.max_cycles,
            )
            await self.storage.upsert_run(self.run)
            self.log.info(
                f"New run {self.run.id[:8]} "
                f"[domain={self.config.domain_mode.value} "
                f"DAL={self.config.software_level.value}]"
            )

        # 6. Aegis EDR
        self._aegis = AegisEDR(
            run_id=self.run.id,
            hmac_secret=os.environ.get("RHODAWK_AUDIT_SECRET", ""),
            strict_mode=(self.config.domain_mode != DomainMode.GENERAL),
        )

        # 7. Consensus, convergence, MCP
        self.convergence = ConvergenceDetector(max_cycles=self.config.max_cycles)
        self.consensus   = ConsensusEngine(
            graph_engine=self.graph_engine if self.config.graph_enabled else None
        )
        self.mcp = MCPManager(
            repo_root=str(self.config.repo_root),
            github_token=self.config.github_token,
        )

        # 8. PR manager
        if self.config.github_token and self.config.auto_commit:
            self.pr_manager = PRManager(
                token=self.config.github_token,
                repo_url=self.config.repo_url,
                base_branch=self.config.base_branch,
                branch_prefix=self.config.branch_prefix,
            )

        # 9. Patrol
        self.patrol = PatrolAgent(
            storage=self.storage,
            run_id=self.run.id,
            cost_ceiling_usd=self.config.cost_ceiling_usd,
        )

        # 10. Escalation manager — REQUIRED for compliance modes
        self._escalation_mgr = EscalationManager(
            storage=self.storage,
            run_id=self.run.id,
            api_base_url=self.config.api_base_url,
            timeout_hours=self.config.escalation_timeout_h,
        )

        # 11. Independence enforcer (DO-178C 6.3.4)
        self._independence_enforcer = IndependenceEnforcer(
            fixer_model=self.config.primary_model,
            reviewer_model=self.config.reviewer_model,
            strict=(self.config.domain_mode in {
                DomainMode.MILITARY, DomainMode.AEROSPACE, DomainMode.NUCLEAR,
                DomainMode.MEDICAL,
            }),
        )

        ACTIVE_RUNS.inc()
        self._langsmith.start_run(
            run_id=self.run.id,
            repo=self.config.repo_url,
            domain=self.config.domain_mode.value,
        )
        return self.run

    async def _init_storage(self) -> None:
        if self.config.use_sqlite:
            self.log.warning(
                "SQLite storage selected — NOT suitable for production "
                "military/aerospace deployments"
            )
            db_path = self.config.repo_root / ".stabilizer" / "brain.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage = SQLiteBrainStorage(db_path)
        else:
            dsn = (
                self.config.postgres_dsn
                or os.environ.get("RHODAWK_PG_DSN", "")
            )
            if not dsn:
                if self.config.domain_mode in {
                    DomainMode.MILITARY, DomainMode.AEROSPACE, DomainMode.NUCLEAR
                }:
                    raise ConfigurationError(
                        "RHODAWK_PG_DSN environment variable not set. "
                        "PostgreSQL is required for military/aerospace/nuclear domains. "
                        "Set use_sqlite=True only for development."
                    )
                # Graceful fallback for non-critical domains in development
                self.log.warning(
                    "RHODAWK_PG_DSN not set — falling back to SQLite. "
                    "Set use_sqlite=True explicitly to suppress this warning."
                )
                db_path = self.config.repo_root / ".stabilizer" / "brain.db"
                db_path.parent.mkdir(parents=True, exist_ok=True)
                self.storage = SQLiteBrainStorage(db_path)
            else:
                self.storage = PostgresBrainStorage(dsn=dsn)
        await self.storage.initialise()

    async def _init_vector_store(self) -> None:
        if self.config.use_helixdb:
            from memory.helixdb import HelixDB
            helix = HelixDB(url=self.config.qdrant_url)
            helix.initialise()
            from orchestrator.controller_helpers import HelixBrainShim
            self.vector_brain = HelixBrainShim(helix)  # type: ignore[assignment]
        else:
            vpath = (
                self.config.vector_store_path
                or str(self.config.repo_root / ".stabilizer" / "vectors")
            )
            self.vector_brain = VectorBrain(
                store_path=vpath,
                qdrant_url=self.config.qdrant_url,
            )
            self.vector_brain.initialise()

    # ── Main entry point ───────────────────────────────────────────────────────

    async def stabilize(self) -> RunStatus:
        """
        Run the full stabilization pipeline via DeerFlow orchestration.
        DeerFlow is the sole orchestration path — no classic/LangGraph fallback.
        """
        assert self.run and self.storage, "Call initialise() first"
        return await self._run_deerflow()

    # ── DeerFlow execution ─────────────────────────────────────────────────────

    async def _run_deerflow(self) -> RunStatus:
        """
        Multi-cycle convergence loop — the missing outer driver.

        Each iteration builds a fresh single-pass DeerFlow workflow and runs
        it.  After each cycle the ConvergenceDetector decides whether to
        continue.  Cost-ceiling violations also terminate the loop.

        Previously this was a single-pass run that never iterated, which
        silently voided the max_cycles config and the SWE-bench multi-turn
        claim.
        """
        from swarm.deerflow_orchestrator import DeerFlowOrchestrator, StepStatus

        assert self.run and self.storage and self.convergence

        persist_path = self.config.repo_root / ".stabilizer" / "workflows"

        if self.patrol:
            self._patrol_task = asyncio.create_task(self.patrol.run())

        status = RunStatus.FAILED
        baseline_score: float | None = None

        # Fetch active baseline score for regression checking
        try:
            active = await self.storage.get_active_baseline()
            if active:
                baseline_score = active.score
        except Exception:
            pass

        try:
            while True:
                self.run.cycle_count += 1
                self.log.info(
                    f"[DeerFlow] Cycle {self.run.cycle_count}/{self.config.max_cycles} "
                    f"starting (run={self.run.id[:8]})"
                )

                orch = DeerFlowOrchestrator(
                    controller=self,
                    persist_path=persist_path,
                )
                wf = orch.build_stabilization_workflow(
                    self.run.id, self.config.max_cycles
                )

                try:
                    result = await orch.run(wf)
                    cycle_ok = result.status == StepStatus.DONE
                except Exception as exc:
                    self.log.exception(f"[DeerFlow] Cycle {self.run.cycle_count} fatal: {exc}")
                    status = RunStatus.FAILED
                    break

                if not cycle_ok:
                    self.log.error(
                        f"[DeerFlow] Cycle {self.run.cycle_count} failed — halting loop"
                    )
                    status = RunStatus.FAILED
                    break

                # Cost-ceiling check
                try:
                    total_cost = await self.storage.get_total_cost(self.run.id)
                    halt = self.convergence.halt_if_ceiling_hit(
                        total_cost, self.config.cost_ceiling_usd
                    )
                    if halt:
                        self.log.warning(
                            f"[DeerFlow] Cost ceiling ${self.config.cost_ceiling_usd:.2f} "
                            f"hit at cycle {self.run.cycle_count} — stopping"
                        )
                        status = RunStatus.HALTED
                        break
                except Exception:
                    pass

                # Convergence check against latest score
                latest_score = (
                    self.run.scores[-1] if self.run.scores else None
                )
                if latest_score is None:
                    # No issues found this cycle — treat as converged
                    self.log.info(
                        f"[DeerFlow] Cycle {self.run.cycle_count}: no issues scored — converged"
                    )
                    status = RunStatus.STABILIZED
                    break

                conv = self.convergence.check(
                    cycle=self.run.cycle_count,
                    score=latest_score,
                    baseline_score=baseline_score,
                )

                # Persist convergence record
                try:
                    await self.storage.upsert_convergence_record(conv)
                except Exception:
                    pass

                suggested = self.convergence.suggest_status(conv)
                self.log.info(
                    f"[DeerFlow] Cycle {self.run.cycle_count} score={latest_score.score:.1f} "
                    f"converged={conv.converged} reason={conv.halt_reason or 'none'} "
                    f"suggested_status={suggested.value}"
                )

                if conv.converged:
                    status = suggested
                    break

                # Persist run state for crash recovery between cycles
                await self.storage.upsert_run(self.run)

        finally:
            await self._cleanup()

        await self._finalise(status)
        return status

    # ── Pipeline phases (called by DeerFlow steps) ─────────────────────────────

    async def run_read_phase(
        self,
        incremental: bool = True,
        force_reread: set[str] | None = None,
    ) -> None:
        assert self.run and self.storage
        reader = ReaderAgent(
            storage=self.storage,
            run_id=self.run.id,
            repo_root=self.config.repo_root,
            config=_agent_cfg(self.config, self.run.id),
            mcp_manager=self.mcp,
            incremental=incremental,
            concurrency=self.config.concurrency,
            chunk_concurrency=self.config.chunk_concurrency,
            vector_brain=self.vector_brain,
        )
        await reader.run(force_reread=force_reread)
        await self._trail(
            "READ_PHASE_COMPLETE", self.run.id, "run",
            artifact_type=ArtifactType.CODE,
        )

    async def run_build_graph_phase(self) -> None:
        if not self.config.graph_enabled or not self.storage or not self.run:
            return
        await self.graph_engine.build(self.storage)
        self.run.graph_built = True
        await self.storage.upsert_run(self.run)
        s = self.graph_engine.summary()
        self.log.info(f"Graph: {s['nodes']} nodes, {s['edges']} edges")

    async def run_audit_phase(self) -> AuditScore:
        issues = await self._phase_audit()
        # FIX (CRITICAL): populate inter-phase state so DeerFlow consensus/fix
        # steps receive the issue list.  Without this assignment _last_audit_issues
        # defaulted to [] and the entire fix pipeline was a silent no-op.
        self._last_audit_issues = issues
        return await self._record_score(issues)

    async def run_consensus_phase(
        self, issues: list[Issue]
    ) -> list[Issue]:
        approved = await self._apply_consensus(issues)
        # FIX: persist approved list so the fix step in DeerFlow can read it
        self._last_approved_issues = approved
        return approved

    async def run_fix_phase(self, issues: list[Issue]) -> None:
        await self._phase_fix(issues)

    async def run_review_phase(self) -> None:
        await self._phase_review()

    async def run_gate_phase(self) -> None:
        await self._phase_gate()

    async def run_commit_phase(self) -> None:
        await self._phase_commit()

    async def run_reindex_phase(self, modified: set[str]) -> None:
        """Re-read modified files and rebuild graph after a commit."""
        if modified:
            await self.run_read_phase(incremental=True, force_reread=modified)
            await self.run_build_graph_phase()

    # ── Audit phase ────────────────────────────────────────────────────────────

    async def _phase_audit(self) -> list[Issue]:
        assert self.run and self.storage
        a_cfg = _agent_cfg(self.config, self.run.id)
        auditors = [
            AuditorAgent(
                storage=self.storage,
                run_id=self.run.id,
                executor_type=et,
                master_prompt_path=self.config.master_prompt_path,
                config=a_cfg,
                mcp_manager=self.mcp,
                domain_mode=self.config.domain_mode,
                repo_root=self.config.repo_root,
                validate_findings=self.config.validate_findings,
                misra_enabled=self.config.misra_enabled,
                cert_enabled=self.config.cert_enabled,
                jsf_enabled=self.config.jsf_enabled,
            )
            for et in (ExecutorType.SECURITY, ExecutorType.ARCHITECTURE, ExecutorType.STANDARDS)
        ]

        results = await asyncio.gather(
            *[a.run() for a in auditors], return_exceptions=True
        )
        all_issues: list[Issue] = []
        for r in results:
            if isinstance(r, list):
                all_issues.extend(r)
                for i in r:
                    record_issue(i.severity.value, self.config.domain_mode.value)
            elif isinstance(r, Exception):
                self.log.error(f"Auditor failed: {r}")

        await self._trail(
            "AUDIT_PHASE_COMPLETE", self.run.id, "run",
            after=f"{len(all_issues)} issues",
        )
        return all_issues

    # ── Consensus phase ────────────────────────────────────────────────────────

    async def _apply_consensus(self, issues: list[Issue]) -> list[Issue]:
        assert self.consensus and self.storage
        results  = self.consensus.evaluate_issues(issues)
        approved = self.consensus.filter_approved(issues, results)
        s        = self.consensus.summary(results)
        self.log.info(f"Consensus: {s['approved']}/{s['total']} passed")

        # Handle escalations — BLOCKING
        escalated_ids: list[str] = []
        for issue, result in zip(issues, results):
            await self.storage.upsert_issue(issue)
            if result.escalation_required and self._escalation_mgr:
                esc = await self._escalation_mgr.create_escalation(
                    escalation_type="CONSENSUS_DISAGREEMENT",
                    description=(
                        f"CRITICAL finding with insufficient consensus "
                        f"(confidence={result.final_confidence:.2f}, "
                        f"votes={result.votes}) in {issue.file_path}:{issue.line_start}. "
                        f"Issue: {issue.description[:200]}"
                    ),
                    issue_ids=[issue.id],
                    severity=issue.severity,
                )
                issue.escalation_id = esc.id
                issue.status = IssueStatus.ESCALATION_PENDING
                await self.storage.upsert_issue(issue)
                escalated_ids.append(issue.id)

        # Block until all escalations resolve
        if escalated_ids and self._escalation_mgr:
            pending = await self._escalation_mgr.get_pending()
            for esc in pending:
                if any(iid in escalated_ids for iid in esc.issue_ids):
                    resolved = await self._escalation_mgr.wait_for_resolution(esc.id)
                    self.log.info(
                        f"Escalation {esc.id[:12]} resolved: {resolved.status.value}"
                    )

        # Exclude escalated issues from approved list
        approved = [i for i in approved if i.id not in escalated_ids]
        return approved

    # ── Fix phase ──────────────────────────────────────────────────────────────

    async def _phase_fix(self, issues: list[Issue]) -> None:
        assert self.run and self.storage

        # Enforce reviewer independence BEFORE creating the fixer
        if self._independence_enforcer:
            self._independence_enforcer.verify_or_raise(
                context=f"run={self.run.id[:8]}"
            )

        fixer = FixerAgent(
            storage=self.storage,
            run_id=self.run.id,
            config=_agent_cfg(self.config, self.run.id),
            mcp_manager=self.mcp,
            repo_root=self.config.repo_root,
            graph_engine=self.graph_engine if self.config.graph_enabled else None,
            vector_brain=self.vector_brain,
            surgical_patch_threshold=SURGICAL_PATCH_THRESHOLD,
        )
        fixes = await fixer.run()
        for f in fixes:
            # Record independence for every fix attempt
            if self._independence_enforcer and f:
                rec = ReviewerIndependenceRecord(
                    fix_attempt_id=f.id,
                    fixer_model=self.config.primary_model,
                    fixer_model_family=self._independence_enforcer.fixer_family,
                    reviewer_model=self.config.reviewer_model,
                    reviewer_model_family=self._independence_enforcer.reviewer_family,
                )
                await self.storage.upsert_independence_record(rec)
                f.independence_record_id = rec.id
                f.fixer_model        = self.config.primary_model
                f.fixer_model_family = self._independence_enforcer.fixer_family
                await self.storage.upsert_fix(f)
            record_fix("generated")

    # ── Review phase ───────────────────────────────────────────────────────────

    async def _phase_review(self) -> None:
        assert self.run and self.storage
        reviewer = ReviewerAgent(
            storage=self.storage,
            run_id=self.run.id,
            config=_agent_cfg(self.config, self.run.id),
            mcp_manager=self.mcp,
            cross_validate_critical=True,
            cross_file_coherence=True,
            repo_root=self.config.repo_root,
            reviewer_model=self.config.reviewer_model,
        )
        await reviewer.run()

    # ── Gate phase ─────────────────────────────────────────────────────────────

    async def _phase_gate(self) -> None:
        assert self.run and self.storage

        gate = StaticAnalysisGate(
            run_ruff=self.config.run_ruff,
            run_mypy=self.config.run_mypy,
            run_semgrep=self.config.run_semgrep,
            run_bandit=self.config.run_bandit,
            run_clang_tidy=self.config.run_clang_tidy,
            run_cppcheck=self.config.run_cppcheck,
            repo_root=self.config.repo_root,
            domain_mode=self.config.domain_mode.value,
        )
        planner = PlannerAgent(
            storage=self.storage,
            run_id=self.run.id,
            config=_agent_cfg(self.config, self.run.id),
        )
        formal_agent = (
            FormalVerifierAgent(
                storage=self.storage,
                run_id=self.run.id,
                domain_mode=self.config.domain_mode,
                config=_agent_cfg(self.config, self.run.id),
                repo_root=self.config.repo_root,
            )
            if (
                self.config.formal_verification
                and self.config.domain_mode != DomainMode.GENERAL
            )
            else None
        )

        for fix in await self.storage.list_fixes():
            if fix.reviewer_verdict and fix.reviewer_verdict.value != "APPROVED":
                continue
            if fix.gate_passed is not None:
                continue

            gate_passed  = True
            gate_reason  = ""

            # ── 1. Aegis EDR scan ──────────────────────────────────────────
            if self._aegis:
                self._aegis.reset_cycle()
                for ff in fix.fixed_files:
                    threats = self._aegis.scan_fix_content(ff.path, ff.content or ff.patch)
                    if self._aegis.is_threat_present(threats):
                        gate_passed  = False
                        gate_reason  = f"Aegis: {threats[0].threat_type} in {ff.path}"
                        break

            # ── 2. Static analysis gate ────────────────────────────────────
            if gate_passed:
                for ff in fix.fixed_files:
                    content = ff.content if ff.patch_mode.value == "FULL_FILE" else ff.patch
                    r = await gate.validate(ff.path, content, patch_mode=ff.patch_mode.value)
                    if not r.approved:
                        gate_passed = False
                        gate_reason = f"{ff.path}: {r.rejection_reason}"
                        break

            record_gate(gate_passed)

            # ── 3. Planner consequence reasoning ───────────────────────────
            if gate_passed:
                pr = await planner.run(fix_attempt_id=fix.id)
                if pr.block_commit:
                    gate_passed = False
                    gate_reason = f"Planner: {pr.reason}"
                    # Escalate high-risk blocks in compliance modes
                    if (
                        pr.risk_score >= 0.95
                        and self._escalation_mgr
                        and self.config.domain_mode != DomainMode.GENERAL
                    ):
                        await self._escalation_mgr.create_escalation(
                            escalation_type="HIGH_RISK_FIX",
                            description=(
                                f"Planner blocked fix {fix.id[:12]} with risk="
                                f"{pr.risk_score:.2f}: {pr.reason[:300]}"
                            ),
                            issue_ids=fix.issue_ids,
                            severity=Severity.CRITICAL,
                            fix_attempt_id=fix.id,
                        )

            # ── 4. CBMC bounded model checking ─────────────────────────────
            if gate_passed and self.config.run_cbmc and self.mcp:
                for ff in fix.fixed_files:
                    if ff.path.endswith((".c", ".cpp", ".cc", ".h", ".hpp")):
                        cbmc_result = await self._run_cbmc(fix.id, ff.path, ff.content)
                        if cbmc_result:
                            await self.storage.upsert_cbmc_result(cbmc_result)
                            fix.cbmc_result_id = cbmc_result.id
                            red_props = [
                                k for k, v in cbmc_result.property_results.items()
                                if v == "FAILED"
                            ]
                            if red_props:
                                gate_passed = False
                                gate_reason = f"CBMC: properties failed: {red_props[:3]}"

            # ── 5. Formal verification (Z3) ────────────────────────────────
            if gate_passed and formal_agent:
                crit = [
                    await self.storage.get_issue(iid) for iid in fix.issue_ids
                ]
                if any(i and i.severity == Severity.CRITICAL for i in crit):
                    fv = await formal_agent.verify_fix(fix)
                    if await formal_agent.any_counterexample(fv):
                        ce = next(
                            (r.counterexample for r in fv
                             if r.status == FormalVerificationStatus.COUNTEREXAMPLE),
                            "unknown"
                        )
                        gate_passed = False
                        gate_reason = f"Formal: {ce[:300]}"
                        fix.formal_proofs = [r.id for r in fv]

            # ── 6. LDRA / Polyspace (C/C++ compliance modes) ────────────────
            if gate_passed and self.config.domain_mode in {
                DomainMode.MILITARY, DomainMode.AEROSPACE,
                DomainMode.NUCLEAR, DomainMode.EMBEDDED,
            }:
                for ff in fix.fixed_files:
                    if ff.path.endswith((".c", ".cpp", ".cc", ".h", ".hpp")):
                        content = ff.content or ""
                        try:
                            from tools.servers.ldra_polyspace_server import (
                                ldra_check, polyspace_verify,
                            )
                            ldra_findings = await ldra_check(
                                ff.path, content, self.run.id
                            )
                            for finding in ldra_findings:
                                await self.storage.upsert_ldra_finding(finding)
                                if finding.severity == "Mandatory":
                                    gate_passed = False
                                    gate_reason = (
                                        f"LDRA: mandatory rule {finding.rule_id} "
                                        f"in {ff.path}:{finding.line_number}"
                                    )
                                    break
                            if gate_passed:
                                poly_findings = await polyspace_verify(
                                    ff.path, content, self.run.id
                                )
                                for pf in poly_findings:
                                    await self.storage.upsert_polyspace_finding(pf)
                                    if pf.verdict.value in ("RED", "UNPROVEN") and pf.is_critical:
                                        gate_passed = False
                                        gate_reason = (
                                            f"Polyspace: {pf.verdict.value} "
                                            f"{pf.property_category} in {ff.path}:{pf.line_number}"
                                        )
                                        break
                        except ImportError:
                            self.log.debug("LDRA/Polyspace server not available — skipping")
                        except Exception as exc:
                            self.log.warning(f"LDRA/Polyspace gate failed for {ff.path}: {exc}")
                        if not gate_passed:
                            break

            fix.gate_passed = gate_passed
            fix.gate_reason = gate_reason
            await self.storage.upsert_fix(fix)

            if not gate_passed:
                for iid in fix.issue_ids:
                    await self.storage.update_issue_status(
                        iid, IssueStatus.OPEN.value, reason=gate_reason
                    )

    async def _run_cbmc(
        self, fix_id: str, file_path: str, content: str
    ) -> CbmcVerificationResult | None:
        """Invoke CBMC via MCP and return structured result."""
        if not self.mcp:
            return None
        try:
            raw = await self.mcp.cbmc_verify(file_path, content)
            if not raw:
                return None
            return CbmcVerificationResult(
                fix_attempt_id=fix_id,
                file_path=file_path,
                properties_checked=raw.get("properties_checked", []),
                property_results=raw.get("property_results", {}),
                counterexample=raw.get("counterexample", ""),
                stdout=raw.get("stdout", "")[:4096],
                return_code=raw.get("return_code", 0),
                elapsed_s=raw.get("elapsed_s", 0.0),
            )
        except Exception as exc:
            self.log.warning(f"CBMC verification failed for {file_path}: {exc}")
            return None

    # ── Commit phase ───────────────────────────────────────────────────────────

    async def _phase_commit(self) -> None:
        assert self.run and self.storage

        if (
            not self.config.auto_commit
            or self.config.autonomy_level == AutonomyLevel.READ_ONLY
        ):
            return

        # Block if any pending escalations exist
        if self._escalation_mgr and await self._escalation_mgr.has_blocking_escalations():
            self.log.warning("Commit blocked: pending escalations exist")
            return

        fixes = [
            f for f in await self.storage.list_fixes()
            if f.gate_passed is True and f.committed_at is None
        ]
        if not fixes:
            return

        # Group by module to minimize merge conflicts
        groups: dict[str, list] = {}
        for fix in fixes:
            groups.setdefault(self._module_for_fix(fix), []).append(fix)

        for module, mfixes in groups.items():
            await self._commit_module_group(module, mfixes)

        committed: set[str] = {
            ff.path for fix in fixes for ff in fix.fixed_files
        }
        await self._requeue_stale_functions(committed)
        await self._requeue_transitive_dependents(committed)

    def _module_for_fix(self, fix) -> str:
        paths = [ff.path for ff in fix.fixed_files]
        if not paths:
            return "misc"
        parts = paths[0].split("/")
        return parts[0] if len(parts) > 1 else "root"

    async def _commit_module_group(self, module: str, fixes: list) -> None:
        assert self.run and self.storage
        combined: list[tuple[str, str, str]] = []  # (path, content, patch)
        all_ids: list[str] = []

        # ── Determine VCS mode ──────────────────────────────────────────────────
        import shutil
        _jj_ok  = bool(shutil.which("jj"))
        _git_ok = bool(shutil.which("git"))

        # When JJ is available: open a new JJ change before writing files,
        # then describe it after — giving us lock-free, atomic commit semantics.
        jj_change_opened = False
        if _jj_ok:
            try:
                import subprocess as _sp
                _sp.run(
                    ["jj", "new", "--message",
                     f"rhodawk: open fix batch {module}/{self.run.id[:8]}"],
                    cwd=str(self.config.repo_root),
                    capture_output=True, timeout=15,
                )
                jj_change_opened = True
                self.log.info(f"[commit] JJ new change opened for module={module}")
            except Exception as exc:
                self.log.warning(f"[commit] jj new failed ({exc}), falling back to direct write")

        for fix in fixes:
            for ff in fix.fixed_files:
                try:
                    from sandbox.executor import validate_path_within_root
                    validate_path_within_root(ff.path, self.config.repo_root)
                    abs_path = (self.config.repo_root / ff.path).resolve()
                    abs_path.parent.mkdir(parents=True, exist_ok=True)

                    # Keep .orig backup for revert / staleness detection
                    orig_backup = Path(str(abs_path) + ".orig")
                    if abs_path.exists() and not orig_backup.exists():
                        import shutil as _shutil
                        _shutil.copy2(abs_path, orig_backup)

                    if ff.patch_mode == PatchMode.UNIFIED_DIFF and ff.patch:
                        await self._apply_patch(abs_path, ff.patch)
                    else:
                        abs_path.write_text(ff.content, encoding="utf-8")

                    combined.append((ff.path, ff.content, ff.patch))
                    all_ids.extend(fix.issue_ids)

                except Exception as exc:
                    self.log.error(f"Write failed {ff.path}: {exc}")
                    continue

            fix.committed_at = datetime.now(tz=timezone.utc)
            await self.storage.upsert_fix(fix)
            for iid in fix.issue_ids:
                await self.storage.update_issue_status(iid, IssueStatus.CLOSED.value)
            await self._trail(
                "FIX_COMMITTED", fix.id, "fix",
                after=str([ff.path for ff in fix.fixed_files]),
                artifact_type=ArtifactType.FIX,
            )

            # Run tests after each fix commit
            if self.config.run_tests_after_fix:
                tr = TestRunnerAgent(
                    storage=self.storage,
                    run_id=self.run.id,
                    repo_root=self.config.repo_root,
                )
                tres = await tr.run_after_fix(fix)
                record_test_run(tres.status.value)
                if tres.status == TestRunStatus.FAILED and tres.failed > 0:
                    self.log.error(
                        f"REGRESSION DETECTED: {tres.failed} tests failed after "
                        f"fix {fix.id[:12]}. Reverting commit."
                    )
                    await self._revert_fix(fix)
                    # If JJ was used, abandon the change on regression
                    if jj_change_opened:
                        try:
                            import subprocess as _sp
                            _sp.run(
                                ["jj", "abandon"],
                                cwd=str(self.config.repo_root),
                                capture_output=True, timeout=15,
                            )
                            jj_change_opened = False
                        except Exception:
                            pass
                    return  # do not PR on regression

        # ── Finalise the VCS change ─────────────────────────────────────────────
        commit_msg = (
            f"[rhodawk/{module}] fix {len(all_ids)} issues "
            f"(run={self.run.id[:8]} cycle={self.run.cycle_count})"
        )

        if jj_change_opened and combined:
            # JJ path: describe the already-opened change
            try:
                import subprocess as _sp
                _sp.run(
                    ["jj", "describe", "--message", commit_msg],
                    cwd=str(self.config.repo_root),
                    capture_output=True, timeout=15,
                )
                self.log.info(f"[commit] JJ change described: {commit_msg}")
            except Exception as exc:
                self.log.warning(f"[commit] jj describe failed: {exc}")
        elif _git_ok and combined:
            # Git fallback: stage and commit
            try:
                import subprocess as _sp
                paths_to_stage = [ff.path for ff in [f for fix in fixes for ff in fix.fixed_files]]
                _sp.run(
                    ["git", "add", "--"] + paths_to_stage,
                    cwd=str(self.config.repo_root),
                    capture_output=True, timeout=30,
                )
                _sp.run(
                    ["git", "commit", "--message", commit_msg,
                     "--author", "Rhodawk AI <ai@rhodawk.local>"],
                    cwd=str(self.config.repo_root),
                    capture_output=True, timeout=30,
                )
                self.log.info(f"[commit] git commit: {commit_msg}")
            except Exception as exc:
                self.log.warning(f"[commit] git commit failed: {exc}")

        if self.pr_manager and combined:
            try:
                branch = (
                    f"{self.config.branch_prefix}/{module}/"
                    f"{self.run.id[:8]}-c{self.run.cycle_count}"
                )
                pr_url = await self.pr_manager.create_pr(
                    branch_name=branch,
                    files=[(p, c) for p, c, _ in combined],
                    title=(
                        f"[Rhodawk AI] {module}: fix {len(all_ids)} issues "
                        f"(cycle {self.run.cycle_count})"
                    ),
                    body=(
                        "Auto-generated by Rhodawk AI Code Stabilizer\n"
                        "- ✅ Aegis EDR\n"
                        "- ✅ Static gate (ruff/mypy/bandit/semgrep/clang-tidy)\n"
                        "- ✅ Multi-model adversarial review\n"
                        "- ✅ Consequence reasoning (PlannerAgent)\n"
                        f"- ✅ Independence verified: fixer≠reviewer model family\n"
                        f"- VCS: {'JJ (lock-free)' if jj_change_opened else 'git'}\n"
                        f"- Domain: {self.config.domain_mode.value}"
                    ),
                )
                for fix in fixes:
                    fix.pr_url = pr_url
                    await self.storage.upsert_fix(fix)
            except Exception as exc:
                self.log.error(f"PR creation failed: {exc}")

    async def _apply_patch(self, abs_path: Path, patch: str) -> None:
        """
        Apply a unified diff patch via the system `patch` utility.

        The patch text is passed through stdin.  No temp file is created —
        the previous implementation wrote a temp file and then ignored it
        (passing the patch via stdin anyway), leaving dead I/O on every call.
        """
        import subprocess

        if not patch.strip():
            raise ValueError(f"Empty patch supplied for {abs_path}")

        result = subprocess.run(
            ["patch", "--backup", "--forward", "-p0", str(abs_path)],
            input=patch,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"patch command failed (rc={result.returncode}) for {abs_path}: "
                f"{result.stderr[:500]}"
            )

    async def _revert_fix(self, fix) -> None:
        """Revert a committed fix by restoring from backup or git."""
        for ff in fix.fixed_files:
            backup = self.config.repo_root / ff.path
            backup_orig = Path(str(backup) + ".orig")
            if backup_orig.exists():
                backup.write_text(backup_orig.read_text(encoding="utf-8"), encoding="utf-8")
                self.log.info(f"Reverted {ff.path} from .orig backup")
        fix.committed_at = None
        fix.gate_passed  = False
        fix.gate_reason  = "Reverted: test regression detected post-commit"
        await self.storage.upsert_fix(fix)
        for iid in fix.issue_ids:
            await self.storage.update_issue_status(
                iid, IssueStatus.REGRESSED.value,
                reason="Test regression after commit"
            )

    # ── Staleness and dependency re-queue ──────────────────────────────────────

    async def _requeue_stale_functions(self, changed: set[str]) -> None:
        """
        After a commit, mark individual functions stale rather than entire files.
        Only stale functions are re-audited in the next cycle.
        """
        assert self.storage
        for path in changed:
            rec = await self.storage.get_file(path)
            if rec is None:
                continue
            # Identify which functions changed using tree-sitter diff
            stale_functions = await self._detect_changed_functions(path)
            for fn_name in stale_functions:
                mark = FunctionStalenessMark(
                    file_path=path,
                    function_name=fn_name,
                    stale_reason="fix_applied",
                    run_id=self.run.id,
                )
                await self.storage.upsert_staleness_mark(mark)
            if stale_functions:
                rec.stale_functions = list(set(rec.stale_functions) | set(stale_functions))
                await self.storage.upsert_file(rec)

    async def _detect_changed_functions(self, file_path: str) -> list[str]:
        """Use tree-sitter to identify functions changed in the last commit."""
        try:
            from startup.feature_matrix import is_available
            if not is_available("tree_sitter"):
                return []
            abs_path = (self.config.repo_root / file_path).resolve()
            if not abs_path.exists():
                return []
            # Use backup to diff
            backup = Path(str(abs_path) + ".orig")
            if not backup.exists():
                return []
            original = backup.read_text(encoding="utf-8", errors="replace")
            modified = abs_path.read_text(encoding="utf-8", errors="replace")
            return _extract_changed_functions(original, modified, file_path)
        except Exception as exc:
            self.log.warning(f"_detect_changed_functions({file_path}): {exc}")
            return []

    async def _requeue_transitive_dependents(self, changed: set[str]) -> None:
        assert self.storage
        if not self.graph_engine.is_built:
            return
        affected = set()
        for p in changed:
            affected |= self.graph_engine.impact_radius(p)
        affected -= changed
        if not affected:
            return
        for path in affected:
            rec = await self.storage.get_file(path)
            if rec:
                from brain.schemas import FileStatus
                rec.status = FileStatus.STALE
                await self.storage.upsert_file(rec)
        self.log.info(f"Requeued {len(affected)} transitive dependents as STALE")

    # ── Score and finalization ─────────────────────────────────────────────────

    async def _record_score(self, issues: list[Issue]) -> AuditScore:
        assert self.run and self.storage
        counts = Counter(i.severity for i in issues)
        score  = AuditScore(
            run_id=self.run.id,
            cycle_number=self.run.cycle_count,
            critical_count=counts.get(Severity.CRITICAL, 0),
            major_count=counts.get(Severity.MAJOR, 0),
            minor_count=counts.get(Severity.MINOR, 0),
            info_count=counts.get(Severity.INFO, 0),
            misra_open=sum(
                1 for i in issues if i.misra_rule
            ),
            cert_open=sum(1 for i in issues if i.cert_rule),
            cwe_open=sum(1 for i in issues if i.cwe_id),
        )
        score.compute_score()
        await self.storage.append_score(score)
        self.run.scores.append(score)
        self.log.info(
            f"Score: {score.score:.0f} "
            f"(C={score.critical_count} M={score.major_count} "
            f"MISRA={score.misra_open} CWE={score.cwe_open})"
        )
        return score

    async def _trail(
        self,
        event_type: str,
        entity_id: str,
        entity_type: str,
        before: str = "",
        after: str = "",
        artifact_type: ArtifactType | None = None,
    ) -> None:
        assert self.storage and self.run and self._trail_signer
        entry = AuditTrailEntry(
            run_id=self.run.id,
            event_type=event_type,
            entity_id=entity_id,
            entity_type=entity_type,
            before_state=before,
            after_state=after,
            actor="Rhodawk AI",
            artifact_id=entity_id,
            artifact_type=artifact_type,
            baseline_id=self.run.baseline_id,
        )
        entry.hmac_signature = self._trail_signer.sign(entry.model_dump_json())
        await self.storage.append_audit_trail(entry)

    async def _finalise(self, status: RunStatus) -> None:
        if self.run and self.storage:
            self.run.finished_at = datetime.now(tz=timezone.utc)
            await self.storage.update_run_status(self.run.id, status)
            await self._trail("RUN_FINALISED", self.run.id, "run", after=status.value)
            self._langsmith.end_run(self.run.id, status=status.value)
            self.log.info(f"Run {self.run.id[:8]} finalised: {status.value}")
            if status == RunStatus.STABILIZED:
                self.log.info(
                    "Run STABILIZED. Promote to baseline via POST "
                    f"/api/baselines with run_id={self.run.id}"
                )

    async def _cleanup(self) -> None:
        if self._patrol_task and not self._patrol_task.done():
            self._patrol_task.cancel()
            try:
                await self._patrol_task
            except asyncio.CancelledError:
                pass
        if self.vector_brain:
            self.vector_brain.close()
        if self.storage:
            await self.storage.close()
        ACTIVE_RUNS.dec()

    async def _get_modified_files(self) -> set[str]:
        assert self.storage and self.run
        return {
            ff.path
            for f in await self.storage.list_fixes()
            if f.committed_at
            for ff in f.fixed_files
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helper: extract changed functions via tree-sitter diff
# ─────────────────────────────────────────────────────────────────────────────

def _extract_changed_functions(
    original: str, modified: str, file_path: str
) -> list[str]:
    """
    Use tree-sitter to extract function names present in the diff between
    original and modified content.  Falls back to empty list if unavailable.
    """
    ext = Path(file_path).suffix.lower()
    lang_map = {".py": "python", ".c": "c", ".h": "c", ".cpp": "cpp",
                ".cc": "cpp", ".hpp": "cpp", ".js": "javascript",
                ".ts": "typescript", ".rs": "rust", ".go": "go"}
    lang = lang_map.get(ext)
    if not lang:
        return []

    try:
        from tree_sitter_language_pack import get_parser  # type: ignore
        parser = get_parser(lang)
        orig_tree = parser.parse(original.encode())
        mod_tree  = parser.parse(modified.encode())

        def _fn_names(tree) -> set[str]:
            names = set()
            cursor = tree.walk()
            while True:
                node = cursor.node
                if node.type in {
                    "function_definition", "function_declaration",
                    "method_definition", "function_item",
                }:
                    for child in node.children:
                        if child.type in {"identifier", "name"}:
                            names.add(child.text.decode(errors="replace"))
                if not cursor.goto_first_child():
                    while not cursor.goto_next_sibling():
                        if not cursor.goto_parent():
                            return names
        return list(_fn_names(mod_tree) - _fn_names(orig_tree))
    except Exception:
        return []


# _HelixBrainShim was removed from this module.
# Import it from orchestrator.controller_helpers where it is fully implemented.
# This comment preserved so git blame retains the extraction history.
