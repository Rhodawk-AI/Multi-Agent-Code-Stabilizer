"""
orchestrator/controller.py  —  Rhodawk AI v1.0
B1-B12 all fixed. New: TieredModelRouter, DeerFlow, LangGraph, AegisEDR, HelixDB, Prometheus.
"""
from __future__ import annotations
import asyncio, logging
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
    AuditRun, AuditScore, AuditTrailEntry, AutonomyLevel, DomainMode,
    ExecutorType, FormalVerificationStatus, Issue, IssueStatus,
    PatrolEvent, RunStatus, Severity, TestRunStatus,
)
from brain.sqlite_storage  import SQLiteBrainStorage
from brain.vector_store    import VectorBrain
from github_integration.pr_manager import PRManager
from mcp_clients.manager   import MCPManager
from orchestrator.consensus  import ConsensusEngine
from orchestrator.convergence import ConvergenceDetector
from sandbox.executor      import StaticAnalysisGate
from utils.audit_trail     import AuditTrailSigner
from security.aegis        import AegisEDR
from metrics.prometheus_exporter import (
    ACTIVE_RUNS, record_issue, record_fix, record_gate, record_test_run,
    update_cost_pct, time_cycle,
)
log = logging.getLogger(__name__)

class StabilizerConfig(BaseModel):
    repo_url:            str
    repo_root:           Path
    master_prompt_path:  Path          = Path("config/prompts/base.md")
    github_token:        str           = ""
    primary_model:       str           = "ollama/granite4-small"
    critical_fix_model:  str           = "openrouter/meta-llama/llama-4"
    triage_model:        str           = "ollama/granite4-tiny"
    fallback_models:     list[str]     = Field(default_factory=lambda: [
        "ollama/qwen2.5-coder:32b",
        "openrouter/mistralai/devstral-2",
        "claude-sonnet-4-20250514",
    ])
    max_cycles:          int           = 50
    cost_ceiling_usd:    float         = 50.0
    concurrency:         int           = 4
    chunk_concurrency:   int           = 4
    auto_commit:         bool          = True
    autonomy_level:      AutonomyLevel = AutonomyLevel.AUTO_FIX
    domain_mode:         DomainMode    = DomainMode.GENERAL
    run_semgrep:         bool          = True
    run_mypy:            bool          = True
    run_bandit:          bool          = True
    run_ruff:            bool          = True
    formal_verification: bool          = False
    run_tests_after_fix: bool          = True
    vector_store_enabled: bool         = False
    vector_store_path:   str           = ""
    graph_enabled:       bool          = True
    validate_findings:   bool          = True
    plugin_paths:        list[Path]    = Field(default_factory=list)
    base_branch:         str           = "main"
    branch_prefix:       str           = "stabilizer"
    use_helixdb:         bool          = False
    qdrant_url:          str           = "http://localhost:6333"
    use_deerflow:        bool          = True
    use_langgraph:       bool          = False
    model_config = {"arbitrary_types_allowed": True}

from agents.base import AgentConfig as _AgentConfig

def _agent_cfg(cfg: StabilizerConfig, run_id: str) -> _AgentConfig:
    return _AgentConfig(
        model=cfg.primary_model, fallback_models=cfg.fallback_models,
        critical_fix_model=cfg.critical_fix_model, triage_model=cfg.triage_model,
        cost_ceiling_usd=cfg.cost_ceiling_usd, run_id=run_id,
    )

class StabilizerController:
    def __init__(self, config: StabilizerConfig) -> None:
        self.config = config
        self.storage: SQLiteBrainStorage | None = None
        self.run: AuditRun | None = None
        self.graph_engine = DependencyGraphEngine()
        self.vector_brain: VectorBrain | None = None
        self.consensus: ConsensusEngine | None = None
        self.convergence: ConvergenceDetector | None = None
        self.patrol: PatrolAgent | None = None
        self.mcp: MCPManager | None = None
        self.pr_manager: PRManager | None = None
        self._trail_signer: AuditTrailSigner | None = None
        self._patrol_task: asyncio.Task | None = None
        self._aegis: AegisEDR | None = None
        self.log = log

    async def initialise(self, resume_run_id: str | None = None) -> AuditRun:
        import os
        db_path = self.config.repo_root / ".stabilizer" / "brain.db"
        self.storage = SQLiteBrainStorage(db_path)
        await self.storage.initialise()

        if self.config.vector_store_enabled:
            if self.config.use_helixdb:
                from memory.helixdb import HelixDB
                helix = HelixDB(url=self.config.qdrant_url)
                helix.initialise()
                self.vector_brain = _HelixBrainShim(helix)  # type: ignore
            else:
                vpath = self.config.vector_store_path or str(
                    self.config.repo_root / ".stabilizer" / "vectors")
                self.vector_brain = VectorBrain(store_path=vpath)
                self.vector_brain.initialise()

        self._trail_signer = AuditTrailSigner()

        if resume_run_id:
            run = await self.storage.get_run(resume_run_id)
            if run:
                self.run = run

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
            self.log.info(f"New run {self.run.id[:8]} [domain={self.config.domain_mode.value}]")

        # B11: Aegis EDR
        self._aegis = AegisEDR(
            run_id=self.run.id,
            hmac_secret=os.environ.get("RHODAWK_AUDIT_SECRET", ""),
            strict_mode=(self.config.domain_mode != DomainMode.GENERAL),
        )

        self.convergence = ConvergenceDetector(max_cycles=self.config.max_cycles)
        self.consensus   = ConsensusEngine(
            graph_engine=self.graph_engine if self.config.graph_enabled else None)
        self.mcp = MCPManager(repo_root=str(self.config.repo_root),
                              github_token=self.config.github_token)

        if self.config.github_token and self.config.auto_commit:
            self.pr_manager = PRManager(
                token=self.config.github_token, repo_url=self.config.repo_url,
                base_branch=self.config.base_branch, branch_prefix=self.config.branch_prefix)

        if self.storage and self.run:
            self.patrol = PatrolAgent(storage=self.storage, run_id=self.run.id,
                                      cost_ceiling_usd=self.config.cost_ceiling_usd)
        ACTIVE_RUNS.inc()
        return self.run

    async def stabilize(self) -> RunStatus:
        assert self.run and self.storage
        if self.config.use_deerflow:
            return await self._run_deerflow()
        if self.config.use_langgraph:
            return await self._run_langgraph()
        return await self._run_classic()

    async def _run_deerflow(self) -> RunStatus:
        from swarm.deerflow_orchestrator import DeerFlowOrchestrator, StepStatus
        orch = DeerFlowOrchestrator(
            controller=self,
            persist_path=self.config.repo_root / ".stabilizer" / "workflows")
        wf = orch.build_stabilization_workflow(self.run.id, self.config.max_cycles)
        if self.patrol:
            self._patrol_task = asyncio.create_task(self.patrol.run())
        try:
            result = await orch.run(wf)
            status = RunStatus.STABILIZED if result.status == StepStatus.DONE else RunStatus.FAILED
        except Exception as exc:
            self.log.exception(f"DeerFlow fatal: {exc}")
            status = RunStatus.FAILED
        finally:
            await self._cleanup()
        await self._finalise(status)
        return status

    async def _run_langgraph(self) -> RunStatus:
        from swarm.langgraph_state import build_stabilizer_graph, SwarmState
        graph = build_stabilizer_graph(self)
        if graph is None:
            return await self._run_classic()
        if self.patrol:
            self._patrol_task = asyncio.create_task(self.patrol.run())
        try:
            await graph.ainvoke(SwarmState(run_id=self.run.id,
                                           max_cycles=self.config.max_cycles).to_dict())
            status = RunStatus.STABILIZED
        except Exception as exc:
            self.log.exception(f"LangGraph fatal: {exc}")
            status = RunStatus.FAILED
        finally:
            await self._cleanup()
        await self._finalise(status)
        return status

    async def _run_classic(self) -> RunStatus:
        if self.patrol:
            self._patrol_task = asyncio.create_task(self.patrol.run())
        try:
            await self.run_read_phase(incremental=False)
            await self._build_graph()
            while True:
                self.run.cycle_count += 1
                await self.storage.upsert_run(self.run)
                self.log.info(f"═══ Cycle {self.run.cycle_count}/{self.config.max_cycles} ═══")
                with time_cycle():
                    issues = await self._phase_audit()
                    score  = await self._record_score(issues)
                terminal = self.convergence.check(score)
                if terminal:
                    await self._finalise(terminal)
                    return terminal
                if not issues:
                    await self._finalise(RunStatus.STABILIZED)
                    return RunStatus.STABILIZED
                approved = await self._apply_consensus(issues)
                if not approved:
                    continue
                await self._phase_fix(approved)
                await self._phase_review()
                await self._phase_gate()
                await self._phase_commit()
                modified = await self._get_modified_files()
                if modified:
                    await self.run_read_phase(incremental=True, force_reread=modified)
                    await self._build_graph()
                total_cost = await self.storage.get_total_cost(self.run.id)
                update_cost_pct(total_cost, self.config.cost_ceiling_usd)
        except Exception as exc:
            self.log.exception(f"Stabilizer fatal: {exc}")
            await self._finalise(RunStatus.FAILED)
            return RunStatus.FAILED
        finally:
            await self._cleanup()

    async def _cleanup(self) -> None:
        if self._patrol_task and not self._patrol_task.done():
            self._patrol_task.cancel()
        if self.vector_brain:
            self.vector_brain.close()
        if self.storage:
            await self.storage.close()
        ACTIVE_RUNS.dec()

    async def run_read_phase(self, incremental: bool = True,
                              force_reread: set[str] | None = None) -> None:
        assert self.run and self.storage
        reader = ReaderAgent(
            storage=self.storage, run_id=self.run.id,
            repo_root=self.config.repo_root,
            config=_agent_cfg(self.config, self.run.id),
            mcp_manager=self.mcp, incremental=incremental,
            concurrency=self.config.concurrency,
            chunk_concurrency=self.config.chunk_concurrency,
            vector_brain=self.vector_brain,
        )
        await reader.run(force_reread=force_reread)
        await self._trail("READ_PHASE_COMPLETE", self.run.id, "run")

    async def _build_graph(self) -> None:
        if not self.config.graph_enabled or not self.storage or not self.run:
            return
        await self.graph_engine.build(self.storage)
        self.run.graph_built = True
        await self.storage.upsert_run(self.run)
        s = self.graph_engine.summary()
        self.log.info(f"Graph: {s['nodes']} nodes, {s['edges']} edges")

    async def _phase_audit(self) -> list[Issue]:
        assert self.run and self.storage
        a_cfg = _agent_cfg(self.config, self.run.id)
        auditors = [
            AuditorAgent(
                storage=self.storage, run_id=self.run.id,
                executor_type=et,
                master_prompt_path=self.config.master_prompt_path,
                config=a_cfg, mcp_manager=self.mcp,
                domain_mode=self.config.domain_mode,
                repo_root=self.config.repo_root,
                validate_findings=self.config.validate_findings,
            )
            for et in (ExecutorType.SECURITY, ExecutorType.ARCHITECTURE, ExecutorType.STANDARDS)
        ]
        results = await asyncio.gather(*[a.run() for a in auditors], return_exceptions=True)
        all_issues: list[Issue] = []
        for r in results:
            if isinstance(r, list):
                all_issues.extend(r)
                for i in r:
                    record_issue(i.severity.value, self.config.domain_mode.value)
            elif isinstance(r, Exception):
                self.log.error(f"Auditor failed: {r}")
        await self._trail("AUDIT_PHASE_COMPLETE", self.run.id, "run",
                          after=f"{len(all_issues)} issues")
        return all_issues

    async def _apply_consensus(self, issues: list[Issue]) -> list[Issue]:
        assert self.consensus and self.storage
        results  = self.consensus.evaluate_issues(issues)
        approved = self.consensus.filter_approved(issues, results)
        s = self.consensus.summary(results)
        self.log.info(f"Consensus: {s['approved']}/{s['total']} passed")
        for issue in issues:
            await self.storage.upsert_issue(issue)
        return approved

    async def _phase_fix(self, issues: list[Issue]) -> None:
        assert self.run and self.storage
        fixer = FixerAgent(
            storage=self.storage, run_id=self.run.id,
            config=_agent_cfg(self.config, self.run.id),
            mcp_manager=self.mcp, repo_root=self.config.repo_root,
            graph_engine=self.graph_engine if self.config.graph_enabled else None,
            vector_brain=self.vector_brain,
        )
        fixes = await fixer.run()
        for f in fixes:
            record_fix("generated")

    async def _phase_review(self) -> None:
        assert self.run and self.storage
        reviewer = ReviewerAgent(
            storage=self.storage, run_id=self.run.id,
            config=_agent_cfg(self.config, self.run.id),
            mcp_manager=self.mcp,
            cross_validate_critical=True, cross_file_coherence=True,
            repo_root=self.config.repo_root,
        )
        await reviewer.run()

    async def _phase_gate(self) -> None:
        assert self.run and self.storage
        gate = StaticAnalysisGate(
            run_ruff=self.config.run_ruff, run_mypy=self.config.run_mypy,
            run_semgrep=self.config.run_semgrep, run_bandit=self.config.run_bandit,
            repo_root=self.config.repo_root, domain_mode=self.config.domain_mode.value,
        )
        planner = PlannerAgent(storage=self.storage, run_id=self.run.id,
                               config=_agent_cfg(self.config, self.run.id))
        formal_agent = FormalVerifierAgent(
            storage=self.storage, run_id=self.run.id,
            domain_mode=self.config.domain_mode,
            config=_agent_cfg(self.config, self.run.id),
            repo_root=self.config.repo_root,
        ) if (self.config.formal_verification and self.config.domain_mode != DomainMode.GENERAL) else None

        for fix in await self.storage.list_fixes():
            if fix.reviewer_verdict and fix.reviewer_verdict.value != "APPROVED":
                continue
            if fix.gate_passed is not None:
                continue
            gate_passed, gate_reason = True, ""

            # B11: Aegis EDR scan before any disk write
            if self._aegis:
                self._aegis.reset_cycle()
                for ff in fix.fixed_files:
                    threats = self._aegis.scan_fix_content(ff.path, ff.content)
                    if self._aegis.is_threat_present(threats):
                        gate_passed = False
                        gate_reason = f"Aegis: {threats[0].threat_type} in {ff.path}"
                        break

            if gate_passed:
                for ff in fix.fixed_files:
                    r = await gate.validate(ff.path, ff.content)
                    if not r.approved:
                        gate_passed, gate_reason = False, f"{ff.path}: {r.rejection_reason}"
                        break
            record_gate(gate_passed)

            if gate_passed:
                pr = await planner.run(fix_attempt_id=fix.id)
                if pr.block_commit:
                    gate_passed, gate_reason = False, f"Planner: {pr.reason}"

            if gate_passed and formal_agent:
                crit = [await self.storage.get_issue(iid) for iid in fix.issue_ids]
                if any(i and i.severity == Severity.CRITICAL for i in crit):
                    fv = await formal_agent.verify_fix(fix)
                    if await formal_agent.any_counterexample(fv):
                        ce = next(r.counterexample for r in fv
                                  if r.status == FormalVerificationStatus.COUNTEREXAMPLE)
                        gate_passed, gate_reason = False, f"Formal: {ce[:300]}"
                        fix.formal_proofs = [r.id for r in fv]

            fix.gate_passed, fix.gate_reason = gate_passed, gate_reason
            await self.storage.upsert_fix(fix)
            if not gate_passed:
                for iid in fix.issue_ids:
                    await self.storage.update_issue_status(
                        iid, IssueStatus.OPEN.value, reason=gate_reason)

    async def _phase_commit(self) -> None:
        assert self.run and self.storage
        if not self.config.auto_commit or self.config.autonomy_level == AutonomyLevel.READ_ONLY:
            return
        fixes = [f for f in await self.storage.list_fixes()
                 if f.gate_passed is True and f.committed_at is None]
        if not fixes:
            return
        groups: dict[str, list] = {}
        for fix in fixes:
            groups.setdefault(self._module_for_fix(fix), []).append(fix)
        for module, mfixes in groups.items():
            await self._commit_module_group(module, mfixes)
        committed: set[str] = {ff.path for fix in fixes for ff in fix.fixed_files}
        await self._requeue_transitive_dependents(committed)

    def _module_for_fix(self, fix) -> str:
        paths = [ff.path for ff in fix.fixed_files]
        if not paths: return "misc"
        parts = paths[0].split("/")
        return parts[0] if len(parts) > 1 else "root"

    async def _commit_module_group(self, module: str, fixes: list) -> None:
        assert self.run and self.storage
        combined, all_ids = [], []
        for fix in fixes:
            for ff in fix.fixed_files:
                try:
                    from sandbox.executor import validate_path_within_root
                    validate_path_within_root(ff.path, self.config.repo_root)
                    abs_path = (self.config.repo_root / ff.path).resolve()
                    abs_path.parent.mkdir(parents=True, exist_ok=True)
                    abs_path.write_text(ff.content, encoding="utf-8")
                    combined.append((ff.path, ff.content))
                    all_ids.extend(fix.issue_ids)
                except Exception as exc:
                    self.log.error(f"Write failed {ff.path}: {exc}")
            fix.committed_at = datetime.now(tz=timezone.utc)
            await self.storage.upsert_fix(fix)
            for iid in fix.issue_ids:
                await self.storage.update_issue_status(iid, IssueStatus.CLOSED.value)
            await self._trail("FIX_COMMITTED", fix.id, "fix",
                              after=str([ff.path for ff in fix.fixed_files]))
            if self.config.run_tests_after_fix:
                tr = TestRunnerAgent(storage=self.storage, run_id=self.run.id,
                                     repo_root=self.config.repo_root)
                tres = await tr.run_after_fix(fix)
                record_test_run(tres.status.value)

        if self.pr_manager and combined:
            try:
                branch = f"{self.config.branch_prefix}/{module}/{self.run.id[:8]}-c{self.run.cycle_count}"
                pr_url = await self.pr_manager.create_pr(
                    branch_name=branch, files=combined,
                    title=f"[Rhodawk AI] {module}: fix {len(all_ids)} issues (c{self.run.cycle_count})",
                    body="Auto-generated by Rhodawk AI\n- ✅ Aegis EDR\n- ✅ Static gate\n- ✅ Multi-model review",
                )
                for fix in fixes:
                    fix.pr_url = pr_url
                    await self.storage.upsert_fix(fix)
            except Exception as exc:
                self.log.error(f"PR creation failed: {exc}")

    async def _requeue_transitive_dependents(self, changed: set[str]) -> None:
        assert self.storage
        if not self.graph_engine.is_built: return
        affected = set()
        for p in changed:
            affected |= self.graph_engine.impact_radius(p)
        affected -= changed
        if not affected: return
        for path in affected:
            rec = await self.storage.get_file(path)
            if rec:
                from brain.schemas import FileStatus
                rec.status = FileStatus.UNREAD
                await self.storage.upsert_file(rec)

    async def _get_modified_files(self) -> set[str]:
        assert self.storage and self.run
        return {ff.path for f in await self.storage.list_fixes()
                if f.committed_at for ff in f.fixed_files}

    async def _record_score(self, issues: list[Issue]) -> AuditScore:
        assert self.run and self.storage
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
        self.log.info(f"Score: {score.score:.0f} (C={score.critical_count} M={score.major_count})")
        return score

    async def _trail(self, event_type: str, entity_id: str, entity_type: str,
                     before: str = "", after: str = "") -> None:
        assert self.storage and self.run and self._trail_signer
        entry = AuditTrailEntry(
            run_id=self.run.id, event_type=event_type, entity_id=entity_id,
            entity_type=entity_type, before_state=before, after_state=after,
            actor="Rhodawk AI",
        )
        entry.hmac_signature = self._trail_signer.sign(entry.model_dump_json())
        await self.storage.append_audit_trail(entry)

    async def _finalise(self, status: RunStatus) -> None:
        if self.run and self.storage:
            await self.storage.update_run_status(self.run.id, status)
            await self._trail("RUN_FINALISED", self.run.id, "run", after=status.value)
            self.log.info(f"Run {self.run.id[:8]} finalised: {status.value}")

    async def run_audit_phase(self) -> AuditScore:
        issues = await self._phase_audit()
        return await self._record_score(issues)


class _HelixBrainShim:
    """Adapts HelixDB to the VectorBrain interface."""
    def __init__(self, helix) -> None:
        self._helix = helix
    @property
    def is_available(self) -> bool: return self._helix.is_available
    def initialise(self) -> None: pass
    def close(self) -> None: self._helix.close()
    def index_chunk(self, chunk_id, file_path, line_start, line_end,
                    language, summary, observations) -> None:
        from memory.helixdb import HelixDocument
        self._helix.index(HelixDocument(
            id=chunk_id, file_path=file_path, line_start=line_start, line_end=line_end,
            language=language, content=" ".join(observations), summary=summary))
    def find_similar_to_issue(self, query: str, n: int = 8):
        results = self._helix.search(query, n=n)
        from brain.vector_store import VectorSearchResult
        return [VectorSearchResult(chunk_id=r.id, file_path=r.file_path,
            line_start=r.line_start, line_end=r.line_end, language=r.language,
            summary=r.summary, distance=1.0 - r.score) for r in results]
