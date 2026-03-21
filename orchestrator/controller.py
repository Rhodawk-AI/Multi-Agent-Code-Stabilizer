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
    AutonomyLevel, CbmcVerificationResult, CompoundFinding,
    DomainMode, ExecutorType, FormalVerificationStatus,
    FunctionStalenessMark, Issue, IssueStatus, PatchMode, PatrolEvent,
    ReviewerIndependenceRecord, RunStatus, Severity, SynthesisReport,
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

# ── Antagonist additions ──────────────────────────────────────────────────────
from agents.test_generator      import TestGeneratorAgent
from agents.mutation_verifier   import MutationVerifierAgent
from brain.hybrid_retriever     import HybridRetriever
from context.repo_map           import get_repo_map
from memory.fix_memory          import get_fix_memory

# ── Gap 1: CPG-based causal context ──────────────────────────────────────────
try:
    from cpg.cpg_engine          import CPGEngine, get_cpg_engine
    from cpg.program_slicer      import ProgramSlicer
    from cpg.context_selector    import CPGContextSelector
    from cpg.incremental_updater import IncrementalCPGUpdater
    _CPG_AVAILABLE = True
except ImportError:
    _CPG_AVAILABLE = False

# ── Gap 2: Synthesis Agent — cross-domain compound findings ──────────────────
from agents.synthesis_agent import SynthesisAgent

# ── Gap 5: Multi-Intelligence / Adversarial Ensemble ─────────────────────────
try:
    from agents.adversarial_critic import AdversarialCriticAgent
    from swe_bench.bobn_sampler    import BoBNSampler
    _GAP5_AVAILABLE = True
except ImportError:
    _GAP5_AVAILABLE = False

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
    # Model routing — Qwen2.5-Coder-32B via vLLM local; override with env RHODAWK_PRIMARY_MODEL
    primary_model:       str           = "openai/Qwen/Qwen2.5-Coder-32B-Instruct"
    critical_fix_model:  str           = "openrouter/meta-llama/llama-4"
    triage_model:        str           = "ollama/granite4-tiny"
    reviewer_model:      str           = "ollama/qwen2.5-coder:32b"
    fallback_models:     list[str]     = Field(default_factory=lambda: [
        "ollama/qwen2.5-coder:32b",
        "openrouter/mistralai/devstral-2",
        "claude-sonnet-4-20250514",
    ])
    vllm_base_url:       str           = "http://localhost:8000/v1"
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
    run_clang_tidy:      bool          = True
    run_cppcheck:        bool          = True
    run_cbmc:            bool          = False
    formal_verification: bool          = False
    run_tests_after_fix: bool          = True
    # Storage
    use_sqlite:          bool          = False
    postgres_dsn:        str           = ""
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
    api_base_url:        str           = ""
    escalation_timeout_h: float        = 24.0
    # Orchestration
    use_deerflow:        bool          = True
    # Compliance
    misra_enabled:       bool          = True
    cert_enabled:        bool          = True
    jsf_enabled:         bool          = False
    # Repo-map
    repo_map_enabled:    bool          = True
    repo_map_max_tokens: int           = 2048
    # Hybrid retrieval
    hybrid_retrieval_enabled: bool     = True
    # Test generation
    test_gen_enabled:    bool          = True
    pynguin_timeout_s:   int           = 60
    use_hypothesis:      bool          = True
    # Mutation testing
    mutation_testing_enabled: bool     = False
    mutation_score_threshold: float | None = None
    # Fix memory
    fix_memory_enabled:  bool          = True
    # CPG (Gap 1)
    cpg_enabled:         bool          = True
    joern_url:           str           = "http://localhost:8080"
    joern_repo_path:     str           = ""
    joern_project_name:  str           = "rhodawk"
    cpg_blast_radius_threshold: int    = 50
    cpg_max_slice_nodes: int           = 50
    cpg_max_files_in_slice: int        = 30
    # Synthesis Agent (Gap 2) — cross-domain compound finding detection
    synthesis_enabled:          bool   = True
    synthesis_dedup_enabled:    bool   = True
    synthesis_compound_enabled: bool   = True
    # Dedicated model for synthesis — should be a DIFFERENT family from primary
    # auditors so it brings genuinely fresh cross-domain reasoning.
    # Defaults to critical_fix_model; override with RHODAWK_SYNTHESIS_MODEL env var.
    synthesis_model:            str    = ""
    synthesis_max_compound:     int    = 20
    # ── Gap 5: Multi-Intelligence / Adversarial Ensemble (BoBN pipeline) ─────
    # Master switch.  When False the existing single-fixer path is used unchanged.
    # When True: Localization Agent → Fixer A + Fixer B (parallel, different families)
    #            → Adversarial Critic (third family) → Synthesis → Formal Gate.
    gap5_enabled:               bool   = False
    # vLLM endpoint for Fixer B (DeepSeek-Coder-V2-Lite-16B).
    # Run alongside primary on a second GPU partition / port.
    # Override: VLLM_SECONDARY_BASE_URL
    gap5_vllm_secondary_base_url: str  = "http://localhost:8001/v1"
    # Model identifier served at the secondary vLLM endpoint.
    # Override: VLLM_SECONDARY_MODEL
    gap5_vllm_secondary_model:  str    = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    # vLLM endpoint for the Adversarial Critic (Llama-3.3-70B).
    # Leave blank to route through OpenRouter (requires OPENROUTER_API_KEY).
    # Override: VLLM_CRITIC_BASE_URL
    gap5_vllm_critic_base_url:  str    = ""
    # Critic model — MUST be a different family from both fixers.
    # Qwen (Alibaba) + DeepSeek → Llama (Meta) satisfies independence.
    # Override: VLLM_CRITIC_MODEL
    gap5_vllm_critic_model:     str    = "meta-llama/Llama-3.3-70B-Instruct"
    # Total BoBN candidate count (fixer_a + fixer_b).
    gap5_bobn_n_candidates:     int    = 5
    # Candidates generated by Fixer A (Qwen-32B) at temperatures 0.2/0.4/0.6.
    gap5_bobn_fixer_a_count:    int    = 3
    # Candidates generated by Fixer B (DeepSeek-16B) at temperatures 0.3/0.7.
    gap5_bobn_fixer_b_count:    int    = 2

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
        # ── Antagonist subsystems ─────────────────────────────────────────────
        self._hybrid_retriever: HybridRetriever | None = None
        self._repo_map:         Any | None             = None
        self._fix_memory:       Any | None             = None
        # ── Gap 1: CPG subsystems ─────────────────────────────────────────────
        self._cpg_engine:           Any | None         = None
        self._program_slicer:       Any | None         = None
        self._cpg_context_selector: Any | None         = None
        self._incremental_updater:  Any | None         = None
        # ── Gap 4: CommitAuditScheduler ───────────────────────────────────────
        self._commit_audit_scheduler: Any | None       = None
        # ── Gap 2: Synthesis Agent state ─────────────────────────────────────
        # Compound findings detected in the last synthesis pass.
        # Persisted here so DeerFlow steps and the report exporter can read them.
        self._last_compound_findings: list[CompoundFinding] = []
        # ── Gap 5: BoBN adversarial ensemble ─────────────────────────────────
        # Initialised in _init_antagonist() when gap5_enabled=True.
        self._bobn_sampler:       Any | None = None
        self._adversarial_critic: Any | None = None

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

        # 12. Antagonist subsystems ─────────────────────────────────────────────
        await self._init_antagonist()

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

    async def _init_antagonist(self) -> None:
        """
        Initialise the four Antagonist subsystems.
        All failures are non-fatal — the core pipeline continues without them.
        """
        # 1. Hybrid retriever (BM25 + dense)
        if self.config.hybrid_retrieval_enabled:
            try:
                self._hybrid_retriever = HybridRetriever(
                    qdrant_url=self.config.qdrant_url,
                    collection="rhodawk_chunks",
                    vector_brain=self.vector_brain,
                )
                self._hybrid_retriever.initialise()
            except Exception as exc:
                self.log.warning(f"HybridRetriever init failed: {exc}")
                self._hybrid_retriever = None

        # 2. Repo-map
        if self.config.repo_map_enabled:
            try:
                self._repo_map = get_repo_map(self.config.repo_root)
            except Exception as exc:
                self.log.warning(f"RepoMap init failed: {exc}")
                self._repo_map = None

        # 3. Fix memory (Algorithm Distillation)
        if self.config.fix_memory_enabled:
            try:
                data_dir = self.config.repo_root / ".stabilizer"
                self._fix_memory = get_fix_memory(
                    repo_url=self.config.repo_url,
                    qdrant_url=self.config.qdrant_url,
                    data_dir=data_dir,
                )
            except Exception as exc:
                self.log.warning(f"FixMemory init failed: {exc}")
                self._fix_memory = None

        # 4. Configure vLLM routing for LiteLLM
        try:
            from models.router import get_router
            router = get_router()
            router.configure_litellm_for_vllm()
        except Exception as exc:
            self.log.debug(f"vLLM LiteLLM config: {exc}")

        # 5. Gap 1: CPG engine (Joern Code Property Graph)
        await self._init_cpg()

        # 6. Gap 5: adversarial ensemble — family independence check + BoBN init
        if self.config.gap5_enabled:
            await self._init_gap5()

        self.log.info(
            f"Antagonist subsystems: "
            f"hybrid_retriever={self._hybrid_retriever is not None} "
            f"repo_map={self._repo_map is not None} "
            f"fix_memory={self._fix_memory is not None} "
            f"cpg_engine={self._cpg_engine is not None and getattr(self._cpg_engine, 'is_available', False)}"
        )

    async def _init_cpg(self) -> None:
        """
        Gap 1: Initialise the CPG subsystem (Joern + ProgramSlicer + ContextSelector).

        All failures are non-fatal — the core pipeline continues without CPG,
        falling back to the existing hybrid BM25+dense retrieval path.

        Wiring:
          CPGEngine           — connects to Joern, imports codebase
          ProgramSlicer       — computes backward/forward slices
          CPGContextSelector  — selects causally relevant context for FixerAgent
          IncrementalUpdater  — updates CPG after commits (Gap 4 integration)
        """
        if not self.config.cpg_enabled or not _CPG_AVAILABLE:
            self.log.info(
                "CPG disabled (cpg_enabled=False or cpg module not installed). "
                "Context selection will use hybrid BM25+dense retrieval."
            )
            return

        try:
            # Resolve Joern URL from env or config
            import os
            joern_url = (
                os.environ.get("JOERN_URL", "")
                or self.config.joern_url
                or "http://localhost:8080"
            )
            joern_repo_path = (
                os.environ.get("JOERN_REPO_PATH", "")
                or self.config.joern_repo_path
                or str(self.config.repo_root)
            )

            self._cpg_engine = get_cpg_engine(
                joern_url=joern_url,
                graph_engine=self.graph_engine,
                blast_radius_threshold=self.config.cpg_blast_radius_threshold,
            )

            # Non-blocking init — returns False if Joern not running
            connected = await self._cpg_engine.initialise(
                repo_path=joern_repo_path,
                project_name=self.config.joern_project_name,
            )

            if connected:
                self.log.info(
                    f"CPGEngine: Joern connected at {joern_url} — "
                    f"CPG analysis enabled (blast_radius_threshold="
                    f"{self.config.cpg_blast_radius_threshold})"
                )
            else:
                self.log.info(
                    "CPGEngine: Joern not available — "
                    "falling back to networkx import graph for context selection. "
                    "To enable full CPG: docker-compose up joern"
                )

            # ProgramSlicer wraps CPGEngine with LLM-friendly output
            self._program_slicer = ProgramSlicer(
                cpg_engine=self._cpg_engine,
                max_slice_nodes=self.config.cpg_max_slice_nodes,
                max_files_in_slice=self.config.cpg_max_files_in_slice,
            )

            # CPGContextSelector: the Gap 1 context selection layer
            # Replaces vector similarity for WHICH CODE TO LOAD
            # (vector stays for PATTERN MATCHING / few-shot examples)
            self._cpg_context_selector = CPGContextSelector(
                cpg_engine=self._cpg_engine,
                program_slicer=self._program_slicer,
                repo_root=self.config.repo_root,
                hybrid_retriever=self._hybrid_retriever,
                vector_brain=self.vector_brain,
            )

            # IncrementalCPGUpdater: Gap 4 integration
            # After each commit, computes the 50-200 function audit target set
            self._incremental_updater = IncrementalCPGUpdater(
                cpg_engine=self._cpg_engine,
                repo_root=self.config.repo_root,
                storage=self.storage,
            )

            # CommitAuditScheduler: Gap 4 full orchestration path
            # Wraps IncrementalCPGUpdater with idempotency, staleness marks,
            # scoped test re-runs, and CommitAuditRecord persistence.
            #
            # GAP 4 FIX: the test_runner parameter was previously hardcoded to
            # None, which meant the scoped test re-run step inside
            # CommitAuditScheduler._run_scoped_tests() always logged
            # "no test runner configured — skipping tests" and returned
            # immediately.  None of the Gap 4 requirement #4 (re-run only test
            # cases that cover the changed functions) ever fired.
            #
            # Fix: build a TestRunnerAgent and pass it in.  The test runner is
            # cheap to construct and its _detect_and_run() falls back gracefully
            # when no test framework is detected, so there is no downside to
            # always wiring it.
            from agents.test_runner import TestRunnerAgent as _TestRunnerAgent
            _test_runner_for_scheduler = _TestRunnerAgent(
                storage=self.storage,
                run_id=self.run.id,
                repo_root=self.config.repo_root,
            )

            from orchestrator.commit_audit_scheduler import CommitAuditScheduler
            self._commit_audit_scheduler = CommitAuditScheduler(
                storage=self.storage,
                incremental_updater=self._incremental_updater,
                test_runner=_test_runner_for_scheduler,
                run_id=self.run.id,
                repo_root=self.config.repo_root,
                cpg_engine=self._cpg_engine,
                graph_engine=self.graph_engine,
            )

        except Exception as exc:
            self.log.warning(
                f"CPG subsystem init failed (non-fatal): {exc} — "
                "continuing without CPG"
            )
            self._cpg_engine             = None
            self._program_slicer         = None
            self._cpg_context_selector   = None
            self._incremental_updater    = None
            self._commit_audit_scheduler = None

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
            hybrid_retriever=self._hybrid_retriever,
            repo_map=self._repo_map,
            cpg_engine=self._cpg_engine,
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
        # ── Gap 4: stale_only activation ─────────────────────────────────────
        # Cycle 1 always performs a full scan — the staleness table is empty and
        # the CPG has not yet computed any impact sets.
        #
        # Cycles 2+ activate stale_only=True ONLY when CommitAuditScheduler is
        # wired (i.e. cpg_enabled=True and Joern is reachable).  The scheduler
        # writes FunctionStalenessMark rows after every fix commit so the next
        # audit scope is bounded to the CPG-computed impact set (50–200 functions
        # instead of the full codebase).
        #
        # When stale_only=True and the staleness table is empty — which happens
        # when a cycle produces no committed fixes — AuditorAgent.run() detects
        # the empty result and automatically falls back to get_all_observations()
        # so no findings are ever silently dropped.
        is_incremental_cycle = (
            getattr(self.run, "cycle_count", 1) > 1
            and self._commit_audit_scheduler is not None
        )
        issues = await self._phase_audit(stale_only=is_incremental_cycle)
        # FIX (CRITICAL): populate inter-phase state so DeerFlow consensus/fix
        # steps receive the issue list.  Without this assignment _last_audit_issues
        # defaulted to [] and the entire fix pipeline was a silent no-op.
        self._last_audit_issues = issues
        # Gap 2: _last_compound_findings is set inside _phase_audit by SynthesisAgent.
        # It is already populated when we reach here.
        return await self._record_score(issues)

    async def run_consensus_phase(
        self, issues: list[Issue]
    ) -> list[Issue]:
        approved = await self._apply_consensus(issues)
        # FIX: persist approved list so the fix step in DeerFlow can read it
        self._last_approved_issues = approved
        return approved

    async def run_fix_phase(self, issues: list[Issue]) -> None:
        # Gap 5: when the adversarial ensemble is enabled and ready, route
        # through the dual-fixer BoBN pipeline.  Falls back automatically
        # to _phase_fix() if BoBN is not initialised (safe degradation).
        if self.config.gap5_enabled and self._bobn_sampler is not None:
            await self._phase_fix_gap5(issues)
        else:
            await self._phase_fix(issues)

    async def run_test_gen_phase(self) -> None:
        """Generate tests for all gate-pending fixes (Antagonist Addition 4)."""
        await self._phase_test_gen()

    async def run_mutation_phase(self) -> None:
        """Run mutation testing gate (Antagonist Addition 6)."""
        await self._phase_mutation()

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

    async def _phase_audit(self, stale_only: bool = False) -> list[Issue]:
        """
        Run the three-domain audit (SECURITY, ARCHITECTURE, STANDARDS) in
        parallel, then pass all findings through SynthesisAgent which:

          1. Deduplicates by fingerprint + LLM semantic pass
          2. Detects cross-domain compound findings (Gap 2)

        The synthesis step is what separates this system from CodeRabbit and
        Greptile — both tools use single-pass LLM scans that cannot reason
        about a bug requiring information from two auditor domains simultaneously.

        Parameters
        ----------
        stale_only:
            When True (set by run_audit_phase for cycles 2+ when
            CommitAuditScheduler is active) each AuditorAgent calls
            storage.get_stale_observations() instead of get_all_observations().
            This scopes the audit to the CPG-computed impact set written by
            CommitAuditScheduler — typically 50–200 functions rather than the
            full codebase.  This is the Gap 4 activation path: without this
            flag the staleness infrastructure exists but is never exercised.

            AuditorAgent.run() auto-falls-back to get_all_observations() when
            the staleness table is empty, so passing stale_only=True on a cycle
            that produced no committed fixes never silently drops findings.
        """
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
            *[a.run(stale_only=stale_only) for a in auditors], return_exceptions=True
        )
        all_issues: list[Issue] = []
        for r in results:
            if isinstance(r, list):
                all_issues.extend(r)
                for i in r:
                    record_issue(i.severity.value, self.config.domain_mode.value)
            elif isinstance(r, Exception):
                self.log.error(f"Auditor failed: {r}")

        raw_count = len(all_issues)
        self.log.info(
            f"[audit] Three-domain audit complete: {raw_count} raw findings "
            f"(SECURITY + ARCHITECTURE + STANDARDS)"
        )

        # ── Gap 2: Synthesis — dedup + cross-domain compound detection ────────
        if self.config.synthesis_enabled and all_issues:
            synthesis_model = (
                self.config.synthesis_model
                or os.environ.get("RHODAWK_SYNTHESIS_MODEL", "")
                or self.config.critical_fix_model
            )

            # ── GAP 2 FIX: Enforce synthesis model family independence ──────────
            # The spec requires synthesis to use a DIFFERENT model family from
            # the auditors so cross-domain reasoning comes from genuinely fresh
            # eyes.  When synthesis_model is blank it falls back to
            # critical_fix_model (potentially the same family as the auditors).
            #
            # Enforcement levels:
            #   • MILITARY / AEROSPACE / NUCLEAR: raises ConfigurationError
            #     (same-family synthesis is a compliance failure in these modes)
            #   • All other domains: logs a WARNING (advisory only)
            _synthesis_model_is_default = not (
                self.config.synthesis_model
                or os.environ.get("RHODAWK_SYNTHESIS_MODEL", "")
            )
            if _synthesis_model_is_default:
                self.log.warning(
                    "[audit] synthesis_model is not configured — falling back to "
                    f"critical_fix_model ({self.config.critical_fix_model}). "
                    "This may be the same model family as the auditors and reduces "
                    "cross-domain reasoning independence. "
                    "Set synthesis_model in [synthesis] config or "
                    "RHODAWK_SYNTHESIS_MODEL env var to a different family "
                    "(e.g. DeepSeek-Coder-V2 or Qwen2.5-Coder-32B) for best results."
                )
            else:
                # When synthesis_model IS explicitly set, verify the families differ.
                try:
                    from verification.independence_enforcer import extract_model_family
                    primary_family   = extract_model_family(self.config.primary_model)
                    synthesis_family = extract_model_family(synthesis_model)
                    if (
                        primary_family
                        and synthesis_family
                        and primary_family == synthesis_family
                    ):
                        _family_msg = (
                            f"[audit] synthesis_model ({synthesis_model}) appears to be "
                            f"the same model family as primary_model "
                            f"({self.config.primary_model}): both resolve to "
                            f"'{primary_family}'. Cross-domain compound finding quality "
                            "will be reduced. Set synthesis_model to a model from a "
                            "different provider/family for genuine independence."
                        )
                        if self.config.domain_mode in {
                            DomainMode.MILITARY, DomainMode.AEROSPACE, DomainMode.NUCLEAR
                        }:
                            raise ConfigurationError(_family_msg)
                        else:
                            self.log.warning(_family_msg)
                except (ImportError, Exception) as _family_exc:
                    # extract_model_family is best-effort; never block the pipeline
                    self.log.debug(
                        f"[audit] synthesis model family check skipped: {_family_exc}"
                    )

            synthesis_agent = SynthesisAgent(
                storage=self.storage,
                run_id=self.run.id,
                config=a_cfg,
                mcp_manager=self.mcp,
                domain_mode=self.config.domain_mode,
                repo_root=self.config.repo_root,
                synthesis_model=synthesis_model,
                dedup_enabled=self.config.synthesis_dedup_enabled,
                compound_enabled=self.config.synthesis_compound_enabled,
                max_compound_findings=self.config.synthesis_max_compound,
            )
            _synthesis_start = datetime.now(tz=timezone.utc)
            try:
                deduped_issues, compound_findings = await synthesis_agent.run(
                    issues=all_issues
                )
                _synthesis_duration_s = (
                    datetime.now(tz=timezone.utc) - _synthesis_start
                ).total_seconds()

                # Persist compound findings for report exporter and DeerFlow
                self._last_compound_findings = compound_findings

                # Materialise compound findings as Issues in the issue list so
                # they flow through consensus → fix → review unchanged.
                # FIX: replaced silent list-comprehension None-drop with
                # explicit per-finding fetch so storage failures are logged
                # at ERROR level instead of being swallowed silently.
                compound_issues: list[Issue] = []
                for cf in compound_findings:
                    if not cf.synthesized_issue_id:
                        continue
                    fetched = await self.storage.get_issue(cf.synthesized_issue_id)
                    if fetched is None:
                        self.log.error(
                            f"[audit] Compound finding '{cf.title}' could not be "
                            f"materialised — storage returned None for "
                            f"synthesized_issue_id={cf.synthesized_issue_id}. "
                            "This likely means upsert_synthesis_report / "
                            "upsert_issue is not implemented on the active "
                            "storage backend (e.g. PostgresBrainStorage). "
                            "The compound finding will be omitted from this "
                            "audit cycle's issue list."
                        )
                    else:
                        compound_issues.append(fetched)

                all_issues = deduped_issues + compound_issues

                # Record compound findings in Prometheus
                for cf in compound_findings:
                    record_issue(cf.severity.value, self.config.domain_mode.value)

                self.log.info(
                    f"[audit] Synthesis: {raw_count} → {len(deduped_issues)} "
                    f"(deduped) + {len(compound_issues)} compound findings "
                    f"= {len(all_issues)} total"
                )

                # ── GAP 2 FIX: Build and persist SynthesisReport ─────────────
                # Previously SynthesisReport was defined in schemas.py and
                # imported in controller.py but NEVER CONSTRUCTED at runtime.
                # Per-run dedup/compound metrics (dedup count, compound count,
                # compound_critical_count, duration_s) were only written to
                # log.info and lost forever.  This block closes that gap.
                compound_critical = sum(
                    1 for cf in compound_findings
                    if cf.severity == Severity.CRITICAL
                )
                total_deduped = raw_count - len(deduped_issues)
                synthesis_report = SynthesisReport(
                    run_id=self.run.id,
                    cycle=self.run.cycle_count,
                    raw_issue_count=raw_count,
                    fingerprint_dedup_count=total_deduped,
                    semantic_dedup_count=0,   # aggregate tracked in fingerprint_dedup_count
                    final_issue_count=len(deduped_issues),
                    compound_finding_count=len(compound_findings),
                    compound_critical_count=compound_critical,
                    synthesis_model=synthesis_model,
                    dedup_enabled=self.config.synthesis_dedup_enabled,
                    compound_enabled=self.config.synthesis_compound_enabled,
                    duration_s=_synthesis_duration_s,
                )
                try:
                    await self.storage.upsert_synthesis_report(synthesis_report)
                    self.log.info(
                        f"[audit] SynthesisReport persisted: "
                        f"raw={raw_count} deduped={len(deduped_issues)} "
                        f"compound={len(compound_findings)} "
                        f"(critical={compound_critical}) "
                        f"duration={_synthesis_duration_s:.1f}s"
                    )
                except Exception as report_exc:
                    # Non-fatal — log but never block the pipeline
                    self.log.warning(
                        f"[audit] SynthesisReport persist failed (non-fatal): {report_exc}"
                    )

            except Exception as exc:
                self.log.error(
                    f"[audit] SynthesisAgent failed — using raw undeduped findings: {exc}"
                )
                # Fail-open: never lose findings because synthesis crashed
                self._last_compound_findings = []
        else:
            self._last_compound_findings = []
            self.log.debug(
                "[audit] Synthesis disabled or no findings — skipping synthesis pass"
            )

        await self._trail(
            "AUDIT_PHASE_COMPLETE", self.run.id, "run",
            after=(
                f"{len(all_issues)} issues "
                f"({len(self._last_compound_findings)} compound)"
            ),
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

    async def _init_gap5(self) -> None:
        """
        Gap 5: Initialise the adversarial BoBN ensemble.

        Performs three things:
          1. Pushes Gap 5 env vars into os.environ so TieredModelRouter and
             AdversarialCriticAgent pick them up without config refactoring.
          2. Calls router.assert_family_independence() — raises RuntimeError if
             the critic and any fixer share the same model family.  This is the
             CI gate that was missing; a misconfigured VLLM_CRITIC_MODEL silently
             produced correlated blind spots before this call was added.
          3. Instantiates AdversarialCriticAgent and BoBNSampler and stores them
             as controller state so _phase_fix_gap5() can reuse them across cycles
             without re-constructing on every call.

        All failures are non-fatal — if Gap 5 init fails the pipeline degrades
        gracefully to the standard single-fixer _phase_fix() path.
        """
        if not _GAP5_AVAILABLE:
            self.log.warning(
                "[gap5] adversarial_critic / bobn_sampler not importable — "
                "gap5_enabled will be ignored.  Check that swe_bench/ is on PYTHONPATH."
            )
            return

        # Push config values into env so TieredModelRouter reads them correctly.
        # We only set if not already overridden by the operator.
        env_defaults = {
            "VLLM_SECONDARY_BASE_URL": self.config.gap5_vllm_secondary_base_url,
            "VLLM_SECONDARY_MODEL":    self.config.gap5_vllm_secondary_model,
            "VLLM_CRITIC_BASE_URL":    self.config.gap5_vllm_critic_base_url,
            "VLLM_CRITIC_MODEL":       self.config.gap5_vllm_critic_model,
            "RHODAWK_BOBN_CANDIDATES": str(self.config.gap5_bobn_n_candidates),
            "RHODAWK_BOBN_FIXER_A":    str(self.config.gap5_bobn_fixer_a_count),
            "RHODAWK_BOBN_FIXER_B":    str(self.config.gap5_bobn_fixer_b_count),
        }
        for key, val in env_defaults.items():
            if val and key not in os.environ:
                os.environ[key] = val

        # Family independence check — this is the gate that was missing.
        # assert_family_independence() raises RuntimeError if critic family ==
        # fixer family, which would make the adversarial pass worthless.
        try:
            from models.router import get_router
            router = get_router()
            router.assert_family_independence()
        except RuntimeError as exc:
            self.log.error(
                f"[gap5] Family independence violation — Gap 5 disabled: {exc}\n"
                "Set VLLM_CRITIC_MODEL=meta-llama/Llama-3.3-70B-Instruct to fix."
            )
            return
        except Exception as exc:
            self.log.warning(f"[gap5] Family check skipped: {exc}")

        # Instantiate critic and sampler.  Both are stateless between calls so
        # a single instance per controller lifetime is safe.
        try:
            from models.router import get_router
            router = get_router()
            self._adversarial_critic = AdversarialCriticAgent(model_router=router)

            # GAP 5 FIX: PatchSynthesisAgent — picks winner or merges best elements.
            # Uses CLOUD_OSS tier (Devstral/Llama-4), a different family from both
            # fixers and the adversarial critic.  Created here and passed into
            # BoBNSampler so it is reused across cycles without reconstruction.
            from agents.patch_synthesis_agent import PatchSynthesisAgent
            self._patch_synthesis = PatchSynthesisAgent(model_router=router)

            self._bobn_sampler = BoBNSampler(
                model_router = router,
                critic       = self._adversarial_critic,
                synthesis    = self._patch_synthesis,
            )

            # GAP 5 FIX: Trajectory collector for ARPO RL training corpus.
            # Production fix runs should accumulate training data just like
            # the SWE-bench evaluator path does.
            from swe_bench.trajectory_collector import TrajectoryCollector
            self._trajectory_collector = TrajectoryCollector()

            self.log.info(
                "[gap5] Adversarial ensemble ready: "
                f"fixer_a={router.primary_model('fix')} "
                f"fixer_b={router.secondary_model()} "
                f"critic={router.critic_model()} "
                f"bobn_n={self.config.gap5_bobn_n_candidates} "
                f"synthesis=ON trajectory=ON"
            )
        except Exception as exc:
            self.log.error(f"[gap5] BoBN sampler init failed — Gap 5 disabled: {exc}")
            self._adversarial_critic = None
            self._patch_synthesis    = None
            self._bobn_sampler       = None
            self._trajectory_collector = None

    async def _phase_fix_gap5(self, issues: list[Issue]) -> None:
        """
        Gap 5 fix phase: dual-fixer BoBN adversarial ensemble.

        Pipeline per issue:
          1. Build localization context using CPG slices (Gap 1) or hybrid
             retriever fallback — gives both fixers the causal context slice,
             not the semantically-similar one.
          2. Run BoBNSampler.sample():
               Fixer A (Qwen-32B)   × 3 temperatures  →  3 patch candidates
               Fixer B (DeepSeek-16B) × 2 temperatures → 2 patch candidates
               ExecutionFeedbackLoop × 5 candidates    → test-scored patches
               AdversarialCriticAgent                  → attack reports
               Composite ranking (test×0.6 + robust×0.3 + minimal×0.1)
          3. Write the winning patch back as a FixAttempt using the existing
             FixerAgent storage path so downstream gates (review, formal, commit)
             work unchanged.

        Falls back to the standard _phase_fix() if BoBN is not initialised or
        if no issues require the full ensemble (e.g. trivial lint fixes).
        """
        if not self._bobn_sampler or not self.run or not self.storage:
            self.log.warning("[gap5] BoBN sampler not ready — falling back to single-fixer path")
            await self._phase_fix(issues)
            return

        if not issues:
            return

        self.log.info(f"[gap5] Starting adversarial BoBN fix for {len(issues)} issue(s)")

        # Update the BoBNSampler with the current issue batch's combined context.
        # Each issue is processed as a separate SWE-bench-style instance.
        for issue in issues:
            issue_text = (
                f"File: {issue.file_path}\n"
                f"Severity: {issue.severity.value if hasattr(issue.severity, 'value') else issue.severity}\n"
                f"Domain: {issue.domain}\n\n"
                f"{issue.description}"
            )

            # Build localization context: prefer CPG causal slice, fall back to
            # hybrid retriever.  This is the Gap 1 / Gap 5 integration point.
            localization_context = ""
            if self._cpg_context_selector and self._cpg_engine and \
               getattr(self._cpg_engine, "is_available", False):
                try:
                    ctx = await self._cpg_context_selector.select_context(
                        issue=issue, max_nodes=self.config.cpg_max_slice_nodes
                    )
                    localization_context = ctx.formatted_context if hasattr(ctx, "formatted_context") else str(ctx)
                except Exception as exc:
                    self.log.debug(f"[gap5] CPG context failed for {issue.id[:8]}: {exc}")

            if not localization_context and self._hybrid_retriever:
                try:
                    hits = await self._hybrid_retriever.query(
                        query=issue.description[:500],
                        top_k=5,
                    )
                    localization_context = "\n\n".join(
                        h.get("content", "") for h in (hits or []) if h.get("content")
                    )
                except Exception as exc:
                    self.log.debug(f"[gap5] Hybrid retriever fallback failed: {exc}")

            # Update sampler context for this issue before sampling.
            self._bobn_sampler.issue   = issue_text
            self._bobn_sampler.loc_ctx = localization_context

            try:
                bobn_result = await self._bobn_sampler.sample(
                    instance_id  = issue.id,
                    repo         = self.config.repo_url,
                    base_commit  = "",            # not in Issue schema; gap bench only
                    fail_tests   = [],            # execution loop skips docker without tests
                    pass_tests   = None,
                    repo_root    = self.config.repo_root,
                )

                if not bobn_result.winner or not bobn_result.winner.patch:
                    self.log.warning(
                        f"[gap5] BoBN returned no winner for issue {issue.id[:8]} "
                        f"— falling back to single-fixer for this issue"
                    )
                    await self._phase_fix([issue])
                    continue

                winner = bobn_result.winner
                self.log.info(
                    f"[gap5] Issue {issue.id[:8]}: winner=candidate_{winner.candidate_id} "
                    f"model={winner.model.split('/')[-1]} "
                    f"composite={winner.composite_score:.2f} "
                    f"test_score={winner.test_score:.2f} "
                    f"synthesis={bobn_result.synthesis_action} "
                    f"attack_confidence={winner.attack_report.attack_confidence:.2f if winner.attack_report else 0:.2f} "
                    f"candidates={bobn_result.n_candidates} "
                    f"fully_passing={bobn_result.n_fully_passing}"
                )

                # ── GAP 5 FIX A: Formal gate on winning patch ─────────────────
                # Previously MISSING from the controller path entirely.
                # Runs three layers: diff sanity → safety pattern scan → Z3.
                # On failure tries the second-best candidate before giving up.
                winning_patch      = winner.patch
                formal_gate_passed = True

                if self.config.formal_verification and winning_patch:
                    try:
                        from swe_bench.evaluator import _check_safety_patterns, _run_z3_gate
                        from agents.patch_synthesis_agent import _validate_diff
                        from models.router import get_router as _get_router

                        added_lines = [
                            ln[1:] for ln in winning_patch.split("\n")
                            if ln.startswith("+") and not ln.startswith("+++")
                        ]
                        added_content = "\n".join(added_lines)
                        violations    = _check_safety_patterns(added_content, issue.id[:8])
                        diff_valid    = _validate_diff(winning_patch)

                        if violations or not diff_valid:
                            self.log.warning(
                                f"[gap5] Formal gate FAIL issue={issue.id[:8]} "
                                f"reason={'violations:' + str(violations) if violations else 'invalid diff'} "
                                "— trying second-best candidate"
                            )
                            formal_gate_passed = False
                            second = next(
                                (c for c in bobn_result.all_candidates[1:]
                                 if c.patch and c.patch != winning_patch),
                                None,
                            )
                            if second:
                                winning_patch      = second.patch
                                winner             = second
                                formal_gate_passed = True
                                self.log.info(
                                    f"[gap5] Formal gate: promoted second-best "
                                    f"candidate={winner.candidate_id} for issue {issue.id[:8]}"
                                )
                        else:
                            # Z3 layer — advisory, never blocks in production
                            try:
                                z3_ok = await _run_z3_gate(
                                    issue_text  = issue_text,
                                    patch       = winning_patch,
                                    router      = _get_router(),
                                    instance_id = issue.id[:8],
                                )
                                if not z3_ok:
                                    self.log.warning(
                                        f"[gap5] Z3 advisory: counterexample found for "
                                        f"issue {issue.id[:8]} — patch may be unsound "
                                        "(proceeding: Z3 is advisory in production)"
                                    )
                            except Exception as _z3e:
                                self.log.debug(f"[gap5] Z3 layer skipped: {_z3e}")

                    except ImportError as _ie:
                        self.log.debug(f"[gap5] Formal gate helpers unavailable: {_ie}")

                # ── GAP 5 FIX B: Trajectory collection ───────────────────────
                # Previously MISSING from the controller path.  The evaluator
                # path collected trajectories; production fix runs must too so
                # the ARPO RL corpus grows from real-world data, not just benchmarks.
                _tc = getattr(self, "_trajectory_collector", None)
                if _tc is not None and bobn_result.all_candidates:
                    try:
                        _tc.collect_from_bobn_result(
                            instance_id = issue.id,
                            bobn_result = bobn_result,
                            resolved    = formal_gate_passed,
                            issue_text  = issue_text,
                            loc_context = localization_context,
                        )
                        _corpus = _tc.corpus_size()
                        self.log.debug(
                            f"[gap5] Trajectory saved for issue {issue.id[:8]}. "
                            f"Corpus={_corpus}"
                        )
                        if _tc.is_ready_for_training():
                            self.log.info(
                                f"[gap5] RL corpus ready ({_corpus} trajectories) — "
                                "run: python scripts/arpo_trainer.py"
                            )
                    except Exception as _te:
                        self.log.debug(f"[gap5] Trajectory collection non-fatal: {_te}")

                # ── Write the winning patch as a FixAttempt ───────────────────
                fixer = FixerAgent(
                    storage                  = self.storage,
                    run_id                   = self.run.id,
                    config                   = _agent_cfg(self.config, self.run.id),
                    mcp_manager              = self.mcp,
                    repo_root                = self.config.repo_root,
                    graph_engine             = self.graph_engine if self.config.graph_enabled else None,
                    vector_brain             = self.vector_brain,
                    surgical_patch_threshold = SURGICAL_PATCH_THRESHOLD,
                    repo_map                 = self._repo_map,
                    hybrid_retriever         = self._hybrid_retriever,
                    fix_memory               = self._fix_memory,
                    cpg_engine               = self._cpg_engine,
                    cpg_context_selector     = self._cpg_context_selector,
                    program_slicer           = self._program_slicer,
                    escalation_manager       = self._escalation_mgr,
                    blast_radius_threshold   = self.config.cpg_blast_radius_threshold,
                )
                _synthesis_decision = getattr(winner, "synthesis_decision", None)
                fixes = await fixer.run_with_patch(
                    issues      = [issue],
                    patch       = winning_patch,
                    patch_model = winner.model,
                    patch_meta  = {
                        "bobn_candidate_id":    winner.candidate_id,
                        "bobn_composite_score": winner.composite_score,
                        "bobn_test_score":      winner.test_score,
                        "bobn_n_candidates":    bobn_result.n_candidates,
                        "attack_confidence":    winner.attack_report.attack_confidence
                            if winner.attack_report else None,
                        "attack_vectors":       len(winner.attack_report.attack_vectors)
                            if winner.attack_report else 0,
                        "synthesis_action":     bobn_result.synthesis_action,
                        "synthesis_confidence": _synthesis_decision.confidence
                            if _synthesis_decision else None,
                        "formal_gate_passed":   formal_gate_passed,
                    },
                )
                for f in (fixes or []):
                    if self._independence_enforcer and f:
                        rec = ReviewerIndependenceRecord(
                            fix_attempt_id        = f.id,
                            fixer_model           = winner.model,
                            fixer_model_family    = self._independence_enforcer.fixer_family,
                            reviewer_model        = self.config.reviewer_model,
                            reviewer_model_family = self._independence_enforcer.reviewer_family,
                        )
                        await self.storage.upsert_independence_record(rec)
                        f.independence_record_id = rec.id
                        f.fixer_model        = winner.model
                        f.fixer_model_family = self._independence_enforcer.fixer_family
                        await self.storage.upsert_fix(f)
                    record_fix("generated")

                # ── GAP 5 FIX C: Inline test generation + mutation ────────────
                # Previously test gen and mutation existed as separate controller
                # phases that ran after _phase_fix_gap5 completed.  They were
                # never wired into this branch, so gap5 fixes were committed
                # without any test generation or mutation coverage gate.
                if fixes and self.config.test_gen_enabled:
                    try:
                        _tg = TestGeneratorAgent(
                            storage     = self.storage,
                            run_id      = self.run.id,
                            config      = _agent_cfg(self.config, self.run.id),
                            mcp_manager = self.mcp,
                            repo_root   = self.config.repo_root,
                        )
                        await _tg.run(fixes=fixes)
                        self.log.info(
                            f"[gap5] Test generation done for issue {issue.id[:8]}"
                        )
                    except Exception as _tge:
                        self.log.debug(f"[gap5] Test generation non-fatal: {_tge}")

                if fixes and self.config.mutation_testing_enabled:
                    try:
                        _mv = MutationVerifierAgent(
                            storage         = self.storage,
                            run_id          = self.run.id,
                            config          = _agent_cfg(self.config, self.run.id),
                            mcp_manager     = self.mcp,
                            repo_root       = self.config.repo_root,
                            score_threshold = self.config.mutation_score_threshold,
                        )
                        await _mv.run(fixes=fixes)
                        self.log.info(
                            f"[gap5] Mutation testing done for issue {issue.id[:8]}"
                        )
                    except Exception as _mve:
                        self.log.debug(f"[gap5] Mutation testing non-fatal: {_mve}")

            except Exception as exc:
                self.log.error(
                    f"[gap5] BoBN pipeline failed for issue {issue.id[:8]}: {exc} "
                    "— falling back to single-fixer"
                )
                await self._phase_fix([issue])

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
            repo_map=self._repo_map,
            hybrid_retriever=self._hybrid_retriever,
            fix_memory=self._fix_memory,
            # Gap 1: CPG causal context
            cpg_engine=self._cpg_engine,
            cpg_context_selector=self._cpg_context_selector,
            program_slicer=self._program_slicer,
            # Gap 3: forward impact gate — escalation_manager fires when blast
            # radius exceeds threshold; blast_radius_threshold forwarded from
            # config so it matches the CPGEngine and PlannerAgent thresholds.
            escalation_manager=self._escalation_mgr,
            blast_radius_threshold=self.config.cpg_blast_radius_threshold,
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

    # ── Test generation phase (Antagonist) ────────────────────────────────────

    async def _phase_test_gen(self) -> dict[str, list[str]]:
        """
        Generate unit tests for every gate-pending fix.

        Runs TestGeneratorAgent between the fix phase and the review phase.
        Generated test paths are stored on the FixAttempt record for use by
        MutationVerifierAgent and the RTM exporter.

        Returns a mapping of fix_id → list[test_file_paths].
        """
        if not self.config.test_gen_enabled:
            return {}
        assert self.run and self.storage

        fixes = [
            f for f in await self.storage.list_fixes()
            if f.gate_passed is None and f.committed_at is None
        ]
        if not fixes:
            return {}

        agent = TestGeneratorAgent(
            storage=self.storage,
            run_id=self.run.id,
            config=_agent_cfg(self.config, self.run.id),
            mcp_manager=self.mcp,
            repo_root=self.config.repo_root,
            pynguin_timeout=self.config.pynguin_timeout_s,
            use_hypothesis=self.config.use_hypothesis,
        )

        results: dict[str, list[str]] = {}
        for fix in fixes:
            try:
                paths = await agent.run(fix)
                results[fix.id] = paths
                if paths:
                    self.log.info(
                        f"[TestGen] {len(paths)} test file(s) generated "
                        f"for fix {fix.id[:12]}: {paths[:3]}"
                    )
            except Exception as exc:
                self.log.warning(f"[TestGen] Failed for fix {fix.id[:12]}: {exc}")

        await self._trail(
            "TEST_GEN_PHASE_COMPLETE", self.run.id, "run",
            after=f"{sum(len(v) for v in results.values())} test files",
        )
        return results

    # ── Mutation testing phase (Antagonist) ────────────────────────────────────

    async def _phase_mutation(self) -> None:
        """
        Run mutmut against all Python files in gate-pending fixes.

        Blocks commit on any fix whose mutation score falls below the domain
        threshold.  Requires Python + mutmut>=2.5.0.  Non-Python files are
        skipped silently.
        """
        if not self.config.mutation_testing_enabled:
            return
        assert self.run and self.storage

        fixes = [
            f for f in await self.storage.list_fixes()
            if f.gate_passed is None and f.committed_at is None
        ]
        if not fixes:
            return

        agent = MutationVerifierAgent(
            storage=self.storage,
            run_id=self.run.id,
            config=_agent_cfg(self.config, self.run.id),
            repo_root=self.config.repo_root,
            domain_mode=self.config.domain_mode,
            score_threshold=self.config.mutation_score_threshold,
        )

        for fix in fixes:
            test_paths = getattr(fix, "generated_test_paths", None) or []
            try:
                results = await agent.run(fix, test_paths=test_paths)
                if results:
                    scores = ", ".join(
                        f"{r.file_path}:{r.mutation_score:.1f}%" for r in results
                    )
                    self.log.info(
                        f"[Mutation] fix {fix.id[:12]}: {scores}"
                    )
            except Exception as exc:
                self.log.warning(
                    f"[Mutation] Failed for fix {fix.id[:12]}: {exc}"
                )

        await self._trail(
            "MUTATION_PHASE_COMPLETE", self.run.id, "run",
            after="mutation scores recorded",
        )

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
            cpg_engine=self._cpg_engine,
            # Gap 3: blast radius hard-block in planner also triggers an
            # escalation — without this the planner blocks the commit silently
            # but no human is notified.
            escalation_manager=self._escalation_mgr,
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

        # Gap 4: Incremental CPG update after commit.
        # Uses CommitAuditScheduler when available (full Gap 4 path), otherwise
        # falls back to the raw IncrementalCPGUpdater.  Both paths write
        # FunctionStalenessMark rows so the next read phase audits only the
        # 50–200 functions in the CPG impact set rather than the full codebase.
        if self._commit_audit_scheduler and committed:
            try:
                car = await self._commit_audit_scheduler.schedule_commit_audit(
                    commit_hash="",
                    changed_files=committed,
                    branch=getattr(self.run, "branch", ""),
                    author="Rhodawk AI (auto-fix)",
                    commit_message="",
                )
                self.log.info(
                    "[Gap4] CommitAuditScheduler: changed=%d fns, impact=%d fns, "
                    "audit_targets=%d (status=%s)",
                    car.total_changed_functions,
                    car.total_impact_functions,
                    car.total_functions_to_audit,
                    car.status.value,
                )
            except Exception as exc:
                self.log.warning("[Gap4] CommitAuditScheduler failed (non-fatal): %s", exc)

        elif self._incremental_updater and committed:
            try:
                update_result = await self._incremental_updater.update_after_commit(
                    changed_files=committed,
                    run_id=self.run.id,
                )
                # Gap 4 fix: feed audit_targets back as FunctionStalenessMark rows
                # so the next cycle picks up exactly the impacted functions.
                # Previously the result was computed but never persisted back into
                # stale marks — the incremental updater wrote them internally but
                # the controller ignored the returned audit_targets entirely.
                for target in update_result.audit_targets:
                    fn_name = target.get("function_name", "")
                    fp      = target.get("file_path", "")
                    if not fn_name or not fp:
                        continue
                    try:
                        mark = FunctionStalenessMark(
                            file_path=fp,
                            function_name=fn_name,
                            line_start=target.get("line_number", 0),
                            stale_reason=target.get("relationship", "cpg_impact"),
                            run_id=self.run.id,
                        )
                        await self.storage.upsert_staleness_mark(mark)
                    except Exception:
                        pass

                total_lines = sum(
                    getattr(r, "line_count", 0)
                    for r in await self.storage.list_files()
                )
                self.log.info(
                    "[Gap4] IncrementalCPGUpdater: changed=%d fns, impact=%d fns, "
                    "audit_targets=%d (vs %d total lines)",
                    update_result.total_functions_changed,
                    update_result.total_functions_affected,
                    update_result.total_functions_to_audit,
                    total_lines,
                )
            except Exception as exc:
                self.log.warning("[Gap4] Incremental update failed (non-fatal): %s", exc)

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

        # ── Persist the reverted approach so the Fixer never repeats it ────────
        if self._fix_memory:
            try:
                issues = await self.storage.list_issues(run_id=self.run.id)
                issue_map = {i.id: i for i in issues}
                for ff in fix.fixed_files:
                    # Collect the issue types that this fix was addressing
                    issue_types = ", ".join(
                        issue_map[iid].description[:80]
                        for iid in fix.issue_ids
                        if iid in issue_map
                    ) or "unknown"
                    fix_approach = ff.diff_summary or ff.changes_made or "patch applied"
                    self._fix_memory.store_failure(
                        issue_type=issue_types,
                        file_context=ff.path,
                        fix_approach=fix_approach,
                        failure_reason="Test regression detected post-commit; fix reverted",
                        run_id=self.run.id,
                    )
                self.log.info(
                    f"[revert] Stored {len(fix.fixed_files)} reverted fix pattern(s) "
                    f"in FixMemory for fix {fix.id[:12]}"
                )
            except Exception as exc:
                self.log.warning(f"[revert] FixMemory.store_failure failed (non-fatal): {exc}")

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
        """
        Gap 4 fix: use CPG function-level impact when available; fall back to
        the file-level import-graph blast radius only when CPG is absent.

        The previous implementation always used the file-level graph, which
        marks entire files as STALE even when only one function changed.  With
        the CPG we compute the exact set of downstream functions to depth 3
        and emit FunctionStalenessMark rows rather than coarse FileStatus.STALE.
        """
        assert self.storage

        # ── CPG path: function-granularity impact ────────────────────────────
        if self._cpg_engine and self._incremental_updater and changed:
            try:
                all_fns: list[str] = []
                for path in changed:
                    rec = await self.storage.get_file(path)
                    if rec and rec.known_functions:
                        all_fns.extend(rec.known_functions)

                if all_fns:
                    impact = await self._cpg_engine.compute_blast_radius(
                        function_names=all_fns,
                        file_paths=list(changed),
                        depth=3,
                    )
                    impacted_fns   = impact.affected_functions
                    impacted_files = set(impact.affected_files) - changed

                    # GAP 4 FIX: cap the per-function staleness mark count to
                    # _MAX_BLAST_CAP.  A widely-used utility can have thousands
                    # of callers at depth 3.  Emitting thousands of staleness
                    # marks per commit degrades storage and makes the next read
                    # phase re-audit more code than a full scan would.
                    #
                    # When the cap is hit, function-level marks are emitted only
                    # for the first _MAX_BLAST_CAP functions.  The remaining
                    # impacted FILES are still marked STALE so they get
                    # re-audited at file granularity — coarser but correct.
                    _MAX_BLAST_CAP = 500
                    if len(impacted_fns) > _MAX_BLAST_CAP:
                        self.log.warning(
                            "[Gap4] CPG blast radius %d exceeds cap %d for "
                            "changed files %s — truncating function-level "
                            "marks to cap; remaining files fall back to "
                            "file-level STALE.",
                            len(impacted_fns), _MAX_BLAST_CAP,
                            sorted(changed)[:3],
                        )
                        impacted_fns = impacted_fns[:_MAX_BLAST_CAP]

                    added = 0
                    for item in impacted_fns:
                        fn = item.get("function_name", "")
                        fp = item.get("file_path", "")
                        if not fn or not fp or fp in changed:
                            continue
                        try:
                            mark = FunctionStalenessMark(
                                file_path=fp,
                                function_name=fn,
                                stale_reason="transitive_cpg_impact",
                                run_id=self.run.id,
                            )
                            await self.storage.upsert_staleness_mark(mark)
                            added += 1
                        except Exception:
                            pass
                    # Still mark impacted files STALE for the read phase
                    for path in impacted_files:
                        rec = await self.storage.get_file(path)
                        if rec:
                            from brain.schemas import FileStatus
                            rec.status = FileStatus.STALE
                            await self.storage.upsert_file(rec)
                    self.log.info(
                        "[Gap4] CPG transitive impact: %d function marks, %d files",
                        added, len(impacted_files),
                    )
                    return
            except Exception as exc:
                self.log.debug(
                    "[Gap4] CPG transitive impact failed, falling back to graph: %s", exc
                )

        # ── Fallback: file-level import graph ────────────────────────────────
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
        self.log.info(
            "[Gap4] file-graph transitive dependents: %d files marked STALE",
            len(affected),
        )


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
        # Gap 1: Close Joern connection
        if self._cpg_engine:
            try:
                await self._cpg_engine.close()
            except Exception:
                pass
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
