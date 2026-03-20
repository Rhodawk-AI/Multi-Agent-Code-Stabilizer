from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from brain.schemas import (
    AuditRun, AuditScore, AuditTrailEntry,
    BaselineRecord, CbmcVerificationResult,
    CommitAuditRecord, CommitAuditStatus,
    ConvergenceRecord,
    EscalationRecord, EscalationStatus,
    FileChunkRecord, FileRecord,
    FixAttempt, FormalVerificationResult,
    FunctionStalenessMark,
    Issue, IssueStatus,
    LdraFinding, PatrolEvent,
    PlannerRecord, PolyspaceFinding,
    RefactorProposal,
    RequirementTraceability, ReviewerIndependenceRecord,
    RunStatus, SoftwareAccomplishmentSummary,
    SoftwareConfigurationIndex, SynthesisReport, TestRunResult,
)


class BrainStorage(ABC):
    """
    Abstract persistence layer.  All methods are async.

    Implementation contract:
    • upsert_* methods must be idempotent — calling twice with the same
      object must produce the same stored state as calling once.
    • list_* methods return empty lists (not None) when no records exist.
    • get_* methods return None when the record does not exist.
    • All writes must be atomic at the row level.
    """

                                                                                 
    @abstractmethod
    async def initialise(self) -> None:
        """Create schema if not exists. Must be idempotent."""

    @abstractmethod
    async def close(self) -> None:
        """Flush and release all connections."""

                                                                                 
    @abstractmethod
    async def upsert_run(self, run: AuditRun) -> None: ...

    @abstractmethod
    async def get_run(self, run_id: str) -> AuditRun | None: ...

    @abstractmethod
    async def update_run_status(self, run_id: str, status: RunStatus) -> None: ...

    @abstractmethod
    async def append_score(self, score: AuditScore) -> None: ...

                                                                                 
    @abstractmethod
    async def upsert_file(self, record: FileRecord) -> None: ...

    @abstractmethod
    async def get_file(self, path: str) -> FileRecord | None: ...

    @abstractmethod
    async def list_files(self, run_id: str = "") -> list[FileRecord]: ...

    @abstractmethod
    async def upsert_chunk(self, chunk: FileChunkRecord) -> None: ...

    @abstractmethod
    async def list_chunks(
        self, file_path: str, run_id: str = ""
    ) -> list[FileChunkRecord]: ...

    @abstractmethod
    async def get_all_observations(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def get_stale_observations(self, run_id: str = "") -> list[dict[str, Any]]:
        """
        Return the minimal observation dicts needed to re-audit only the
        functions marked stale by CommitAuditScheduler.

        The returned dicts have the same shape as get_all_observations()
        ({file_path, language, content, line_start, line_end, function_name,
        dependencies}) but are scoped to chunks whose function_name appears in
        the function_staleness table for the given run_id.

        When run_id is empty, all stale marks across all runs are included.
        When the function_staleness table is empty, returns an empty list —
        callers must fall back to get_all_observations() in that case.
        """
        ...

                                                                                 
    @abstractmethod
    async def upsert_issue(self, issue: Issue) -> None: ...

    @abstractmethod
    async def get_issue(self, issue_id: str) -> Issue | None: ...

    @abstractmethod
    async def list_issues(
        self, run_id: str = "", status: str | None = None
    ) -> list[Issue]: ...

    @abstractmethod
    async def update_issue_status(
        self, issue_id: str, status: str, reason: str = ""
    ) -> None: ...

    @abstractmethod
    async def get_total_cost(self, run_id: str) -> float: ...

                                                                                 
    @abstractmethod
    async def upsert_fix(self, fix: FixAttempt) -> None: ...

    @abstractmethod
    async def get_fix(self, fix_id: str) -> FixAttempt | None: ...

    @abstractmethod
    async def list_fixes(self, run_id: str = "") -> list[FixAttempt]: ...

                                                                                 
    @abstractmethod
    async def upsert_planner_record(self, record: PlannerRecord) -> None: ...

                                                                                 
    @abstractmethod
    async def append_audit_trail(self, entry: AuditTrailEntry) -> None: ...

    @abstractmethod
    async def list_audit_trail(
        self, run_id: str, limit: int = 1000
    ) -> list[AuditTrailEntry]: ...

                                                                                 
    @abstractmethod
    async def append_patrol_event(self, event: PatrolEvent) -> None: ...

                                                                                 
    @abstractmethod
    async def upsert_test_result(self, result: TestRunResult) -> None: ...

                                                                                 
    @abstractmethod
    async def upsert_formal_result(self, result: FormalVerificationResult) -> None: ...

                                                                               
    @abstractmethod
    async def upsert_escalation(self, esc: EscalationRecord) -> None:
        """Persist an escalation record.  Must be idempotent on id."""

    @abstractmethod
    async def get_escalation(self, escalation_id: str) -> EscalationRecord | None:
        """Return the escalation with the given id, or None."""

    @abstractmethod
    async def list_escalations(
        self,
        run_id: str = "",
        status: EscalationStatus | None = None,
    ) -> list[EscalationRecord]:
        """List escalations, optionally filtered by run_id and/or status."""

                                                                                
    @abstractmethod
    async def upsert_refactor_proposal(self, proposal: RefactorProposal) -> None:
        """Persist a refactor proposal.  Must be idempotent on id."""

    @abstractmethod
    async def get_refactor_proposal(self, proposal_id: str) -> RefactorProposal | None:
        """Return the refactor proposal with the given id, or None."""

    @abstractmethod
    async def list_refactor_proposals(self, run_id: str = "") -> list[RefactorProposal]:
        """List all refactor proposals for a run."""

                                                                                 
    @abstractmethod
    async def upsert_baseline(self, baseline: BaselineRecord) -> None:
        """Persist a baseline record."""

    @abstractmethod
    async def get_baseline(self, baseline_id: str) -> BaselineRecord | None:
        """Return the baseline with the given id, or None."""

    @abstractmethod
    async def get_active_baseline(self, run_id: str = "") -> BaselineRecord | None:
        """Return the currently active (is_active=True) baseline for a run."""

    @abstractmethod
    async def list_baselines(self, run_id: str = "") -> list[BaselineRecord]:
        """List all baselines, optionally filtered by run_id."""

                                                                                 
    @abstractmethod
    async def upsert_staleness_mark(self, mark: FunctionStalenessMark) -> None:
        """Record a function as stale, requiring targeted re-audit."""

    @abstractmethod
    async def list_stale_functions(
        self, file_path: str = "", run_id: str = ""
    ) -> list[FunctionStalenessMark]:
        """List all stale function marks, optionally filtered."""

    @abstractmethod
    async def clear_staleness_mark(self, file_path: str, function_name: str) -> None:
        """Remove a staleness mark after re-audit completes."""

                                                                                 
    @abstractmethod
    async def upsert_ldra_finding(self, finding: LdraFinding) -> None: ...

    @abstractmethod
    async def list_ldra_findings(
        self, run_id: str = "", file_path: str = ""
    ) -> list[LdraFinding]: ...

    @abstractmethod
    async def upsert_polyspace_finding(self, finding: PolyspaceFinding) -> None: ...

    @abstractmethod
    async def list_polyspace_findings(
        self, run_id: str = ""
    ) -> list[PolyspaceFinding]: ...

    @abstractmethod
    async def upsert_cbmc_result(self, result: CbmcVerificationResult) -> None: ...

    @abstractmethod
    async def get_cbmc_result(self, result_id: str) -> CbmcVerificationResult | None: ...

                                                                                 
    @abstractmethod
    async def upsert_rtm_entry(self, entry: RequirementTraceability) -> None: ...

    @abstractmethod
    async def get_rtm_for_issue(
        self, issue_id: str
    ) -> RequirementTraceability | None: ...

    @abstractmethod
    async def list_rtm_entries(
        self, run_id: str = ""
    ) -> list[RequirementTraceability]: ...

                                                                                
    @abstractmethod
    async def upsert_independence_record(
        self, record: ReviewerIndependenceRecord
    ) -> None: ...

    @abstractmethod
    async def get_independence_record(
        self, fix_attempt_id: str
    ) -> ReviewerIndependenceRecord | None: ...

                                                                                
    @abstractmethod
    async def upsert_sas(self, sas: SoftwareAccomplishmentSummary) -> None: ...

    @abstractmethod
    async def get_sas(self, run_id: str) -> SoftwareAccomplishmentSummary | None: ...

                                                                                
    @abstractmethod
    async def upsert_sci(self, sci: SoftwareConfigurationIndex) -> None: ...

    @abstractmethod
    async def get_sci(self, baseline_id: str) -> SoftwareConfigurationIndex | None: ...

                                                                                 
    @abstractmethod
    async def upsert_convergence_record(self, record: "ConvergenceRecord") -> None:
        """Persist a convergence check result for audit trail and resume logic."""
        ...

    @abstractmethod
    async def list_convergence_records(
        self, run_id: str
    ) -> list["ConvergenceRecord"]:
        """Return all convergence records for a run, ordered by cycle ascending."""
        ...

                                                                                 
    @abstractmethod
    async def log_llm_session(self, session: dict) -> None:
        """
        Persist an LLM call record for cost tracking and reproducibility.
        The session dict contains: run_id, agent_type, model, prompt_tokens,
        completion_tokens, cost_usd, duration_ms, success, error.
        """
        ...

                                                                                
    @abstractmethod
    async def upsert_synthesis_report(self, report: SynthesisReport) -> None:
        """
        Persist a SynthesisReport produced after each audit cycle.

        Records deduplication effectiveness (raw → deduped counts), compound
        finding yield, synthesis model used, and wall-clock duration.
        Idempotent on (run_id, cycle) — re-running the same cycle overwrites
        the previous report for that cycle.
        """
        ...

    @abstractmethod
    async def get_synthesis_report(
        self,
        run_id: str,
        cycle: int | None = None,
    ) -> SynthesisReport | None:
        """
        Retrieve a SynthesisReport.

        If cycle is None, returns the latest (highest cycle) report for the run.
        If cycle is provided, returns the report for that specific cycle.
        Returns None when no matching report exists.
        """
        ...

    @abstractmethod
    async def list_synthesis_reports(
        self,
        run_id: str | None = None,
    ) -> list[SynthesisReport]:
        """
        List SynthesisReports, optionally filtered by run_id.

        Returns empty list (never None) when no reports exist.
        Ordered by (run_id, cycle) ascending so callers can track trends.
        """
        ...

    @abstractmethod
    async def list_compound_findings(
        self,
        run_id: str | None = None,
        severity: str | None = None,
    ) -> list[Issue]:
        """
        List compound findings — Issues with executor_type=SYNTHESIS.

        Cross-domain vulnerabilities detected by SynthesisAgent that no
        single-domain auditor can identify alone.

        Optionally filtered by run_id and/or severity string
        (e.g. 'CRITICAL', 'MAJOR'). Returns empty list (never None).
        """
        ...

    # ── Gap 4: Commit-granularity incremental audit persistence ─────────────

    @abstractmethod
    async def upsert_commit_audit_record(self, record: CommitAuditRecord) -> None:
        """
        Persist a CommitAuditRecord.  Idempotent on id.

        Called by CommitAuditScheduler at each state transition so that:
        • interrupted audits can be resumed on worker restart
        • dashboards can report per-commit compute savings
        • CI systems can poll for DONE/FAILED status
        """
        ...

    @abstractmethod
    async def get_commit_audit_record(self, record_id: str) -> CommitAuditRecord | None:
        """Return the CommitAuditRecord with the given id, or None."""
        ...

    @abstractmethod
    async def get_commit_audit_record_by_hash(
        self, commit_hash: str, run_id: str = ""
    ) -> CommitAuditRecord | None:
        """
        Return the most recent CommitAuditRecord for a given commit_hash.

        When run_id is supplied, scopes the lookup to that run.  Used by
        the webhook endpoint to avoid scheduling duplicate audits for the
        same commit.
        """
        ...

    @abstractmethod
    async def list_commit_audit_records(
        self,
        run_id: str = "",
        status: CommitAuditStatus | None = None,
        limit: int = 100,
    ) -> list[CommitAuditRecord]:
        """
        List CommitAuditRecords, optionally filtered by run_id and/or status.

        Returns empty list (never None).  Ordered by created_at descending so
        callers see the most recent commits first.
        """
        ...
