from __future__ import annotations

from abc import ABC, abstractmethod

from brain.schemas import (
    AuditRun,
    AuditScore,
    AuditTrailEntry,
    FileChunkRecord,
    FileRecord,
    FixAttempt,
    Issue,
    IssueFingerprint,
    LLMSession,
    PatrolEvent,
    PlannerRecord,
    ReviewResult,
    RunStatus,
    Severity,
)


class BrainStorage(ABC):

    @abstractmethod
    async def initialise(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def upsert_run(self, run: AuditRun) -> None: ...

    @abstractmethod
    async def get_run(self, run_id: str) -> AuditRun | None: ...

    @abstractmethod
    async def update_run_status(self, run_id: str, status: RunStatus) -> None: ...

    @abstractmethod
    async def append_score(self, score: AuditScore) -> None: ...

    @abstractmethod
    async def get_scores(self, run_id: str) -> list[AuditScore]: ...

    @abstractmethod
    async def upsert_file(self, record: FileRecord) -> None: ...

    @abstractmethod
    async def get_file(self, path: str) -> FileRecord | None: ...

    @abstractmethod
    async def list_files(self, run_id: str | None = None) -> list[FileRecord]: ...

    @abstractmethod
    async def mark_file_read(self, path: str, summary: str) -> None: ...

    @abstractmethod
    async def append_chunk(self, chunk: FileChunkRecord) -> None: ...

    @abstractmethod
    async def get_chunks(self, file_path: str) -> list[FileChunkRecord]: ...

    @abstractmethod
    async def upsert_issue(self, issue: Issue) -> None: ...

    @abstractmethod
    async def get_issue(self, issue_id: str) -> Issue | None: ...

    @abstractmethod
    async def list_issues(
        self,
        run_id: str | None = None,
        status: str | None = None,
        severity: Severity | None = None,
        file_path: str | None = None,
    ) -> list[Issue]: ...

    @abstractmethod
    async def update_issue_status(
        self, issue_id: str, status: str, reason: str = ""
    ) -> None: ...

    @abstractmethod
    async def increment_fix_attempts(self, issue_id: str) -> int: ...

    @abstractmethod
    async def get_fingerprint(self, fingerprint: str) -> IssueFingerprint | None: ...

    @abstractmethod
    async def upsert_fingerprint(self, fp: IssueFingerprint) -> None: ...

    @abstractmethod
    async def upsert_fix(self, fix: FixAttempt) -> None: ...

    @abstractmethod
    async def get_fix(self, fix_id: str) -> FixAttempt | None: ...

    @abstractmethod
    async def list_fixes(self, issue_id: str | None = None) -> list[FixAttempt]: ...

    @abstractmethod
    async def upsert_review(self, review: ReviewResult) -> None: ...

    @abstractmethod
    async def get_review(self, fix_attempt_id: str) -> ReviewResult | None: ...

    @abstractmethod
    async def upsert_planner_record(self, record: PlannerRecord) -> None: ...

    @abstractmethod
    async def get_planner_records(self, fix_attempt_id: str) -> list[PlannerRecord]: ...

    @abstractmethod
    async def log_patrol_event(self, event: PatrolEvent) -> None: ...

    @abstractmethod
    async def get_patrol_events(self, run_id: str) -> list[PatrolEvent]: ...

    @abstractmethod
    async def log_llm_session(self, session: LLMSession) -> None: ...

    @abstractmethod
    async def get_total_cost(self, run_id: str) -> float: ...

    @abstractmethod
    async def append_audit_trail(self, entry: AuditTrailEntry) -> None: ...

    @abstractmethod
    async def get_audit_trail(self, run_id: str) -> list[AuditTrailEntry]: ...

    async def count_open_critical(self, run_id: str | None = None) -> int:
        issues = await self.list_issues(run_id=run_id, status="OPEN", severity=Severity.CRITICAL)
        return len(issues)

    async def all_read(self) -> bool:
        files = await self.list_files()
        return all(f.fully_read for f in files) if files else False
