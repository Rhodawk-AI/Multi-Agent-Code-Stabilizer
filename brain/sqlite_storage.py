"""
brain/sqlite_storage.py
Production SQLite brain storage using aiosqlite.
Handles millions of lines of code via incremental indexing.

PATCH LOG:
  - append_score: replaced str(id(score)) memory-address PK with score.id (UUID)
  - upsert_issue: added run_id to INSERT statement (was NULL for every issue, breaking
    all list_issues(run_id=...) queries and multi-run isolation entirely)
  - Added asyncio.Lock for write serialization — aiosqlite with a single shared
    connection is not concurrency-safe for writes without explicit serialization
  - _conn() context manager now uses write_lock for all mutations
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

import aiosqlite

from brain.schemas import (
    AuditRun,
    AuditScore,
    FileChunkRecord,
    FileRecord,
    FileStatus,
    FixAttempt,
    FixedFile,
    Issue,
    IssueFingerprint,
    IssueStatus,
    LLMSession,
    PatrolEvent,
    ReviewDecision,
    ReviewResult,
    ReviewVerdict,
    RunStatus,
    Severity,
)
from brain.storage import BrainStorage

log = logging.getLogger(__name__)

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS audit_runs (
    id                  TEXT PRIMARY KEY,
    repo_url            TEXT NOT NULL,
    repo_name           TEXT NOT NULL,
    branch              TEXT NOT NULL DEFAULT 'main',
    master_prompt_path  TEXT,
    status              TEXT NOT NULL DEFAULT 'RUNNING',
    cycle_count         INTEGER DEFAULT 0,
    max_cycles          INTEGER DEFAULT 50,
    files_total         INTEGER DEFAULT 0,
    files_read          INTEGER DEFAULT 0,
    metadata            TEXT DEFAULT '{}',
    started_at          TEXT NOT NULL,
    completed_at        TEXT
);

CREATE TABLE IF NOT EXISTS audit_scores (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES audit_runs(id),
    total_issues    INTEGER DEFAULT 0,
    critical_count  INTEGER DEFAULT 0,
    major_count     INTEGER DEFAULT 0,
    minor_count     INTEGER DEFAULT 0,
    info_count      INTEGER DEFAULT 0,
    score           REAL DEFAULT 0.0,
    scored_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    path            TEXT PRIMARY KEY,
    content_hash    TEXT,
    size_lines      INTEGER DEFAULT 0,
    size_bytes      INTEGER DEFAULT 0,
    language        TEXT DEFAULT 'unknown',
    status          TEXT DEFAULT 'UNREAD',
    chunk_strategy  TEXT DEFAULT 'FULL',
    chunks_total    INTEGER DEFAULT 0,
    chunks_read     INTEGER DEFAULT 0,
    summary         TEXT DEFAULT '',
    is_load_bearing INTEGER DEFAULT 0,
    last_hash_check TEXT,
    last_read_at    TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS file_chunks (
    chunk_id            TEXT PRIMARY KEY,
    file_path           TEXT NOT NULL REFERENCES files(path),
    chunk_index         INTEGER NOT NULL,
    total_chunks        INTEGER NOT NULL,
    line_start          INTEGER NOT NULL,
    line_end            INTEGER NOT NULL,
    symbols_defined     TEXT DEFAULT '[]',
    symbols_referenced  TEXT DEFAULT '[]',
    dependencies        TEXT DEFAULT '[]',
    summary             TEXT DEFAULT '',
    raw_observations    TEXT DEFAULT '[]',
    token_count         INTEGER DEFAULT 0,
    read_at             TEXT NOT NULL,
    UNIQUE(file_path, chunk_index)
);

CREATE TABLE IF NOT EXISTS issues (
    id                      TEXT PRIMARY KEY,
    run_id                  TEXT REFERENCES audit_runs(id),
    severity                TEXT NOT NULL,
    file_path               TEXT NOT NULL,
    line_start              INTEGER DEFAULT 0,
    line_end                INTEGER DEFAULT 0,
    executor_type           TEXT NOT NULL,
    master_prompt_section   TEXT DEFAULT '',
    description             TEXT NOT NULL,
    fix_requires_files      TEXT DEFAULT '[]',
    status                  TEXT DEFAULT 'OPEN',
    fix_attempt_count       INTEGER DEFAULT 0,
    fingerprint             TEXT DEFAULT '',
    escalated_reason        TEXT,
    created_at              TEXT NOT NULL,
    closed_at               TEXT
);

CREATE INDEX IF NOT EXISTS idx_issues_run_status  ON issues(run_id, status);
CREATE INDEX IF NOT EXISTS idx_issues_severity    ON issues(severity);
CREATE INDEX IF NOT EXISTS idx_issues_file        ON issues(file_path);
CREATE INDEX IF NOT EXISTS idx_issues_fingerprint ON issues(fingerprint);

CREATE TABLE IF NOT EXISTS issue_fingerprints (
    fingerprint TEXT PRIMARY KEY,
    issue_id    TEXT NOT NULL,
    seen_count  INTEGER DEFAULT 1,
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fix_attempts (
    id                  TEXT PRIMARY KEY,
    issue_ids           TEXT NOT NULL DEFAULT '[]',
    fixed_files         TEXT NOT NULL DEFAULT '[]',
    reviewer_verdict    TEXT,
    reviewer_reason     TEXT DEFAULT '',
    reviewer_confidence REAL DEFAULT 0.0,
    commit_sha          TEXT,
    pr_url              TEXT,
    created_at          TEXT NOT NULL,
    committed_at        TEXT
);

CREATE TABLE IF NOT EXISTS review_results (
    review_id           TEXT PRIMARY KEY,
    fix_attempt_id      TEXT NOT NULL REFERENCES fix_attempts(id),
    decisions           TEXT NOT NULL DEFAULT '[]',
    overall_score       REAL DEFAULT 0.0,
    overall_note        TEXT DEFAULT '',
    approve_for_commit  INTEGER DEFAULT 0,
    reviewed_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS patrol_log (
    id           TEXT PRIMARY KEY,
    event_type   TEXT NOT NULL,
    detail       TEXT NOT NULL,
    action_taken TEXT NOT NULL,
    run_id       TEXT NOT NULL,
    timestamp    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_patrol_run ON patrol_log(run_id);

CREATE TABLE IF NOT EXISTS llm_sessions (
    id                  TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL,
    agent_type          TEXT NOT NULL,
    model               TEXT NOT NULL,
    prompt_tokens       INTEGER DEFAULT 0,
    completion_tokens   INTEGER DEFAULT 0,
    cost_usd            REAL DEFAULT 0.0,
    duration_ms         INTEGER DEFAULT 0,
    success             INTEGER DEFAULT 1,
    error               TEXT,
    started_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_llm_run ON llm_sessions(run_id);
"""


def _now() -> str:
    return datetime.utcnow().isoformat()


class SQLiteBrainStorage(BrainStorage):
    def __init__(self, db_path: str | Path = ".stabilizer/brain.db") -> None:
        self._path = Path(db_path)
        self._db: aiosqlite.Connection | None = None
        # FIX: asyncio.Lock for write serialization.
        # aiosqlite wraps sqlite3 in a thread but a single shared connection is
        # not safe for concurrent async writers without explicit serialization.
        self._write_lock = asyncio.Lock()

    @asynccontextmanager
    async def _conn(self) -> AsyncIterator[aiosqlite.Connection]:
        if self._db is None:
            raise RuntimeError("Storage not initialised — call await storage.initialise() first")
        yield self._db

    @asynccontextmanager
    async def _write(self) -> AsyncIterator[aiosqlite.Connection]:
        """Context manager for write operations — serialises concurrent writes."""
        async with self._write_lock:
            if self._db is None:
                raise RuntimeError("Storage not initialised — call await storage.initialise() first")
            yield self._db

    async def initialise(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._path))
        self._db.row_factory = aiosqlite.Row
        async with self._db.executescript(DDL):
            pass
        await self._db.commit()
        log.info(f"Brain storage initialised at {self._path}")

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # ── Audit Run ─────────────────────────────────────────────────────────

    async def upsert_run(self, run: AuditRun) -> None:
        async with self._write() as db:
            await db.execute("""
                INSERT INTO audit_runs
                    (id, repo_url, repo_name, branch, master_prompt_path,
                     status, cycle_count, max_cycles, files_total, files_read,
                     metadata, started_at, completed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    cycle_count=excluded.cycle_count,
                    files_total=excluded.files_total,
                    files_read=excluded.files_read,
                    metadata=excluded.metadata,
                    completed_at=excluded.completed_at
            """, (
                run.id, run.repo_url, run.repo_name, run.branch,
                run.master_prompt_path, run.status.value,
                run.cycle_count, run.max_cycles,
                run.files_total, run.files_read,
                json.dumps(run.metadata),
                run.started_at.isoformat(),
                run.completed_at.isoformat() if run.completed_at else None,
            ))
            await db.commit()

    async def get_run(self, run_id: str) -> AuditRun | None:
        async with self._conn() as db:
            async with db.execute(
                "SELECT * FROM audit_runs WHERE id=?", (run_id,)
            ) as cur:
                row = await cur.fetchone()
                if not row:
                    return None
                return AuditRun(
                    id=row["id"],
                    repo_url=row["repo_url"],
                    repo_name=row["repo_name"],
                    branch=row["branch"],
                    master_prompt_path=row["master_prompt_path"] or "",
                    status=RunStatus(row["status"]),
                    cycle_count=row["cycle_count"],
                    max_cycles=row["max_cycles"],
                    files_total=row["files_total"],
                    files_read=row["files_read"],
                    metadata=json.loads(row["metadata"] or "{}"),
                    started_at=datetime.fromisoformat(row["started_at"]),
                    completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                )

    async def update_run_status(self, run_id: str, status: RunStatus) -> None:
        async with self._write() as db:
            completed = _now() if status in (
                RunStatus.STABILIZED, RunStatus.HALTED,
                RunStatus.FAILED, RunStatus.ESCALATED,
            ) else None
            await db.execute(
                "UPDATE audit_runs SET status=?, completed_at=? WHERE id=?",
                (status.value, completed, run_id),
            )
            await db.commit()

    async def append_score(self, score: AuditScore) -> None:
        # FIX: was using str(id(score)) — Python object memory address — as PK.
        # This is non-deterministic and guaranteed to collide across interpreter sessions.
        # AuditScore now has a proper UUID id field.
        async with self._write() as db:
            await db.execute("""
                INSERT INTO audit_scores
                    (id, run_id, total_issues, critical_count, major_count,
                     minor_count, info_count, score, scored_at)
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO NOTHING
            """, (
                score.id, score.run_id,
                score.total_issues, score.critical_count,
                score.major_count, score.minor_count,
                score.info_count, score.score,
                score.scored_at.isoformat(),
            ))
            await db.commit()

    async def get_scores(self, run_id: str) -> list[AuditScore]:
        async with self._conn() as db:
            async with db.execute(
                "SELECT * FROM audit_scores WHERE run_id=? ORDER BY scored_at",
                (run_id,),
            ) as cur:
                rows = await cur.fetchall()
                return [
                    AuditScore(
                        id=r["id"],
                        run_id=r["run_id"],
                        total_issues=r["total_issues"],
                        critical_count=r["critical_count"],
                        major_count=r["major_count"],
                        minor_count=r["minor_count"],
                        info_count=r["info_count"],
                        score=r["score"],
                        scored_at=datetime.fromisoformat(r["scored_at"]),
                    )
                    for r in rows
                ]

    # ── Files ──────────────────────────────────────────────────────────────

    async def upsert_file(self, record: FileRecord) -> None:
        async with self._write() as db:
            await db.execute("""
                INSERT INTO files
                    (path, content_hash, size_lines, size_bytes, language,
                     status, chunk_strategy, chunks_total, chunks_read,
                     summary, is_load_bearing, last_hash_check, last_read_at,
                     created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(path) DO UPDATE SET
                    content_hash=excluded.content_hash,
                    size_lines=excluded.size_lines,
                    size_bytes=excluded.size_bytes,
                    status=excluded.status,
                    chunk_strategy=excluded.chunk_strategy,
                    chunks_total=excluded.chunks_total,
                    chunks_read=excluded.chunks_read,
                    summary=excluded.summary,
                    is_load_bearing=excluded.is_load_bearing,
                    last_hash_check=excluded.last_hash_check,
                    last_read_at=excluded.last_read_at,
                    updated_at=excluded.updated_at
            """, (
                record.path, record.content_hash,
                record.size_lines, record.size_bytes,
                record.language, record.status.value,
                record.chunk_strategy.value,
                record.chunks_total, record.chunks_read,
                record.summary, int(record.is_load_bearing),
                record.last_hash_check.isoformat() if record.last_hash_check else None,
                record.last_read_at.isoformat() if record.last_read_at else None,
                record.created_at.isoformat(),
                _now(),
            ))
            await db.commit()

    async def get_file(self, path: str) -> FileRecord | None:
        async with self._conn() as db:
            async with db.execute("SELECT * FROM files WHERE path=?", (path,)) as cur:
                row = await cur.fetchone()
                if not row:
                    return None
                return self._row_to_file(row)

    async def list_files(self, run_id: str | None = None) -> list[FileRecord]:
        async with self._conn() as db:
            async with db.execute("SELECT * FROM files ORDER BY path") as cur:
                rows = await cur.fetchall()
                return [self._row_to_file(r) for r in rows]

    def _row_to_file(self, row: aiosqlite.Row) -> FileRecord:
        from brain.schemas import ChunkStrategy
        return FileRecord(
            path=row["path"],
            content_hash=row["content_hash"] or "",
            size_lines=row["size_lines"],
            size_bytes=row["size_bytes"],
            language=row["language"],
            status=FileStatus(row["status"]),
            chunk_strategy=ChunkStrategy(row["chunk_strategy"]),
            chunks_total=row["chunks_total"],
            chunks_read=row["chunks_read"],
            summary=row["summary"] or "",
            is_load_bearing=bool(row["is_load_bearing"]),
            last_hash_check=datetime.fromisoformat(row["last_hash_check"]) if row["last_hash_check"] else None,
            last_read_at=datetime.fromisoformat(row["last_read_at"]) if row["last_read_at"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    async def mark_file_read(self, path: str, summary: str) -> None:
        async with self._write() as db:
            await db.execute("""
                UPDATE files
                SET status='READ', summary=?, last_read_at=?, updated_at=?,
                    chunks_read=chunks_total
                WHERE path=?
            """, (summary, _now(), _now(), path))
            await db.commit()

    async def append_chunk(self, chunk: FileChunkRecord) -> None:
        async with self._write() as db:
            await db.execute("""
                INSERT OR REPLACE INTO file_chunks
                    (chunk_id, file_path, chunk_index, total_chunks,
                     line_start, line_end, symbols_defined, symbols_referenced,
                     dependencies, summary, raw_observations, token_count, read_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                chunk.chunk_id, chunk.file_path,
                chunk.chunk_index, chunk.total_chunks,
                chunk.line_start, chunk.line_end,
                json.dumps(chunk.symbols_defined),
                json.dumps(chunk.symbols_referenced),
                json.dumps(chunk.dependencies),
                chunk.summary,
                json.dumps(chunk.raw_observations),
                chunk.token_count,
                chunk.read_at.isoformat(),
            ))
            # Update chunks_read counter
            await db.execute("""
                UPDATE files SET chunks_read=chunks_read+1, updated_at=?
                WHERE path=?
            """, (_now(), chunk.file_path))
            await db.commit()

    async def get_chunks(self, file_path: str) -> list[FileChunkRecord]:
        async with self._conn() as db:
            async with db.execute(
                "SELECT * FROM file_chunks WHERE file_path=? ORDER BY chunk_index",
                (file_path,),
            ) as cur:
                rows = await cur.fetchall()
                return [
                    FileChunkRecord(
                        chunk_id=r["chunk_id"],
                        file_path=r["file_path"],
                        chunk_index=r["chunk_index"],
                        total_chunks=r["total_chunks"],
                        line_start=r["line_start"],
                        line_end=r["line_end"],
                        symbols_defined=json.loads(r["symbols_defined"]),
                        symbols_referenced=json.loads(r["symbols_referenced"]),
                        dependencies=json.loads(r["dependencies"]),
                        summary=r["summary"],
                        raw_observations=json.loads(r["raw_observations"]),
                        token_count=r["token_count"],
                        read_at=datetime.fromisoformat(r["read_at"]),
                    )
                    for r in rows
                ]

    # ── Issues ─────────────────────────────────────────────────────────────

    async def upsert_issue(self, issue: Issue) -> None:
        # FIX: run_id was completely absent from the INSERT.
        # The DDL column existed, but was never populated.
        # list_issues(run_id=...) therefore always returned empty results,
        # breaking score computation, cost ceilings, and all per-run queries.
        async with self._write() as db:
            await db.execute("""
                INSERT INTO issues
                    (id, run_id, severity, file_path, line_start, line_end,
                     executor_type, master_prompt_section, description,
                     fix_requires_files, status, fix_attempt_count,
                     fingerprint, escalated_reason, created_at, closed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    fix_attempt_count=excluded.fix_attempt_count,
                    escalated_reason=excluded.escalated_reason,
                    closed_at=excluded.closed_at
            """, (
                issue.id, issue.run_id,
                issue.severity.value,
                issue.file_path, issue.line_start, issue.line_end,
                issue.executor_type.value,
                issue.master_prompt_section,
                issue.description,
                json.dumps(issue.fix_requires_files),
                issue.status.value,
                issue.fix_attempt_count,
                issue.fingerprint,
                issue.escalated_reason,
                issue.created_at.isoformat(),
                issue.closed_at.isoformat() if issue.closed_at else None,
            ))
            await db.commit()

    async def get_issue(self, issue_id: str) -> Issue | None:
        async with self._conn() as db:
            async with db.execute("SELECT * FROM issues WHERE id=?", (issue_id,)) as cur:
                row = await cur.fetchone()
                if not row:
                    return None
                return self._row_to_issue(row)

    async def list_issues(
        self,
        run_id: str | None = None,
        status: str | None = None,
        severity: Severity | None = None,
        file_path: str | None = None,
    ) -> list[Issue]:
        query = "SELECT * FROM issues WHERE 1=1"
        params: list = []
        if run_id:
            query += " AND run_id=?"
            params.append(run_id)
        if status:
            query += " AND status=?"
            params.append(status)
        if severity:
            query += " AND severity=?"
            params.append(severity.value)
        if file_path:
            query += " AND file_path=?"
            params.append(file_path)
        query += (
            " ORDER BY CASE severity WHEN 'CRITICAL' THEN 0 WHEN 'MAJOR' THEN 1"
            " WHEN 'MINOR' THEN 2 ELSE 3 END, created_at"
        )
        async with self._conn() as db:
            async with db.execute(query, params) as cur:
                rows = await cur.fetchall()
                return [self._row_to_issue(r) for r in rows]

    def _row_to_issue(self, row: aiosqlite.Row) -> Issue:
        from brain.schemas import ExecutorType
        return Issue(
            id=row["id"],
            run_id=row["run_id"] or "",
            severity=Severity(row["severity"]),
            file_path=row["file_path"],
            line_start=row["line_start"],
            line_end=row["line_end"],
            executor_type=ExecutorType(row["executor_type"]),
            master_prompt_section=row["master_prompt_section"] or "",
            description=row["description"],
            fix_requires_files=json.loads(row["fix_requires_files"] or "[]"),
            status=IssueStatus(row["status"]),
            fix_attempt_count=row["fix_attempt_count"],
            fingerprint=row["fingerprint"] or "",
            escalated_reason=row["escalated_reason"],
            created_at=datetime.fromisoformat(row["created_at"]),
            closed_at=datetime.fromisoformat(row["closed_at"]) if row["closed_at"] else None,
        )

    async def update_issue_status(self, issue_id: str, status: str, reason: str = "") -> None:
        async with self._write() as db:
            closed = _now() if status in ("CLOSED", "ESCALATED") else None
            await db.execute("""
                UPDATE issues SET status=?, escalated_reason=?, closed_at=?
                WHERE id=?
            """, (status, reason or None, closed, issue_id))
            await db.commit()

    async def increment_fix_attempts(self, issue_id: str) -> int:
        async with self._write() as db:
            await db.execute(
                "UPDATE issues SET fix_attempt_count=fix_attempt_count+1 WHERE id=?",
                (issue_id,),
            )
            await db.commit()
            async with db.execute(
                "SELECT fix_attempt_count FROM issues WHERE id=?", (issue_id,)
            ) as cur:
                row = await cur.fetchone()
                return row["fix_attempt_count"] if row else 0

    # ── Fingerprints ───────────────────────────────────────────────────────

    async def get_fingerprint(self, fingerprint: str) -> IssueFingerprint | None:
        async with self._conn() as db:
            async with db.execute(
                "SELECT * FROM issue_fingerprints WHERE fingerprint=?", (fingerprint,)
            ) as cur:
                row = await cur.fetchone()
                if not row:
                    return None
                return IssueFingerprint(
                    fingerprint=row["fingerprint"],
                    issue_id=row["issue_id"],
                    seen_count=row["seen_count"],
                    first_seen=datetime.fromisoformat(row["first_seen"]),
                    last_seen=datetime.fromisoformat(row["last_seen"]),
                )

    async def upsert_fingerprint(self, fp: IssueFingerprint) -> None:
        async with self._write() as db:
            await db.execute("""
                INSERT INTO issue_fingerprints (fingerprint, issue_id, seen_count, first_seen, last_seen)
                VALUES (?,?,?,?,?)
                ON CONFLICT(fingerprint) DO UPDATE SET
                    seen_count=seen_count+1,
                    last_seen=excluded.last_seen
            """, (
                fp.fingerprint, fp.issue_id, fp.seen_count,
                fp.first_seen.isoformat(), fp.last_seen.isoformat(),
            ))
            await db.commit()

    # ── Fix Attempts ───────────────────────────────────────────────────────

    async def upsert_fix(self, fix: FixAttempt) -> None:
        async with self._write() as db:
            await db.execute("""
                INSERT INTO fix_attempts
                    (id, issue_ids, fixed_files, reviewer_verdict, reviewer_reason,
                     reviewer_confidence, commit_sha, pr_url, created_at, committed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    reviewer_verdict=excluded.reviewer_verdict,
                    reviewer_reason=excluded.reviewer_reason,
                    reviewer_confidence=excluded.reviewer_confidence,
                    commit_sha=excluded.commit_sha,
                    pr_url=excluded.pr_url,
                    committed_at=excluded.committed_at
            """, (
                fix.id,
                json.dumps(fix.issue_ids),
                json.dumps([f.model_dump() for f in fix.fixed_files]),
                fix.reviewer_verdict.value if fix.reviewer_verdict else None,
                fix.reviewer_reason,
                fix.reviewer_confidence,
                fix.commit_sha,
                fix.pr_url,
                fix.created_at.isoformat(),
                fix.committed_at.isoformat() if fix.committed_at else None,
            ))
            await db.commit()

    async def get_fix(self, fix_id: str) -> FixAttempt | None:
        async with self._conn() as db:
            async with db.execute("SELECT * FROM fix_attempts WHERE id=?", (fix_id,)) as cur:
                row = await cur.fetchone()
                if not row:
                    return None
                files_data = json.loads(row["fixed_files"] or "[]")
                return FixAttempt(
                    id=row["id"],
                    issue_ids=json.loads(row["issue_ids"]),
                    fixed_files=[FixedFile(**f) for f in files_data],
                    reviewer_verdict=ReviewVerdict(row["reviewer_verdict"]) if row["reviewer_verdict"] else None,
                    reviewer_reason=row["reviewer_reason"] or "",
                    reviewer_confidence=row["reviewer_confidence"],
                    commit_sha=row["commit_sha"],
                    pr_url=row["pr_url"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    committed_at=datetime.fromisoformat(row["committed_at"]) if row["committed_at"] else None,
                )

    async def list_fixes(self, issue_id: str | None = None) -> list[FixAttempt]:
        async with self._conn() as db:
            if issue_id:
                async with db.execute(
                    "SELECT * FROM fix_attempts WHERE issue_ids LIKE ? ORDER BY created_at",
                    (f"%{issue_id}%",),
                ) as cur:
                    rows = await cur.fetchall()
            else:
                async with db.execute("SELECT * FROM fix_attempts ORDER BY created_at") as cur:
                    rows = await cur.fetchall()
            result = []
            for row in rows:
                files_data = json.loads(row["fixed_files"] or "[]")
                result.append(FixAttempt(
                    id=row["id"],
                    issue_ids=json.loads(row["issue_ids"]),
                    fixed_files=[FixedFile(**f) for f in files_data],
                    reviewer_verdict=ReviewVerdict(row["reviewer_verdict"]) if row["reviewer_verdict"] else None,
                    reviewer_reason=row["reviewer_reason"] or "",
                    reviewer_confidence=row["reviewer_confidence"],
                    commit_sha=row["commit_sha"],
                    pr_url=row["pr_url"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    committed_at=datetime.fromisoformat(row["committed_at"]) if row["committed_at"] else None,
                ))
            return result

    # ── Reviews ────────────────────────────────────────────────────────────

    async def upsert_review(self, review: ReviewResult) -> None:
        async with self._write() as db:
            await db.execute("""
                INSERT INTO review_results
                    (review_id, fix_attempt_id, decisions, overall_score,
                     overall_note, approve_for_commit, reviewed_at)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(review_id) DO UPDATE SET
                    decisions=excluded.decisions,
                    overall_score=excluded.overall_score,
                    overall_note=excluded.overall_note,
                    approve_for_commit=excluded.approve_for_commit
            """, (
                review.review_id,
                review.fix_attempt_id,
                json.dumps([d.model_dump() for d in review.decisions]),
                review.overall_score,
                review.overall_note,
                int(review.approve_for_commit),
                review.reviewed_at.isoformat(),
            ))
            await db.commit()

    async def get_review(self, fix_attempt_id: str) -> ReviewResult | None:
        async with self._conn() as db:
            async with db.execute(
                "SELECT * FROM review_results WHERE fix_attempt_id=?", (fix_attempt_id,)
            ) as cur:
                row = await cur.fetchone()
                if not row:
                    return None
                decisions_data = json.loads(row["decisions"] or "[]")
                return ReviewResult(
                    review_id=row["review_id"],
                    fix_attempt_id=row["fix_attempt_id"],
                    decisions=[ReviewDecision(**d) for d in decisions_data],
                    overall_score=row["overall_score"],
                    overall_note=row["overall_note"] or "",
                    approve_for_commit=bool(row["approve_for_commit"]),
                    reviewed_at=datetime.fromisoformat(row["reviewed_at"]),
                )

    # ── Patrol ─────────────────────────────────────────────────────────────

    async def log_patrol_event(self, event: PatrolEvent) -> None:
        async with self._write() as db:
            await db.execute("""
                INSERT INTO patrol_log (id, event_type, detail, action_taken, run_id, timestamp)
                VALUES (?,?,?,?,?,?)
            """, (
                event.id, event.event_type, event.detail,
                event.action_taken, event.run_id, event.timestamp.isoformat(),
            ))
            await db.commit()

    async def get_patrol_events(self, run_id: str) -> list[PatrolEvent]:
        async with self._conn() as db:
            async with db.execute(
                "SELECT * FROM patrol_log WHERE run_id=? ORDER BY timestamp",
                (run_id,),
            ) as cur:
                rows = await cur.fetchall()
                return [
                    PatrolEvent(
                        id=row["id"],
                        event_type=row["event_type"],
                        detail=row["detail"],
                        action_taken=row["action_taken"],
                        run_id=row["run_id"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                    )
                    for row in rows
                ]

    # ── LLM sessions ───────────────────────────────────────────────────────

    async def log_llm_session(self, session: LLMSession) -> None:
        async with self._write() as db:
            await db.execute("""
                INSERT INTO llm_sessions
                    (id, run_id, agent_type, model, prompt_tokens,
                     completion_tokens, cost_usd, duration_ms, success, error, started_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                session.id, session.run_id, session.agent_type.value,
                session.model, session.prompt_tokens,
                session.completion_tokens, session.cost_usd,
                session.duration_ms, int(session.success),
                session.error, session.started_at.isoformat(),
            ))
            await db.commit()

    async def get_total_cost(self, run_id: str) -> float:
        async with self._conn() as db:
            async with db.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) as total FROM llm_sessions WHERE run_id=?",
                (run_id,),
            ) as cur:
                row = await cur.fetchone()
                return float(row["total"]) if row else 0.0
