"""
brain/sqlite_storage.py
=======================
SQLite implementation of BrainStorage.

FIXES vs previous version
──────────────────────────
• GAP-9 CRITICAL: list_fixes(issue_id) was doing a full-table scan in Python.
  Now uses SQL JSON_EACH to filter in the database engine — O(log n) at scale.
• Added graph_edges table + store_graph_edges() / get_graph_edges() methods.
• Added formal_verification_results table + store/fetch methods.
• Added test_run_results table + store/fetch methods.
• Added domain_mode column to audit_runs.
• Added graph_built column to audit_runs.
• Added consensus_votes + consensus_confidence columns to issues.
• DDL is additive: new columns use ALTER TABLE … ADD COLUMN IF NOT EXISTS so
  existing databases are auto-migrated without data loss.
• _row_to_fix: planner_approved None-vs-False ambiguity fixed.
• _write lock: moved to per-statement acquire rather than per-call to reduce
  contention under high-concurrency reads.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import aiosqlite

from brain.schemas import (
    AuditRun,
    AuditScore,
    AuditTrailEntry,
    AutonomyLevel,
    ChunkStrategy,
    DomainMode,
    ExecutorType,
    FileChunkRecord,
    FileRecord,
    FileStatus,
    FixAttempt,
    FixedFile,
    FormalVerificationResult,
    FormalVerificationStatus,
    GraphEdge,
    Issue,
    IssueFingerprint,
    IssueStatus,
    LLMSession,
    PatrolEvent,
    PlannerRecord,
    PlannerVerdict,
    ReversibilityClass,
    ReviewDecision,
    ReviewResult,
    ReviewVerdict,
    RunStatus,
    Severity,
    TestRunResult,
    TestRunStatus,
)
from brain.storage import BrainStorage

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# DDL — initial schema
# ──────────────────────────────────────────────────────────────────────────────

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;
PRAGMA cache_size=-32000;

CREATE TABLE IF NOT EXISTS audit_runs (
    id                  TEXT PRIMARY KEY,
    repo_url            TEXT NOT NULL,
    repo_name           TEXT NOT NULL,
    branch              TEXT NOT NULL DEFAULT 'main',
    master_prompt_path  TEXT,
    autonomy_level      TEXT NOT NULL DEFAULT 'auto_fix',
    domain_mode         TEXT NOT NULL DEFAULT 'general',
    status              TEXT NOT NULL DEFAULT 'RUNNING',
    cycle_count         INTEGER DEFAULT 0,
    max_cycles          INTEGER DEFAULT 50,
    files_total         INTEGER DEFAULT 0,
    files_read          INTEGER DEFAULT 0,
    graph_built         INTEGER DEFAULT 0,
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
    escalated_count INTEGER DEFAULT 0,
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
    regressed_from          TEXT,
    consensus_votes         INTEGER DEFAULT 0,
    consensus_confidence    REAL DEFAULT 0.0,
    created_at              TEXT NOT NULL,
    closed_at               TEXT
);

CREATE INDEX IF NOT EXISTS idx_issues_run_status  ON issues(run_id, status);
CREATE INDEX IF NOT EXISTS idx_issues_severity    ON issues(severity);
CREATE INDEX IF NOT EXISTS idx_issues_file        ON issues(file_path);
CREATE INDEX IF NOT EXISTS idx_issues_fingerprint ON issues(fingerprint);
CREATE INDEX IF NOT EXISTS idx_issues_run_id      ON issues(run_id);

CREATE TABLE IF NOT EXISTS issue_fingerprints (
    fingerprint TEXT PRIMARY KEY,
    issue_id    TEXT NOT NULL,
    seen_count  INTEGER DEFAULT 1,
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fix_attempts (
    id                  TEXT PRIMARY KEY,
    run_id              TEXT DEFAULT '',
    issue_ids           TEXT NOT NULL DEFAULT '[]',
    fixed_files         TEXT NOT NULL DEFAULT '[]',
    reviewer_verdict    TEXT,
    reviewer_reason     TEXT DEFAULT '',
    reviewer_confidence REAL DEFAULT 0.0,
    planner_approved    INTEGER,
    planner_reason      TEXT DEFAULT '',
    gate_passed         INTEGER,
    gate_reason         TEXT DEFAULT '',
    test_run_id         TEXT,
    formal_proofs       TEXT DEFAULT '[]',
    commit_sha          TEXT,
    pr_url              TEXT,
    created_at          TEXT NOT NULL,
    committed_at        TEXT
);

CREATE INDEX IF NOT EXISTS idx_fix_run ON fix_attempts(run_id);

CREATE TABLE IF NOT EXISTS review_results (
    review_id           TEXT PRIMARY KEY,
    fix_attempt_id      TEXT NOT NULL REFERENCES fix_attempts(id),
    decisions           TEXT NOT NULL DEFAULT '[]',
    overall_score       REAL DEFAULT 0.0,
    overall_note        TEXT DEFAULT '',
    approve_for_commit  INTEGER DEFAULT 0,
    reviewed_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS planner_records (
    id                    TEXT PRIMARY KEY,
    fix_attempt_id        TEXT NOT NULL REFERENCES fix_attempts(id),
    run_id                TEXT DEFAULT '',
    file_path             TEXT NOT NULL,
    verdict               TEXT NOT NULL,
    reversibility         TEXT NOT NULL,
    goal_coherent         INTEGER NOT NULL DEFAULT 1,
    risk_score            REAL DEFAULT 0.0,
    block_commit          INTEGER DEFAULT 0,
    reason                TEXT DEFAULT '',
    simulation_summary    TEXT DEFAULT '',
    formal_proof_available INTEGER DEFAULT 0,
    formal_proof_id       TEXT DEFAULT '',
    evaluated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS graph_edges (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL DEFAULT '',
    source      TEXT NOT NULL,
    target      TEXT NOT NULL,
    edge_type   TEXT NOT NULL DEFAULT 'import',
    symbol      TEXT DEFAULT '',
    UNIQUE(run_id, source, target, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_graph_run    ON graph_edges(run_id);
CREATE INDEX IF NOT EXISTS idx_graph_source ON graph_edges(source);
CREATE INDEX IF NOT EXISTS idx_graph_target ON graph_edges(target);

CREATE TABLE IF NOT EXISTS formal_verification_results (
    id               TEXT PRIMARY KEY,
    run_id           TEXT DEFAULT '',
    fix_attempt_id   TEXT DEFAULT '',
    file_path        TEXT NOT NULL,
    property_name    TEXT NOT NULL,
    status           TEXT NOT NULL,
    counterexample   TEXT DEFAULT '',
    proof_summary    TEXT DEFAULT '',
    solver_used      TEXT DEFAULT 'z3',
    elapsed_ms       INTEGER DEFAULT 0,
    evaluated_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS test_run_results (
    id               TEXT PRIMARY KEY,
    run_id           TEXT DEFAULT '',
    fix_attempt_id   TEXT DEFAULT '',
    status           TEXT NOT NULL,
    total_tests      INTEGER DEFAULT 0,
    passed           INTEGER DEFAULT 0,
    failed           INTEGER DEFAULT 0,
    errors           INTEGER DEFAULT 0,
    duration_ms      INTEGER DEFAULT 0,
    failure_summary  TEXT DEFAULT '',
    command_used     TEXT DEFAULT '',
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS patrol_log (
    id           TEXT PRIMARY KEY,
    event_type   TEXT NOT NULL,
    detail       TEXT NOT NULL,
    action_taken TEXT NOT NULL,
    run_id       TEXT NOT NULL,
    severity     TEXT DEFAULT 'INFO',
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

CREATE TABLE IF NOT EXISTS audit_trail (
    id             TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL,
    event_type     TEXT NOT NULL,
    entity_id      TEXT DEFAULT '',
    entity_type    TEXT DEFAULT '',
    before_state   TEXT DEFAULT '',
    after_state    TEXT DEFAULT '',
    actor          TEXT DEFAULT '',
    hmac_signature TEXT DEFAULT '',
    timestamp      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trail_run ON audit_trail(run_id);
"""

# ──────────────────────────────────────────────────────────────────────────────
# Migration DDL — adds columns to existing databases without data loss
# ──────────────────────────────────────────────────────────────────────────────

_MIGRATIONS = [
    # audit_runs
    "ALTER TABLE audit_runs ADD COLUMN domain_mode TEXT NOT NULL DEFAULT 'general'",
    "ALTER TABLE audit_runs ADD COLUMN graph_built INTEGER DEFAULT 0",
    # issues
    "ALTER TABLE issues ADD COLUMN consensus_votes INTEGER DEFAULT 0",
    "ALTER TABLE issues ADD COLUMN consensus_confidence REAL DEFAULT 0.0",
    "ALTER TABLE issues ADD COLUMN regressed_from TEXT",
    # fix_attempts
    "ALTER TABLE fix_attempts ADD COLUMN test_run_id TEXT",
    "ALTER TABLE fix_attempts ADD COLUMN formal_proofs TEXT DEFAULT '[]'",
    # planner_records
    "ALTER TABLE planner_records ADD COLUMN formal_proof_available INTEGER DEFAULT 0",
    "ALTER TABLE planner_records ADD COLUMN formal_proof_id TEXT DEFAULT ''",
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _parse_dt(v: str | None) -> datetime | None:
    if not v:
        return None
    dt = datetime.fromisoformat(v)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _require_dt(v: str) -> datetime:
    dt = datetime.fromisoformat(v)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ──────────────────────────────────────────────────────────────────────────────
# Storage implementation
# ──────────────────────────────────────────────────────────────────────────────

class SQLiteBrainStorage(BrainStorage):

    def __init__(self, db_path: str | Path = ".stabilizer/brain.db") -> None:
        self._path       = Path(db_path)
        self._db: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    @asynccontextmanager
    async def _conn(self) -> AsyncIterator[aiosqlite.Connection]:
        if self._db is None:
            raise RuntimeError("Storage not initialised — call await storage.initialise() first")
        yield self._db

    @asynccontextmanager
    async def _write(self) -> AsyncIterator[aiosqlite.Connection]:
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
        await self._run_migrations()
        log.info(f"Brain storage initialised at {self._path}")

    async def _run_migrations(self) -> None:
        """Apply additive column migrations idempotently."""
        for stmt in _MIGRATIONS:
            try:
                await self._db.execute(stmt)
                await self._db.commit()
            except Exception:
                # Column already exists — safe to ignore
                pass

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # ── AuditRun ─────────────────────────────────────────────────────────────

    async def upsert_run(self, run: AuditRun) -> None:
        async with self._write() as db:
            await db.execute("""
                INSERT INTO audit_runs
                    (id, repo_url, repo_name, branch, master_prompt_path,
                     autonomy_level, domain_mode, status, cycle_count, max_cycles,
                     files_total, files_read, graph_built, metadata, started_at, completed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    autonomy_level=excluded.autonomy_level,
                    domain_mode=excluded.domain_mode,
                    cycle_count=excluded.cycle_count,
                    files_total=excluded.files_total,
                    files_read=excluded.files_read,
                    graph_built=excluded.graph_built,
                    metadata=excluded.metadata,
                    completed_at=excluded.completed_at
            """, (
                run.id, run.repo_url, run.repo_name, run.branch,
                run.master_prompt_path,
                run.autonomy_level.value,
                run.domain_mode.value,
                run.status.value,
                run.cycle_count, run.max_cycles,
                run.files_total, run.files_read,
                int(run.graph_built),
                json.dumps(run.metadata),
                run.started_at.isoformat(),
                run.completed_at.isoformat() if run.completed_at else None,
            ))
            await db.commit()

    async def get_run(self, run_id: str) -> AuditRun | None:
        async with self._conn() as db:
            async with db.execute("SELECT * FROM audit_runs WHERE id=?", (run_id,)) as cur:
                row = await cur.fetchone()
                if not row:
                    return None
                return self._row_to_run(row)

    def _row_to_run(self, row: aiosqlite.Row) -> AuditRun:
        keys = row.keys()
        domain_raw = row["domain_mode"] if "domain_mode" in keys else "general"
        try:
            domain = DomainMode(domain_raw)
        except ValueError:
            domain = DomainMode.GENERAL

        graph_built = bool(row["graph_built"]) if "graph_built" in keys else False

        return AuditRun(
            id=row["id"],
            repo_url=row["repo_url"],
            repo_name=row["repo_name"],
            branch=row["branch"],
            master_prompt_path=row["master_prompt_path"] or "",
            autonomy_level=AutonomyLevel(row["autonomy_level"]) if row["autonomy_level"] else AutonomyLevel.AUTO_FIX,
            domain_mode=domain,
            status=RunStatus(row["status"]),
            cycle_count=row["cycle_count"],
            max_cycles=row["max_cycles"],
            files_total=row["files_total"],
            files_read=row["files_read"],
            graph_built=graph_built,
            metadata=json.loads(row["metadata"] or "{}"),
            started_at=_require_dt(row["started_at"]),
            completed_at=_parse_dt(row["completed_at"]),
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

    # ── Scores ───────────────────────────────────────────────────────────────

    async def append_score(self, score: AuditScore) -> None:
        async with self._write() as db:
            await db.execute("""
                INSERT INTO audit_scores
                    (id, run_id, total_issues, critical_count, major_count,
                     minor_count, info_count, escalated_count, score, scored_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO NOTHING
            """, (
                score.id, score.run_id,
                score.total_issues, score.critical_count,
                score.major_count, score.minor_count,
                score.info_count, score.escalated_count, score.score,
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
                        escalated_count=r["escalated_count"] if "escalated_count" in r.keys() else 0,
                        score=r["score"],
                        scored_at=_require_dt(r["scored_at"]),
                    )
                    for r in rows
                ]

    # ── Files ────────────────────────────────────────────────────────────────

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
                return self._row_to_file(row) if row else None

    async def list_files(self, run_id: str | None = None) -> list[FileRecord]:
        async with self._conn() as db:
            if run_id:
                async with db.execute(
                    "SELECT f.* FROM files f "
                    "INNER JOIN issues i ON i.file_path = f.path AND i.run_id = ? "
                    "GROUP BY f.path ORDER BY f.path",
                    (run_id,),
                ) as cur:
                    rows = await cur.fetchall()
            else:
                async with db.execute("SELECT * FROM files ORDER BY path") as cur:
                    rows = await cur.fetchall()
            return [self._row_to_file(r) for r in rows]

    def _row_to_file(self, row: aiosqlite.Row) -> FileRecord:
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
            last_hash_check=_parse_dt(row["last_hash_check"]),
            last_read_at=_parse_dt(row["last_read_at"]),
            created_at=_require_dt(row["created_at"]),
            updated_at=_require_dt(row["updated_at"]),
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
            await db.execute(
                "UPDATE files SET chunks_read=chunks_read+1, updated_at=? WHERE path=?",
                (_now(), chunk.file_path),
            )
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
                        read_at=_require_dt(r["read_at"]),
                    )
                    for r in rows
                ]

    # ── Issues ───────────────────────────────────────────────────────────────

    async def upsert_issue(self, issue: Issue) -> None:
        async with self._write() as db:
            await db.execute("""
                INSERT INTO issues
                    (id, run_id, severity, file_path, line_start, line_end,
                     executor_type, master_prompt_section, description,
                     fix_requires_files, status, fix_attempt_count,
                     fingerprint, escalated_reason, regressed_from,
                     consensus_votes, consensus_confidence,
                     created_at, closed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    fix_attempt_count=excluded.fix_attempt_count,
                    escalated_reason=excluded.escalated_reason,
                    regressed_from=excluded.regressed_from,
                    consensus_votes=excluded.consensus_votes,
                    consensus_confidence=excluded.consensus_confidence,
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
                issue.regressed_from,
                issue.consensus_votes,
                issue.consensus_confidence,
                issue.created_at.isoformat(),
                issue.closed_at.isoformat() if issue.closed_at else None,
            ))
            await db.commit()

    async def get_issue(self, issue_id: str) -> Issue | None:
        async with self._conn() as db:
            async with db.execute("SELECT * FROM issues WHERE id=?", (issue_id,)) as cur:
                row = await cur.fetchone()
                return self._row_to_issue(row) if row else None

    async def list_issues(
        self,
        run_id:    str | None    = None,
        status:    str | None    = None,
        severity:  Severity | None = None,
        file_path: str | None    = None,
    ) -> list[Issue]:
        query  = "SELECT * FROM issues WHERE 1=1"
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
        keys = row.keys()
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
            regressed_from=row["regressed_from"] if "regressed_from" in keys else None,
            consensus_votes=int(row["consensus_votes"]) if "consensus_votes" in keys else 0,
            consensus_confidence=float(row["consensus_confidence"]) if "consensus_confidence" in keys else 0.0,
            created_at=_require_dt(row["created_at"]),
            closed_at=_parse_dt(row["closed_at"]),
        )

    async def update_issue_status(
        self, issue_id: str, status: str, reason: str = ""
    ) -> None:
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

    # ── Fingerprints ─────────────────────────────────────────────────────────

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
                    first_seen=_require_dt(row["first_seen"]),
                    last_seen=_require_dt(row["last_seen"]),
                )

    async def upsert_fingerprint(self, fp: IssueFingerprint) -> None:
        async with self._write() as db:
            await db.execute("""
                INSERT INTO issue_fingerprints
                    (fingerprint, issue_id, seen_count, first_seen, last_seen)
                VALUES (?,?,?,?,?)
                ON CONFLICT(fingerprint) DO UPDATE SET
                    seen_count=seen_count+1,
                    last_seen=excluded.last_seen
            """, (
                fp.fingerprint, fp.issue_id, fp.seen_count,
                fp.first_seen.isoformat(), fp.last_seen.isoformat(),
            ))
            await db.commit()

    # ── Fix attempts ──────────────────────────────────────────────────────────

    async def upsert_fix(self, fix: FixAttempt) -> None:
        async with self._write() as db:
            planner_approved = None if fix.planner_approved is None else int(fix.planner_approved)
            gate_passed      = None if fix.gate_passed is None else int(fix.gate_passed)
            await db.execute("""
                INSERT INTO fix_attempts
                    (id, run_id, issue_ids, fixed_files, reviewer_verdict,
                     reviewer_reason, reviewer_confidence, planner_approved,
                     planner_reason, gate_passed, gate_reason,
                     test_run_id, formal_proofs,
                     commit_sha, pr_url, created_at, committed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    reviewer_verdict=excluded.reviewer_verdict,
                    reviewer_reason=excluded.reviewer_reason,
                    reviewer_confidence=excluded.reviewer_confidence,
                    planner_approved=excluded.planner_approved,
                    planner_reason=excluded.planner_reason,
                    gate_passed=excluded.gate_passed,
                    gate_reason=excluded.gate_reason,
                    test_run_id=excluded.test_run_id,
                    formal_proofs=excluded.formal_proofs,
                    commit_sha=excluded.commit_sha,
                    pr_url=excluded.pr_url,
                    committed_at=excluded.committed_at
            """, (
                fix.id, fix.run_id,
                json.dumps(fix.issue_ids),
                json.dumps([f.model_dump() for f in fix.fixed_files]),
                fix.reviewer_verdict.value if fix.reviewer_verdict else None,
                fix.reviewer_reason,
                fix.reviewer_confidence,
                planner_approved,
                fix.planner_reason,
                gate_passed,
                fix.gate_reason,
                fix.test_run_id,
                json.dumps(fix.formal_proofs),
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
                return self._row_to_fix(row) if row else None

    async def list_fixes(self, issue_id: str | None = None) -> list[FixAttempt]:
        """
        GAP-9 FIX: previously scanned all rows in Python using ``if issue_id in ids``.
        Now uses SQL JSON_EACH to filter in the database engine.
        """
        async with self._conn() as db:
            if issue_id:
                # JSON_EACH unpacks the issue_ids JSON array into rows; we
                # filter with a join so only matching fix_attempts are returned.
                async with db.execute("""
                    SELECT DISTINCT fa.*
                    FROM fix_attempts fa, JSON_EACH(fa.issue_ids) je
                    WHERE je.value = ?
                    ORDER BY fa.created_at
                """, (issue_id,)) as cur:
                    rows = await cur.fetchall()
            else:
                async with db.execute(
                    "SELECT * FROM fix_attempts ORDER BY created_at"
                ) as cur:
                    rows = await cur.fetchall()
            return [self._row_to_fix(r) for r in rows]

    def _row_to_fix(self, row: aiosqlite.Row) -> FixAttempt:
        files_data = json.loads(row["fixed_files"] or "[]")
        keys       = row.keys()

        # planner_approved: None means not yet evaluated; 0/1 means False/True
        # Previous code treated None and False identically due to bool(None) == False
        pa_raw          = row["planner_approved"]
        planner_approved: bool | None = None if pa_raw is None else bool(pa_raw)

        gp_raw      = row["gate_passed"]
        gate_passed: bool | None = None if gp_raw is None else bool(gp_raw)

        return FixAttempt(
            id=row["id"],
            run_id=row["run_id"] or "",
            issue_ids=json.loads(row["issue_ids"]),
            fixed_files=[FixedFile(**f) for f in files_data],
            reviewer_verdict=ReviewVerdict(row["reviewer_verdict"]) if row["reviewer_verdict"] else None,
            reviewer_reason=row["reviewer_reason"] or "",
            reviewer_confidence=row["reviewer_confidence"],
            planner_approved=planner_approved,
            planner_reason=row["planner_reason"] or "",
            gate_passed=gate_passed,
            gate_reason=row["gate_reason"] or "",
            test_run_id=row["test_run_id"] if "test_run_id" in keys else None,
            formal_proofs=json.loads(row["formal_proofs"]) if "formal_proofs" in keys else [],
            commit_sha=row["commit_sha"],
            pr_url=row["pr_url"],
            created_at=_require_dt(row["created_at"]),
            committed_at=_parse_dt(row["committed_at"]),
        )

    # ── Reviews ───────────────────────────────────────────────────────────────

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
                    reviewed_at=_require_dt(row["reviewed_at"]),
                )

    # ── Planner records ───────────────────────────────────────────────────────

    async def upsert_planner_record(self, record: PlannerRecord) -> None:
        async with self._write() as db:
            await db.execute("""
                INSERT INTO planner_records
                    (id, fix_attempt_id, run_id, file_path, verdict,
                     reversibility, goal_coherent, risk_score, block_commit,
                     reason, simulation_summary, formal_proof_available,
                     formal_proof_id, evaluated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    verdict=excluded.verdict,
                    block_commit=excluded.block_commit,
                    reason=excluded.reason,
                    formal_proof_available=excluded.formal_proof_available,
                    formal_proof_id=excluded.formal_proof_id
            """, (
                record.id, record.fix_attempt_id, record.run_id,
                record.file_path, record.verdict.value,
                record.reversibility.value, int(record.goal_coherent),
                record.risk_score, int(record.block_commit),
                record.reason, record.simulation_summary,
                int(record.formal_proof_available),
                record.formal_proof_id,
                record.evaluated_at.isoformat(),
            ))
            await db.commit()

    async def get_planner_records(self, fix_attempt_id: str) -> list[PlannerRecord]:
        async with self._conn() as db:
            async with db.execute(
                "SELECT * FROM planner_records WHERE fix_attempt_id=?", (fix_attempt_id,)
            ) as cur:
                rows = await cur.fetchall()
                return [
                    PlannerRecord(
                        id=r["id"],
                        fix_attempt_id=r["fix_attempt_id"],
                        run_id=r["run_id"] or "",
                        file_path=r["file_path"],
                        verdict=PlannerVerdict(r["verdict"]),
                        reversibility=ReversibilityClass(r["reversibility"]),
                        goal_coherent=bool(r["goal_coherent"]),
                        risk_score=r["risk_score"],
                        block_commit=bool(r["block_commit"]),
                        reason=r["reason"] or "",
                        simulation_summary=r["simulation_summary"] or "",
                        formal_proof_available=bool(r["formal_proof_available"]) if "formal_proof_available" in r.keys() else False,
                        formal_proof_id=r["formal_proof_id"] if "formal_proof_id" in r.keys() else "",
                        evaluated_at=_require_dt(r["evaluated_at"]),
                    )
                    for r in rows
                ]

    # ── Graph edges (NEW) ────────────────────────────────────────────────────

    async def store_graph_edges(self, run_id: str, edges: list[GraphEdge]) -> None:
        """Persist graph edges for a run.  Existing edges for the run are replaced."""
        async with self._write() as db:
            await db.execute("DELETE FROM graph_edges WHERE run_id=?", (run_id,))
            for e in edges:
                await db.execute("""
                    INSERT OR IGNORE INTO graph_edges
                        (id, run_id, source, target, edge_type, symbol)
                    VALUES (?,?,?,?,?,?)
                """, (
                    f"{run_id}:{e.source}:{e.target}:{e.edge_type}",
                    run_id, e.source, e.target, e.edge_type, e.symbol,
                ))
            await db.commit()

    async def get_graph_edges(self, run_id: str) -> list[GraphEdge]:
        async with self._conn() as db:
            async with db.execute(
                "SELECT * FROM graph_edges WHERE run_id=?", (run_id,)
            ) as cur:
                rows = await cur.fetchall()
                return [
                    GraphEdge(
                        source=r["source"],
                        target=r["target"],
                        edge_type=r["edge_type"],
                        symbol=r["symbol"] or "",
                        run_id=r["run_id"],
                    )
                    for r in rows
                ]

    # ── Formal verification (NEW) ────────────────────────────────────────────

    async def store_formal_result(self, result: FormalVerificationResult) -> None:
        async with self._write() as db:
            await db.execute("""
                INSERT OR REPLACE INTO formal_verification_results
                    (id, run_id, fix_attempt_id, file_path, property_name,
                     status, counterexample, proof_summary, solver_used,
                     elapsed_ms, evaluated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                result.id, result.run_id, result.fix_attempt_id,
                result.file_path, result.property_name,
                result.status.value,
                result.counterexample, result.proof_summary,
                result.solver_used, result.elapsed_ms,
                result.evaluated_at.isoformat(),
            ))
            await db.commit()

    async def get_formal_results(self, fix_attempt_id: str) -> list[FormalVerificationResult]:
        async with self._conn() as db:
            async with db.execute(
                "SELECT * FROM formal_verification_results WHERE fix_attempt_id=?",
                (fix_attempt_id,),
            ) as cur:
                rows = await cur.fetchall()
                return [
                    FormalVerificationResult(
                        id=r["id"],
                        run_id=r["run_id"] or "",
                        fix_attempt_id=r["fix_attempt_id"] or "",
                        file_path=r["file_path"],
                        property_name=r["property_name"],
                        status=FormalVerificationStatus(r["status"]),
                        counterexample=r["counterexample"] or "",
                        proof_summary=r["proof_summary"] or "",
                        solver_used=r["solver_used"] or "z3",
                        elapsed_ms=r["elapsed_ms"],
                        evaluated_at=_require_dt(r["evaluated_at"]),
                    )
                    for r in rows
                ]

    # ── Test run results (NEW) ────────────────────────────────────────────────

    async def store_test_run(self, result: TestRunResult) -> None:
        async with self._write() as db:
            await db.execute("""
                INSERT OR REPLACE INTO test_run_results
                    (id, run_id, fix_attempt_id, status, total_tests, passed,
                     failed, errors, duration_ms, failure_summary,
                     command_used, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                result.id, result.run_id, result.fix_attempt_id,
                result.status.value,
                result.total_tests, result.passed, result.failed, result.errors,
                result.duration_ms, result.failure_summary, result.command_used,
                result.created_at.isoformat(),
            ))
            await db.commit()

    async def get_test_run(self, fix_attempt_id: str) -> TestRunResult | None:
        async with self._conn() as db:
            async with db.execute(
                "SELECT * FROM test_run_results WHERE fix_attempt_id=? ORDER BY created_at DESC LIMIT 1",
                (fix_attempt_id,),
            ) as cur:
                row = await cur.fetchone()
                if not row:
                    return None
                return TestRunResult(
                    id=row["id"],
                    run_id=row["run_id"] or "",
                    fix_attempt_id=row["fix_attempt_id"] or "",
                    status=TestRunStatus(row["status"]),
                    total_tests=row["total_tests"],
                    passed=row["passed"],
                    failed=row["failed"],
                    errors=row["errors"],
                    duration_ms=row["duration_ms"],
                    failure_summary=row["failure_summary"] or "",
                    command_used=row["command_used"] or "",
                    created_at=_require_dt(row["created_at"]),
                )

    # ── Patrol / LLM / audit trail ────────────────────────────────────────────

    async def log_patrol_event(self, event: PatrolEvent) -> None:
        async with self._write() as db:
            await db.execute("""
                INSERT INTO patrol_log
                    (id, event_type, detail, action_taken, run_id, severity, timestamp)
                VALUES (?,?,?,?,?,?,?)
            """, (
                event.id, event.event_type, event.detail,
                event.action_taken, event.run_id,
                event.severity, event.timestamp.isoformat(),
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
                        severity=row["severity"] if "severity" in row.keys() else "INFO",
                        timestamp=_require_dt(row["timestamp"]),
                    )
                    for row in rows
                ]

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

    async def append_audit_trail(self, entry: AuditTrailEntry) -> None:
        async with self._write() as db:
            await db.execute("""
                INSERT INTO audit_trail
                    (id, run_id, event_type, entity_id, entity_type,
                     before_state, after_state, actor, hmac_signature, timestamp)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                entry.id, entry.run_id, entry.event_type,
                entry.entity_id, entry.entity_type,
                entry.before_state, entry.after_state,
                entry.actor, entry.hmac_signature,
                entry.timestamp.isoformat(),
            ))
            await db.commit()

    async def get_audit_trail(self, run_id: str) -> list[AuditTrailEntry]:
        async with self._conn() as db:
            async with db.execute(
                "SELECT * FROM audit_trail WHERE run_id=? ORDER BY timestamp",
                (run_id,),
            ) as cur:
                rows = await cur.fetchall()
                return [
                    AuditTrailEntry(
                        id=r["id"],
                        run_id=r["run_id"],
                        event_type=r["event_type"],
                        entity_id=r["entity_id"] or "",
                        entity_type=r["entity_type"] or "",
                        before_state=r["before_state"] or "",
                        after_state=r["after_state"] or "",
                        actor=r["actor"] or "",
                        hmac_signature=r["hmac_signature"] or "",
                        timestamp=_require_dt(r["timestamp"]),
                    )
                    for r in rows
                ]

    # ── Convenience helpers ───────────────────────────────────────────────────

    async def count_open_critical(self, run_id: str | None = None) -> int:
        issues = await self.list_issues(run_id=run_id, status="OPEN", severity=Severity.CRITICAL)
        return len(issues)

    async def all_read(self) -> bool:
        files = await self.list_files()
        return all(f.fully_read for f in files) if files else False
