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
    CommitAuditRecord,
    CommitAuditStatus,
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
    PatrolEvent,
    PlannerRecord,
    PlannerVerdict,
    ReversibilityClass,
    ReviewVerdict,
    RunStatus,
    Severity,
    TestRunResult,
    TestRunStatus,
)
from brain.storage import BrainStorage

log = logging.getLogger(__name__)

                                                                                
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
    -- Gap 4 additions: content stored so get_all_observations/get_stale_observations
    -- can return auditable dicts without re-reading every file from disk.
    content             TEXT DEFAULT '',
    function_name       TEXT DEFAULT '',
    language            TEXT DEFAULT 'unknown',
    run_id              TEXT DEFAULT '',
    UNIQUE(file_path, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_file      ON file_chunks(file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_fn        ON file_chunks(function_name);
CREATE INDEX IF NOT EXISTS idx_chunks_run       ON file_chunks(run_id);

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
    id                    TEXT PRIMARY KEY,
    run_id                TEXT DEFAULT '',
    issue_ids             TEXT NOT NULL DEFAULT '[]',
    fixed_files           TEXT NOT NULL DEFAULT '[]',
    reviewer_verdict      TEXT,
    reviewer_reason       TEXT DEFAULT '',
    reviewer_confidence   REAL DEFAULT 0.0,
    planner_approved      INTEGER,
    planner_reason        TEXT DEFAULT '',
    gate_passed           INTEGER,
    gate_reason           TEXT DEFAULT '',
    test_run_id           TEXT,
    formal_proofs         TEXT DEFAULT '[]',
    commit_sha            TEXT,
    pr_url                TEXT,
    created_at            TEXT NOT NULL,
    committed_at          TEXT,
    blast_radius_exceeded INTEGER NOT NULL DEFAULT 0,
    refactor_proposal_id  TEXT    NOT NULL DEFAULT ''
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

CREATE TABLE IF NOT EXISTS function_staleness (
    id            TEXT PRIMARY KEY,
    file_path     TEXT NOT NULL,
    function_name TEXT NOT NULL,
    line_start    INTEGER DEFAULT 0,
    line_end      INTEGER DEFAULT 0,
    stale_reason  TEXT DEFAULT '',
    stale_since   TEXT NOT NULL,
    run_id        TEXT DEFAULT '',
    UNIQUE(file_path, function_name)
);

CREATE INDEX IF NOT EXISTS idx_staleness_run  ON function_staleness(run_id);
CREATE INDEX IF NOT EXISTS idx_staleness_file ON function_staleness(file_path);

CREATE TABLE IF NOT EXISTS commit_audit_records (
    id                       TEXT PRIMARY KEY,
    run_id                   TEXT NOT NULL DEFAULT '',
    commit_hash              TEXT NOT NULL DEFAULT '',
    branch                   TEXT NOT NULL DEFAULT '',
    author                   TEXT NOT NULL DEFAULT '',
    commit_message           TEXT NOT NULL DEFAULT '',
    changed_files            TEXT NOT NULL DEFAULT '[]',
    changed_functions        TEXT NOT NULL DEFAULT '{}',
    all_changed_functions    TEXT NOT NULL DEFAULT '[]',
    new_functions            TEXT NOT NULL DEFAULT '[]',
    deleted_functions        TEXT NOT NULL DEFAULT '[]',
    impact_functions         TEXT NOT NULL DEFAULT '[]',
    impact_files             TEXT NOT NULL DEFAULT '[]',
    audit_targets            TEXT NOT NULL DEFAULT '[]',
    total_changed_functions  INTEGER NOT NULL DEFAULT 0,
    total_impact_functions   INTEGER NOT NULL DEFAULT 0,
    total_functions_to_audit INTEGER NOT NULL DEFAULT 0,
    test_files_to_run        TEXT NOT NULL DEFAULT '[]',
    test_functions_to_run    TEXT NOT NULL DEFAULT '[]',
    status                   TEXT NOT NULL DEFAULT 'PENDING',
    cpg_updated              INTEGER NOT NULL DEFAULT 0,
    joern_update_status      TEXT NOT NULL DEFAULT '',
    error_detail             TEXT NOT NULL DEFAULT '',
    created_at               TEXT NOT NULL,
    started_at               TEXT,
    finished_at              TEXT
);

CREATE INDEX IF NOT EXISTS idx_car_run_id      ON commit_audit_records(run_id);
CREATE INDEX IF NOT EXISTS idx_car_commit_hash ON commit_audit_records(commit_hash);
CREATE INDEX IF NOT EXISTS idx_car_status      ON commit_audit_records(status);

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

-- escalations (human escalation workflow, DO-178C compliance)
CREATE TABLE IF NOT EXISTS escalations (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    issue_ids TEXT,
    fix_attempt_id TEXT,
    escalation_type TEXT,
    description TEXT,
    severity TEXT,
    mil882e_category TEXT,
    status TEXT,
    approved_by TEXT,
    approved_at TEXT,
    approval_rationale TEXT,
    risk_acceptance TEXT,
    notified_via TEXT,
    notified_at TEXT,
    timeout_at TEXT,
    created_at TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_escalations_run_id ON escalations(run_id);
CREATE INDEX IF NOT EXISTS idx_escalations_status ON escalations(status);

-- refactor_proposals (blast-radius-exceeding changes requiring human approval)
CREATE TABLE IF NOT EXISTS refactor_proposals (
    id TEXT PRIMARY KEY,
    fix_attempt_id TEXT,
    run_id TEXT,
    issue_ids TEXT,
    changed_functions TEXT,
    affected_function_count INTEGER,
    affected_file_count INTEGER,
    test_files_affected TEXT,
    blast_radius_score REAL,
    importing_modules TEXT,
    importing_module_count INTEGER,
    affected_components TEXT,
    proposed_refactoring TEXT,
    migration_steps TEXT,
    estimated_scope TEXT,
    risks TEXT,
    recommendation TEXT,
    escalation_id TEXT,
    requires_human_review INTEGER,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_refactor_run_id ON refactor_proposals(run_id);

-- baselines (approved score snapshots for regression gating)
CREATE TABLE IF NOT EXISTS baselines (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    baseline_name TEXT,
    software_level TEXT,
    commit_hash TEXT,
    issue_count TEXT,
    score_snapshot REAL,
    file_hashes TEXT,
    approved_by TEXT,
    approved_at TEXT,
    is_active INTEGER DEFAULT 0,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_baselines_run_id    ON baselines(run_id);
CREATE INDEX IF NOT EXISTS idx_baselines_is_active ON baselines(is_active);

-- convergence_records (per-cycle convergence check audit trail)
CREATE TABLE IF NOT EXISTS convergence_records (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    data TEXT
);
CREATE INDEX IF NOT EXISTS idx_convergence_run_id ON convergence_records(run_id);

-- synthesis_reports (Gap 2: per-cycle dedup and compound-finding metrics)
CREATE TABLE IF NOT EXISTS synthesis_reports (
    id      TEXT PRIMARY KEY,
    run_id  TEXT NOT NULL,
    cycle   INTEGER NOT NULL DEFAULT 0,
    data    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_synthesis_reports_run ON synthesis_reports(run_id);

-- ldra_findings (LDRA static analysis results, DO-178C toolchain)
CREATE TABLE IF NOT EXISTS ldra_findings (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    fix_attempt_id TEXT,
    file_path TEXT,
    rule_id TEXT,
    severity TEXT,
    message TEXT,
    line_number INTEGER DEFAULT 0,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_ldra_run_id ON ldra_findings(run_id);

-- polyspace_findings (Polyspace Code Prover results)
CREATE TABLE IF NOT EXISTS polyspace_findings (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    fix_attempt_id TEXT,
    file_path TEXT,
    check_name TEXT,
    color TEXT,
    message TEXT,
    line_number INTEGER DEFAULT 0,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_polyspace_run_id ON polyspace_findings(run_id);

-- cbmc_results (CBMC bounded model checker results)
CREATE TABLE IF NOT EXISTS cbmc_results (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    fix_attempt_id TEXT,
    file_path TEXT,
    function_name TEXT,
    status TEXT,
    counterexample TEXT,
    elapsed_ms INTEGER DEFAULT 0,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_cbmc_run_id ON cbmc_results(run_id);

-- rtm_entries (Requirements Traceability Matrix)
CREATE TABLE IF NOT EXISTS rtm_entries (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    requirement_id TEXT,
    source_file TEXT,
    coverage_status TEXT,
    notes TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_rtm_run_id ON rtm_entries(run_id);

-- independence_records (DO-178C independence verification)
CREATE TABLE IF NOT EXISTS independence_records (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    fix_attempt_id TEXT,
    fixer_model TEXT,
    reviewer_model TEXT,
    same_family INTEGER DEFAULT 0,
    violation_reason TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_independence_run_id ON independence_records(run_id);

-- sas_records (Software Accomplishment Summary)
CREATE TABLE IF NOT EXISTS sas_records (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    data TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_sas_run_id ON sas_records(run_id);

-- sci_records (Software Configuration Index)
CREATE TABLE IF NOT EXISTS sci_records (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    data TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_sci_run_id ON sci_records(run_id);
"""

                                                                                
_MIGRATIONS = [
                
    "ALTER TABLE audit_runs ADD COLUMN domain_mode TEXT NOT NULL DEFAULT 'general'",
    "ALTER TABLE audit_runs ADD COLUMN graph_built INTEGER DEFAULT 0",
            
    "ALTER TABLE issues ADD COLUMN consensus_votes INTEGER DEFAULT 0",
    "ALTER TABLE issues ADD COLUMN consensus_confidence REAL DEFAULT 0.0",
    "ALTER TABLE issues ADD COLUMN regressed_from TEXT",
                  
    "ALTER TABLE fix_attempts ADD COLUMN test_run_id TEXT",
    "ALTER TABLE fix_attempts ADD COLUMN formal_proofs TEXT DEFAULT '[]'",
                     
    "ALTER TABLE planner_records ADD COLUMN formal_proof_available INTEGER DEFAULT 0",
    "ALTER TABLE planner_records ADD COLUMN formal_proof_id TEXT DEFAULT ''",

    # Bug 2 fix: persist blast-radius gate state so get_fix() round-trips correctly.
    "ALTER TABLE fix_attempts ADD COLUMN blast_radius_exceeded INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE fix_attempts ADD COLUMN refactor_proposal_id  TEXT    NOT NULL DEFAULT ''",

    # Gap 4 fix: add content, function_name, language, run_id to file_chunks so
    # get_all_observations / get_stale_observations can serve auditable dicts
    # without re-reading every source file from disk on every audit cycle.
    # All four are additive and safe to apply to existing databases.
    "ALTER TABLE file_chunks ADD COLUMN content       TEXT    DEFAULT ''",
    "ALTER TABLE file_chunks ADD COLUMN function_name TEXT    DEFAULT ''",
    "ALTER TABLE file_chunks ADD COLUMN language      TEXT    DEFAULT 'unknown'",
    "ALTER TABLE file_chunks ADD COLUMN run_id        TEXT    DEFAULT ''",
]


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
                                                        
                pass

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

                                                                               
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

    # ── Chunk persistence (implements BrainStorage abstract contract) ─────────
    #
    # The old names append_chunk / get_chunks did not match the ABC and used
    # the non-existent chunk.chunk_id attribute.  Both are replaced here by
    # the correct upsert_chunk / list_chunks pair that use chunk.id and also
    # persist the new content / function_name / language / run_id columns.

    async def upsert_chunk(self, chunk: FileChunkRecord) -> None:
        """Idempotent upsert on (file_path, chunk_index)."""
        async with self._write() as db:
            await db.execute(
                """
                INSERT INTO file_chunks
                    (chunk_id, file_path, chunk_index, total_chunks,
                     line_start, line_end, symbols_defined, symbols_referenced,
                     dependencies, summary, raw_observations, token_count,
                     read_at, content, function_name, language, run_id)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(file_path, chunk_index) DO UPDATE SET
                    chunk_id        = excluded.chunk_id,
                    total_chunks    = excluded.total_chunks,
                    line_end        = excluded.line_end,
                    symbols_defined = excluded.symbols_defined,
                    symbols_referenced = excluded.symbols_referenced,
                    dependencies    = excluded.dependencies,
                    summary         = excluded.summary,
                    raw_observations = excluded.raw_observations,
                    token_count     = excluded.token_count,
                    read_at         = excluded.read_at,
                    content         = excluded.content,
                    function_name   = excluded.function_name,
                    language        = excluded.language,
                    run_id          = excluded.run_id
                """,
                (
                    chunk.id,
                    chunk.file_path,
                    chunk.chunk_index,
                    chunk.total_chunks,
                    chunk.line_start,
                    chunk.line_end,
                    json.dumps(chunk.symbols_defined),
                    json.dumps(chunk.symbols_referenced),
                    json.dumps(chunk.dependencies),
                    chunk.summary,
                    json.dumps(chunk.raw_observations),
                    getattr(chunk, "token_count", 0),
                    chunk.created_at.isoformat(),
                    chunk.content,
                    chunk.function_name,
                    chunk.language,
                    chunk.run_id,
                ),
            )
            await db.commit()

    async def list_chunks(
        self, file_path: str, run_id: str = ""
    ) -> list[FileChunkRecord]:
        """Return all chunks for a file, ordered by chunk_index."""
        async with self._conn() as db:
            if run_id:
                sql    = ("SELECT * FROM file_chunks "
                          "WHERE file_path=? AND run_id=? ORDER BY chunk_index")
                params: tuple = (file_path, run_id)
            else:
                sql    = ("SELECT * FROM file_chunks "
                          "WHERE file_path=? ORDER BY chunk_index")
                params = (file_path,)
            async with db.execute(sql, params) as cur:
                rows = await cur.fetchall()
                return [self._row_to_chunk(r) for r in rows]

    def _row_to_chunk(self, r: "aiosqlite.Row") -> FileChunkRecord:
        """Convert a file_chunks row to a FileChunkRecord."""
        return FileChunkRecord(
            id=r["chunk_id"],
            file_path=r["file_path"],
            chunk_index=r["chunk_index"],
            total_chunks=r["total_chunks"],
            line_start=r["line_start"],
            line_end=r["line_end"],
            symbols_defined=json.loads(r["symbols_defined"] or "[]"),
            symbols_referenced=json.loads(r["symbols_referenced"] or "[]"),
            dependencies=json.loads(r["dependencies"] or "[]"),
            summary=r["summary"] or "",
            raw_observations=json.loads(r["raw_observations"] or "[]"),
            content=r["content"] or "",
            function_name=r["function_name"] or "",
            language=r["language"] or "unknown",
            run_id=r["run_id"] or "",
        )

    def _row_to_obs_dict(self, r: "aiosqlite.Row") -> dict:
        """Convert a file_chunks row to the observation dict shape the auditor expects."""
        return {
            "file_path":     r["file_path"],
            "language":      r["language"] or "unknown",
            "content":       r["content"] or "",
            "line_start":    r["line_start"],
            "line_end":      r["line_end"],
            "function_name": r["function_name"] or "",
            "dependencies":  json.loads(r["dependencies"] or "[]"),
            "run_id":        r["run_id"] or "",
        }

    async def get_all_observations(self) -> list[dict]:
        """
        Return every stored chunk as an observation dict for the auditor.

        Implements the BrainStorage abstract method.  Each dict contains the
        keys the auditor's _audit_chunk() expects:
            file_path, language, content, line_start, line_end,
            function_name, dependencies
        """
        async with self._conn() as db:
            try:
                async with db.execute(
                    "SELECT * FROM file_chunks ORDER BY file_path, chunk_index"
                ) as cur:
                    rows = await cur.fetchall()
                return [self._row_to_obs_dict(r) for r in rows if r["content"]]
            except Exception as exc:
                log.warning("get_all_observations failed: %s", exc)
                return []

    async def get_stale_observations(self, run_id: str = "") -> list[dict]:
        """
        Return only the chunk observation dicts whose function_name appears in
        the function_staleness table.

        Gap 4 fix: the auditor calls this instead of get_all_observations()
        during incremental (commit-triggered) audit cycles so that only the
        CPG-computed impact set is re-audited rather than the whole codebase.

        When run_id is supplied the staleness lookup is scoped to that run.
        When the staleness table is empty this returns an empty list and the
        caller must fall back to get_all_observations().
        """
        async with self._conn() as db:
            try:
                if run_id:
                    stale_sql = (
                        "SELECT DISTINCT function_name FROM function_staleness "
                        "WHERE run_id = ?"
                    )
                    stale_params: tuple = (run_id,)
                else:
                    stale_sql    = "SELECT DISTINCT function_name FROM function_staleness"
                    stale_params = ()

                async with db.execute(stale_sql, stale_params) as cur:
                    stale_names = {row["function_name"] for row in await cur.fetchall()
                                   if row["function_name"]}

                if not stale_names:
                    return []

                # SQLite IN clause: bind one ? per stale function name.
                placeholders = ",".join("?" * len(stale_names))
                chunk_sql = (
                    f"SELECT * FROM file_chunks "
                    f"WHERE function_name IN ({placeholders}) "
                    f"ORDER BY file_path, chunk_index"
                )
                async with db.execute(chunk_sql, tuple(stale_names)) as cur:
                    rows = await cur.fetchall()

                return [self._row_to_obs_dict(r) for r in rows if r["content"]]
            except Exception as exc:
                log.warning("get_stale_observations failed: %s", exc)
                return []

    # ── Legacy aliases kept for callers that pre-date the ABC rename ─────────

    async def append_chunk(self, chunk: FileChunkRecord) -> None:
        """Deprecated alias for upsert_chunk.  Use upsert_chunk instead."""
        await self.upsert_chunk(chunk)

    async def get_chunks(self, file_path: str) -> list[FileChunkRecord]:
        """Deprecated alias for list_chunks.  Use list_chunks instead."""
        return await self.list_chunks(file_path)

                                                                               
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
                issue.fix_attempts,
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
            fix_attempts=row["fix_attempt_count"],
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
                     commit_sha, pr_url, created_at, committed_at,
                     blast_radius_exceeded, refactor_proposal_id)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                    committed_at=excluded.committed_at,
                    blast_radius_exceeded=excluded.blast_radius_exceeded,
                    refactor_proposal_id=excluded.refactor_proposal_id
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
                int(fix.blast_radius_exceeded),
                fix.refactor_proposal_id,
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

                                                                              
        pa_raw          = row["planner_approved"]
        planner_approved: bool | None = None if pa_raw is None else bool(pa_raw)

        gp_raw      = row["gate_passed"]
        gate_passed: bool | None = None if gp_raw is None else bool(gp_raw)

        # blast_radius_exceeded and refactor_proposal_id were added in the
        # Bug 2 schema migration.  Use `in keys` guards so the method still
        # works against pre-migration databases (e.g. test fixtures created
        # before the ALTER TABLE is applied to an existing SQLite file).
        blast_radius_exceeded = bool(row["blast_radius_exceeded"]) \
            if "blast_radius_exceeded" in keys else False
        refactor_proposal_id = (row["refactor_proposal_id"] or "") \
            if "refactor_proposal_id" in keys else ""

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
            blast_radius_exceeded=blast_radius_exceeded,
            refactor_proposal_id=refactor_proposal_id,
        )

                                                                                
    async def upsert_review(self, review) -> None:
        """Persist a ReviewResult.  Accepts ReviewResult model or compatible duck-type."""
        decisions_json = json.dumps(
            [d.model_dump() if hasattr(d, "model_dump") else d for d in review.decisions]
        )
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
                decisions_json,
                review.overall_score,
                review.overall_note,
                int(review.approve_for_commit),
                review.reviewed_at.isoformat(),
            ))
            await db.commit()

    async def get_review(self, fix_attempt_id: str):
        """Return a ReviewResult for the given fix_attempt_id, or None."""
        from brain.schemas import ReviewDecision as _RD, ReviewResult as _RR, ReviewVerdict as _RV
        async with self._conn() as db:
            async with db.execute(
                "SELECT * FROM review_results WHERE fix_attempt_id=?", (fix_attempt_id,)
            ) as cur:
                row = await cur.fetchone()
                if not row:
                    return None
                decisions_data = json.loads(row["decisions"] or "[]")
                decisions = []
                for d in decisions_data:
                    try:
                        decisions.append(_RD(**d))
                    except Exception:
                        pass
                return _RR(
                    review_id=row["review_id"],
                    fix_attempt_id=row["fix_attempt_id"],
                    decisions=decisions,
                    overall_score=row["overall_score"],
                    overall_note=row["overall_note"] or "",
                    approve_for_commit=bool(row["approve_for_commit"]),
                    reviewed_at=_require_dt(row["reviewed_at"]),
                )

                                                                                
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

                                                                                
    async def upsert_test_result(self, result: TestRunResult) -> None:
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

                                                                                
    async def append_patrol_event(self, event: PatrolEvent) -> None:
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

    async def log_llm_session(self, session) -> None:
        """Accept either a dict (abstract contract) or an LLMSession model."""
        from brain.schemas import LLMSession as _LLMSession
        if isinstance(session, dict):
            s_id          = session.get("id", str(uuid.uuid4()))
            s_run_id      = session.get("run_id", "")
            s_agent_type  = session.get("agent_type", "GENERAL")
            if hasattr(s_agent_type, "value"):
                s_agent_type = s_agent_type.value
            s_model       = session.get("model", "")
            s_ptok        = session.get("prompt_tokens", 0)
            s_ctok        = session.get("completion_tokens", 0)
            s_cost        = session.get("cost_usd", 0.0)
            s_dur         = session.get("duration_ms", 0)
            s_ok          = int(session.get("success", True))
            s_err         = session.get("error", "")
            s_start       = session.get("started_at", datetime.now(tz=timezone.utc))
            if isinstance(s_start, datetime):
                s_start = s_start.isoformat()
        else:
            s_id          = session.id
            s_run_id      = session.run_id
            s_agent_type  = session.agent_type.value if hasattr(session.agent_type, "value") else str(session.agent_type)
            s_model       = session.model
            s_ptok        = session.prompt_tokens
            s_ctok        = session.completion_tokens
            s_cost        = session.cost_usd
            s_dur         = session.duration_ms
            s_ok          = int(session.success)
            s_err         = session.error
            s_start       = session.started_at.isoformat()

        import uuid as _uuid
        async with self._write() as db:
            await db.execute("""
                INSERT INTO llm_sessions
                    (id, run_id, agent_type, model, prompt_tokens,
                     completion_tokens, cost_usd, duration_ms, success, error, started_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO NOTHING
            """, (s_id, s_run_id, s_agent_type, s_model, s_ptok,
                  s_ctok, s_cost, s_dur, s_ok, s_err, s_start))
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

                                                                                
    async def count_open_critical(self, run_id: str | None = None) -> int:
        issues = await self.list_issues(run_id=run_id, status="OPEN", severity=Severity.CRITICAL)
        return len(issues)

    async def all_read(self) -> bool:
        files = await self.list_files()
        return all(f.fully_read for f in files) if files else False

                                                                                 
    async def list_audit_trail(self, run_id: str, limit: int = 1000) -> list[AuditTrailEntry]:
        return (await self.get_audit_trail(run_id))[:limit]

    async def upsert_escalation(self, esc) -> None:
        async with self._write() as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS escalations (
                    id TEXT PRIMARY KEY,
                    run_id TEXT, issue_ids TEXT, fix_attempt_id TEXT,
                    escalation_type TEXT, description TEXT,
                    severity TEXT, mil882e_category TEXT, status TEXT,
                    approved_by TEXT, approved_at TEXT, approval_rationale TEXT,
                    risk_acceptance TEXT, notified_via TEXT, notified_at TEXT,
                    timeout_at TEXT, created_at TEXT, updated_at TEXT
                )
            """)
            await db.execute("""
                INSERT OR REPLACE INTO escalations
                    (id, run_id, issue_ids, fix_attempt_id, escalation_type,
                     description, severity, mil882e_category, status,
                     approved_by, approved_at, approval_rationale, risk_acceptance,
                     notified_via, notified_at, timeout_at, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                esc.id, esc.run_id,
                __import__("json").dumps(esc.issue_ids),
                esc.fix_attempt_id, esc.escalation_type,
                esc.description, esc.severity.value, esc.mil882e_category.value,
                esc.status.value, esc.approved_by,
                esc.approved_at.isoformat() if esc.approved_at else None,
                esc.approval_rationale, esc.risk_acceptance,
                __import__("json").dumps(esc.notified_via),
                esc.notified_at.isoformat() if esc.notified_at else None,
                esc.timeout_at.isoformat() if esc.timeout_at else None,
                esc.created_at.isoformat(), esc.updated_at.isoformat(),
            ))
            await db.commit()

    async def get_escalation(self, escalation_id: str):
        await self._ensure_escalations_table()
        async with self._conn() as db:
            async with db.execute(
                "SELECT * FROM escalations WHERE id=?", (escalation_id,)
            ) as cur:
                row = await cur.fetchone()
                return self._row_to_escalation(row) if row else None

    async def list_escalations(self, run_id: str = "", status=None) -> list:
        await self._ensure_escalations_table()
        async with self._conn() as db:
            q = "SELECT * FROM escalations WHERE 1=1"
            params: list = []
            if run_id:
                q += " AND run_id=?"; params.append(run_id)
            if status is not None:
                q += " AND status=?"; params.append(status.value if hasattr(status, "value") else status)
            async with db.execute(q, params) as cur:
                rows = await cur.fetchall()
                return [self._row_to_escalation(r) for r in rows if r]

    async def _ensure_escalations_table(self) -> None:
        async with self._write() as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS escalations (
                    id TEXT PRIMARY KEY, run_id TEXT, issue_ids TEXT,
                    fix_attempt_id TEXT, escalation_type TEXT, description TEXT,
                    severity TEXT, mil882e_category TEXT, status TEXT,
                    approved_by TEXT, approved_at TEXT, approval_rationale TEXT,
                    risk_acceptance TEXT, notified_via TEXT, notified_at TEXT,
                    timeout_at TEXT, created_at TEXT, updated_at TEXT
                )
            """)
            await db.commit()

    def _row_to_escalation(self, row):
        from brain.schemas import EscalationRecord, EscalationStatus, Severity, MilStd882eCategory
        import json as _json, datetime as _dt
        return EscalationRecord(
            id=row["id"], run_id=row["run_id"] or "",
            issue_ids=_json.loads(row["issue_ids"] or "[]"),
            fix_attempt_id=row["fix_attempt_id"] or "",
            escalation_type=row["escalation_type"] or "",
            description=row["description"] or "",
            severity=Severity(row["severity"]) if row["severity"] else Severity.CRITICAL,
            mil882e_category=MilStd882eCategory(row["mil882e_category"]) if row["mil882e_category"] else MilStd882eCategory.CAT_I,
            status=EscalationStatus(row["status"]) if row["status"] else EscalationStatus.PENDING,
            approved_by=row["approved_by"] or "",
            approval_rationale=row["approval_rationale"] or "",
            risk_acceptance=row["risk_acceptance"] or "",
            notified_via=_json.loads(row["notified_via"] or "[]"),
        )

                                                                                 
    async def upsert_refactor_proposal(self, proposal) -> None:
        import json as _json
        async with self._write() as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS refactor_proposals (
                    id TEXT PRIMARY KEY,
                    fix_attempt_id TEXT,
                    run_id TEXT,
                    issue_ids TEXT,
                    changed_functions TEXT,
                    affected_function_count INTEGER,
                    affected_file_count INTEGER,
                    test_files_affected TEXT,
                    blast_radius_score REAL,
                    importing_modules TEXT,
                    importing_module_count INTEGER,
                    affected_components TEXT,
                    proposed_refactoring TEXT,
                    migration_steps TEXT,
                    estimated_scope TEXT,
                    risks TEXT,
                    recommendation TEXT,
                    escalation_id TEXT,
                    requires_human_review INTEGER,
                    created_at TEXT
                )
            """)
            # ADD COLUMN is idempotent on SQLite when the column doesn't exist
            # yet — needed for databases created before this migration.
            for col, typedef in (
                ("importing_modules",      "TEXT"),
                ("importing_module_count", "INTEGER"),
            ):
                try:
                    await db.execute(
                        f"ALTER TABLE refactor_proposals ADD COLUMN {col} {typedef}"
                    )
                except Exception:
                    pass  # column already exists
            await db.execute("""
                INSERT OR REPLACE INTO refactor_proposals
                    (id, fix_attempt_id, run_id, issue_ids, changed_functions,
                     affected_function_count, affected_file_count, test_files_affected,
                     blast_radius_score, importing_modules, importing_module_count,
                     affected_components, proposed_refactoring,
                     migration_steps, estimated_scope, risks, recommendation,
                     escalation_id, requires_human_review, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                proposal.id, proposal.fix_attempt_id, proposal.run_id,
                _json.dumps(proposal.issue_ids),
                _json.dumps(proposal.changed_functions),
                proposal.affected_function_count,
                proposal.affected_file_count,
                _json.dumps(proposal.test_files_affected),
                proposal.blast_radius_score,
                _json.dumps(getattr(proposal, 'importing_modules', [])),
                getattr(proposal, 'importing_module_count', 0),
                _json.dumps(proposal.affected_components),
                proposal.proposed_refactoring,
                _json.dumps(proposal.migration_steps),
                proposal.estimated_scope,
                _json.dumps(proposal.risks),
                proposal.recommendation,
                proposal.escalation_id,
                1 if proposal.requires_human_review else 0,
                proposal.created_at.isoformat(),
            ))
            await db.commit()

    async def get_refactor_proposal(self, proposal_id: str):
        async with self._conn() as db:
            try:
                async with db.execute(
                    "SELECT * FROM refactor_proposals WHERE id=?", (proposal_id,)
                ) as cur:
                    row = await cur.fetchone()
                    return self._row_to_refactor_proposal(row) if row else None
            except Exception:
                return None

    async def list_refactor_proposals(self, run_id: str = "") -> list:
        async with self._conn() as db:
            try:
                if run_id:
                    sql = "SELECT * FROM refactor_proposals WHERE run_id=? ORDER BY created_at DESC"
                    params: list = [run_id]
                else:
                    sql = "SELECT * FROM refactor_proposals ORDER BY created_at DESC"
                    params = []
                async with db.execute(sql, params) as cur:
                    rows = await cur.fetchall()
                    return [self._row_to_refactor_proposal(r) for r in rows if r]
            except Exception:
                return []

    def _row_to_refactor_proposal(self, row):
        from brain.schemas import RefactorProposal
        import json as _json
        return RefactorProposal(
            id=row["id"],
            fix_attempt_id=row["fix_attempt_id"] or "",
            run_id=row["run_id"] or "",
            issue_ids=_json.loads(row["issue_ids"] or "[]"),
            changed_functions=_json.loads(row["changed_functions"] or "[]"),
            affected_function_count=row["affected_function_count"] or 0,
            affected_file_count=row["affected_file_count"] or 0,
            test_files_affected=_json.loads(row["test_files_affected"] or "[]"),
            blast_radius_score=row["blast_radius_score"] or 0.0,
            importing_modules=_json.loads(row["importing_modules"] or "[]")
                if "importing_modules" in row.keys() else [],
            importing_module_count=row["importing_module_count"] or 0
                if "importing_module_count" in row.keys() else 0,
            affected_components=_json.loads(row["affected_components"] or "[]"),
            proposed_refactoring=row["proposed_refactoring"] or "",
            migration_steps=_json.loads(row["migration_steps"] or "[]"),
            estimated_scope=row["estimated_scope"] or "",
            risks=_json.loads(row["risks"] or "[]"),
            recommendation=row["recommendation"] or "",
            escalation_id=row["escalation_id"] or "",
            requires_human_review=bool(row["requires_human_review"]),
            # Bug 5 fix: restore persisted timestamp instead of defaulting to
            # now().  Without this, every API response showed the query time
            # rather than when the proposal was actually created.
            created_at=_require_dt(row["created_at"]),
        )
    async def upsert_baseline(self, baseline) -> None:
        await self._ensure_baselines_table()
        async with self._write() as db:
            await db.execute("""
                INSERT OR REPLACE INTO baselines
                    (id, run_id, baseline_name, software_level, commit_hash,
                     issue_count, score_snapshot, file_hashes, approved_by,
                     approved_at, is_active, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                baseline.id, baseline.run_id, baseline.baseline_name,
                baseline.software_level.value, baseline.commit_hash,
                __import__("json").dumps(baseline.issue_count),
                baseline.score_snapshot,
                __import__("json").dumps(baseline.file_hashes),
                baseline.approved_by,
                baseline.approved_at.isoformat() if baseline.approved_at else None,
                1 if baseline.is_active else 0,
                baseline.created_at.isoformat(),
            ))
            await db.commit()

    async def _ensure_baselines_table(self) -> None:
        async with self._write() as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS baselines (
                    id TEXT PRIMARY KEY, run_id TEXT, baseline_name TEXT,
                    software_level TEXT, commit_hash TEXT, issue_count TEXT,
                    score_snapshot REAL, file_hashes TEXT, approved_by TEXT,
                    approved_at TEXT, is_active INTEGER DEFAULT 0, created_at TEXT
                )
            """)
            await db.commit()

    async def get_baseline(self, baseline_id: str):
        await self._ensure_baselines_table()
        async with self._conn() as db:
            async with db.execute("SELECT * FROM baselines WHERE id=?", (baseline_id,)) as cur:
                row = await cur.fetchone()
                return self._row_to_baseline(row) if row else None

    async def get_active_baseline(self, run_id: str = ""):
        await self._ensure_baselines_table()
        async with self._conn() as db:
            if run_id:
                async with db.execute(
                    "SELECT * FROM baselines WHERE run_id=? AND is_active=1 LIMIT 1", (run_id,)
                ) as cur:
                    row = await cur.fetchone()
            else:
                async with db.execute(
                    "SELECT * FROM baselines WHERE is_active=1 ORDER BY created_at DESC LIMIT 1"
                ) as cur:
                    row = await cur.fetchone()
            return self._row_to_baseline(row) if row else None

    async def list_baselines(self, run_id: str = "") -> list:
        await self._ensure_baselines_table()
        async with self._conn() as db:
            q = "SELECT * FROM baselines" + (" WHERE run_id=?" if run_id else "")
            async with db.execute(q, (run_id,) if run_id else ()) as cur:
                rows = await cur.fetchall()
                return [self._row_to_baseline(r) for r in rows]

    def _row_to_baseline(self, row):
        from brain.schemas import BaselineRecord, SoftwareLevel
        import json as _json
        return BaselineRecord(
            id=row["id"], run_id=row["run_id"] or "",
            baseline_name=row["baseline_name"] or "",
            software_level=SoftwareLevel(row["software_level"]) if row["software_level"] else SoftwareLevel.NONE,
            commit_hash=row["commit_hash"] or "",
            issue_count=_json.loads(row["issue_count"] or "{}"),
            score_snapshot=float(row["score_snapshot"] or 0),
            file_hashes=_json.loads(row["file_hashes"] or "{}"),
            approved_by=row["approved_by"] or "",
            is_active=bool(row["is_active"]),
        )

    # ── Function staleness (Gap 4) ────────────────────────────────────────────

    async def upsert_staleness_mark(self, mark) -> None:
        """
        Upsert a FunctionStalenessMark.  The table is created in the main DDL
        block so we never re-issue CREATE TABLE here.  Uses UNIQUE(file_path,
        function_name) so repeated marks for the same function are idempotent.
        """
        from brain.schemas import FunctionStalenessMark  # local to avoid circulars
        async with self._write() as db:
            await db.execute(
                """
                INSERT INTO function_staleness
                    (id, file_path, function_name, line_start, line_end,
                     stale_reason, stale_since, run_id)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(file_path, function_name) DO UPDATE SET
                    stale_reason = excluded.stale_reason,
                    stale_since  = excluded.stale_since,
                    run_id       = excluded.run_id,
                    line_start   = excluded.line_start,
                    line_end     = excluded.line_end
                """,
                (
                    mark.id, mark.file_path, mark.function_name,
                    mark.line_start, mark.line_end, mark.stale_reason,
                    mark.stale_since.isoformat(), mark.run_id,
                ),
            )
            await db.commit()

    async def list_stale_functions(self, file_path: str = "", run_id: str = "") -> list:
        from brain.schemas import FunctionStalenessMark
        async with self._conn() as db:
            try:
                conditions: list[str] = []
                params:     list      = []
                if file_path:
                    conditions.append("file_path=?"); params.append(file_path)
                if run_id:
                    conditions.append("run_id=?");    params.append(run_id)
                where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
                sql   = f"SELECT * FROM function_staleness {where} ORDER BY stale_since"
                async with db.execute(sql, params) as cur:
                    rows = await cur.fetchall()
                return [
                    FunctionStalenessMark(
                        id=r["id"],
                        file_path=r["file_path"],
                        function_name=r["function_name"],
                        line_start=r["line_start"] or 0,
                        line_end=r["line_end"] or 0,
                        stale_reason=r["stale_reason"] or "",
                        run_id=r["run_id"] or "",
                    )
                    for r in rows
                ]
            except Exception:
                return []

    async def clear_staleness_mark(self, file_path: str, function_name: str) -> None:
        async with self._write() as db:
            try:
                await db.execute(
                    "DELETE FROM function_staleness WHERE file_path=? AND function_name=?",
                    (file_path, function_name),
                )
                await db.commit()
            except Exception:
                pass

    # ── Commit-audit records (Gap 4) ─────────────────────────────────────────

    async def upsert_commit_audit_record(self, record: "CommitAuditRecord") -> None:  # type: ignore[override]
        async with self._write() as db:
            await db.execute(
                """
                INSERT INTO commit_audit_records (
                    id, run_id, commit_hash, branch, author, commit_message,
                    changed_files, changed_functions, all_changed_functions,
                    new_functions, deleted_functions, impact_functions,
                    impact_files, audit_targets,
                    total_changed_functions, total_impact_functions,
                    total_functions_to_audit, test_files_to_run,
                    test_functions_to_run, status, cpg_updated,
                    joern_update_status, error_detail,
                    created_at, started_at, finished_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    status                   = excluded.status,
                    cpg_updated              = excluded.cpg_updated,
                    joern_update_status      = excluded.joern_update_status,
                    error_detail             = excluded.error_detail,
                    impact_functions         = excluded.impact_functions,
                    impact_files             = excluded.impact_files,
                    audit_targets            = excluded.audit_targets,
                    total_changed_functions  = excluded.total_changed_functions,
                    total_impact_functions   = excluded.total_impact_functions,
                    total_functions_to_audit = excluded.total_functions_to_audit,
                    test_files_to_run        = excluded.test_files_to_run,
                    test_functions_to_run    = excluded.test_functions_to_run,
                    started_at               = excluded.started_at,
                    finished_at              = excluded.finished_at
                """,
                (
                    record.id, record.run_id, record.commit_hash,
                    record.branch, record.author, record.commit_message,
                    json.dumps(record.changed_files),
                    json.dumps(record.changed_functions),
                    json.dumps(record.all_changed_functions),
                    json.dumps(record.new_functions),
                    json.dumps(record.deleted_functions),
                    json.dumps(record.impact_functions),
                    json.dumps(record.impact_files),
                    json.dumps(record.audit_targets),
                    record.total_changed_functions,
                    record.total_impact_functions,
                    record.total_functions_to_audit,
                    json.dumps(record.test_files_to_run),
                    json.dumps(record.test_functions_to_run),
                    record.status.value,
                    int(record.cpg_updated),
                    record.joern_update_status,
                    record.error_detail,
                    record.created_at.isoformat(),
                    record.started_at.isoformat() if record.started_at else None,
                    record.finished_at.isoformat() if record.finished_at else None,
                ),
            )
            await db.commit()

    def _row_to_commit_audit_record(self, row: aiosqlite.Row) -> "CommitAuditRecord":
        from brain.schemas import CommitAuditRecord, CommitAuditStatus
        return CommitAuditRecord(
            id=row["id"],
            run_id=row["run_id"] or "",
            commit_hash=row["commit_hash"] or "",
            branch=row["branch"] or "",
            author=row["author"] or "",
            commit_message=row["commit_message"] or "",
            changed_files=json.loads(row["changed_files"] or "[]"),
            changed_functions=json.loads(row["changed_functions"] or "{}"),
            all_changed_functions=json.loads(row["all_changed_functions"] or "[]"),
            new_functions=json.loads(row["new_functions"] or "[]"),
            deleted_functions=json.loads(row["deleted_functions"] or "[]"),
            impact_functions=json.loads(row["impact_functions"] or "[]"),
            impact_files=json.loads(row["impact_files"] or "[]"),
            audit_targets=json.loads(row["audit_targets"] or "[]"),
            total_changed_functions=row["total_changed_functions"] or 0,
            total_impact_functions=row["total_impact_functions"] or 0,
            total_functions_to_audit=row["total_functions_to_audit"] or 0,
            test_files_to_run=json.loads(row["test_files_to_run"] or "[]"),
            test_functions_to_run=json.loads(row["test_functions_to_run"] or "[]"),
            status=CommitAuditStatus(row["status"]),
            cpg_updated=bool(row["cpg_updated"]),
            joern_update_status=row["joern_update_status"] or "",
            error_detail=row["error_detail"] or "",
            created_at=_require_dt(row["created_at"]),
            started_at=_parse_dt(row["started_at"]),
            finished_at=_parse_dt(row["finished_at"]),
        )

    async def get_commit_audit_record(self, record_id: str) -> "CommitAuditRecord | None":
        async with self._conn() as db:
            async with db.execute(
                "SELECT * FROM commit_audit_records WHERE id=?", (record_id,)
            ) as cur:
                row = await cur.fetchone()
                return self._row_to_commit_audit_record(row) if row else None

    async def get_commit_audit_record_by_hash(
        self, commit_hash: str, run_id: str = ""
    ) -> "CommitAuditRecord | None":
        async with self._conn() as db:
            try:
                if run_id:
                    sql    = ("SELECT * FROM commit_audit_records "
                              "WHERE commit_hash=? AND run_id=? "
                              "ORDER BY created_at DESC LIMIT 1")
                    params = (commit_hash, run_id)
                else:
                    sql    = ("SELECT * FROM commit_audit_records "
                              "WHERE commit_hash=? ORDER BY created_at DESC LIMIT 1")
                    params = (commit_hash,)
                async with db.execute(sql, params) as cur:
                    row = await cur.fetchone()
                    return self._row_to_commit_audit_record(row) if row else None
            except Exception:
                return None

    async def list_commit_audit_records(
        self,
        run_id: str = "",
        status: "CommitAuditStatus | None" = None,
        limit: int = 100,
    ) -> list:
        from brain.schemas import CommitAuditStatus as _CAS
        async with self._conn() as db:
            try:
                conditions: list[str] = []
                params:     list      = []
                if run_id:
                    conditions.append("run_id=?");   params.append(run_id)
                if status is not None:
                    conditions.append("status=?");   params.append(status.value)
                where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
                params.append(limit)
                sql = (
                    f"SELECT * FROM commit_audit_records {where} "
                    f"ORDER BY created_at DESC LIMIT ?"
                )
                async with db.execute(sql, params) as cur:
                    rows = await cur.fetchall()
                return [self._row_to_commit_audit_record(r) for r in rows]
            except Exception:
                return []



                                                                      
    async def upsert_ldra_finding(self, finding) -> None:
        await self._upsert_json_table("ldra_findings", finding.id, finding.model_dump_json())

    async def list_ldra_findings(self, run_id: str = "", file_path: str = "") -> list:
        from brain.schemas import LdraFinding
        return await self._list_json_table("ldra_findings", LdraFinding)

    async def upsert_polyspace_finding(self, finding) -> None:
        await self._upsert_json_table("polyspace_findings", finding.id, finding.model_dump_json())

    async def list_polyspace_findings(self, run_id: str = "") -> list:
        from brain.schemas import PolyspaceFinding
        return await self._list_json_table("polyspace_findings", PolyspaceFinding)

    async def upsert_cbmc_result(self, result) -> None:
        await self._upsert_json_table("cbmc_results", result.id, result.model_dump_json())

    async def get_cbmc_result(self, result_id: str):
        from brain.schemas import CbmcVerificationResult
        return await self._get_json_table("cbmc_results", result_id, CbmcVerificationResult)

    async def upsert_rtm_entry(self, entry) -> None:
        await self._upsert_json_table("rtm_entries", entry.id, entry.model_dump_json())

    async def get_rtm_for_issue(self, issue_id: str):
        from brain.schemas import RequirementTraceability
        async with self._conn() as db:
            try:
                async with db.execute(
                    "SELECT data FROM rtm_entries WHERE data LIKE ?",
                    (f'%{issue_id}%',)
                ) as cur:
                    row = await cur.fetchone()
                    if row:
                        return RequirementTraceability.model_validate_json(row["data"])
            except Exception:
                pass
        return None

    async def list_rtm_entries(self, run_id: str = "") -> list:
        from brain.schemas import RequirementTraceability
        return await self._list_json_table("rtm_entries", RequirementTraceability)

    async def upsert_independence_record(self, record) -> None:
        await self._upsert_json_table("independence_records", record.id, record.model_dump_json())

    async def get_independence_record(self, fix_attempt_id: str):
        from brain.schemas import ReviewerIndependenceRecord
        async with self._conn() as db:
            try:
                async with db.execute(
                    "SELECT data FROM independence_records WHERE data LIKE ?",
                    (f'%{fix_attempt_id}%',)
                ) as cur:
                    row = await cur.fetchone()
                    if row:
                        return ReviewerIndependenceRecord.model_validate_json(row["data"])
            except Exception:
                pass
        return None

    async def upsert_sas(self, sas) -> None:
        await self._upsert_json_table("sas_records", sas.id, sas.model_dump_json())

    async def get_sas(self, run_id: str):
        from brain.schemas import SoftwareAccomplishmentSummary
        async with self._conn() as db:
            try:
                async with db.execute(
                    "SELECT data FROM sas_records WHERE data LIKE ?",
                    (f'%{run_id}%',)
                ) as cur:
                    row = await cur.fetchone()
                    if row:
                        return SoftwareAccomplishmentSummary.model_validate_json(row["data"])
            except Exception:
                pass
        return None

    async def upsert_sci(self, sci) -> None:
        await self._upsert_json_table("sci_records", sci.id, sci.model_dump_json())

    async def get_sci(self, baseline_id: str):
        from brain.schemas import SoftwareConfigurationIndex
        async with self._conn() as db:
            try:
                async with db.execute(
                    "SELECT data FROM sci_records WHERE data LIKE ?",
                    (f'%{baseline_id}%',)
                ) as cur:
                    row = await cur.fetchone()
                    if row:
                        return SoftwareConfigurationIndex.model_validate_json(row["data"])
            except Exception:
                pass
        return None

                                     
    async def _upsert_json_table(self, table: str, id_: str, data: str) -> None:
        async with self._write() as db:
            await db.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, data TEXT)
            """)
            await db.execute(
                f"INSERT OR REPLACE INTO {table} (id, data) VALUES (?,?)",
                (id_, data),
            )
            await db.commit()

    async def _get_json_table(self, table: str, id_: str, model_class):
        async with self._conn() as db:
            try:
                async with db.execute(
                    f"SELECT data FROM {table} WHERE id=?", (id_,)
                ) as cur:
                    row = await cur.fetchone()
                    if row:
                        return model_class.model_validate_json(row["data"])
            except Exception:
                pass
        return None

    async def _list_json_table(self, table: str, model_class) -> list:
        async with self._conn() as db:
            try:
                async with db.execute(f"SELECT data FROM {table}") as cur:
                    rows = await cur.fetchall()
                    result = []
                    for row in rows:
                        try:
                            result.append(model_class.model_validate_json(row["data"]))
                        except Exception:
                            pass
                    return result
            except Exception:
                return []

                                                                                 
    async def upsert_convergence_record(self, record: "ConvergenceRecord") -> None:                          
        from brain.schemas import ConvergenceRecord as _CR
        async with self._write() as db:
            await db.execute(
                "CREATE TABLE IF NOT EXISTS convergence_records "
                "(id TEXT PRIMARY KEY, run_id TEXT, data TEXT)"
            )
            await db.execute(
                "INSERT OR REPLACE INTO convergence_records (id, run_id, data) VALUES (?,?,?)",
                (record.id, record.run_id, record.model_dump_json()),
            )
            await db.commit()

    async def list_convergence_records(self, run_id: str) -> list["ConvergenceRecord"]:                          
        from brain.schemas import ConvergenceRecord as _CR
        async with self._conn() as db:
            try:
                await db.execute(
                    "CREATE TABLE IF NOT EXISTS convergence_records "
                    "(id TEXT PRIMARY KEY, run_id TEXT, data TEXT)"
                )
                async with db.execute(
                    "SELECT data FROM convergence_records WHERE run_id=? ORDER BY rowid ASC",
                    (run_id,),
                ) as cur:
                    rows = await cur.fetchall()
                result = []
                for row in rows:
                    try:
                        result.append(_CR.model_validate_json(row["data"]))
                    except Exception:
                        pass
                return result
            except Exception:
                return []

                                                                                
    async def _ensure_synthesis_reports_table(self, db) -> None:
        """Create synthesis_reports table if it does not exist."""
        await db.execute("""
            CREATE TABLE IF NOT EXISTS synthesis_reports (
                id      TEXT PRIMARY KEY,
                run_id  TEXT NOT NULL,
                cycle   INTEGER NOT NULL DEFAULT 0,
                data    TEXT NOT NULL
            )
        """)
        try:
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_synthesis_reports_run "
                "ON synthesis_reports(run_id)"
            )
        except Exception:
            pass

    async def upsert_synthesis_report(self, report: "SynthesisReport") -> None:                          
        """
        Persist a SynthesisReport produced by SynthesisAgent after each
        synthesis pass.  One report per run per cycle — the id field is the
        natural primary key.

        Gap 2 fix: previously SynthesisReport was defined in schemas.py and
        imported in controller.py but never constructed or stored.  Per-run
        synthesis quality metrics (dedup counts, compound finding counts) were
        only emitted to log.info and lost.  This method closes that gap.
        """
        from brain.schemas import SynthesisReport
        async with self._write() as db:
            await self._ensure_synthesis_reports_table(db)
            await db.execute(
                "INSERT OR REPLACE INTO synthesis_reports (id, run_id, cycle, data) "
                "VALUES (?,?,?,?)",
                (report.id, report.run_id, report.cycle, report.model_dump_json()),
            )
            await db.commit()

    async def get_synthesis_report(self, run_id: str, cycle: int | None = None) -> "SynthesisReport | None":                          
        """
        Retrieve the SynthesisReport for a run.

        If cycle is specified, returns the report for that cycle.
        If cycle is None, returns the most recent report for the run.
        """
        from brain.schemas import SynthesisReport
        async with self._conn() as db:
            try:
                await self._ensure_synthesis_reports_table(db)
                if cycle is not None:
                    sql = (
                        "SELECT data FROM synthesis_reports "
                        "WHERE run_id=? AND cycle=? LIMIT 1"
                    )
                    params = (run_id, cycle)
                else:
                    sql = (
                        "SELECT data FROM synthesis_reports "
                        "WHERE run_id=? ORDER BY cycle DESC LIMIT 1"
                    )
                    params = (run_id,)
                async with db.execute(sql, params) as cur:
                    row = await cur.fetchone()
                    if row:
                        return SynthesisReport.model_validate_json(row["data"])
            except Exception:
                pass
        return None

    async def list_synthesis_reports(self, run_id: str | None = None) -> "list[SynthesisReport]":                          
        """
        List all SynthesisReports, optionally filtered by run_id.
        Ordered by run_id ASC, cycle ASC.
        """
        from brain.schemas import SynthesisReport
        async with self._conn() as db:
            try:
                await self._ensure_synthesis_reports_table(db)
                if run_id:
                    sql = (
                        "SELECT data FROM synthesis_reports "
                        "WHERE run_id=? ORDER BY cycle ASC"
                    )
                    params = (run_id,)
                else:
                    sql = "SELECT data FROM synthesis_reports ORDER BY run_id, cycle ASC"
                    params = ()
                async with db.execute(sql, params) as cur:
                    rows = await cur.fetchall()
                result = []
                for row in rows:
                    try:
                        result.append(SynthesisReport.model_validate_json(row["data"]))
                    except Exception:
                        pass
                return result
            except Exception:
                return []

                                                                                
    async def list_compound_findings(
        self,
        run_id: str | None = None,
        severity: str | None = None,
    ) -> list:
        """
        Return all Issues with executor_type=SYNTHESIS — these ARE the
        compound findings.  SynthesisAgent materialises each CompoundFinding
        as a regular Issue (executor_type=SYNTHESIS) so they flow through the
        normal consensus → fix → review pipeline.

        This query surfaces them as first-class objects for the API and
        external consumers (dashboards, webhooks, DeerFlow).

        Parameters
        ----------
        run_id:
            If provided, filter to a specific run.
        severity:
            If provided (e.g. "CRITICAL"), filter by severity value.
        """
        from brain.schemas import ExecutorType
        async with self._conn() as db:
            try:
                conditions = ["executor_type = ?"]
                params: list = [ExecutorType.SYNTHESIS.value]
                if run_id:
                    conditions.append("run_id = ?")
                    params.append(run_id)
                if severity:
                    conditions.append("severity = ?")
                    params.append(severity)
                where = " AND ".join(conditions)
                sql = f"SELECT * FROM issues WHERE {where} ORDER BY created_at DESC"
                async with db.execute(sql, params) as cur:
                    rows = await cur.fetchall()
                return [self._row_to_issue(row) for row in rows]
            except Exception:
                return []
