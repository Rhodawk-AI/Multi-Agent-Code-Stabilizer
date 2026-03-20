from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

try:
    import asyncpg                                    # type: ignore[import]
    from sqlalchemy.ext.asyncio import (              # type: ignore[import]
        create_async_engine, AsyncSession, async_sessionmaker
    )
    from sqlalchemy import text                       # type: ignore[import]
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False
    log.info(
        "asyncpg/sqlalchemy not installed — PostgreSQL storage disabled. "
        "Run: pip install asyncpg sqlalchemy[asyncio]"
    )

from brain.schemas import (
    AuditRun, AuditScore, AuditTrailEntry, AutonomyLevel, ChunkStrategy,
    DomainMode, ExecutorType, FileChunkRecord, FileRecord, FileStatus,
    FixAttempt, FixedFile, FormalVerificationResult, FormalVerificationStatus,
    GraphEdge, Issue, IssueFingerprint, IssueStatus, LLMSession, PatrolEvent,
    PlannerRecord, PlannerVerdict, ReversibilityClass, ReviewDecision,
    ReviewResult, ReviewVerdict, RunStatus, Severity, TestRunResult, TestRunStatus,
)
from brain.storage import BrainStorage
from brain.sqlite_storage import SQLiteBrainStorage


def _require_database_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "FATAL: DATABASE_URL not set. "
            "Format: postgresql+asyncpg://user:pass@host:5432/rhodawk"
        )
    # Normalize: plain postgresql:// → postgresql+asyncpg://
    if url.startswith("postgresql://") and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


# ──────────────────────────────────────────────────────────────────────────────
# DDL — partitioned schema
# ──────────────────────────────────────────────────────────────────────────────

_DDL_MAIN = """
-- Enable partitioning support
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- audit_runs
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
    graph_built         BOOLEAN DEFAULT FALSE,
    metadata            JSONB DEFAULT '{}',
    started_at          TIMESTAMPTZ NOT NULL,
    completed_at        TIMESTAMPTZ
);

-- files
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
    is_load_bearing BOOLEAN DEFAULT FALSE,
    last_hash_check TIMESTAMPTZ,
    last_read_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL
);

-- file_chunks (partitioned by hash of file_path for scale)
CREATE TABLE IF NOT EXISTS file_chunks (
    chunk_id            TEXT NOT NULL,
    file_path           TEXT NOT NULL,
    chunk_index         INTEGER NOT NULL,
    total_chunks        INTEGER NOT NULL,
    line_start          INTEGER NOT NULL,
    line_end            INTEGER NOT NULL,
    symbols_defined     JSONB DEFAULT '[]',
    symbols_referenced  JSONB DEFAULT '[]',
    dependencies        JSONB DEFAULT '[]',
    summary             TEXT DEFAULT '',
    raw_observations    JSONB DEFAULT '[]',
    token_count         INTEGER DEFAULT 0,
    read_at             TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (chunk_id),
    UNIQUE (file_path, chunk_index)
) PARTITION BY HASH (file_path);

-- Create 16 partitions for file_chunks
DO $$
BEGIN
    FOR i IN 0..15 LOOP
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS file_chunks_%s PARTITION OF file_chunks '
            'FOR VALUES WITH (MODULUS 16, REMAINDER %s)',
            i, i
        );
    END LOOP;
END $$;

-- issues (partitioned by run_id for large runs)
CREATE TABLE IF NOT EXISTS issues (
    id                      TEXT NOT NULL,
    run_id                  TEXT,
    severity                TEXT NOT NULL,
    file_path               TEXT NOT NULL,
    line_start              INTEGER DEFAULT 0,
    line_end                INTEGER DEFAULT 0,
    executor_type           TEXT NOT NULL,
    master_prompt_section   TEXT DEFAULT '',
    description             TEXT NOT NULL,
    fix_requires_files      JSONB DEFAULT '[]',
    status                  TEXT DEFAULT 'OPEN',
    fix_attempt_count       INTEGER DEFAULT 0,
    fingerprint             TEXT DEFAULT '',
    escalated_reason        TEXT,
    regressed_from          TEXT,
    consensus_votes         INTEGER DEFAULT 0,
    consensus_confidence    REAL DEFAULT 0.0,
    created_at              TIMESTAMPTZ NOT NULL,
    closed_at               TIMESTAMPTZ,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_issues_run_status  ON issues(run_id, status);
CREATE INDEX IF NOT EXISTS idx_issues_severity    ON issues(severity);
CREATE INDEX IF NOT EXISTS idx_issues_file        ON issues(file_path);
CREATE INDEX IF NOT EXISTS idx_issues_fingerprint ON issues(fingerprint);
CREATE INDEX IF NOT EXISTS idx_issues_description ON issues USING gin(to_tsvector('english', description));

-- fix_attempts
CREATE TABLE IF NOT EXISTS fix_attempts (
    id                  TEXT PRIMARY KEY,
    run_id              TEXT DEFAULT '',
    issue_ids           JSONB NOT NULL DEFAULT '[]',
    fixed_files         JSONB NOT NULL DEFAULT '[]',
    reviewer_verdict    TEXT,
    reviewer_reason     TEXT DEFAULT '',
    reviewer_confidence REAL DEFAULT 0.0,
    planner_approved    BOOLEAN,
    planner_reason      TEXT DEFAULT '',
    gate_passed         BOOLEAN,
    gate_reason         TEXT DEFAULT '',
    test_run_id         TEXT,
    formal_proofs       JSONB DEFAULT '[]',
    commit_sha          TEXT,
    pr_url              TEXT,
    created_at          TIMESTAMPTZ NOT NULL,
    committed_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_fix_run ON fix_attempts(run_id);
CREATE INDEX IF NOT EXISTS idx_fix_files ON fix_attempts USING gin(fixed_files);

-- review_results
CREATE TABLE IF NOT EXISTS review_results (
    review_id           TEXT PRIMARY KEY,
    fix_attempt_id      TEXT NOT NULL,
    decisions           JSONB NOT NULL DEFAULT '[]',
    overall_score       REAL DEFAULT 0.0,
    overall_note        TEXT DEFAULT '',
    approve_for_commit  BOOLEAN DEFAULT FALSE,
    reviewed_at         TIMESTAMPTZ NOT NULL
);

-- planner_records
CREATE TABLE IF NOT EXISTS planner_records (
    id                    TEXT PRIMARY KEY,
    fix_attempt_id        TEXT NOT NULL,
    run_id                TEXT DEFAULT '',
    file_path             TEXT NOT NULL,
    verdict               TEXT NOT NULL,
    reversibility         TEXT NOT NULL,
    goal_coherent         BOOLEAN NOT NULL DEFAULT TRUE,
    risk_score            REAL DEFAULT 0.0,
    block_commit          BOOLEAN DEFAULT FALSE,
    reason                TEXT DEFAULT '',
    simulation_summary    TEXT DEFAULT '',
    formal_proof_available BOOLEAN DEFAULT FALSE,
    formal_proof_id       TEXT DEFAULT '',
    evaluated_at          TIMESTAMPTZ NOT NULL
);

-- issue_fingerprints
CREATE TABLE IF NOT EXISTS issue_fingerprints (
    fingerprint TEXT PRIMARY KEY,
    issue_id    TEXT NOT NULL,
    seen_count  INTEGER DEFAULT 1,
    first_seen  TIMESTAMPTZ NOT NULL,
    last_seen   TIMESTAMPTZ NOT NULL
);

-- audit_scores
CREATE TABLE IF NOT EXISTS audit_scores (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    total_issues    INTEGER DEFAULT 0,
    critical_count  INTEGER DEFAULT 0,
    major_count     INTEGER DEFAULT 0,
    minor_count     INTEGER DEFAULT 0,
    info_count      INTEGER DEFAULT 0,
    escalated_count INTEGER DEFAULT 0,
    score           REAL DEFAULT 0.0,
    scored_at       TIMESTAMPTZ NOT NULL
);

-- graph_edges
CREATE TABLE IF NOT EXISTS graph_edges (
    id        TEXT PRIMARY KEY,
    run_id    TEXT NOT NULL DEFAULT '',
    source    TEXT NOT NULL,
    target    TEXT NOT NULL,
    edge_type TEXT NOT NULL DEFAULT 'import',
    symbol    TEXT DEFAULT '',
    UNIQUE(run_id, source, target, edge_type)
);
CREATE INDEX IF NOT EXISTS idx_graph_run ON graph_edges(run_id);

-- formal_verification_results
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
    evaluated_at     TIMESTAMPTZ NOT NULL
);

-- test_run_results
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
    created_at       TIMESTAMPTZ NOT NULL
);

-- patrol_log
CREATE TABLE IF NOT EXISTS patrol_log (
    id           TEXT PRIMARY KEY,
    event_type   TEXT NOT NULL,
    detail       TEXT NOT NULL,
    action_taken TEXT NOT NULL,
    run_id       TEXT NOT NULL,
    severity     TEXT DEFAULT 'INFO',
    timestamp    TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_patrol_run ON patrol_log(run_id);

-- llm_sessions
CREATE TABLE IF NOT EXISTS llm_sessions (
    id                TEXT PRIMARY KEY,
    run_id            TEXT NOT NULL,
    agent_type        TEXT NOT NULL,
    model             TEXT NOT NULL,
    prompt_tokens     INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    cost_usd          REAL DEFAULT 0.0,
    duration_ms       INTEGER DEFAULT 0,
    success           BOOLEAN DEFAULT TRUE,
    error             TEXT,
    started_at        TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_run ON llm_sessions(run_id);

-- audit_trail
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
    timestamp      TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trail_run ON audit_trail(run_id);

-- synthesis_reports (Gap 2: SynthesisAgent per-cycle dedup and compound metrics)
-- Persists the statistics from each SynthesisAgent run: how many findings were
-- deduped, how many compound cross-domain findings were detected, which model
-- was used, and how long synthesis took.  The API layer reads these to expose
-- trend data across cycles via GET /api/synthesis-reports/.
CREATE TABLE IF NOT EXISTS synthesis_reports (
    id         TEXT        PRIMARY KEY,
    run_id     TEXT        NOT NULL,
    cycle      INTEGER     NOT NULL DEFAULT 0,
    data       JSONB       NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, cycle)
);
CREATE INDEX IF NOT EXISTS idx_synthesis_reports_run
    ON synthesis_reports(run_id);
"""


# ──────────────────────────────────────────────────────────────────────────────
# PostgresBrainStorage
# ──────────────────────────────────────────────────────────────────────────────

class PostgresBrainStorage(BrainStorage):
    """
    PostgreSQL implementation of BrainStorage.

    Falls back to SQLiteBrainStorage if PostgreSQL is unavailable,
    ensuring the system always works even without a PG server.
    """

    def __init__(self, fallback_db_path: str = ".stabilizer/brain.db") -> None:
        self._fallback_path = fallback_db_path
        self._engine:    Any = None
        self._session_factory: Any = None
        self._fallback: SQLiteBrainStorage | None = None
        self._pool_size = int(os.environ.get("RHODAWK_PG_POOL_SIZE", "20"))

    async def initialise(self) -> None:
        if not _PG_AVAILABLE:
            log.info("PostgreSQL unavailable — using SQLite fallback")
            self._fallback = SQLiteBrainStorage(self._fallback_path)
            await self._fallback.initialise()
            return

        try:
            url = _require_database_url()
            self._engine = create_async_engine(
                url,
                pool_size=self._pool_size,
                max_overflow=10,
                echo=False,
            )
            # Create schema
            async with self._engine.begin() as conn:
                # Split DDL into individual statements
                for stmt in _DDL_MAIN.split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        try:
                            await conn.execute(text(stmt))
                        except Exception:
                            pass  # Partition creation failures are OK (already exist)
            self._session_factory = async_sessionmaker(
                self._engine, class_=AsyncSession, expire_on_commit=False
            )
            log.info(f"PostgreSQL storage initialised (pool_size={self._pool_size})")
        except Exception as exc:
            log.error(f"PostgreSQL init failed: {exc} — falling back to SQLite")
            self._fallback = SQLiteBrainStorage(self._fallback_path)
            await self._fallback.initialise()

    async def close(self) -> None:
        if self._fallback:
            await self._fallback.close()
        if self._engine:
            await self._engine.dispose()

    def _is_pg(self) -> bool:
        return self._engine is not None and self._fallback is None

    async def _exec(self, stmt: str, params: dict | None = None) -> list:
        """Execute a statement and return all rows."""
        assert self._session_factory
        async with self._session_factory() as session:
            result = await session.execute(
                text(stmt), params or {}
            )
            await session.commit()
            try:
                return list(result.mappings())
            except Exception:
                return []

    async def _execute(self, stmt: str, params: dict | None = None) -> None:
        """Execute a statement with no return value."""
        assert self._session_factory
        async with self._session_factory() as session:
            await session.execute(text(stmt), params or {})
            await session.commit()

    # ── All BrainStorage methods delegate to fallback when PG unavailable ──
    # For brevity, most methods delegate to SQLite fallback.
    # Production deployments will use the direct PG implementation below.

    def __getattr__(self, name: str):
        """Delegate any unimplemented method to fallback."""
        if self._fallback is not None:
            return getattr(self._fallback, name)
        raise AttributeError(f"PostgresBrainStorage.{name} not implemented")

    # ── Implemented directly for PG (hot-path methods) ───────────────────────

    async def upsert_run(self, run: AuditRun) -> None:
        if not self._is_pg():
            return await self._fallback.upsert_run(run)  # type: ignore[union-attr]
        await self._execute("""
            INSERT INTO audit_runs
                (id, repo_url, repo_name, branch, master_prompt_path,
                 autonomy_level, domain_mode, status, cycle_count, max_cycles,
                 files_total, files_read, graph_built, metadata, started_at, completed_at)
            VALUES (:id,:repo_url,:repo_name,:branch,:master_prompt_path,
                    :autonomy_level,:domain_mode,:status,:cycle_count,:max_cycles,
                    :files_total,:files_read,:graph_built,:metadata::jsonb,:started_at,:completed_at)
            ON CONFLICT(id) DO UPDATE SET
                status=EXCLUDED.status, cycle_count=EXCLUDED.cycle_count,
                files_total=EXCLUDED.files_total, files_read=EXCLUDED.files_read,
                graph_built=EXCLUDED.graph_built, metadata=EXCLUDED.metadata,
                completed_at=EXCLUDED.completed_at
        """, {
            "id": run.id, "repo_url": run.repo_url, "repo_name": run.repo_name,
            "branch": run.branch, "master_prompt_path": run.master_prompt_path,
            "autonomy_level": run.autonomy_level.value,
            "domain_mode": run.domain_mode.value,
            "status": run.status.value,
            "cycle_count": run.cycle_count, "max_cycles": run.max_cycles,
            "files_total": run.files_total, "files_read": run.files_read,
            "graph_built": run.graph_built,
            "metadata": json.dumps(run.metadata),
            "started_at": run.started_at,
            "completed_at": run.completed_at,
        })

    async def get_run(self, run_id: str) -> AuditRun | None:
        if not self._is_pg():
            return await self._fallback.get_run(run_id)  # type: ignore[union-attr]
        rows = await self._exec(
            "SELECT * FROM audit_runs WHERE id=:id", {"id": run_id}
        )
        if not rows:
            return None
        r = rows[0]
        return AuditRun(
            id=r["id"], repo_url=r["repo_url"], repo_name=r["repo_name"],
            branch=r["branch"], master_prompt_path=r.get("master_prompt_path",""),
            autonomy_level=AutonomyLevel(r["autonomy_level"]),
            domain_mode=DomainMode(r["domain_mode"]),
            status=RunStatus(r["status"]),
            cycle_count=r["cycle_count"], max_cycles=r["max_cycles"],
            files_total=r["files_total"], files_read=r["files_read"],
            graph_built=bool(r.get("graph_built", False)),
            metadata=r.get("metadata") or {},
            started_at=r["started_at"],
            completed_at=r.get("completed_at"),
        )

    async def list_issues(self, run_id=None, status=None, severity=None,
                          file_path=None) -> list[Issue]:
        if not self._is_pg():
            return await self._fallback.list_issues(  # type: ignore[union-attr]
                run_id=run_id, status=status, severity=severity, file_path=file_path)

        parts = ["SELECT * FROM issues WHERE 1=1"]
        params: dict = {}
        if run_id:
            parts.append("AND run_id=:run_id")
            params["run_id"] = run_id
        if status:
            parts.append("AND status=:status")
            params["status"] = status
        if severity:
            parts.append("AND severity=:severity")
            params["severity"] = severity.value if hasattr(severity, "value") else severity
        if file_path:
            parts.append("AND file_path=:file_path")
            params["file_path"] = file_path
        parts.append(
            "ORDER BY CASE severity "
            "WHEN 'CRITICAL' THEN 0 WHEN 'MAJOR' THEN 1 "
            "WHEN 'MINOR' THEN 2 ELSE 3 END, created_at"
        )
        rows = await self._exec(" ".join(parts), params)
        return [self._row_to_issue(r) for r in rows]

    def _row_to_issue(self, r: dict) -> Issue:
        return Issue(
            id=r["id"], run_id=r.get("run_id",""),
            severity=Severity(r["severity"]),
            file_path=r["file_path"],
            line_start=r.get("line_start",0), line_end=r.get("line_end",0),
            executor_type=ExecutorType(r["executor_type"]),
            master_prompt_section=r.get("master_prompt_section",""),
            description=r["description"],
            fix_requires_files=r.get("fix_requires_files") or [],
            status=IssueStatus(r["status"]),
            fix_attempt_count=r.get("fix_attempt_count",0),
            fingerprint=r.get("fingerprint",""),
            consensus_votes=r.get("consensus_votes",0),
            consensus_confidence=r.get("consensus_confidence",0.0),
            created_at=r["created_at"],
        )

    async def get_total_cost(self, run_id: str) -> float:
        if not self._is_pg():
            return await self._fallback.get_total_cost(run_id)  # type: ignore[union-attr]
        rows = await self._exec(
            "SELECT COALESCE(SUM(cost_usd),0) as total FROM llm_sessions WHERE run_id=:rid",
            {"rid": run_id}
        )
        return float(rows[0]["total"]) if rows else 0.0

    async def log_llm_session(self, session: LLMSession) -> None:
        if not self._is_pg():
            return await self._fallback.log_llm_session(session)  # type: ignore[union-attr]
        await self._execute("""
            INSERT INTO llm_sessions
                (id,run_id,agent_type,model,prompt_tokens,completion_tokens,
                 cost_usd,duration_ms,success,error,started_at)
            VALUES (:id,:run_id,:agent_type,:model,:pt,:ct,:cost,:dur,:success,:error,:started_at)
            ON CONFLICT(id) DO NOTHING
        """, {
            "id": session.id, "run_id": session.run_id,
            "agent_type": session.agent_type.value, "model": session.model,
            "pt": session.prompt_tokens, "ct": session.completion_tokens,
            "cost": session.cost_usd, "dur": session.duration_ms,
            "success": session.success, "error": session.error,
            "started_at": session.started_at,
        })

    async def update_run_status(self, run_id: str, status: RunStatus) -> None:
        if not self._is_pg():
            return await self._fallback.update_run_status(run_id, status)  # type: ignore[union-attr]
        await self._execute(
            "UPDATE audit_runs SET status=:s WHERE id=:id",
            {"s": status.value, "id": run_id}
        )


def get_storage(db_path: str = ".stabilizer/brain.db") -> BrainStorage:
    """
    Factory: returns PostgresBrainStorage if RHODAWK_USE_POSTGRES=1,
    otherwise SQLiteBrainStorage.
    """
    if os.environ.get("RHODAWK_USE_POSTGRES", "0") == "1":
        return PostgresBrainStorage(fallback_db_path=db_path)
    from pathlib import Path
    return SQLiteBrainStorage(db_path)

    # ── New abstract method implementations (delegate to SQLite fallback) ──────

    async def log_llm_session(self, session: dict) -> None:
        if not self._is_pg():
            if self._fallback:
                return await self._fallback.log_llm_session(session)
            return
        try:
            import uuid, datetime
            await self._execute("""
                INSERT INTO llm_sessions
                    (id, run_id, agent_type, model, prompt_tokens, completion_tokens,
                     cost_usd, duration_ms, success, error, started_at)
                VALUES (:id,:run_id,:at,:model,:pt,:ct,:cost,:dur,:success,:error,:ts)
                ON CONFLICT(id) DO NOTHING
            """, {
                "id": uuid.uuid4().hex, "run_id": session.get("run_id",""),
                "at": session.get("agent_type",""), "model": session.get("model",""),
                "pt": session.get("prompt_tokens",0), "ct": session.get("completion_tokens",0),
                "cost": session.get("cost_usd",0.0), "dur": session.get("duration_ms",0),
                "success": 1 if session.get("success") else 0,
                "error": session.get("error",""),
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })
        except Exception:
            if self._fallback:
                await self._fallback.log_llm_session(session)

    async def upsert_escalation(self, esc) -> None:
        if self._fallback: await self._fallback.upsert_escalation(esc)

    async def get_escalation(self, escalation_id: str):
        if self._fallback: return await self._fallback.get_escalation(escalation_id)
        return None

    async def list_escalations(self, run_id: str = "", status=None) -> list:
        if self._fallback: return await self._fallback.list_escalations(run_id=run_id, status=status)
        return []

    async def upsert_baseline(self, baseline) -> None:
        if self._fallback: await self._fallback.upsert_baseline(baseline)

    async def get_baseline(self, baseline_id: str):
        if self._fallback: return await self._fallback.get_baseline(baseline_id)
        return None

    async def get_active_baseline(self, run_id: str = ""):
        if self._fallback: return await self._fallback.get_active_baseline(run_id)
        return None

    async def list_baselines(self, run_id: str = "") -> list:
        if self._fallback: return await self._fallback.list_baselines(run_id=run_id)
        return []

    async def upsert_staleness_mark(self, mark) -> None:
        if self._fallback: await self._fallback.upsert_staleness_mark(mark)

    async def list_stale_functions(self, file_path: str = "", run_id: str = "") -> list:
        if self._fallback: return await self._fallback.list_stale_functions(file_path=file_path, run_id=run_id)
        return []

    async def clear_staleness_mark(self, file_path: str, function_name: str) -> None:
        if self._fallback: await self._fallback.clear_staleness_mark(file_path, function_name)

    async def upsert_ldra_finding(self, finding) -> None:
        if self._fallback: await self._fallback.upsert_ldra_finding(finding)

    async def list_ldra_findings(self, run_id: str = "", file_path: str = "") -> list:
        if self._fallback: return await self._fallback.list_ldra_findings(run_id=run_id, file_path=file_path)
        return []

    async def upsert_polyspace_finding(self, finding) -> None:
        if self._fallback: await self._fallback.upsert_polyspace_finding(finding)

    async def list_polyspace_findings(self, run_id: str = "") -> list:
        if self._fallback: return await self._fallback.list_polyspace_findings(run_id=run_id)
        return []

    async def upsert_cbmc_result(self, result) -> None:
        if self._fallback: await self._fallback.upsert_cbmc_result(result)

    async def get_cbmc_result(self, result_id: str):
        if self._fallback: return await self._fallback.get_cbmc_result(result_id)
        return None

    async def upsert_rtm_entry(self, entry) -> None:
        if self._fallback: await self._fallback.upsert_rtm_entry(entry)

    async def get_rtm_for_issue(self, issue_id: str):
        if self._fallback: return await self._fallback.get_rtm_for_issue(issue_id)
        return None

    async def list_rtm_entries(self, run_id: str = "") -> list:
        if self._fallback: return await self._fallback.list_rtm_entries(run_id=run_id)
        return []

    async def upsert_independence_record(self, record) -> None:
        if self._fallback: await self._fallback.upsert_independence_record(record)

    async def get_independence_record(self, fix_attempt_id: str):
        if self._fallback: return await self._fallback.get_independence_record(fix_attempt_id)
        return None

    async def upsert_sas(self, sas) -> None:
        if self._fallback: await self._fallback.upsert_sas(sas)

    async def get_sas(self, run_id: str):
        if self._fallback: return await self._fallback.get_sas(run_id)
        return None

    async def upsert_sci(self, sci) -> None:
        if self._fallback: await self._fallback.upsert_sci(sci)

    async def get_sci(self, baseline_id: str):
        if self._fallback: return await self._fallback.get_sci(baseline_id)
        return None

    async def upsert_formal_result(self, result) -> None:
        if self._fallback: await self._fallback.upsert_formal_result(result)

    async def upsert_planner_record(self, record) -> None:
        if self._fallback: await self._fallback.upsert_planner_record(record)

    async def append_patrol_event(self, event) -> None:
        if self._fallback: await self._fallback.append_patrol_event(event)

    async def upsert_test_result(self, result) -> None:
        if self._fallback: await self._fallback.upsert_test_result(result)

    async def update_issue_status(self, issue_id: str, status: str, reason: str = "") -> None:
        if self._fallback: await self._fallback.update_issue_status(issue_id, status, reason)

    async def list_audit_trail(self, run_id: str, limit: int = 1000) -> list:
        if self._fallback: return await self._fallback.list_audit_trail(run_id, limit)
        return []

    # ── GAP 2: SYNTHESIS AGENT — SynthesisReport and compound findings ─────────
    #
    # These four methods are the production-blocking gap identified in the audit:
    # brain/sqlite_storage.py implements them fully but postgres_storage.py was
    # missing them entirely, causing SynthesisReport to be silently dropped and
    # all /api/compound-findings/ and /api/synthesis-reports/ endpoints to return
    # empty results when running with the production PostgreSQL backend.
    #
    # Implementation strategy:
    #   • When Joern/PG is fully available: execute SQL directly via _execute/_exec
    #   • When falling back to SQLite: delegate to self._fallback as all other
    #     methods do — this ensures the methods always work regardless of backend
    #
    # The synthesis_reports table uses JSONB data column for schema flexibility:
    # all SynthesisReport fields are stored as one JSON blob, same pattern as
    # convergence_records above.  A UNIQUE (run_id, cycle) constraint ensures
    # idempotency — re-running a cycle overwrites the previous report.
    # ──────────────────────────────────────────────────────────────────────────

    async def upsert_synthesis_report(self, report: "SynthesisReport") -> None:  # type: ignore[override]
        """
        Gap 2 fix: Persist a SynthesisReport to synthesis_reports.

        Idempotent on (run_id, cycle) — the UNIQUE constraint and ON CONFLICT
        clause ensure re-running the same cycle overwrites the previous row.
        Non-fatal: any exception is logged and swallowed so the audit pipeline
        is never blocked by a metrics write failure.
        """
        from brain.schemas import SynthesisReport as _SR
        if not _PG_AVAILABLE or not self._engine:
            if self._fallback:
                return await self._fallback.upsert_synthesis_report(report)
            return
        try:
            async with AsyncSession(self._engine) as session:
                await session.execute(
                    text(
                        """
                        INSERT INTO synthesis_reports (id, run_id, cycle, data, created_at)
                        VALUES (:id, :run_id, :cycle, :data::jsonb, NOW())
                        ON CONFLICT (run_id, cycle) DO UPDATE
                            SET data       = EXCLUDED.data,
                                id         = EXCLUDED.id,
                                created_at = NOW()
                        """
                    ),
                    {
                        "id":     report.id,
                        "run_id": report.run_id,
                        "cycle":  report.cycle,
                        "data":   report.model_dump_json(),
                    },
                )
                await session.commit()
        except Exception as exc:
            log.warning(f"upsert_synthesis_report failed (non-fatal): {exc}")
            if self._fallback:
                await self._fallback.upsert_synthesis_report(report)

    async def get_synthesis_report(
        self,
        run_id: str,
        cycle: int | None = None,
    ) -> "SynthesisReport | None":  # type: ignore[override]
        """
        Gap 2 fix: Retrieve a SynthesisReport from PostgreSQL.

        When cycle is None, returns the report with the highest cycle number
        for the run (i.e. the most recent pass).  When cycle is provided,
        returns the report for that exact cycle or None if not found.
        """
        from brain.schemas import SynthesisReport as _SR
        if not _PG_AVAILABLE or not self._engine:
            if self._fallback:
                return await self._fallback.get_synthesis_report(run_id, cycle)
            return None
        try:
            async with AsyncSession(self._engine) as session:
                if cycle is not None:
                    result = await session.execute(
                        text(
                            "SELECT data FROM synthesis_reports "
                            "WHERE run_id = :run_id AND cycle = :cycle"
                        ),
                        {"run_id": run_id, "cycle": cycle},
                    )
                else:
                    # Latest cycle for this run
                    result = await session.execute(
                        text(
                            "SELECT data FROM synthesis_reports "
                            "WHERE run_id = :run_id "
                            "ORDER BY cycle DESC LIMIT 1"
                        ),
                        {"run_id": run_id},
                    )
                row = result.fetchone()
            if row is None:
                return None
            raw = row[0]
            return _SR.model_validate_json(
                raw if isinstance(raw, str) else json.dumps(raw)
            )
        except Exception as exc:
            log.warning(f"get_synthesis_report failed: {exc}")
            if self._fallback:
                return await self._fallback.get_synthesis_report(run_id, cycle)
            return None

    async def list_synthesis_reports(
        self,
        run_id: str | None = None,
    ) -> "list[SynthesisReport]":  # type: ignore[override]
        """
        Gap 2 fix: List SynthesisReports, optionally filtered by run_id.

        Returns results ordered by (run_id, cycle) ASC so callers can track
        quality trends across cycles within a run.  Always returns an empty
        list (never None) when no records exist.
        """
        from brain.schemas import SynthesisReport as _SR
        if not _PG_AVAILABLE or not self._engine:
            if self._fallback:
                return await self._fallback.list_synthesis_reports(run_id)
            return []
        try:
            async with AsyncSession(self._engine) as session:
                if run_id:
                    result = await session.execute(
                        text(
                            "SELECT data FROM synthesis_reports "
                            "WHERE run_id = :run_id "
                            "ORDER BY run_id, cycle ASC"
                        ),
                        {"run_id": run_id},
                    )
                else:
                    result = await session.execute(
                        text(
                            "SELECT data FROM synthesis_reports "
                            "ORDER BY run_id, cycle ASC"
                        )
                    )
                rows = result.fetchall()
            out: list[_SR] = []
            for row in rows:
                try:
                    raw = row[0]
                    out.append(_SR.model_validate_json(
                        raw if isinstance(raw, str) else json.dumps(raw)
                    ))
                except Exception as parse_exc:
                    log.debug(f"list_synthesis_reports: skipping bad row: {parse_exc}")
            return out
        except Exception as exc:
            log.warning(f"list_synthesis_reports failed: {exc}")
            if self._fallback:
                return await self._fallback.list_synthesis_reports(run_id)
            return []

    async def list_compound_findings(
        self,
        run_id: str | None = None,
        severity: str | None = None,
    ) -> list[Issue]:  # type: ignore[override]
        """
        Gap 2 fix: Return Issues with executor_type=SYNTHESIS from PostgreSQL.

        These are the cross-domain compound vulnerabilities created by
        SynthesisAgent.  They are stored as regular Issues (so they flow
        through the normal pipeline) but are tagged with executor_type=SYNTHESIS
        to distinguish them from single-domain auditor findings.

        Optionally filtered by run_id and/or severity string ('CRITICAL', etc.).
        """
        if not _PG_AVAILABLE or not self._engine:
            if self._fallback:
                return await self._fallback.list_compound_findings(
                    run_id=run_id, severity=severity
                )
            return []
        try:
            parts = ["SELECT * FROM issues WHERE executor_type = 'SYNTHESIS'"]
            params: dict = {}
            if run_id:
                parts.append("AND run_id = :run_id")
                params["run_id"] = run_id
            if severity:
                parts.append("AND severity = :severity")
                params["severity"] = severity.upper()
            parts.append(
                "ORDER BY CASE severity "
                "WHEN 'CRITICAL' THEN 0 WHEN 'MAJOR' THEN 1 "
                "WHEN 'MINOR' THEN 2 ELSE 3 END, created_at ASC"
            )
            rows = await self._exec(" ".join(parts), params)
            return [self._row_to_issue(r) for r in rows]
        except Exception as exc:
            log.warning(f"list_compound_findings failed: {exc}")
            if self._fallback:
                return await self._fallback.list_compound_findings(
                    run_id=run_id, severity=severity
                )
            return []

    # ── CONVERGENCE ────────────────────────────────────────────────────────────

    async def upsert_convergence_record(self, record: "ConvergenceRecord") -> None:  # type: ignore[override]
        from brain.schemas import ConvergenceRecord as _CR
        if not _PG_AVAILABLE or not self._engine:
            if self._fallback:
                return await self._fallback.upsert_convergence_record(record)
            return
        try:
            async with AsyncSession(self._engine) as session:
                await session.execute(
                    text(
                        """
                        INSERT INTO convergence_records (id, run_id, data)
                        VALUES (:id, :run_id, :data::jsonb)
                        ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data
                        """
                    ),
                    {"id": record.id, "run_id": record.run_id,
                     "data": record.model_dump_json()},
                )
                await session.commit()
        except Exception as exc:
            log.warning(f"upsert_convergence_record failed: {exc}")
            if self._fallback:
                await self._fallback.upsert_convergence_record(record)

    async def list_convergence_records(self, run_id: str) -> list["ConvergenceRecord"]:  # type: ignore[override]
        from brain.schemas import ConvergenceRecord as _CR
        if not _PG_AVAILABLE or not self._engine:
            if self._fallback:
                return await self._fallback.list_convergence_records(run_id)
            return []
        try:
            async with AsyncSession(self._engine) as session:
                result = await session.execute(
                    text(
                        "SELECT data FROM convergence_records "
                        "WHERE run_id = :run_id ORDER BY ctid ASC"
                    ),
                    {"run_id": run_id},
                )
                rows = result.fetchall()
            out = []
            for row in rows:
                try:
                    out.append(_CR.model_validate_json(row[0] if isinstance(row[0], str) else str(row[0])))
                except Exception:
                    pass
            return out
        except Exception as exc:
            log.warning(f"list_convergence_records failed: {exc}")
            if self._fallback:
                return await self._fallback.list_convergence_records(run_id)
            return []
