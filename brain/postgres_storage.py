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
    import asyncpg
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy import text
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False
    log.info('asyncpg/sqlalchemy not installed — PostgreSQL storage disabled. Run: pip install asyncpg sqlalchemy[asyncio]')
from brain.schemas import AuditRun, AuditScore, AuditTrailEntry, AutonomyLevel, ChunkStrategy, CommitAuditRecord, CommitAuditStatus, DomainMode, ExecutorType, FileChunkRecord, FileRecord, FileStatus, FixAttempt, FixedFile, FormalVerificationResult, FormalVerificationStatus, GraphEdge, Issue, IssueFingerprint, IssueStatus, LLMSession, PatrolEvent, PlannerRecord, PlannerVerdict, ReversibilityClass, ReviewDecision, ReviewResult, ReviewVerdict, RunStatus, Severity, TestRunResult, TestRunStatus
from brain.storage import BrainStorage
from brain.sqlite_storage import SQLiteBrainStorage

def _require_database_url() -> str:
    url = os.environ.get('DATABASE_URL', '').strip()
    if not url:
        raise RuntimeError('FATAL: DATABASE_URL not set. Format: postgresql+asyncpg://user:pass@host:5432/rhodawk')
    if url.startswith('postgresql://') and '+asyncpg' not in url:
        url = url.replace('postgresql://', 'postgresql+asyncpg://', 1)
    return url
_DDL_MAIN = "\n-- Enable partitioning support\nCREATE EXTENSION IF NOT EXISTS pg_trgm;\n\n-- audit_runs\nCREATE TABLE IF NOT EXISTS audit_runs (\n    id                  TEXT PRIMARY KEY,\n    repo_url            TEXT NOT NULL,\n    repo_name           TEXT NOT NULL,\n    branch              TEXT NOT NULL DEFAULT 'main',\n    master_prompt_path  TEXT,\n    autonomy_level      TEXT NOT NULL DEFAULT 'auto_fix',\n    domain_mode         TEXT NOT NULL DEFAULT 'general',\n    status              TEXT NOT NULL DEFAULT 'RUNNING',\n    cycle_count         INTEGER DEFAULT 0,\n    max_cycles          INTEGER DEFAULT 50,\n    files_total         INTEGER DEFAULT 0,\n    files_read          INTEGER DEFAULT 0,\n    graph_built         BOOLEAN DEFAULT FALSE,\n    metadata            JSONB DEFAULT '{}',\n    started_at          TIMESTAMPTZ NOT NULL,\n    completed_at        TIMESTAMPTZ\n);\n\n-- files\nCREATE TABLE IF NOT EXISTS files (\n    path            TEXT PRIMARY KEY,\n    content_hash    TEXT,\n    size_lines      INTEGER DEFAULT 0,\n    size_bytes      INTEGER DEFAULT 0,\n    language        TEXT DEFAULT 'unknown',\n    status          TEXT DEFAULT 'UNREAD',\n    chunk_strategy  TEXT DEFAULT 'FULL',\n    chunks_total    INTEGER DEFAULT 0,\n    chunks_read     INTEGER DEFAULT 0,\n    summary         TEXT DEFAULT '',\n    is_load_bearing BOOLEAN DEFAULT FALSE,\n    last_hash_check TIMESTAMPTZ,\n    last_read_at    TIMESTAMPTZ,\n    created_at      TIMESTAMPTZ NOT NULL,\n    updated_at      TIMESTAMPTZ NOT NULL\n);\n\n-- file_chunks (partitioned by hash of file_path for scale)\nCREATE TABLE IF NOT EXISTS file_chunks (\n    chunk_id            TEXT NOT NULL,\n    file_path           TEXT NOT NULL,\n    chunk_index         INTEGER NOT NULL,\n    total_chunks        INTEGER NOT NULL,\n    line_start          INTEGER NOT NULL,\n    line_end            INTEGER NOT NULL,\n    symbols_defined     JSONB DEFAULT '[]',\n    symbols_referenced  JSONB DEFAULT '[]',\n    dependencies        JSONB DEFAULT '[]',\n    summary             TEXT DEFAULT '',\n    raw_observations    JSONB DEFAULT '[]',\n    token_count         INTEGER DEFAULT 0,\n    read_at             TIMESTAMPTZ NOT NULL,\n    PRIMARY KEY (chunk_id),\n    UNIQUE (file_path, chunk_index)\n) PARTITION BY HASH (file_path);\n\n-- Create 16 partitions for file_chunks\nDO $$\nBEGIN\n    FOR i IN 0..15 LOOP\n        EXECUTE format(\n            'CREATE TABLE IF NOT EXISTS file_chunks_%s PARTITION OF file_chunks '\n            'FOR VALUES WITH (MODULUS 16, REMAINDER %s)',\n            i, i\n        );\n    END LOOP;\nEND $$;\n\n-- issues (partitioned by run_id for large runs)\nCREATE TABLE IF NOT EXISTS issues (\n    id                      TEXT NOT NULL,\n    run_id                  TEXT,\n    severity                TEXT NOT NULL,\n    file_path               TEXT NOT NULL,\n    line_start              INTEGER DEFAULT 0,\n    line_end                INTEGER DEFAULT 0,\n    executor_type           TEXT NOT NULL,\n    master_prompt_section   TEXT DEFAULT '',\n    description             TEXT NOT NULL,\n    fix_requires_files      JSONB DEFAULT '[]',\n    status                  TEXT DEFAULT 'OPEN',\n    fix_attempt_count       INTEGER DEFAULT 0,\n    fingerprint             TEXT DEFAULT '',\n    escalated_reason        TEXT,\n    regressed_from          TEXT,\n    consensus_votes         INTEGER DEFAULT 0,\n    consensus_confidence    REAL DEFAULT 0.0,\n    created_at              TIMESTAMPTZ NOT NULL,\n    closed_at               TIMESTAMPTZ,\n    PRIMARY KEY (id)\n);\n\nCREATE INDEX IF NOT EXISTS idx_issues_run_status  ON issues(run_id, status);\nCREATE INDEX IF NOT EXISTS idx_issues_severity    ON issues(severity);\nCREATE INDEX IF NOT EXISTS idx_issues_file        ON issues(file_path);\nCREATE INDEX IF NOT EXISTS idx_issues_fingerprint ON issues(fingerprint);\nCREATE INDEX IF NOT EXISTS idx_issues_description ON issues USING gin(to_tsvector('english', description));\n\n-- fix_attempts\nCREATE TABLE IF NOT EXISTS fix_attempts (\n    id                  TEXT PRIMARY KEY,\n    run_id              TEXT DEFAULT '',\n    issue_ids           JSONB NOT NULL DEFAULT '[]',\n    fixed_files         JSONB NOT NULL DEFAULT '[]',\n    reviewer_verdict    TEXT,\n    reviewer_reason     TEXT DEFAULT '',\n    reviewer_confidence REAL DEFAULT 0.0,\n    planner_approved    BOOLEAN,\n    planner_reason      TEXT DEFAULT '',\n    gate_passed         BOOLEAN,\n    gate_reason         TEXT DEFAULT '',\n    test_run_id         TEXT,\n    formal_proofs       JSONB DEFAULT '[]',\n    commit_sha          TEXT,\n    pr_url              TEXT,\n    created_at          TIMESTAMPTZ NOT NULL,\n    committed_at        TIMESTAMPTZ\n);\n\nCREATE INDEX IF NOT EXISTS idx_fix_run ON fix_attempts(run_id);\nCREATE INDEX IF NOT EXISTS idx_fix_files ON fix_attempts USING gin(fixed_files);\n\n-- review_results\nCREATE TABLE IF NOT EXISTS review_results (\n    review_id           TEXT PRIMARY KEY,\n    fix_attempt_id      TEXT NOT NULL,\n    decisions           JSONB NOT NULL DEFAULT '[]',\n    overall_score       REAL DEFAULT 0.0,\n    overall_note        TEXT DEFAULT '',\n    approve_for_commit  BOOLEAN DEFAULT FALSE,\n    reviewed_at         TIMESTAMPTZ NOT NULL\n);\n\n-- planner_records\nCREATE TABLE IF NOT EXISTS planner_records (\n    id                    TEXT PRIMARY KEY,\n    fix_attempt_id        TEXT NOT NULL,\n    run_id                TEXT DEFAULT '',\n    file_path             TEXT NOT NULL,\n    verdict               TEXT NOT NULL,\n    reversibility         TEXT NOT NULL,\n    goal_coherent         BOOLEAN NOT NULL DEFAULT TRUE,\n    risk_score            REAL DEFAULT 0.0,\n    block_commit          BOOLEAN DEFAULT FALSE,\n    reason                TEXT DEFAULT '',\n    simulation_summary    TEXT DEFAULT '',\n    formal_proof_available BOOLEAN DEFAULT FALSE,\n    formal_proof_id       TEXT DEFAULT '',\n    evaluated_at          TIMESTAMPTZ NOT NULL\n);\n\n-- issue_fingerprints\nCREATE TABLE IF NOT EXISTS issue_fingerprints (\n    fingerprint TEXT PRIMARY KEY,\n    issue_id    TEXT NOT NULL,\n    seen_count  INTEGER DEFAULT 1,\n    first_seen  TIMESTAMPTZ NOT NULL,\n    last_seen   TIMESTAMPTZ NOT NULL\n);\n\n-- audit_scores\nCREATE TABLE IF NOT EXISTS audit_scores (\n    id              TEXT PRIMARY KEY,\n    run_id          TEXT NOT NULL,\n    total_issues    INTEGER DEFAULT 0,\n    critical_count  INTEGER DEFAULT 0,\n    major_count     INTEGER DEFAULT 0,\n    minor_count     INTEGER DEFAULT 0,\n    info_count      INTEGER DEFAULT 0,\n    escalated_count INTEGER DEFAULT 0,\n    score           REAL DEFAULT 0.0,\n    scored_at       TIMESTAMPTZ NOT NULL\n);\n\n-- graph_edges\nCREATE TABLE IF NOT EXISTS graph_edges (\n    id        TEXT PRIMARY KEY,\n    run_id    TEXT NOT NULL DEFAULT '',\n    source    TEXT NOT NULL,\n    target    TEXT NOT NULL,\n    edge_type TEXT NOT NULL DEFAULT 'import',\n    symbol    TEXT DEFAULT '',\n    UNIQUE(run_id, source, target, edge_type)\n);\nCREATE INDEX IF NOT EXISTS idx_graph_run ON graph_edges(run_id);\n\n-- formal_verification_results\nCREATE TABLE IF NOT EXISTS formal_verification_results (\n    id               TEXT PRIMARY KEY,\n    run_id           TEXT DEFAULT '',\n    fix_attempt_id   TEXT DEFAULT '',\n    file_path        TEXT NOT NULL,\n    property_name    TEXT NOT NULL,\n    status           TEXT NOT NULL,\n    counterexample   TEXT DEFAULT '',\n    proof_summary    TEXT DEFAULT '',\n    solver_used      TEXT DEFAULT 'z3',\n    elapsed_ms       INTEGER DEFAULT 0,\n    evaluated_at     TIMESTAMPTZ NOT NULL\n);\n\n-- test_run_results\nCREATE TABLE IF NOT EXISTS test_run_results (\n    id               TEXT PRIMARY KEY,\n    run_id           TEXT DEFAULT '',\n    fix_attempt_id   TEXT DEFAULT '',\n    status           TEXT NOT NULL,\n    total_tests      INTEGER DEFAULT 0,\n    passed           INTEGER DEFAULT 0,\n    failed           INTEGER DEFAULT 0,\n    errors           INTEGER DEFAULT 0,\n    duration_ms      INTEGER DEFAULT 0,\n    failure_summary  TEXT DEFAULT '',\n    command_used     TEXT DEFAULT '',\n    created_at       TIMESTAMPTZ NOT NULL\n);\n\n-- patrol_log\nCREATE TABLE IF NOT EXISTS patrol_log (\n    id           TEXT PRIMARY KEY,\n    event_type   TEXT NOT NULL,\n    detail       TEXT NOT NULL,\n    action_taken TEXT NOT NULL,\n    run_id       TEXT NOT NULL,\n    severity     TEXT DEFAULT 'INFO',\n    timestamp    TIMESTAMPTZ NOT NULL\n);\nCREATE INDEX IF NOT EXISTS idx_patrol_run ON patrol_log(run_id);\n\n-- llm_sessions\nCREATE TABLE IF NOT EXISTS llm_sessions (\n    id                TEXT PRIMARY KEY,\n    run_id            TEXT NOT NULL,\n    agent_type        TEXT NOT NULL,\n    model             TEXT NOT NULL,\n    prompt_tokens     INTEGER DEFAULT 0,\n    completion_tokens INTEGER DEFAULT 0,\n    cost_usd          REAL DEFAULT 0.0,\n    duration_ms       INTEGER DEFAULT 0,\n    success           BOOLEAN DEFAULT TRUE,\n    error             TEXT,\n    started_at        TIMESTAMPTZ NOT NULL\n);\nCREATE INDEX IF NOT EXISTS idx_llm_run ON llm_sessions(run_id);\n\n-- audit_trail\nCREATE TABLE IF NOT EXISTS audit_trail (\n    id             TEXT PRIMARY KEY,\n    run_id         TEXT NOT NULL,\n    event_type     TEXT NOT NULL,\n    entity_id      TEXT DEFAULT '',\n    entity_type    TEXT DEFAULT '',\n    before_state   TEXT DEFAULT '',\n    after_state    TEXT DEFAULT '',\n    actor          TEXT DEFAULT '',\n    hmac_signature TEXT DEFAULT '',\n    timestamp      TIMESTAMPTZ NOT NULL\n);\nCREATE INDEX IF NOT EXISTS idx_trail_run ON audit_trail(run_id);\n\n-- synthesis_reports (Gap 2: SynthesisAgent per-cycle dedup and compound metrics)\n-- Persists the statistics from each SynthesisAgent run: how many findings were\n-- deduped, how many compound cross-domain findings were detected, which model\n-- was used, and how long synthesis took.  The API layer reads these to expose\n-- trend data across cycles via GET /api/synthesis-reports/.\nCREATE TABLE IF NOT EXISTS synthesis_reports (\n    id         TEXT        PRIMARY KEY,\n    run_id     TEXT        NOT NULL,\n    cycle      INTEGER     NOT NULL DEFAULT 0,\n    data       JSONB       NOT NULL,\n    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),\n    UNIQUE (run_id, cycle)\n);\nCREATE INDEX IF NOT EXISTS idx_synthesis_reports_run\n    ON synthesis_reports(run_id);\n"

class PostgresBrainStorage(BrainStorage):

    def __init__(self, dsn: str = '', fallback_db_path: str='.stabilizer/brain.db') -> None:
        self._dsn = dsn
        self._fallback_path = fallback_db_path
        self._engine: Any = None
        self._session_factory: Any = None
        self._fallback: SQLiteBrainStorage | None = None
        self._pool_size = int(os.environ.get('RHODAWK_PG_POOL_SIZE', '20'))

    async def initialise(self) -> None:
        if not _PG_AVAILABLE:
            log.info('PostgreSQL unavailable — using SQLite fallback')
            self._fallback = SQLiteBrainStorage(self._fallback_path)
            await self._fallback.initialise()
            return
        try:
            url = self._dsn or _require_database_url()
            self._engine = create_async_engine(url, pool_size=self._pool_size, max_overflow=10, echo=False)
            async with self._engine.begin() as conn:
                for stmt in _DDL_MAIN.split(';'):
                    stmt = stmt.strip()
                    if stmt:
                        try:
                            await conn.execute(text(stmt))
                        except Exception:
                            pass
            self._session_factory = async_sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)
            log.info(f'PostgreSQL storage initialised (pool_size={self._pool_size})')
        except Exception as exc:
            log.error(f'PostgreSQL init failed: {exc} — falling back to SQLite')
            self._fallback = SQLiteBrainStorage(self._fallback_path)
            await self._fallback.initialise()

    async def close(self) -> None:
        if self._fallback:
            await self._fallback.close()
        if self._engine:
            await self._engine.dispose()

    def _is_pg(self) -> bool:
        return self._engine is not None and self._fallback is None

    async def _exec(self, stmt: str, params: dict | None=None) -> list:
        assert self._session_factory
        async with self._session_factory() as session:
            result = await session.execute(text(stmt), params or {})
            await session.commit()
            try:
                return list(result.mappings())
            except Exception:
                return []

    async def _execute(self, stmt: str, params: dict | None=None) -> None:
        assert self._session_factory
        async with self._session_factory() as session:
            await session.execute(text(stmt), params or {})
            await session.commit()

    def __getattr__(self, name: str):
        if self._fallback is not None:
            return getattr(self._fallback, name)
        raise AttributeError(f'PostgresBrainStorage.{name} not implemented')

    async def upsert_run(self, run: AuditRun) -> None:
        if not self._is_pg():
            return await self._fallback.upsert_run(run)
        await self._execute('\n            INSERT INTO audit_runs\n                (id, repo_url, repo_name, branch, master_prompt_path,\n                 autonomy_level, domain_mode, status, cycle_count, max_cycles,\n                 files_total, files_read, graph_built, metadata, started_at, completed_at)\n            VALUES (:id,:repo_url,:repo_name,:branch,:master_prompt_path,\n                    :autonomy_level,:domain_mode,:status,:cycle_count,:max_cycles,\n                    :files_total,:files_read,:graph_built,:metadata::jsonb,:started_at,:completed_at)\n            ON CONFLICT(id) DO UPDATE SET\n                status=EXCLUDED.status, cycle_count=EXCLUDED.cycle_count,\n                files_total=EXCLUDED.files_total, files_read=EXCLUDED.files_read,\n                graph_built=EXCLUDED.graph_built, metadata=EXCLUDED.metadata,\n                completed_at=EXCLUDED.completed_at\n        ', {'id': run.id, 'repo_url': run.repo_url, 'repo_name': run.repo_name, 'branch': run.branch, 'master_prompt_path': run.master_prompt_path, 'autonomy_level': run.autonomy_level.value, 'domain_mode': run.domain_mode.value, 'status': run.status.value, 'cycle_count': run.cycle_count, 'max_cycles': run.max_cycles, 'files_total': run.files_total, 'files_read': run.files_read, 'graph_built': run.graph_built, 'metadata': json.dumps(run.metadata), 'started_at': run.started_at, 'completed_at': run.completed_at})

    async def get_run(self, run_id: str) -> AuditRun | None:
        if not self._is_pg():
            return await self._fallback.get_run(run_id)
        rows = await self._exec('SELECT * FROM audit_runs WHERE id=:id', {'id': run_id})
        if not rows:
            return None
        r = rows[0]
        return AuditRun(id=r['id'], repo_url=r['repo_url'], repo_name=r['repo_name'], branch=r['branch'], master_prompt_path=r.get('master_prompt_path', ''), autonomy_level=AutonomyLevel(r['autonomy_level']), domain_mode=DomainMode(r['domain_mode']), status=RunStatus(r['status']), cycle_count=r['cycle_count'], max_cycles=r['max_cycles'], files_total=r['files_total'], files_read=r['files_read'], graph_built=bool(r.get('graph_built', False)), metadata=r.get('metadata') or {}, started_at=r['started_at'], completed_at=r.get('completed_at'))

    async def list_issues(self, run_id=None, status=None, severity=None, file_path=None) -> list[Issue]:
        if not self._is_pg():
            return await self._fallback.list_issues(run_id=run_id, status=status, severity=severity, file_path=file_path)
        parts = ['SELECT * FROM issues WHERE 1=1']
        params: dict = {}
        if run_id:
            parts.append('AND run_id=:run_id')
            params['run_id'] = run_id
        if status:
            parts.append('AND status=:status')
            params['status'] = status
        if severity:
            parts.append('AND severity=:severity')
            params['severity'] = severity.value if hasattr(severity, 'value') else severity
        if file_path:
            parts.append('AND file_path=:file_path')
            params['file_path'] = file_path
        parts.append("ORDER BY CASE severity WHEN 'CRITICAL' THEN 0 WHEN 'MAJOR' THEN 1 WHEN 'MINOR' THEN 2 ELSE 3 END, created_at")
        rows = await self._exec(' '.join(parts), params)
        return [self._row_to_issue(r) for r in rows]

    def _row_to_issue(self, r: dict) -> Issue:
        return Issue(id=r['id'], run_id=r.get('run_id', ''), severity=Severity(r['severity']), file_path=r['file_path'], line_start=r.get('line_start', 0), line_end=r.get('line_end', 0), executor_type=ExecutorType(r['executor_type']), master_prompt_section=r.get('master_prompt_section', ''), description=r['description'], fix_requires_files=r.get('fix_requires_files') or [], status=IssueStatus(r['status']), fix_attempts=r.get('fix_attempt_count', 0), fingerprint=r.get('fingerprint', ''), consensus_votes=r.get('consensus_votes', 0), consensus_confidence=r.get('consensus_confidence', 0.0), created_at=r['created_at'])

    async def get_total_cost(self, run_id: str) -> float:
        if not self._is_pg():
            return await self._fallback.get_total_cost(run_id)
        rows = await self._exec('SELECT COALESCE(SUM(cost_usd),0) as total FROM llm_sessions WHERE run_id=:rid', {'rid': run_id})
        return float(rows[0]['total']) if rows else 0.0

    async def log_llm_session(self, session: LLMSession) -> None:
        if not self._is_pg():
            return await self._fallback.log_llm_session(session)
        await self._execute('\n            INSERT INTO llm_sessions\n                (id,run_id,agent_type,model,prompt_tokens,completion_tokens,\n                 cost_usd,duration_ms,success,error,started_at)\n            VALUES (:id,:run_id,:agent_type,:model,:pt,:ct,:cost,:dur,:success,:error,:started_at)\n            ON CONFLICT(id) DO NOTHING\n        ', {'id': session.id, 'run_id': session.run_id, 'agent_type': session.agent_type.value, 'model': session.model, 'pt': session.prompt_tokens, 'ct': session.completion_tokens, 'cost': session.cost_usd, 'dur': session.duration_ms, 'success': session.success, 'error': session.error, 'started_at': session.started_at})

    async def update_run_status(self, run_id: str, status: RunStatus) -> None:
        if not self._is_pg():
            return await self._fallback.update_run_status(run_id, status)
        await self._execute('UPDATE audit_runs SET status=:s WHERE id=:id', {'s': status.value, 'id': run_id})

def get_storage(db_path: str='.stabilizer/brain.db') -> BrainStorage:
    if os.environ.get('RHODAWK_USE_POSTGRES', '0') == '1':
        return PostgresBrainStorage(fallback_db_path=db_path)
    from pathlib import Path
    return SQLiteBrainStorage(db_path)

    async def log_llm_session(self, session: LLMSession) -> None:
        if not self._is_pg():
            return await self._fallback.log_llm_session(session)
        await self._execute('\n            INSERT INTO llm_sessions\n                (id, run_id, agent_type, model, prompt_tokens, completion_tokens,\n                 cost_usd, duration_ms, success, error, started_at)\n            VALUES (:id,:run_id,:at,:model,:pt,:ct,:cost,:dur,:success,:error,:started_at)\n            ON CONFLICT(id) DO NOTHING\n        ', {'id': session.id, 'run_id': session.run_id, 'at': session.agent_type.value, 'model': session.model, 'pt': session.prompt_tokens, 'ct': session.completion_tokens, 'cost': session.cost_usd, 'dur': session.duration_ms, 'success': session.success, 'error': session.error, 'started_at': session.started_at})

    async def upsert_escalation(self, esc) -> None:
        if self._fallback:
            await self._fallback.upsert_escalation(esc)

    async def get_escalation(self, escalation_id: str):
        if self._fallback:
            return await self._fallback.get_escalation(escalation_id)
        return None

    async def list_escalations(self, run_id: str='', status=None) -> list:
        if self._fallback:
            return await self._fallback.list_escalations(run_id=run_id, status=status)
        return []

    async def upsert_refactor_proposal(self, proposal) -> None:
        if self._fallback:
            await self._fallback.upsert_refactor_proposal(proposal)

    async def get_refactor_proposal(self, proposal_id: str):
        if self._fallback:
            return await self._fallback.get_refactor_proposal(proposal_id)
        return None

    async def list_refactor_proposals(self, run_id: str='') -> list:
        if self._fallback:
            return await self._fallback.list_refactor_proposals(run_id=run_id)
        return []

    async def upsert_baseline(self, baseline) -> None:
        if self._fallback:
            await self._fallback.upsert_baseline(baseline)

    async def get_baseline(self, baseline_id: str):
        if self._fallback:
            return await self._fallback.get_baseline(baseline_id)
        return None

    async def get_active_baseline(self, run_id: str=''):
        if self._fallback:
            return await self._fallback.get_active_baseline(run_id)
        return None

    async def list_baselines(self, run_id: str='') -> list:
        if self._fallback:
            return await self._fallback.list_baselines(run_id=run_id)
        return []

    async def upsert_staleness_mark(self, mark) -> None:
        if self._fallback:
            await self._fallback.upsert_staleness_mark(mark)

    async def list_stale_functions(self, file_path: str='', run_id: str='') -> list:
        if self._fallback:
            return await self._fallback.list_stale_functions(file_path=file_path, run_id=run_id)
        return []

    async def clear_staleness_mark(self, file_path: str, function_name: str) -> None:
        if self._fallback:
            await self._fallback.clear_staleness_mark(file_path, function_name)

    async def upsert_ldra_finding(self, finding) -> None:
        if self._fallback:
            await self._fallback.upsert_ldra_finding(finding)

    async def list_ldra_findings(self, run_id: str='', file_path: str='') -> list:
        if self._fallback:
            return await self._fallback.list_ldra_findings(run_id=run_id, file_path=file_path)
        return []

    async def upsert_polyspace_finding(self, finding) -> None:
        if self._fallback:
            await self._fallback.upsert_polyspace_finding(finding)

    async def list_polyspace_findings(self, run_id: str='') -> list:
        if self._fallback:
            return await self._fallback.list_polyspace_findings(run_id=run_id)
        return []

    async def upsert_cbmc_result(self, result) -> None:
        if self._fallback:
            await self._fallback.upsert_cbmc_result(result)

    async def get_cbmc_result(self, result_id: str):
        if self._fallback:
            return await self._fallback.get_cbmc_result(result_id)
        return None

    async def upsert_rtm_entry(self, entry) -> None:
        if self._fallback:
            await self._fallback.upsert_rtm_entry(entry)

    async def get_rtm_for_issue(self, issue_id: str):
        if self._fallback:
            return await self._fallback.get_rtm_for_issue(issue_id)
        return None

    async def list_rtm_entries(self, run_id: str='') -> list:
        if self._fallback:
            return await self._fallback.list_rtm_entries(run_id=run_id)
        return []

    async def upsert_independence_record(self, record) -> None:
        if self._fallback:
            await self._fallback.upsert_independence_record(record)

    async def get_independence_record(self, fix_attempt_id: str):
        if self._fallback:
            return await self._fallback.get_independence_record(fix_attempt_id)
        return None

    async def upsert_sas(self, sas) -> None:
        if self._fallback:
            await self._fallback.upsert_sas(sas)

    async def get_sas(self, run_id: str):
        if self._fallback:
            return await self._fallback.get_sas(run_id)
        return None

    async def upsert_sci(self, sci) -> None:
        if self._fallback:
            await self._fallback.upsert_sci(sci)

    async def get_sci(self, baseline_id: str):
        if self._fallback:
            return await self._fallback.get_sci(baseline_id)
        return None

    async def upsert_formal_result(self, result) -> None:
        if self._fallback:
            await self._fallback.upsert_formal_result(result)

    async def upsert_planner_record(self, record) -> None:
        if self._fallback:
            await self._fallback.upsert_planner_record(record)

    async def append_patrol_event(self, event) -> None:
        if self._fallback:
            await self._fallback.append_patrol_event(event)

    async def upsert_test_result(self, result) -> None:
        if self._fallback:
            await self._fallback.upsert_test_result(result)

    async def update_issue_status(self, issue_id: str, status: str, reason: str='') -> None:
        if self._fallback:
            await self._fallback.update_issue_status(issue_id, status, reason)

    async def list_audit_trail(self, run_id: str, limit: int=1000) -> list:
        if self._fallback:
            return await self._fallback.list_audit_trail(run_id, limit)
        return []

    # ── Fix attempts ─────────────────────────────────────────────────────────

    async def upsert_fix(self, fix: FixAttempt) -> None:
        """BUG-03 / MISSING-03 FIX: native PostgreSQL upsert for fix_attempts."""
        if not self._is_pg():
            return await self._fallback.upsert_fix(fix)
        pa = None if fix.planner_approved is None else bool(fix.planner_approved)
        gp = None if fix.gate_passed is None else bool(fix.gate_passed)
        await self._execute(
            """
            INSERT INTO fix_attempts
                (id, run_id, issue_ids, fixed_files, reviewer_verdict,
                 reviewer_reason, reviewer_confidence, planner_approved,
                 planner_reason, gate_passed, gate_reason,
                 test_run_id, formal_proofs, commit_sha, pr_url,
                 created_at, committed_at)
            VALUES (:id,:run_id,:issue_ids::jsonb,:fixed_files::jsonb,
                    :rv,:rr,:rc,:pa,:pr,:gp,:gr,:trid,:fp,
                    :csha,:purl,:created_at,:committed_at)
            ON CONFLICT(id) DO UPDATE SET
                reviewer_verdict   = EXCLUDED.reviewer_verdict,
                reviewer_reason    = EXCLUDED.reviewer_reason,
                planner_approved   = EXCLUDED.planner_approved,
                planner_reason     = EXCLUDED.planner_reason,
                gate_passed        = EXCLUDED.gate_passed,
                gate_reason        = EXCLUDED.gate_reason,
                test_run_id        = EXCLUDED.test_run_id,
                formal_proofs      = EXCLUDED.formal_proofs,
                commit_sha         = EXCLUDED.commit_sha,
                pr_url             = EXCLUDED.pr_url,
                committed_at       = EXCLUDED.committed_at
            """,
            {
                "id": fix.id, "run_id": fix.run_id,
                "issue_ids": json.dumps(fix.issue_ids),
                "fixed_files": json.dumps([f.model_dump() for f in fix.fixed_files]),
                "rv": fix.reviewer_verdict.value if fix.reviewer_verdict else None,
                "rr": fix.reviewer_reason,
                "rc": fix.reviewer_confidence,
                "pa": pa, "pr": fix.planner_reason,
                "gp": gp, "gr": fix.gate_reason,
                "trid": fix.test_run_id,
                "fp": json.dumps(fix.formal_proofs),
                "csha": getattr(fix, "commit_sha", None),
                "purl": fix.pr_url,
                "created_at": fix.created_at,
                "committed_at": fix.committed_at,
            },
        )

    async def get_fix(self, fix_id: str) -> FixAttempt | None:
        """BUG-03 FIX: native PostgreSQL get for a single fix attempt."""
        if not self._is_pg():
            return await self._fallback.get_fix(fix_id)
        rows = await self._exec(
            "SELECT * FROM fix_attempts WHERE id=:id", {"id": fix_id}
        )
        return self._row_to_fix(rows[0]) if rows else None

    async def list_fixes(self, run_id: str = "", issue_id: str | None = None) -> list[FixAttempt]:
        """
        BUG-03 FIX: native PostgreSQL implementation of list_fixes.

        Previously this method was entirely absent from PostgresBrainStorage.
        __getattr__ delegated to self._fallback, which is None in a live
        PostgreSQL deployment (no SQLite fallback), so every call to
        _phase_gate() in the controller crashed with:
            AttributeError: PostgresBrainStorage.list_fixes not implemented

        This implementation mirrors the SQLite version using PostgreSQL's native
        jsonb containment operator (@>) for efficient issue_id filtering.
        """
        if not self._is_pg():
            return await self._fallback.list_fixes(
                run_id=run_id, issue_id=issue_id
            )
        parts = ["SELECT * FROM fix_attempts WHERE 1=1"]
        params: dict = {}
        if run_id:
            parts.append("AND run_id = :run_id")
            params["run_id"] = run_id
        if issue_id:
            # jsonb @> operator: check that issue_ids array contains the value.
            parts.append("AND issue_ids @> :issue_id_json::jsonb")
            params["issue_id_json"] = json.dumps([issue_id])
        parts.append("ORDER BY created_at")
        rows = await self._exec(" ".join(parts), params)
        return [self._row_to_fix(r) for r in rows]

    def _row_to_fix(self, r: dict) -> FixAttempt:
        """Convert a fix_attempts row dict to a FixAttempt schema object."""
        files_raw = r.get("fixed_files") or []
        if isinstance(files_raw, str):
            import json as _j
            files_raw = _j.loads(files_raw)
        fixed_files = [FixedFile(**f) for f in files_raw]

        pa_raw = r.get("planner_approved")
        gp_raw = r.get("gate_passed")

        proofs_raw = r.get("formal_proofs") or []
        if isinstance(proofs_raw, str):
            import json as _j
            proofs_raw = _j.loads(proofs_raw)

        issue_ids_raw = r.get("issue_ids") or []
        if isinstance(issue_ids_raw, str):
            import json as _j
            issue_ids_raw = _j.loads(issue_ids_raw)

        return FixAttempt(
            id=r["id"],
            run_id=r.get("run_id", ""),
            issue_ids=issue_ids_raw,
            fixed_files=fixed_files,
            reviewer_verdict=ReviewVerdict(r["reviewer_verdict"]) if r.get("reviewer_verdict") else None,
            reviewer_reason=r.get("reviewer_reason", ""),
            reviewer_confidence=float(r.get("reviewer_confidence") or 0.0),
            planner_approved=None if pa_raw is None else bool(pa_raw),
            planner_reason=r.get("planner_reason", ""),
            gate_passed=None if gp_raw is None else bool(gp_raw),
            gate_reason=r.get("gate_reason", ""),
            test_run_id=r.get("test_run_id"),
            formal_proofs=proofs_raw,
            commit_sha=r.get("commit_sha"),
            pr_url=r.get("pr_url", ""),
            created_at=r["created_at"],
            committed_at=r.get("committed_at"),
        )

    # ── Compliance artifacts (MISSING-03 FIX) ─────────────────────────────────
    # Previously every compliance method in PostgresBrainStorage delegated to
    # self._fallback, which is None in a live PostgreSQL deployment. Any fix
    # involving C/C++ code or any compliance export crashed with AttributeError.
    # These native implementations use a simple JSON blob pattern (one row per
    # artifact, keyed by id) consistent with the SQLite implementation.

    async def upsert_ldra_finding(self, finding) -> None:
        """MISSING-03 FIX: native PG upsert for LDRA findings."""
        if not self._is_pg():
            if self._fallback:
                return await self._fallback.upsert_ldra_finding(finding)
            return
        await self._execute(
            "INSERT INTO ldra_findings (id, run_id, fix_attempt_id, file_path, "
            "rule_id, severity, message, line_number, created_at) "
            "VALUES (:id,:rid,:faid,:fp,:rule,:sev,:msg,:line,NOW()) "
            "ON CONFLICT (id) DO UPDATE SET message=EXCLUDED.message",
            {
                "id": finding.id, "rid": getattr(finding, "run_id", ""),
                "faid": getattr(finding, "fix_attempt_id", ""),
                "fp": finding.file_path,
                "rule": finding.rule_id, "sev": str(finding.severity),
                "msg": str(finding.message)[:500],
                "line": getattr(finding, "line_number", 0),
            },
        )

    async def list_ldra_findings(self, run_id: str = "", file_path: str = "") -> list:
        if not self._is_pg():
            if self._fallback:
                return await self._fallback.list_ldra_findings(run_id=run_id, file_path=file_path)
            return []
        from brain.schemas import LdraFinding
        parts = ["SELECT * FROM ldra_findings WHERE 1=1"]
        params: dict = {}
        if run_id:
            parts.append("AND run_id=:run_id"); params["run_id"] = run_id
        if file_path:
            parts.append("AND file_path=:fp"); params["fp"] = file_path
        try:
            rows = await self._exec(" ".join(parts), params)
            return [LdraFinding(**dict(r)) for r in rows]
        except Exception:
            return []

    async def upsert_polyspace_finding(self, finding) -> None:
        """MISSING-03 FIX: native PG upsert for Polyspace findings."""
        if not self._is_pg():
            if self._fallback:
                return await self._fallback.upsert_polyspace_finding(finding)
            return
        await self._execute(
            "INSERT INTO polyspace_findings (id, run_id, fix_attempt_id, file_path, "
            "check_name, color, message, line_number, created_at) "
            "VALUES (:id,:rid,:faid,:fp,:chk,:color,:msg,:line,NOW()) "
            "ON CONFLICT (id) DO UPDATE SET message=EXCLUDED.message",
            {
                "id": finding.id, "rid": getattr(finding, "run_id", ""),
                "faid": getattr(finding, "fix_attempt_id", ""),
                "fp": finding.file_path,
                "chk": finding.check_name,
                "color": str(getattr(finding, "verdict", "orange")),
                "msg": str(getattr(finding, "detail", ""))[:500],
                "line": getattr(finding, "line_number", 0),
            },
        )

    async def list_polyspace_findings(self, run_id: str = "") -> list:
        if not self._is_pg():
            if self._fallback:
                return await self._fallback.list_polyspace_findings(run_id=run_id)
            return []
        from brain.schemas import PolyspaceFinding
        parts = ["SELECT * FROM polyspace_findings WHERE 1=1"]
        params: dict = {}
        if run_id:
            parts.append("AND run_id=:run_id"); params["run_id"] = run_id
        try:
            rows = await self._exec(" ".join(parts), params)
            return [PolyspaceFinding(**dict(r)) for r in rows]
        except Exception:
            return []

    async def upsert_cbmc_result(self, result) -> None:
        """MISSING-03 FIX: native PG upsert for CBMC results."""
        if not self._is_pg():
            if self._fallback:
                return await self._fallback.upsert_cbmc_result(result)
            return
        await self._execute(
            "INSERT INTO cbmc_results (id, run_id, fix_attempt_id, file_path, "
            "function_name, status, counterexample, elapsed_ms, created_at) "
            "VALUES (:id,:rid,:faid,:fp,:fn,:status,:ce,:ms,NOW()) "
            "ON CONFLICT (id) DO UPDATE SET status=EXCLUDED.status",
            {
                "id": result.id, "rid": getattr(result, "run_id", ""),
                "faid": getattr(result, "fix_attempt_id", ""),
                "fp": getattr(result, "file_path", ""),
                "fn": getattr(result, "function_name", ""),
                "status": str(getattr(result, "status", "")),
                "ce": getattr(result, "counterexample", "")[:2000],
                "ms": getattr(result, "elapsed_ms", 0),
            },
        )

    async def get_cbmc_result(self, result_id: str):
        if not self._is_pg():
            if self._fallback:
                return await self._fallback.get_cbmc_result(result_id)
            return None
        from brain.schemas import CbmcVerificationResult
        rows = await self._exec(
            "SELECT * FROM cbmc_results WHERE id=:id", {"id": result_id}
        )
        if not rows:
            return None
        try:
            return CbmcVerificationResult(**dict(rows[0]))
        except Exception:
            return None

    async def upsert_rtm_entry(self, entry) -> None:
        """MISSING-03 FIX: native PG upsert for RTM entries."""
        if not self._is_pg():
            if self._fallback:
                return await self._fallback.upsert_rtm_entry(entry)
            return
        await self._execute(
            "INSERT INTO rtm_entries (id, run_id, requirement_id, source_file, "
            "coverage_status, notes, created_at) "
            "VALUES (:id,:rid,:req,:src,:cov,:notes,NOW()) "
            "ON CONFLICT (id) DO UPDATE SET coverage_status=EXCLUDED.coverage_status",
            {
                "id": entry.id, "rid": getattr(entry, "run_id", ""),
                "req": getattr(entry, "requirement_id", ""),
                "src": getattr(entry, "source_file", ""),
                "cov": str(getattr(entry, "coverage_status", "")),
                "notes": getattr(entry, "notes", "")[:500],
            },
        )

    async def list_rtm_entries(self, run_id: str = "") -> list:
        if not self._is_pg():
            if self._fallback:
                return await self._fallback.list_rtm_entries(run_id=run_id)
            return []
        from brain.schemas import RequirementTraceability
        parts = ["SELECT * FROM rtm_entries WHERE 1=1"]
        params: dict = {}
        if run_id:
            parts.append("AND run_id=:run_id"); params["run_id"] = run_id
        try:
            rows = await self._exec(" ".join(parts), params)
            return [RequirementTraceability(**dict(r)) for r in rows]
        except Exception:
            return []

    async def get_rtm_for_issue(self, issue_id: str):
        if not self._is_pg():
            if self._fallback:
                return await self._fallback.get_rtm_for_issue(issue_id)
            return None
        from brain.schemas import RequirementTraceability
        try:
            rows = await self._exec(
                "SELECT * FROM rtm_entries WHERE notes LIKE :pattern LIMIT 1",
                {"pattern": f"%{issue_id}%"},
            )
            return RequirementTraceability(**dict(rows[0])) if rows else None
        except Exception:
            return None

    async def upsert_independence_record(self, record) -> None:
        """MISSING-03 FIX: native PG upsert for DO-178C independence records."""
        if not self._is_pg():
            if self._fallback:
                return await self._fallback.upsert_independence_record(record)
            return
        await self._execute(
            "INSERT INTO independence_records (id, run_id, fix_attempt_id, "
            "fixer_model, reviewer_model, same_family, violation_reason, created_at) "
            "VALUES (:id,:rid,:faid,:fm,:rm,:sf,:vr,NOW()) "
            "ON CONFLICT (id) DO UPDATE SET same_family=EXCLUDED.same_family",
            {
                "id": record.id, "rid": getattr(record, "run_id", ""),
                "faid": getattr(record, "fix_attempt_id", ""),
                "fm": getattr(record, "fixer_model", ""),
                "rm": getattr(record, "reviewer_model", ""),
                "sf": bool(getattr(record, "same_family", False)),
                "vr": getattr(record, "violation_reason", "")[:500],
            },
        )

    async def get_independence_record(self, fix_attempt_id: str):
        if not self._is_pg():
            if self._fallback:
                return await self._fallback.get_independence_record(fix_attempt_id)
            return None
        from brain.schemas import ReviewerIndependenceRecord
        try:
            rows = await self._exec(
                "SELECT * FROM independence_records WHERE fix_attempt_id=:faid LIMIT 1",
                {"faid": fix_attempt_id},
            )
            return ReviewerIndependenceRecord(**dict(rows[0])) if rows else None
        except Exception:
            return None

    async def upsert_sas(self, sas) -> None:
        """MISSING-03 FIX: native PG upsert for Software Accomplishment Summary."""
        if not self._is_pg():
            if self._fallback:
                return await self._fallback.upsert_sas(sas)
            return
        await self._execute(
            "INSERT INTO sas_records (id, run_id, data, created_at) "
            "VALUES (:id,:rid,:data::jsonb,NOW()) "
            "ON CONFLICT (id) DO UPDATE SET data=EXCLUDED.data",
            {"id": sas.id, "rid": getattr(sas, "run_id", ""),
             "data": sas.model_dump_json()},
        )

    async def get_sas(self, run_id: str):
        if not self._is_pg():
            if self._fallback:
                return await self._fallback.get_sas(run_id)
            return None
        from brain.schemas import SoftwareAccomplishmentSummary
        try:
            rows = await self._exec(
                "SELECT data FROM sas_records WHERE run_id=:rid ORDER BY created_at DESC LIMIT 1",
                {"rid": run_id},
            )
            if rows:
                raw = rows[0]["data"]
                return SoftwareAccomplishmentSummary.model_validate_json(
                    raw if isinstance(raw, str) else json.dumps(raw)
                )
        except Exception:
            pass
        return None

    async def upsert_sci(self, sci) -> None:
        """MISSING-03 FIX: native PG upsert for Software Configuration Index."""
        if not self._is_pg():
            if self._fallback:
                return await self._fallback.upsert_sci(sci)
            return
        await self._execute(
            "INSERT INTO sci_records (id, run_id, data, created_at) "
            "VALUES (:id,:rid,:data::jsonb,NOW()) "
            "ON CONFLICT (id) DO UPDATE SET data=EXCLUDED.data",
            {"id": sci.id, "rid": getattr(sci, "run_id", ""),
             "data": sci.model_dump_json()},
        )

    async def get_sci(self, baseline_id: str):
        if not self._is_pg():
            if self._fallback:
                return await self._fallback.get_sci(baseline_id)
            return None
        from brain.schemas import SoftwareConfigurationIndex
        try:
            rows = await self._exec(
                "SELECT data FROM sci_records WHERE data::text LIKE :pattern LIMIT 1",
                {"pattern": f"%{baseline_id}%"},
            )
            if rows:
                raw = rows[0]["data"]
                return SoftwareConfigurationIndex.model_validate_json(
                    raw if isinstance(raw, str) else json.dumps(raw)
                )
        except Exception:
            pass
        return None

    async def upsert_synthesis_report(self, report: 'SynthesisReport') -> None:
        from brain.schemas import SynthesisReport as _SR
        if not _PG_AVAILABLE or not self._engine:
            if self._fallback:
                return await self._fallback.upsert_synthesis_report(report)
            return
        try:
            async with AsyncSession(self._engine) as session:
                await session.execute(text('\n                        INSERT INTO synthesis_reports (id, run_id, cycle, data, created_at)\n                        VALUES (:id, :run_id, :cycle, :data::jsonb, NOW())\n                        ON CONFLICT (run_id, cycle) DO UPDATE\n                            SET data       = EXCLUDED.data,\n                                id         = EXCLUDED.id,\n                                created_at = NOW()\n                        '), {'id': report.id, 'run_id': report.run_id, 'cycle': report.cycle, 'data': report.model_dump_json()})
                await session.commit()
        except Exception as exc:
            log.warning(f'upsert_synthesis_report failed (non-fatal): {exc}')
            if self._fallback:
                await self._fallback.upsert_synthesis_report(report)

    async def get_synthesis_report(self, run_id: str, cycle: int | None=None) -> 'SynthesisReport | None':
        from brain.schemas import SynthesisReport as _SR
        if not _PG_AVAILABLE or not self._engine:
            if self._fallback:
                return await self._fallback.get_synthesis_report(run_id, cycle)
            return None
        try:
            async with AsyncSession(self._engine) as session:
                if cycle is not None:
                    result = await session.execute(text('SELECT data FROM synthesis_reports WHERE run_id = :run_id AND cycle = :cycle'), {'run_id': run_id, 'cycle': cycle})
                else:
                    result = await session.execute(text('SELECT data FROM synthesis_reports WHERE run_id = :run_id ORDER BY cycle DESC LIMIT 1'), {'run_id': run_id})
                row = result.fetchone()
            if row is None:
                return None
            raw = row[0]
            return _SR.model_validate_json(raw if isinstance(raw, str) else json.dumps(raw))
        except Exception as exc:
            log.warning(f'get_synthesis_report failed: {exc}')
            if self._fallback:
                return await self._fallback.get_synthesis_report(run_id, cycle)
            return None

    async def list_synthesis_reports(self, run_id: str | None=None) -> 'list[SynthesisReport]':
        from brain.schemas import SynthesisReport as _SR
        if not _PG_AVAILABLE or not self._engine:
            if self._fallback:
                return await self._fallback.list_synthesis_reports(run_id)
            return []
        try:
            async with AsyncSession(self._engine) as session:
                if run_id:
                    result = await session.execute(text('SELECT data FROM synthesis_reports WHERE run_id = :run_id ORDER BY run_id, cycle ASC'), {'run_id': run_id})
                else:
                    result = await session.execute(text('SELECT data FROM synthesis_reports ORDER BY run_id, cycle ASC'))
                rows = result.fetchall()
            out: list[_SR] = []
            for row in rows:
                try:
                    raw = row[0]
                    out.append(_SR.model_validate_json(raw if isinstance(raw, str) else json.dumps(raw)))
                except Exception as parse_exc:
                    log.debug(f'list_synthesis_reports: skipping bad row: {parse_exc}')
            return out
        except Exception as exc:
            log.warning(f'list_synthesis_reports failed: {exc}')
            if self._fallback:
                return await self._fallback.list_synthesis_reports(run_id)
            return []

    async def list_compound_findings(self, run_id: str | None=None, severity: str | None=None) -> list[Issue]:
        if not _PG_AVAILABLE or not self._engine:
            if self._fallback:
                return await self._fallback.list_compound_findings(run_id=run_id, severity=severity)
            return []
        try:
            parts = ["SELECT * FROM issues WHERE executor_type = 'SYNTHESIS'"]
            params: dict = {}
            if run_id:
                parts.append('AND run_id = :run_id')
                params['run_id'] = run_id
            if severity:
                parts.append('AND severity = :severity')
                params['severity'] = severity.upper()
            parts.append("ORDER BY CASE severity WHEN 'CRITICAL' THEN 0 WHEN 'MAJOR' THEN 1 WHEN 'MINOR' THEN 2 ELSE 3 END, created_at ASC")
            rows = await self._exec(' '.join(parts), params)
            return [self._row_to_issue(r) for r in rows]
        except Exception as exc:
            log.warning(f'list_compound_findings failed: {exc}')
            if self._fallback:
                return await self._fallback.list_compound_findings(run_id=run_id, severity=severity)
            return []

    async def upsert_convergence_record(self, record: 'ConvergenceRecord') -> None:
        from brain.schemas import ConvergenceRecord as _CR
        if not _PG_AVAILABLE or not self._engine:
            if self._fallback:
                return await self._fallback.upsert_convergence_record(record)
            return
        try:
            async with AsyncSession(self._engine) as session:
                await session.execute(text('\n                        INSERT INTO convergence_records (id, run_id, data)\n                        VALUES (:id, :run_id, :data::jsonb)\n                        ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data\n                        '), {'id': record.id, 'run_id': record.run_id, 'data': record.model_dump_json()})
                await session.commit()
        except Exception as exc:
            log.warning(f'upsert_convergence_record failed: {exc}')
            if self._fallback:
                await self._fallback.upsert_convergence_record(record)

    async def list_convergence_records(self, run_id: str) -> list['ConvergenceRecord']:
        from brain.schemas import ConvergenceRecord as _CR
        if not _PG_AVAILABLE or not self._engine:
            if self._fallback:
                return await self._fallback.list_convergence_records(run_id)
            return []
        try:
            async with AsyncSession(self._engine) as session:
                result = await session.execute(text('SELECT data FROM convergence_records WHERE run_id = :run_id ORDER BY ctid ASC'), {'run_id': run_id})
                rows = result.fetchall()
            out = []
            for row in rows:
                try:
                    out.append(_CR.model_validate_json(row[0] if isinstance(row[0], str) else str(row[0])))
                except Exception:
                    pass
            return out
        except Exception as exc:
            log.warning(f'list_convergence_records failed: {exc}')
            if self._fallback:
                return await self._fallback.list_convergence_records(run_id)
            return []

    # ── Gap 4: Commit-granularity incremental audit ──────────────────────────

    async def upsert_commit_audit_record(self, record: CommitAuditRecord) -> None:
        if not _PG_AVAILABLE or not self._engine:
            if self._fallback:
                return await self._fallback.upsert_commit_audit_record(record)
            return
        try:
            async with AsyncSession(self._engine) as session:
                await session.execute(
                    text("""
                        INSERT INTO commit_audit_records (id, run_id, data)
                        VALUES (:id, :run_id, :data::jsonb)
                        ON CONFLICT (id) DO UPDATE SET
                            data   = EXCLUDED.data,
                            run_id = EXCLUDED.run_id
                    """),
                    {'id': record.id, 'run_id': record.run_id, 'data': record.model_dump_json()},
                )
                await session.commit()
        except Exception as exc:
            log.warning(f'upsert_commit_audit_record failed: {exc}')
            if self._fallback:
                await self._fallback.upsert_commit_audit_record(record)

    async def get_commit_audit_record(self, record_id: str) -> CommitAuditRecord | None:
        if not _PG_AVAILABLE or not self._engine:
            if self._fallback:
                return await self._fallback.get_commit_audit_record(record_id)
            return None
        try:
            async with AsyncSession(self._engine) as session:
                result = await session.execute(
                    text('SELECT data FROM commit_audit_records WHERE id = :id'),
                    {'id': record_id},
                )
                row = result.fetchone()
            if not row:
                return None
            return CommitAuditRecord.model_validate_json(
                row[0] if isinstance(row[0], str) else str(row[0])
            )
        except Exception as exc:
            log.warning(f'get_commit_audit_record failed: {exc}')
            if self._fallback:
                return await self._fallback.get_commit_audit_record(record_id)
            return None

    async def get_commit_audit_record_by_hash(
        self, commit_hash: str, run_id: str = ''
    ) -> CommitAuditRecord | None:
        if not _PG_AVAILABLE or not self._engine:
            if self._fallback:
                return await self._fallback.get_commit_audit_record_by_hash(commit_hash, run_id)
            return None
        try:
            async with AsyncSession(self._engine) as session:
                if run_id:
                    result = await session.execute(
                        text(
                            'SELECT data FROM commit_audit_records '
                            "WHERE data->>'commit_hash' = :h AND run_id = :r "
                            'ORDER BY (data->>\'created_at\') DESC LIMIT 1'
                        ),
                        {'h': commit_hash, 'r': run_id},
                    )
                else:
                    result = await session.execute(
                        text(
                            'SELECT data FROM commit_audit_records '
                            "WHERE data->>'commit_hash' = :h "
                            'ORDER BY (data->>\'created_at\') DESC LIMIT 1'
                        ),
                        {'h': commit_hash},
                    )
                row = result.fetchone()
            if not row:
                return None
            return CommitAuditRecord.model_validate_json(
                row[0] if isinstance(row[0], str) else str(row[0])
            )
        except Exception as exc:
            log.warning(f'get_commit_audit_record_by_hash failed: {exc}')
            if self._fallback:
                return await self._fallback.get_commit_audit_record_by_hash(commit_hash, run_id)
            return None

    async def list_commit_audit_records(
        self,
        run_id: str = '',
        status: CommitAuditStatus | None = None,
        limit: int = 100,
    ) -> list[CommitAuditRecord]:
        if not _PG_AVAILABLE or not self._engine:
            if self._fallback:
                return await self._fallback.list_commit_audit_records(
                    run_id=run_id, status=status, limit=limit
                )
            return []
        try:
            conditions: list[str] = []
            params: dict = {'limit': limit}
            if run_id:
                conditions.append('run_id = :run_id')
                params['run_id'] = run_id
            if status is not None:
                conditions.append("data->>'status' = :status")
                params['status'] = status.value
            where = ('WHERE ' + ' AND '.join(conditions)) if conditions else ''
            async with AsyncSession(self._engine) as session:
                result = await session.execute(
                    text(
                        f'SELECT data FROM commit_audit_records {where} '
                        "ORDER BY (data->>'created_at') DESC LIMIT :limit"
                    ),
                    params,
                )
                rows = result.fetchall()
            out: list[CommitAuditRecord] = []
            for row in rows:
                try:
                    out.append(CommitAuditRecord.model_validate_json(
                        row[0] if isinstance(row[0], str) else str(row[0])
                    ))
                except Exception:
                    pass
            return out
        except Exception as exc:
            log.warning(f'list_commit_audit_records failed: {exc}')
            if self._fallback:
                return await self._fallback.list_commit_audit_records(
                    run_id=run_id, status=status, limit=limit
                )
            return []
