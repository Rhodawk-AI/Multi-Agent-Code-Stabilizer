"""
brain/schemas.py
Pydantic v2 data models for the entire OpenMOSS agent brain.
These are the canonical data structures shared by all agents.

PATCH LOG:
  - Issue: added run_id field (fixes multi-run isolation — upsert_issue was inserting NULL run_id)
  - AuditScore: added id field with UUID factory (fixes sqlite_storage using id(score) as PK)
  - ReviewResult: added overall_note field (fixes AttributeError in reviewer._store_result)
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────

class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    MAJOR    = "MAJOR"
    MINOR    = "MINOR"
    INFO     = "INFO"


class IssueStatus(str, Enum):
    OPEN            = "OPEN"
    FIX_QUEUED      = "FIX_QUEUED"
    FIX_GENERATED   = "FIX_GENERATED"
    REVIEWING       = "REVIEWING"
    APPROVED        = "APPROVED"
    REJECTED        = "REJECTED"
    CLOSED          = "CLOSED"
    ESCALATED       = "ESCALATED"      # human required
    REGRESSED       = "REGRESSED"      # reopened after close


class FileStatus(str, Enum):
    UNREAD   = "UNREAD"
    READING  = "READING"
    READ     = "READ"
    MODIFIED = "MODIFIED"
    LOCKED   = "LOCKED"               # architectural lock — human approval required


class ReviewVerdict(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ESCALATE = "ESCALATE"


class ExecutorType(str, Enum):
    SECURITY      = "SECURITY"
    ARCHITECTURE  = "ARCHITECTURE"
    STANDARDS     = "STANDARDS"
    GENERAL       = "GENERAL"
    READER        = "READER"
    FIXER         = "FIXER"
    REVIEWER      = "REVIEWER"
    PATROL        = "PATROL"
    PLANNER       = "PLANNER"


class ChunkStrategy(str, Enum):
    FULL          = "FULL"            # < 200 lines: read whole file
    HALF          = "HALF"            # 200-1000: top / bottom halves
    AST_NODES     = "AST_NODES"       # 1000-5000: by top-level AST blocks
    SKELETON      = "SKELETON"        # 5000-20000: structural skeleton first
    SKELETON_ONLY = "SKELETON_ONLY"   # > 20000: skeleton + targeted reads


class RunStatus(str, Enum):
    RUNNING     = "RUNNING"
    STABILIZED  = "STABILIZED"
    HALTED      = "HALTED"
    ESCALATED   = "ESCALATED"
    FAILED      = "FAILED"


# ─────────────────────────────────────────────────────────────
# File-level models
# ─────────────────────────────────────────────────────────────

class FileChunkRecord(BaseModel):
    """Written by the Reader agent after each LLM session for a chunk."""
    chunk_id:            str    = Field(default_factory=lambda: str(uuid.uuid4()))
    file_path:           str
    chunk_index:         int
    total_chunks:        int
    line_start:          int
    line_end:            int
    symbols_defined:     list[str] = Field(default_factory=list)
    symbols_referenced:  list[str] = Field(default_factory=list)
    dependencies:        list[str] = Field(default_factory=list)   # other file paths
    summary:             str   = ""
    raw_observations:    list[str] = Field(default_factory=list)
    token_count:         int   = 0
    read_at:             datetime = Field(default_factory=datetime.utcnow)


class FileRecord(BaseModel):
    """Master record for a single file in the repo."""
    path:            str
    content_hash:    str   = ""
    size_lines:      int   = 0
    size_bytes:      int   = 0
    language:        str   = "unknown"
    status:          FileStatus = FileStatus.UNREAD
    chunk_strategy:  ChunkStrategy = ChunkStrategy.FULL
    chunks_total:    int   = 0
    chunks_read:     int   = 0
    summary:         str   = ""
    is_load_bearing: bool  = False    # safety-critical file, needs human approval
    last_hash_check: datetime | None = None
    last_read_at:    datetime | None = None
    created_at:      datetime = Field(default_factory=datetime.utcnow)
    updated_at:      datetime = Field(default_factory=datetime.utcnow)

    @property
    def fully_read(self) -> bool:
        return self.chunks_read >= self.chunks_total > 0


# ─────────────────────────────────────────────────────────────
# Issue models
# ─────────────────────────────────────────────────────────────

class Issue(BaseModel):
    """A single discovered problem in the codebase."""
    id:                    str = Field(default_factory=lambda: f"ISS-{str(uuid.uuid4())[:8].upper()}")
    # FIX: run_id added — without this, list_issues(run_id=...) always returns empty set
    # because the DB column was never populated. Critical for multi-run isolation.
    run_id:                str = ""
    severity:              Severity
    file_path:             str
    line_start:            int = 0
    line_end:              int = 0
    executor_type:         ExecutorType
    master_prompt_section: str = ""
    description:           str
    fix_requires_files:    list[str] = Field(default_factory=list)
    status:                IssueStatus = IssueStatus.OPEN
    fix_attempt_count:     int = 0
    fingerprint:           str = ""    # hash(file+lines+description) for dedup
    created_at:            datetime = Field(default_factory=datetime.utcnow)
    closed_at:             datetime | None = None
    escalated_reason:      str | None = None

    @field_validator("line_end")
    @classmethod
    def line_end_gte_start(cls, v: int, info: Any) -> int:
        start = info.data.get("line_start", 0)
        return max(v, start)


class IssueFingerprint(BaseModel):
    """Used to detect identical issues across runs (loop prevention)."""
    fingerprint: str
    issue_id:    str
    seen_count:  int = 1
    first_seen:  datetime = Field(default_factory=datetime.utcnow)
    last_seen:   datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────
# Fix models
# ─────────────────────────────────────────────────────────────

class FixedFile(BaseModel):
    """A single file returned by the Fixer agent — COMPLETE content, no diffs."""
    path:          str
    content:       str         # full file content, every line present
    issues_resolved: list[str] = Field(default_factory=list)
    changes_made:  str = ""    # human-readable summary of changes
    line_count:    int = 0


class FixAttempt(BaseModel):
    """One fix attempt for a batch of issues."""
    id:                str = Field(default_factory=lambda: str(uuid.uuid4()))
    issue_ids:         list[str]
    fixed_files:       list[FixedFile] = Field(default_factory=list)
    reviewer_verdict:  ReviewVerdict | None = None
    reviewer_reason:   str = ""
    reviewer_confidence: float = 0.0
    commit_sha:        str | None = None
    pr_url:            str | None = None
    created_at:        datetime = Field(default_factory=datetime.utcnow)
    committed_at:      datetime | None = None


# ─────────────────────────────────────────────────────────────
# Review models
# ─────────────────────────────────────────────────────────────

class ReviewDecision(BaseModel):
    issue_id:   str
    fix_path:   str
    verdict:    ReviewVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    reason:     str


class ReviewResult(BaseModel):
    review_id:         str = Field(default_factory=lambda: str(uuid.uuid4()))
    fix_attempt_id:    str
    decisions:         list[ReviewDecision]
    overall_score:     float = Field(default=0.0, ge=0.0, le=1.0)
    # FIX: overall_note was accessed in reviewer._store_result but this field was missing.
    # AttributeError: 'ReviewResult' object has no attribute 'overall_note'
    overall_note:      str = ""
    approve_for_commit: bool = False
    reviewed_at:       datetime = Field(default_factory=datetime.utcnow)

    def compute_approval(self) -> None:
        """Approve only if ALL decisions are APPROVED."""
        self.approve_for_commit = all(
            d.verdict == ReviewVerdict.APPROVED for d in self.decisions
        )
        if self.decisions:
            approved = [d for d in self.decisions if d.verdict == ReviewVerdict.APPROVED]
            self.overall_score = sum(d.confidence for d in approved) / len(self.decisions)
        else:
            self.overall_score = 0.0


# ─────────────────────────────────────────────────────────────
# Audit run models
# ─────────────────────────────────────────────────────────────

class AuditScore(BaseModel):
    """Snapshot of audit health at a point in time."""
    # FIX: id field added — sqlite_storage was using str(id(score)) (Python object memory
    # address) as the primary key, which is non-deterministic and collides across sessions.
    id:             str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id:         str
    total_issues:   int = 0
    critical_count: int = 0
    major_count:    int = 0
    minor_count:    int = 0
    info_count:     int = 0
    score:          float = 0.0   # weighted: critical*10 + major*3 + minor*1
    scored_at:      datetime = Field(default_factory=datetime.utcnow)

    def compute_score(self) -> None:
        self.score = (
            self.critical_count * 10 +
            self.major_count    * 3  +
            self.minor_count    * 1
        )
        self.total_issues = (
            self.critical_count +
            self.major_count    +
            self.minor_count    +
            self.info_count
        )


class AuditRun(BaseModel):
    """A complete audit stabilization run."""
    id:                 str = Field(default_factory=lambda: str(uuid.uuid4()))
    repo_url:           str
    repo_name:          str
    branch:             str = "main"
    master_prompt_path: str = ""
    status:             RunStatus = RunStatus.RUNNING
    cycle_count:        int = 0
    max_cycles:         int = 50
    scores:             list[AuditScore] = Field(default_factory=list)
    files_total:        int = 0
    files_read:         int = 0
    started_at:         datetime = Field(default_factory=datetime.utcnow)
    completed_at:       datetime | None = None
    metadata:           dict[str, Any] = Field(default_factory=dict)

    @property
    def latest_score(self) -> AuditScore | None:
        return self.scores[-1] if self.scores else None

    @property
    def is_stabilized(self) -> bool:
        s = self.latest_score
        return s is not None and s.critical_count == 0 and s.major_count == 0


# ─────────────────────────────────────────────────────────────
# Patrol models
# ─────────────────────────────────────────────────────────────

class PatrolEvent(BaseModel):
    id:           str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type:   str   # TIMEOUT, REGRESSION, COST_LIMIT, THRASH, etc.
    detail:       str
    action_taken: str
    run_id:       str
    timestamp:    datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────
# LLM session tracking
# ─────────────────────────────────────────────────────────────

class LLMSession(BaseModel):
    id:           str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id:       str
    agent_type:   ExecutorType
    model:        str
    prompt_tokens:    int = 0
    completion_tokens: int = 0
    cost_usd:     float = 0.0
    duration_ms:  int = 0
    success:      bool = True
    error:        str | None = None
    started_at:   datetime = Field(default_factory=datetime.utcnow)
