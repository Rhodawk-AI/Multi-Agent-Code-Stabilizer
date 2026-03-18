"""
brain/schemas.py
================
All Pydantic v2 data models for MACS (Multi-Agent Code Stabilizer).

CHANGELOG vs previous version
──────────────────────────────
• Added DomainMode enum (general / finance / medical / military / embedded)
• Added ConsensusRule + ConsensusRequirement for the weighted voting engine
• Added GraphNode + GraphEdge + DependencySnapshot for the dependency graph layer
• Added FormalVerificationResult for Z3/CBMC proofs attached to planner records
• Added TestRunResult for post-fix test execution tracking
• Extended PlannerRecord with formal_proof_available flag
• Extended FixAttempt with test_run_id foreign key
• All existing models preserved without breaking changes
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# ──────────────────────────────────────────────────────────────────────────────
# Core enumerations
# ──────────────────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    MAJOR    = "MAJOR"
    MINOR    = "MINOR"
    INFO     = "INFO"


class IssueStatus(str, Enum):
    OPEN           = "OPEN"
    FIX_QUEUED     = "FIX_QUEUED"
    FIX_GENERATED  = "FIX_GENERATED"
    REVIEWING      = "REVIEWING"
    APPROVED       = "APPROVED"
    REJECTED       = "REJECTED"
    CLOSED         = "CLOSED"
    ESCALATED      = "ESCALATED"
    REGRESSED      = "REGRESSED"


class FileStatus(str, Enum):
    UNREAD   = "UNREAD"
    READING  = "READING"
    READ     = "READ"
    MODIFIED = "MODIFIED"
    LOCKED   = "LOCKED"


class ReviewVerdict(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ESCALATE = "ESCALATE"


class ExecutorType(str, Enum):
    SECURITY     = "SECURITY"
    ARCHITECTURE = "ARCHITECTURE"
    STANDARDS    = "STANDARDS"
    GENERAL      = "GENERAL"
    READER       = "READER"
    FIXER        = "FIXER"
    REVIEWER     = "REVIEWER"
    PATROL       = "PATROL"
    PLANNER      = "PLANNER"
    FORMAL       = "FORMAL"
    DOMAIN       = "DOMAIN"
    TEST_RUNNER  = "TEST_RUNNER"


class ChunkStrategy(str, Enum):
    FULL          = "FULL"
    HALF          = "HALF"
    AST_NODES     = "AST_NODES"
    SKELETON      = "SKELETON"
    SKELETON_ONLY = "SKELETON_ONLY"


class RunStatus(str, Enum):
    RUNNING     = "RUNNING"
    STABILIZED  = "STABILIZED"
    HALTED      = "HALTED"
    ESCALATED   = "ESCALATED"
    FAILED      = "FAILED"


class AutonomyLevel(str, Enum):
    READ_ONLY    = "read_only"
    PROPOSE_ONLY = "propose_only"
    AUTO_FIX     = "auto_fix"


class PlannerVerdict(str, Enum):
    SAFE              = "SAFE"
    SAFE_WITH_WARNING = "SAFE_WITH_WARNING"
    UNSAFE            = "UNSAFE"
    NEEDS_SIMULATION  = "NEEDS_SIMULATION"


class ReversibilityClass(str, Enum):
    REVERSIBLE   = "REVERSIBLE"
    IRREVERSIBLE = "IRREVERSIBLE"
    CONDITIONAL  = "CONDITIONAL"


# ──────────────────────────────────────────────────────────────────────────────
# NEW — Domain modes for mission-critical contexts
# ──────────────────────────────────────────────────────────────────────────────

class DomainMode(str, Enum):
    """Selects which mission-critical rule set to inject into auditors and the gate."""
    GENERAL   = "general"    # Default — standard best-practices
    FINANCE   = "finance"    # PCI-DSS, no float on monetary values, atomic balance ops
    MEDICAL   = "medical"    # IEC 62304, HIPAA, dosage safety, audit trail completeness
    MILITARY  = "military"   # MISRA C:2012, DO-178C, deterministic execution, no malloc
    EMBEDDED  = "embedded"   # RTOS rules: no dynamic alloc, bounded loops, no stdio


# ──────────────────────────────────────────────────────────────────────────────
# NEW — Consensus engine schemas
# ──────────────────────────────────────────────────────────────────────────────

class DisagreementAction(str, Enum):
    ESCALATE_HUMAN  = "ESCALATE_HUMAN"
    DEEPER_ANALYSIS = "DEEPER_ANALYSIS"
    FLAG_UNCERTAIN  = "FLAG_UNCERTAIN"
    AUTO_REJECT     = "AUTO_REJECT"


class ConsensusRule(BaseModel):
    """
    Defines what it takes for a finding at a given severity to proceed to fix.
    Enforced by ConsensusEngine in orchestrator/consensus.py.
    """
    minimum_agents:         int                = 2
    required_domains:       list[ExecutorType] = Field(default_factory=list)
    confidence_floor:       float              = Field(0.75, ge=0.0, le=1.0)
    disagreement_action:    DisagreementAction = DisagreementAction.FLAG_UNCERTAIN
    high_centrality_raises: bool               = True
    """If the target file has betweenness centrality > 0.8, multiply confidence_floor by 1.2."""


class ConsensusVote(BaseModel):
    """Single agent's vote on a fingerprinted finding."""
    agent_id:     str
    domain:       ExecutorType
    confirms:     bool
    confidence:   float = Field(ge=0.0, le=1.0)
    reasoning:    str   = ""
    evidence_lines: list[str] = Field(default_factory=list)


class ConsensusResult(BaseModel):
    """Output of ConsensusEngine.evaluate() for a single fingerprinted finding."""
    fingerprint:         str
    approved:            bool
    weighted_confidence: float  = Field(ge=0.0, le=1.0)
    votes:               list[ConsensusVote] = Field(default_factory=list)
    action:              DisagreementAction | None = None
    reason:              str    = ""
    evaluated_at:        datetime = Field(default_factory=_utcnow)


# ──────────────────────────────────────────────────────────────────────────────
# NEW — Dependency graph schemas
# ──────────────────────────────────────────────────────────────────────────────

class EdgeType(str, Enum):
    IMPORT      = "import"     # file A imports file B
    CALL        = "call"       # function in A calls function in B
    INHERITANCE = "inheritance"# class in A inherits from class in B
    DATA_FLOW   = "data_flow"  # data flows from A to B


class GraphNode(BaseModel):
    path:            str
    language:        str             = "unknown"
    is_load_bearing: bool            = False
    size_lines:      int             = 0
    centrality:      float           = 0.0
    """Betweenness centrality — updated after each graph build."""
    page_rank:       float           = 0.0


class GraphEdge(BaseModel):
    source:    str
    target:    str
    edge_type: EdgeType = EdgeType.IMPORT
    symbol:    str      = ""
    """For CALL edges: the fully-qualified symbol name."""
    run_id:    str      = ""


class DependencySnapshot(BaseModel):
    """Serialized graph snapshot stored in the brain after each read phase."""
    id:         str      = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id:     str
    node_count: int      = 0
    edge_count: int      = 0
    created_at: datetime = Field(default_factory=_utcnow)


# ──────────────────────────────────────────────────────────────────────────────
# NEW — Formal verification result
# ──────────────────────────────────────────────────────────────────────────────

class FormalVerificationStatus(str, Enum):
    PROVEN_SAFE    = "PROVEN_SAFE"     # Z3 returned UNSAT — property always holds
    COUNTEREXAMPLE = "COUNTEREXAMPLE"  # Z3 returned SAT — bug found
    TIMEOUT        = "TIMEOUT"         # Solver exceeded time budget
    UNSUPPORTED    = "UNSUPPORTED"     # Language/construct not supported
    ERROR          = "ERROR"           # Solver error


class FormalVerificationResult(BaseModel):
    id:              str      = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id:          str      = ""
    fix_attempt_id:  str      = ""
    file_path:       str
    property_name:   str
    """Human-readable name of the invariant being verified, e.g. 'balance_non_negative'."""
    status:          FormalVerificationStatus
    counterexample:  str      = ""
    """Z3 model in string form when status == COUNTEREXAMPLE."""
    proof_summary:   str      = ""
    solver_used:     str      = "z3"
    elapsed_ms:      int      = 0
    evaluated_at:    datetime = Field(default_factory=_utcnow)


# ──────────────────────────────────────────────────────────────────────────────
# NEW — Test runner result
# ──────────────────────────────────────────────────────────────────────────────

class TestRunStatus(str, Enum):
    PASSED       = "PASSED"
    FAILED       = "FAILED"
    ERROR        = "ERROR"
    NO_TESTS     = "NO_TESTS"
    TIMED_OUT    = "TIMED_OUT"
    SKIPPED      = "SKIPPED"


class TestRunResult(BaseModel):
    id:              str           = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id:          str           = ""
    fix_attempt_id:  str           = ""
    status:          TestRunStatus
    total_tests:     int           = 0
    passed:          int           = 0
    failed:          int           = 0
    errors:          int           = 0
    duration_ms:     int           = 0
    failure_summary: str           = ""
    command_used:    str           = ""
    created_at:      datetime      = Field(default_factory=_utcnow)


# ──────────────────────────────────────────────────────────────────────────────
# Existing models — unchanged except noted
# ──────────────────────────────────────────────────────────────────────────────

class FileChunkRecord(BaseModel):
    chunk_id:           str      = Field(default_factory=lambda: str(uuid.uuid4()))
    file_path:          str
    chunk_index:        int
    total_chunks:       int
    line_start:         int
    line_end:           int
    symbols_defined:    list[str] = Field(default_factory=list)
    symbols_referenced: list[str] = Field(default_factory=list)
    dependencies:       list[str] = Field(default_factory=list)
    summary:            str       = ""
    raw_observations:   list[str] = Field(default_factory=list)
    token_count:        int       = 0
    read_at:            datetime  = Field(default_factory=_utcnow)


class FileRecord(BaseModel):
    path:             str
    content_hash:     str             = ""
    size_lines:       int             = 0
    size_bytes:       int             = 0
    language:         str             = "unknown"
    status:           FileStatus      = FileStatus.UNREAD
    chunk_strategy:   ChunkStrategy   = ChunkStrategy.FULL
    chunks_total:     int             = 0
    chunks_read:      int             = 0
    summary:          str             = ""
    is_load_bearing:  bool            = False
    last_hash_check:  datetime | None = None
    last_read_at:     datetime | None = None
    created_at:       datetime        = Field(default_factory=_utcnow)
    updated_at:       datetime        = Field(default_factory=_utcnow)

    @property
    def fully_read(self) -> bool:
        return self.chunks_read >= self.chunks_total > 0


class Issue(BaseModel):
    id:                  str      = Field(default_factory=lambda: f"ISS-{str(uuid.uuid4())[:8].upper()}")
    run_id:              str      = ""
    severity:            Severity
    file_path:           str
    line_start:          int      = 0
    line_end:            int      = 0
    executor_type:       ExecutorType
    master_prompt_section: str    = ""
    description:         str
    fix_requires_files:  list[str] = Field(default_factory=list)
    status:              IssueStatus = IssueStatus.OPEN
    fix_attempt_count:   int      = 0
    fingerprint:         str      = ""
    created_at:          datetime = Field(default_factory=_utcnow)
    closed_at:           datetime | None = None
    escalated_reason:    str | None = None
    regressed_from:      str | None = None
    # NEW — consensus fields
    consensus_votes:     int      = 0
    """Number of independent agents that confirmed this finding."""
    consensus_confidence: float   = Field(0.0, ge=0.0, le=1.0)

    @field_validator("line_end")
    @classmethod
    def line_end_gte_start(cls, v: int, info: Any) -> int:
        start = info.data.get("line_start", 0)
        return max(v, start)

    @model_validator(mode="after")
    def ensure_fix_requires_files(self) -> "Issue":
        if not self.fix_requires_files and self.file_path:
            self.fix_requires_files = [self.file_path]
        return self


class IssueFingerprint(BaseModel):
    fingerprint: str
    issue_id:    str
    seen_count:  int      = 1
    first_seen:  datetime = Field(default_factory=_utcnow)
    last_seen:   datetime = Field(default_factory=_utcnow)


class FixedFile(BaseModel):
    path:                 str
    content:              str
    issues_resolved:      list[str] = Field(default_factory=list)
    changes_made:         str       = ""
    line_count:           int       = 0
    original_line_count:  int       = 0
    diff_summary:         str       = ""

    @model_validator(mode="after")
    def compute_line_count(self) -> "FixedFile":
        if not self.line_count and self.content:
            self.line_count = len(self.content.splitlines())
        return self


class FixAttempt(BaseModel):
    id:                   str      = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id:               str      = ""
    issue_ids:            list[str]
    fixed_files:          list[FixedFile] = Field(default_factory=list)
    reviewer_verdict:     ReviewVerdict | None = None
    reviewer_reason:      str      = ""
    reviewer_confidence:  float    = 0.0
    planner_approved:     bool | None = None
    planner_reason:       str      = ""
    gate_passed:          bool | None = None
    gate_reason:          str      = ""
    commit_sha:           str | None = None
    pr_url:               str | None = None
    created_at:           datetime = Field(default_factory=_utcnow)
    committed_at:         datetime | None = None
    # NEW
    test_run_id:          str | None = None
    """Foreign key to TestRunResult — populated after post-fix test execution."""
    formal_proofs:        list[str] = Field(default_factory=list)
    """IDs of FormalVerificationResult records attached to this fix."""


class ReviewDecision(BaseModel):
    issue_id:       str
    fix_path:       str
    verdict:        ReviewVerdict
    confidence:     float = Field(ge=0.0, le=1.0)
    reason:         str
    line_references: list[str] = Field(default_factory=list)


class ReviewResult(BaseModel):
    review_id:         str      = Field(default_factory=lambda: str(uuid.uuid4()))
    fix_attempt_id:    str
    decisions:         list[ReviewDecision]
    overall_score:     float    = Field(default=0.0, ge=0.0, le=1.0)
    overall_note:      str      = ""
    approve_for_commit: bool    = False
    reviewed_at:       datetime = Field(default_factory=_utcnow)

    def compute_approval(self) -> None:
        if not self.decisions:
            self.approve_for_commit = False
            self.overall_score = 0.0
            return
        approved = [d for d in self.decisions if d.verdict == ReviewVerdict.APPROVED]
        self.approve_for_commit = len(approved) == len(self.decisions)
        self.overall_score = (
            sum(d.confidence for d in approved) / len(self.decisions)
            if self.decisions else 0.0
        )


class PlannerRecord(BaseModel):
    id:                  str      = Field(default_factory=lambda: str(uuid.uuid4()))
    fix_attempt_id:      str
    run_id:              str      = ""
    file_path:           str
    verdict:             PlannerVerdict
    reversibility:       ReversibilityClass
    goal_coherent:       bool
    risk_score:          float    = Field(ge=0.0, le=1.0)
    block_commit:        bool
    reason:              str
    simulation_summary:  str      = ""
    # NEW
    formal_proof_available: bool  = False
    formal_proof_id:     str      = ""
    evaluated_at:        datetime = Field(default_factory=_utcnow)


class AuditScore(BaseModel):
    id:              str      = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id:          str
    total_issues:    int      = 0
    critical_count:  int      = 0
    major_count:     int      = 0
    minor_count:     int      = 0
    info_count:      int      = 0
    escalated_count: int      = 0
    score:           float    = 0.0
    scored_at:       datetime = Field(default_factory=_utcnow)

    def compute_score(self) -> None:
        self.score = (
            self.critical_count * 10
            + self.major_count * 3
            + self.minor_count * 1
        )
        self.total_issues = (
            self.critical_count
            + self.major_count
            + self.minor_count
            + self.info_count
        )


class AuditRun(BaseModel):
    id:                  str          = Field(default_factory=lambda: str(uuid.uuid4()))
    repo_url:            str
    repo_name:           str
    branch:              str          = "main"
    master_prompt_path:  str          = ""
    autonomy_level:      AutonomyLevel = AutonomyLevel.AUTO_FIX
    domain_mode:         DomainMode    = DomainMode.GENERAL
    status:              RunStatus     = RunStatus.RUNNING
    cycle_count:         int           = 0
    max_cycles:          int           = 50
    scores:              list[AuditScore] = Field(default_factory=list)
    files_total:         int           = 0
    files_read:          int           = 0
    graph_built:         bool          = False
    started_at:          datetime      = Field(default_factory=_utcnow)
    completed_at:        datetime | None = None
    metadata:            dict[str, Any] = Field(default_factory=dict)

    @property
    def latest_score(self) -> AuditScore | None:
        return self.scores[-1] if self.scores else None

    @property
    def is_stabilized(self) -> bool:
        s = self.latest_score
        return s is not None and s.critical_count == 0 and s.major_count == 0


class PatrolEvent(BaseModel):
    id:           str      = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type:   str
    detail:       str
    action_taken: str
    run_id:       str
    severity:     str      = "INFO"
    timestamp:    datetime = Field(default_factory=_utcnow)


class LLMSession(BaseModel):
    id:                str      = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id:            str
    agent_type:        ExecutorType
    model:             str
    prompt_tokens:     int      = 0
    completion_tokens: int      = 0
    cost_usd:          float    = 0.0
    duration_ms:       int      = 0
    success:           bool     = True
    error:             str | None = None
    started_at:        datetime = Field(default_factory=_utcnow)


class AuditTrailEntry(BaseModel):
    id:              str      = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id:          str
    event_type:      str
    entity_id:       str      = ""
    entity_type:     str      = ""
    before_state:    str      = ""
    after_state:     str      = ""
    actor:           str      = ""
    hmac_signature:  str      = ""
    timestamp:       datetime = Field(default_factory=_utcnow)
