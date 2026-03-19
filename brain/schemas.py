"""
brain/schemas.py
================
All Pydantic v2 data models for Rhodawk AI Code Stabilizer.

PRODUCTION FIXES vs audit report
──────────────────────────────────
• Added MilStd882eCategory (CAT_I–IV) mapped to Severity
• Added RequirementTraceability for DO-178C RTM (Table A-5)
• Added BaselineRecord for DO-178C Sec 11 configuration management
• Added SoftwareConfigurationIndex for DO-178C SCM evidence
• Added SoftwareAccomplishmentSummary for DO-178C Sec 11.20
• Added ReviewerIndependenceRecord enforcing model-family separation (DO-178C 6.3.4)
• Added FunctionStalenessMark for function-level incremental re-audit
• Added EscalationRecord with blocking approval (DO-178C 6.3.4 / MIL-STD-882E Task 402)
• Added ComplianceViolation, DeviationRecord for rule-level findings
• Added CbmcVerificationResult for CBMC bounded model checker evidence
• Added PolyspaceFinding for abstract interpretation evidence
• Added LdraFinding for LDRA Testbed MISRA/DO-178C traceability
• Extended Issue: requirement_id, test_case_id, cwe_id, misra_rule, jsf_rule,
  cert_rule, mil882e_category, deviation_record, function_name
• Extended FixAttempt: reviewer_model_family, fixer_model_family, patch_mode,
  cbmc_result_id, polyspace_verdict, unified-diff support
• Extended AuditTrailEntry: artifact_id, artifact_type, baseline_id, change_request_id
• Extended AuditRun: software_level, tool_qualification_level, baseline_id
• Added PatchMode enum (FULL_FILE / UNIFIED_DIFF / AST_REWRITE)
• Removed FIX_RATIO_MIN — replaced by compiler-based correctness gate
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


# ─────────────────────────────────────────────────────────────────────────────
# Core enumerations
# ─────────────────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    MAJOR    = "MAJOR"
    MINOR    = "MINOR"
    INFO     = "INFO"


class MilStd882eCategory(str, Enum):
    """MIL-STD-882E Mishap Risk Assessment Code — software-level mapping."""
    CAT_I   = "CAT_I"    # Catastrophic
    CAT_II  = "CAT_II"   # Critical
    CAT_III = "CAT_III"  # Marginal
    CAT_IV  = "CAT_IV"   # Negligible
    NONE    = "NONE"


SEVERITY_TO_MIL882E: dict[Severity, MilStd882eCategory] = {
    Severity.CRITICAL: MilStd882eCategory.CAT_I,
    Severity.MAJOR:    MilStd882eCategory.CAT_II,
    Severity.MINOR:    MilStd882eCategory.CAT_III,
    Severity.INFO:     MilStd882eCategory.CAT_IV,
}


class IssueStatus(str, Enum):
    OPEN                = "OPEN"
    FIX_QUEUED          = "FIX_QUEUED"
    FIX_GENERATED       = "FIX_GENERATED"
    REVIEWING           = "REVIEWING"
    APPROVED            = "APPROVED"
    REJECTED            = "REJECTED"
    CLOSED              = "CLOSED"
    ESCALATED           = "ESCALATED"
    ESCALATION_PENDING  = "ESCALATION_PENDING"
    ESCALATION_APPROVED = "ESCALATION_APPROVED"
    ESCALATION_REJECTED = "ESCALATION_REJECTED"
    REGRESSED           = "REGRESSED"
    DEFERRED            = "DEFERRED"
    BASELINE_LOCKED     = "BASELINE_LOCKED"


class FileStatus(str, Enum):
    UNREAD   = "UNREAD"
    READING  = "READING"
    READ     = "READ"
    MODIFIED = "MODIFIED"
    LOCKED   = "LOCKED"
    STALE    = "STALE"
    PARTIAL  = "PARTIAL"


class ReviewVerdict(str, Enum):
    APPROVED         = "APPROVED"
    REJECTED         = "REJECTED"
    ESCALATE         = "ESCALATE"
    APPROVED_WARNING = "APPROVED_WARNING"


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
    CBMC         = "CBMC"
    POLYSPACE    = "POLYSPACE"
    LDRA         = "LDRA"
    MISRA        = "MISRA"
    CERT         = "CERT"
    JSF          = "JSF"


class ChunkStrategy(str, Enum):
    FULL          = "FULL"
    HALF          = "HALF"
    AST_NODES     = "AST_NODES"
    SKELETON      = "SKELETON"
    SKELETON_ONLY = "SKELETON_ONLY"
    FUNCTION      = "FUNCTION"
    PREPROCESSED  = "PREPROCESSED"


class RunStatus(str, Enum):
    RUNNING           = "RUNNING"
    STABILIZED        = "STABILIZED"
    HALTED            = "HALTED"
    ESCALATED         = "ESCALATED"
    FAILED            = "FAILED"
    BASELINE_PENDING  = "BASELINE_PENDING"
    BASELINE_APPROVED = "BASELINE_APPROVED"


class AutonomyLevel(str, Enum):
    READ_ONLY       = "READ_ONLY"
    SUGGEST         = "SUGGEST"
    AUTO_FIX        = "AUTO_FIX"
    AUTO_FIX_PR     = "AUTO_FIX_PR"
    FULL_AUTONOMOUS = "FULL_AUTONOMOUS"


class DomainMode(str, Enum):
    GENERAL   = "GENERAL"
    FINANCE   = "FINANCE"
    MEDICAL   = "MEDICAL"
    MILITARY  = "MILITARY"
    EMBEDDED  = "EMBEDDED"
    AEROSPACE = "AEROSPACE"
    NUCLEAR   = "NUCLEAR"


class ReversibilityClass(str, Enum):
    REVERSIBLE   = "REVERSIBLE"
    CONDITIONAL  = "CONDITIONAL"
    IRREVERSIBLE = "IRREVERSIBLE"


class PlannerVerdict(str, Enum):
    SAFE              = "SAFE"
    SAFE_WITH_WARNING = "SAFE_WITH_WARNING"
    UNSAFE            = "UNSAFE"
    NEEDS_HUMAN       = "NEEDS_HUMAN"


class DisagreementAction(str, Enum):
    FLAG_UNCERTAIN = "FLAG_UNCERTAIN"
    ESCALATE_HUMAN = "ESCALATE_HUMAN"
    BLOCK          = "BLOCK"
    AUTO_RESOLVE   = "AUTO_RESOLVE"


class FormalVerificationStatus(str, Enum):
    PROVED         = "PROVED"
    REFUTED        = "REFUTED"
    UNKNOWN        = "UNKNOWN"
    COUNTEREXAMPLE = "COUNTEREXAMPLE"
    TIMEOUT        = "TIMEOUT"
    ERROR          = "ERROR"
    SKIPPED        = "SKIPPED"


class TestRunStatus(str, Enum):
    PASSED  = "PASSED"
    FAILED  = "FAILED"
    ERROR   = "ERROR"
    SKIPPED = "SKIPPED"
    PARTIAL = "PARTIAL"


class ComplianceStandard(str, Enum):
    DO_178C   = "DO-178C"
    MISRA_C   = "MISRA-C:2023"
    MISRA_CPP = "MISRA-C++:2023"
    CERT_C    = "CERT-C"
    CERT_CPP  = "CERT-C++"
    MIL_882E  = "MIL-STD-882E"
    JSF_AV    = "JSF-AV-RULES"
    CWE       = "CWE"
    OWASP     = "OWASP"
    IEC_61513 = "IEC-61513"
    IEC_62304 = "IEC-62304"
    AUTOSAR   = "AUTOSAR-C++14"


class PatchMode(str, Enum):
    FULL_FILE    = "FULL_FILE"
    UNIFIED_DIFF = "UNIFIED_DIFF"
    AST_REWRITE  = "AST_REWRITE"


class ArtifactType(str, Enum):
    REQUIREMENT = "REQUIREMENT"
    DESIGN      = "DESIGN"
    CODE        = "CODE"
    TEST_CASE   = "TEST_CASE"
    TEST_RESULT = "TEST_RESULT"
    COVERAGE    = "COVERAGE"
    EVIDENCE    = "EVIDENCE"
    FIX         = "FIX"
    AUDIT_ENTRY = "AUDIT_ENTRY"
    BASELINE    = "BASELINE"
    SAS         = "SAS"


class ToolQualificationLevel(str, Enum):
    TQL_1 = "TQL-1"
    TQL_2 = "TQL-2"
    TQL_3 = "TQL-3"
    TQL_4 = "TQL-4"
    TQL_5 = "TQL-5"
    NONE  = "NONE"


class SoftwareLevel(str, Enum):
    DAL_A = "DAL-A"
    DAL_B = "DAL-B"
    DAL_C = "DAL-C"
    DAL_D = "DAL-D"
    DAL_E = "DAL-E"
    NONE  = "NONE"


class PolyspaceVerdict(str, Enum):
    GREEN  = "GREEN"
    ORANGE = "ORANGE"
    RED    = "RED"
    GRAY   = "GRAY"


# ─────────────────────────────────────────────────────────────────────────────
# Compliance — deviation and violation
# ─────────────────────────────────────────────────────────────────────────────

class DeviationRecord(BaseModel):
    deviation_id:    str      = Field(default_factory=_new_id)
    rule_id:         str      = ""
    standard:        ComplianceStandard = ComplianceStandard.MISRA_C
    justification:   str      = ""
    approved_by:     str      = ""
    approved_at:     datetime | None = None
    expiry_date:     datetime | None = None
    risk_acceptance: str      = ""


class ComplianceViolation(BaseModel):
    rule_id:          str                  = ""
    standard:         ComplianceStandard   = ComplianceStandard.MISRA_C
    rule_description: str                  = ""
    is_mandatory:     bool                 = False
    deviation_record: DeviationRecord | None = None
    tool_detected_by: str                  = ""
    confidence:       float                = Field(ge=0.0, le=1.0, default=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Traceability (DO-178C Table A-5)
# ─────────────────────────────────────────────────────────────────────────────

class RequirementTraceability(BaseModel):
    id:               str      = Field(default_factory=_new_id)
    requirement_id:   str      = ""
    requirement_text: str      = ""
    test_case_id:     str      = ""
    test_result_id:   str      = ""
    coverage_pct:     float    = 0.0
    mcdc_coverage:    float    = 0.0
    issue_id:         str      = ""
    fix_attempt_id:   str      = ""
    verified_by:      str      = ""
    verified_at:      datetime | None = None
    do178c_objective: str      = ""


# ─────────────────────────────────────────────────────────────────────────────
# Reviewer independence (DO-178C 6.3.4)
# ─────────────────────────────────────────────────────────────────────────────

class ReviewerIndependenceRecord(BaseModel):
    id:                    str      = Field(default_factory=_new_id)
    fix_attempt_id:        str      = ""
    fixer_model:           str      = ""
    fixer_model_family:    str      = ""
    reviewer_model:        str      = ""
    reviewer_model_family: str      = ""
    independence_verified: bool     = False
    violation_reason:      str      = ""
    created_at:            datetime = Field(default_factory=_utcnow)

    @model_validator(mode="after")
    def _check_independence(self) -> "ReviewerIndependenceRecord":
        same = self.fixer_model_family.lower() == self.reviewer_model_family.lower()
        self.independence_verified = not same
        if same and not self.violation_reason:
            self.violation_reason = (
                f"DO-178C 6.3.4 violation: fixer ({self.fixer_model_family}) "
                f"and reviewer ({self.reviewer_model_family}) share model family."
            )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Formal verification
# ─────────────────────────────────────────────────────────────────────────────

class FormalVerificationResult(BaseModel):
    id:             str      = Field(default_factory=_new_id)
    fix_attempt_id: str      = ""
    file_path:      str      = ""
    property_name:  str      = ""
    status:         FormalVerificationStatus = FormalVerificationStatus.UNKNOWN
    counterexample: str      = ""
    proof_script:   str      = ""
    solver:         str      = "z3"
    elapsed_s:      float    = 0.0
    verified_at:    datetime = Field(default_factory=_utcnow)
    evidence_path:  str      = ""


class CbmcVerificationResult(BaseModel):
    id:                 str      = Field(default_factory=_new_id)
    fix_attempt_id:     str      = ""
    file_path:          str      = ""
    function_name:      str      = ""
    unwind_bound:       int      = 10
    properties_checked: list[str] = Field(default_factory=list)
    property_results:   dict[str, str] = Field(default_factory=dict)
    counterexample:     str      = ""
    stdout:             str      = ""
    return_code:        int      = 0
    elapsed_s:          float    = 0.0
    verified_at:        datetime = Field(default_factory=_utcnow)


class PolyspaceFinding(BaseModel):
    id:             str             = Field(default_factory=_new_id)
    fix_attempt_id: str             = ""
    file_path:      str             = ""
    line_number:    int             = 0
    check_name:     str             = ""
    verdict:        PolyspaceVerdict = PolyspaceVerdict.ORANGE
    category:       str             = ""
    detail:         str             = ""
    run_id:         str             = ""


class LdraFinding(BaseModel):
    id:              str               = Field(default_factory=_new_id)
    file_path:       str               = ""
    line_number:     int               = 0
    rule_id:         str               = ""
    standard:        ComplianceStandard = ComplianceStandard.MISRA_C
    severity:        str               = ""
    message:         str               = ""
    is_suppressed:   bool              = False
    suppression_ref: str               = ""
    run_id:          str               = ""


# ─────────────────────────────────────────────────────────────────────────────
# Function-level staleness
# ─────────────────────────────────────────────────────────────────────────────

class FunctionStalenessMark(BaseModel):
    id:            str      = Field(default_factory=_new_id)
    file_path:     str      = ""
    function_name: str      = ""
    line_start:    int      = 0
    line_end:      int      = 0
    stale_reason:  str      = ""
    stale_since:   datetime = Field(default_factory=_utcnow)
    run_id:        str      = ""


# ─────────────────────────────────────────────────────────────────────────────
# Human escalation (DO-178C 6.3.4 / MIL-STD-882E Task 402)
# ─────────────────────────────────────────────────────────────────────────────

class EscalationStatus(str, Enum):
    PENDING       = "PENDING"
    APPROVED      = "APPROVED"
    REJECTED      = "REJECTED"
    TIMEOUT       = "TIMEOUT"
    AUTO_RESOLVED = "AUTO_RESOLVED"


class EscalationRecord(BaseModel):
    id:               str               = Field(default_factory=_new_id)
    run_id:           str               = ""
    issue_ids:        list[str]         = Field(default_factory=list)
    fix_attempt_id:   str               = ""
    escalation_type:  str               = ""
    description:      str               = ""
    severity:         Severity          = Severity.CRITICAL
    mil882e_category: MilStd882eCategory = MilStd882eCategory.CAT_I
    status:           EscalationStatus  = EscalationStatus.PENDING
    approved_by:      str               = ""
    approved_at:      datetime | None   = None
    approval_rationale: str             = ""
    risk_acceptance:  str               = ""
    notified_via:     list[str]         = Field(default_factory=list)
    notified_at:      datetime | None   = None
    timeout_at:       datetime | None   = None
    created_at:       datetime          = Field(default_factory=_utcnow)
    updated_at:       datetime          = Field(default_factory=_utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline and configuration management (DO-178C Sec. 11)
# ─────────────────────────────────────────────────────────────────────────────

class BaselineRecord(BaseModel):
    id:               str         = Field(default_factory=_new_id)
    run_id:           str         = ""
    baseline_name:    str         = ""
    software_level:   SoftwareLevel = SoftwareLevel.NONE
    commit_hash:      str         = ""
    issue_count:      dict[str, int] = Field(default_factory=dict)
    score_snapshot:   float       = 0.0
    file_hashes:      dict[str, str] = Field(default_factory=dict)
    approved_by:      str         = ""
    approved_at:      datetime | None = None
    approval_token:   str         = ""
    change_request_id: str        = ""
    rtm_export_path:  str         = ""
    sas_export_path:  str         = ""
    created_at:       datetime    = Field(default_factory=_utcnow)
    is_active:        bool        = False


class SoftwareConfigurationIndex(BaseModel):
    id:               str      = Field(default_factory=_new_id)
    baseline_id:      str      = ""
    run_id:           str      = ""
    controlled_files: list[dict[str, str]] = Field(default_factory=list)
    tool_versions:    dict[str, str] = Field(default_factory=dict)
    compiler_config:  dict[str, str] = Field(default_factory=dict)
    generated_at:     datetime = Field(default_factory=_utcnow)
    generated_by:     str      = "Rhodawk AI"


class SoftwareAccomplishmentSummary(BaseModel):
    id:                       str         = Field(default_factory=_new_id)
    baseline_id:              str         = ""
    run_id:                   str         = ""
    software_level:           SoftwareLevel = SoftwareLevel.NONE
    tool_qualification_level: ToolQualificationLevel = ToolQualificationLevel.NONE
    total_cycles:             int         = 0
    total_issues_found:       int         = 0
    total_issues_closed:      int         = 0
    total_escalations:        int         = 0
    total_deviations:         int         = 0
    tools_used:               list[dict[str, str]] = Field(default_factory=list)
    misra_violations_open:    int         = 0
    misra_violations_closed:  int         = 0
    cert_violations_open:     int         = 0
    cert_violations_closed:   int         = 0
    cwe_findings_open:        int         = 0
    cwe_findings_closed:      int         = 0
    do178c_objectives_met:    list[str]   = Field(default_factory=list)
    do178c_objectives_open:   list[str]   = Field(default_factory=list)
    prepared_by:              str         = "Rhodawk AI"
    reviewed_by:              str         = ""
    approved_by:              str         = ""
    approved_at:              datetime | None = None
    generated_at:             datetime    = Field(default_factory=_utcnow)
    psac_deviations:          list[str]   = Field(default_factory=list)
    certification_basis:      str         = ""


# ─────────────────────────────────────────────────────────────────────────────
# File and chunk records
# ─────────────────────────────────────────────────────────────────────────────

class FileRecord(BaseModel):
    path:               str         = ""
    language:           str         = "unknown"
    status:             FileStatus  = FileStatus.UNREAD
    hash:               str         = ""
    line_count:         int         = 0
    chunk_count:        int         = 0
    summary:            str         = ""
    key_symbols:        list[str]   = Field(default_factory=list)
    dependencies:       list[str]   = Field(default_factory=list)
    all_observations:   list[str]   = Field(default_factory=list)
    is_load_bearing:    bool        = False
    run_id:             str         = ""
    last_read_at:       datetime | None = None
    known_functions:    list[str]   = Field(default_factory=list)
    stale_functions:    list[str]   = Field(default_factory=list)
    in_safety_partition: bool       = False
    software_level:     SoftwareLevel = SoftwareLevel.NONE


class FileChunkRecord(BaseModel):
    id:                  str           = Field(default_factory=_new_id)
    file_path:           str           = ""
    run_id:              str           = ""
    chunk_index:         int           = 0
    total_chunks:        int           = 1
    line_start:          int           = 0
    line_end:            int           = 0
    language:            str           = "unknown"
    strategy:            ChunkStrategy = ChunkStrategy.FULL
    summary:             str           = ""
    symbols_defined:     list[str]     = Field(default_factory=list)
    symbols_referenced:  list[str]     = Field(default_factory=list)
    dependencies:        list[str]     = Field(default_factory=list)
    raw_observations:    list[str]     = Field(default_factory=list)
    vector_id:           str           = ""
    function_name:       str           = ""
    all_functions:       list[str]     = Field(default_factory=list)
    preprocessed:        bool          = False
    created_at:          datetime      = Field(default_factory=_utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Issue
# ─────────────────────────────────────────────────────────────────────────────

class IssueFingerprint(BaseModel):
    file_path:        str = ""
    rule_key:         str = ""
    line_start:       int = 0
    description_hash: str = ""


class Issue(BaseModel):
    id:                    str          = Field(default_factory=_new_id)
    run_id:                str          = ""
    severity:              Severity     = Severity.MINOR
    file_path:             str          = ""
    line_start:            int          = 0
    line_end:              int          = 0
    function_name:         str          = ""
    master_prompt_section: str          = ""
    description:           str          = ""
    status:                IssueStatus  = IssueStatus.OPEN
    executor_type:         ExecutorType | None = None
    domain_mode:           DomainMode   = DomainMode.GENERAL
    fix_attempts:          int          = 0
    max_fix_attempts:      int          = 3
    fix_requires_files:    list[str]    = Field(default_factory=list)
    confidence:            float        = Field(ge=0.0, le=1.0, default=0.85)
    fingerprint:           str          = ""
    # Compliance
    cwe_id:                str          = ""
    misra_rule:            str          = ""
    jsf_rule:              str          = ""
    cert_rule:             str          = ""
    compliance_violations: list[ComplianceViolation] = Field(default_factory=list)
    # MIL-STD-882E
    mil882e_category:      MilStd882eCategory = MilStd882eCategory.NONE
    hazard_description:    str          = ""
    # DO-178C traceability
    requirement_id:        str          = ""
    test_case_id:          str          = ""
    do178c_objective:      str          = ""
    deviation_record:      DeviationRecord | None = None
    # Consensus
    consensus_votes:       int          = 0
    consensus_confidence:  float        = 0.0
    # Timestamps
    detected_at:           datetime     = Field(default_factory=_utcnow)
    closed_at:             datetime | None = None
    last_updated:          datetime     = Field(default_factory=_utcnow)
    escalation_id:         str          = ""


# ─────────────────────────────────────────────────────────────────────────────
# Fix attempt
# ─────────────────────────────────────────────────────────────────────────────

class FixedFile(BaseModel):
    path:          str       = ""
    content:       str       = ""
    patch:         str       = ""
    patch_mode:    PatchMode = PatchMode.UNIFIED_DIFF
    changes_made:  str       = ""
    diff_summary:  str       = ""
    confidence:    float     = Field(ge=0.0, le=1.0, default=0.85)
    original_hash: str       = ""
    patched_hash:  str       = ""
    lines_changed: int       = 0


class FixAttempt(BaseModel):
    id:                     str              = Field(default_factory=_new_id)
    run_id:                 str              = ""
    issue_ids:              list[str]        = Field(default_factory=list)
    fixed_files:            list[FixedFile]  = Field(default_factory=list)
    fixer_model:            str              = ""
    fixer_model_family:     str              = ""
    reviewer_model:         str              = ""
    reviewer_model_family:  str              = ""
    independence_record_id: str              = ""
    reviewer_verdict:       ReviewVerdict | None = None
    reviewer_notes:         str              = ""
    gate_passed:            bool | None      = None
    gate_reason:            str              = ""
    planner_approved:       bool | None      = None
    planner_reason:         str              = ""
    formal_proofs:          list[str]        = Field(default_factory=list)
    cbmc_result_id:         str              = ""
    polyspace_finding_ids:  list[str]        = Field(default_factory=list)
    test_run_id:            str              = ""
    patch_mode:             PatchMode        = PatchMode.UNIFIED_DIFF
    requirement_id:         str              = ""
    test_case_id:           str              = ""
    committed_at:           datetime | None  = None
    pr_url:                 str              = ""
    created_at:             datetime         = Field(default_factory=_utcnow)
    updated_at:             datetime         = Field(default_factory=_utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Planner
# ─────────────────────────────────────────────────────────────────────────────

class PlannerRecord(BaseModel):
    id:                    str              = Field(default_factory=_new_id)
    fix_attempt_id:        str              = ""
    run_id:                str              = ""
    file_path:             str              = ""
    verdict:               PlannerVerdict   = PlannerVerdict.SAFE
    reversibility:         ReversibilityClass = ReversibilityClass.REVERSIBLE
    goal_coherent:         bool             = True
    risk_score:            float            = Field(ge=0.0, le=1.0, default=0.0)
    block_commit:          bool             = False
    reason:                str              = ""
    simulation_summary:    str              = ""
    formal_proof_available: bool            = False
    evaluated_at:          datetime         = Field(default_factory=_utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Audit score and run
# ─────────────────────────────────────────────────────────────────────────────

class AuditScore(BaseModel):
    id:             str   = Field(default_factory=_new_id)
    run_id:         str   = ""
    cycle_number:   int   = 0
    critical_count: int   = 0
    major_count:    int   = 0
    minor_count:    int   = 0
    info_count:     int   = 0
    score:          float = 100.0
    misra_open:     int   = 0
    cert_open:      int   = 0
    cwe_open:       int   = 0
    jsf_open:       int   = 0
    created_at:     datetime = Field(default_factory=_utcnow)

    def compute_score(self) -> None:
        c_pen = min(self.critical_count * 15, 60)
        m_pen = min(self.major_count    *  5, 30)
        n_pen = min(self.minor_count    *  1, 10)
        self.score = max(0.0, 100.0 - c_pen - m_pen - n_pen)


class ConsensusVote(BaseModel):
    agent:      ExecutorType = ExecutorType.GENERAL
    confirmed:  bool         = False
    confidence: float        = Field(ge=0.0, le=1.0, default=0.0)
    notes:      str          = ""


class ConsensusResult(BaseModel):
    issue_fingerprint:   str          = ""
    votes:               list[ConsensusVote] = Field(default_factory=list)
    final_confidence:    float        = Field(ge=0.0, le=1.0, default=0.0)
    approved:            bool         = False
    disagreement_action: DisagreementAction = DisagreementAction.FLAG_UNCERTAIN
    high_centrality:     bool         = False
    escalation_required: bool         = False


class ConsensusRule(BaseModel):
    minimum_agents:         int                = 1
    required_domains:       list[ExecutorType] = Field(default_factory=list)
    confidence_floor:       float              = Field(ge=0.0, le=1.0, default=0.5)
    disagreement_action:    DisagreementAction = DisagreementAction.FLAG_UNCERTAIN
    high_centrality_raises: bool               = False


class AuditRun(BaseModel):
    id:                       str               = Field(default_factory=_new_id)
    repo_url:                 str               = ""
    repo_name:                str               = ""
    branch:                   str               = "main"
    master_prompt_path:       str               = ""
    autonomy_level:           AutonomyLevel     = AutonomyLevel.AUTO_FIX
    domain_mode:              DomainMode        = DomainMode.GENERAL
    software_level:           SoftwareLevel     = SoftwareLevel.NONE
    tool_qualification_level: ToolQualificationLevel = ToolQualificationLevel.NONE
    max_cycles:               int               = 50
    cycle_count:              int               = 0
    status:                   RunStatus         = RunStatus.RUNNING
    scores:                   list[AuditScore]  = Field(default_factory=list)
    graph_built:              bool              = False
    baseline_id:              str               = ""
    active_escalations:       int               = 0
    total_escalations:        int               = 0
    started_at:               datetime          = Field(default_factory=_utcnow)
    finished_at:              datetime | None   = None


# ─────────────────────────────────────────────────────────────────────────────
# Audit trail (DO-178C SCM evidence)
# ─────────────────────────────────────────────────────────────────────────────

class AuditTrailEntry(BaseModel):
    id:                str           = Field(default_factory=_new_id)
    run_id:            str           = ""
    event_type:        str           = ""
    entity_id:         str           = ""
    entity_type:       str           = ""
    before_state:      str           = ""
    after_state:       str           = ""
    actor:             str           = "Rhodawk AI"
    artifact_id:       str           = ""
    artifact_type:     ArtifactType | None = None
    baseline_id:       str           = ""
    change_request_id: str           = ""
    model_name:        str           = ""
    model_version:     str           = ""
    hmac_signature:    str           = ""
    created_at:        datetime      = Field(default_factory=_utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Patrol events
# ─────────────────────────────────────────────────────────────────────────────

class PatrolEvent(BaseModel):
    id:         str      = Field(default_factory=_new_id)
    run_id:     str      = ""
    event_type: str      = ""
    detail:     str      = ""
    severity:   str      = "INFO"
    created_at: datetime = Field(default_factory=_utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

class TestRunResult(BaseModel):
    id:             str           = Field(default_factory=_new_id)
    run_id:         str           = ""
    fix_attempt_id: str           = ""
    status:         TestRunStatus = TestRunStatus.SKIPPED
    passed:         int           = 0
    failed:         int           = 0
    errors:         int           = 0
    skipped:        int           = 0
    coverage_pct:   float         = 0.0
    mcdc_coverage:  float         = 0.0
    output:         str           = ""
    duration_s:     float         = 0.0
    created_at:     datetime      = Field(default_factory=_utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Graph models
# ─────────────────────────────────────────────────────────────────────────────

class GraphNode(BaseModel):
    path:                str   = ""
    language:            str   = "unknown"
    is_load_bearing:     bool  = False
    centrality:          float = 0.0
    page_rank:           float = 0.0
    in_safety_partition: bool  = False


class GraphEdge(BaseModel):
    source:    str   = ""
    target:    str   = ""
    edge_type: str   = "import"
    symbol:    str   = ""
    weight:    float = 1.0


class DependencySnapshot(BaseModel):
    id:         str      = Field(default_factory=_new_id)
    run_id:     str      = ""
    node_count: int      = 0
    edge_count: int      = 0
    built_at:   datetime = Field(default_factory=_utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Convergence
# ─────────────────────────────────────────────────────────────────────────────

class ConvergenceRecord(BaseModel):
    run_id:      str     = ""
    cycle:       int     = 0
    score:       float   = 0.0
    converged:   bool    = False
    halt_reason: str     = ""
    recorded_at: datetime = Field(default_factory=_utcnow)
