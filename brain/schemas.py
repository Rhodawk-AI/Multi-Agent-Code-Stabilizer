from __future__ import annotations
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator

def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)

def _new_id() -> str:
    return str(uuid.uuid4())

class Severity(str, Enum):
    CRITICAL = 'CRITICAL'
    MAJOR = 'MAJOR'
    MINOR = 'MINOR'
    INFO = 'INFO'

class MilStd882eCategory(str, Enum):
    CAT_I = 'CAT_I'
    CAT_II = 'CAT_II'
    CAT_III = 'CAT_III'
    CAT_IV = 'CAT_IV'
    NONE = 'NONE'
SEVERITY_TO_MIL882E: dict[Severity, MilStd882eCategory] = {Severity.CRITICAL: MilStd882eCategory.CAT_I, Severity.MAJOR: MilStd882eCategory.CAT_II, Severity.MINOR: MilStd882eCategory.CAT_III, Severity.INFO: MilStd882eCategory.CAT_IV}

class IssueStatus(str, Enum):
    OPEN = 'OPEN'
    FIX_QUEUED = 'FIX_QUEUED'
    FIX_GENERATED = 'FIX_GENERATED'
    REVIEWING = 'REVIEWING'
    APPROVED = 'APPROVED'
    REJECTED = 'REJECTED'
    CLOSED = 'CLOSED'
    ESCALATED = 'ESCALATED'
    ESCALATION_PENDING = 'ESCALATION_PENDING'
    ESCALATION_APPROVED = 'ESCALATION_APPROVED'
    ESCALATION_REJECTED = 'ESCALATION_REJECTED'
    REGRESSED = 'REGRESSED'
    DEFERRED = 'DEFERRED'
    BASELINE_LOCKED = 'BASELINE_LOCKED'

class FileStatus(str, Enum):
    UNREAD = 'UNREAD'
    READING = 'READING'
    READ = 'READ'
    MODIFIED = 'MODIFIED'
    LOCKED = 'LOCKED'
    STALE = 'STALE'
    PARTIAL = 'PARTIAL'

class ReviewVerdict(str, Enum):
    APPROVED = 'APPROVED'
    REJECTED = 'REJECTED'
    ESCALATE = 'ESCALATE'
    APPROVED_WARNING = 'APPROVED_WARNING'

class ExecutorType(str, Enum):
    SECURITY = 'SECURITY'
    ARCHITECTURE = 'ARCHITECTURE'
    STANDARDS = 'STANDARDS'
    GENERAL = 'GENERAL'
    READER = 'READER'
    FIXER = 'FIXER'
    REVIEWER = 'REVIEWER'
    PATROL = 'PATROL'
    PLANNER = 'PLANNER'
    FORMAL = 'FORMAL'
    DOMAIN = 'DOMAIN'
    TEST_RUNNER = 'TEST_RUNNER'
    CBMC = 'CBMC'
    POLYSPACE = 'POLYSPACE'
    LDRA = 'LDRA'
    MISRA = 'MISRA'
    CERT = 'CERT'
    JSF = 'JSF'
    SYNTHESIS = 'SYNTHESIS'

class ChunkStrategy(str, Enum):
    FULL = 'FULL'
    HALF = 'HALF'
    AST_NODES = 'AST_NODES'
    SKELETON = 'SKELETON'
    SKELETON_ONLY = 'SKELETON_ONLY'
    FUNCTION = 'FUNCTION'
    PREPROCESSED = 'PREPROCESSED'

class RunStatus(str, Enum):
    RUNNING = 'RUNNING'
    STABILIZED = 'STABILIZED'
    HALTED = 'HALTED'
    ESCALATED = 'ESCALATED'
    FAILED = 'FAILED'
    BASELINE_PENDING = 'BASELINE_PENDING'
    BASELINE_APPROVED = 'BASELINE_APPROVED'

class AutonomyLevel(str, Enum):
    READ_ONLY = 'READ_ONLY'
    SUGGEST = 'SUGGEST'
    AUTO_FIX = 'AUTO_FIX'
    AUTO_FIX_PR = 'AUTO_FIX_PR'
    FULL_AUTONOMOUS = 'FULL_AUTONOMOUS'

class DomainMode(str, Enum):
    GENERAL = 'GENERAL'
    FINANCE = 'FINANCE'
    MEDICAL = 'MEDICAL'
    MILITARY = 'MILITARY'
    EMBEDDED = 'EMBEDDED'
    AEROSPACE = 'AEROSPACE'
    NUCLEAR = 'NUCLEAR'

class ReversibilityClass(str, Enum):
    REVERSIBLE = 'REVERSIBLE'
    CONDITIONAL = 'CONDITIONAL'
    IRREVERSIBLE = 'IRREVERSIBLE'

class PlannerVerdict(str, Enum):
    SAFE = 'SAFE'
    SAFE_WITH_WARNING = 'SAFE_WITH_WARNING'
    UNSAFE = 'UNSAFE'
    NEEDS_HUMAN = 'NEEDS_HUMAN'

class DisagreementAction(str, Enum):
    FLAG_UNCERTAIN = 'FLAG_UNCERTAIN'
    ESCALATE_HUMAN = 'ESCALATE_HUMAN'
    BLOCK = 'BLOCK'
    AUTO_RESOLVE = 'AUTO_RESOLVE'

class FormalVerificationStatus(str, Enum):
    PROVED = 'PROVED'
    REFUTED = 'REFUTED'
    UNKNOWN = 'UNKNOWN'
    COUNTEREXAMPLE = 'COUNTEREXAMPLE'
    TIMEOUT = 'TIMEOUT'
    ERROR = 'ERROR'
    SKIPPED = 'SKIPPED'

class TestRunStatus(str, Enum):
    PASSED = 'PASSED'
    FAILED = 'FAILED'
    ERROR = 'ERROR'
    SKIPPED = 'SKIPPED'
    PARTIAL = 'PARTIAL'
    NO_TESTS = 'NO_TESTS'

class ComplianceStandard(str, Enum):
    DO_178C = 'DO-178C'
    MISRA_C = 'MISRA-C:2023'
    MISRA_CPP = 'MISRA-C++:2023'
    CERT_C = 'CERT-C'
    CERT_CPP = 'CERT-C++'
    MIL_882E = 'MIL-STD-882E'
    JSF_AV = 'JSF-AV-RULES'
    CWE = 'CWE'
    OWASP = 'OWASP'
    IEC_61513 = 'IEC-61513'
    IEC_62304 = 'IEC-62304'
    AUTOSAR = 'AUTOSAR-C++14'

class PatchMode(str, Enum):
    FULL_FILE = 'FULL_FILE'
    UNIFIED_DIFF = 'UNIFIED_DIFF'
    AST_REWRITE = 'AST_REWRITE'

class ArtifactType(str, Enum):
    REQUIREMENT = 'REQUIREMENT'
    DESIGN = 'DESIGN'
    CODE = 'CODE'
    TEST_CASE = 'TEST_CASE'
    TEST_RESULT = 'TEST_RESULT'
    COVERAGE = 'COVERAGE'
    EVIDENCE = 'EVIDENCE'
    FIX = 'FIX'
    AUDIT_ENTRY = 'AUDIT_ENTRY'
    BASELINE = 'BASELINE'
    SAS = 'SAS'

class ToolQualificationLevel(str, Enum):
    TQL_1 = 'TQL-1'
    TQL_2 = 'TQL-2'
    TQL_3 = 'TQL-3'
    TQL_4 = 'TQL-4'
    TQL_5 = 'TQL-5'
    NONE = 'NONE'

class SoftwareLevel(str, Enum):
    DAL_A = 'DAL-A'
    DAL_B = 'DAL-B'
    DAL_C = 'DAL-C'
    DAL_D = 'DAL-D'
    DAL_E = 'DAL-E'
    NONE = 'NONE'

class PolyspaceVerdict(str, Enum):
    GREEN = 'GREEN'
    ORANGE = 'ORANGE'
    RED = 'RED'
    GRAY = 'GRAY'

class DeviationRecord(BaseModel):
    deviation_id: str = Field(default_factory=_new_id)
    rule_id: str = ''
    standard: ComplianceStandard = ComplianceStandard.MISRA_C
    justification: str = ''
    approved_by: str = ''
    approved_at: datetime | None = None
    expiry_date: datetime | None = None
    risk_acceptance: str = ''

class ComplianceViolation(BaseModel):
    rule_id: str = ''
    standard: ComplianceStandard = ComplianceStandard.MISRA_C
    rule_description: str = ''
    is_mandatory: bool = False
    deviation_record: DeviationRecord | None = None
    tool_detected_by: str = ''
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)

class RequirementTraceability(BaseModel):
    id: str = Field(default_factory=_new_id)
    requirement_id: str = ''
    requirement_text: str = ''
    test_case_id: str = ''
    test_result_id: str = ''
    coverage_pct: float = 0.0
    mcdc_coverage: float = 0.0
    issue_id: str = ''
    fix_attempt_id: str = ''
    verified_by: str = ''
    verified_at: datetime | None = None
    do178c_objective: str = ''

class ReviewerIndependenceRecord(BaseModel):
    id: str = Field(default_factory=_new_id)
    fix_attempt_id: str = ''
    fixer_model: str = ''
    fixer_model_family: str = ''
    reviewer_model: str = ''
    reviewer_model_family: str = ''
    independence_verified: bool = False
    violation_reason: str = ''
    created_at: datetime = Field(default_factory=_utcnow)

    @model_validator(mode='after')
    def _check_independence(self) -> 'ReviewerIndependenceRecord':
        same = self.fixer_model_family.lower() == self.reviewer_model_family.lower()
        self.independence_verified = not same
        if same and (not self.violation_reason):
            self.violation_reason = f'DO-178C 6.3.4 violation: fixer ({self.fixer_model_family}) and reviewer ({self.reviewer_model_family}) share model family.'
        return self

class FormalVerificationResult(BaseModel):
    id: str = Field(default_factory=_new_id)
    fix_attempt_id: str = ''
    file_path: str = ''
    property_name: str = ''
    status: FormalVerificationStatus = FormalVerificationStatus.UNKNOWN
    counterexample: str = ''
    proof_script: str = ''
    solver: str = 'z3'
    elapsed_s: float = 0.0
    verified_at: datetime = Field(default_factory=_utcnow)
    evidence_path: str = ''

class CbmcVerificationResult(BaseModel):
    id: str = Field(default_factory=_new_id)
    fix_attempt_id: str = ''
    file_path: str = ''
    function_name: str = ''
    unwind_bound: int = 10
    properties_checked: list[str] = Field(default_factory=list)
    property_results: dict[str, str] = Field(default_factory=dict)
    counterexample: str = ''
    stdout: str = ''
    return_code: int = 0
    elapsed_s: float = 0.0
    verified_at: datetime = Field(default_factory=_utcnow)

class PolyspaceFinding(BaseModel):
    id: str = Field(default_factory=_new_id)
    fix_attempt_id: str = ''
    file_path: str = ''
    line_number: int = 0
    check_name: str = ''
    verdict: PolyspaceVerdict = PolyspaceVerdict.ORANGE
    category: str = ''
    detail: str = ''
    run_id: str = ''

class LdraFinding(BaseModel):
    id: str = Field(default_factory=_new_id)
    file_path: str = ''
    line_number: int = 0
    rule_id: str = ''
    standard: ComplianceStandard = ComplianceStandard.MISRA_C
    severity: str = ''
    message: str = ''
    is_suppressed: bool = False
    suppression_ref: str = ''
    run_id: str = ''

class FunctionStalenessMark(BaseModel):
    id: str = Field(default_factory=_new_id)
    file_path: str = ''
    function_name: str = ''
    line_start: int = 0
    line_end: int = 0
    stale_reason: str = ''
    stale_since: datetime = Field(default_factory=_utcnow)
    run_id: str = ''

class EscalationStatus(str, Enum):
    PENDING = 'PENDING'
    APPROVED = 'APPROVED'
    REJECTED = 'REJECTED'
    TIMEOUT = 'TIMEOUT'
    AUTO_RESOLVED = 'AUTO_RESOLVED'

class EscalationRecord(BaseModel):
    id: str = Field(default_factory=_new_id)
    run_id: str = ''
    issue_ids: list[str] = Field(default_factory=list)
    fix_attempt_id: str = ''
    escalation_type: str = ''
    description: str = ''
    severity: Severity = Severity.CRITICAL
    mil882e_category: MilStd882eCategory = MilStd882eCategory.CAT_I
    status: EscalationStatus = EscalationStatus.PENDING
    approved_by: str = ''
    approved_at: datetime | None = None
    approval_rationale: str = ''
    risk_acceptance: str = ''
    notified_via: list[str] = Field(default_factory=list)
    notified_at: datetime | None = None
    timeout_at: datetime | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

class BaselineRecord(BaseModel):
    id: str = Field(default_factory=_new_id)
    run_id: str = ''
    baseline_name: str = ''
    software_level: SoftwareLevel = SoftwareLevel.NONE
    commit_hash: str = ''
    issue_count: dict[str, int] = Field(default_factory=dict)
    score_snapshot: float = 0.0
    file_hashes: dict[str, str] = Field(default_factory=dict)
    approved_by: str = ''
    approved_at: datetime | None = None
    approval_token: str = ''
    change_request_id: str = ''
    rtm_export_path: str = ''
    sas_export_path: str = ''
    created_at: datetime = Field(default_factory=_utcnow)
    is_active: bool = False

class SoftwareConfigurationIndex(BaseModel):
    id: str = Field(default_factory=_new_id)
    baseline_id: str = ''
    run_id: str = ''
    controlled_files: list[dict[str, str]] = Field(default_factory=list)
    tool_versions: dict[str, str] = Field(default_factory=dict)
    compiler_config: dict[str, str] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=_utcnow)
    generated_by: str = 'Rhodawk AI'

class SoftwareAccomplishmentSummary(BaseModel):
    id: str = Field(default_factory=_new_id)
    baseline_id: str = ''
    run_id: str = ''
    software_level: SoftwareLevel = SoftwareLevel.NONE
    tool_qualification_level: ToolQualificationLevel = ToolQualificationLevel.NONE
    total_cycles: int = 0
    total_issues_found: int = 0
    total_issues_closed: int = 0
    total_escalations: int = 0
    total_deviations: int = 0
    tools_used: list[dict[str, str]] = Field(default_factory=list)
    misra_violations_open: int = 0
    misra_violations_closed: int = 0
    cert_violations_open: int = 0
    cert_violations_closed: int = 0
    cwe_findings_open: int = 0
    cwe_findings_closed: int = 0
    do178c_objectives_met: list[str] = Field(default_factory=list)
    do178c_objectives_open: list[str] = Field(default_factory=list)
    prepared_by: str = 'Rhodawk AI'
    reviewed_by: str = ''
    approved_by: str = ''
    approved_at: datetime | None = None
    generated_at: datetime = Field(default_factory=_utcnow)
    psac_deviations: list[str] = Field(default_factory=list)
    certification_basis: str = ''

class FileRecord(BaseModel):
    path: str = ''
    language: str = 'unknown'
    status: FileStatus = FileStatus.UNREAD
    hash: str = ''
    line_count: int = 0
    chunk_count: int = 0
    summary: str = ''
    key_symbols: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    all_observations: list[str] = Field(default_factory=list)
    is_load_bearing: bool = False
    run_id: str = ''
    last_read_at: datetime | None = None
    known_functions: list[str] = Field(default_factory=list)
    stale_functions: list[str] = Field(default_factory=list)
    in_safety_partition: bool = False
    software_level: SoftwareLevel = SoftwareLevel.NONE

class FileChunkRecord(BaseModel):
    id: str = Field(default_factory=_new_id)
    file_path: str = ''
    run_id: str = ''
    chunk_index: int = 0
    total_chunks: int = 1
    line_start: int = 0
    line_end: int = 0
    language: str = 'unknown'
    strategy: ChunkStrategy = ChunkStrategy.FULL
    # The actual source text for this chunk.  Populated by chunk_file() and
    # persisted to the file_chunks table so get_all_observations() and
    # get_stale_observations() can return auditable dicts without re-reading
    # every file from disk on every audit cycle.
    content: str = ''
    summary: str = ''
    symbols_defined: list[str] = Field(default_factory=list)
    symbols_referenced: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    raw_observations: list[str] = Field(default_factory=list)
    vector_id: str = ''
    function_name: str = ''
    all_functions: list[str] = Field(default_factory=list)
    preprocessed: bool = False
    created_at: datetime = Field(default_factory=_utcnow)

class IssueFingerprint(BaseModel):
    file_path: str = ''
    rule_key: str = ''
    line_start: int = 0
    description_hash: str = ''

class Issue(BaseModel):
    id: str = Field(default_factory=_new_id)
    run_id: str = ''
    severity: Severity = Severity.MINOR
    file_path: str = ''
    line_start: int = 0
    line_end: int = 0
    function_name: str = ''
    master_prompt_section: str = ''
    description: str = ''
    status: IssueStatus = IssueStatus.OPEN
    executor_type: ExecutorType | None = None
    domain_mode: DomainMode = DomainMode.GENERAL
    fix_attempts: int = 0
    max_fix_attempts: int = 3
    fix_requires_files: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.85)
    fingerprint: str = ''
    cwe_id: str = ''
    misra_rule: str = ''
    jsf_rule: str = ''
    cert_rule: str = ''
    compliance_violations: list[ComplianceViolation] = Field(default_factory=list)
    mil882e_category: MilStd882eCategory = MilStd882eCategory.NONE
    hazard_description: str = ''
    requirement_id: str = ''
    test_case_id: str = ''
    do178c_objective: str = ''
    deviation_record: DeviationRecord | None = None
    consensus_votes: int = 0
    consensus_confidence: float = 0.0
    is_mandatory: bool = False
    compound_finding_id: str = ''
    detected_at: datetime = Field(default_factory=_utcnow)
    closed_at: datetime | None = None
    last_updated: datetime = Field(default_factory=_utcnow)
    escalation_id: str = ''
    # BoBN / SWE-bench integration fields.
    # fail_tests: test IDs that must be fixed by the patch (FAIL_TO_PASS signal).
    # pass_tests: test IDs that must remain green after the patch (PASS_TO_PASS signal).
    # base_commit: the git commit SHA the issue was detected against; passed to
    #              ExecutionFeedbackLoop so Docker containers start from the right
    #              snapshot.  These were previously hardcoded to [] / None / "" in
    #              _phase_fix_gap5, which zeroed every BoBN test_score.
    fail_tests:   list[str]       = Field(default_factory=list)
    pass_tests:   list[str]       = Field(default_factory=list)
    base_commit:  str             = ''

class FixedFile(BaseModel):
    path: str = ''
    content: str = ''
    patch: str = ''
    patch_mode: PatchMode = PatchMode.UNIFIED_DIFF
    changes_made: str = ''
    diff_summary: str = ''
    confidence: float = Field(ge=0.0, le=1.0, default=0.85)
    original_hash: str = ''
    patched_hash: str = ''
    lines_changed: int = 0

class FixAttempt(BaseModel):
    id: str = Field(default_factory=_new_id)
    run_id: str = ''
    issue_ids: list[str] = Field(default_factory=list)
    fixed_files: list[FixedFile] = Field(default_factory=list)
    fixer_model: str = ''
    fixer_model_family: str = ''
    reviewer_model: str = ''
    reviewer_model_family: str = ''
    independence_record_id: str = ''
    reviewer_verdict: ReviewVerdict | None = None
    reviewer_notes: str = ''
    gate_passed: bool | None = None
    gate_reason: str = ''
    planner_approved: bool | None = None
    planner_reason: str = ''
    formal_proofs: list[str] = Field(default_factory=list)
    cbmc_result_id: str = ''
    polyspace_finding_ids: list[str] = Field(default_factory=list)
    test_run_id: str = ''
    patch_mode: PatchMode = PatchMode.UNIFIED_DIFF
    requirement_id: str = ''
    test_case_id: str = ''
    committed_at: datetime | None = None
    pr_url: str = ''
    blast_radius_exceeded: bool = False
    refactor_proposal_id: str = ''
    # Gap 5: free-form metadata from the BoBN adversarial ensemble
    # Stores JSON-serialised BoBN scoring fields (candidate_id, composite_score,
    # test_score, n_candidates, attack_confidence) so the audit trail and API
    # can surface provenance for every adversarially-selected patch.
    extra_notes: str = ''
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

class BugRecurrenceSignal(BaseModel):
    """Captures the result of a historical-recurrence check performed before
    the fixer generates a patch.

    When the same bug class (matched by semantic similarity against fix_memory)
    has been patched N or more times within the recurrence window, this is
    strong evidence that the root cause is structural — a patch will not
    eliminate the recurrence.  The fixer uses this signal to route to a
    refactor proposal instead of generating another patch.

    Fields
    ------
    recurrence_count:
        Number of prior successful fixes (non-reverted) for the same bug class
        found in fix_memory within the time window.
    reverted_count:
        Number of prior fix attempts for the same bug class that were reverted
        (regression detected).  Combined with recurrence_count this measures
        fix instability.
    window_days:
        The time window in days over which recurrences were counted (default 180).
    dominant_file_context:
        The most common file context seen in the historical entries, i.e. the
        module most frequently implicated in this bug class.
    coupling_score:
        CPG-derived architectural coupling score for the implicated functions
        (0.0 = well-encapsulated, 1.0 = maximally coupled).  Set to -1.0 when
        CPG is unavailable.
    distinct_caller_modules:
        Number of distinct caller modules identified by the CPG coupling query.
        High values (> coupling_module_threshold) are an independent smell signal.
    coupling_module_threshold:
        The threshold above which distinct_caller_modules triggers a coupling
        smell escalation regardless of recurrence_count.
    is_structural:
        True when recurrence_count >= recurrence_threshold OR
        coupling_score >= coupling_score_threshold OR
        distinct_caller_modules >= coupling_module_threshold.
    escalation_reason:
        Human-readable summary of why is_structural was set.
    """
    recurrence_count:           int   = 0
    reverted_count:             int   = 0
    window_days:                int   = 180
    dominant_file_context:      str   = ''
    coupling_score:             float = -1.0
    distinct_caller_modules:    int   = 0
    coupling_module_threshold:  int   = 5
    is_structural:              bool  = False
    escalation_reason:          str   = ''


class RefactorProposal(BaseModel):
    id: str = Field(default_factory=_new_id)
    fix_attempt_id: str = ''
    run_id: str = ''
    issue_ids: list[str] = Field(default_factory=list)
    changed_functions: list[str] = Field(default_factory=list)
    affected_function_count: int = 0
    affected_file_count: int = 0
    test_files_affected: list[str] = Field(default_factory=list)
    blast_radius_score: float = 0.0
    # Import-only blast radius: files that import a changed symbol without
    # calling any changed function.  Broken by type/signature changes even
    # though the call graph does not reach them.  Tracked separately so the
    # human reviewer knows the full surface area, not just call-site count.
    importing_modules: list[str] = Field(default_factory=list)
    importing_module_count: int = 0
    affected_components: list[str] = Field(default_factory=list)
    proposed_refactoring: str = ''
    migration_steps: list[str] = Field(default_factory=list)
    estimated_scope: str = ''
    risks: list[str] = Field(default_factory=list)
    recommendation: str = ''
    escalation_id: str = ''
    requires_human_review: bool = True
    # ── Recurrence / coupling smell fields (Gap 3 proactive arch detection) ──
    # Populated when the refactor proposal was triggered by recurrence or CPG
    # coupling analysis rather than (or in addition to) blast radius overflow.
    recurrence_count: int = 0
    reverted_count: int = 0
    distinct_caller_modules: int = 0
    coupling_score: float = -1.0
    recurrence_escalation_reason: str = ''
    # Trigger source: 'blast_radius' | 'recurrence' | 'coupling_smell' |
    # 'architectural_symptom' | 'combined'
    trigger_source: str = 'blast_radius'
    created_at: datetime = Field(default_factory=_utcnow)

class PlannerRecord(BaseModel):
    id: str = Field(default_factory=_new_id)
    fix_attempt_id: str = ''
    run_id: str = ''
    file_path: str = ''
    verdict: PlannerVerdict = PlannerVerdict.SAFE
    reversibility: ReversibilityClass = ReversibilityClass.REVERSIBLE
    goal_coherent: bool = True
    risk_score: float = Field(ge=0.0, le=1.0, default=0.0)
    block_commit: bool = False
    reason: str = ''
    simulation_summary: str = ''
    formal_proof_available: bool = False
    evaluated_at: datetime = Field(default_factory=_utcnow)

class CompoundFindingCategory(str, Enum):
    SECURITY_ARCHITECTURE = 'SECURITY_ARCHITECTURE'
    SECURITY_STANDARDS = 'SECURITY_STANDARDS'
    ARCHITECTURE_STANDARDS = 'ARCHITECTURE_STANDARDS'
    ALL_DOMAINS = 'ALL_DOMAINS'
    SECURITY_ONLY = 'SECURITY_ONLY'
    ARCHITECTURE_ONLY = 'ARCHITECTURE_ONLY'
    STANDARDS_ONLY = 'STANDARDS_ONLY'

class CompoundFinding(BaseModel):
    id: str = Field(default_factory=_new_id)
    run_id: str = ''
    title: str = ''
    description: str = ''
    severity: Severity = Severity.CRITICAL
    category: CompoundFindingCategory = CompoundFindingCategory.SECURITY_ARCHITECTURE
    contributing_issue_ids: list[str] = Field(default_factory=list)
    synthesized_issue_id: str = ''
    domains_involved: list[str] = Field(default_factory=list)
    amplification_factor: float = Field(ge=1.0, le=10.0, default=2.0)
    fix_complexity: str = 'HIGH'
    rationale: str = ''
    cwe_id: str = ''
    misra_rule: str = ''
    cert_rule: str = ''
    mil882e_category: MilStd882eCategory = MilStd882eCategory.CAT_I
    is_mandatory: bool = True
    created_at: datetime = Field(default_factory=_utcnow)

class SynthesisReport(BaseModel):
    id: str = Field(default_factory=_new_id)
    run_id: str = ''
    cycle: int = 0
    raw_issue_count: int = 0
    fingerprint_dedup_count: int = 0
    semantic_dedup_count: int = 0
    final_issue_count: int = 0
    compound_finding_count: int = 0
    compound_critical_count: int = 0
    synthesis_model: str = ''
    dedup_enabled: bool = True
    compound_enabled: bool = True
    synthesis_summary: str = ''
    duration_s: float = 0.0
    created_at: datetime = Field(default_factory=_utcnow)
    # ARCH-03 FIX: track how many issues have no fail_tests so operators know
    # when BoBN composite scoring is operating at reduced signal (40% instead of
    # 100%). Issues sourced from static analysis never have fail_tests populated,
    # so the test_score=0 collapse is the dominant case in production usage.
    # These fields are exposed in the API response and included in benchmark
    # score output so published results are qualified appropriately.
    issues_without_fail_tests: int = 0    # count of deduped issues with no fail_tests
    bobn_inactive_pct: float = 0.0        # percentage of issues where BoBN test signal is absent
    heuristic_tests_matched: int = 0      # issues that received fail_tests via heuristic matching

class AuditScore(BaseModel):
    id: str = Field(default_factory=_new_id)
    run_id: str = ''
    cycle_number: int = 0
    critical_count: int = 0
    major_count: int = 0
    minor_count: int = 0
    info_count: int = 0
    score: float = 100.0
    misra_open: int = 0
    cert_open: int = 0
    cwe_open: int = 0
    jsf_open: int = 0
    created_at: datetime = Field(default_factory=_utcnow)

    def compute_score(self) -> None:
        c_pen = min(self.critical_count * 15, 60)
        m_pen = min(self.major_count * 5, 30)
        n_pen = min(self.minor_count * 1, 10)
        self.score = max(0.0, 100.0 - c_pen - m_pen - n_pen)

class ConsensusVote(BaseModel):
    agent: ExecutorType = ExecutorType.GENERAL
    confirmed: bool = False
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    notes: str = ''

class ConsensusResult(BaseModel):
    issue_fingerprint: str = ''
    votes: list[ConsensusVote] = Field(default_factory=list)
    final_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    approved: bool = False
    disagreement_action: DisagreementAction = DisagreementAction.FLAG_UNCERTAIN
    high_centrality: bool = False
    escalation_required: bool = False

class ConsensusRule(BaseModel):
    minimum_agents: int = 1
    required_domains: list[ExecutorType] = Field(default_factory=list)
    confidence_floor: float = Field(ge=0.0, le=1.0, default=0.5)
    disagreement_action: DisagreementAction = DisagreementAction.FLAG_UNCERTAIN
    high_centrality_raises: bool = False

class AuditRun(BaseModel):
    id: str = Field(default_factory=_new_id)
    repo_url: str = ''
    repo_name: str = ''
    branch: str = 'main'
    master_prompt_path: str = ''
    autonomy_level: AutonomyLevel = AutonomyLevel.AUTO_FIX
    domain_mode: DomainMode = DomainMode.GENERAL
    software_level: SoftwareLevel = SoftwareLevel.NONE
    tool_qualification_level: ToolQualificationLevel = ToolQualificationLevel.NONE
    max_cycles: int = 50
    cycle_count: int = 0
    status: RunStatus = RunStatus.RUNNING
    scores: list[AuditScore] = Field(default_factory=list)
    graph_built: bool = False
    baseline_id: str = ''
    active_escalations: int = 0
    total_escalations: int = 0
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None

class AuditTrailEntry(BaseModel):
    id: str = Field(default_factory=_new_id)
    run_id: str = ''
    event_type: str = ''
    entity_id: str = ''
    entity_type: str = ''
    before_state: str = ''
    after_state: str = ''
    actor: str = 'Rhodawk AI'
    artifact_id: str = ''
    artifact_type: ArtifactType | None = None
    baseline_id: str = ''
    change_request_id: str = ''
    model_name: str = ''
    model_version: str = ''
    hmac_signature: str = ''
    created_at: datetime = Field(default_factory=_utcnow)

class PatrolEvent(BaseModel):
    id: str = Field(default_factory=_new_id)
    run_id: str = ''
    event_type: str = ''
    detail: str = ''
    severity: str = 'INFO'
    created_at: datetime = Field(default_factory=_utcnow)

class TestRunResult(BaseModel):
    id: str = Field(default_factory=_new_id)
    run_id: str = ''
    fix_attempt_id: str = ''
    status: TestRunStatus = TestRunStatus.SKIPPED
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    coverage_pct: float = 0.0
    mcdc_coverage: float = 0.0
    output: str = ''
    duration_s: float = 0.0
    created_at: datetime = Field(default_factory=_utcnow)

class GraphNode(BaseModel):
    path: str = ''
    language: str = 'unknown'
    is_load_bearing: bool = False
    centrality: float = 0.0
    page_rank: float = 0.0
    in_safety_partition: bool = False

class GraphEdge(BaseModel):
    source: str = ''
    target: str = ''
    edge_type: str = 'import'
    symbol: str = ''
    weight: float = 1.0

class DependencySnapshot(BaseModel):
    id: str = Field(default_factory=_new_id)
    run_id: str = ''
    node_count: int = 0
    edge_count: int = 0
    built_at: datetime = Field(default_factory=_utcnow)

class ConvergenceRecord(BaseModel):
    run_id: str = ''
    cycle: int = 0
    score: float = 0.0
    converged: bool = False
    halt_reason: str = ''


# ---------------------------------------------------------------------------
# Review / LLM session types
# ---------------------------------------------------------------------------

class ReviewDecision(BaseModel):
    """Per-issue verdict produced by ReviewerAgent."""
    issue_id:       str           = ''
    fix_path:       str           = ''
    verdict:        ReviewVerdict = ReviewVerdict.REJECTED
    confidence:     float         = Field(ge=0.0, le=1.0, default=0.5)
    reason:         str           = ''
    concerns:       list[str]     = Field(default_factory=list)
    cross_file_ok:  bool          = True


class ReviewResult(BaseModel):
    """Aggregated review outcome for a single FixAttempt."""
    review_id:         str                   = Field(default_factory=_new_id)
    fix_attempt_id:    str                   = ''
    decisions:         list[ReviewDecision]  = Field(default_factory=list)
    overall_score:     float                 = 0.0
    overall_note:      str                   = ''
    approve_for_commit: bool                 = False
    reviewed_at:       datetime              = Field(default_factory=_utcnow)

    def compute_approval(self) -> None:
        """
        Approve for commit iff every decision is APPROVED or APPROVED_WARNING
        and at least one decision exists.  Sets overall_score to the mean
        confidence of all decisions.
        """
        if not self.decisions:
            self.approve_for_commit = False
            self.overall_score      = 0.0
            return
        approved_verdicts = {ReviewVerdict.APPROVED, ReviewVerdict.APPROVED_WARNING}
        all_approved       = all(d.verdict in approved_verdicts for d in self.decisions)
        self.approve_for_commit = all_approved
        self.overall_score      = sum(d.confidence for d in self.decisions) / len(self.decisions)


class LLMSession(BaseModel):
    """Record of a single LLM API call for cost tracking and audit trail."""
    id:                str           = Field(default_factory=_new_id)
    run_id:            str           = ''
    agent_type:        ExecutorType  = ExecutorType.GENERAL
    model:             str           = ''
    prompt_tokens:     int           = 0
    completion_tokens: int           = 0
    cost_usd:          float         = 0.0
    duration_ms:       int           = 0
    success:           bool          = True
    error:             str           = ''
    started_at:        datetime      = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Gap 4: Commit-granularity incremental audit tracking
# ---------------------------------------------------------------------------

class CommitAuditStatus(str, Enum):
    PENDING   = 'PENDING'    # diff parsed, impact set queued
    RUNNING   = 'RUNNING'    # auditing impact set functions
    DONE      = 'DONE'       # all impact-set functions re-audited
    FAILED    = 'FAILED'     # scheduler encountered a fatal error
    SKIPPED   = 'SKIPPED'    # no changed functions found in diff


class CommitAuditRecord(BaseModel):
    """
    Persists the state of a single commit-triggered incremental audit.

    One record is created per commit that reaches the system (via webhook,
    CI push, or post-fix commit).  The scheduler uses this record to resume
    interrupted audits and to report compute savings (functions_to_audit vs
    full codebase line count).
    """
    id: str = Field(default_factory=_new_id)
    run_id: str = ''

    # Git provenance
    commit_hash: str = ''
    branch: str = ''
    author: str = ''
    commit_message: str = ''

    # Diff statistics (function-granularity, not file-granularity)
    changed_files: list[str] = Field(default_factory=list)
    changed_functions: dict[str, list[str]] = Field(default_factory=dict)
    # flat list derived from changed_functions for easy querying
    all_changed_functions: list[str] = Field(default_factory=list)
    new_functions: list[str] = Field(default_factory=list)
    deleted_functions: list[str] = Field(default_factory=list)

    # CPG impact set (transitive dependents to depth 3)
    impact_functions: list[str] = Field(default_factory=list)
    impact_files: list[str] = Field(default_factory=list)

    # Audit scope — the minimal set fed to re-audit
    audit_targets: list[dict] = Field(default_factory=list)

    # Counts for compute-savings reporting
    total_changed_functions: int = 0
    total_impact_functions: int = 0
    total_functions_to_audit: int = 0

    # Test re-run scope
    test_files_to_run: list[str] = Field(default_factory=list)
    test_functions_to_run: list[str] = Field(default_factory=list)

    # Execution state
    status: CommitAuditStatus = CommitAuditStatus.PENDING
    cpg_updated: bool = False
    joern_update_status: str = ''
    error_detail: str = ''

    # Timing
    created_at: datetime = Field(default_factory=_utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------
# Tests and external callers import `Escalation` rather than `EscalationRecord`.
# The alias keeps those imports working without duplicating the class.
Escalation = EscalationRecord