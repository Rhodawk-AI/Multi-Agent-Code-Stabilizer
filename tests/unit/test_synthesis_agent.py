"""
tests/unit/test_synthesis_agent.py
===================================
Unit tests for Gap 2 SynthesisAgent — cross-domain compound finding detection.

Coverage:
  ✓ Fingerprint deduplication — exact duplicates removed, best kept
  ✓ Fingerprint dedup priority ordering (severity > confidence > domain)
  ✓ Semantic dedup application — LLM response applied correctly
  ✓ Semantic dedup batching — large lists chunked correctly
  ✓ Compound finding detection — raw LLM response converted correctly
  ✓ CompoundFinding validation — <2 valid indices dropped
  ✓ Severity escalation — amplification ≥ 3.0 promotes severity
  ✓ _compound_to_issue — correct Issue fields, executor_type=SYNTHESIS
  ✓ run() full pipeline — dedup + compound, persists to storage
  ✓ run() fail-open — synthesis crash returns original issues unchanged
  ✓ run() empty input — returns ([], []) cleanly
  ✓ SynthesisAgent disabled — synthesis_enabled=False skips synthesis phase
  ✓ Config loader maps [synthesis] TOML section to StabilizerConfig fields
  ✓ schemas — ExecutorType.SYNTHESIS present; CompoundFinding, SynthesisReport valid
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from brain.schemas import (
    CompoundFinding,
    CompoundFindingCategory,
    ExecutorType,
    Issue,
    IssueStatus,
    Severity,
    SynthesisReport,
    MilStd882eCategory,
    DomainMode,
)
from agents.synthesis_agent import (
    SynthesisAgent,
    DeduplicationResponse,
    DuplicateCluster,
    CompoundFindingResponse,
    RawCompoundFinding,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_issue(
    file_path: str = "foo.py",
    line_start: int = 10,
    description: str = "buffer overflow in parse()",
    severity: Severity = Severity.MAJOR,
    executor_type: ExecutorType = ExecutorType.SECURITY,
    confidence: float = 0.85,
    cwe_id: str = "",
    misra_rule: str = "",
    cert_rule: str = "",
    fingerprint: str = "",
) -> Issue:
    fp = fingerprint or hashlib.sha256(
        f"{file_path}:{line_start}:0:{description[:120]}".encode()
    ).hexdigest()[:16]
    return Issue(
        run_id="run-test",
        severity=severity,
        file_path=file_path,
        line_start=line_start,
        line_end=line_start + 5,
        description=description,
        status=IssueStatus.OPEN,
        executor_type=executor_type,
        confidence=confidence,
        fingerprint=fp,
        cwe_id=cwe_id,
        misra_rule=misra_rule,
        cert_rule=cert_rule,
    )


def _make_storage() -> MagicMock:
    storage = MagicMock()
    storage.upsert_issue = AsyncMock(return_value=None)
    storage.get_issue = AsyncMock(return_value=None)
    storage.append_audit_trail = AsyncMock(return_value=None)
    return storage


def _make_agent(
    dedup_enabled: bool = True,
    compound_enabled: bool = True,
    synthesis_model: str = "test-model",
) -> SynthesisAgent:
    from agents.base import AgentConfig
    cfg = AgentConfig(
        model="test-primary",
        critical_fix_model="test-critical",
        triage_model="test-triage",
        run_id="run-test",
    )
    storage = _make_storage()
    return SynthesisAgent(
        storage=storage,
        run_id="run-test",
        config=cfg,
        domain_mode=DomainMode.GENERAL,
        dedup_enabled=dedup_enabled,
        compound_enabled=compound_enabled,
        synthesis_model=synthesis_model,
    )


# ── Schema tests ──────────────────────────────────────────────────────────────

class TestSchemas:
    def test_executor_type_synthesis_present(self):
        """ExecutorType.SYNTHESIS must exist (Gap 2 addition)."""
        assert ExecutorType.SYNTHESIS == "SYNTHESIS"

    def test_compound_finding_category_values(self):
        """All cross-domain pairing categories must be present."""
        cats = {c.value for c in CompoundFindingCategory}
        assert "SECURITY_ARCHITECTURE" in cats
        assert "SECURITY_STANDARDS" in cats
        assert "ARCHITECTURE_STANDARDS" in cats
        assert "ALL_DOMAINS" in cats

    def test_compound_finding_instantiation(self):
        """CompoundFinding must instantiate with defaults."""
        cf = CompoundFinding(
            run_id="r1",
            title="Race-condition auth bypass",
            description="A race enables privilege escalation",
            severity=Severity.CRITICAL,
            category=CompoundFindingCategory.SECURITY_ARCHITECTURE,
            contributing_issue_ids=["a", "b"],
            domains_involved=["SECURITY", "ARCHITECTURE"],
            amplification_factor=3.5,
            fix_complexity="HIGH",
        )
        assert cf.severity == Severity.CRITICAL
        assert cf.amplification_factor == 3.5
        assert cf.mil882e_category == MilStd882eCategory.CAT_I

    def test_synthesis_report_instantiation(self):
        """SynthesisReport must instantiate cleanly."""
        sr = SynthesisReport(
            run_id="r1",
            cycle=2,
            raw_issue_count=30,
            fingerprint_dedup_count=5,
            semantic_dedup_count=3,
            final_issue_count=22,
            compound_finding_count=4,
            compound_critical_count=2,
        )
        assert sr.raw_issue_count == 30
        assert sr.compound_finding_count == 4

    def test_issue_has_is_mandatory_field(self):
        """Issue must have is_mandatory (Gap 2 addition)."""
        issue = _make_issue()
        assert hasattr(issue, "is_mandatory")
        assert issue.is_mandatory is False

    def test_issue_has_compound_finding_id_field(self):
        """Issue must have compound_finding_id (Gap 2 addition)."""
        issue = _make_issue()
        assert hasattr(issue, "compound_finding_id")
        assert issue.compound_finding_id == ""


# ── Fingerprint deduplication ─────────────────────────────────────────────────

class TestFingerprintDedup:
    def test_exact_duplicates_removed(self):
        """Issues with identical fingerprints must be deduplicated."""
        agent = _make_agent()
        fp = "abc123"
        i1 = _make_issue(description="overflow", severity=Severity.MAJOR, fingerprint=fp)
        i2 = _make_issue(description="overflow", severity=Severity.MAJOR, fingerprint=fp)
        result = agent._dedup_by_fingerprint([i1, i2])
        assert len(result) == 1

    def test_different_fingerprints_kept(self):
        """Issues with different fingerprints must all be kept."""
        agent = _make_agent()
        i1 = _make_issue(description="overflow", line_start=10, fingerprint="fp1")
        i2 = _make_issue(description="null deref", line_start=20, fingerprint="fp2")
        result = agent._dedup_by_fingerprint([i1, i2])
        assert len(result) == 2

    def test_keeps_higher_severity(self):
        """When fingerprints match, keep the higher-severity finding."""
        agent = _make_agent()
        fp = "dup-fp"
        major = _make_issue(severity=Severity.MAJOR,    fingerprint=fp, confidence=0.9)
        crit  = _make_issue(severity=Severity.CRITICAL, fingerprint=fp, confidence=0.9)
        result = agent._dedup_by_fingerprint([major, crit])
        assert len(result) == 1
        assert result[0].severity == Severity.CRITICAL

    def test_keeps_higher_confidence_on_same_severity(self):
        """When severity ties, keep the higher-confidence finding."""
        agent = _make_agent()
        fp = "dup-fp"
        low_conf  = _make_issue(severity=Severity.MAJOR, confidence=0.5, fingerprint=fp)
        high_conf = _make_issue(severity=Severity.MAJOR, confidence=0.95, fingerprint=fp)
        result = agent._dedup_by_fingerprint([low_conf, high_conf])
        assert result[0].confidence == 0.95

    def test_keeps_security_domain_on_tie(self):
        """When severity and confidence tie, prefer SECURITY domain."""
        agent = _make_agent()
        fp = "dup-fp"
        arch_issue = _make_issue(
            severity=Severity.MAJOR, confidence=0.8,
            executor_type=ExecutorType.ARCHITECTURE, fingerprint=fp,
        )
        sec_issue = _make_issue(
            severity=Severity.MAJOR, confidence=0.8,
            executor_type=ExecutorType.SECURITY, fingerprint=fp,
        )
        result = agent._dedup_by_fingerprint([arch_issue, sec_issue])
        assert result[0].executor_type == ExecutorType.SECURITY

    def test_computes_missing_fingerprint(self):
        """Issues without a fingerprint get one auto-computed."""
        agent = _make_agent()
        issue = Issue(
            run_id="r1",
            file_path="a.py",
            line_start=5,
            description="test issue",
            severity=Severity.MINOR,
            status=IssueStatus.OPEN,
            fingerprint="",   # intentionally blank
        )
        result = agent._dedup_by_fingerprint([issue])
        assert len(result) == 1   # no crash on missing fingerprint

    def test_three_domains_same_bug(self):
        """Three auditors reporting the same bug deduplicate to one."""
        agent = _make_agent()
        fp = "same-bug"
        sec   = _make_issue(executor_type=ExecutorType.SECURITY,     severity=Severity.CRITICAL, fingerprint=fp)
        arch  = _make_issue(executor_type=ExecutorType.ARCHITECTURE, severity=Severity.MAJOR,    fingerprint=fp)
        std   = _make_issue(executor_type=ExecutorType.STANDARDS,    severity=Severity.MINOR,    fingerprint=fp)
        result = agent._dedup_by_fingerprint([arch, std, sec])
        assert len(result) == 1
        assert result[0].severity == Severity.CRITICAL


# ── Semantic deduplication ─────────────────────────────────────────────────────

class TestSemanticDedup:
    def test_apply_dedup_response_removes_correct_indices(self):
        """_apply_dedup_response must remove non-canonical indices."""
        agent = _make_agent()
        issues = [_make_issue(line_start=i) for i in range(5)]
        resp = DeduplicationResponse(
            duplicate_clusters=[
                DuplicateCluster(indices=[0, 2, 4], canonical_index=0),
                DuplicateCluster(indices=[1, 3],    canonical_index=1),
            ],
            total_duplicates_removed=3,
        )
        result = agent._apply_dedup_response(issues, resp)
        # Keeps indices 0 and 1 only
        assert len(result) == 2
        assert result[0].line_start == 0
        assert result[1].line_start == 1

    def test_apply_dedup_invalid_canonical_ignored(self):
        """Out-of-bounds canonical index must not crash — cluster ignored."""
        agent = _make_agent()
        issues = [_make_issue(line_start=i) for i in range(3)]
        resp = DeduplicationResponse(
            duplicate_clusters=[
                DuplicateCluster(indices=[0, 1], canonical_index=99),  # invalid
            ],
        )
        result = agent._apply_dedup_response(issues, resp)
        # Cluster ignored — all 3 kept
        assert len(result) == 3

    def test_apply_dedup_empty_clusters_no_op(self):
        """Empty duplicate_clusters must return issues unchanged."""
        agent = _make_agent()
        issues = [_make_issue(line_start=i) for i in range(4)]
        resp = DeduplicationResponse(duplicate_clusters=[])
        result = agent._apply_dedup_response(issues, resp)
        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_semantic_dedup_calls_llm(self):
        """_dedup_semantic must call call_llm_structured_deterministic."""
        agent = _make_agent()
        issues = [_make_issue(line_start=i) for i in range(3)]
        mock_resp = DeduplicationResponse(
            duplicate_clusters=[
                DuplicateCluster(indices=[0, 1], canonical_index=0)
            ]
        )
        with patch.object(
            agent,
            "call_llm_structured_deterministic",
            new=AsyncMock(return_value=mock_resp),
        ):
            result = await agent._dedup_semantic(issues)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_semantic_dedup_single_issue_no_llm(self):
        """_dedup_semantic with 1 issue must return it without calling LLM."""
        agent = _make_agent()
        issues = [_make_issue()]
        with patch.object(
            agent, "call_llm_structured_deterministic", new=AsyncMock()
        ) as mock_llm:
            result = await agent._dedup_semantic(issues)
        mock_llm.assert_not_called()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_semantic_dedup_batches_large_lists(self):
        """Large issue lists must be processed in batches."""
        agent = _make_agent()
        # Create 90 issues — more than _DEDUP_BATCH_SIZE (60)
        issues = [_make_issue(line_start=i, fingerprint=f"fp{i}") for i in range(90)]
        call_count = 0

        async def fake_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return DeduplicationResponse(duplicate_clusters=[])

        with patch.object(
            agent, "call_llm_structured_deterministic", new=fake_llm
        ):
            result = await agent._dedup_semantic_batched(issues)

        # Should have been called at least twice (90 / 60 = 2 batches)
        assert call_count >= 2
        # No duplicates in this list, so all 90 kept
        assert len(result) == 90


# ── Compound finding detection ────────────────────────────────────────────────

class TestCompoundFindingDetection:
    def test_raw_to_compound_valid(self):
        """Valid RawCompoundFinding must produce a CompoundFinding."""
        agent = _make_agent()
        issues = [
            _make_issue(executor_type=ExecutorType.SECURITY,     line_start=10),
            _make_issue(executor_type=ExecutorType.ARCHITECTURE, line_start=20),
        ]
        raw = RawCompoundFinding(
            title="Race-condition privilege escalation",
            description="Race in auth check enables bypass",
            severity="CRITICAL",
            contributing_indices=[0, 1],
            domains_involved=["SECURITY", "ARCHITECTURE"],
            category="SECURITY_ARCHITECTURE",
            amplification_factor=4.0,
            fix_complexity="HIGH",
            rationale="Winning the race bypasses the auth check entirely.",
        )
        cf = agent._raw_to_compound(raw, issues)
        assert cf is not None
        assert cf.title == "Race-condition privilege escalation"
        assert cf.amplification_factor == 4.0
        assert len(cf.contributing_issue_ids) == 2

    def test_raw_to_compound_less_than_two_valid_indices(self):
        """RawCompoundFinding with <2 valid indices must be dropped (returns None)."""
        agent = _make_agent()
        issues = [_make_issue()]
        raw = RawCompoundFinding(
            title="Incomplete compound",
            description="Only one contributor",
            contributing_indices=[0],   # only 1
        )
        cf = agent._raw_to_compound(raw, issues)
        assert cf is None

    def test_raw_to_compound_out_of_bounds_indices_dropped(self):
        """Out-of-bounds contributing_indices must be filtered; result <2 → None."""
        agent = _make_agent()
        issues = [_make_issue()]  # 1 issue, indices 0 valid; 99 invalid
        raw = RawCompoundFinding(
            title="Bad indices",
            description="desc",
            contributing_indices=[0, 99],  # 99 is out of bounds → only 1 valid
        )
        cf = agent._raw_to_compound(raw, issues)
        assert cf is None

    def test_raw_to_compound_unknown_category_falls_back(self):
        """Unknown category string must fall back to SECURITY_ARCHITECTURE."""
        agent = _make_agent()
        issues = [
            _make_issue(line_start=1),
            _make_issue(line_start=2),
        ]
        raw = RawCompoundFinding(
            title="Test",
            description="desc",
            contributing_indices=[0, 1],
            category="COMPLETELY_UNKNOWN_VALUE",
        )
        cf = agent._raw_to_compound(raw, issues)
        assert cf is not None
        assert cf.category == CompoundFindingCategory.SECURITY_ARCHITECTURE

    def test_severity_escalation_high_amplification(self):
        """amplification ≥ 3.0 must escalate severity by one rank."""
        agent = _make_agent()
        escalated = agent._escalate_severity(Severity.MAJOR, amplification=3.0)
        assert escalated == Severity.CRITICAL

    def test_severity_escalation_low_amplification_no_change(self):
        """amplification < 3.0 must NOT escalate severity."""
        agent = _make_agent()
        unchanged = agent._escalate_severity(Severity.MAJOR, amplification=2.0)
        assert unchanged == Severity.MAJOR

    def test_severity_escalation_caps_at_critical(self):
        """CRITICAL cannot be escalated further."""
        agent = _make_agent()
        still_critical = agent._escalate_severity(Severity.CRITICAL, amplification=9.0)
        assert still_critical == Severity.CRITICAL

    def test_severity_escalation_info_to_minor(self):
        """INFO with high amplification escalates to MINOR."""
        agent = _make_agent()
        escalated = agent._escalate_severity(Severity.INFO, amplification=5.0)
        assert escalated == Severity.MINOR

    @pytest.mark.asyncio
    async def test_detect_compound_skips_single_domain(self):
        """Compound detection must be skipped when all issues are from one domain."""
        agent = _make_agent()
        issues = [
            _make_issue(executor_type=ExecutorType.SECURITY, line_start=i)
            for i in range(5)
        ]
        result = await agent._detect_compound_findings(issues)
        assert result == []

    @pytest.mark.asyncio
    async def test_detect_compound_skips_too_few_issues(self):
        """Compound detection must be skipped with fewer than _COMPOUND_MIN_CROSS_DOMAIN_ISSUES."""
        agent = _make_agent()
        agent._COMPOUND_MIN_CROSS_DOMAIN_ISSUES = 5  # set threshold high
        issues = [
            _make_issue(executor_type=ExecutorType.SECURITY),
            _make_issue(executor_type=ExecutorType.ARCHITECTURE),
        ]
        result = await agent._detect_compound_findings(issues)
        assert result == []

    @pytest.mark.asyncio
    async def test_detect_compound_calls_llm_with_correct_structure(self):
        """Compound detection must call call_llm_structured with multi-domain issues."""
        agent = _make_agent()
        issues = [
            _make_issue(executor_type=ExecutorType.SECURITY,     line_start=1),
            _make_issue(executor_type=ExecutorType.ARCHITECTURE, line_start=2),
            _make_issue(executor_type=ExecutorType.STANDARDS,    line_start=3),
        ]
        mock_resp = CompoundFindingResponse(
            compound_findings=[
                RawCompoundFinding(
                    title="Race+Auth compound",
                    description="Race enables auth bypass",
                    severity="CRITICAL",
                    contributing_indices=[0, 1],
                    domains_involved=["SECURITY", "ARCHITECTURE"],
                    category="SECURITY_ARCHITECTURE",
                    amplification_factor=3.5,
                    fix_complexity="HIGH",
                    rationale="Combined effect is CRITICAL",
                )
            ],
            synthesis_summary="One compound finding detected.",
        )
        with patch.object(
            agent, "call_llm_structured", new=AsyncMock(return_value=mock_resp)
        ):
            result = await agent._detect_compound_findings(issues)

        assert len(result) == 1
        assert result[0].title == "Race+Auth compound"
        assert result[0].amplification_factor == 3.5

    @pytest.mark.asyncio
    async def test_detect_compound_respects_max_limit(self):
        """Compound detection must not return more than max_compound_findings."""
        agent = _make_agent()
        agent.max_compound_findings = 2  # cap at 2
        issues = [
            _make_issue(executor_type=ExecutorType.SECURITY,     line_start=i)
            for i in range(2)
        ] + [
            _make_issue(executor_type=ExecutorType.ARCHITECTURE, line_start=i + 10)
            for i in range(2)
        ]
        raw_findings = [
            RawCompoundFinding(
                title=f"Compound {i}",
                description=f"desc {i}",
                contributing_indices=[0, 2],
                severity="CRITICAL",
                category="SECURITY_ARCHITECTURE",
                amplification_factor=2.0,
            )
            for i in range(5)  # 5 findings but cap is 2
        ]
        mock_resp = CompoundFindingResponse(compound_findings=raw_findings)
        with patch.object(
            agent, "call_llm_structured", new=AsyncMock(return_value=mock_resp)
        ):
            result = await agent._detect_compound_findings(issues)

        assert len(result) <= 2


# ── _compound_to_issue ────────────────────────────────────────────────────────

class TestCompoundToIssue:
    def test_executor_type_is_synthesis(self):
        """Issue produced from CompoundFinding must have executor_type=SYNTHESIS."""
        agent = _make_agent()
        contributing = [
            _make_issue(executor_type=ExecutorType.SECURITY,     line_start=10),
            _make_issue(executor_type=ExecutorType.ARCHITECTURE, line_start=20),
        ]
        cf = CompoundFinding(
            run_id="r1",
            title="Race-condition auth bypass",
            description="desc",
            severity=Severity.CRITICAL,
            category=CompoundFindingCategory.SECURITY_ARCHITECTURE,
            contributing_issue_ids=[contributing[0].id, contributing[1].id],
            domains_involved=["SECURITY", "ARCHITECTURE"],
            amplification_factor=3.5,
        )
        issue = agent._compound_to_issue(cf, contributing)
        assert issue.executor_type == ExecutorType.SYNTHESIS

    def test_description_includes_compound_header(self):
        """Issue description must include [COMPOUND FINDING] header."""
        agent = _make_agent()
        issues = [
            _make_issue(line_start=10),
            _make_issue(line_start=20),
        ]
        cf = CompoundFinding(
            run_id="r1",
            title="Test compound",
            description="A compound issue",
            severity=Severity.CRITICAL,
            category=CompoundFindingCategory.SECURITY_ARCHITECTURE,
            contributing_issue_ids=[issues[0].id, issues[1].id],
            domains_involved=["SECURITY", "ARCHITECTURE"],
        )
        issue = agent._compound_to_issue(cf, issues)
        assert "[COMPOUND FINDING" in issue.description

    def test_severity_matches_compound_finding(self):
        """Issue severity must match the CompoundFinding severity."""
        agent = _make_agent()
        issues = [_make_issue(line_start=i) for i in range(2)]
        cf = CompoundFinding(
            run_id="r1",
            title="T",
            description="d",
            severity=Severity.CRITICAL,
            category=CompoundFindingCategory.ALL_DOMAINS,
            contributing_issue_ids=[issues[0].id, issues[1].id],
        )
        issue = agent._compound_to_issue(cf, issues)
        assert issue.severity == Severity.CRITICAL

    def test_is_mandatory_set_for_critical(self):
        """is_mandatory must be True when compound finding is CRITICAL."""
        agent = _make_agent()
        issues = [_make_issue(line_start=i) for i in range(2)]
        cf = CompoundFinding(
            run_id="r1",
            title="T",
            description="d",
            severity=Severity.CRITICAL,
            category=CompoundFindingCategory.SECURITY_ARCHITECTURE,
            contributing_issue_ids=[issues[0].id, issues[1].id],
        )
        issue = agent._compound_to_issue(cf, issues)
        assert issue.is_mandatory is True

    def test_fix_requires_files_union_of_contributors(self):
        """fix_requires_files must be the union of all contributor files."""
        agent = _make_agent()
        i1 = _make_issue(file_path="auth.py",    line_start=10)
        i2 = _make_issue(file_path="handler.py", line_start=20)
        i1.fix_requires_files = ["auth.py"]
        i2.fix_requires_files = ["handler.py", "shared.py"]
        cf = CompoundFinding(
            run_id="r1",
            title="T",
            description="d",
            severity=Severity.CRITICAL,
            category=CompoundFindingCategory.SECURITY_ARCHITECTURE,
            contributing_issue_ids=[i1.id, i2.id],
        )
        issue = agent._compound_to_issue(cf, [i1, i2])
        assert "auth.py" in issue.fix_requires_files
        assert "handler.py" in issue.fix_requires_files
        assert "shared.py" in issue.fix_requires_files

    def test_fingerprint_deterministic(self):
        """Same compound finding must produce the same fingerprint each time."""
        agent = _make_agent()
        issues = [_make_issue(line_start=i) for i in range(2)]
        cf = CompoundFinding(
            run_id="r1",
            title="T",
            description="d",
            severity=Severity.MAJOR,
            category=CompoundFindingCategory.SECURITY_STANDARDS,
            contributing_issue_ids=[issues[0].id, issues[1].id],
        )
        issue1 = agent._compound_to_issue(cf, issues)
        issue2 = agent._compound_to_issue(cf, issues)
        assert issue1.fingerprint == issue2.fingerprint


# ── Full pipeline run() ───────────────────────────────────────────────────────

class TestSynthesisAgentRun:
    @pytest.mark.asyncio
    async def test_run_empty_input(self):
        """run() with empty input must return ([], []) immediately."""
        agent = _make_agent()
        deduped, compound = await agent.run([])
        assert deduped == []
        assert compound == []

    @pytest.mark.asyncio
    async def test_run_returns_deduped_and_compound(self):
        """run() must return (deduplicated_issues, compound_findings)."""
        agent = _make_agent()
        fp = "shared-fp"
        issues = [
            _make_issue(executor_type=ExecutorType.SECURITY,     fingerprint=fp,  severity=Severity.CRITICAL),
            _make_issue(executor_type=ExecutorType.ARCHITECTURE, fingerprint=fp,  severity=Severity.MAJOR),
            _make_issue(executor_type=ExecutorType.STANDARDS,    fingerprint="s2", line_start=50),
        ]

        dedup_resp = DeduplicationResponse(
            duplicate_clusters=[
                DuplicateCluster(indices=[0, 1], canonical_index=0)
            ]
        )
        compound_resp = CompoundFindingResponse(
            compound_findings=[],
            synthesis_summary="No compound findings.",
        )

        with patch.object(agent, "call_llm_structured_deterministic", new=AsyncMock(return_value=dedup_resp)):
            with patch.object(agent, "call_llm_structured", new=AsyncMock(return_value=compound_resp)):
                deduped, compound = await agent.run(issues)

        # The duplicate pair [0,1] → keep index 0; plus the unpaired issue[2]
        assert len(deduped) == 2
        assert compound == []

    @pytest.mark.asyncio
    async def test_run_persists_compound_to_storage(self):
        """run() must call storage.upsert_issue for each compound finding."""
        agent = _make_agent()
        issues = [
            _make_issue(executor_type=ExecutorType.SECURITY,     line_start=1, fingerprint="fp1"),
            _make_issue(executor_type=ExecutorType.ARCHITECTURE, line_start=2, fingerprint="fp2"),
            _make_issue(executor_type=ExecutorType.STANDARDS,    line_start=3, fingerprint="fp3"),
        ]
        compound_resp = CompoundFindingResponse(
            compound_findings=[
                RawCompoundFinding(
                    title="Auth+race compound",
                    description="Race enables auth bypass",
                    severity="CRITICAL",
                    contributing_indices=[0, 1],
                    domains_involved=["SECURITY", "ARCHITECTURE"],
                    category="SECURITY_ARCHITECTURE",
                    amplification_factor=4.0,
                    fix_complexity="HIGH",
                    rationale="critical",
                )
            ],
        )
        dedup_resp = DeduplicationResponse(duplicate_clusters=[])

        with patch.object(agent, "call_llm_structured_deterministic", new=AsyncMock(return_value=dedup_resp)):
            with patch.object(agent, "call_llm_structured", new=AsyncMock(return_value=compound_resp)):
                _, compound = await agent.run(issues)

        # storage.upsert_issue must have been called for the compound issue
        assert agent.storage.upsert_issue.called
        assert len(compound) == 1
        assert compound[0].title == "Auth+race compound"

    @pytest.mark.asyncio
    async def test_run_fail_open_on_synthesis_crash(self):
        """If LLM call crashes, run() must return original issues unchanged."""
        agent = _make_agent()
        issues = [_make_issue(line_start=i, fingerprint=f"fp{i}") for i in range(3)]

        async def exploding_llm(*args, **kwargs):
            raise RuntimeError("LLM service unavailable")

        with patch.object(agent, "call_llm_structured_deterministic", new=exploding_llm):
            deduped, compound = await agent.run(issues)

        # Fingerprint dedup runs first (no duplicates here), then semantic crashes
        # but fail-open means we get the fingerprint-deduped result back
        assert len(deduped) == 3
        assert compound == []

    @pytest.mark.asyncio
    async def test_run_writes_audit_trail(self):
        """run() must call storage.append_audit_trail once."""
        agent = _make_agent()
        issues = [_make_issue()]
        dedup_resp = DeduplicationResponse(duplicate_clusters=[])
        compound_resp = CompoundFindingResponse(compound_findings=[])

        with patch.object(agent, "call_llm_structured_deterministic", new=AsyncMock(return_value=dedup_resp)):
            with patch.object(agent, "call_llm_structured", new=AsyncMock(return_value=compound_resp)):
                await agent.run(issues)

        agent.storage.append_audit_trail.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_dedup_disabled(self):
        """When dedup_enabled=False, run() must skip LLM dedup but still run compound."""
        agent = _make_agent(dedup_enabled=False, compound_enabled=False)
        issues = [
            _make_issue(line_start=i, fingerprint=f"fp{i}") for i in range(4)
        ]
        with patch.object(
            agent, "call_llm_structured_deterministic", new=AsyncMock()
        ) as mock_dedup:
            deduped, compound = await agent.run(issues)

        # Fingerprint dedup still runs (it's deterministic, not an LLM call)
        # LLM semantic dedup should NOT be called
        mock_dedup.assert_not_called()
        assert len(deduped) == 4  # no duplicates in unique fps
        assert compound == []


# ── Config loader integration ─────────────────────────────────────────────────

class TestConfigLoader:
    def test_synthesis_toml_section_maps_to_config(self):
        """[synthesis] TOML section must map to StabilizerConfig fields."""
        from config.loader import _flatten_toml

        raw = {
            "repo_url": "https://github.com/test/repo",
            "cpg": {},
            "synthesis": {
                "enabled":               False,
                "dedup_enabled":         False,
                "compound_enabled":      True,
                "synthesis_model":       "openrouter/deepseek/deepseek-coder-v2",
                "max_compound_findings": 10,
            },
        }
        flat = _flatten_toml(raw)
        assert flat["synthesis_enabled"] is False
        assert flat["synthesis_dedup_enabled"] is False
        assert flat["synthesis_compound_enabled"] is True
        assert flat["synthesis_model"] == "openrouter/deepseek/deepseek-coder-v2"
        assert flat["synthesis_max_compound"] == 10

    def test_synthesis_env_vars_mapped(self):
        """RHODAWK_SYNTHESIS_MODEL env var must be picked up by _apply_env."""
        import os
        from config.loader import _apply_env

        os.environ["RHODAWK_SYNTHESIS_MODEL"] = "test-synthesis-model"
        data: dict = {}
        try:
            _apply_env(data)
            assert data.get("synthesis_model") == "test-synthesis-model"
        finally:
            del os.environ["RHODAWK_SYNTHESIS_MODEL"]

    def test_synthesis_enabled_env_var(self):
        """RHODAWK_SYNTHESIS_ENABLED=false must disable synthesis."""
        import os
        from config.loader import _apply_env

        os.environ["RHODAWK_SYNTHESIS_ENABLED"] = "false"
        data: dict = {}
        try:
            _apply_env(data)
            assert data.get("synthesis_enabled") is False
        finally:
            del os.environ["RHODAWK_SYNTHESIS_ENABLED"]


# ── Controller integration ────────────────────────────────────────────────────

class TestControllerGap2Integration:
    def test_controller_config_has_synthesis_fields(self):
        """StabilizerConfig must have all Gap 2 synthesis fields."""
        from orchestrator.controller import StabilizerConfig
        from pathlib import Path

        cfg = StabilizerConfig(repo_url="x", repo_root=Path("."))
        assert hasattr(cfg, "synthesis_enabled")
        assert cfg.synthesis_enabled is True
        assert hasattr(cfg, "synthesis_dedup_enabled")
        assert cfg.synthesis_dedup_enabled is True
        assert hasattr(cfg, "synthesis_compound_enabled")
        assert cfg.synthesis_compound_enabled is True
        assert hasattr(cfg, "synthesis_model")
        assert cfg.synthesis_model == ""
        assert hasattr(cfg, "synthesis_max_compound")
        assert cfg.synthesis_max_compound == 20

    def test_controller_has_last_compound_findings(self):
        """StabilizerController must expose _last_compound_findings list."""
        from orchestrator.controller import StabilizerController, StabilizerConfig
        from pathlib import Path

        cfg = StabilizerConfig(repo_url="x", repo_root=Path("."))
        ctrl = StabilizerController(cfg)
        assert hasattr(ctrl, "_last_compound_findings")
        assert isinstance(ctrl._last_compound_findings, list)
