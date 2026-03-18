"""
tests/unit/test_consensus.py
============================
Unit tests for orchestrator.consensus.ConsensusEngine.

Covers:
• Single-agent vote: passes if above floor, blocked if below
• Multi-agent vote: weighted confidence computation
• Required-domain enforcement for CRITICAL issues
• High-centrality file raises confidence floor
• DisagreementAction routing
• filter_approved: mutates issue.consensus_votes and consensus_confidence
• Domain weights: SECURITY domain weighted 2× for CRITICAL findings
• Summary dict structure
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from brain.schemas import (
    ConsensusRule,
    DisagreementAction,
    ExecutorType,
    Issue,
    IssueStatus,
    Severity,
)
from orchestrator.consensus import ConsensusEngine, DEFAULT_RULES


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _issue(
    severity:   Severity     = Severity.MAJOR,
    executor:   ExecutorType = ExecutorType.ARCHITECTURE,
    file_path:  str          = "agents/fixer.py",
    fingerprint: str         = "",
    confidence: float        = 0.85,
    run_id:     str          = "",
) -> Issue:
    fp = fingerprint or uuid.uuid4().hex[:16]
    return Issue(
        run_id=run_id or str(uuid.uuid4()),
        severity=severity,
        file_path=file_path,
        executor_type=executor,
        description="Test finding",
        fingerprint=fp,
        consensus_confidence=confidence,
    )


def _engine(**kwargs) -> ConsensusEngine:
    return ConsensusEngine(**kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Basic evaluation
# ──────────────────────────────────────────────────────────────────────────────

class TestSingleAgentVote:
    def test_minor_single_agent_above_floor_approved(self):
        engine  = _engine()
        issues  = [_issue(severity=Severity.MINOR, confidence=0.60)]
        results = engine.evaluate_issues(issues)
        assert len(results) == 1
        assert results[0].approved is True

    def test_minor_single_agent_below_floor_rejected(self):
        engine = _engine(rules={
            Severity.MINOR: ConsensusRule(minimum_agents=1, confidence_floor=0.80)
        })
        issues  = [_issue(severity=Severity.MINOR, confidence=0.40)]
        results = engine.evaluate_issues(issues)
        assert results[0].approved is False

    def test_info_always_passes(self):
        engine  = _engine()
        issues  = [_issue(severity=Severity.INFO, confidence=0.0)]
        results = engine.evaluate_issues(issues)
        assert results[0].approved is True

    def test_major_requires_two_agents_single_fails(self):
        engine  = _engine()
        issues  = [_issue(severity=Severity.MAJOR, confidence=0.95)]
        results = engine.evaluate_issues(issues)
        # DEFAULT_RULES MAJOR requires minimum_agents=2
        assert results[0].approved is False
        assert "2" in results[0].reason  # message mentions the required count

    def test_major_two_agents_same_fingerprint_passes(self):
        engine = _engine()
        fp     = "fp_major_001"
        issues = [
            _issue(severity=Severity.MAJOR, executor=ExecutorType.SECURITY,
                   fingerprint=fp, confidence=0.80),
            _issue(severity=Severity.MAJOR, executor=ExecutorType.ARCHITECTURE,
                   fingerprint=fp, confidence=0.75),
        ]
        results = engine.evaluate_issues(issues)
        assert len(results) == 1           # grouped by fingerprint
        assert results[0].approved is True
        assert len(results[0].votes) == 2


class TestCriticalRequiredDomain:
    def test_critical_without_security_domain_blocked(self):
        engine = _engine()
        fp     = "critical_fp_001"
        issues = [
            _issue(severity=Severity.CRITICAL, executor=ExecutorType.ARCHITECTURE,
                   fingerprint=fp, confidence=0.95),
            _issue(severity=Severity.CRITICAL, executor=ExecutorType.STANDARDS,
                   fingerprint=fp, confidence=0.92),
        ]
        results = engine.evaluate_issues(issues)
        assert results[0].approved is False
        assert "SECURITY" in results[0].reason

    def test_critical_with_security_domain_passes(self):
        engine = _engine()
        fp     = "critical_fp_002"
        issues = [
            _issue(severity=Severity.CRITICAL, executor=ExecutorType.SECURITY,
                   fingerprint=fp, confidence=0.90),
            _issue(severity=Severity.CRITICAL, executor=ExecutorType.ARCHITECTURE,
                   fingerprint=fp, confidence=0.88),
        ]
        results = engine.evaluate_issues(issues)
        assert results[0].approved is True

    def test_critical_security_only_one_agent_below_two_fails(self):
        engine = _engine()
        fp     = "critical_fp_003"
        issues = [
            _issue(severity=Severity.CRITICAL, executor=ExecutorType.SECURITY,
                   fingerprint=fp, confidence=0.95),
        ]
        results = engine.evaluate_issues(issues)
        assert results[0].approved is False
        assert "2" in results[0].reason


# ──────────────────────────────────────────────────────────────────────────────
# Weighted confidence computation
# ──────────────────────────────────────────────────────────────────────────────

class TestWeightedConfidence:
    def test_security_weighted_2x_for_critical(self):
        """SECURITY vote weighted 2.0, ARCH vote weighted 1.0 for CRITICAL."""
        engine = _engine()
        fp     = "wc_001"
        # Security conf=0.90 (weight 2), Architecture conf=0.60 (weight 1)
        # Expected weighted = (0.90*2 + 0.60*1) / (2+1) = 2.40/3 = 0.80
        issues = [
            _issue(severity=Severity.CRITICAL, executor=ExecutorType.SECURITY,
                   fingerprint=fp, confidence=0.90),
            _issue(severity=Severity.CRITICAL, executor=ExecutorType.ARCHITECTURE,
                   fingerprint=fp, confidence=0.60),
        ]
        results = engine.evaluate_issues(issues)
        expected = (0.90 * 2 + 0.60 * 1) / 3
        assert abs(results[0].weighted_confidence - expected) < 0.01

    def test_equal_weight_domains_average(self):
        """Two ARCHITECTURE votes weighted equally → simple average."""
        engine = _engine()
        fp     = "wc_002"
        issues = [
            _issue(severity=Severity.MAJOR, executor=ExecutorType.ARCHITECTURE,
                   fingerprint=fp, confidence=0.80),
            _issue(severity=Severity.MAJOR, executor=ExecutorType.STANDARDS,
                   fingerprint=fp, confidence=0.70),
        ]
        results = engine.evaluate_issues(issues)
        # Arch weight = 1.5 for MAJOR, Standards weight = 1.0
        expected = (0.80 * 1.5 + 0.70 * 1.0) / (1.5 + 1.0)
        assert abs(results[0].weighted_confidence - expected) < 0.01

    def test_confidence_never_exceeds_1(self):
        engine = _engine()
        fp     = "wc_003"
        issues = [
            _issue(severity=Severity.MINOR, fingerprint=fp, confidence=0.99),
        ]
        results = engine.evaluate_issues(issues)
        assert results[0].weighted_confidence <= 1.0

    def test_confidence_never_below_0(self):
        engine = _engine()
        fp     = "wc_004"
        issues = [
            _issue(severity=Severity.INFO, fingerprint=fp, confidence=0.0),
        ]
        results = engine.evaluate_issues(issues)
        assert results[0].weighted_confidence >= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# High-centrality file raises floor
# ──────────────────────────────────────────────────────────────────────────────

class TestHighCentralityRaisesFloor:
    class _MockGraphEngine:
        """Mock engine that reports a fixed centrality score."""
        is_built = True

        def __init__(self, centrality: float) -> None:
            self._centrality = centrality

        def centrality_score(self, path: str) -> float:
            return self._centrality

    def test_high_centrality_raises_floor_below_raised_floor_fails(self):
        """A finding that would pass the normal floor fails for a hub file."""
        # Normal floor for MAJOR = 0.70; high_centrality_multiplier = 1.20
        # Raised floor = 0.70 * 1.20 = 0.84
        # Provide weighted confidence = 0.75 → passes 0.70 but fails 0.84
        mock_graph = self._MockGraphEngine(centrality=0.80)
        engine = ConsensusEngine(graph_engine=mock_graph)
        fp     = "hc_001"
        issues = [
            _issue(severity=Severity.MAJOR, executor=ExecutorType.SECURITY,
                   fingerprint=fp, confidence=0.75),
            _issue(severity=Severity.MAJOR, executor=ExecutorType.ARCHITECTURE,
                   fingerprint=fp, confidence=0.75),
        ]
        results = engine.evaluate_issues(issues)
        # weighted ≈ 0.75; raised floor ≈ 0.84 → should fail
        assert results[0].approved is False

    def test_low_centrality_uses_normal_floor(self):
        """A finding for a low-centrality file uses the unmodified floor."""
        mock_graph = self._MockGraphEngine(centrality=0.20)
        engine = ConsensusEngine(graph_engine=mock_graph)
        fp     = "hc_002"
        issues = [
            _issue(severity=Severity.MAJOR, executor=ExecutorType.SECURITY,
                   fingerprint=fp, confidence=0.75),
            _issue(severity=Severity.MAJOR, executor=ExecutorType.ARCHITECTURE,
                   fingerprint=fp, confidence=0.75),
        ]
        results = engine.evaluate_issues(issues)
        # weighted ≈ (0.75*2 + 0.75*1)/(2+1) = 0.75 > 0.70 → passes
        assert results[0].approved is True

    def test_no_graph_engine_no_centrality_adjustment(self):
        """Without a graph engine, floor is never adjusted."""
        engine = ConsensusEngine(graph_engine=None)
        fp     = "hc_003"
        issues = [
            _issue(severity=Severity.MAJOR, executor=ExecutorType.SECURITY,
                   fingerprint=fp, confidence=0.75),
            _issue(severity=Severity.MAJOR, executor=ExecutorType.ARCHITECTURE,
                   fingerprint=fp, confidence=0.75),
        ]
        results = engine.evaluate_issues(issues)
        assert results[0].approved is True


# ──────────────────────────────────────────────────────────────────────────────
# Disagreement action routing
# ──────────────────────────────────────────────────────────────────────────────

class TestDisagreementAction:
    def test_escalate_human_on_critical_failure(self):
        engine  = _engine()
        fp      = "da_001"
        # Single agent, no SECURITY domain → should fail with ESCALATE_HUMAN action
        issues  = [_issue(severity=Severity.CRITICAL, executor=ExecutorType.ARCHITECTURE,
                          fingerprint=fp, confidence=0.95)]
        results = engine.evaluate_issues(issues)
        assert results[0].approved is False
        assert results[0].action == DisagreementAction.ESCALATE_HUMAN

    def test_flag_uncertain_on_major_failure(self):
        engine  = _engine()
        fp      = "da_002"
        # Single agent → below minimum_agents for MAJOR → FLAG_UNCERTAIN
        issues  = [_issue(severity=Severity.MAJOR, confidence=0.95, fingerprint=fp)]
        results = engine.evaluate_issues(issues)
        assert results[0].action == DisagreementAction.FLAG_UNCERTAIN

    def test_custom_rule_auto_reject(self):
        custom_rules = {
            Severity.MINOR: ConsensusRule(
                minimum_agents=1,
                confidence_floor=0.90,
                disagreement_action=DisagreementAction.AUTO_REJECT,
            )
        }
        engine  = ConsensusEngine(rules=custom_rules)
        fp      = "da_003"
        issues  = [_issue(severity=Severity.MINOR, confidence=0.50, fingerprint=fp)]
        results = engine.evaluate_issues(issues)
        assert results[0].action == DisagreementAction.AUTO_REJECT


# ──────────────────────────────────────────────────────────────────────────────
# filter_approved
# ──────────────────────────────────────────────────────────────────────────────

class TestFilterApproved:
    def test_approved_issues_returned(self):
        engine = _engine()
        fp1    = "fa_001"
        fp2    = "fa_002"
        issues = [
            _issue(severity=Severity.MINOR, fingerprint=fp1, confidence=0.70),
            _issue(severity=Severity.MINOR, fingerprint=fp2, confidence=0.70),
        ]
        results  = engine.evaluate_issues(issues)
        approved = engine.filter_approved(issues, results)
        assert len(approved) == 2

    def test_rejected_issues_not_returned(self):
        engine = _engine()
        fp     = "fa_003"
        # Below floor for MINOR
        issues = [_issue(severity=Severity.MINOR, fingerprint=fp, confidence=0.20)]
        custom_rules = {
            Severity.MINOR: ConsensusRule(minimum_agents=1, confidence_floor=0.80)
        }
        engine2  = ConsensusEngine(rules=custom_rules)
        results  = engine2.evaluate_issues(issues)
        approved = engine2.filter_approved(issues, results)
        assert len(approved) == 0

    def test_consensus_metadata_mutated_on_issue(self):
        """filter_approved should mutate issue.consensus_votes and consensus_confidence."""
        engine  = _engine()
        fp      = "fa_004"
        issues  = [_issue(severity=Severity.MINOR, fingerprint=fp, confidence=0.70)]
        results = engine.evaluate_issues(issues)
        engine.filter_approved(issues, results)
        assert issues[0].consensus_votes >= 1
        assert issues[0].consensus_confidence > 0.0

    def test_deduplication_by_fingerprint(self):
        """Two issues with same fingerprint: only one representative in approved list."""
        engine = _engine()
        fp     = "fa_005"
        issues = [
            _issue(severity=Severity.MINOR, fingerprint=fp, confidence=0.70),
            _issue(severity=Severity.MINOR, fingerprint=fp, confidence=0.70),
        ]
        results  = engine.evaluate_issues(issues)
        approved = engine.filter_approved(issues, results)
        # Only one representative returned per fingerprint
        assert len(approved) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

class TestSummary:
    def test_summary_keys_present(self):
        engine  = _engine()
        results = engine.evaluate_issues([
            _issue(severity=Severity.MINOR, confidence=0.70),
        ])
        s = engine.summary(results)
        assert "total" in s
        assert "approved" in s
        assert "rejected" in s
        assert "escalated" in s
        assert "mean_confidence" in s

    def test_summary_counts(self):
        engine = _engine()
        fp1    = "s_001"
        fp2    = "s_002"
        # fp1: MINOR passes, fp2: MAJOR single agent fails (escalation)
        issues = [
            _issue(severity=Severity.MINOR, fingerprint=fp1, confidence=0.70),
            _issue(severity=Severity.CRITICAL, fingerprint=fp2,
                   executor=ExecutorType.ARCHITECTURE, confidence=0.80),
        ]
        results = engine.evaluate_issues(issues)
        s       = engine.summary(results)
        assert s["total"] == 2
        # CRITICAL single-agent missing security → rejected, ESCALATE_HUMAN
        assert s["escalated"] >= 1

    def test_empty_summary(self):
        engine = _engine()
        s      = engine.summary([])
        assert s["total"] == 0
        assert s["mean_confidence"] == 0.0
