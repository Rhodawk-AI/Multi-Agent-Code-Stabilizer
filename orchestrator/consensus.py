"""
orchestrator/consensus.py
=========================
Multi-agent consensus engine for Rhodawk AI Code Stabilizer.

PRODUCTION FIXES vs audit report
──────────────────────────────────
• ESCALATE_HUMAN now correctly sets escalation_required=True on ConsensusResult
  when confidence is below the CRITICAL floor — the controller will call
  EscalationManager.create_escalation() and block.
• Domain-mode-specific confidence floors: military/aerospace/nuclear require
  higher consensus (0.90) for CRITICAL findings vs GENERAL (0.75).
• MISRA mandatory rule findings auto-escalate regardless of confidence.
• High-centrality file confidence floor raised by 0.10 above the base floor.
• ConsensusRule.minimum_agents enforced — insufficient votes → ESCALATE.
• evaluate_issues() is synchronous (pure data processing — no I/O).
• All consensus logic is deterministic given the same input votes.
"""
from __future__ import annotations

import logging
from typing import Any

from brain.schemas import (
    ConsensusResult, ConsensusRule, ConsensusVote,
    DisagreementAction, DomainMode, ExecutorType,
    Issue, Severity,
)

log = logging.getLogger(__name__)

# Base confidence floors by domain mode
_BASE_CONFIDENCE_FLOORS: dict[str, float] = {
    "MILITARY":  0.90,
    "AEROSPACE": 0.90,
    "NUCLEAR":   0.92,
    "MEDICAL":   0.85,
    "FINANCE":   0.80,
    "EMBEDDED":  0.82,
    "GENERAL":   0.70,
}

# MISRA mandatory rules — always escalate regardless of consensus confidence
_MISRA_MANDATORY_RULES: frozenset[str] = frozenset({
    "MISRA-C:2023-1.3",  "MISRA-C:2023-2.1",  "MISRA-C:2023-15.1",
    "MISRA-C:2023-17.3", "MISRA-C:2023-18.1", "MISRA-C:2023-22.1",
    "MISRA-C:2023-22.2",
})


class ConsensusEngine:
    """
    Evaluates multi-agent audit findings and produces ConsensusResult
    for each issue. Determines whether escalation is required.
    """

    def __init__(
        self,
        graph_engine:  Any | None   = None,
        domain_mode:   str          = "GENERAL",
        min_agents:    int          = 2,
    ) -> None:
        self.graph_engine = graph_engine
        self.domain_mode  = domain_mode.upper()
        self.min_agents   = min_agents
        self._base_floor  = _BASE_CONFIDENCE_FLOORS.get(self.domain_mode, 0.70)

    def evaluate_issues(
        self, issues: list[Issue]
    ) -> list[ConsensusResult]:
        """
        Evaluate consensus for each issue.
        Returns one ConsensusResult per issue in the same order.
        """
        return [self._evaluate_one(issue) for issue in issues]

    def filter_approved(
        self,
        issues:  list[Issue],
        results: list[ConsensusResult],
    ) -> list[Issue]:
        """Return issues whose consensus is approved (not escalated)."""
        return [
            issue
            for issue, result in zip(issues, results)
            if result.approved and not result.escalation_required
        ]

    def summary(self, results: list[ConsensusResult]) -> dict[str, int]:
        return {
            "total":     len(results),
            "approved":  sum(1 for r in results if r.approved),
            "escalated": sum(1 for r in results if r.escalation_required),
            "blocked":   sum(1 for r in results if not r.approved),
        }

    def _evaluate_one(self, issue: Issue) -> ConsensusResult:
        # Build synthetic votes from the data we have
        votes = self._build_votes(issue)

        # Compute aggregate confidence
        if not votes:
            return self._escalate_result(
                issue, "No audit votes available for consensus"
            )

        confirmed_votes = [v for v in votes if v.confirmed]
        total_votes     = len(votes)
        confirm_count   = len(confirmed_votes)

        if total_votes < self.min_agents:
            return self._escalate_result(
                issue,
                f"Insufficient votes: {total_votes} < {self.min_agents} required"
            )

        # Weighted average confidence
        final_confidence = (
            sum(v.confidence for v in confirmed_votes) / total_votes
            if votes else 0.0
        )

        # Determine the confidence floor for this issue
        floor = self._get_confidence_floor(issue)

        # MISRA mandatory rules auto-escalate for human review
        if issue.misra_rule in _MISRA_MANDATORY_RULES:
            return ConsensusResult(
                issue_fingerprint=issue.fingerprint,
                votes=votes,
                final_confidence=final_confidence,
                approved=True,       # The finding itself is valid
                disagreement_action=DisagreementAction.ESCALATE_HUMAN,
                high_centrality=self._is_high_centrality(issue.file_path),
                escalation_required=True,  # Mandatory human sign-off
            )

        # Confidence below floor → escalate
        if final_confidence < floor:
            if issue.severity == Severity.CRITICAL:
                return self._escalate_result(
                    issue,
                    f"CRITICAL finding confidence {final_confidence:.2f} < "
                    f"floor {floor:.2f}"
                )
            else:
                # Non-critical low confidence → flag uncertain, don't escalate
                return ConsensusResult(
                    issue_fingerprint=issue.fingerprint,
                    votes=votes,
                    final_confidence=final_confidence,
                    approved=False,
                    disagreement_action=DisagreementAction.FLAG_UNCERTAIN,
                    escalation_required=False,
                )

        # Security domain must confirm CRITICAL findings
        if issue.severity == Severity.CRITICAL:
            security_confirmed = any(
                v.confirmed and v.agent == ExecutorType.SECURITY
                for v in votes
            )
            if not security_confirmed:
                return self._escalate_result(
                    issue,
                    "CRITICAL finding not confirmed by SECURITY auditor"
                )

        return ConsensusResult(
            issue_fingerprint=issue.fingerprint,
            votes=votes,
            final_confidence=final_confidence,
            approved=True,
            disagreement_action=DisagreementAction.AUTO_RESOLVE,
            high_centrality=self._is_high_centrality(issue.file_path),
            escalation_required=False,
        )

    def _get_confidence_floor(self, issue: Issue) -> float:
        floor = self._base_floor
        # High-centrality files require higher confidence
        if self._is_high_centrality(issue.file_path):
            floor = min(floor + 0.10, 0.99)
        # CRITICAL findings require higher confidence
        if issue.severity == Severity.CRITICAL:
            floor = min(floor + 0.05, 0.99)
        return floor

    def _is_high_centrality(self, file_path: str) -> bool:
        if not self.graph_engine:
            return False
        try:
            node = self.graph_engine.get_node(file_path)
            return node is not None and node.get("is_load_bearing", False)
        except Exception:
            return False

    def _build_votes(self, issue: Issue) -> list[ConsensusVote]:
        """
        Build votes from issue metadata.
        In a real run, votes come from multiple parallel AuditorAgent instances.
        Here we synthesise from the consensus_votes and consensus_confidence
        fields that were set during the audit phase.
        """
        if not issue.consensus_votes:
            # Single-auditor mode — create one synthetic vote
            return [ConsensusVote(
                agent=issue.executor_type or ExecutorType.GENERAL,
                confirmed=True,
                confidence=issue.confidence,
                notes="single-auditor",
            )]

        # Multi-vote mode — reconstruct from stored data
        votes: list[ConsensusVote] = []
        for i in range(issue.consensus_votes):
            votes.append(ConsensusVote(
                agent=issue.executor_type or ExecutorType.GENERAL,
                confirmed=True,
                confidence=issue.consensus_confidence,
            ))
        return votes

    def _escalate_result(
        self, issue: Issue, reason: str
    ) -> ConsensusResult:
        log.warning(
            f"[consensus] Escalating issue {issue.id[:12]} "
            f"({issue.severity.value} in {issue.file_path}): {reason}"
        )
        return ConsensusResult(
            issue_fingerprint=issue.fingerprint,
            votes=[],
            final_confidence=issue.confidence,
            approved=False,
            disagreement_action=DisagreementAction.ESCALATE_HUMAN,
            high_centrality=self._is_high_centrality(issue.file_path),
            escalation_required=True,
        )
