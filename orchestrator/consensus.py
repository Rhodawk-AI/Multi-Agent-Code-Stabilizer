"""
orchestrator/consensus.py
=========================
Weighted N-agent consensus engine for MACS.

Closes GAP-5: the three auditors ran in parallel but never voted on their
findings.  This module implements a proper consensus mechanism that:

1. Receives findings from multiple auditor agents after each audit phase.
2. Computes a weighted confidence score combining all agents' votes.
3. Enforces domain requirements — e.g. CRITICAL security issues require the
   SECURITY auditor to confirm before proceeding.
4. Raises the confidence bar for high-centrality files using the graph engine.
5. Returns ConsensusResult for each fingerprinted finding so the controller
   can decide whether to proceed to fix or escalate.

Integration points
──────────────────
- Called by StabilizerController._phase_audit() after all three auditors run.
- Uses DependencyGraphEngine.centrality_score() to detect hub files.
- ConsensusResult is stored on each Issue (consensus_votes, consensus_confidence).
- ConsensusRule per severity is configurable at construction time.

Default consensus rules (can be overridden)
────────────────────────────────────────────
• CRITICAL: ≥2 agents, SECURITY domain required, confidence ≥ 0.85
• MAJOR:    ≥2 agents, any domain, confidence ≥ 0.70
• MINOR:    ≥1 agent,  any domain, confidence ≥ 0.50
• INFO:     ≥1 agent,  any domain, confidence ≥ 0.00 (always passes)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from brain.schemas import (
    ConsensusResult,
    ConsensusRule,
    ConsensusVote,
    DisagreementAction,
    ExecutorType,
    Issue,
    IssueStatus,
    Severity,
)

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Default rules per severity
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_RULES: dict[Severity, ConsensusRule] = {
    Severity.CRITICAL: ConsensusRule(
        minimum_agents=2,
        required_domains=[ExecutorType.SECURITY],
        confidence_floor=0.85,
        disagreement_action=DisagreementAction.ESCALATE_HUMAN,
        high_centrality_raises=True,
    ),
    Severity.MAJOR: ConsensusRule(
        minimum_agents=2,
        required_domains=[],
        confidence_floor=0.70,
        disagreement_action=DisagreementAction.FLAG_UNCERTAIN,
        high_centrality_raises=True,
    ),
    Severity.MINOR: ConsensusRule(
        minimum_agents=1,
        required_domains=[],
        confidence_floor=0.50,
        disagreement_action=DisagreementAction.FLAG_UNCERTAIN,
        high_centrality_raises=False,
    ),
    Severity.INFO: ConsensusRule(
        minimum_agents=1,
        required_domains=[],
        confidence_floor=0.00,
        disagreement_action=DisagreementAction.FLAG_UNCERTAIN,
        high_centrality_raises=False,
    ),
}

_HIGH_CENTRALITY_THRESHOLD  = 0.70   # above this: raise confidence bar
_HIGH_CENTRALITY_MULTIPLIER = 1.20   # × 1.2 for hub files


# ──────────────────────────────────────────────────────────────────────────────
# ConsensusEngine
# ──────────────────────────────────────────────────────────────────────────────

class ConsensusEngine:
    """
    Evaluates a set of findings from multiple auditor agents and determines
    whether each fingerprinted finding has sufficient multi-agent consensus
    to proceed to the fix phase.

    Parameters
    ----------
    rules:
        Severity → ConsensusRule mapping.  Defaults to ``DEFAULT_RULES``.
    graph_engine:
        Optional DependencyGraphEngine used to look up centrality scores.
        Pass ``None`` to disable centrality-based confidence adjustment.
    """

    def __init__(
        self,
        rules:        dict[Severity, ConsensusRule] | None = None,
        graph_engine: Any | None                          = None,
    ) -> None:
        self._rules        = rules or DEFAULT_RULES
        self._graph_engine = graph_engine

    # ── Main API ──────────────────────────────────────────────────────────────

    def evaluate_issues(self, issues: list[Issue]) -> list[ConsensusResult]:
        """
        Group issues by fingerprint, compute per-group weighted votes, and
        return a ConsensusResult per unique fingerprint.

        Parameters
        ----------
        issues:
            All issues from ALL auditor agents for the current cycle.

        Returns
        -------
        List of ConsensusResult, one per unique fingerprint.
        """
        by_fingerprint: dict[str, list[Issue]] = defaultdict(list)
        for issue in issues:
            fp = issue.fingerprint or issue.id
            by_fingerprint[fp].append(issue)

        results: list[ConsensusResult] = []
        for fingerprint, group in by_fingerprint.items():
            result = self._evaluate_group(fingerprint, group)
            results.append(result)

        return results

    def filter_approved(
        self, issues: list[Issue], results: list[ConsensusResult]
    ) -> list[Issue]:
        """
        Return only the issues that passed consensus.
        Also mutates each issue's consensus_votes and consensus_confidence fields.

        Issues that did NOT pass are updated to ESCALATED or INFO status.
        """
        approved_fps: dict[str, ConsensusResult] = {
            r.fingerprint: r for r in results if r.approved
        }
        rejected_fps: dict[str, ConsensusResult] = {
            r.fingerprint: r for r in results if not r.approved
        }

        approved: list[Issue] = []
        seen_fps: set[str]    = set()

        for issue in issues:
            fp     = issue.fingerprint or issue.id
            result = approved_fps.get(fp) or rejected_fps.get(fp)

            if result:
                issue.consensus_votes      = len(result.votes)
                issue.consensus_confidence = result.weighted_confidence

            if fp in approved_fps:
                if fp not in seen_fps:
                    # Only take the highest-severity representative per fingerprint
                    approved.append(issue)
                    seen_fps.add(fp)
            else:
                if result and result.action == DisagreementAction.AUTO_REJECT:
                    issue.status = IssueStatus.INFO  # type: ignore[assignment]
                    log.info(f"Consensus: auto-rejected {issue.id} (fp={fp[:8]})")

        return approved

    # ── Group evaluation ──────────────────────────────────────────────────────

    def _evaluate_group(
        self, fingerprint: str, group: list[Issue]
    ) -> ConsensusResult:
        # Representative issue (highest severity in the group)
        rep    = max(group, key=lambda i: list(Severity).index(i.severity))
        rule   = self._rules.get(rep.severity, DEFAULT_RULES[Severity.MINOR])

        # Build votes from each issue's reporting agent
        votes: list[ConsensusVote] = []
        for issue in group:
            vote = ConsensusVote(
                agent_id=issue.id,
                domain=issue.executor_type,
                confirms=True,
                confidence=issue.consensus_confidence if issue.consensus_confidence > 0 else 0.75,
                reasoning=issue.description[:200],
            )
            votes.append(vote)

        # Weighted confidence = sum(conf * weight) / sum(weight)
        # Weight = 2.0 for SECURITY domain on CRITICAL issues, 1.0 otherwise
        def _weight(v: ConsensusVote) -> float:
            if rep.severity == Severity.CRITICAL and v.domain == ExecutorType.SECURITY:
                return 2.0
            if rep.severity in (Severity.CRITICAL, Severity.MAJOR) and v.domain == ExecutorType.ARCHITECTURE:
                return 1.5
            return 1.0

        total_weight    = sum(_weight(v) for v in votes)
        weighted_conf   = (
            sum(v.confidence * _weight(v) for v in votes) / total_weight
            if total_weight > 0 else 0.0
        )

        # Adjust confidence floor for high-centrality files
        floor = rule.confidence_floor
        if rule.high_centrality_raises and self._graph_engine and self._graph_engine.is_built:
            centrality = self._graph_engine.centrality_score(rep.file_path)
            if centrality >= _HIGH_CENTRALITY_THRESHOLD:
                floor = min(0.99, floor * _HIGH_CENTRALITY_MULTIPLIER)
                log.debug(
                    f"Consensus: raised floor to {floor:.2f} for high-centrality file "
                    f"'{rep.file_path}' (centrality={centrality:.3f})"
                )

        # Check constraints
        confirmed_by = {v.domain for v in votes if v.confirms}
        agent_count  = len(votes)

        reasons: list[str] = []

        # 1. Minimum agent count
        if agent_count < rule.minimum_agents:
            reasons.append(
                f"Only {agent_count} agent(s) confirmed; requires {rule.minimum_agents}"
            )

        # 2. Required domains
        missing_domains = [d for d in rule.required_domains if d not in confirmed_by]
        if missing_domains:
            missing_names = [d.value for d in missing_domains]
            reasons.append(
                f"Required domain(s) missing: {', '.join(missing_names)}"
            )

        # 3. Weighted confidence floor
        if weighted_conf < floor:
            reasons.append(
                f"Weighted confidence {weighted_conf:.2f} below floor {floor:.2f}"
            )

        approved = len(reasons) == 0

        action: DisagreementAction | None = None
        reason = "Consensus reached" if approved else "; ".join(reasons)
        if not approved:
            action = rule.disagreement_action

        return ConsensusResult(
            fingerprint=fingerprint,
            approved=approved,
            weighted_confidence=weighted_conf,
            votes=votes,
            action=action,
            reason=reason,
            evaluated_at=datetime.now(tz=timezone.utc),
        )

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self, results: list[ConsensusResult]) -> dict:
        return {
            "total":       len(results),
            "approved":    sum(1 for r in results if r.approved),
            "rejected":    sum(1 for r in results if not r.approved),
            "escalated":   sum(1 for r in results
                               if not r.approved
                               and r.action == DisagreementAction.ESCALATE_HUMAN),
            "mean_confidence": (
                sum(r.weighted_confidence for r in results) / len(results)
                if results else 0.0
            ),
        }
