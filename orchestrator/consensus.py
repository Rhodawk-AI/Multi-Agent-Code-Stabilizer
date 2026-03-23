"""
orchestrator/consensus.py
=========================
Multi-agent consensus engine for Rhodawk AI Code Stabilizer.

TEST-02 FIX (audit report V2)
──────────────────────────────
Four bugs caused all consensus tests to fail:

  BUG A — evaluate_issues() returned one result *per issue* instead of one
    result *per fingerprint group*.  Tests pass multiple Issue objects that
    share a fingerprint to represent votes from different auditor domains.
    Fixed: issues are grouped by fingerprint before evaluation.

  BUG B — weighted_confidence, action, reason were not populated on
    ConsensusResult.  All three are now computed and set on every result.

  BUG C — _is_high_centrality() called graph_engine.get_node() but the
    documented (and tested) graph-engine interface exposes centrality_score().
    Fixed: uses centrality_score() with a 0.5 threshold.

  BUG D — confidence floor for high-centrality files used an additive +0.10
    instead of a multiplicative ×1.20 factor.  The test comment explicitly
    states "raised floor = 0.70 × 1.20 = 0.84".  Fixed.

Additional gaps closed:
  • DEFAULT_RULES exported so tests can import it.
  • ConsensusEngine accepts a rules= mapping to override per-severity defaults.
  • summary() now includes 'rejected' and 'mean_confidence' keys.
  • filter_approved() deduplicates by fingerprint and mutates issue metadata.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from brain.schemas import (
    ConsensusResult, ConsensusRule, ConsensusVote,
    DisagreementAction, ExecutorType,
    Issue, Severity,
)

log = logging.getLogger(__name__)

# ── Base confidence floors by domain mode ─────────────────────────────────────
_BASE_CONFIDENCE_FLOORS: dict[str, float] = {
    "MILITARY":  0.90,
    "AEROSPACE": 0.90,
    "NUCLEAR":   0.92,
    "MEDICAL":   0.85,
    "FINANCE":   0.80,
    "EMBEDDED":  0.82,
    "GENERAL":   0.70,
}

# ── MISRA mandatory rules — always escalate ───────────────────────────────────
_MISRA_MANDATORY_RULES: frozenset[str] = frozenset({
    "MISRA-C:2023-1.3",  "MISRA-C:2023-2.1",  "MISRA-C:2023-15.1",
    "MISRA-C:2023-17.3", "MISRA-C:2023-18.1", "MISRA-C:2023-22.1",
    "MISRA-C:2023-22.2",
})

# ── Domain weights by severity ────────────────────────────────────────────────
# Used when computing weighted_confidence across a fingerprint group.
# SECURITY findings are weighted 2× for CRITICAL (highest-stakes domain).
# ARCHITECTURE findings are weighted 1.5× for MAJOR (structural impact domain).
_DOMAIN_WEIGHTS: dict[Severity, dict[ExecutorType, float]] = {
    Severity.CRITICAL: {
        ExecutorType.SECURITY:     2.0,
        ExecutorType.ARCHITECTURE: 1.0,
        ExecutorType.STANDARDS:    1.0,
        ExecutorType.GENERAL:      1.0,
        ExecutorType.SYNTHESIS:    1.0,
    },
    Severity.MAJOR: {
        ExecutorType.SECURITY:     1.0,
        ExecutorType.ARCHITECTURE: 1.5,
        ExecutorType.STANDARDS:    1.0,
        ExecutorType.GENERAL:      1.0,
        ExecutorType.SYNTHESIS:    1.0,
    },
}

def _domain_weight(severity: Severity, executor: ExecutorType) -> float:
    """Return the vote weight for a domain at a given severity level."""
    weights = _DOMAIN_WEIGHTS.get(severity, {})
    return weights.get(executor, 1.0)


# ── Default consensus rules per severity ──────────────────────────────────────
DEFAULT_RULES: dict[Severity, ConsensusRule] = {
    Severity.INFO: ConsensusRule(
        minimum_agents=1,
        confidence_floor=0.0,
        disagreement_action=DisagreementAction.AUTO_RESOLVE,
    ),
    Severity.MINOR: ConsensusRule(
        minimum_agents=1,
        confidence_floor=0.50,
        disagreement_action=DisagreementAction.FLAG_UNCERTAIN,
    ),
    Severity.MAJOR: ConsensusRule(
        minimum_agents=2,
        confidence_floor=0.70,
        disagreement_action=DisagreementAction.FLAG_UNCERTAIN,
    ),
    Severity.CRITICAL: ConsensusRule(
        minimum_agents=2,
        confidence_floor=0.75,
        required_domains=[ExecutorType.SECURITY],
        disagreement_action=DisagreementAction.ESCALATE_HUMAN,
    ),
}

# High-centrality multiplier: raise the confidence floor by this factor.
_HIGH_CENTRALITY_MULTIPLIER: float = 1.20
# Centrality threshold: scores above this value are considered high-centrality.
_HIGH_CENTRALITY_THRESHOLD: float = 0.50


class ConsensusEngine:
    """
    Evaluates multi-agent audit findings and produces ConsensusResult
    for each fingerprint group. Determines whether escalation is required.
    """

    def __init__(
        self,
        graph_engine:  Any | None                    = None,
        domain_mode:   str                           = "GENERAL",
        min_agents:    int                           = 2,
        rules:         dict[Severity, ConsensusRule] | None = None,
    ) -> None:
        self.graph_engine = graph_engine
        self.domain_mode  = domain_mode.upper()
        self.min_agents   = min_agents
        self._base_floor  = _BASE_CONFIDENCE_FLOORS.get(self.domain_mode, 0.70)
        # Per-severity rules: caller-supplied rules override defaults.
        self._rules: dict[Severity, ConsensusRule] = {**DEFAULT_RULES, **(rules or {})}

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate_issues(
        self, issues: list[Issue]
    ) -> list[ConsensusResult]:
        """
        Evaluate consensus for each fingerprint group.

        Issues sharing the same fingerprint are votes from different auditor
        domains on the same finding.  Returns one ConsensusResult per unique
        fingerprint, in the order they first appear in *issues*.
        """
        # Group by fingerprint, preserving insertion order.
        groups: dict[str, list[Issue]] = defaultdict(list)
        order:  list[str]              = []
        for issue in issues:
            fp = issue.fingerprint or issue.id  # fall back to id if no fingerprint
            if fp not in groups:
                order.append(fp)
            groups[fp].append(issue)

        return [self._evaluate_group(groups[fp]) for fp in order]

    def filter_approved(
        self,
        issues:  list[Issue],
        results: list[ConsensusResult],
    ) -> list[Issue]:
        """
        Return one representative Issue per approved fingerprint group.

        Also mutates each issue's consensus_votes and consensus_confidence
        fields so callers can persist the consensus decision without a
        separate lookup.
        """
        # Build a fingerprint → result map for fast lookup.
        result_map: dict[str, ConsensusResult] = {r.issue_fingerprint: r for r in results}

        approved_fps: set[str]   = set()
        approved:     list[Issue] = []

        for issue in issues:
            fp = issue.fingerprint or issue.id
            result = result_map.get(fp)
            if result is None:
                continue

            # Always mutate metadata so callers can read updated fields.
            issue.consensus_votes      = len(result.votes)
            issue.consensus_confidence = result.weighted_confidence or result.final_confidence

            if result.approved and not result.escalation_required:
                if fp not in approved_fps:
                    approved_fps.add(fp)
                    approved.append(issue)

        return approved

    def summary(self, results: list[ConsensusResult]) -> dict:
        total      = len(results)
        approved   = sum(1 for r in results if r.approved and not r.escalation_required)
        escalated  = sum(1 for r in results if r.escalation_required)
        rejected   = total - approved - escalated
        mean_conf  = (
            sum(r.weighted_confidence for r in results) / total
            if total else 0.0
        )
        return {
            "total":           total,
            "approved":        approved,
            "rejected":        rejected,
            "escalated":       escalated,
            "blocked":         total - approved,       # legacy key — kept for compat
            "mean_confidence": round(mean_conf, 4),
        }

    # ── Internal evaluation ───────────────────────────────────────────────────

    def _evaluate_group(self, group: list[Issue]) -> ConsensusResult:
        """
        Evaluate a group of issues that share the same fingerprint.
        Each issue in the group represents one auditor-domain vote.
        """
        # Use the first issue as the representative for severity, file_path, etc.
        rep = group[0]

        # SYNTHESIS issues route through a dedicated path (no vote grouping needed).
        if rep.executor_type == ExecutorType.SYNTHESIS:
            return self._evaluate_synthesis(rep)

        # Build one vote per issue in the group.
        votes = [
            ConsensusVote(
                agent=issue.executor_type or ExecutorType.GENERAL,
                confirmed=True,
                confidence=issue.consensus_confidence if issue.consensus_confidence > 0.0
                           else issue.confidence,
                notes=f"auditor:{issue.executor_type.value if issue.executor_type else 'GENERAL'}",
            )
            for issue in group
        ]

        total_votes = len(votes)

        # Retrieve the rule for this severity.
        rule = self._rules.get(rep.severity, DEFAULT_RULES.get(rep.severity, ConsensusRule()))

        # ── Minimum agents check ──────────────────────────────────────────────
        if total_votes < rule.minimum_agents:
            reason = (
                f"Insufficient votes: {total_votes} < {rule.minimum_agents} required "
                f"for {rep.severity.value} findings"
            )
            return self._make_result(
                rep, votes,
                approved=False,
                weighted_confidence=votes[0].confidence if votes else 0.0,
                reason=reason,
                action=rule.disagreement_action,
                escalation_required=(rule.disagreement_action == DisagreementAction.ESCALATE_HUMAN),
            )

        # ── Weighted confidence ───────────────────────────────────────────────
        weighted_confidence = self._compute_weighted_confidence(group, rep.severity)

        # ── Confidence floor (with high-centrality uplift) ────────────────────
        floor = self._get_confidence_floor(rep, rule)

        # ── MISRA mandatory rules — always escalate ───────────────────────────
        if getattr(rep, "misra_rule", None) in _MISRA_MANDATORY_RULES:
            return self._make_result(
                rep, votes,
                approved=True,
                weighted_confidence=weighted_confidence,
                reason=f"MISRA mandatory rule {rep.misra_rule} requires human sign-off",
                action=DisagreementAction.ESCALATE_HUMAN,
                escalation_required=True,
            )

        # ── Confidence below floor ────────────────────────────────────────────
        if weighted_confidence < floor:
            reason = (
                f"{rep.severity.value} finding weighted confidence "
                f"{weighted_confidence:.2f} < floor {floor:.2f}"
            )
            if rep.severity == Severity.CRITICAL:
                return self._make_result(
                    rep, votes,
                    approved=False,
                    weighted_confidence=weighted_confidence,
                    reason=reason,
                    action=DisagreementAction.ESCALATE_HUMAN,
                    escalation_required=True,
                )
            else:
                return self._make_result(
                    rep, votes,
                    approved=False,
                    weighted_confidence=weighted_confidence,
                    reason=reason,
                    action=rule.disagreement_action,
                    escalation_required=False,
                )

        # ── Required-domain check ─────────────────────────────────────────────
        if rule.required_domains:
            present_domains = {
                issue.executor_type for issue in group if issue.executor_type
            }
            missing = [
                d.value for d in rule.required_domains
                if d not in present_domains
            ]
            if missing:
                reason = (
                    f"{rep.severity.value} finding requires confirmation from "
                    f"{', '.join(missing)} domain(s)"
                )
                return self._make_result(
                    rep, votes,
                    approved=False,
                    weighted_confidence=weighted_confidence,
                    reason=reason,
                    action=DisagreementAction.ESCALATE_HUMAN,
                    escalation_required=True,
                )

        # ── Approved ──────────────────────────────────────────────────────────
        return self._make_result(
            rep, votes,
            approved=True,
            weighted_confidence=weighted_confidence,
            reason="",
            action=DisagreementAction.AUTO_RESOLVE,
            escalation_required=False,
        )

    # ── SYNTHESIS path ────────────────────────────────────────────────────────

    def _evaluate_synthesis(self, issue: Issue) -> ConsensusResult:
        """
        Dedicated consensus path for compound findings (executor_type=SYNTHESIS).
        Bypasses min_agents and required_domain checks — see module docstring.
        """
        final_confidence = issue.confidence if issue.confidence > 0.0 else 0.9
        rule  = self._rules.get(issue.severity, ConsensusRule())
        floor = self._get_confidence_floor(issue, rule)

        votes = [ConsensusVote(
            agent=ExecutorType.SYNTHESIS,
            confirmed=True,
            confidence=final_confidence,
            notes="compound-finding: cross-domain synthesis pass",
        )]

        if getattr(issue, "misra_rule", None) in _MISRA_MANDATORY_RULES:
            return self._make_result(
                issue, votes,
                approved=True,
                weighted_confidence=final_confidence,
                reason=f"MISRA mandatory rule {issue.misra_rule}",
                action=DisagreementAction.ESCALATE_HUMAN,
                escalation_required=True,
            )

        if final_confidence < floor:
            return self._make_result(
                issue, votes,
                approved=False,
                weighted_confidence=final_confidence,
                reason=f"Compound finding confidence {final_confidence:.2f} < floor {floor:.2f}",
                action=DisagreementAction.ESCALATE_HUMAN,
                escalation_required=True,
            )

        return self._make_result(
            issue, votes,
            approved=True,
            weighted_confidence=final_confidence,
            reason="",
            action=DisagreementAction.AUTO_RESOLVE,
            escalation_required=False,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_weighted_confidence(
        self, group: list[Issue], severity: Severity
    ) -> float:
        """
        Compute domain-weighted average confidence across the fingerprint group.
        Each issue's executor_type determines its weight.
        """
        total_weight = 0.0
        weighted_sum = 0.0
        for issue in group:
            w = _domain_weight(severity, issue.executor_type or ExecutorType.GENERAL)
            conf = (
                issue.consensus_confidence if issue.consensus_confidence > 0.0
                else issue.confidence
            )
            weighted_sum  += w * conf
            total_weight  += w
        if total_weight == 0.0:
            return 0.0
        return max(0.0, min(1.0, weighted_sum / total_weight))

    def _get_confidence_floor(
        self, issue: Issue, rule: ConsensusRule | None = None
    ) -> float:
        """
        Return the effective confidence floor for this issue.

        Base floor comes from rule.confidence_floor if a rule is provided,
        otherwise from the domain-mode base floor.
        High-centrality files raise the floor multiplicatively by 1.20.
        """
        if rule is not None:
            floor = rule.confidence_floor
        else:
            floor = self._base_floor
            if issue.severity == Severity.CRITICAL:
                floor = min(floor + 0.05, 0.99)

        if self._is_high_centrality(issue.file_path):
            floor = min(floor * _HIGH_CENTRALITY_MULTIPLIER, 0.99)

        return floor

    def _is_high_centrality(self, file_path: str) -> bool:
        """Return True if the file is a high-centrality hub in the call graph."""
        if not self.graph_engine:
            return False
        try:
            # Use centrality_score() — the documented graph-engine interface.
            score = self.graph_engine.centrality_score(file_path)
            return score >= _HIGH_CENTRALITY_THRESHOLD
        except Exception:
            return False

    def _make_result(
        self,
        rep:                Issue,
        votes:              list[ConsensusVote],
        *,
        approved:           bool,
        weighted_confidence: float,
        reason:             str,
        action:             DisagreementAction,
        escalation_required: bool,
    ) -> ConsensusResult:
        """Construct a fully-populated ConsensusResult."""
        high = self._is_high_centrality(rep.file_path)
        log.debug(
            "[consensus] fp=%s sev=%s approved=%s wconf=%.2f reason=%s",
            rep.fingerprint[:12] if rep.fingerprint else "?",
            rep.severity.value,
            approved,
            weighted_confidence,
            reason or "(approved)",
        )
        return ConsensusResult(
            issue_fingerprint   =rep.fingerprint,
            votes               =votes,
            final_confidence    =weighted_confidence,
            weighted_confidence =weighted_confidence,
            approved            =approved,
            disagreement_action =action,
            action              =action,
            reason              =reason,
            high_centrality     =high,
            escalation_required =escalation_required,
        )


# ── GAP 5: BoBN Candidate Ranking ─────────────────────────────────────────────

def rank_candidates(candidates: list[dict]) -> list[dict]:
    """
    Rank BoBN patch candidates by composite score.

    composite = 0.6 × test_score
              + 0.3 × (1 − attack_confidence)
              + 0.1 × minimality_score

    Returns candidates sorted descending by composite_score.
    """
    for c in candidates:
        test_score  = float(c.get("test_score",         0.0))
        attack_conf = float(c.get("attack_confidence",  0.5))
        minimality  = float(c.get("minimality_score",   0.5))
        c["composite_score"] = (
            0.6 * test_score
            + 0.3 * (1.0 - attack_conf)
            + 0.1 * minimality
        )

    ranked = sorted(candidates, key=lambda c: c["composite_score"], reverse=True)
    for i, c in enumerate(ranked):
        c["rank"] = i + 1
        log.debug(
            "[consensus] BoBN rank=%d id=%s composite=%.3f test=%.2f robust=%.2f minimal=%.2f",
            i + 1, c.get("id", "?"), c["composite_score"],
            c.get("test_score", 0), 1 - c.get("attack_confidence", 0.5),
            c.get("minimality_score", 0.5),
        )
    return ranked
