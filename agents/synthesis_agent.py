"""
agents/synthesis_agent.py
=========================
Synthesis Agent for Rhodawk AI Code Stabilizer — Gap 2 fix.

GAP 2 PROBLEM
─────────────
Three AuditorAgent instances run in parallel — SECURITY, ARCHITECTURE,
STANDARDS — all calling the same LLM with different prompts.  On large
codebases this produces:

  • Duplicate findings (same bug reported 3 ways in different languages)
  • Missed findings (each auditor stays in its lane, never combining signals)
  • No cross-auditor reasoning (the security auditor never learns what the
    architecture auditor found, and vice versa)

THE FIX
───────
SynthesisAgent receives ALL three auditor outputs simultaneously, then:

  1. DEDUPLICATION — fingerprint + semantic similarity pass removes
     findings that describe the same root defect, keeping the highest-
     confidence/highest-severity representative.

  2. CROSS-DOMAIN COMPOUND FINDING DETECTION — the synthesis LLM is
     given all three auditor outputs together and asked to identify findings
     that span domain boundaries.  A race condition that is also a privilege
     escalation is a CompoundFinding worth 10 separate single-domain findings.
     No commercial tool finds these.  This one does.

  3. SEVERITY ESCALATION — compound findings are promoted to the maximum
     severity of their constituent findings, and a new CompoundFinding record
     is persisted with the contributing issue IDs.

  4. INDEPENDENT MODEL — synthesis uses config.synthesis_model (defaults to
     the critical_fix_model, a different family than the auditors).  Seeing
     the combined output with fresh eyes is the structural advantage.

OUTPUT
──────
Returns (deduplicated_issues, compound_findings) where:
  • deduplicated_issues  — the original flat list with duplicates removed
  • compound_findings    — new CompoundFinding objects, each backed by an
                           Issue in storage with ExecutorType.SYNTHESIS

PRODUCTION BEHAVIOUR
────────────────────
• Falls back gracefully if LLM call fails — returns original issues unchanged
  so the pipeline never silently loses findings.
• Temperature=0.0 for dedup decisions (deterministic, auditable).
• Temperature=0.1 for compound finding generation (slight creativity needed).
• All CompoundFinding Issues are persisted to storage before return.
• Audit trail entry written on completion.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from pydantic import BaseModel, Field

from agents.base import AgentConfig, BaseAgent
from brain.schemas import (
    CompoundFinding,
    CompoundFindingCategory,
    DomainMode,
    ExecutorType,
    Issue,
    IssueStatus,
    MilStd882eCategory,
    SEVERITY_TO_MIL882E,
    Severity,
    SynthesisReport,
)
from brain.storage import BrainStorage

log = logging.getLogger(__name__)


# ── Response models for LLM structured calls ─────────────────────────────────

class DuplicateCluster(BaseModel):
    """LLM identifies a group of issue indices that describe the same defect."""
    indices:          list[int] = Field(default_factory=list)
    canonical_index:  int       = 0   # which one to keep
    reason:           str       = ""


class DeduplicationResponse(BaseModel):
    duplicate_clusters: list[DuplicateCluster] = Field(default_factory=list)
    total_duplicates_removed: int = 0


class RawCompoundFinding(BaseModel):
    """LLM-generated cross-domain compound finding."""
    title:              str       = ""
    description:        str       = ""
    severity:           str       = "CRITICAL"
    contributing_indices: list[int] = Field(default_factory=list)
    domains_involved:   list[str] = Field(default_factory=list)
    category:           str       = "SECURITY_ARCHITECTURE"
    # How much more severe is this compound issue than any single finding?
    amplification_factor: float   = Field(ge=1.0, le=10.0, default=2.0)
    fix_complexity:     str       = "HIGH"   # LOW | MEDIUM | HIGH | ARCHITECTURAL
    rationale:          str       = ""
    cwe_id:             str       = ""
    misra_rule:         str       = ""
    cert_rule:          str       = ""


class CompoundFindingResponse(BaseModel):
    compound_findings: list[RawCompoundFinding] = Field(default_factory=list)
    synthesis_summary: str = ""


# ── Synthesis Agent ───────────────────────────────────────────────────────────

class SynthesisAgent(BaseAgent):
    """
    Receives all auditor outputs, deduplicates, and generates cross-domain
    compound findings.

    This is the Gap 2 structural fix: instead of three isolated auditors
    each staying in their lane, a synthesis pass reasons across all findings
    simultaneously to identify compound vulnerabilities that no single-domain
    scan can detect.
    """

    agent_type = ExecutorType.SYNTHESIS

    # Minimum semantic similarity to classify two findings as duplicates
    # when fingerprints differ (0.0 = never deduplicate by semantics only,
    # 1.0 = deduplicate everything).
    _SEMANTIC_DUP_THRESHOLD: float = 0.85

    # Maximum findings per LLM dedup call (avoids context overflow)
    _DEDUP_BATCH_SIZE: int = 60

    # Overlap between adjacent dedup batches.  When len(issues) > _DEDUP_BATCH_SIZE
    # the list is split into windows of size _DEDUP_BATCH_SIZE that each share
    # _DEDUP_BATCH_OVERLAP issues with the previous window.  Without overlap a
    # duplicate pair that straddles a batch boundary would never be seen by the
    # same LLM call and would survive deduplication.  The overlap guarantees that
    # every pair of adjacent issues appears in at least one batch together as long
    # as their index distance is < _DEDUP_BATCH_SIZE.
    _DEDUP_BATCH_OVERLAP: int = 15

    # Minimum number of issues from ≥2 different domains required before
    # attempting compound finding detection.
    #
    # BUG FIX: was 3, which silently skipped the canonical two-issue compound
    # case — one SECURITY finding + one ARCHITECTURE finding — that the gap
    # spec explicitly demonstrates (auth_bypass + async_handler race =
    # privilege escalation).  Exactly 2 issues from 2 distinct domains is the
    # minimum meaningful cross-domain signal; anything lower is impossible
    # (the ``len(domains_present) < 2`` guard above already enforces ≥2 domains).
    _COMPOUND_MIN_CROSS_DOMAIN_ISSUES: int = 2

    def __init__(
        self,
        storage:          BrainStorage,
        run_id:           str,
        config:           AgentConfig | None = None,
        mcp_manager:      Any | None         = None,
        domain_mode:      DomainMode         = DomainMode.GENERAL,
        repo_root:        Any | None         = None,
        synthesis_model:  str                = "",
        # Gap 2 knobs
        dedup_enabled:    bool               = True,
        compound_enabled: bool               = True,
        max_compound_findings: int           = 20,
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.domain_mode           = domain_mode
        self.repo_root             = repo_root
        self._synthesis_model      = synthesis_model or (
            config.critical_fix_model if config else ""
        )
        self.dedup_enabled         = dedup_enabled
        self.compound_enabled      = compound_enabled
        self.max_compound_findings = max_compound_findings

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(
        self,
        issues: list[Issue],
        **kwargs: Any,
    ) -> tuple[list[Issue], list[CompoundFinding]]:
        """
        Run the synthesis pipeline.

        Parameters
        ----------
        issues:
            The combined raw findings from all auditor agents (SECURITY +
            ARCHITECTURE + STANDARDS).  May contain duplicates.

        Returns
        -------
        (deduplicated_issues, compound_findings)
            deduplicated_issues  — flat list, duplicates removed
            compound_findings    — new cross-domain compound issues
        """
        if not issues:
            return [], []

        self.log.info(
            f"[synthesis] Starting synthesis over {len(issues)} raw findings "
            f"from {len({i.executor_type for i in issues})} auditor domains"
        )

        # ── Step 1: Fast fingerprint-based deduplication ──────────────────────
        deduped = self._dedup_by_fingerprint(issues)
        self.log.info(
            f"[synthesis] Fingerprint dedup: "
            f"{len(issues)} → {len(deduped)} "
            f"(-{len(issues) - len(deduped)} exact duplicates)"
        )

        # ── Step 2: Semantic deduplication via LLM (batched) ──────────────────
        if self.dedup_enabled and len(deduped) > 1:
            try:
                deduped = await self._dedup_semantic(deduped)
                self.log.info(
                    f"[synthesis] Semantic dedup result: {len(deduped)} findings"
                )
            except Exception as exc:
                self.log.warning(
                    f"[synthesis] Semantic dedup failed (using fingerprint result): {exc}"
                )

        # ── Step 3: Cross-domain compound finding detection ───────────────────
        compound_findings: list[CompoundFinding] = []
        if self.compound_enabled:
            try:
                compound_findings = await self._detect_compound_findings(deduped)
                self.log.info(
                    f"[synthesis] Detected {len(compound_findings)} compound findings"
                )
            except Exception as exc:
                self.log.warning(
                    f"[synthesis] Compound detection failed: {exc}"
                )

        # ── Step 4: Persist compound findings as Issues ───────────────────────
        compound_issues: list[Issue] = []
        for cf in compound_findings:
            issue = self._compound_to_issue(cf, deduped)
            await self.storage.upsert_issue(issue)
            compound_issues.append(issue)
            cf.synthesized_issue_id = issue.id

        # ── Step 5: Audit trail ───────────────────────────────────────────────
        await self._write_audit_trail(
            raw_count=len(issues),
            deduped_count=len(deduped),
            compound_count=len(compound_findings),
        )

        # Combine deduplicated issues + compound synthesis issues
        all_issues = deduped + compound_issues
        self.log.info(
            f"[synthesis] Complete: {len(all_issues)} final issues "
            f"({len(deduped)} deduped + {len(compound_issues)} compound)"
        )
        return deduped, compound_findings

    # ── Step 1: Fingerprint deduplication ─────────────────────────────────────

    def _dedup_by_fingerprint(self, issues: list[Issue]) -> list[Issue]:
        """
        Remove exact duplicate findings by fingerprint hash.

        When two issues share the same fingerprint (file + lines + description
        hash), keep the one with:
          1. Higher severity
          2. Higher confidence (tie-break)
          3. SECURITY domain (tie-break — security findings take precedence)
        """
        seen: dict[str, Issue] = {}
        for issue in issues:
            fp = issue.fingerprint or self._compute_fingerprint(issue)
            if fp not in seen:
                seen[fp] = issue
            else:
                existing = seen[fp]
                # Keep the "better" finding
                if self._issue_priority(issue) > self._issue_priority(existing):
                    seen[fp] = issue
        return list(seen.values())

    @staticmethod
    def _compute_fingerprint(issue: Issue) -> str:
        """Compute a fingerprint for issues that lack one."""
        raw = f"{issue.file_path}:{issue.line_start}:{issue.line_end}:{issue.description[:120]}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _issue_priority(issue: Issue) -> tuple[int, float, int]:
        """Priority tuple for duplicate resolution: (severity_rank, confidence, domain_rank)."""
        sev_rank = {
            Severity.CRITICAL: 4,
            Severity.MAJOR:    3,
            Severity.MINOR:    2,
            Severity.INFO:     1,
        }.get(issue.severity, 0)
        domain_rank = {
            ExecutorType.SECURITY:     3,
            ExecutorType.STANDARDS:    2,
            ExecutorType.ARCHITECTURE: 1,
        }.get(issue.executor_type, 0)
        return (sev_rank, issue.confidence, domain_rank)

    # ── Step 2: Semantic deduplication ────────────────────────────────────────

    async def _dedup_semantic(self, issues: list[Issue]) -> list[Issue]:
        """
        Use LLM to identify semantically duplicate findings that share the same
        root defect but are described differently by different auditors.

        Processes in batches of _DEDUP_BATCH_SIZE to avoid context overflow.
        """
        if len(issues) <= 1:
            return issues

        # For very large lists, batch the deduplication
        if len(issues) > self._DEDUP_BATCH_SIZE:
            return await self._dedup_semantic_batched(issues)

        summary_lines = self._build_issue_summary(issues)
        prompt = (
            f"## Code Audit Findings — Duplicate Detection\n\n"
            f"The following {len(issues)} findings were generated by three independent "
            f"auditors (SECURITY, ARCHITECTURE, STANDARDS) scanning the same codebase.\n\n"
            f"Identify groups of findings that describe the **same underlying defect** "
            f"from different angles.  Two findings are duplicates if fixing one would "
            f"fix the other.\n\n"
            f"## Findings (0-indexed)\n{summary_lines}\n\n"
            f"## Instructions\n"
            f"- For each duplicate cluster, identify the canonical (best) finding to keep.\n"
            f"- Prefer: higher severity > higher confidence > SECURITY domain.\n"
            f"- Only cluster findings that describe EXACTLY the same root defect.\n"
            f"- Do NOT cluster findings that are merely related or in the same file.\n"
            f"- If no duplicates exist, return an empty duplicate_clusters list.\n\n"
            f"Return JSON with duplicate_clusters and total_duplicates_removed."
        )

        resp = await self.call_llm_structured_deterministic(
            prompt=prompt,
            response_model=DeduplicationResponse,
            system=(
                "You are a senior code auditor identifying duplicate findings "
                "across multiple auditor agents. Be conservative — only cluster "
                "findings you are certain describe the same root defect."
            ),
            model_override=self._synthesis_model or self.config.triage_model,
        )

        return self._apply_dedup_response(issues, resp)

    async def _dedup_semantic_batched(self, issues: list[Issue]) -> list[Issue]:\n        \"\"\"\n        For lists larger than _DEDUP_BATCH_SIZE, deduplicate in overlapping\n        windows to prevent cross-batch duplicate pairs from surviving.\n\n        CORRECTNESS FIX\n        ---------------\n        The original implementation processed non-overlapping sequential windows\n        (``result[offset:offset + batch_size]``) and incremented ``offset`` by\n        the deduped window length.  A duplicate pair whose two members straddle a\n        batch boundary — issue A in the last position of window N, issue B in the\n        first position of window N+1 — would never appear in the same LLM call\n        and would survive deduplication.  This is a silent correctness failure\n        that compounds on large codebases (thousands of findings across three\n        auditor domains).\n\n        Fix strategy\n        ------------\n        1. **Cross-batch structural pre-pass**: before any LLM call, remove\n           exact structural duplicates (same file + line range + description\n           prefix) regardless of batch position.  This catches the majority of\n           cross-batch duplicates cheaply without an LLM round-trip.\n\n        2. **Overlapping windows**: each window of size _DEDUP_BATCH_SIZE overlaps\n           the previous window by _DEDUP_BATCH_OVERLAP issues.  Every pair of\n           adjacent issues therefore appears in at least one shared window as long\n           as their index distance is ≤ _DEDUP_BATCH_SIZE - _DEDUP_BATCH_OVERLAP.\n           Indices removed in one window are excluded from all subsequent windows\n           so the overlap does not re-introduce already-removed issues.\n        \"\"\"\n        if len(issues) <= self._DEDUP_BATCH_SIZE:\n            return await self._dedup_semantic(issues)\n\n        # ── Step 1: Cross-batch structural pre-pass ───────────────────────────\n        # Catch duplicates that would straddle batch boundaries using a\n        # deterministic structural key: (file_path, line_start, description_64).\n        # This is cheaper than the LLM dedup pass and catches the common case\n        # where the same finding is emitted by two different auditors in the\n        # same file at the same location.\n        struct_seen: dict[str, int] = {}   # key → first occurrence index\n        struct_remove: set[int] = set()\n        for idx, issue in enumerate(issues):\n            struct_key = (\n                f\"{issue.file_path}:{issue.line_start}:{issue.line_end}:\"\n                f\"{issue.description[:64].lower().strip()}\"\n            )\n            if struct_key in struct_seen:\n                # Keep the higher-priority one between the two\n                existing_idx = struct_seen[struct_key]\n                if self._issue_priority(issue) > self._issue_priority(issues[existing_idx]):\n                    struct_remove.add(existing_idx)\n                    struct_seen[struct_key] = idx\n                else:\n                    struct_remove.add(idx)\n            else:\n                struct_seen[struct_key] = idx\n\n        pre_deduped = [i for j, i in enumerate(issues) if j not in struct_remove]\n        if len(struct_remove) > 0:\n            self.log.info(\n                f\"[synthesis] Cross-batch structural pre-pass: \"\n                f\"removed {len(struct_remove)} duplicates before LLM batching\"\n            )\n\n        # ── Step 2: Overlapping window LLM dedup pass ─────────────────────────\n        # Track globally removed indices so overlap re-encounters don't re-add\n        # issues that an earlier window already removed.\n        removed_ids: set[str] = set()   # use issue.id for stable identity\n        result = list(pre_deduped)\n\n        step       = self._DEDUP_BATCH_SIZE - self._DEDUP_BATCH_OVERLAP\n        offset     = 0\n        iterations = 0\n        max_iter   = (len(result) // max(step, 1)) + 2   # safety bound\n\n        while offset < len(result) and iterations < max_iter:\n            iterations += 1\n            # Build the window, excluding already-removed issues\n            window_raw = result[offset : offset + self._DEDUP_BATCH_SIZE]\n            window     = [i for i in window_raw if i.id not in removed_ids]\n            if not window:\n                offset += step\n                continue\n\n            deduped_window = await self._dedup_semantic(window)\n\n            # Accumulate ids removed by this window\n            surviving_ids = {i.id for i in deduped_window}\n            for i in window:\n                if i.id not in surviving_ids:\n                    removed_ids.add(i.id)\n\n            removed_count = len(window) - len(deduped_window)\n            if removed_count > 0:\n                self.log.debug(\n                    f\"[synthesis] Overlapping batch @offset={offset}: \"\n                    f\"removed {removed_count} duplicates (window={len(window)})\"\n                )\n\n            offset += step\n\n        # Rebuild the final list in original order, excluding all removed ids\n        final = [i for i in pre_deduped if i.id not in removed_ids]\n        self.log.info(\n            f\"[synthesis] Batched dedup complete: \"\n            f\"{len(issues)} → {len(final)} \"\n            f\"(structural={len(struct_remove)} LLM={len(removed_ids)})\"\n        )\n        return final

    def _apply_dedup_response(
        self,
        issues: list[Issue],
        resp: DeduplicationResponse,
    ) -> list[Issue]:
        """Apply the LLM deduplication response to the issue list."""
        indices_to_remove: set[int] = set()
        for cluster in resp.duplicate_clusters:
            canonical = cluster.canonical_index
            if not (0 <= canonical < len(issues)):
                continue
            for idx in cluster.indices:
                if 0 <= idx < len(issues) and idx != canonical:
                    indices_to_remove.add(idx)
                    self.log.debug(
                        f"[synthesis] Dedup: removing issue[{idx}] "
                        f"(duplicate of [{canonical}]): "
                        f"{cluster.reason[:100]}"
                    )

        return [i for j, i in enumerate(issues) if j not in indices_to_remove]

    # ── Step 3: Compound finding detection ───────────────────────────────────

    async def _detect_compound_findings(
        self, issues: list[Issue]
    ) -> list[CompoundFinding]:
        """
        Detect cross-domain compound findings.

        A compound finding is a vulnerability that:
          - Requires information from ≥2 auditor domains to identify
          - Is MORE severe than any single-domain finding because the
            combined effect enables an attack vector that neither domain
            can see alone
          - Requires a more comprehensive fix than patching a single issue

        Examples:
          • Race condition (ARCHITECTURE) + auth bypass (SECURITY)
            → race-condition-enabled privilege escalation
          • Integer overflow (SECURITY/STANDARDS) + missing bounds check
            (STANDARDS) + network input path (ARCHITECTURE)
            → remote integer overflow exploit
          • Memory management violation (STANDARDS) + callback into freed
            object (ARCHITECTURE) + user-controlled trigger (SECURITY)
            → use-after-free with user-controlled exploitation path
        """
        # Need findings from ≥2 different domains for cross-domain analysis
        domains_present = {i.executor_type for i in issues}
        if len(domains_present) < 2:
            self.log.debug(
                "[synthesis] Skipping compound detection: "
                f"only {len(domains_present)} domain(s) present"
            )
            return []

        if len(issues) < self._COMPOUND_MIN_CROSS_DOMAIN_ISSUES:
            self.log.debug(
                "[synthesis] Skipping compound detection: "
                f"only {len(issues)} issues (min={self._COMPOUND_MIN_CROSS_DOMAIN_ISSUES})"
            )
            return []

        # Build a rich summary grouping by domain
        domain_summaries = self._build_domain_grouped_summary(issues)
        summary_lines    = self._build_issue_summary(issues)

        prompt = (
            f"## Cross-Domain Code Audit Synthesis\n\n"
            f"Three specialized auditors have scanned the same codebase:\n"
            f"  • SECURITY auditor  — finds vulnerabilities, CWE violations\n"
            f"  • ARCHITECTURE auditor — finds structural, design, coupling issues\n"
            f"  • STANDARDS auditor — finds rule violations (MISRA, CERT, coding standards)\n\n"
            f"## Findings Grouped by Domain\n{domain_summaries}\n\n"
            f"## All Findings (0-indexed, for reference)\n{summary_lines}\n\n"
            f"## Task: Identify Cross-Domain Compound Findings\n\n"
            f"A compound finding is a vulnerability that CANNOT be identified by looking "
            f"at findings from a single domain.  It emerges from the COMBINATION of "
            f"findings across domains.\n\n"
            f"Look specifically for:\n"
            f"1. **Security+Architecture**: Race condition that enables auth bypass; "
            f"   architectural violation that creates a security-exploitable path\n"
            f"2. **Security+Standards**: CWE violation made exploitable by missing "
            f"   standard protections; integer overflow that bypasses an auth check\n"
            f"3. **Architecture+Standards**: Standard violation that causes an "
            f"   architectural invariant to break, creating system-level failure\n"
            f"4. **All three domains**: The rarest and most severe — a defect chain "
            f"   spanning all three auditor domains\n\n"
            f"## Constraints\n"
            f"- Only report findings where the COMBINATION is more severe than any "
            f"  individual finding.  Do not re-report single-domain findings.\n"
            f"- Each compound finding MUST reference ≥2 contributing_indices from "
            f"  different domains.\n"
            f"- **Confidence weighting**: each finding line shows `conf=X.XX`.  "
            f"  A compound finding whose contributing findings both have conf < 0.60 "
            f"  should receive amplification_factor ≤ 1.5.  Reserve amplification "
            f"  ≥ 3.0 for combinations where both contributors have conf ≥ 0.80.  "
            f"  Do not construct high-amplification compound findings from speculative "
            f"  low-confidence inputs — this inflates severity without evidence.\n"
            f"- Maximum {self.max_compound_findings} compound findings.\n"
            f"- If no genuine compound findings exist, return an empty list.\n\n"
            f"Return JSON with compound_findings array and synthesis_summary."
        )

        resp = await self.call_llm_structured(
            prompt=prompt,
            response_model=CompoundFindingResponse,
            system=(
                "You are a principal security architect performing cross-domain "
                "synthesis of code audit findings.  Your goal is to identify "
                "compound vulnerabilities that no single-domain auditor can detect. "
                "Be specific and precise — each compound finding must clearly "
                "explain why the COMBINATION is more dangerous than the parts."
            ),
            model_override=self._synthesis_model or self.config.critical_fix_model,
        )

        self.log.info(
            f"[synthesis] LLM returned {len(resp.compound_findings)} compound findings. "
            f"Summary: {resp.synthesis_summary[:200]}"
        )

        # Convert raw LLM response → CompoundFinding objects
        compound: list[CompoundFinding] = []
        for raw in resp.compound_findings[:self.max_compound_findings]:
            cf = self._raw_to_compound(raw, issues)
            if cf:
                compound.append(cf)

        return compound

    def _build_domain_grouped_summary(self, issues: list[Issue]) -> str:
        """Build a per-domain grouped summary for the compound detection prompt.

        Confidence scores are included on each finding line so the synthesis
        LLM knows which auditor signals are high-confidence vs. speculative.
        A compound finding built from two low-confidence findings is materially
        different from one built from two high-confidence findings and should
        receive a proportionally lower amplification_factor.
        """
        by_domain: dict[str, list[Issue]] = {}
        for issue in issues:
            domain = (issue.executor_type or ExecutorType.GENERAL).value
            by_domain.setdefault(domain, []).append(issue)

        lines: list[str] = []
        for domain, domain_issues in sorted(by_domain.items()):
            lines.append(f"\n### {domain} ({len(domain_issues)} findings)")
            for i, iss in enumerate(domain_issues[:20]):  # cap per domain
                conf = f" conf={iss.confidence:.2f}" if iss.confidence is not None else ""
                lines.append(
                    f"  [{issues.index(iss)}] [{iss.severity.value}]{conf} "
                    f"{iss.file_path}:{iss.line_start} — {iss.description[:150]}"
                )
            if len(domain_issues) > 20:
                lines.append(f"  ... +{len(domain_issues) - 20} more")
        return "\n".join(lines)

    def _build_issue_summary(self, issues: list[Issue]) -> str:
        """Build a flat indexed summary of issues for LLM prompts.

        Each line includes the auditor confidence score so the synthesis LLM
        can weight higher-confidence findings more heavily when resolving
        duplicate clusters and constructing compound findings.  A speculative
        low-confidence finding (conf=0.40) from one auditor should not be
        treated as equivalent to a high-confidence finding (conf=0.95) from
        another, even if they describe the same file and line.
        """
        lines: list[str] = []
        for i, issue in enumerate(issues):
            domain = (issue.executor_type or ExecutorType.GENERAL).value
            cwe    = f" [{issue.cwe_id}]" if issue.cwe_id else ""
            misra  = f" [{issue.misra_rule}]" if issue.misra_rule else ""
            conf   = f" conf={issue.confidence:.2f}" if issue.confidence is not None else ""
            lines.append(
                f"{i}: [{domain}][{issue.severity.value}]{cwe}{misra}{conf} "
                f"{issue.file_path}:{issue.line_start} — "
                f"{issue.description[:160]}"
            )
        return "\n".join(lines)

    def _raw_to_compound(
        self,
        raw: RawCompoundFinding,
        all_issues: list[Issue],
    ) -> CompoundFinding | None:
        """Convert a raw LLM compound finding to a CompoundFinding schema object."""
        if not raw.title or not raw.contributing_indices:
            return None

        # Validate contributing indices
        valid_indices = [
            idx for idx in raw.contributing_indices
            if 0 <= idx < len(all_issues)
        ]
        if len(valid_indices) < 2:
            self.log.debug(
                f"[synthesis] Dropping compound finding '{raw.title}': "
                f"fewer than 2 valid contributing indices"
            )
            return None

        contributing_issue_ids = [all_issues[i].id for i in valid_indices]

        # Compute max severity from contributors
        contributing_severities = [all_issues[i].severity for i in valid_indices]
        max_severity = max(
            contributing_severities,
            key=lambda s: {
                Severity.CRITICAL: 4,
                Severity.MAJOR:    3,
                Severity.MINOR:    2,
                Severity.INFO:     1,
            }.get(s, 0),
        )
        # Escalate severity for truly compound findings
        escalated_severity = self._escalate_severity(max_severity, raw.amplification_factor)

        # Parse category
        try:
            category = CompoundFindingCategory(raw.category.upper())
        except ValueError:
            category = CompoundFindingCategory.SECURITY_ARCHITECTURE

        # Parse fix complexity
        fix_complexity = raw.fix_complexity.upper() if raw.fix_complexity else "HIGH"

        return CompoundFinding(
            run_id=self.run_id,
            title=raw.title[:500],
            description=raw.description,
            severity=escalated_severity,
            category=category,
            contributing_issue_ids=contributing_issue_ids,
            domains_involved=[d.upper() for d in raw.domains_involved],
            amplification_factor=raw.amplification_factor,
            fix_complexity=fix_complexity,
            rationale=raw.rationale,
            cwe_id=raw.cwe_id,
            misra_rule=raw.misra_rule,
            cert_rule=raw.cert_rule,
            mil882e_category=SEVERITY_TO_MIL882E.get(
                escalated_severity, MilStd882eCategory.NONE
            ),
        )

    @staticmethod
    def _escalate_severity(base: Severity, amplification: float) -> Severity:
        """
        Escalate severity based on compound amplification.

        A compound finding with amplification ≥ 3.0 and base MAJOR gets
        escalated to CRITICAL because the combination creates a qualitatively
        worse vulnerability.
        """
        if amplification >= 3.0:
            rank = {
                Severity.INFO:     1,
                Severity.MINOR:    2,
                Severity.MAJOR:    3,
                Severity.CRITICAL: 4,
            }.get(base, 1)
            escalated_rank = min(rank + 1, 4)
            reverse = {1: Severity.INFO, 2: Severity.MINOR, 3: Severity.MAJOR, 4: Severity.CRITICAL}
            return reverse[escalated_rank]
        return base

    # ── Step 4: Convert CompoundFinding → Issue ───────────────────────────────

    def _compound_to_issue(
        self, cf: CompoundFinding, all_issues: list[Issue]
    ) -> Issue:
        """
        Create a full Issue record from a CompoundFinding so it flows through
        the normal consensus → fix → review pipeline.

        The issue is marked executor_type=SYNTHESIS so downstream agents know
        it originated from cross-domain analysis.
        """
        # Derive file_path from the highest-severity contributing issue
        contributing = [
            i for i in all_issues if i.id in cf.contributing_issue_ids
        ]
        primary = max(contributing, key=self._issue_priority) if contributing else None

        # Build compliance tags from contributors
        cwe_ids   = list({i.cwe_id   for i in contributing if i.cwe_id})
        misra_ids = list({i.misra_rule for i in contributing if i.misra_rule})
        cert_ids  = list({i.cert_rule  for i in contributing if i.cert_rule})

        description = (
            f"[COMPOUND FINDING — {cf.category.value}] {cf.title}\n\n"
            f"{cf.description}\n\n"
            f"Domains involved: {', '.join(cf.domains_involved)}\n"
            f"Amplification factor: {cf.amplification_factor:.1f}x\n"
            f"Fix complexity: {cf.fix_complexity}\n"
            f"Contributing issues: {len(cf.contributing_issue_ids)}\n\n"
            f"Rationale:\n{cf.rationale}"
        )

        fp_raw = (
            f"COMPOUND:{cf.title}:{':'.join(sorted(cf.contributing_issue_ids))}"
        )
        fingerprint = hashlib.sha256(fp_raw.encode()).hexdigest()[:16]

        return Issue(
            run_id=self.run_id,
            severity=cf.severity,
            file_path=primary.file_path if primary else "",
            line_start=primary.line_start if primary else 0,
            line_end=primary.line_end if primary else 0,
            function_name=primary.function_name if primary else "",
            description=description,
            status=IssueStatus.OPEN,
            executor_type=ExecutorType.SYNTHESIS,
            domain_mode=self.domain_mode,
            fix_requires_files=list({
                f for i in contributing for f in i.fix_requires_files
            }),
            confidence=0.9,  # Synthesis findings are high-confidence by construction
            fingerprint=fingerprint,
            cwe_id=cwe_ids[0] if cwe_ids else cf.cwe_id,
            misra_rule=misra_ids[0] if misra_ids else cf.misra_rule,
            cert_rule=cert_ids[0] if cert_ids else cf.cert_rule,
            mil882e_category=cf.mil882e_category,
            is_mandatory=cf.severity == Severity.CRITICAL,
        )

    # ── Audit trail ───────────────────────────────────────────────────────────

    async def _write_audit_trail(
        self,
        raw_count:      int,
        deduped_count:  int,
        compound_count: int,
    ) -> None:
        """Write a synthesis completion entry to the audit trail."""
        try:
            from brain.schemas import AuditTrailEntry
            entry = AuditTrailEntry(
                run_id=self.run_id,
                event_type="SYNTHESIS_COMPLETE",
                actor="SynthesisAgent",
                entity_id=self.run_id,
                entity_type="run",
                after=(
                    f"raw={raw_count} deduped={deduped_count} "
                    f"compound={compound_count} "
                    f"removed={raw_count - deduped_count}"
                ),
            )
            await self.storage.append_audit_trail(entry)
        except Exception as exc:
            self.log.debug(f"[synthesis] Audit trail write failed: {exc}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def agent_name(self) -> str:
        return "SynthesisAgent"
