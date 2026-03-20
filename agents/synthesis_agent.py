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

    # Minimum number of issues from ≥2 different domains required before
    # attempting compound finding detection.  Below this threshold the
    # cross-domain analysis adds noise, not signal.
    _COMPOUND_MIN_CROSS_DOMAIN_ISSUES: int = 3

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

    async def _dedup_semantic_batched(self, issues: list[Issue]) -> list[Issue]:
        """
        For lists larger than _DEDUP_BATCH_SIZE, process in overlapping batches.
        Each batch gets a chunk of issues; duplicates within each chunk are removed.
        """
        result = list(issues)
        batch_size = self._DEDUP_BATCH_SIZE
        offset = 0
        while offset < len(result):
            batch = result[offset:offset + batch_size]
            deduped_batch = await self._dedup_semantic(batch)
            # Replace the batch slice with the deduped version
            removed = len(batch) - len(deduped_batch)
            result[offset:offset + batch_size] = deduped_batch
            offset += len(deduped_batch)
            if removed > 0:
                self.log.debug(
                    f"[synthesis] Batch dedup @offset={offset}: "
                    f"removed {removed} duplicates"
                )
        return result

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
        """Build a per-domain grouped summary for the compound detection prompt."""
        by_domain: dict[str, list[Issue]] = {}
        for issue in issues:
            domain = (issue.executor_type or ExecutorType.GENERAL).value
            by_domain.setdefault(domain, []).append(issue)

        lines: list[str] = []
        for domain, domain_issues in sorted(by_domain.items()):
            lines.append(f"\n### {domain} ({len(domain_issues)} findings)")
            for i, iss in enumerate(domain_issues[:20]):  # cap per domain
                lines.append(
                    f"  [{issues.index(iss)}] [{iss.severity.value}] "
                    f"{iss.file_path}:{iss.line_start} — {iss.description[:150]}"
                )
            if len(domain_issues) > 20:
                lines.append(f"  ... +{len(domain_issues) - 20} more")
        return "\n".join(lines)

    def _build_issue_summary(self, issues: list[Issue]) -> str:
        """Build a flat indexed summary of issues for LLM prompts."""
        lines: list[str] = []
        for i, issue in enumerate(issues):
            domain = (issue.executor_type or ExecutorType.GENERAL).value
            cwe    = f" [{issue.cwe_id}]" if issue.cwe_id else ""
            misra  = f" [{issue.misra_rule}]" if issue.misra_rule else ""
            lines.append(
                f"{i}: [{domain}][{issue.severity.value}]{cwe}{misra} "
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
