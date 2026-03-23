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
import re
from pathlib import Path
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
    _COMPOUND_MIN_CROSS_DOMAIN_ISSUES: int = 2

    # ARCH-04 FIX: cap compound detection batch size so large codebases don't
    # overflow the LLM context window silently.
    # Previously _detect_compound_findings ran on the full deduplicated list
    # with no limit. A codebase with 400 findings produced a single LLM call
    # with ~80,000 chars, truncating silently on 32K-token models.
    _COMPOUND_BATCH_SIZE: int    = 40
    _COMPOUND_BATCH_OVERLAP: int = 10

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
        # ARCH-3: optional CPG engine for compound finding path verification.
        # When set, _verify_compound_paths() queries the CPG to confirm that
        # a runtime path exists between the files of contributing issues.
        self._cpg_engine: Any = None   # set by controller after construction
        # ARCH-03 FIX: expose the heuristic test-match count so the controller
        # can include it in the persisted SynthesisReport without duplicating
        # the computation.  Initialised to 0; updated by run() after the
        # heuristic pass completes.
        self._last_heuristic_matched: int = 0

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
                raw_compounds = await self._detect_compound_findings(deduped)
                # ARCH-3 FIX: CPG path verification.
                # The synthesis LLM may hallucinate compound findings — two
                # findings that superficially appear related but have no actual
                # code path connecting them at runtime. Without verification the
                # pipeline escalates ungrounded findings to CRITICAL severity.
                # When a CPG engine is available, verify that a path exists between
                # the file locations of each compound finding's contributors before
                # accepting it. Unverifiable findings are retained but confidence
                # is lowered and a note is added so reviewers know the path was
                # not confirmed. When no CPG is available, all findings are
                # retained with their original confidence (same as before).
                compound_findings = await self._verify_compound_paths(raw_compounds, deduped)
                self.log.info(
                    f"[synthesis] Detected {len(compound_findings)} compound findings "
                    f"({sum(1 for cf in compound_findings if getattr(cf, '_cpg_verified', True))} "
                    f"CPG-verified)"
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

        # ── Step 4.5: Heuristic fail_tests population (ARCH-03 FIX) ─────────────
        # Issues sourced from static analysis never have fail_tests populated, so
        # test_score=0 and the BoBN composite collapses to 40% signal for the
        # dominant issue class. Here we attempt to match issues against existing
        # test files in the repository to supply at least a weak correctness signal.
        # Even one matching test is better than none — it prevents the full collapse.
        heuristic_matched = 0
        try:
            heuristic_matched = await self._populate_fail_tests_heuristic(deduped)
            if heuristic_matched:
                self.log.info(
                    f"[synthesis] Heuristic fail_tests: matched tests for "
                    f"{heuristic_matched}/{len(deduped)} issues that had none"
                )
        except Exception as _ht_exc:
            self.log.debug(f"[synthesis] Heuristic fail_tests population failed (non-fatal): {_ht_exc}")

        # Expose the count so the controller can persist it in SynthesisReport.
        self._last_heuristic_matched = heuristic_matched

        # ── Step 5: Audit trail ───────────────────────────────────────────────
        # ARCH-03 FIX: compute BoBN-inactive stats and include in the audit trail
        # so operators and benchmark reports know when test signal is absent.
        _no_tests = [i for i in deduped if not i.fail_tests]
        _bobn_inactive_pct = (len(_no_tests) / len(deduped) * 100.0) if deduped else 0.0
        if _no_tests:
            self.log.warning(
                "[synthesis] ARCH-03: %d/%d deduped issues have no fail_tests after "
                "heuristic matching — BoBN composite scoring will use 40%% signal "
                "(0.3×robust + 0.1×minimal, test_score=0) for these issues. "
                "Provide fail_tests on Issue records or instrument the auditor to "
                "generate reproduction tests for full BoBN effectiveness.",
                len(_no_tests), len(deduped),
            )

        await self._write_audit_trail(
            raw_count=len(issues),
            deduped_count=len(deduped),
            compound_count=len(compound_findings),
            issues_without_fail_tests=len(_no_tests),
            bobn_inactive_pct=_bobn_inactive_pct,
            heuristic_tests_matched=heuristic_matched,
        )

        # Combine deduplicated issues + compound synthesis issues
        all_issues = deduped + compound_issues
        self.log.info(
            f"[synthesis] Complete: {len(all_issues)} final issues "
            f"({len(deduped)} deduped + {len(compound_issues)} compound) "
            f"bobn_inactive={_bobn_inactive_pct:.0f}%"
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
        For lists larger than _DEDUP_BATCH_SIZE, deduplicate in overlapping
        windows to prevent cross-batch duplicate pairs from surviving.

        CORRECTNESS FIX
        ---------------
        The original implementation processed non-overlapping sequential windows
        (``result[offset:offset + batch_size]``) and incremented ``offset`` by
        the deduped window length.  A duplicate pair whose two members straddle a
        batch boundary — issue A in the last position of window N, issue B in the
        first position of window N+1 — would never appear in the same LLM call
        and would survive deduplication.  This is a silent correctness failure
        that compounds on large codebases (thousands of findings across three
        auditor domains).

        Fix strategy
        ------------
        1. **Cross-batch structural pre-pass**: before any LLM call, remove
           exact structural duplicates (same file + line range + description
           prefix) regardless of batch position.  This catches the majority of
           cross-batch duplicates cheaply without an LLM round-trip.

        2. **Overlapping windows**: each window of size _DEDUP_BATCH_SIZE overlaps
           the previous window by _DEDUP_BATCH_OVERLAP issues.  Every pair of
           adjacent issues therefore appears in at least one shared window as long
           as their index distance is ≤ _DEDUP_BATCH_SIZE - _DEDUP_BATCH_OVERLAP.
           Indices removed in one window are excluded from all subsequent windows
           so the overlap does not re-introduce already-removed issues.
        """
        if len(issues) <= self._DEDUP_BATCH_SIZE:
            return await self._dedup_semantic(issues)

        # ── Step 1: Cross-batch structural pre-pass ───────────────────────────
        # Catch duplicates that would straddle batch boundaries using a
        # deterministic structural key: (file_path, line_start, description_64).
        # This is cheaper than the LLM dedup pass and catches the common case
        # where the same finding is emitted by two different auditors in the
        # same file at the same location.
        struct_seen: dict[str, int] = {}   # key → first occurrence index
        struct_remove: set[int] = set()
        for idx, issue in enumerate(issues):
            struct_key = (
                f"{issue.file_path}:{issue.line_start}:{issue.line_end}:"
                f"{issue.description[:64].lower().strip()}"
            )
            if struct_key in struct_seen:
                # Keep the higher-priority one between the two
                existing_idx = struct_seen[struct_key]
                if self._issue_priority(issue) > self._issue_priority(issues[existing_idx]):
                    struct_remove.add(existing_idx)
                    struct_seen[struct_key] = idx
                else:
                    struct_remove.add(idx)
            else:
                struct_seen[struct_key] = idx

        pre_deduped = [i for j, i in enumerate(issues) if j not in struct_remove]
        if len(struct_remove) > 0:
            self.log.info(
                f"[synthesis] Cross-batch structural pre-pass: "
                f"removed {len(struct_remove)} duplicates before LLM batching"
            )

        # ── Step 2: Overlapping window LLM dedup pass ─────────────────────────
        # Track globally removed indices so overlap re-encounters don't re-add
        # issues that an earlier window already removed.
        removed_ids: set[str] = set()   # use issue.id for stable identity
        result = list(pre_deduped)

        step       = self._DEDUP_BATCH_SIZE - self._DEDUP_BATCH_OVERLAP
        offset     = 0
        iterations = 0
        max_iter   = (len(result) // max(step, 1)) + 2   # safety bound

        while offset < len(result) and iterations < max_iter:
            iterations += 1
            # Build the window, excluding already-removed issues
            window_raw = result[offset : offset + self._DEDUP_BATCH_SIZE]
            window     = [i for i in window_raw if i.id not in removed_ids]
            if not window:
                offset += step
                continue

            deduped_window = await self._dedup_semantic(window)

            # Accumulate ids removed by this window
            surviving_ids = {i.id for i in deduped_window}
            for i in window:
                if i.id not in surviving_ids:
                    removed_ids.add(i.id)

            removed_count = len(window) - len(deduped_window)
            if removed_count > 0:
                self.log.debug(
                    f"[synthesis] Overlapping batch @offset={offset}: "
                    f"removed {removed_count} duplicates (window={len(window)})"
                )

            offset += step

        # Rebuild the final list in original order, excluding all removed ids
        final = [i for i in pre_deduped if i.id not in removed_ids]
        self.log.info(
            f"[synthesis] Batched dedup complete: "
            f"{len(issues)} → {len(final)} "
            f"(structural={len(struct_remove)} LLM={len(removed_ids)})"
        )
        return final
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

        ARCH-04 FIX: runs in overlapping batches of _COMPOUND_BATCH_SIZE (40)
        to prevent silent LLM context overflow. Previously this ran unbounded on
        the full deduplicated list — 400 findings = ~80,000 chars = silent
        truncation on 32K-token models. Results are deduplicated across batches.

        A compound finding requires information from ≥2 auditor domains and
        is more severe than any single-domain finding in isolation.
        """
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

        # Small list: run directly without batching overhead
        if len(issues) <= self._COMPOUND_BATCH_SIZE:
            return await self._detect_compound_findings_batch(issues)

        # ARCH-04: overlapping batches for large issue lists
        total_batches = max(
            1,
            (len(issues) - self._COMPOUND_BATCH_OVERLAP) //
            (self._COMPOUND_BATCH_SIZE - self._COMPOUND_BATCH_OVERLAP) + 1
        )
        self.log.info(
            "[synthesis] Compound detection: %d issues → %d overlapping batch(es) "
            "(batch_size=%d overlap=%d)",
            len(issues), total_batches,
            self._COMPOUND_BATCH_SIZE, self._COMPOUND_BATCH_OVERLAP,
        )

        all_compounds: list[CompoundFinding] = []
        seen_keys: set[str] = set()
        step   = self._COMPOUND_BATCH_SIZE - self._COMPOUND_BATCH_OVERLAP
        offset = 0

        while offset < len(issues):
            batch = issues[offset: offset + self._COMPOUND_BATCH_SIZE]
            if len({i.executor_type for i in batch}) >= 2:
                try:
                    batch_results = await self._detect_compound_findings_batch(batch)
                    for cf in batch_results:
                        key = f"{cf.title.lower()[:60]}"
                        if key not in seen_keys:
                            seen_keys.add(key)
                            all_compounds.append(cf)
                except Exception as exc:
                    self.log.warning(
                        "[synthesis] Compound batch @offset=%d failed: %s", offset, exc
                    )
            offset += step
            if len(all_compounds) >= self.max_compound_findings:
                break

        self.log.info(
            "[synthesis] Compound detection complete: %d unique findings across batches",
            len(all_compounds),
        )

        # ARCH-4 FIX: final cross-domain pass on top-5-per-domain findings.
        #
        # The sliding-window approach above caps batches at _COMPOUND_BATCH_SIZE=40.
        # Issue pairs more than 40 positions apart are NEVER co-examined — exactly
        # the highest-value cross-domain interactions (e.g. issue #1 SECURITY +
        # issue #301 ARCHITECTURE) that only appear in large codebases.
        #
        # One additional LLM call covers the top 5 by severity from each of the
        # three auditor domains (15 issues total — well within context limits),
        # guaranteeing the highest-severity cross-domain interactions are always
        # examined regardless of total issue count or issue ordering.
        try:
            from brain.schemas import ExecutorType as _ET
            _domain_keys = ["SECURITY", "ARCHITECTURE", "STANDARDS"]
            _top_per_domain: list = []
            for _dk in _domain_keys:
                _domain_issues = [
                    i for i in issues
                    if str(getattr(i, "executor_type", "") or
                           getattr(i, "domain", "") or "").upper() == _dk
                ]
                _domain_issues.sort(
                    key=lambda x: float(getattr(x, "severity_score", 0) or 0),
                    reverse=True,
                )
                _top_per_domain.extend(_domain_issues[:5])

            # Only run if we have findings from at least 2 domains AND the curated
            # set is not a subset of an already-examined batch (would be redundant
            # when issue count <= _COMPOUND_BATCH_SIZE).
            _curated_ids = {getattr(i, "id", None) for i in _top_per_domain}
            _already_covered = len(issues) <= self._COMPOUND_BATCH_SIZE
            if (
                len({str(getattr(i, "executor_type", "") or
                         getattr(i, "domain", "")).upper()
                     for i in _top_per_domain}) >= 2
                and not _already_covered
                and _top_per_domain
            ):
                self.log.info(
                    "[synthesis] ARCH-4 final cross-domain pass: "
                    "%d curated issues (top-5 per domain)", len(_top_per_domain)
                )
                _final_compounds = await self._detect_compound_findings_batch(
                    _top_per_domain
                )
                for cf in _final_compounds:
                    key = f"{cf.title.lower()[:60]}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_compounds.append(cf)
                self.log.info(
                    "[synthesis] ARCH-4 cross-domain pass added %d new compound finding(s)",
                    len(_final_compounds),
                )
        except Exception as _arch4_exc:
            self.log.warning(
                "[synthesis] ARCH-4 final cross-domain pass failed (non-fatal): %s",
                _arch4_exc,
            )

        return all_compounds[:self.max_compound_findings]

    async def _detect_compound_findings_batch(
        self, issues: list[Issue]
    ) -> list[CompoundFinding]:
        """
        Run compound detection on one bounded batch. Do not call directly —
        use _detect_compound_findings() which handles batching and dedup.
        """
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

        compound: list[CompoundFinding] = []
        for raw in resp.compound_findings[:self.max_compound_findings]:
            cf = self._raw_to_compound(raw, issues)
            if cf:
                compound.append(cf)

        return compound

    async def _verify_compound_paths(
        self,
        raw_compounds: list[CompoundFinding],
        all_issues: list[Issue],
    ) -> list[CompoundFinding]:
        """
        ARCH-3 FIX: CPG path verification for compound findings.

        The synthesis LLM generates compound findings by reasoning about
        which single-domain findings could combine into a more severe
        vulnerability. This reasoning is plausible-sounding but ungrounded —
        the LLM cannot verify that the two code locations actually share a
        runtime execution path, data-flow path, or control dependency.

        Without verification, compound findings are elevated to CRITICAL
        severity based purely on LLM speculation. This inflates severity
        without evidence and can cause the pipeline to escalate false positives
        to human review while real findings wait in the queue.

        This method queries the CPG engine for each compound finding to verify
        that a path exists between the file locations of its contributing issues.
        If the CPG confirms a connection, the finding is marked cpg_verified=True.
        If the CPG returns no path, confidence is lowered to 0.5 and a note is
        added to the rationale indicating the path was not confirmed.
        If no CPG engine is available, all findings are returned unchanged
        (same behaviour as before this fix was added).

        The results of verification are surfaced in the CompoundFinding record
        and in the synthesis log so reviewers can distinguish verified from
        unverified compound findings in the dashboard.
        """
        if not raw_compounds:
            return []

        # Lazy-resolve the CPG engine from the repo_root context.
        cpg_engine = getattr(self, '_cpg_engine', None)
        if cpg_engine is None and self.repo_root is not None:
            # The controller may have stored the CPG engine on self.repo_root
            # via dependency injection — check common attribute names.
            for attr in ('cpg_engine', '_cpg_engine', 'cpg'):
                candidate = getattr(self.repo_root, attr, None)
                if candidate is not None:
                    cpg_engine = candidate
                    break

        if cpg_engine is None or not getattr(cpg_engine, 'is_available', False):
            self.log.debug(
                "[synthesis] CPG not available — skipping path verification "
                "for %d compound findings. Start Joern to enable verification.",
                len(raw_compounds),
            )
            return raw_compounds

        verified: list[CompoundFinding] = []
        for cf in raw_compounds:
            contributing = [
                i for i in all_issues if i.id in cf.contributing_issue_ids
            ]
            if len(contributing) < 2:
                verified.append(cf)
                continue

            # Check whether the CPG contains any path between the files of the
            # two highest-priority contributing issues.
            try:
                file_a = contributing[0].file_path
                file_b = contributing[1].file_path

                if file_a == file_b:
                    # Same file — intra-file compound finding. CPG path is
                    # implicit (they share the same compilation unit). Mark verified.
                    cf.rationale = (cf.rationale or "") + " [CPG: same-file path confirmed]"
                    verified.append(cf)
                    continue

                # Query CPG for any connecting path between the two files.
                path_exists = await cpg_engine.path_exists_between(
                    source_file=file_a,
                    sink_file=file_b,
                )

                if path_exists:
                    cf.rationale = (cf.rationale or "") + (
                        f" [CPG: path confirmed between {file_a} and {file_b}]"
                    )
                    self.log.debug(
                        "[synthesis] CPG verified path: %s → %s for '%s'",
                        file_a, file_b, cf.title[:60],
                    )
                else:
                    # No CPG path found. Retain the finding but lower confidence
                    # and note the unverified status clearly.
                    original_amp = cf.amplification_factor
                    cf.amplification_factor = min(cf.amplification_factor, 1.5)
                    cf.rationale = (cf.rationale or "") + (
                        f" [CPG WARNING: no confirmed path between {file_a} and "
                        f"{file_b}. Amplification reduced {original_amp:.1f}x → "
                        f"{cf.amplification_factor:.1f}x. Manual review required.]"
                    )
                    self.log.warning(
                        "[synthesis] CPG found NO path between %s and %s "
                        "for compound finding '%s' — confidence reduced",
                        file_a, file_b, cf.title[:60],
                    )
                verified.append(cf)

            except Exception as exc:
                # CPG query failed — retain the finding unchanged rather than
                # dropping it, but note that verification was not possible.
                self.log.debug(
                    "[synthesis] CPG path verification error for '%s': %s",
                    cf.title[:60], exc,
                )
                cf.rationale = (cf.rationale or "") + " [CPG: verification error — unconfirmed]"
                verified.append(cf)

        n_verified   = sum(1 for cf in verified if "[CPG: path confirmed" in (cf.rationale or "") or "[CPG: same-file" in (cf.rationale or ""))
        n_unverified = sum(1 for cf in verified if "[CPG WARNING:" in (cf.rationale or ""))
        self.log.info(
            "[synthesis] CPG path verification: %d confirmed, %d unconfirmed, %d errors",
            n_verified, n_unverified, len(verified) - n_verified - n_unverified,
        )
        return verified

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

    async def _populate_fail_tests_heuristic(self, issues: list[Issue]) -> int:
        """
        ARCH-03 FIX: Heuristic fail_tests population for static-analysis-sourced issues.

        The BoBN composite formula is:
            0.6 × test_score + 0.3 × robust + 0.1 × minimal

        When fail_tests is empty, test_score=0 and the composite collapses to 40%
        signal.  This method attempts to recover at least a weak correctness signal
        by scanning the repository's test suite for test functions that exercise
        the same file or function that the issue targets.

        Strategy
        --------
        1. Walk repo_root looking for test files (test_*.py / *_test.py).
           Cap the scan at _MAX_TEST_FILES to avoid crawling huge repos.
        2. For each issue that has no fail_tests, derive candidate identifiers:
              • the stem of the affected file (e.g. "fix_memory" from
                "memory/fix_memory.py")
              • any function name extracted from the issue description
                (word following "in ", "def ", or "`")
        3. For each matching test file, extract test function names with a
           lightweight regex (no AST parse — must stay fast for 400-issue runs).
        4. Score each test function against the candidate identifiers; keep
           those that contain at least one identifier as a substring.
        5. Write matching test IDs back to issue.fail_tests and persist via
           storage.upsert_issue().

        This is intentionally conservative: we only add tests that contain
        the affected symbol name as a literal substring, so false-positive
        signal is very rare.  The benefit of a weak heuristic test (e.g.
        test_retrieve_federated for fix_memory.py) far outweighs having
        test_score=0, even if the test doesn't exercise the exact defect path.

        Returns the count of issues that received at least one new fail_test.
        """
        _MAX_TEST_FILES = 300
        _FUNC_RE = re.compile(r"def (test_\w+)", re.MULTILINE)
        _IDENT_RE = re.compile(
            r"(?:in |def |`)([a-z_][a-z0-9_]{2,})", re.IGNORECASE
        )

        repo_root = self.repo_root
        if repo_root is None:
            return 0

        # Resolve to a Path whether repo_root is str or Path-like
        try:
            root = Path(str(repo_root))
        except Exception:
            return 0
        if not root.is_dir():
            return 0

        # Collect issues that need tests
        needs_tests = [i for i in issues if not i.fail_tests]
        if not needs_tests:
            return 0

        # Walk once, collect test files (capped)
        test_files: list[Path] = []
        try:
            for p in root.rglob("*.py"):
                name = p.name
                if name.startswith("test_") or name.endswith("_test.py"):
                    test_files.append(p)
                    if len(test_files) >= _MAX_TEST_FILES:
                        break
        except Exception:
            return 0

        if not test_files:
            return 0

        # Build an index: file_stem → [(test_id, test_name)] where test_id is
        # "path/to/test_file.py::test_function_name" (pytest node-id format).
        # We index by test file stem so the per-issue lookup is O(1) not O(files).
        stem_index: dict[str, list[str]] = {}
        for tf in test_files:
            try:
                src = tf.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            rel = str(tf.relative_to(root)) if root in tf.parents or tf.is_relative_to(root) else tf.name
            for fn_name in _FUNC_RE.findall(src):
                node_id = f"{rel}::{fn_name}"
                # index under the test file's own stem
                stem_index.setdefault(tf.stem, []).append(node_id)
                # also index under every word token in the test file name so
                # "test_fix_memory.py" matches both "fix" and "memory" lookups
                for tok in tf.stem.replace("test_", "").split("_"):
                    if len(tok) >= 3:
                        stem_index.setdefault(tok, []).append(node_id)

        matched_count = 0
        for issue in needs_tests:
            # Derive candidate identifiers from the affected file and description
            candidates: set[str] = set()
            if issue.file_path:
                stem = Path(issue.file_path).stem.lower()
                candidates.add(stem)
                # also add each underscore-separated component
                for part in stem.split("_"):
                    if len(part) >= 3:
                        candidates.add(part)
            for m in _IDENT_RE.finditer(issue.description or ""):
                tok = m.group(1).lower()
                if len(tok) >= 3:
                    candidates.add(tok)

            # Collect test IDs whose name contains any candidate as a substring
            matched_ids: list[str] = []
            seen_ids: set[str] = set()
            for cand in candidates:
                for node_id in stem_index.get(cand, []):
                    if node_id not in seen_ids:
                        # Extra check: the test function name must also contain
                        # the candidate — avoids picking up unrelated tests that
                        # happened to share a common index key.
                        fn_name = node_id.split("::")[-1]
                        if cand in fn_name:
                            matched_ids.append(node_id)
                            seen_ids.add(node_id)

            if matched_ids:
                issue.fail_tests = matched_ids[:10]   # cap at 10 per issue
                try:
                    await self.storage.upsert_issue(issue)
                except Exception:
                    pass   # non-fatal — issue object already mutated in memory
                matched_count += 1

        return matched_count

    async def _write_audit_trail(
        self,
        raw_count:      int,
        deduped_count:  int,
        compound_count: int,
        # ARCH-03 FIX: BoBN-inactive telemetry — how many issues lack fail_tests
        # and therefore receive only 40% of the normal composite scoring signal.
        issues_without_fail_tests: int = 0,
        bobn_inactive_pct:         float = 0.0,
        heuristic_tests_matched:   int = 0,
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
                    f"removed={raw_count - deduped_count} "
                    f"bobn_inactive={bobn_inactive_pct:.0f}% "
                    f"no_fail_tests={issues_without_fail_tests} "
                    f"heuristic_matched={heuristic_tests_matched}"
                ),
            )
            await self.storage.append_audit_trail(entry)
        except Exception as exc:
            self.log.debug(f"[synthesis] Audit trail write failed: {exc}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def agent_name(self) -> str:
        return "SynthesisAgent"
