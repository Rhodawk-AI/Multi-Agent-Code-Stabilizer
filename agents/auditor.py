"""
agents/auditor.py
=================
Finds issues in the codebase using domain-specialised LLM auditors.

FIXES vs previous version
──────────────────────────
• GAP-2 CRITICAL: auditors worked exclusively from pre-computed file summaries.
  Logic errors the reader missed were invisible.  Now:
  - Files under DIRECT_AUDIT_LINE_THRESHOLD (5 000 lines) are audited from
    ACTUAL source code, not summaries.
  - Files above the threshold continue to use summary + observations (the only
    practical approach at scale), with a note to the LLM that full source is
    unavailable.
  - The fixer receives the same direct-source context for targeted regions.
• GAP-20: added ``_validate_findings()`` — a lightweight second-pass that asks
  the LLM to confirm each finding is actually present in the code.  Findings
  the validator rejects are downgraded to INFO rather than discarded entirely
  (they're logged for review).
• Domain rules injection: ``DOMAIN_EXTRA_INSTRUCTIONS`` maps DomainMode values
  to additional audit instructions injected into the system prompt, so
  finance/medical/military repos automatically get domain-specific scrutiny
  without changing any agent code.
• ``_build_brain_summary`` now accepts a repo_root so direct-source batches
  can interleave summary-mode files with source-mode files efficiently.
• Consensus metadata (``consensus_votes``, ``consensus_confidence``) is now
  populated on each Issue so the ConsensusEngine can use it.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agents.base import AgentConfig, BaseAgent, wrap_content
from brain.schemas import (
    DomainMode,
    ExecutorType,
    Issue,
    IssueFingerprint,
    IssueStatus,
    Severity,
)
from brain.storage import BrainStorage

log = logging.getLogger(__name__)

MAX_FIX_ATTEMPTS  = 3
AUDIT_BATCH_SIZE  = 20        # reduced from 30 — smaller batches = higher quality
DIRECT_AUDIT_LINE_THRESHOLD = 5_000  # files below this: audit from real source


# ──────────────────────────────────────────────────────────────────────────────
# LLM response models
# ──────────────────────────────────────────────────────────────────────────────

class AuditIssue(BaseModel):
    severity: Severity
    file_path: str = Field(description="Relative path to the affected file")
    line_start: int = Field(ge=0, default=0)
    line_end:   int = Field(ge=0, default=0)
    master_prompt_section: str = Field(
        description="Which section of the master prompt is violated"
    )
    description: str = Field(description="Precise, actionable description of the issue")
    fix_requires_files: list[str] = Field(
        default_factory=list,
        description="All files needed to implement the fix",
    )
    confidence: float = Field(
        ge=0.0, le=1.0, default=0.85,
        description="Your confidence this issue is real (0–1)"
    )


class AuditOutput(BaseModel):
    domain:     str
    issues:     list[AuditIssue] = Field(default_factory=list)
    confidence: float            = Field(ge=0.0, le=1.0, default=0.85)
    notes:      str              = ""


class FindingValidation(BaseModel):
    """Second-pass hallucination filter."""
    issue_id:  str
    confirmed: bool
    reason:    str = ""


class FindingValidationBatch(BaseModel):
    validations: list[FindingValidation] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Domain-specific audit instruction add-ons
# ──────────────────────────────────────────────────────────────────────────────

DOMAIN_EXTRA_INSTRUCTIONS: dict[DomainMode, str] = {
    DomainMode.FINANCE: (
        "\n\n## Finance Domain — Additional Requirements\n"
        "- Flag ALL float arithmetic on monetary values (must use Decimal).\n"
        "- Flag ALL non-atomic balance mutations.\n"
        "- Flag ALL non-cryptographic random usage.\n"
        "- Flag ALL missing transaction rollback on error paths.\n"
        "- Check every SQL query for injection vectors.\n"
        "- Apply PCI-DSS: card data must never be logged.\n"
    ),
    DomainMode.MEDICAL: (
        "\n\n## Medical Domain — Additional Requirements\n"
        "- Flag ALL float arithmetic on dosage values.\n"
        "- Flag ANY code path that can disable a safety alarm.\n"
        "- Verify every patient data access writes an immutable audit log entry.\n"
        "- Apply IEC 62304: Class C functions (dosage, infusion, alarm) require\n"
        "  formal pre/post conditions or explicit assertion guards.\n"
        "- Flag ANY nullable patient_id.\n"
    ),
    DomainMode.MILITARY: (
        "\n\n## Military / Safety-Critical Domain — Additional Requirements\n"
        "- Flag ALL dynamic memory allocation (malloc/calloc/new) outside init.\n"
        "- Flag ALL use of goto (MISRA Rule 15.1).\n"
        "- Flag ALL stdio functions (printf/scanf/gets) in RTOS context.\n"
        "- Flag ALL unbounded loops — every loop must have a provable upper bound.\n"
        "- Flag ALL non-deterministic operations.\n"
        "- Apply DO-178C: every safety-critical function must have test coverage.\n"
    ),
    DomainMode.EMBEDDED: (
        "\n\n## Embedded / RTOS Domain — Additional Requirements\n"
        "- Flag ALL dynamic memory allocation after initialisation phase.\n"
        "- Flag ALL blocking operations in interrupt handlers.\n"
        "- Flag ALL stack allocations exceeding 512 bytes in a single frame.\n"
        "- Verify all shared-memory accesses use appropriate locking primitives.\n"
    ),
    DomainMode.GENERAL: "",
}

# Domain-specialised system prompts per executor type
DOMAIN_SYSTEM_PROMPTS: dict[ExecutorType, str] = {
    ExecutorType.SECURITY: (
        "security auditor specializing in CERT, CWE, OWASP, and OS-level security. "
        "Focus on: injection vulnerabilities, unsafe deserialization, credential exposure, "
        "privilege escalation, sandboxing failures, and supply-chain risks."
    ),
    ExecutorType.ARCHITECTURE: (
        "principal software architect auditing for GII (General Interactive Intelligence) "
        "architectural correctness. Focus on: cognitive loop integrity, safety gate "
        "completeness, consequence-grounded reasoning, exception propagation, state machine "
        "correctness, dead code paths, and cross-component invariant violations."
    ),
    ExecutorType.STANDARDS: (
        "software quality engineer auditing for MISRA-C, DO-178C, CERT, and general "
        "industry coding standards. Focus on: missing error handling, resource leaks, "
        "undefined behaviour, test coverage gaps, documentation gaps, and type safety."
    ),
    ExecutorType.GENERAL: (
        "senior software engineer performing general code quality audit. "
        "Focus on: bugs, logic errors, performance issues, and maintainability problems."
    ),
    ExecutorType.DOMAIN: (
        "domain expert auditor applying mission-critical standards."
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────

class AuditorAgent(BaseAgent):

    def __init__(
        self,
        storage:             BrainStorage,
        run_id:              str,
        executor_type:       ExecutorType,
        master_prompt_path:  str | Path,
        config:              AgentConfig | None = None,
        mcp_manager:         Any | None         = None,
        domain_mode:         DomainMode          = DomainMode.GENERAL,
        repo_root:           Path | None         = None,
        validate_findings:   bool                = True,
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.agent_type        = executor_type
        self.master_prompt_path = Path(master_prompt_path)
        self.domain_mode       = domain_mode
        self.repo_root         = repo_root
        self.validate_findings_enabled = validate_findings
        self._master_prompt: str | None = None

    # ── Prompt loading ────────────────────────────────────────────────────────

    async def _load_master_prompt(self) -> str:
        if self._master_prompt is None:
            if self.master_prompt_path.exists():
                self._master_prompt = self.master_prompt_path.read_text(encoding="utf-8")
            else:
                self._master_prompt = self._default_master_prompt()
        return self._master_prompt

    def _default_master_prompt(self) -> str:
        return (
            "# Master Audit Prompt\n\n"
            "## 1. Safety Gates\n"
            "Every action must pass through a consequence reasoner before execution.\n"
            "## 2. Exception Safety\n"
            "All external calls must have explicit exception handlers with defined recovery.\n"
            "## 3. State Machine Integrity\n"
            "State transitions must be deterministic and all terminal states handled.\n"
            "## 4. Resource Management\n"
            "All resources (files, connections, threads) must have guaranteed cleanup.\n"
            "## 5. Security\n"
            "No credential exposure, no unsafe deserialization, all inputs validated.\n"
        )

    # ── Main run ──────────────────────────────────────────────────────────────

    async def run(self, **kwargs: Any) -> list[Issue]:
        master_prompt = await self._load_master_prompt()
        # Append domain-specific instructions
        domain_extra  = DOMAIN_EXTRA_INSTRUCTIONS.get(self.domain_mode, "")
        full_prompt   = master_prompt + domain_extra

        brain_summary = await self._build_brain_summary()
        self.log.info(
            f"Auditor [{self.agent_type.value}] [{self.domain_mode.value}]: "
            f"auditing {brain_summary['file_count']} files "
            f"({brain_summary['direct_count']} direct-source)"
        )

        all_issues: list[Issue] = []

        # ── Batch 1: direct source audit for small files ─────────────────────
        if brain_summary["direct_files"]:
            direct_batches = self._batch_files(brain_summary["direct_files"], AUDIT_BATCH_SIZE)
            for batch_idx, batch in enumerate(direct_batches):
                self.log.info(
                    f"Auditor [{self.agent_type.value}]: direct-source batch "
                    f"{batch_idx + 1}/{len(direct_batches)}"
                )
                issues = await self._audit_direct_batch(batch, full_prompt, brain_summary["global_context"])
                all_issues.extend(issues)
                await self.check_cost_ceiling()

        # ── Batch 2: summary-based audit for large files ─────────────────────
        if brain_summary["summary_files"]:
            summary_batches = self._batch_files(brain_summary["summary_files"], AUDIT_BATCH_SIZE)
            for batch_idx, batch in enumerate(summary_batches):
                self.log.info(
                    f"Auditor [{self.agent_type.value}]: summary batch "
                    f"{batch_idx + 1}/{len(summary_batches)}"
                )
                issues = await self._audit_summary_batch(batch, full_prompt, brain_summary["global_context"])
                all_issues.extend(issues)
                await self.check_cost_ceiling()

        # ── GAP-20: validate findings (hallucination filter) ─────────────────
        if self.validate_findings_enabled and all_issues:
            all_issues = await self._validate_findings(all_issues)

        new_issues = await self._dedup_and_store(all_issues)
        self.log.info(
            f"Auditor [{self.agent_type.value}]: {len(new_issues)} new issues "
            f"(from {len(all_issues)} raw)"
        )
        return new_issues

    # ── Brain summary building ────────────────────────────────────────────────

    async def _build_brain_summary(self) -> dict[str, Any]:
        files      = await self.storage.list_files()
        read_files = [f for f in files if f.status.value == "READ"]

        direct_files:  list[str] = []  # full source text
        summary_files: list[str] = []  # observation-based summaries

        for f in read_files:
            if f.size_lines <= DIRECT_AUDIT_LINE_THRESHOLD and self.repo_root:
                # Try to read actual source
                try:
                    from sandbox.executor import validate_path_within_root
                    validate_path_within_root(f.path, self.repo_root)
                    abs_path = (self.repo_root / f.path).resolve()
                    content  = abs_path.read_text(encoding="utf-8", errors="replace")
                    entry = (
                        f"FILE: {f.path}\n"
                        f"Language: {f.language} | Lines: {f.size_lines}\n"
                        f"=== SOURCE ===\n"
                        + wrap_content(content[:12_000])   # cap at 12k chars per file
                    )
                    direct_files.append(entry)
                    continue
                except Exception:
                    pass  # fall through to summary mode

            # Summary mode (large files or no repo_root)
            chunks = await self.storage.get_chunks(f.path)
            observations: list[str] = []
            deps: list[str]         = []
            for c in chunks:
                observations.extend(c.raw_observations)
                deps.extend(c.dependencies)

            summary_entry = (
                f"FILE: {f.path}\n"
                f"Language: {f.language} | Lines: {f.size_lines}\n"
                f"Summary: {f.summary}\n"
                f"Dependencies: {', '.join(set(deps))[:300]}\n"
                f"Observations: {chr(10).join(observations[:20])}\n"
                "(NOTE: full source not provided — this file exceeds the direct-audit threshold)\n"
            )
            summary_files.append(summary_entry)

        global_context = (
            f"Repository contains {len(files)} total files, "
            f"{len(read_files)} fully read.\n"
            f"Languages: {', '.join(set(f.language for f in read_files))}\n"
            f"Domain mode: {self.domain_mode.value}\n"
        )

        return {
            "file_count":    len(read_files),
            "direct_count":  len(direct_files),
            "direct_files":  direct_files,
            "summary_files": summary_files,
            "global_context": global_context,
        }

    def _batch_files(
        self, files: list[str], batch_size: int = AUDIT_BATCH_SIZE
    ) -> list[list[str]]:
        batches: list[list[str]] = []
        for i in range(0, len(files), batch_size):
            batches.append(files[i:i + batch_size])
        return batches if batches else [[]]

    # ── Direct-source audit batch ─────────────────────────────────────────────

    async def _audit_direct_batch(
        self,
        file_entries:   list[str],
        master_prompt:  str,
        global_context: str,
    ) -> list[Issue]:
        system = self.build_system_prompt(
            DOMAIN_SYSTEM_PROMPTS.get(self.agent_type, "code auditor")
        )

        files_text = wrap_content("\n---\n".join(file_entries))

        prompt = (
            f"# AUDIT DOMAIN: {self.agent_type.value} | MODE: {self.domain_mode.value}\n\n"
            f"## Global Repository Context\n{global_context}\n\n"
            f"## Master Audit Specification\n{wrap_content(master_prompt)}\n\n"
            f"## Files to Audit (FULL SOURCE PROVIDED)\n{files_text}\n\n"
            "## Your Task\n"
            "Audit every function in every file above against the master specification. "
            "You have the actual source code — audit line by line for the domain you specialise in. "
            "Report EVERY issue with exact file path and line numbers. "
            f"Focus only on {self.agent_type.value} domain issues. "
            "Set confidence 0.0-1.0 per finding. "
            "Be exhaustive — a missed CRITICAL issue is worse than a false positive."
        )

        output = await self.call_llm_structured(
            prompt=prompt,
            response_model=AuditOutput,
            system=system,
        )
        return self._convert_issues(output)

    # ── Summary-based audit batch ─────────────────────────────────────────────

    async def _audit_summary_batch(
        self,
        file_summaries: list[str],
        master_prompt:  str,
        global_context: str,
    ) -> list[Issue]:
        system = self.build_system_prompt(
            DOMAIN_SYSTEM_PROMPTS.get(self.agent_type, "code auditor")
        )

        files_text = wrap_content("\n---\n".join(file_summaries))

        prompt = (
            f"# AUDIT DOMAIN: {self.agent_type.value} | MODE: {self.domain_mode.value}\n\n"
            f"## Global Repository Context\n{global_context}\n\n"
            f"## Master Audit Specification\n{wrap_content(master_prompt)}\n\n"
            f"## File Summaries and Observations\n{files_text}\n\n"
            "## Your Task\n"
            "Audit every file against the master specification using the summaries and "
            "observations provided (full source unavailable — files exceed size threshold). "
            "Report EVERY issue — do not skip minor issues, do not consolidate issues in "
            "different files. For each issue, provide exact file path and best-estimate "
            "line numbers. "
            f"Focus only on {self.agent_type.value} domain issues. "
            "Set confidence 0.0-1.0 per finding. Note that confidence should be slightly "
            "lower when based on summaries rather than direct source."
        )

        output = await self.call_llm_structured(
            prompt=prompt,
            response_model=AuditOutput,
            system=system,
        )
        return self._convert_issues(output)

    def _convert_issues(self, output: AuditOutput) -> list[Issue]:
        return [
            Issue(
                run_id=self.run_id,
                severity=ai.severity,
                file_path=ai.file_path,
                line_start=ai.line_start,
                line_end=ai.line_end,
                executor_type=self.agent_type,
                master_prompt_section=ai.master_prompt_section,
                description=ai.description,
                fix_requires_files=ai.fix_requires_files or [ai.file_path],
                status=IssueStatus.OPEN,
                fingerprint=self._fingerprint_issue(ai),
                consensus_votes=1,
                consensus_confidence=ai.confidence,
                created_at=datetime.now(tz=timezone.utc),
            )
            for ai in output.issues
        ]

    # ── GAP-20: finding validation ────────────────────────────────────────────

    async def _validate_findings(self, issues: list[Issue]) -> list[Issue]:
        """
        Second-pass hallucination filter.

        For each issue, ask the LLM: "Is this finding actually present in the
        code, or did you hallucinate it?"  Findings the validator rejects are
        downgraded to INFO (not discarded) so they remain visible for review.

        We batch up to 20 validations per LLM call to keep costs manageable.
        Only validates issues where we have source available (small files).
        """
        if not self.repo_root:
            return issues  # no source access, skip validation

        # Only validate findings for files we can read
        to_validate: list[Issue]    = []
        skip_validate: list[Issue]  = []

        for issue in issues:
            try:
                from sandbox.executor import validate_path_within_root
                validate_path_within_root(issue.file_path, self.repo_root)
                abs_path = (self.repo_root / issue.file_path).resolve()
                if abs_path.exists():
                    to_validate.append(issue)
                    continue
            except Exception:
                pass
            skip_validate.append(issue)

        if not to_validate:
            return issues

        validated: list[Issue] = list(skip_validate)

        batch_size = 20
        for i in range(0, len(to_validate), batch_size):
            batch = to_validate[i:i + batch_size]
            validated.extend(await self._validate_batch(batch))

        return validated

    async def _validate_batch(self, issues: list[Issue]) -> list[Issue]:
        """Validate a single batch of findings against their source files."""
        # Build context: one source snippet per unique file
        file_contents: dict[str, str] = {}
        for issue in issues:
            if issue.file_path not in file_contents and self.repo_root:
                try:
                    abs_path = (self.repo_root / issue.file_path).resolve()
                    content  = abs_path.read_text(encoding="utf-8", errors="replace")
                    # Only include lines around the finding
                    lines = content.splitlines()
                    start = max(0, issue.line_start - 10)
                    end   = min(len(lines), issue.line_end + 10)
                    file_contents[issue.file_path] = "\n".join(lines[start:end])
                except Exception:
                    pass

        issue_list = "\n".join(
            f"ID: {iss.id} | File: {iss.file_path} L{iss.line_start}-{iss.line_end}\n"
            f"  Finding: {iss.description[:200]}\n"
            for iss in issues
        )
        source_ctx = "\n\n".join(
            f"=== {path} ===\n{wrap_content(snippet)}"
            for path, snippet in file_contents.items()
        )

        system = self.build_system_prompt(
            "senior code auditor performing validation of claimed findings"
        )
        prompt = (
            "## Findings to Validate\n"
            f"{issue_list}\n\n"
            "## Relevant Source Excerpts\n"
            f"{source_ctx}\n\n"
            "## Your Task\n"
            "For each finding ID above, confirm whether it is actually present in the "
            "source excerpts shown.  A finding is confirmed if the described issue is "
            "directly observable in the code.  A finding should be rejected ONLY if it "
            "is clearly not present (hallucinated).  When in doubt, confirm it."
        )

        try:
            validation = await self.call_llm_structured(
                prompt=prompt,
                response_model=FindingValidationBatch,
                system=system,
                model_override=self.config.triage_model,  # cheap model for validation
            )
        except Exception as exc:
            self.log.warning(f"Finding validation failed: {exc} — keeping all findings")
            return issues

        validation_map = {v.issue_id: v for v in validation.validations}
        result: list[Issue] = []
        for issue in issues:
            v = validation_map.get(issue.id)
            if v and not v.confirmed:
                # Downgrade to INFO rather than discard — remains visible
                self.log.debug(
                    f"Validator rejected finding {issue.id}: {v.reason[:100]}. "
                    "Downgrading to INFO."
                )
                issue.severity = Severity.INFO
                issue.description = f"[VALIDATOR DOWNGRADED] {issue.description}"
            result.append(issue)

        return result

    # ── Dedup and store ───────────────────────────────────────────────────────

    def _fingerprint_issue(self, issue: "AuditIssue") -> str:
        raw = f"{issue.file_path}:{issue.line_start}:{issue.description[:80]}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    async def _dedup_and_store(self, issues: list[Issue]) -> list[Issue]:
        new_issues: list[Issue] = []

        for issue in issues:
            fp = await self.storage.get_fingerprint(issue.fingerprint)
            if fp:
                if fp.seen_count >= MAX_FIX_ATTEMPTS:
                    issue.status = IssueStatus.ESCALATED
                    issue.escalated_reason = (
                        f"Issue seen {fp.seen_count} times without convergence"
                    )
                    self.log.warning(f"Escalating persistent issue: {issue.id}")
                fp.seen_count += 1
                fp.last_seen  = datetime.now(tz=timezone.utc)
                await self.storage.upsert_fingerprint(fp)
            else:
                await self.storage.upsert_fingerprint(IssueFingerprint(
                    fingerprint=issue.fingerprint,
                    issue_id=issue.id,
                ))

            await self.storage.upsert_issue(issue)
            new_issues.append(issue)

        return new_issues
