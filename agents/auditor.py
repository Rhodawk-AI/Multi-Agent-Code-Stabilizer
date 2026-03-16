"""
agents/auditor.py
Phase 2: Auditor Agent.
Synthesizes the brain into a structured issue list against the master prompt.
Runs in parallel across Security / Architecture / Standards domains.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agents.base import AgentConfig, BaseAgent
from brain.schemas import (
    ExecutorType,
    Issue,
    IssueFingerprint,
    IssueStatus,
    Severity,
)
from brain.storage import BrainStorage

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Structured output model for auditors
# ─────────────────────────────────────────────────────────────

class AuditIssue(BaseModel):
    severity:              Severity
    file_path:             str = Field(description="Relative path to the affected file")
    line_start:            int = Field(ge=0)
    line_end:              int = Field(ge=0)
    master_prompt_section: str = Field(description="Which section of the master prompt is violated")
    description:           str = Field(description="Precise, actionable description of the issue")
    fix_requires_files:    list[str] = Field(
        default_factory=list,
        description="All files needed to implement the fix (may be >1 for cross-file issues)"
    )


class AuditOutput(BaseModel):
    domain:     str
    issues:     list[AuditIssue] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.85)
    notes:      str = ""


# ─────────────────────────────────────────────────────────────
# Domain audit prompts
# ─────────────────────────────────────────────────────────────

DOMAIN_SYSTEM_PROMPTS: dict[ExecutorType, str] = {
    ExecutorType.SECURITY: (
        "security auditor specializing in CERT, CWE, OWASP, and OS-level security. "
        "Focus on: injection vulnerabilities, unsafe deserialization, credential exposure, "
        "privilege escalation, sandboxing failures, and supply-chain risks."
    ),
    ExecutorType.ARCHITECTURE: (
        "principal software architect auditing for GII (General Interactive Intelligence) "
        "architectural correctness. Focus on: cognitive loop integrity, safety gate completeness, "
        "consequence-grounded reasoning, exception propagation, state machine correctness, "
        "dead code paths, and cross-component invariant violations."
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
}


class AuditorAgent(BaseAgent):
    """
    Synthesizes the accumulated brain into a structured issue list.
    One instance runs per domain (Security / Architecture / Standards).
    Can be parallelised.
    """

    def __init__(
        self,
        storage: BrainStorage,
        run_id: str,
        executor_type: ExecutorType,
        master_prompt_path: str | Path,
        config: AgentConfig | None = None,
        mcp_manager: Any | None = None,
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.agent_type         = executor_type
        self.master_prompt_path = Path(master_prompt_path)
        self._master_prompt: str | None = None

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

    async def run(self, **kwargs: Any) -> list[Issue]:  # type: ignore[override]
        """
        Run audit for this domain. Returns list of Issues written to the brain.
        """
        master_prompt = await self._load_master_prompt()
        brain_summary = await self._build_brain_summary()

        self.log.info(
            f"Auditor [{self.agent_type.value}]: running audit on "
            f"{brain_summary['file_count']} files"
        )

        # Split brain into batches to handle large codebases
        file_batches = self._batch_files(brain_summary["file_summaries"])
        all_issues: list[Issue] = []

        for batch_idx, batch in enumerate(file_batches):
            self.log.info(
                f"Auditor [{self.agent_type.value}]: batch {batch_idx + 1}/{len(file_batches)}"
            )
            batch_issues = await self._audit_batch(
                batch, master_prompt, brain_summary["global_context"]
            )
            all_issues.extend(batch_issues)
            await self.check_cost_ceiling()

        # Dedup by fingerprint and write to brain
        new_issues = await self._dedup_and_store(all_issues)
        self.log.info(
            f"Auditor [{self.agent_type.value}]: found {len(new_issues)} new issues"
        )
        return new_issues

    async def _build_brain_summary(self) -> dict[str, Any]:
        """Compile the brain into a compact, LLM-consumable summary."""
        files = await self.storage.list_files()
        read_files = [f for f in files if f.status.value == "READ"]

        file_summaries: list[str] = []
        for f in read_files:
            chunks = await self.storage.get_chunks(f.path)
            observations = []
            for c in chunks:
                observations.extend(c.raw_observations)

            deps = []
            for c in chunks:
                deps.extend(c.dependencies)

            summary_text = (
                f"FILE: {f.path}\n"
                f"Language: {f.language} | Lines: {f.size_lines}\n"
                f"Summary: {f.summary}\n"
                f"Dependencies: {', '.join(set(deps))[:200]}\n"
                f"Observations: {chr(10).join(observations[:20])}\n"
            )
            file_summaries.append(summary_text)

        global_context = (
            f"Repository contains {len(files)} total files, "
            f"{len(read_files)} fully read.\n"
            f"Languages: {', '.join(set(f.language for f in read_files))}\n"
        )

        return {
            "file_count": len(read_files),
            "file_summaries": file_summaries,
            "global_context": global_context,
        }

    def _batch_files(self, summaries: list[str], batch_size: int = 30) -> list[list[str]]:
        """Split file summaries into batches for audit."""
        batches = []
        for i in range(0, len(summaries), batch_size):
            batches.append(summaries[i:i + batch_size])
        return batches if batches else [[]]

    async def _audit_batch(
        self,
        file_summaries: list[str],
        master_prompt: str,
        global_context: str,
    ) -> list[Issue]:
        system = self.build_system_prompt(
            DOMAIN_SYSTEM_PROMPTS.get(self.agent_type, "code auditor")
        )

        files_text = "\n---\n".join(file_summaries)

        prompt = (
            f"# AUDIT DOMAIN: {self.agent_type.value}\n\n"
            f"## Global Repository Context\n{global_context}\n\n"
            f"## Master Audit Specification\n{master_prompt}\n\n"
            f"## File Summaries and Observations\n{files_text}\n\n"
            "## Your Task\n"
            "Audit every file against the master specification. "
            "Report EVERY issue — do not skip minor issues, do not consolidate issues that are in different files. "
            "For each issue, provide exact file path and line numbers from the observations above. "
            f"Focus only on {self.agent_type.value} domain issues. "
            "Be exhaustive. Missing a CRITICAL issue is worse than a false positive."
        )

        output = await self.call_llm_structured(
            prompt=prompt,
            response_model=AuditOutput,
            system=system,
        )

        return [
            Issue(
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
                created_at=datetime.utcnow(),
            )
            for ai in output.issues
        ]

    def _fingerprint_issue(self, issue: AuditIssue) -> str:
        raw = f"{issue.file_path}:{issue.line_start}:{issue.description[:80]}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    async def _dedup_and_store(self, issues: list[Issue]) -> list[Issue]:
        """
        Dedup against existing fingerprints.
        Issues seen > MAX_FIX_ATTEMPTS times are escalated, not queued.
        """
        MAX_FIX_ATTEMPTS = 3
        new_issues: list[Issue] = []

        for issue in issues:
            fp = await self.storage.get_fingerprint(issue.fingerprint)
            if fp:
                if fp.seen_count >= MAX_FIX_ATTEMPTS:
                    issue.status = IssueStatus.ESCALATED
                    issue.escalated_reason = (
                        f"Issue seen {fp.seen_count} times without convergence"
                    )
                    self.log.warning(
                        f"Escalating persistent issue: {issue.id} in {issue.file_path}"
                    )
                fp.seen_count += 1
                fp.last_seen = datetime.utcnow()
                await self.storage.upsert_fingerprint(fp)
            else:
                await self.storage.upsert_fingerprint(IssueFingerprint(
                    fingerprint=issue.fingerprint,
                    issue_id=issue.id,
                ))

            await self.storage.upsert_issue(issue)
            new_issues.append(issue)

        return new_issues
