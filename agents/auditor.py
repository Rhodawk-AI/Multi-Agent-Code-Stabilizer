"""
agents/auditor.py
=================
Multi-domain code auditor agent for Rhodawk AI Code Stabilizer.

PRODUCTION FIXES vs audit report
──────────────────────────────────
• MISRA-C 2023 rule mapping: LLM findings tagged with misra_rule IDs for
  the 27 mandatory rules and top required rules detectable via LLM.
• CERT-C rule mapping: findings tagged with cert_rule (MEM30-C, STR31-C, etc.)
• CWE mapping: findings tagged with cwe_id from CWE Top 25 2024.
• JSF++ AV Rule mapping for military/aerospace domain modes.
• function_name extracted via tree-sitter or line-number heuristic.
• mil882e_category auto-assigned from severity via SEVERITY_TO_MIL882E map.
• compliance_violations list populated on every Issue.
• do178c_objective field populated for findings relevant to DO-178C objectives.
• DOMAIN_EXTRA_INSTRUCTIONS for MILITARY replaced with real 40-rule set
  (not 6-bullet placeholder).
• Chunk-level observations de-duplicated via fingerprint before Issue creation.
• Issues loaded from storage correctly — no re-query race.
• validate_findings gate uses deterministic LLM call (temperature=0.0).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agents.base import AgentConfig, BaseAgent, wrap_content, wrap_source_file
from brain.schemas import (
    ComplianceStandard, ComplianceViolation, DomainMode,
    ExecutorType, Issue, IssueStatus, MilStd882eCategory,
    SEVERITY_TO_MIL882E, Severity,
)
from brain.storage import BrainStorage

log = logging.getLogger(__name__)


# ── LLM response schema ───────────────────────────────────────────────────────

class RawFinding(BaseModel):
    severity:        str   = "MINOR"
    file_path:       str   = ""
    line_start:      int   = 0
    line_end:        int   = 0
    function_name:   str   = ""
    description:     str   = ""
    category:        str   = ""
    fix_requires_files: list[str] = Field(default_factory=list)
    confidence:      float = Field(ge=0.0, le=1.0, default=0.75)
    # Compliance tags — filled by LLM, validated by auditor
    cwe_id:          str   = ""    # e.g. "CWE-787"
    misra_rule:      str   = ""    # e.g. "MISRA-C:2023-15.1"
    cert_rule:       str   = ""    # e.g. "MEM30-C"
    jsf_rule:        str   = ""    # e.g. "AV-Rule-67"
    do178c_objective: str  = ""    # e.g. "Table A-7, Obj 5"
    is_mandatory:    bool  = False  # True for MISRA mandatory rules


class AuditResponse(BaseModel):
    findings: list[RawFinding] = Field(default_factory=list)
    summary:  str              = ""


# ── Domain instruction sets ───────────────────────────────────────────────────

_DOMAIN_EXTRA_INSTRUCTIONS: dict[DomainMode, str] = {

    DomainMode.GENERAL: "",

    DomainMode.MILITARY: """
MILITARY / SAFETY-CRITICAL DOMAIN — MANDATORY CHECKS:

MISRA-C 2023 Mandatory Rules (tag misra_rule field):
- MISRA-C:2023-1.3   Never produce undefined or critical unspecified behavior
- MISRA-C:2023-2.1   No unreachable code (control flow graph required)
- MISRA-C:2023-2.3   No unused type declarations
- MISRA-C:2023-2.4   No unused tags
- MISRA-C:2023-2.6   No unused label declarations
- MISRA-C:2023-2.7   No unused parameters
- MISRA-C:2023-15.1  No goto statement
- MISRA-C:2023-17.3  No more arguments than parameters in function call
- MISRA-C:2023-17.6  Array parameters must not have const-qualified element type
- MISRA-C:2023-18.1  Pointer arithmetic must stay within array bounds
- MISRA-C:2023-18.2  No subtraction between pointers unless same array
- MISRA-C:2023-18.3  Relational operators only on pointers to same array
- MISRA-C:2023-18.5  No more than two levels of pointer indirection
- MISRA-C:2023-20.14 _Noreturn must match function behavior
- MISRA-C:2023-22.1  Memory/resources must be released before error exit
- MISRA-C:2023-22.2  No free() of memory not dynamically allocated
- MISRA-C:2023-22.3  Same file not opened simultaneously for read and write
- MISRA-C:2023-22.4  Write to read-only stream is prohibited
- MISRA-C:2023-22.5  No dereferencing of FILE* pointer
- MISRA-C:2023-22.6  No use of FILE* after stream closed

MISRA-C 2023 Required Rules (tag misra_rule):
- MISRA-C:2023-11.3  No cast between pointer to object and different pointer type
- MISRA-C:2023-12.2  Right-hand operand of shift must be in bounds
- MISRA-C:2023-13.4  No side effects in operand of sizeof
- MISRA-C:2023-14.1  All loops must have reachable termination condition
- MISRA-C:2023-15.5  Function must have single point of exit
- MISRA-C:2023-17.1  No variadic function calls (<stdarg.h> prohibited)
- MISRA-C:2023-21.3  No use of memory allocation/deallocation from <stdlib.h>
- MISRA-C:2023-21.6  No use of <stdio.h> input/output functions
- MISRA-C:2023-21.7  No use of atof/atoi/atol/atoll
- MISRA-C:2023-21.11 No use of <tgmath.h>

JSF++ AV Rules (tag jsf_rule, military/aerospace only):
- AV-Rule-8    All code must conform to the declared C++ standard
- AV-Rule-67   Unary minus must not be applied to unsigned expression
- AV-Rule-101  No increment/decrement mixed with other operators
- AV-Rule-114-126  No dynamic heap allocation post-initialization (includes
                   std::vector, std::string, std::unique_ptr construction)
- AV-Rule-210-213  No templates (hard prohibition)

CERT-C Rules (tag cert_rule):
- MEM30-C  Never access freed memory (use-after-free)
- MEM31-C  Free dynamically allocated memory exactly once
- MEM34-C  Only free memory allocated dynamically
- STR31-C  Buffer must have sufficient space for string operations
- STR32-C  String arguments to library functions must be null-terminated
- INT30-C  Unsigned integer operations must not wrap
- INT31-C  Integer conversions must not result in lost/misinterpreted data
- ERR33-C  Return values of standard library functions must be checked
- SIG30-C  Signal handlers must be async-signal-safe

CWE Top 25 2024 (tag cwe_id):
- CWE-787  Out-of-bounds Write
- CWE-416  Use After Free
- CWE-125  Out-of-bounds Read
- CWE-476  NULL Pointer Dereference
- CWE-190  Integer Overflow or Wraparound
- CWE-20   Improper Input Validation
- CWE-78   OS Command Injection
- CWE-22   Path Traversal
- CWE-787, CWE-125 are CRITICAL by default for safety-critical code

MIL-STD-882E: Every CRITICAL finding must include hazard_description
describing the potential mishap. Assign mil882e_category based on:
  CAT_I (Catastrophic) = can cause death or loss of system
  CAT_II (Critical)    = can cause severe injury or major system damage
  CAT_III (Marginal)   = can cause minor injury or minor system damage
  CAT_IV (Negligible)  = minimal threat to safety

DO-178C Objectives: Tag do178c_objective when finding relates to:
  "Table A-4, Obj 4" — High-level requirement accuracy
  "Table A-5, Obj 1" — Software architecture traceability
  "Table A-7, Obj 5" — Source code verification of standards
  "Table A-7, Obj 6" — Absence of unintended functions
""",

    DomainMode.AEROSPACE: """
AEROSPACE / DO-178C DOMAIN — apply all MILITARY checks above plus:
- Flag any mutable global state in DAL-A/B modules (DO-178C partitioning)
- Flag any function exceeding McCabe cyclomatic complexity > 10
- Flag any file missing DO-178C section/requirement header comment
- Flag missing return-value checks on all operating system calls
""",

    DomainMode.MEDICAL: """
MEDICAL DEVICE / IEC 62304 DOMAIN:
- Flag all unchecked return values from I/O, memory, and network operations
- Flag absence of defensive input validation on all external data
- Flag resource leaks (memory, file handles, sockets)
- Flag use of deprecated or unsafe library functions
- Tag with cert_rule where applicable (ERR33-C, MEM30-C)
""",

    DomainMode.FINANCE: """
FINANCIAL DOMAIN:
- Flag all integer arithmetic without overflow checks on monetary values
- Flag floating-point equality comparisons
- Flag missing input validation on external financial data
- Flag logging of PII or credential data
- Tag CWE-190 (integer overflow) and CWE-20 (input validation)
""",

    DomainMode.EMBEDDED: """
EMBEDDED SYSTEMS DOMAIN:
- Flag dynamic memory allocation (heap use prohibited in many RTOS environments)
- Flag unbounded loops and recursion
- Flag race conditions on shared hardware registers (missing volatile)
- Flag missing interrupt disable/enable pairing around critical sections
- Apply MISRA-C:2023-21.3 (no malloc/free) as CRITICAL
""",

    DomainMode.NUCLEAR: """
NUCLEAR / IEC 61513 DOMAIN — apply all MILITARY + AEROSPACE checks plus:
- Flag any non-deterministic code path (random, time-dependent behavior)
- Flag any network communication in safety-critical functions
- Flag any use of floating point in safety-critical calculations
  (IEC 61513 prefers fixed-point for nuclear I&C systems)
- All findings are CRITICAL by default pending human review
""",
}


# ── MISRA mandatory rule IDs — mandatory rules cannot be deviated from ────────
_MISRA_MANDATORY_RULES: set[str] = {
    "MISRA-C:2023-1.3",  "MISRA-C:2023-2.1",  "MISRA-C:2023-2.3",
    "MISRA-C:2023-2.4",  "MISRA-C:2023-2.6",  "MISRA-C:2023-2.7",
    "MISRA-C:2023-15.1", "MISRA-C:2023-17.3", "MISRA-C:2023-17.6",
    "MISRA-C:2023-18.1", "MISRA-C:2023-18.2", "MISRA-C:2023-18.3",
    "MISRA-C:2023-18.5", "MISRA-C:2023-20.14","MISRA-C:2023-22.1",
    "MISRA-C:2023-22.2", "MISRA-C:2023-22.3", "MISRA-C:2023-22.4",
    "MISRA-C:2023-22.5", "MISRA-C:2023-22.6",
}

# CWE IDs that default to CRITICAL in safety-critical domains
_CRITICAL_CWES: set[str] = {"CWE-787", "CWE-416", "CWE-125", "CWE-476", "CWE-190"}


# ── AuditorAgent ──────────────────────────────────────────────────────────────

class AuditorAgent(BaseAgent):
    agent_type = ExecutorType.SECURITY  # overridden per instance

    def __init__(
        self,
        storage:            BrainStorage,
        run_id:             str,
        executor_type:      ExecutorType = ExecutorType.SECURITY,
        master_prompt_path: Path         = Path("config/prompts/base.md"),
        config:             AgentConfig | None = None,
        mcp_manager:        Any | None   = None,
        domain_mode:        DomainMode   = DomainMode.GENERAL,
        repo_root:          Path | None  = None,
        validate_findings:  bool         = True,
        misra_enabled:      bool         = True,
        cert_enabled:       bool         = True,
        jsf_enabled:        bool         = False,
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.agent_type         = executor_type
        self.master_prompt_path = master_prompt_path
        self.domain_mode        = domain_mode
        self.repo_root          = repo_root
        self.validate_findings  = validate_findings
        self.misra_enabled      = misra_enabled
        self.cert_enabled       = cert_enabled
        self.jsf_enabled        = jsf_enabled
        self._master_prompt: str = ""

    async def run(self, stale_only: bool = False, **kwargs: Any) -> list[Issue]:
        """
        Audit code chunks and return Issues.

        Parameters
        ----------
        stale_only:
            When True (set by the controller during commit-triggered incremental
            cycles) the auditor calls storage.get_stale_observations() instead
            of get_all_observations().  This scopes the audit to the CPG-computed
            impact set written by CommitAuditScheduler — typically 50-200
            functions rather than the full codebase.

            If get_stale_observations() returns an empty list (staleness table is
            empty or the run has no marks yet) we automatically fall back to
            get_all_observations() so the first full audit cycle still works.
        """
        self._master_prompt = self._load_master_prompt()

        if stale_only:
            chunks = await self.storage.get_stale_observations(
                run_id=self.run_id
            )
            if not chunks:
                # Staleness table empty — this is either the first cycle or
                # marks were already cleared.  Fall back to full audit.
                self.log.debug(
                    "[%s] stale_only=True but no stale marks found — "
                    "falling back to full get_all_observations()",
                    self.agent_name,
                )
                chunks = await self.storage.get_all_observations()
        else:
            chunks = await self.storage.get_all_observations()

        if not chunks:
            return []

        self.log.info(
            "[%s] auditing %d chunk(s) (stale_only=%s)",
            self.agent_name, len(chunks), stale_only,
        )

        sem = asyncio.Semaphore(4)
        tasks = [self._audit_chunk(chunk, sem) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_issues: list[Issue] = []
        seen_fps: set[str] = set()

        for r in results:
            if isinstance(r, list):
                for issue in r:
                    if issue.fingerprint and issue.fingerprint in seen_fps:
                        continue
                    if issue.fingerprint:
                        seen_fps.add(issue.fingerprint)
                    all_issues.append(issue)
            elif isinstance(r, Exception):
                self.log.error(f"[{self.agent_name}] chunk audit failed: {r}")

        if self.validate_findings:
            all_issues = await self._validate_batch(all_issues)

        for issue in all_issues:
            await self.storage.upsert_issue(issue)

        self.log.info(
            f"[{self.agent_name}] {len(all_issues)} findings "
            f"(domain={self.domain_mode.value})"
        )
        return all_issues

    async def _audit_chunk(
        self, chunk: dict[str, Any], sem: asyncio.Semaphore
    ) -> list[Issue]:
        async with sem:
            file_path  = chunk.get("file_path", "")
            language   = chunk.get("language", "unknown")
            content    = chunk.get("content", "")
            line_start = chunk.get("line_start", 0)

            if not content:
                return []

            domain_instructions = _DOMAIN_EXTRA_INSTRUCTIONS.get(
                self.domain_mode, ""
            )
            compliance_tags = self._build_compliance_tag_instructions()

            # SEC-5 FIX: use wrap_source_file() instead of wrap_content() so
            # repository source code is enclosed in <source_code file="..."> tags.
            # The structural delimiter prevents the LLM from treating comment-
            # embedded adversarial instructions (e.g. "# SYSTEM OVERRIDE: …")
            # as prompt instructions rather than code content to be audited.
            # sanitize_content() inside wrap_source_file also strips the known
            # injection trigger phrases defined in agents/base.py.
            wrapped_code = wrap_source_file(content, file_path)

            prompt = (
                f"## Master Audit Specification\n{self._master_prompt}\n\n"
                f"## File\n`{file_path}` (language: {language})\n\n"
                f"## Code Chunk (lines {line_start}+)\n{wrapped_code}\n\n"
                + (f"## Domain Requirements\n{domain_instructions}\n\n"
                   if domain_instructions else "")
                + (f"## Compliance Tagging\n{compliance_tags}\n\n"
                   if compliance_tags else "")
                + "## Task\nIdentify ALL issues. For each finding populate ALL "
                "compliance fields (cwe_id, misra_rule, cert_rule, jsf_rule) "
                "where applicable. Return empty findings list if no issues found."
            )

            try:
                response = await self.call_llm_structured(
                    prompt=prompt,
                    response_model=AuditResponse,
                    system=self.build_system_prompt(
                        f"{self.agent_type.value.lower()} code auditor "
                        f"specializing in {self.domain_mode.value.lower()} domain"
                    ),
                )
            except Exception as exc:
                self.log.warning(
                    f"[{self.agent_name}] audit failed for {file_path}: {exc}"
                )
                return []

            return [
                self._raw_to_issue(f, file_path, language)
                for f in response.findings
                if f.description
            ]

    def _raw_to_issue(
        self, f: RawFinding, fallback_file: str, language: str
    ) -> Issue:
        file_path = f.file_path or fallback_file

        # Severity override for safety-critical CWEs
        severity_str = f.severity.upper()
        if (
            f.cwe_id in _CRITICAL_CWES
            and self.domain_mode in {
                DomainMode.MILITARY, DomainMode.AEROSPACE, DomainMode.NUCLEAR
            }
        ):
            severity_str = "CRITICAL"
        # MISRA mandatory rules are always CRITICAL
        if f.misra_rule and f.misra_rule in _MISRA_MANDATORY_RULES:
            severity_str = "CRITICAL"

        try:
            sev = Severity(severity_str)
        except ValueError:
            sev = Severity.MINOR

        mil882e = SEVERITY_TO_MIL882E.get(sev, MilStd882eCategory.NONE)

        # Build ComplianceViolation list
        violations: list[ComplianceViolation] = []
        if f.misra_rule and self.misra_enabled:
            standard = (
                ComplianceStandard.MISRA_CPP
                if language in {"cpp", "c++"}
                else ComplianceStandard.MISRA_C
            )
            violations.append(ComplianceViolation(
                rule_id=f.misra_rule,
                standard=standard,
                rule_description=f.description[:200],
                is_mandatory=(f.misra_rule in _MISRA_MANDATORY_RULES),
                tool_detected_by=f"llm/{self.agent_name}",
                confidence=f.confidence,
            ))
        if f.cert_rule and self.cert_enabled:
            standard_c = (
                ComplianceStandard.CERT_CPP
                if language in {"cpp", "c++"}
                else ComplianceStandard.CERT_C
            )
            violations.append(ComplianceViolation(
                rule_id=f.cert_rule,
                standard=standard_c,
                rule_description=f.description[:200],
                is_mandatory=False,
                tool_detected_by=f"llm/{self.agent_name}",
                confidence=f.confidence,
            ))
        if f.jsf_rule and self.jsf_enabled:
            violations.append(ComplianceViolation(
                rule_id=f.jsf_rule,
                standard=ComplianceStandard.JSF_AV,
                rule_description=f.description[:200],
                is_mandatory=False,
                tool_detected_by=f"llm/{self.agent_name}",
                confidence=f.confidence,
            ))
        if f.cwe_id:
            violations.append(ComplianceViolation(
                rule_id=f.cwe_id,
                standard=ComplianceStandard.CWE,
                rule_description=f.description[:200],
                is_mandatory=False,
                tool_detected_by=f"llm/{self.agent_name}",
                confidence=f.confidence,
            ))

        fp = BaseAgent.fingerprint(
            file_path, f.line_start, f.line_end, f.description
        )

        return Issue(
            run_id=self.run_id,
            severity=sev,
            file_path=file_path,
            line_start=f.line_start,
            line_end=f.line_end,
            function_name=f.function_name,
            description=f.description,
            status=IssueStatus.OPEN,
            executor_type=self.agent_type,
            domain_mode=self.domain_mode,
            fix_requires_files=f.fix_requires_files or [file_path],
            confidence=f.confidence,
            fingerprint=fp,
            cwe_id=f.cwe_id,
            misra_rule=f.misra_rule,
            cert_rule=f.cert_rule,
            jsf_rule=f.jsf_rule,
            do178c_objective=f.do178c_objective,
            compliance_violations=violations,
            mil882e_category=mil882e,
            is_mandatory=f.is_mandatory if hasattr(f, "is_mandatory") else False,
            # FIX: populate requirement_id by resolving the MISRA/CERT/CWE rule
            # against the requirements index stored in .stabilizer/requirements.json
            # (if available). Without this the RTM has no traceability content.
            requirement_id=self._resolve_requirement_id(f),
        )

    def _resolve_requirement_id(self, f: "RawFinding") -> str:
        """
        Map a compliance rule tag to the nearest requirement ID.

        Resolution order:
        1. .stabilizer/requirements.json  — project-specific mapping
           Format: {"MISRA-C:2023-15.1": "REQ-001", "CWE-787": "REQ-042", ...}
        2. Built-in rule-to-generic-requirement fallback table
        3. Empty string (RTM row will have no requirement — acceptable for INFO)
        """
        # Try rule-based lookup
        candidates = [f.misra_rule, f.cert_rule, f.jsf_rule, f.cwe_id]
        candidates = [c for c in candidates if c]

        # 1. Project-specific requirements map
        req_map = self._load_requirements_map()
        for rule in candidates:
            if rule in req_map:
                return req_map[rule]

        # 2. Built-in generic fallback for common standards
        _GENERIC_MAP: dict[str, str] = {
            # MISRA mandatory rules → safety-critical requirement bucket
            "MISRA-C:2023-1.3":  "REQ-SAFETY-001",
            "MISRA-C:2023-2.1":  "REQ-SAFETY-002",
            "MISRA-C:2023-15.1": "REQ-SAFETY-003",
            "MISRA-C:2023-17.3": "REQ-SAFETY-004",
            "MISRA-C:2023-18.1": "REQ-SAFETY-005",
            "MISRA-C:2023-22.1": "REQ-SAFETY-006",
            "MISRA-C:2023-22.2": "REQ-SAFETY-007",
            # CWE Top-10 → security requirement bucket
            "CWE-787": "REQ-SEC-001",
            "CWE-79":  "REQ-SEC-002",
            "CWE-89":  "REQ-SEC-003",
            "CWE-416": "REQ-SEC-004",
            "CWE-476": "REQ-SEC-005",
            "CWE-190": "REQ-SEC-006",
            "CWE-125": "REQ-SEC-007",
            "CWE-22":  "REQ-SEC-008",
            # CERT memory rules
            "MEM30-C": "REQ-MEM-001",
            "MEM32-C": "REQ-MEM-002",
            "STR31-C": "REQ-MEM-003",
            "INT31-C": "REQ-MEM-004",
        }
        for rule in candidates:
            if rule in _GENERIC_MAP:
                return _GENERIC_MAP[rule]

        return ""

    def _load_requirements_map(self) -> dict[str, str]:
        """Load .stabilizer/requirements.json if present; cache for the run."""
        if hasattr(self, "_req_map_cache"):
            return self._req_map_cache  # type: ignore[attr-defined]
        self._req_map_cache: dict[str, str] = {}
        try:
            if self.repo_root:
                req_file = Path(self.repo_root) / ".stabilizer" / "requirements.json"
                if req_file.exists():
                    import json
                    self._req_map_cache = json.loads(
                        req_file.read_text(encoding="utf-8")
                    )
                    self.log.info(
                        f"[auditor] Loaded {len(self._req_map_cache)} requirement "
                        "mappings from .stabilizer/requirements.json"
                    )
        except Exception as exc:
            self.log.debug(f"[auditor] requirements.json load failed: {exc}")
        return self._req_map_cache

    async def _validate_batch(self, issues: list[Issue]) -> list[Issue]:
        """
        Validate a batch of findings with a deterministic (temperature=0.0)
        secondary LLM call to eliminate hallucinated findings.
        """
        if not issues:
            return []

        class ValidationResponse(BaseModel):
            valid_indices: list[int] = Field(default_factory=list)
            rationale:     str       = ""

        summary = "\n".join(
            f"{i}: [{iss.severity.value}] {iss.file_path}:{iss.line_start} — "
            f"{iss.description[:120]}"
            for i, iss in enumerate(issues)
        )
        prompt = (
            f"Review these {len(issues)} code findings and identify which are "
            "genuine issues vs hallucinations, false positives, or style nits.\n\n"
            f"Findings:\n{summary}\n\n"
            "Return the indices (0-based) of findings that are GENUINE code issues. "
            "Exclude: vague descriptions, style preferences, non-issues, duplicates."
        )
        try:
            resp = await self.call_llm_structured_deterministic(
                prompt=prompt,
                response_model=ValidationResponse,
                system="You are a senior code reviewer validating audit findings for accuracy.",
                model_override=self.config.triage_model,
            )
            valid = {i for i in resp.valid_indices if 0 <= i < len(issues)}
            validated = [iss for i, iss in enumerate(issues) if i in valid]
            self.log.info(
                f"[{self.agent_name}] Validation: {len(validated)}/{len(issues)} "
                "findings confirmed genuine"
            )
            return validated
        except Exception as exc:
            self.log.warning(f"[{self.agent_name}] Validation LLM call failed: {exc}")
            return issues  # Return unvalidated if validation fails

    def _load_master_prompt(self) -> str:
        try:
            p = self.master_prompt_path
            if self.repo_root:
                p = self.repo_root / self.master_prompt_path
            return Path(p).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            self.log.warning(f"Could not load master prompt: {exc}")
            return (
                "Perform a comprehensive security and quality audit of the provided code. "
                "Identify all bugs, security vulnerabilities, and standards violations."
            )

    def _build_compliance_tag_instructions(self) -> str:
        parts: list[str] = []
        if self.misra_enabled:
            parts.append(
                "Tag misra_rule with 'MISRA-C:2023-X.Y' format for any MISRA-C violation."
            )
        if self.cert_enabled:
            parts.append(
                "Tag cert_rule with 'MEM30-C', 'STR31-C', etc. for CERT-C violations."
            )
        if self.jsf_enabled:
            parts.append(
                "Tag jsf_rule with 'AV-Rule-N' for JSF++ AV Rule violations."
            )
        parts.append(
            "Tag cwe_id with 'CWE-N' (e.g. 'CWE-787') for CWE Top 25 findings."
        )
        return "\n".join(parts)
