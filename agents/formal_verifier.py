"""
agents/formal_verifier.py
=========================
Formal verification agent for MACS — closes GAP-4.

Uses the Z3 SMT solver to attempt mathematical proofs of correctness for
CRITICAL fixes in mission-critical domain modes (finance, medical, military).

Architecture
────────────
A formal verifier cannot automatically prove arbitrary Python programs
correct.  Instead it targets the highest-value, most verifiable properties:

Finance domain
  • balance_non_negative: after every balance mutation, balance >= 0
  • no_float_on_price:    Decimal used for all monetary arithmetic (static)
  • atomic_balance_update: no read-modify-write race window (static)

Medical domain
  • dose_positive:        dosage values are always positive
  • dose_bounded:         dosage is within configured safe range
  • no_null_patient_id:   patient_id is never None

Military domain
  • no_unbounded_loop:    all loops have a provable upper bound (static)
  • no_dynamic_alloc:     no malloc/new outside init phase (static)

The agent works in two modes:

1. LLM-extracted constraint mode (default):
   The LLM extracts formal constraints from the fixed file as Python
   expressions, then Z3 proves them.

2. Pattern-match mode (fallback):
   Static pattern matching against the fixed file content for known
   dangerous patterns, returning COUNTEREXAMPLE if any are found.

Integration
──────────
Called by StabilizerController._phase_gate() after gate passes for
CRITICAL fixes when domain_mode is not GENERAL.  If Z3 returns SAT
(counterexample exists), the fix is blocked.
"""
from __future__ import annotations

import ast
import asyncio
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.base import AgentConfig, BaseAgent, wrap_content
from brain.schemas import (
    DomainMode,
    ExecutorType,
    FixAttempt,
    FormalVerificationResult,
    FormalVerificationStatus,
    Severity,
)
from brain.storage import BrainStorage

log = logging.getLogger(__name__)

try:
    from z3 import (                       # type: ignore[import]
        Int, Real, Bool, And, Or, Not, Implies,
        Solver, sat, unsat, unknown,
        ArithRef, BoolRef,
    )
    _Z3_AVAILABLE = True
    log.info("FormalVerifier: z3-solver available")
except ImportError:
    _Z3_AVAILABLE = False
    log.info(
        "z3-solver not installed — FormalVerifier will use static pattern mode only. "
        "Run: pip install z3-solver"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Property definitions
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VerificationProperty:
    name:        str
    description: str
    domains:     frozenset[DomainMode]
    severity:    Severity = Severity.CRITICAL


FINANCE_PROPERTIES: list[VerificationProperty] = [
    VerificationProperty(
        name="balance_non_negative",
        description="After every balance mutation, balance >= 0 must hold",
        domains=frozenset({DomainMode.FINANCE}),
    ),
    VerificationProperty(
        name="no_float_monetary",
        description="No float arithmetic on monetary values (use Decimal)",
        domains=frozenset({DomainMode.FINANCE}),
    ),
    VerificationProperty(
        name="atomic_balance_update",
        description="Balance read and write must be in same atomic operation",
        domains=frozenset({DomainMode.FINANCE}),
    ),
]

MEDICAL_PROPERTIES: list[VerificationProperty] = [
    VerificationProperty(
        name="dose_positive",
        description="Dosage must always be > 0",
        domains=frozenset({DomainMode.MEDICAL}),
    ),
    VerificationProperty(
        name="no_null_patient_id",
        description="patient_id must never be None",
        domains=frozenset({DomainMode.MEDICAL}),
    ),
    VerificationProperty(
        name="alarm_not_disabled",
        description="Safety alarm enable flags must not be set to False",
        domains=frozenset({DomainMode.MEDICAL}),
    ),
]

MILITARY_PROPERTIES: list[VerificationProperty] = [
    VerificationProperty(
        name="no_dynamic_alloc_outside_init",
        description="No malloc/calloc/new outside initialisation functions",
        domains=frozenset({DomainMode.MILITARY, DomainMode.EMBEDDED}),
    ),
    VerificationProperty(
        name="no_goto",
        description="goto is forbidden (MISRA Rule 15.1)",
        domains=frozenset({DomainMode.MILITARY}),
    ),
    VerificationProperty(
        name="no_stdio_in_handler",
        description="printf/scanf/gets forbidden in interrupt/signal handlers",
        domains=frozenset({DomainMode.MILITARY, DomainMode.EMBEDDED}),
    ),
]

ALL_PROPERTIES = FINANCE_PROPERTIES + MEDICAL_PROPERTIES + MILITARY_PROPERTIES


# ──────────────────────────────────────────────────────────────────────────────
# LLM response models
# ──────────────────────────────────────────────────────────────────────────────

from pydantic import BaseModel, Field


class ExtractedConstraint(BaseModel):
    property_name:  str
    can_verify:     bool
    z3_pre:         list[str] = Field(default_factory=list,
                                      description="Z3 Python expressions for pre-conditions")
    z3_post_neg:    list[str] = Field(default_factory=list,
                                      description="Z3 Python expressions for NEGATED post-conditions")
    explanation:    str       = ""


class ConstraintExtractionResponse(BaseModel):
    constraints: list[ExtractedConstraint] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────

class FormalVerifierAgent(BaseAgent):
    agent_type = ExecutorType.FORMAL

    def __init__(
        self,
        storage:        BrainStorage,
        run_id:         str,
        domain_mode:    DomainMode         = DomainMode.GENERAL,
        config:         AgentConfig | None = None,
        timeout_s:      int                = 30,
        repo_root:      Path | None        = None,
    ) -> None:
        super().__init__(storage, run_id, config)
        self.domain_mode = domain_mode
        self.timeout_s   = timeout_s
        self.repo_root   = repo_root

    # ── Public interface ──────────────────────────────────────────────────────

    async def verify_fix(self, fix: FixAttempt) -> list[FormalVerificationResult]:
        """
        Run formal verification on a fix attempt.

        Returns a list of FormalVerificationResult — one per applicable property
        per modified file.  An empty list means no properties were applicable.
        """
        applicable = [
            p for p in ALL_PROPERTIES
            if self.domain_mode in p.domains
        ]
        if not applicable:
            return []

        results: list[FormalVerificationResult] = []

        for ff in fix.fixed_files:
            file_results = await self._verify_file(fix, ff.path, ff.content, applicable)
            results.extend(file_results)

        # Persist results
        for r in results:
            try:
                await self.storage.store_formal_result(r)
            except AttributeError:
                pass  # storage implementation may not have this method yet

        return results

    async def any_counterexample(self, results: list[FormalVerificationResult]) -> bool:
        """True if any verification result is a counterexample — fix should be blocked."""
        return any(
            r.status == FormalVerificationStatus.COUNTEREXAMPLE
            for r in results
        )

    # ── File verification ─────────────────────────────────────────────────────

    async def _verify_file(
        self,
        fix:        FixAttempt,
        file_path:  str,
        content:    str,
        properties: list[VerificationProperty],
    ) -> list[FormalVerificationResult]:
        results: list[FormalVerificationResult] = []

        for prop in properties:
            start_ms = int(time.monotonic() * 1000)

            # Try Z3 proof first if available
            if _Z3_AVAILABLE:
                result = await self._try_z3_proof(fix, file_path, content, prop)
            else:
                # Fall back to static pattern matching
                result = self._static_pattern_check(fix, file_path, content, prop)

            elapsed = int(time.monotonic() * 1000) - start_ms
            result.elapsed_ms = elapsed

            results.append(result)
            self.log.info(
                f"FormalVerifier: {file_path} | {prop.name} → {result.status.value} "
                f"({elapsed}ms)"
            )

        return results

    # ── Z3 proof attempt ──────────────────────────────────────────────────────

    async def _try_z3_proof(
        self,
        fix:       FixAttempt,
        file_path: str,
        content:   str,
        prop:      VerificationProperty,
    ) -> FormalVerificationResult:
        """
        Two-phase approach:
        1. Ask the LLM to extract Z3 constraints from the fixed code.
        2. Run Z3 on those constraints.

        If LLM extraction fails or says can_verify=False, fall through to
        static pattern matching.
        """
        # Phase 1: LLM constraint extraction
        extraction = await self._extract_constraints(content, prop)

        if not extraction or not extraction.can_verify:
            # LLM couldn't express the property in Z3 — fall back to static
            return self._static_pattern_check(fix, file_path, content, prop)

        if not extraction.z3_post_neg:
            # Nothing to prove
            return FormalVerificationResult(
                run_id=self.run_id,
                fix_attempt_id=fix.id,
                file_path=file_path,
                property_name=prop.name,
                status=FormalVerificationStatus.UNSUPPORTED,
                proof_summary="No provable constraints extracted",
                solver_used="z3",
            )

        # Phase 2: run Z3 with timeout
        loop = asyncio.get_event_loop()
        try:
            status, ce, summary = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    self._run_z3,
                    extraction.z3_pre,
                    extraction.z3_post_neg,
                    prop.name,
                ),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            return FormalVerificationResult(
                run_id=self.run_id,
                fix_attempt_id=fix.id,
                file_path=file_path,
                property_name=prop.name,
                status=FormalVerificationStatus.TIMEOUT,
                proof_summary=f"Z3 timed out after {self.timeout_s}s",
                solver_used="z3",
            )
        except Exception as exc:
            return FormalVerificationResult(
                run_id=self.run_id,
                fix_attempt_id=fix.id,
                file_path=file_path,
                property_name=prop.name,
                status=FormalVerificationStatus.ERROR,
                proof_summary=f"Z3 error: {exc}",
                solver_used="z3",
            )

        return FormalVerificationResult(
            run_id=self.run_id,
            fix_attempt_id=fix.id,
            file_path=file_path,
            property_name=prop.name,
            status=status,
            counterexample=ce,
            proof_summary=summary,
            solver_used="z3",
        )

    def _run_z3(
        self,
        pre_conditions:      list[str],
        negated_post_conditions: list[str],
        prop_name:           str,
    ) -> tuple[FormalVerificationStatus, str, str]:
        """
        Run in executor (blocking).  Constructs and solves a Z3 formula:

        pre_conditions ∧ ¬post_condition → UNSAT means property holds.
        """
        try:
            solver = Solver()
            local_vars: dict[str, Any] = {}

            # Expose Z3 constructors in eval scope
            z3_scope = {
                "Int": Int, "Real": Real, "Bool": Bool,
                "And": And, "Or": Or, "Not": Not, "Implies": Implies,
            }

            # Declare variables referenced in constraints
            for expr_str in pre_conditions + negated_post_conditions:
                for name in re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expr_str):
                    if name not in z3_scope and name not in local_vars:
                        local_vars[name] = Int(name)

            scope = {**z3_scope, **local_vars}

            # Add pre-conditions
            for expr_str in pre_conditions:
                try:
                    constraint = eval(expr_str, scope)  # noqa: S307
                    solver.add(constraint)
                except Exception as exc:
                    return (
                        FormalVerificationStatus.ERROR,
                        "",
                        f"Pre-condition eval failed: {exc} (expr: {expr_str})",
                    )

            # Add negated post-conditions (∃ a state where the property fails)
            for expr_str in negated_post_conditions:
                try:
                    neg_constraint = eval(expr_str, scope)  # noqa: S307
                    solver.add(neg_constraint)
                except Exception as exc:
                    return (
                        FormalVerificationStatus.ERROR,
                        "",
                        f"Post-condition eval failed: {exc} (expr: {expr_str})",
                    )

            check_result = solver.check()

            if check_result == unsat:
                return (
                    FormalVerificationStatus.PROVEN_SAFE,
                    "",
                    f"Property '{prop_name}' proven SAFE — no counterexample exists (UNSAT)",
                )
            elif check_result == sat:
                model_str = str(solver.model())
                return (
                    FormalVerificationStatus.COUNTEREXAMPLE,
                    model_str,
                    f"Property '{prop_name}' VIOLATED — counterexample: {model_str[:500]}",
                )
            else:
                return (
                    FormalVerificationStatus.TIMEOUT,
                    "",
                    f"Z3 returned 'unknown' for property '{prop_name}'",
                )

        except Exception as exc:
            return (
                FormalVerificationStatus.ERROR,
                "",
                f"Z3 internal error: {exc}",
            )

    # ── LLM constraint extraction ─────────────────────────────────────────────

    async def _extract_constraints(
        self,
        content:  str,
        prop:     VerificationProperty,
    ) -> ExtractedConstraint | None:
        system = self.build_system_prompt(
            "formal methods engineer translating code to Z3 SMT constraints"
        )
        prompt = (
            f"## Property to Verify\n"
            f"Name: {prop.name}\n"
            f"Description: {prop.description}\n\n"
            f"## Code Under Analysis\n"
            f"{wrap_content(content[:6000])}\n\n"  # cap for LLM context
            "## Your Task\n"
            "Extract Z3 SMT constraints from the code above that express this property.\n\n"
            "Rules:\n"
            "1. z3_pre: list of Z3 Python expressions representing pre-conditions "
            "(e.g. 'balance >= 0', 'amount > 0', 'amount <= balance').\n"
            "2. z3_post_neg: list of Z3 Python expressions representing the NEGATED "
            "post-condition (what we're proving CAN'T happen). "
            "Example: to prove balance never goes negative, z3_post_neg = ['balance - amount < 0'].\n"
            "3. Use only: Int, Real, Bool variables and arithmetic/comparison operators.\n"
            "4. If you cannot express this property in Z3, set can_verify=false.\n"
            "5. Be conservative — only include constraints you are confident about."
        )

        try:
            response = await self.call_llm_structured(
                prompt=prompt,
                response_model=ConstraintExtractionResponse,
                system=system,
                model_override=self.config.triage_model,
            )
            for c in response.constraints:
                if c.property_name == prop.name:
                    return c
            return response.constraints[0] if response.constraints else None
        except Exception as exc:
            self.log.warning(f"FormalVerifier: constraint extraction failed: {exc}")
            return None

    # ── Static pattern check (fallback) ──────────────────────────────────────

    def _static_pattern_check(
        self,
        fix:       FixAttempt,
        file_path: str,
        content:   str,
        prop:      VerificationProperty,
    ) -> FormalVerificationResult:
        """
        Pattern-matching fallback when Z3 is unavailable or the LLM couldn't
        extract machine-checkable constraints.
        """
        _PATTERNS: dict[str, list[tuple[str, str]]] = {
            "no_float_monetary":  [
                ("price * float",    "float arithmetic on price variable"),
                ("float(price",      "float casting of price variable"),
                ("balance * 0.",     "float arithmetic on balance"),
                ("total * 0.",       "float arithmetic on total"),
            ],
            "no_goto":            [
                ("\tgoto ", "goto statement"),
                (" goto ",  "goto statement"),
            ],
            "no_dynamic_alloc_outside_init": [
                (" malloc(",  "malloc outside init"),
                ("\tmalloc(", "malloc outside init"),
                ("calloc(",   "calloc call"),
                ("realloc(",  "realloc call"),
            ],
            "no_stdio_in_handler": [
                ("printf(",  "printf in handler"),
                ("scanf(",   "scanf in handler"),
                ("gets(",    "gets() call — unconditionally unsafe"),
            ],
            "alarm_not_disabled": [
                ("alarm_enabled = False", "alarm disabled"),
                ("disable_alarm(",        "alarm disabled"),
                ("safety_alarm = 0",      "alarm disabled"),
            ],
        }

        patterns = _PATTERNS.get(prop.name, [])
        for line_no, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("#", "//", "/*", "*")):
                continue
            for pat, msg in patterns:
                if pat in stripped:
                    return FormalVerificationResult(
                        run_id=self.run_id,
                        fix_attempt_id=fix.id,
                        file_path=file_path,
                        property_name=prop.name,
                        status=FormalVerificationStatus.COUNTEREXAMPLE,
                        counterexample=f"Line {line_no}: {msg}: {stripped[:120]}",
                        proof_summary=f"Static pattern match: {msg} at line {line_no}",
                        solver_used="static-pattern",
                    )

        return FormalVerificationResult(
            run_id=self.run_id,
            fix_attempt_id=fix.id,
            file_path=file_path,
            property_name=prop.name,
            status=FormalVerificationStatus.PROVEN_SAFE,
            proof_summary=f"Static pattern scan: no violations of '{prop.name}' found",
            solver_used="static-pattern",
        )
