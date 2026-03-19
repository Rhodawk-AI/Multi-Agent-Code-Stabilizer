"""
swarm/crew_roles.py
===================
CrewAI role-based crews for Rhodawk AI.

Crews
──────
SecurityCrew         — OWASP/CWE/CERT deep-dive + CVE cross-reference
ArchitectureCrew     — Dependency analysis + design pattern review
DomainSpecialistCrew — Finance/Medical/Military domain expert panel
SWEBenchCrew         — SWE-bench task decomposition and fix generation

Each crew has agents with distinct roles, goals, and backstories wired
to the tiered model router.
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

try:
    from crewai import Agent, Crew, Task, Process  # type: ignore[import]
    _CREWAI_AVAILABLE = True
except ImportError:
    _CREWAI_AVAILABLE = False
    log.warning(
        "crewai not installed — CrewAI integration disabled. "
        "Run: pip install crewai"
    )

from models.router import get_router, ModelTier


# ──────────────────────────────────────────────────────────────────────────────
# LLM adapter
# ──────────────────────────────────────────────────────────────────────────────

def _crewai_llm(task_type: str = "audit") -> Any:
    """Return a CrewAI-compatible LLM object for the given task type."""
    if not _CREWAI_AVAILABLE:
        return None
    router = get_router()
    model  = router.primary_model(task_type)
    try:
        from crewai import LLM  # type: ignore[import]
        return LLM(model=model)
    except ImportError:
        return model


# ──────────────────────────────────────────────────────────────────────────────
# Security Crew
# ──────────────────────────────────────────────────────────────────────────────

def build_security_crew(code_context: str, repo_path: str) -> Any | None:
    """
    Three-agent security review crew:
    1. Threat modeler — enumerates attack surfaces
    2. Vulnerability analyst — maps findings to CWE/CVE
    3. Remediation specialist — proposes concrete fixes
    """
    if not _CREWAI_AVAILABLE:
        return None

    threat_modeler = Agent(
        role="Senior Threat Modeler",
        goal="Enumerate all attack surfaces and data flows that could be exploited",
        backstory=(
            "You have 15 years of red-team experience at a Fortune 500 financial institution. "
            "You think like an attacker and map every untrusted input, network boundary, "
            "and privilege boundary."
        ),
        llm=_crewai_llm("audit"),
        verbose=False,
        allow_delegation=False,
    )

    vuln_analyst = Agent(
        role="Vulnerability Analyst",
        goal="Map threat model findings to specific CWE numbers and public CVEs",
        backstory=(
            "You maintain the NVD database and can correlate code patterns to "
            "OWASP Top 10, CWE Top 25, and live CVE feeds instantly."
        ),
        llm=_crewai_llm("audit"),
        verbose=False,
        allow_delegation=False,
    )

    remediation = Agent(
        role="Remediation Engineer",
        goal="Produce minimal, correct patches for every confirmed vulnerability",
        backstory=(
            "You specialise in security-preserving refactors: you never break functionality "
            "but you always close the vulnerability completely."
        ),
        llm=_crewai_llm("critical_fix"),
        verbose=False,
        allow_delegation=False,
    )

    task_model = Task(
        description=(
            f"Analyze this code repository for security vulnerabilities.\n"
            f"Repo path: {repo_path}\n\n"
            f"Code context:\n{code_context[:8000]}\n\n"
            "Produce a ranked list of attack surfaces with severity (CRITICAL/MAJOR/MINOR)."
        ),
        agent=threat_modeler,
        expected_output="JSON list of attack surfaces with severity and description",
    )

    task_analyze = Task(
        description=(
            "For each attack surface identified, map to CWE ID and check for public CVEs. "
            "Produce a structured vulnerability report."
        ),
        agent=vuln_analyst,
        expected_output="JSON vulnerability report with CWE IDs and CVE references",
        context=[task_model],
    )

    task_fix = Task(
        description=(
            "For each CRITICAL and MAJOR vulnerability, produce a minimal patch. "
            "Return complete fixed file content for each modified file."
        ),
        agent=remediation,
        expected_output="JSON list of fixed files with path and complete content",
        context=[task_analyze],
    )

    return Crew(
        agents=[threat_modeler, vuln_analyst, remediation],
        tasks=[task_model, task_analyze, task_fix],
        process=Process.sequential,
        verbose=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SWE-Bench Crew (optimised for benchmark tasks)
# ──────────────────────────────────────────────────────────────────────────────

def build_swe_bench_crew(issue_text: str, repo_context: str) -> Any | None:
    """
    Crew optimised for SWE-bench style issue resolution.

    Structure:
    1. Issue analyst — parses the issue, identifies root cause
    2. Test author   — writes a failing test that captures the bug
    3. Fix engineer  — implements the minimal fix that passes the test
    4. Verifier      — confirms fix doesn't regress other tests
    """
    if not _CREWAI_AVAILABLE:
        return None

    analyst = Agent(
        role="Issue Analyst",
        goal="Precisely identify the root cause of the reported issue",
        backstory=(
            "You are an expert at reading GitHub issues, understanding stack traces, "
            "and pinpointing the exact line and condition that causes the bug."
        ),
        llm=_crewai_llm("audit"),
        verbose=False,
    )

    test_author = Agent(
        role="Test Author",
        goal="Write a minimal failing test that captures exactly the reported bug",
        backstory=(
            "You write test-first. Your tests are the specification. "
            "You never write tests that pass without the fix."
        ),
        llm=_crewai_llm("simple_codegen"),
        verbose=False,
    )

    fix_engineer = Agent(
        role="Fix Engineer",
        goal="Implement the minimal correct fix that makes the failing test pass",
        backstory=(
            "You are surgical. You touch the minimum amount of code. "
            "Every line you change has a reason. You never introduce regressions."
        ),
        llm=_crewai_llm("critical_fix"),
        verbose=False,
    )

    verifier = Agent(
        role="Code Verifier",
        goal="Confirm the fix is complete, correct, and doesn't regress other behavior",
        backstory=(
            "You are the last line of defense before code ships. "
            "You read the diff adversarially looking for anything that could break."
        ),
        llm=_crewai_llm("review"),
        verbose=False,
    )

    t1 = Task(
        description=(
            f"Analyze this GitHub issue and identify the root cause.\n"
            f"Issue:\n{issue_text}\n\n"
            f"Relevant code:\n{repo_context[:6000]}"
        ),
        agent=analyst,
        expected_output="Root cause analysis with exact file, function, and line range",
    )

    t2 = Task(
        description="Write a minimal pytest test that will FAIL before the fix and PASS after.",
        agent=test_author,
        expected_output="Complete Python test file content",
        context=[t1],
    )

    t3 = Task(
        description="Implement the minimal fix. Return complete file content for each changed file.",
        agent=fix_engineer,
        expected_output="JSON: {fixed_files: [{path, content, changes_summary}]}",
        context=[t1, t2],
    )

    t4 = Task(
        description=(
            "Review the fix diff. Confirm it fixes the issue without introducing regressions. "
            "Return {approved: bool, reason: str, confidence: float}."
        ),
        agent=verifier,
        expected_output="JSON approval decision",
        context=[t3],
    )

    return Crew(
        agents=[analyst, test_author, fix_engineer, verifier],
        tasks=[t1, t2, t3, t4],
        process=Process.sequential,
        verbose=False,
    )
