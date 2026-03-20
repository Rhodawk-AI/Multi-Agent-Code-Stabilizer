"""
api/routes/compound_findings.py
================================
REST endpoints for Gap 2 compound findings and synthesis reports.

GAP 2 FIX — Missing API surface
─────────────────────────────────
Previously compound findings were materialised as Issues with
executor_type=SYNTHESIS and stored via the normal issues table.  This meant
their compound-specific metadata (amplification_factor, contributing_issue_ids,
category, rationale, domains_involved) was only accessible by parsing the
description string — inaccessible to external consumers.

This module exposes compound findings as first-class API objects alongside
their SynthesisReport provenance so dashboards, webhooks, and the DeerFlow
orchestrator can query them structurally without string-parsing.

Endpoints
─────────
  GET /api/compound-findings/
      List all compound findings, optionally filtered by run_id / severity.
      Returns structured CompoundFindingOut objects with all metadata fields.

  GET /api/compound-findings/{issue_id}
      Retrieve a single compound finding by its Issue ID.

  GET /api/compound-findings/summary/{run_id}
      Aggregated statistics for a run: finding counts by severity, category,
      domain pairing, and average amplification factor.

  GET /api/synthesis-reports/
      List SynthesisReports persisted by the controller after each run cycle.
      Filtered by run_id; ordered by cycle ASC.

  GET /api/synthesis-reports/{run_id}
      Most recent SynthesisReport for a run (latest cycle).

  GET /api/synthesis-reports/{run_id}/{cycle}
      SynthesisReport for a specific cycle of a run.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from brain.schemas import ExecutorType, Severity
from brain.sqlite_storage import SQLiteBrainStorage

router = APIRouter(prefix="/api", tags=["compound-findings"])


# ── Storage helper ────────────────────────────────────────────────────────────

def _storage(repo_path: str) -> SQLiteBrainStorage:
    return SQLiteBrainStorage(Path(repo_path) / ".stabilizer" / "brain.db")


# ── Response models ───────────────────────────────────────────────────────────

class CompoundFindingOut(BaseModel):
    """
    Structured representation of a compound (cross-domain) finding.

    Compound findings are Issues with executor_type=SYNTHESIS.  Their
    compound-specific metadata is stored in the description field as a
    structured block; this model parses and exposes it cleanly.
    """
    id:                    str
    run_id:                str
    severity:              str
    file_path:             str
    line_start:            int
    line_end:              int
    status:                str
    confidence:            float
    cwe_id:                str
    misra_rule:            str
    # Compound-specific fields parsed from description
    compound_category:     str   = ""
    title:                 str   = ""
    domains_involved:      list[str] = []
    amplification_factor:  float = 0.0
    fix_complexity:        str   = ""
    contributing_issue_count: int = 0
    rationale:             str   = ""
    # Raw description (always included for full traceability)
    description:           str
    created_at:            str


class SynthesisReportOut(BaseModel):
    id:                      str
    run_id:                  str
    cycle:                   int
    raw_issue_count:         int
    fingerprint_dedup_count: int
    semantic_dedup_count:    int
    final_issue_count:       int
    compound_finding_count:  int
    compound_critical_count: int
    synthesis_model:         str
    dedup_enabled:           bool
    compound_enabled:        bool
    duration_s:              float
    created_at:              str


class CompoundFindingSummary(BaseModel):
    run_id:                  str
    total_compound:          int
    by_severity:             dict[str, int]
    avg_amplification_factor: float
    domains_seen:            list[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_compound_description(desc: str) -> dict[str, Any]:
    """
    Extract compound-specific fields from the structured description block
    written by SynthesisAgent._compound_to_issue().

    Description format (written by synthesis_agent.py):
      [COMPOUND FINDING — {category}] {title}

      {description_body}

      Domains involved: {d1}, {d2}
      Amplification factor: {x.x}x
      Fix complexity: {HIGH|MEDIUM|LOW|ARCHITECTURAL}
      Contributing issues: {n}

      Rationale:
      {rationale_text}
    """
    parsed: dict[str, Any] = {}

    # Category and title from the first line
    m = re.match(r"\[COMPOUND FINDING — ([^\]]+)\]\s*(.+)", desc)
    if m:
        parsed["compound_category"] = m.group(1).strip()
        parsed["title"] = m.group(2).strip()

    # Domains involved
    m = re.search(r"Domains involved:\s*(.+)", desc)
    if m:
        parsed["domains_involved"] = [
            d.strip() for d in m.group(1).split(",") if d.strip()
        ]

    # Amplification factor
    m = re.search(r"Amplification factor:\s*([\d.]+)x", desc)
    if m:
        try:
            parsed["amplification_factor"] = float(m.group(1))
        except ValueError:
            pass

    # Fix complexity
    m = re.search(r"Fix complexity:\s*(\S+)", desc)
    if m:
        parsed["fix_complexity"] = m.group(1).strip()

    # Contributing issues count
    m = re.search(r"Contributing issues:\s*(\d+)", desc)
    if m:
        try:
            parsed["contributing_issue_count"] = int(m.group(1))
        except ValueError:
            pass

    # Rationale (everything after "Rationale:\n")
    m = re.search(r"Rationale:\n(.+)", desc, re.DOTALL)
    if m:
        parsed["rationale"] = m.group(1).strip()[:1000]

    return parsed


def _issue_to_compound_out(issue) -> CompoundFindingOut:
    """Convert an Issue with executor_type=SYNTHESIS to CompoundFindingOut."""
    parsed = _parse_compound_description(issue.description or "")
    return CompoundFindingOut(
        id=issue.id,
        run_id=issue.run_id,
        severity=issue.severity.value,
        file_path=issue.file_path,
        line_start=issue.line_start,
        line_end=issue.line_end,
        status=issue.status.value,
        confidence=issue.confidence,
        cwe_id=issue.cwe_id or "",
        misra_rule=issue.misra_rule or "",
        compound_category=parsed.get("compound_category", ""),
        title=parsed.get("title", ""),
        domains_involved=parsed.get("domains_involved", []),
        amplification_factor=parsed.get("amplification_factor", 0.0),
        fix_complexity=parsed.get("fix_complexity", ""),
        contributing_issue_count=parsed.get("contributing_issue_count", 0),
        rationale=parsed.get("rationale", ""),
        description=issue.description or "",
        created_at=issue.created_at.isoformat(),
    )


def _report_to_out(report) -> SynthesisReportOut:
    return SynthesisReportOut(
        id=report.id,
        run_id=report.run_id,
        cycle=report.cycle,
        raw_issue_count=report.raw_issue_count,
        fingerprint_dedup_count=report.fingerprint_dedup_count,
        semantic_dedup_count=report.semantic_dedup_count,
        final_issue_count=report.final_issue_count,
        compound_finding_count=report.compound_finding_count,
        compound_critical_count=report.compound_critical_count,
        synthesis_model=report.synthesis_model,
        dedup_enabled=report.dedup_enabled,
        compound_enabled=report.compound_enabled,
        duration_s=report.duration_s,
        created_at=report.created_at.isoformat(),
    )


# ── Compound findings endpoints ───────────────────────────────────────────────

@router.get("/compound-findings/", response_model=list[CompoundFindingOut])
async def list_compound_findings(
    run_id:    str = Query(default="", description="Filter by run ID"),
    severity:  str = Query(default="", description="Filter by severity (CRITICAL/MAJOR/MINOR/INFO)"),
    repo_path: str = Query(default="."),
) -> list[CompoundFindingOut]:
    """
    List all compound findings (cross-domain issues produced by SynthesisAgent).

    Compound findings have executor_type=SYNTHESIS and are structurally
    distinct from single-domain auditor findings: they emerge from the
    combination of ≥2 auditor domains and are more severe than any individual
    contributing finding alone.
    """
    storage = _storage(repo_path)
    try:
        await storage.initialise()
        issues = await storage.list_compound_findings(
            run_id=run_id or None,
            severity=severity or None,
        )
        return [_issue_to_compound_out(i) for i in issues]
    finally:
        await storage.close()


@router.get("/compound-findings/summary/{run_id}", response_model=CompoundFindingSummary)
async def compound_findings_summary(
    run_id:    str,
    repo_path: str = Query(default="."),
) -> CompoundFindingSummary:
    """
    Aggregated statistics for compound findings in a run.

    Returns severity distribution, all distinct domain pairings seen,
    and average amplification factor across all compound findings.
    """
    storage = _storage(repo_path)
    try:
        await storage.initialise()
        issues = await storage.list_compound_findings(run_id=run_id)
        if not issues:
            return CompoundFindingSummary(
                run_id=run_id,
                total_compound=0,
                by_severity={},
                avg_amplification_factor=0.0,
                domains_seen=[],
            )

        by_severity: dict[str, int] = {}
        domains_seen: set[str] = set()
        amp_factors: list[float] = []

        for issue in issues:
            sev = issue.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1
            parsed = _parse_compound_description(issue.description or "")
            for d in parsed.get("domains_involved", []):
                domains_seen.add(d)
            af = parsed.get("amplification_factor", 0.0)
            if af > 0:
                amp_factors.append(af)

        return CompoundFindingSummary(
            run_id=run_id,
            total_compound=len(issues),
            by_severity=by_severity,
            avg_amplification_factor=(
                round(sum(amp_factors) / len(amp_factors), 2) if amp_factors else 0.0
            ),
            domains_seen=sorted(domains_seen),
        )
    finally:
        await storage.close()


@router.get("/compound-findings/{issue_id}", response_model=CompoundFindingOut)
async def get_compound_finding(
    issue_id:  str,
    repo_path: str = Query(default="."),
) -> CompoundFindingOut:
    """
    Retrieve a single compound finding by its Issue ID.

    Returns 404 if not found or if the issue is not a SYNTHESIS-type finding.
    """
    storage = _storage(repo_path)
    try:
        await storage.initialise()
        issue = await storage.get_issue(issue_id)
        if not issue:
            raise HTTPException(
                status_code=404,
                detail=f"Issue {issue_id} not found",
            )
        if issue.executor_type != ExecutorType.SYNTHESIS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Issue {issue_id} is type {issue.executor_type.value}, "
                    f"not a compound finding (SYNTHESIS). "
                    f"Use GET /api/issues/{issue_id} for regular findings."
                ),
            )
        return _issue_to_compound_out(issue)
    finally:
        await storage.close()


# ── Synthesis report endpoints ────────────────────────────────────────────────

@router.get("/synthesis-reports/", response_model=list[SynthesisReportOut])
async def list_synthesis_reports(
    run_id:    str = Query(default="", description="Filter by run ID"),
    repo_path: str = Query(default="."),
) -> list[SynthesisReportOut]:
    """
    List SynthesisReports persisted by the controller after each audit cycle.

    Each report records deduplication effectiveness (raw → deduped counts),
    compound finding yield, synthesis model used, and wall-clock duration.
    Ordered by cycle ASC so callers can track quality trends across cycles.
    """
    storage = _storage(repo_path)
    try:
        await storage.initialise()
        reports = await storage.list_synthesis_reports(run_id=run_id or None)
        return [_report_to_out(r) for r in reports]
    finally:
        await storage.close()


@router.get("/synthesis-reports/{run_id}", response_model=SynthesisReportOut)
async def get_latest_synthesis_report(
    run_id:    str,
    repo_path: str = Query(default="."),
) -> SynthesisReportOut:
    """Most recent SynthesisReport for a run (latest cycle)."""
    storage = _storage(repo_path)
    try:
        await storage.initialise()
        report = await storage.get_synthesis_report(run_id=run_id)
        if not report:
            raise HTTPException(
                status_code=404,
                detail=f"No SynthesisReport found for run {run_id}",
            )
        return _report_to_out(report)
    finally:
        await storage.close()


@router.get("/synthesis-reports/{run_id}/{cycle}", response_model=SynthesisReportOut)
async def get_synthesis_report_by_cycle(
    run_id:    str,
    cycle:     int,
    repo_path: str = Query(default="."),
) -> SynthesisReportOut:
    """SynthesisReport for a specific cycle of a run."""
    storage = _storage(repo_path)
    try:
        await storage.initialise()
        report = await storage.get_synthesis_report(run_id=run_id, cycle=cycle)
        if not report:
            raise HTTPException(
                status_code=404,
                detail=f"No SynthesisReport found for run {run_id} cycle {cycle}",
            )
        return _report_to_out(report)
    finally:
        await storage.close()
