"""
compliance/sas_generator.py — DO-178C Software Accomplishment Summary generator.
Implements Gap 9 from audit report.

IMPORTANT DISCLAIMER: Rhodawk AI is an advisory tool. All outputs
(fixes, SAS documents, RTM entries) require independent human verification
before they can be cited as DO-178C compliance evidence. This tool does NOT
hold a Tool Qualification Document (TQD) and is NOT qualified under
DO-330 at any TQL level. Until formal tool qualification is obtained,
all generated artifacts must be treated as advisory drafts, not certified
evidence.
"""
from __future__ import annotations
import json, logging
from datetime import datetime, timezone
from brain.schemas import (
    SoftwareAccomplishmentSummary, SoftwareLevel, ToolQualificationLevel,
    ComplianceStandard, Severity,
)
from brain.storage import BrainStorage
log = logging.getLogger(__name__)

class SASGenerator:
    TOOL_INVENTORY = [
        {"name":"clang-tidy","version":"17+","tql":"TQL-5","qualified_for":"MISRA-C static analysis"},
        {"name":"cppcheck","version":"2.x","tql":"TQL-5","qualified_for":"C/C++ static analysis"},
        {"name":"cbmc","version":"5.x","tql":"TQL-4","qualified_for":"Bounded model checking"},
        {"name":"z3","version":"4.x","tql":"TQL-4","qualified_for":"SMT property proving"},
        {"name":"semgrep","version":"1.x","tql":"TQL-5","qualified_for":"Pattern analysis"},
        {"name":"bandit","version":"1.x","tql":"TQL-5","qualified_for":"Python security"},
        {"name":"Rhodawk AI FixerAgent","version":"1.0","tql":"TQL-5",
         "qualified_for":"Fix candidate generation (output independently verified)"},
    ]

    def __init__(self, storage: BrainStorage) -> None:
        self.storage = storage

    async def generate(
        self,
        run_id: str,
        software_level: SoftwareLevel = SoftwareLevel.NONE,
        tql: ToolQualificationLevel = ToolQualificationLevel.NONE,
        baseline_id: str = "",
        prepared_by: str = "Rhodawk AI",
    ) -> SoftwareAccomplishmentSummary:
        issues = await self.storage.list_issues(run_id=run_id)
        fixes  = await self.storage.list_fixes(run_id=run_id)
        trail  = await self.storage.list_audit_trail(run_id=run_id)
        escs   = await self.storage.list_escalations(run_id=run_id)
        ldra   = await self.storage.list_ldra_findings(run_id=run_id)
        poly   = await self.storage.list_polyspace_findings(run_id=run_id)

        misra_open   = sum(1 for i in issues if i.misra_rule and i.status.value not in ("CLOSED","DEFERRED"))
        misra_closed = sum(1 for i in issues if i.misra_rule and i.status.value == "CLOSED")
        cert_open    = sum(1 for i in issues if i.cert_rule and i.status.value not in ("CLOSED","DEFERRED"))
        cert_closed  = sum(1 for i in issues if i.cert_rule and i.status.value == "CLOSED")
        cwe_open     = sum(1 for i in issues if i.cwe_id and i.status.value not in ("CLOSED","DEFERRED"))
        cwe_closed   = sum(1 for i in issues if i.cwe_id and i.status.value == "CLOSED")

        do178c_met:  list[str] = []
        do178c_open: list[str] = []

        has_misra = misra_closed > 0 or misra_open > 0
        has_traceability = any(
            getattr(i, "requirement_id", "") for i in issues
        )
        has_formal = any(
            e.event_type in ("formal_verification", "cbmc_verification")
            for e in trail
        )
        has_test_coverage = any(
            e.event_type == "test_run" for e in trail
        )
        has_reviews = any(
            e.event_type in ("review", "consensus_vote") for e in trail
        )
        critical_open = sum(
            1 for i in issues
            if i.severity == Severity.CRITICAL
            and i.status.value not in ("CLOSED", "DEFERRED")
        )

        if has_misra and misra_open == 0:
            do178c_met.append(
                "Table A-7 Obj 5 — Source code standards verification (all MISRA findings closed)"
            )
        elif has_misra:
            do178c_open.append(
                f"Table A-7 Obj 5 — Source code standards verification ({misra_open} MISRA findings open)"
            )
        else:
            do178c_open.append(
                "Table A-7 Obj 5 — Source code standards verification (not assessed — no MISRA analysis run)"
            )

        if has_traceability:
            traced = sum(1 for i in issues if getattr(i, "requirement_id", ""))
            if traced >= len(issues) * 0.8:
                do178c_met.append("Table A-5 — Traceability (requirements traced)")
            else:
                do178c_open.append(
                    f"Table A-5 — Traceability ({traced}/{len(issues)} issues traced — expand coverage)"
                )
        else:
            do178c_open.append("Table A-5 — Traceability (no requirement_id links found)")

        do178c_open.append(
            "Table A-4 — High-level requirements accuracy (requires human DER review)"
        )

        if has_formal:
            do178c_met.append("Table A-7 Obj 7 — Formal methods applied")
        else:
            do178c_open.append(
                "Table A-7 Obj 7 — Formal methods (CBMC/Z3 not applied in this run)"
            )

        if has_test_coverage:
            do178c_met.append("Table A-6 Obj 3 — Test execution completed")
        else:
            do178c_open.append("Table A-6 Obj 3 — Test execution (no test runs recorded)")

        do178c_open.append(
            "Table A-7 Obj 6 — Absence of unintended functions (requires structural coverage analysis)"
        )

        sas = SoftwareAccomplishmentSummary(
            baseline_id=baseline_id,
            run_id=run_id,
            software_level=software_level,
            tool_qualification_level=tql,
            total_cycles=0,
            total_issues_found=len(issues),
            total_issues_closed=sum(1 for i in issues if i.status.value == "CLOSED"),
            total_escalations=len(escs),
            total_deviations=sum(1 for i in issues if i.deviation_record),
            tools_used=self.TOOL_INVENTORY,
            misra_violations_open=misra_open,
            misra_violations_closed=misra_closed,
            cert_violations_open=cert_open,
            cert_violations_closed=cert_closed,
            cwe_findings_open=cwe_open,
            cwe_findings_closed=cwe_closed,
            do178c_objectives_met=do178c_met,
            do178c_objectives_open=do178c_open,
            prepared_by=prepared_by,
            certification_basis=(
                "DO-178C (RTCA/EUROCAE, December 2011), "
                "MISRA-C:2023, MIL-STD-882E, CERT-C 2016"
            ),
            advisory_disclaimer=(
                "ADVISORY ONLY: This SAS document was generated by Rhodawk AI, "
                "an unqualified tool (no TQD per DO-330). All objectives, findings, "
                "and compliance assessments require independent human verification "
                "by a qualified DER before they may be cited as certification evidence."
            ),
        )
        await self.storage.upsert_sas(sas)
        return sas

    async def export_json(self, run_id: str) -> str:
        sas = await self.storage.get_sas(run_id)
        if not isinstance(sas, SoftwareAccomplishmentSummary):
            sas = await self.generate(run_id)
        return sas.model_dump_json(indent=2)
