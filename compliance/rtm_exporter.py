"""
compliance/rtm_exporter.py — DO-178C Requirements Traceability Matrix exporter.
Implements Gap 3 from audit report.
"""
from __future__ import annotations
import csv, io, json, logging
from datetime import datetime, timezone
from brain.schemas import RequirementTraceability, AuditRun
from brain.storage import BrainStorage
log = logging.getLogger(__name__)

class RTMExporter:
    def __init__(self, storage: BrainStorage) -> None:
        self.storage = storage

    async def export_csv(self, run_id: str) -> str:
        entries = await self.storage.list_rtm_entries(run_id=run_id)
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=[
            "id","requirement_id","requirement_text","test_case_id",
            "test_result_id","coverage_pct","mcdc_coverage","issue_id",
            "fix_attempt_id","verified_by","verified_at","do178c_objective",
        ])
        w.writeheader()
        for e in entries:
            w.writerow({
                "id": e.id,
                "requirement_id": e.requirement_id,
                "requirement_text": e.requirement_text[:200],
                "test_case_id": e.test_case_id,
                "test_result_id": e.test_result_id,
                "coverage_pct": f"{e.coverage_pct:.1f}",
                "mcdc_coverage": f"{e.mcdc_coverage:.1f}",
                "issue_id": e.issue_id,
                "fix_attempt_id": e.fix_attempt_id,
                "verified_by": e.verified_by,
                "verified_at": e.verified_at.isoformat() if e.verified_at else "",
                "do178c_objective": e.do178c_objective,
            })
        return buf.getvalue()

    async def export_json(self, run_id: str) -> str:
        entries = await self.storage.list_rtm_entries(run_id=run_id)
        return json.dumps([e.model_dump() for e in entries], indent=2, default=str)
