"""tests/unit/test_compliance.py — RTMExporter and SASGenerator unit tests."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from compliance.rtm_exporter import RTMExporter
from compliance.sas_generator import SASGenerator
from brain.schemas import (
    RequirementTraceability,
    SoftwareAccomplishmentSummary,
    SoftwareLevel,
    ToolQualificationLevel,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_rtm_entry(**kwargs) -> RequirementTraceability:
    defaults = dict(
        id="rtm-1",
        run_id="run1",
        requirement_id="REQ-001",
        requirement_text="System shall not crash on null input",
        test_case_id="test_no_crash",
        test_result_id="result-1",
        coverage_pct=90.0,
        mcdc_coverage=85.0,
        issue_id="issue-1",
        fix_attempt_id="fix-1",
        verified_by="Rhodawk AI",
        verified_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        do178c_objective="A-5",
    )
    defaults.update(kwargs)
    return RequirementTraceability(**defaults)


def _make_storage_for_rtm(entries=None) -> AsyncMock:
    s = AsyncMock()
    s.list_rtm_entries = AsyncMock(return_value=entries or [])
    s.list_issues = AsyncMock(return_value=[])
    s.list_fixes = AsyncMock(return_value=[])
    s.list_audit_trail = AsyncMock(return_value=[])
    s.list_escalations = AsyncMock(return_value=[])
    s.list_ldra_findings = AsyncMock(return_value=[])
    return s


# ── RTMExporter ───────────────────────────────────────────────────────────────

class TestRTMExporterCSV:
    @pytest.mark.asyncio
    async def test_empty_entries_returns_header_only(self):
        storage = _make_storage_for_rtm([])
        exporter = RTMExporter(storage=storage)
        csv_text = await exporter.export_csv("run1")
        lines = csv_text.strip().splitlines()
        assert len(lines) == 1  # header only
        assert "requirement_id" in lines[0]

    @pytest.mark.asyncio
    async def test_csv_contains_entry_data(self):
        entry = _make_rtm_entry()
        storage = _make_storage_for_rtm([entry])
        exporter = RTMExporter(storage=storage)
        csv_text = await exporter.export_csv("run1")
        assert "REQ-001" in csv_text
        assert "test_no_crash" in csv_text

    @pytest.mark.asyncio
    async def test_csv_truncates_long_requirement_text(self):
        long_text = "A" * 500
        entry = _make_rtm_entry(requirement_text=long_text)
        storage = _make_storage_for_rtm([entry])
        exporter = RTMExporter(storage=storage)
        csv_text = await exporter.export_csv("run1")
        # Text is truncated to 200 chars in the exporter
        lines = csv_text.strip().splitlines()
        data_line = lines[1]
        assert "A" * 201 not in data_line

    @pytest.mark.asyncio
    async def test_csv_handles_none_verified_at(self):
        entry = _make_rtm_entry(verified_at=None)
        storage = _make_storage_for_rtm([entry])
        exporter = RTMExporter(storage=storage)
        csv_text = await exporter.export_csv("run1")
        assert csv_text  # no crash

    @pytest.mark.asyncio
    async def test_csv_multiple_entries(self):
        entries = [
            _make_rtm_entry(id=f"rtm-{i}", requirement_id=f"REQ-00{i}")
            for i in range(5)
        ]
        storage = _make_storage_for_rtm(entries)
        exporter = RTMExporter(storage=storage)
        csv_text = await exporter.export_csv("run1")
        lines = csv_text.strip().splitlines()
        assert len(lines) == 6  # header + 5 rows


class TestRTMExporterJSON:
    @pytest.mark.asyncio
    async def test_json_empty(self):
        storage = _make_storage_for_rtm([])
        exporter = RTMExporter(storage=storage)
        result = await exporter.export_json("run1")
        data = json.loads(result)
        assert data == []

    @pytest.mark.asyncio
    async def test_json_contains_entry(self):
        entry = _make_rtm_entry()
        storage = _make_storage_for_rtm([entry])
        exporter = RTMExporter(storage=storage)
        result = await exporter.export_json("run1")
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["requirement_id"] == "REQ-001"

    @pytest.mark.asyncio
    async def test_json_is_valid(self):
        entry = _make_rtm_entry()
        storage = _make_storage_for_rtm([entry])
        exporter = RTMExporter(storage=storage)
        result = await exporter.export_json("run1")
        # Should not raise
        json.loads(result)


# ── SASGenerator ─────────────────────────────────────────────────────────────

class TestSASGenerator:
    @pytest.mark.asyncio
    async def test_generate_returns_sas_object(self):
        storage = _make_storage_for_rtm()
        gen = SASGenerator(storage=storage)
        sas = await gen.generate(run_id="run1")
        assert isinstance(sas, SoftwareAccomplishmentSummary)

    @pytest.mark.asyncio
    async def test_generate_sets_run_id(self):
        storage = _make_storage_for_rtm()
        gen = SASGenerator(storage=storage)
        sas = await gen.generate(run_id="run-xyz")
        assert sas.run_id == "run-xyz"

    @pytest.mark.asyncio
    async def test_generate_with_software_level(self):
        storage = _make_storage_for_rtm()
        gen = SASGenerator(storage=storage)
        sas = await gen.generate(run_id="run1", software_level=SoftwareLevel.DAL_A)
        assert sas.software_level == SoftwareLevel.DAL_A

    @pytest.mark.asyncio
    async def test_generate_prepared_by(self):
        storage = _make_storage_for_rtm()
        gen = SASGenerator(storage=storage)
        sas = await gen.generate(run_id="run1", prepared_by="QA Team")
        assert sas.prepared_by == "QA Team"

    @pytest.mark.asyncio
    async def test_tool_inventory_non_empty(self):
        assert len(SASGenerator.TOOL_INVENTORY) > 0
        assert all("name" in t for t in SASGenerator.TOOL_INVENTORY)

    @pytest.mark.asyncio
    async def test_export_json_valid(self):
        storage = _make_storage_for_rtm()
        gen = SASGenerator(storage=storage)
        result = await gen.export_json(run_id="run1")
        data = json.loads(result)
        assert isinstance(data, list) or isinstance(data, dict)
