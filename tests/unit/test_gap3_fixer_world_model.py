"""
tests/unit/test_gap3_fixer_world_model.py
==========================================
Regression tests for Gap 3 — "The Fixer Has No World Model".

Covers the three defects identified in the audit:

  Defect 1 — Revert memory path (previously zero coverage):
    - All probe rounds exhausting triggers _store_failure_memory()
    - fix_memory.store_failure() persists a [REVERTED]-prefixed entry
    - A subsequent retrieve() returns the [REVERTED] entry
    - format_as_few_shot() places reverted entries in the negative section

  Defect 2 — _generate_refactor_proposal() (previously zero coverage):
    - Blast radius >= threshold aborts patch and calls _generate_refactor_proposal()
    - RefactorProposal is persisted to storage
    - EscalationManager is called with BLAST_RADIUS_EXCEEDED
    - Returned FixAttempt has blast_radius_exceeded=True and no fixed_files

  Defect 3 — Time-window filter (previously unenforced at call site):
    - retrieve(max_age_days=N) drops entries older than N days
    - Fresh entries are still returned
    - max_age_days=0 means no filtering (unlimited)
    - Entries without created_at are retained for backward compat
    - _get_memory_examples explicitly passes max_age_days=180
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_storage(tmp_path: Path):
    from brain.sqlite_storage import SQLiteBrainStorage
    db = tmp_path / "brain.db"
    return SQLiteBrainStorage(db)


def _make_fixer(storage, run_id="test-run", repo_root=None, fix_memory=None,
                cpg_engine=None, escalation_manager=None, blast_radius_threshold=50):
    from agents.fixer import FixerAgent
    from agents.base import AgentConfig
    return FixerAgent(
        storage=storage,
        run_id=run_id,
        config=AgentConfig(model="test-model", run_id=run_id),
        repo_root=repo_root,
        fix_memory=fix_memory,
        cpg_engine=cpg_engine,
        escalation_manager=escalation_manager,
        blast_radius_threshold=blast_radius_threshold,
    )


def _make_issue(file_path="foo.py", severity="MEDIUM", description="null deref"):
    from brain.schemas import Issue, Severity, IssueStatus
    return Issue(
        run_id="test-run",
        file_path=file_path,
        line_start=10,
        severity=Severity(severity),
        description=description,
        status=IssueStatus.OPEN,
    )


def _json_fm(tmp_path: Path):
    """FixMemory wired to JSON fallback (no external deps required)."""
    from memory.fix_memory import FixMemory
    fm = FixMemory(repo_url="https://example.com/repo", data_dir=tmp_path)
    fm.initialise()
    assert fm._backend == "json"
    return fm


def _inject_record(json_path: Path, days_old: int, **extra):
    """Write a single record with a specific age directly into the JSON store."""
    records = []
    if json_path.exists():
        records = json.loads(json_path.read_text())
    ts = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    record = {
        "issue_type":   "old_null_deref",
        "file_context": "legacy.py",
        "fix_approach": "Old inline guard",
        "test_result":  "passed=1 failed=0",
        "run_id":       "old-run",
        "created_at":   ts,
        "reverted":     False,
    }
    record.update(extra)
    records.append(record)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(records))


# ===========================================================================
# Defect 1 — Revert memory: store_failure called after all probe rounds fail
# ===========================================================================

class TestRevertMemoryCallPath:
    """
    FixerAgent._store_failure_memory must be called when every feedback
    round returns test_passed=False.
    """

    @pytest.mark.asyncio
    async def test_store_failure_called_when_all_rounds_fail(self, tmp_path):
        storage = _make_storage(tmp_path)
        await storage.initialise()

        mock_fm = MagicMock()
        mock_fm.store_failure = MagicMock()

        fixer = _make_fixer(storage, fix_memory=mock_fm, repo_root=tmp_path)
        fixer._probe_candidate = AsyncMock(
            return_value=(False, "AssertionError: expected 1 got 0")
        )

        from agents.fixer import FixResponse
        fixer.call_llm_structured = AsyncMock(
            return_value=FixResponse(fixed_files=[], overall_notes="")
        )
        fixer._load_file = AsyncMock(return_value="def foo():\n    pass\n")

        issues = [_make_issue()]
        await fixer._fix_group(frozenset(["foo.py"]), issues)

        assert mock_fm.store_failure.call_count == 1, (
            "store_failure must be called exactly once when all probe rounds fail"
        )

    @pytest.mark.asyncio
    async def test_store_failure_not_called_when_probe_passes(self, tmp_path):
        storage = _make_storage(tmp_path)
        await storage.initialise()

        mock_fm = MagicMock()
        mock_fm.store_failure = MagicMock()

        fixer = _make_fixer(storage, fix_memory=mock_fm, repo_root=tmp_path)
        fixer._probe_candidate = AsyncMock(
            return_value=(True, "passed=5 failed=0")
        )

        from agents.fixer import FixResponse
        fixer.call_llm_structured = AsyncMock(
            return_value=FixResponse(fixed_files=[], overall_notes="")
        )
        fixer._load_file = AsyncMock(return_value="def foo():\n    pass\n")

        issues = [_make_issue()]
        await fixer._fix_group(frozenset(["foo.py"]), issues)

        mock_fm.store_failure.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_failure_receives_last_round_output(self, tmp_path):
        """
        The failure_reason passed to store_failure must be the output from the
        final (third) probe round, not an earlier one.
        """
        storage = _make_storage(tmp_path)
        await storage.initialise()

        call_log: list[dict] = []

        class TrackingFixMemory:
            def store_failure(self, issue_type, file_context, fix_approach,
                              failure_reason, run_id=""):
                call_log.append({"failure_reason": failure_reason})

            def retrieve(self, issue_description, n=3, max_age_days=180):
                return []

            def format_as_few_shot(self, entries):
                return ""

        fixer = _make_fixer(storage, fix_memory=TrackingFixMemory(),
                             repo_root=tmp_path)

        outputs = [
            (False, "round-1 failure"),
            (False, "round-2 failure"),
            (False, "round-3 FINAL failure output"),
        ]
        fixer._probe_candidate = AsyncMock(side_effect=outputs)

        from agents.fixer import FixResponse, FixedFileFullResponse
        fixer.call_llm_structured = AsyncMock(return_value=FixResponse(
            fixed_files=[FixedFileFullResponse(
                path="foo.py", content="def foo(): pass",
                issues_resolved=[], changes_made="added guard",
                diff_summary="Added None check", confidence=0.7,
            )],
            overall_notes="",
        ))
        fixer._load_file = AsyncMock(
            return_value="def foo():\n    return x.attr\n"
        )

        issues = [_make_issue(description="null deref on x.attr")]
        await fixer._fix_group(frozenset(["foo.py"]), issues)

        assert len(call_log) == 1
        assert "round-3 FINAL failure output" in call_log[0]["failure_reason"]

    @pytest.mark.asyncio
    async def test_store_failure_no_crash_without_fix_memory(self, tmp_path):
        """_store_failure_memory is a no-op when fix_memory is not wired."""
        storage = _make_storage(tmp_path)
        await storage.initialise()

        fixer = _make_fixer(storage, fix_memory=None, repo_root=tmp_path)
        fixer._probe_candidate = AsyncMock(return_value=(False, "failed"))

        from agents.fixer import FixResponse
        fixer.call_llm_structured = AsyncMock(
            return_value=FixResponse(fixed_files=[], overall_notes="")
        )
        fixer._load_file = AsyncMock(return_value="def foo(): pass\n")

        result = await fixer._fix_group(frozenset(["foo.py"]), [_make_issue()])
        assert result is not None


# ---------------------------------------------------------------------------
# Defect 1 (cont.) — FixMemory round-trip: store_failure → retrieve → format
# ---------------------------------------------------------------------------

class TestFixMemoryRevertRoundTrip:

    def test_store_failure_writes_reverted_prefix(self, tmp_path):
        fm = _json_fm(tmp_path)
        fm.store_failure(
            issue_type="null_deref",
            file_context="auth.py:validate_session",
            fix_approach="Added early return on None",
            failure_reason="test_validate_session_raises failed",
            run_id="run-001",
        )
        records = json.loads((tmp_path / "fix_memory.json").read_text())
        assert len(records) == 1
        assert records[0]["fix_approach"].startswith("[REVERTED]")
        assert records[0]["reverted"] is True

    def test_retrieve_returns_reverted_entry(self, tmp_path):
        fm = _json_fm(tmp_path)
        fm.store_failure(
            issue_type="null_deref",
            file_context="auth.py",
            fix_approach="Added early return",
            failure_reason="regression in test_auth",
        )
        entries = fm.retrieve("null_deref null pointer exception")
        reverted = [e for e in entries if e.fix_approach.startswith("[REVERTED]")]
        assert len(reverted) == 1, "retrieve must return [REVERTED] entries"

    def test_format_as_few_shot_separates_negatives(self, tmp_path):
        fm = _json_fm(tmp_path)
        fm.store_success(
            issue_type="null_deref",
            file_context="auth.py",
            fix_approach="Added is None guard",
            test_result="passed=10 failed=0",
        )
        fm.store_failure(
            issue_type="null_deref",
            file_context="auth.py",
            fix_approach="Raised ValueError instead",
            failure_reason="broke 3 callers",
        )
        entries = fm.retrieve("null deref auth.py")
        text = fm.format_as_few_shot(entries)

        assert "Successful Fix Patterns" in text
        assert "Previously Reverted" in text
        assert "DO NOT REPEAT" in text
        assert text.index("Successful") < text.index("Previously Reverted")

    def test_format_as_few_shot_empty_returns_empty_string(self, tmp_path):
        fm = _json_fm(tmp_path)
        assert fm.format_as_few_shot([]) == ""

    def test_store_success_not_marked_reverted(self, tmp_path):
        fm = _json_fm(tmp_path)
        fm.store_success(
            issue_type="sql_injection",
            file_context="db.py",
            fix_approach="Parameterised query",
            test_result="passed=5 failed=0",
        )
        entries = fm.retrieve("sql injection db.py")
        assert all(not e.fix_approach.startswith("[REVERTED]") for e in entries)


# ===========================================================================
# Defect 2 — _generate_refactor_proposal called when blast exceeds gate
# ===========================================================================

class TestRefactorProposalPath:

    def _make_blast(self, fn_count: int, threshold: int):
        from cpg.cpg_engine import CPGBlastRadius
        return CPGBlastRadius(
            changed_functions=["payment_service.process_payment"],
            affected_functions=[
                {"function_name": f"fn_{i}", "file_path": "dep.py",
                 "line_number": i, "relationship": "direct_caller"}
                for i in range(fn_count)
            ],
            affected_files=["dep.py", "test_dep.py"],
            affected_function_count=fn_count,
            affected_file_count=2,
            test_files_affected=["test_dep.py"],
            blast_radius_score=min(fn_count / 200.0, 1.0),
            requires_human_review=(fn_count >= threshold),
            source="cpg",
        )

    def _fake_refactor_response(self):
        from pydantic import BaseModel, Field as F

        class _R(BaseModel):
            affected_components: list[str] = F(default_factory=list)
            proposed_refactoring: str = "Use adapter pattern"
            migration_steps: list[str] = F(default_factory=list)
            estimated_scope: str = "3 days"
            risks: list[str] = F(default_factory=list)
            recommendation: str = "Route to human review"

        return _R()

    @pytest.mark.asyncio
    async def test_blast_exceeded_aborts_patch(self, tmp_path):
        """blast_radius_exceeded=True and no fixed_files when gate fires."""
        storage = _make_storage(tmp_path)
        await storage.initialise()

        mock_cpg = AsyncMock()
        mock_cpg.is_available = True
        mock_cpg.compute_blast_radius = AsyncMock(
            return_value=self._make_blast(fn_count=100, threshold=50)
        )

        fixer = _make_fixer(storage, repo_root=tmp_path,
                             cpg_engine=mock_cpg, blast_radius_threshold=50)
        fixer._extract_file_symbols = AsyncMock(return_value={"process_payment"})
        fixer._load_file = AsyncMock(return_value="def process_payment(): pass\n")
        fixer.call_llm_structured = AsyncMock(
            return_value=self._fake_refactor_response()
        )

        result = await fixer._fix_group(
            frozenset(["payment_service.py"]),
            [_make_issue(file_path="payment_service.py")],
        )

        assert result.blast_radius_exceeded is True
        assert result.fixed_files == []
        assert result.refactor_proposal_id != ""

    @pytest.mark.asyncio
    async def test_refactor_proposal_persisted_to_storage(self, tmp_path):
        """RefactorProposal must be retrievable from storage after gate fires."""
        storage = _make_storage(tmp_path)
        await storage.initialise()

        mock_cpg = AsyncMock()
        mock_cpg.is_available = True
        mock_cpg.compute_blast_radius = AsyncMock(
            return_value=self._make_blast(fn_count=75, threshold=50)
        )

        fixer = _make_fixer(storage, repo_root=tmp_path,
                             cpg_engine=mock_cpg, blast_radius_threshold=50)
        fixer._extract_file_symbols = AsyncMock(return_value={"do_work"})
        fixer._load_file = AsyncMock(return_value="def do_work(): pass\n")
        fixer.call_llm_structured = AsyncMock(
            return_value=self._fake_refactor_response()
        )

        fix = await fixer._fix_group(frozenset(["foo.py"]), [_make_issue()])

        proposal = await storage.get_refactor_proposal(fix.refactor_proposal_id)
        assert proposal is not None, "RefactorProposal must be persisted to storage"
        assert proposal.requires_human_review is True
        assert proposal.affected_function_count == 75

    @pytest.mark.asyncio
    async def test_escalation_manager_called_on_blast_exceeded(self, tmp_path):
        """EscalationManager.create_escalation called with BLAST_RADIUS_EXCEEDED."""
        storage = _make_storage(tmp_path)
        await storage.initialise()

        mock_cpg = AsyncMock()
        mock_cpg.is_available = True
        mock_cpg.compute_blast_radius = AsyncMock(
            return_value=self._make_blast(fn_count=60, threshold=50)
        )

        from brain.schemas import Escalation
        mock_esc = AsyncMock()
        mock_esc.create_escalation = AsyncMock(return_value=Escalation(
            run_id="test-run",
            escalation_type="BLAST_RADIUS_EXCEEDED",
            description="test",
        ))

        fixer = _make_fixer(storage, repo_root=tmp_path,
                             cpg_engine=mock_cpg, escalation_manager=mock_esc,
                             blast_radius_threshold=50)
        fixer._extract_file_symbols = AsyncMock(return_value={"fn"})
        fixer._load_file = AsyncMock(return_value="def fn(): pass\n")
        fixer.call_llm_structured = AsyncMock(
            return_value=self._fake_refactor_response()
        )

        await fixer._fix_group(frozenset(["foo.py"]), [_make_issue()])

        mock_esc.create_escalation.assert_called_once()
        kwargs = mock_esc.create_escalation.call_args
        etype = kwargs.kwargs.get("escalation_type") or (
            kwargs.args[0] if kwargs.args else ""
        )
        assert etype == "BLAST_RADIUS_EXCEEDED"

    @pytest.mark.asyncio
    async def test_below_threshold_does_not_trigger_refactor(self, tmp_path):
        """blast_radius=30 (< threshold 50) must not abort patch generation."""
        storage = _make_storage(tmp_path)
        await storage.initialise()

        mock_cpg = AsyncMock()
        mock_cpg.is_available = True
        mock_cpg.compute_blast_radius = AsyncMock(
            return_value=self._make_blast(fn_count=30, threshold=50)
        )

        fixer = _make_fixer(storage, repo_root=tmp_path,
                             cpg_engine=mock_cpg, blast_radius_threshold=50)
        fixer._extract_file_symbols = AsyncMock(return_value={"fn"})
        fixer._load_file = AsyncMock(return_value="def fn(): pass\n")
        fixer._probe_candidate = AsyncMock(return_value=(True, "passed=1 failed=0"))

        from agents.fixer import FixResponse
        fixer.call_llm_structured = AsyncMock(
            return_value=FixResponse(fixed_files=[], overall_notes="")
        )

        result = await fixer._fix_group(frozenset(["foo.py"]), [_make_issue()])
        assert result.blast_radius_exceeded is False


# ===========================================================================
# Defect 3 — Time-window filter on FixMemory.retrieve()
# ===========================================================================

class TestFixMemoryTimeWindowFilter:
    """
    retrieve(max_age_days=N) excludes entries older than N days.
    max_age_days=0 means unlimited (no filter).
    Entries without created_at are retained (backward compat).
    """

    def test_old_entries_excluded_by_default(self, tmp_path):
        """Entries older than 180 days (default) are not returned."""
        fm = _json_fm(tmp_path)
        _inject_record(tmp_path / "fix_memory.json", days_old=200)

        entries = fm.retrieve("null deref legacy.py")
        old = [e for e in entries if e.issue_type == "old_null_deref"]
        assert old == [], "Entries > 180 days old must not be returned by default"

    def test_fresh_entries_returned(self, tmp_path):
        """Entries from yesterday are within the window and must be returned."""
        fm = _json_fm(tmp_path)
        fm.store_success(
            issue_type="null_deref",
            file_context="fresh.py",
            fix_approach="Added guard",
            test_result="passed=5 failed=0",
        )
        entries = fm.retrieve("null deref fresh.py")
        fresh = [e for e in entries if e.issue_type == "null_deref"]
        assert len(fresh) >= 1

    def test_explicit_max_age_days_respected(self, tmp_path):
        """max_age_days=7 excludes a 30-day-old entry but keeps a fresh one."""
        fm = _json_fm(tmp_path)
        _inject_record(tmp_path / "fix_memory.json", days_old=30)
        fm.store_success(
            issue_type="null_deref",
            file_context="fresh.py",
            fix_approach="Added guard",
            test_result="passed=5 failed=0",
        )

        entries = fm.retrieve("null deref", max_age_days=7)
        old   = [e for e in entries if e.issue_type == "old_null_deref"]
        fresh = [e for e in entries if e.issue_type == "null_deref"]
        assert old == [], "30-day-old entry must be excluded with max_age_days=7"
        assert len(fresh) >= 1

    def test_max_age_days_zero_means_unlimited(self, tmp_path):
        """max_age_days=0 disables age filtering — all entries are returned."""
        fm = _json_fm(tmp_path)
        _inject_record(tmp_path / "fix_memory.json", days_old=365)

        entries = fm.retrieve("null deref old legacy.py", max_age_days=0)
        old = [e for e in entries if e.issue_type == "old_null_deref"]
        assert len(old) >= 1, "max_age_days=0 must disable filtering"

    def test_boundary_entry_within_window_included(self, tmp_path):
        """An entry 179 days old must be included when max_age_days=180."""
        fm = _json_fm(tmp_path)
        _inject_record(tmp_path / "fix_memory.json", days_old=179)

        entries = fm.retrieve("null deref old legacy.py", max_age_days=180)
        old = [e for e in entries if e.issue_type == "old_null_deref"]
        assert len(old) >= 1, "Entry at 179 days must be within max_age_days=180"

    def test_reverted_entry_also_subject_to_time_window(self, tmp_path):
        """A [REVERTED] entry from 200 days ago must be excluded."""
        fm = _json_fm(tmp_path)
        json_path = tmp_path / "fix_memory.json"
        old_ts = (
            datetime.now(timezone.utc) - timedelta(days=200)
        ).isoformat()
        json_path.write_text(json.dumps([{
            "issue_type":   "null_deref",
            "file_context": "legacy.py",
            "fix_approach": "[REVERTED] Old bad approach",
            "test_result":  "REGRESSION: broke 5 tests",
            "run_id":       "old-run",
            "created_at":   old_ts,
            "reverted":     True,
        }]))

        entries = fm.retrieve("null deref legacy.py")
        reverted = [e for e in entries if e.fix_approach.startswith("[REVERTED]")]
        assert reverted == [], "[REVERTED] entries must also respect max_age_days"

    def test_missing_created_at_retained_for_backward_compat(self, tmp_path):
        """
        Records without created_at (pre-timestamp legacy entries) are RETAINED
        — the filter must not silently drop them on upgrade.
        """
        fm = _json_fm(tmp_path)
        json_path = tmp_path / "fix_memory.json"
        json_path.write_text(json.dumps([{
            "issue_type":   "null_deref",
            "file_context": "legacy.py",
            "fix_approach": "No timestamp",
            "test_result":  "passed=1 failed=0",
            "run_id":       "run-legacy",
            # deliberately omit created_at
        }]))

        entries = fm.retrieve("null deref", max_age_days=30)
        no_ts = [e for e in entries if e.fix_approach == "No timestamp"]
        assert len(no_ts) >= 1, (
            "Records without created_at must be retained (backward compat)"
        )


# ===========================================================================
# Defect 3 (cont.) — fixer._get_memory_examples passes max_age_days=180
# ===========================================================================

class TestFixerPassesMaxAgeDaysToRetrieve:
    """
    _get_memory_examples must explicitly pass max_age_days=180 so the
    6-month staleness window is an enforced call contract.
    """

    @pytest.mark.asyncio
    async def test_get_memory_examples_passes_max_age_days(self, tmp_path):
        storage = _make_storage(tmp_path)
        await storage.initialise()

        retrieve_calls: list[dict] = []

        class RecordingFixMemory:
            def retrieve(self, issue_description, n=3, max_age_days=180):
                retrieve_calls.append({
                    "n": n,
                    "max_age_days": max_age_days,
                })
                return []

            def format_as_few_shot(self, entries):
                return ""

        fixer = _make_fixer(storage, fix_memory=RecordingFixMemory())
        await fixer._get_memory_examples([_make_issue(description="null pointer")])

        assert len(retrieve_calls) == 1
        assert retrieve_calls[0]["max_age_days"] == 180, (
            "_get_memory_examples must pass max_age_days=180 to retrieve()"
        )


# ===========================================================================
# Defect 2 (cont.) — API route for refactor proposals
# ===========================================================================

class TestRefactorProposalsAPIRoute:
    """
    Verifies the refactor_proposals router exists, exposes correct endpoints,
    and returns stored proposals through the FastAPI test client.
    """

    def test_router_is_importable(self):
        from api.routes import refactor_proposals
        assert hasattr(refactor_proposals, "router")

    def test_router_has_list_and_detail_routes(self):
        from api.routes.refactor_proposals import router
        paths = {r.path for r in router.routes}  # type: ignore[attr-defined]
        # Router prefix is /api/refactor-proposals (hyphens)
        assert any(p.endswith("/") and "refactor" in p for p in paths), (
            f"List route missing. Found: {paths}"
        )
        assert any("{proposal_id}" in p for p in paths), (
            f"Detail route missing. Found: {paths}"
        )

    @pytest.mark.asyncio
    async def test_list_returns_stored_proposals(self, tmp_path):
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from api.routes.refactor_proposals import router
        from brain.sqlite_storage import SQLiteBrainStorage
        from brain.schemas import RefactorProposal

        app = FastAPI()
        app.include_router(router)

        storage = SQLiteBrainStorage(tmp_path / "brain.db")
        await storage.initialise()

        proposal = RefactorProposal(
            run_id="run-abc",
            issue_ids=["issue-1"],
            affected_function_count=60,
            affected_file_count=5,
            proposed_refactoring="Adapter pattern",
            recommendation="Start with shim layer",
            requires_human_review=True,
        )
        await storage.upsert_refactor_proposal(proposal)

        @app.middleware("http")
        async def inject_storage(request: Request, call_next):
            request.app.state.storage = storage
            return await call_next(request)

        # Use the actual prefix from the router
        from api.routes.refactor_proposals import router as rp_router
        prefix = rp_router.prefix  # /api/refactor-proposals

        client = TestClient(app)
        resp = client.get(f"{prefix}/", params={"run_id": "run-abc"})
        assert resp.status_code == 200
        body = resp.json()
        # List endpoint may return a list or a dict with "items"
        items = body if isinstance(body, list) else body.get("items", [])
        assert len(items) >= 1
        assert items[0]["run_id"] == "run-abc"
        assert items[0]["affected_function_count"] == 60

    @pytest.mark.asyncio
    async def test_detail_returns_404_for_unknown(self, tmp_path):
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from api.routes.refactor_proposals import router
        from brain.sqlite_storage import SQLiteBrainStorage

        app = FastAPI()
        app.include_router(router)

        storage = SQLiteBrainStorage(tmp_path / "brain2.db")
        await storage.initialise()

        @app.middleware("http")
        async def inject_storage(request: Request, call_next):
            request.app.state.storage = storage
            return await call_next(request)

        prefix = router.prefix
        client = TestClient(app)
        resp = client.get(f"{prefix}/nonexistent-id-xyz")
        assert resp.status_code == 404

    def test_router_registered_in_app(self):
        """app.py must register api.routes.refactor_proposals."""
        import inspect
        import api.app as app_module
        src = inspect.getsource(app_module)
        assert "refactor_proposals" in src
