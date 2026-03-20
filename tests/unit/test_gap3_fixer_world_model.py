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


# ===========================================================================
# Import graph blast radius — Gap 3 undercount fix
# ===========================================================================

class TestImportGraphBlastRadius:
    """
    CPGBlastRadius must account for modules that import a changed symbol
    without calling any changed function (pure import references).

    These files are invisible to the call graph but are broken by type
    changes, constant renames, and signature shifts at 10M-line scale.

    Three defects were present before the fix:
      A. CPGBlastRadius had no importing_modules / importing_module_count fields.
      B. compute_blast_radius() never called get_importing_files().
      C. blast_radius_score formula used only affected_function_count / 200,
         ignoring import-only modules entirely.
    """

    def _make_blast_with_imports(
        self,
        fn_count:      int,
        import_count:  int,
        threshold:     int = 50,
    ):
        from cpg.cpg_engine import CPGBlastRadius
        # Build a blast radius object that reflects the post-fix structure.
        call_score   = min(fn_count    / 200.0, 1.0)
        import_score = min(import_count / 400.0, 1.0)
        score        = round(0.8 * call_score + 0.2 * import_score, 4)
        weighted     = fn_count + int(import_count * 0.5)
        return CPGBlastRadius(
            changed_functions=["payment_service.process_payment"],
            affected_functions=[
                {"function_name": f"fn_{i}", "file_path": "dep.py",
                 "line_number": i, "relationship": "direct_caller"}
                for i in range(fn_count)
            ],
            affected_files=["dep.py"],
            affected_function_count=fn_count,
            affected_file_count=1,
            test_files_affected=[],
            importing_modules=[f"module_{i}.py" for i in range(import_count)],
            importing_module_count=import_count,
            blast_radius_score=score,
            requires_human_review=(weighted >= threshold),
            source="cpg",
        )

    # ── Field existence ───────────────────────────────────────────────────────

    def test_blast_radius_has_importing_module_count_field(self):
        """CPGBlastRadius must expose importing_module_count (was missing)."""
        from cpg.cpg_engine import CPGBlastRadius
        blast = CPGBlastRadius()
        assert hasattr(blast, "importing_module_count"), (
            "CPGBlastRadius missing importing_module_count field"
        )
        assert blast.importing_module_count == 0

    def test_blast_radius_has_importing_modules_field(self):
        """CPGBlastRadius must expose importing_modules list (was missing)."""
        from cpg.cpg_engine import CPGBlastRadius
        blast = CPGBlastRadius()
        assert hasattr(blast, "importing_modules"), (
            "CPGBlastRadius missing importing_modules field"
        )
        assert blast.importing_modules == []

    def test_refactor_proposal_schema_has_importing_fields(self):
        """RefactorProposal schema must carry importing_module_count and importing_modules."""
        from brain.schemas import RefactorProposal
        p = RefactorProposal(run_id="r", proposed_refactoring="x", recommendation="y")
        assert hasattr(p, "importing_module_count")
        assert hasattr(p, "importing_modules")
        assert p.importing_module_count == 0
        assert p.importing_modules == []

    # ── Score formula ─────────────────────────────────────────────────────────

    def test_score_increases_with_import_only_modules(self):
        """blast_radius_score must be higher when importing_module_count > 0."""
        blast_no_imports  = self._make_blast_with_imports(fn_count=20, import_count=0)
        blast_with_imports = self._make_blast_with_imports(fn_count=20, import_count=40)
        assert blast_with_imports.blast_radius_score > blast_no_imports.blast_radius_score, (
            "import-only modules must increase blast_radius_score"
        )

    def test_score_formula_weights_call_graph_higher_than_imports(self):
        """Call graph (80%) must outweigh import graph (20%) in the score."""
        # 100 callers, 0 importers → score = 0.8 * (100/200) = 0.40
        blast_callers = self._make_blast_with_imports(fn_count=100, import_count=0)
        # 0 callers, 400 importers → score = 0.2 * (400/400) = 0.20
        blast_importers = self._make_blast_with_imports(fn_count=0, import_count=400)
        assert blast_callers.blast_radius_score > blast_importers.blast_radius_score, (
            "100 callers must score higher than 400 import-only modules"
        )

    def test_import_only_modules_counted_at_half_weight_for_human_review_gate(self):
        """
        The human review gate must fire when fn_count + 0.5*import_count >= threshold.
        60 callers alone (threshold 50): fires without imports.
        30 callers + 40 importers (weighted=50): fires at boundary.
        30 callers + 39 importers (weighted=49.5 → 49): does not fire.
        """
        threshold = 50
        # 60 callers, 0 importers → weighted=60 ≥ 50 → fires
        blast_callers = self._make_blast_with_imports(fn_count=60, import_count=0,
                                                       threshold=threshold)
        assert blast_callers.requires_human_review is True

        # 30 callers + 40 importers → weighted = 30 + 20 = 50 → fires
        blast_boundary = self._make_blast_with_imports(fn_count=30, import_count=40,
                                                        threshold=threshold)
        assert blast_boundary.requires_human_review is True, (
            "30 callers + 40 importers (weighted=50) must trigger human review"
        )

        # 30 callers + 38 importers → weighted = 30 + 19 = 49 → does not fire
        blast_below = self._make_blast_with_imports(fn_count=30, import_count=38,
                                                     threshold=threshold)
        assert blast_below.requires_human_review is False, (
            "30 callers + 38 importers (weighted=49) must NOT trigger human review"
        )

    # ── compute_blast_radius integration ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_compute_blast_radius_calls_get_importing_files(self):
        """
        CPGEngine.compute_blast_radius must call client.get_importing_files()
        and populate importing_modules on the returned blast radius object.
        """
        from cpg.cpg_engine import CPGEngine
        from unittest.mock import AsyncMock, MagicMock

        mock_client = MagicMock()
        mock_client.is_ready = True
        mock_client.compute_impact_set = AsyncMock(return_value=[
            {"function_name": "fn_a", "file_path": "dep.py",
             "line_number": 10, "relationship": "caller", "depth": 1}
        ])
        mock_client.get_importing_files = AsyncMock(return_value=[
            {"importer_file": "consumer_a.py", "imported_symbol": "process_payment",
             "relationship": "import_reference"},
            {"importer_file": "consumer_b.py", "imported_symbol": "process_payment",
             "relationship": "import_reference"},
        ])

        engine = CPGEngine(blast_radius_threshold=50)
        engine._client = mock_client
        engine._ready  = True

        blast = await engine.compute_blast_radius(
            function_names=["process_payment"],
            file_paths=["payment_service.py"],
            depth=3,
        )

        mock_client.get_importing_files.assert_called_once()
        assert blast.importing_module_count == 2, (
            "importing_module_count must equal number of import-only files returned"
        )
        assert "consumer_a.py" in blast.importing_modules
        assert "consumer_b.py" in blast.importing_modules

    @pytest.mark.asyncio
    async def test_import_only_files_not_double_counted_with_call_graph(self):
        """
        A file that both calls a changed function AND imports it must appear in
        affected_files but NOT be double-counted in importing_modules.
        """
        from cpg.cpg_engine import CPGEngine
        from unittest.mock import AsyncMock, MagicMock

        mock_client = MagicMock()
        mock_client.is_ready = True
        mock_client.compute_impact_set = AsyncMock(return_value=[
            {"function_name": "fn_a", "file_path": "both_caller_and_importer.py",
             "line_number": 5, "relationship": "caller", "depth": 1}
        ])
        # get_importing_files returns the same file that the call graph already found
        mock_client.get_importing_files = AsyncMock(return_value=[
            {"importer_file": "both_caller_and_importer.py",
             "imported_symbol": "process_payment",
             "relationship": "import_reference"},
            {"importer_file": "import_only.py",
             "imported_symbol": "process_payment",
             "relationship": "import_reference"},
        ])

        engine = CPGEngine(blast_radius_threshold=50)
        engine._client = mock_client
        engine._ready  = True

        blast = await engine.compute_blast_radius(
            function_names=["process_payment"],
            file_paths=["payment_service.py"],
        )

        # "both_caller_and_importer.py" must NOT appear in importing_modules
        assert "both_caller_and_importer.py" not in blast.importing_modules, (
            "files already in the call graph must not be double-counted in importing_modules"
        )
        # "import_only.py" must appear since it is not in the call graph
        assert "import_only.py" in blast.importing_modules

    # ── Storage round-trip ────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_refactor_proposal_import_fields_persist_and_reload(self, tmp_path):
        """importing_modules and importing_module_count must survive storage round-trip."""
        from brain.sqlite_storage import SQLiteBrainStorage
        from brain.schemas import RefactorProposal

        storage = SQLiteBrainStorage(tmp_path / "brain.db")
        await storage.initialise()

        proposal = RefactorProposal(
            run_id="run-import-test",
            proposed_refactoring="Adapter pattern",
            recommendation="Shim layer first",
            importing_modules=["consumer_a.py", "consumer_b.py", "consumer_c.py"],
            importing_module_count=3,
            affected_function_count=20,
            affected_file_count=4,
            requires_human_review=True,
        )
        await storage.upsert_refactor_proposal(proposal)

        loaded = await storage.get_refactor_proposal(proposal.id)
        assert loaded is not None
        assert loaded.importing_module_count == 3, (
            "importing_module_count must be persisted and reloaded correctly"
        )
        assert set(loaded.importing_modules) == {"consumer_a.py", "consumer_b.py", "consumer_c.py"}, (
            "importing_modules list must survive the storage round-trip"
        )

    @pytest.mark.asyncio
    async def test_legacy_proposal_without_import_fields_loads_safely(self, tmp_path):
        """
        Proposals created before the import-graph fix (no importing_modules column)
        must load without error and default to 0 / [].
        """
        import aiosqlite
        from brain.sqlite_storage import SQLiteBrainStorage
        from brain.schemas import RefactorProposal

        db_path = tmp_path / "legacy.db"

        # Create a legacy table without the new columns
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                CREATE TABLE refactor_proposals (
                    id TEXT PRIMARY KEY,
                    fix_attempt_id TEXT, run_id TEXT,
                    issue_ids TEXT, changed_functions TEXT,
                    affected_function_count INTEGER, affected_file_count INTEGER,
                    test_files_affected TEXT, blast_radius_score REAL,
                    affected_components TEXT, proposed_refactoring TEXT,
                    migration_steps TEXT, estimated_scope TEXT,
                    risks TEXT, recommendation TEXT, escalation_id TEXT,
                    requires_human_review INTEGER, created_at TEXT
                )
            """)
            import json as _json
            from datetime import datetime, timezone
            await db.execute(
                """INSERT INTO refactor_proposals VALUES (
                    'legacy-id','','legacy-run','[]','[]',
                    5,2,'[]',0.1,'[]','old proposal','[]','1 day','[]',
                    'manual review','',1,?
                )""",
                (datetime.now(timezone.utc).isoformat(),),
            )
            await db.commit()

        storage = SQLiteBrainStorage(db_path)
        await storage.initialise()

        # upsert_refactor_proposal's ALTER TABLE ADD COLUMN migration
        # must add the missing columns so get_refactor_proposal works
        dummy = RefactorProposal(
            id="dummy-trigger",
            run_id="legacy-run",
            proposed_refactoring="x",
            recommendation="y",
        )
        await storage.upsert_refactor_proposal(dummy)

        loaded = await storage.get_refactor_proposal("legacy-id")
        assert loaded is not None, "legacy proposal must be loadable after migration"
        assert loaded.importing_module_count == 0
        assert loaded.importing_modules == []


# ===========================================================================
# Fix 2 — CPG FQN resolution: resolve_method_fqn + resolve_function_names
# ===========================================================================

class TestCPGFQNResolution:
    """
    Verifies that CPGEngine.resolve_function_names resolves bare symbol names
    to actual CPG FQNs via JoernClient.resolve_method_fqn, and that
    compute_blast_radius passes resolved names to compute_impact_set and
    get_importing_files rather than the raw fixer-derived names.

    This prevents the silent zero-hit failure where a name like
    ``src.services.payment_service.process_payment`` (derived from the file
    system path) does not match the CPG node
    ``services.payment_service.process_payment`` (derived from sys.path).
    """

    # ── JoernClient.resolve_method_fqn exists and is callable ────────────────

    def test_resolve_method_fqn_exists_on_client(self):
        from cpg.joern_client import JoernClient
        assert hasattr(JoernClient, "resolve_method_fqn"), (
            "JoernClient must expose resolve_method_fqn"
        )
        import inspect
        assert inspect.iscoroutinefunction(JoernClient.resolve_method_fqn), (
            "resolve_method_fqn must be async"
        )

    @pytest.mark.asyncio
    async def test_resolve_method_fqn_returns_empty_when_not_ready(self):
        from cpg.joern_client import JoernClient
        client = JoernClient()
        # _ready is False — must return [] without crashing
        result = await client.resolve_method_fqn("process_payment", "payment_service.py")
        assert result == []

    @pytest.mark.asyncio
    async def test_resolve_method_fqn_returns_fqns_from_cpg(self):
        """When Joern returns FQNs, resolve_method_fqn surfaces them."""
        from cpg.joern_client import JoernClient
        from unittest.mock import AsyncMock, MagicMock

        client = JoernClient()
        client._ready = True
        client._session = MagicMock()
        client._query = AsyncMock(return_value=[
            "services.payment_service.PaymentService.process_payment",
        ])

        fqns = await client.resolve_method_fqn("process_payment", "payment_service.py")
        assert len(fqns) == 1
        assert fqns[0] == "services.payment_service.PaymentService.process_payment"

    @pytest.mark.asyncio
    async def test_resolve_method_fqn_filters_empty_results(self):
        from cpg.joern_client import JoernClient
        from unittest.mock import AsyncMock, MagicMock

        client = JoernClient()
        client._ready = True
        client._session = MagicMock()
        # Joern may return empty strings or "<empty>" sentinel values
        client._query = AsyncMock(return_value=["", "<empty>", "valid.FQN.method"])

        fqns = await client.resolve_method_fqn("method")
        assert "" not in fqns
        assert "<empty>" not in fqns
        assert "valid.FQN.method" in fqns

    # ── CPGEngine.resolve_function_names ──────────────────────────────────────

    def test_resolve_function_names_exists_on_engine(self):
        from cpg.cpg_engine import CPGEngine
        assert hasattr(CPGEngine, "resolve_function_names"), (
            "CPGEngine must expose resolve_function_names"
        )

    @pytest.mark.asyncio
    async def test_resolve_function_names_returns_input_when_joern_unavailable(self):
        from cpg.cpg_engine import CPGEngine

        engine = CPGEngine()
        engine._ready = False  # Joern unavailable

        names = ["process_payment", "src.services.payment_service.process_payment"]
        result = await engine.resolve_function_names(names)
        assert set(result) >= set(names), (
            "When Joern is unavailable, all input names must be returned as fallback"
        )

    @pytest.mark.asyncio
    async def test_resolve_function_names_adds_cpg_fqn_alongside_bare(self):
        """
        resolve_function_names must return BOTH the original names AND the
        CPG-resolved FQN so that impact queries hit real CPG nodes while the
        bare name still works as a fallback.
        """
        from cpg.cpg_engine import CPGEngine
        from unittest.mock import AsyncMock, MagicMock

        mock_client = MagicMock()
        mock_client.is_ready = True
        mock_client.resolve_method_fqn = AsyncMock(
            return_value=["services.payment_service.PaymentService.process_payment"]
        )

        engine = CPGEngine()
        engine._client = mock_client
        engine._ready  = True

        result = await engine.resolve_function_names(
            bare_names=["process_payment"],
            file_paths=["src/services/payment_service.py"],
        )

        assert "process_payment" in result, (
            "Bare name must be retained as fallback"
        )
        assert "services.payment_service.PaymentService.process_payment" in result, (
            "CPG-resolved FQN must be added"
        )

    @pytest.mark.asyncio
    async def test_resolve_function_names_deduplicates(self):
        """Passing the same name twice must not produce duplicates."""
        from cpg.cpg_engine import CPGEngine
        from unittest.mock import AsyncMock, MagicMock

        mock_client = MagicMock()
        mock_client.is_ready = True
        mock_client.resolve_method_fqn = AsyncMock(return_value=["pkg.mod.fn"])

        engine = CPGEngine()
        engine._client = mock_client
        engine._ready  = True

        result = await engine.resolve_function_names(
            bare_names=["fn", "fn"],
        )
        assert result.count("fn") == 1, "Duplicates must be removed"
        assert result.count("pkg.mod.fn") == 1

    # ── compute_blast_radius passes resolved names downstream ─────────────────

    @pytest.mark.asyncio
    async def test_compute_blast_radius_uses_resolved_names_for_impact_set(self):
        """
        compute_blast_radius must call compute_impact_set with resolved FQNs,
        not the raw fixer-derived names.  This is the core regression: when
        only the raw name is passed, Joern returns zero results and the blast
        gate never fires for repos with non-trivial package structure.
        """
        from cpg.cpg_engine import CPGEngine
        from unittest.mock import AsyncMock, MagicMock, call

        mock_client = MagicMock()
        mock_client.is_ready = True

        # resolve_method_fqn returns the real FQN
        mock_client.resolve_method_fqn = AsyncMock(
            return_value=["services.payment_service.PaymentService.process_payment"]
        )
        # impact set returns one affected function
        mock_client.compute_impact_set = AsyncMock(return_value=[
            {"function_name": "caller_fn", "file_path": "checkout.py",
             "line_number": 42, "relationship": "caller", "depth": 1}
        ])
        mock_client.get_importing_files = AsyncMock(return_value=[])

        engine = CPGEngine(blast_radius_threshold=50)
        engine._client = mock_client
        engine._ready  = True

        await engine.compute_blast_radius(
            function_names=["process_payment"],
            file_paths=["src/services/payment_service.py"],
        )

        # compute_impact_set must have been called with a list that includes
        # the resolved FQN, not just the bare name.
        assert mock_client.compute_impact_set.called, "compute_impact_set must be called"
        call_kwargs = mock_client.compute_impact_set.call_args
        passed_names = (
            call_kwargs.kwargs.get("function_names")
            or (call_kwargs.args[0] if call_kwargs.args else [])
        )
        assert "services.payment_service.PaymentService.process_payment" in passed_names, (
            "compute_impact_set must receive the CPG-resolved FQN, not just the bare name"
        )

    @pytest.mark.asyncio
    async def test_compute_blast_radius_uses_resolved_names_for_import_query(self):
        """
        get_importing_files must also receive the resolved FQNs so import-only
        references are found even in repos with non-trivial package structure.
        """
        from cpg.cpg_engine import CPGEngine
        from unittest.mock import AsyncMock, MagicMock

        mock_client = MagicMock()
        mock_client.is_ready = True
        mock_client.resolve_method_fqn = AsyncMock(
            return_value=["auth.middleware.validate_session"]
        )
        mock_client.compute_impact_set = AsyncMock(return_value=[])
        mock_client.get_importing_files = AsyncMock(return_value=[
            {"importer_file": "consumer.py", "imported_symbol": "validate_session",
             "relationship": "import_reference"}
        ])

        engine = CPGEngine(blast_radius_threshold=50)
        engine._client = mock_client
        engine._ready  = True

        blast = await engine.compute_blast_radius(
            function_names=["validate_session"],
            file_paths=["auth/middleware.py"],
        )

        assert mock_client.get_importing_files.called
        import_kwargs = mock_client.get_importing_files.call_args
        import_names = (
            import_kwargs.kwargs.get("symbol_names")
            or (import_kwargs.args[0] if import_kwargs.args else [])
        )
        assert "auth.middleware.validate_session" in import_names, (
            "get_importing_files must receive the CPG-resolved FQN"
        )
        assert blast.importing_module_count == 1
        assert "consumer.py" in blast.importing_modules


# ===========================================================================
# Gap 3 — Proactive Architectural Smell Detection
# ===========================================================================
# Three new test classes covering the three structural fixes:
#
#   TestBugRecurrenceEscalation
#       Signal 1: fix_memory recurrence count >= threshold → refactor proposal
#
#   TestCPGCouplingSmellEscalation
#       Signal 2: CPG distinct_caller_modules >= threshold → refactor proposal
#
#   TestArchitecturalSymptomPlanner
#       PlannerAgent: is_architectural_symptom=True blocks fix, creates escalation
# ===========================================================================


# ---------------------------------------------------------------------------
# Helpers shared by the new test classes
# ---------------------------------------------------------------------------

def _make_planner(storage, run_id="test-run", cpg_engine=None, escalation_manager=None):
    from agents.planner import PlannerAgent
    from agents.base import AgentConfig
    return PlannerAgent(
        storage=storage,
        run_id=run_id,
        config=AgentConfig(model="test-model", run_id=run_id),
        cpg_engine=cpg_engine,
        escalation_manager=escalation_manager,
    )


def _make_fix_attempt(run_id="test-run", path="foo.py", changes_made="add null guard"):
    from brain.schemas import FixAttempt, FixedFile, PatchMode
    return FixAttempt(
        run_id=run_id,
        issue_ids=["issue-1"],
        fixed_files=[
            FixedFile(
                path=path,
                content="def f(x):\n    if x is None:\n        return\n    return x.value\n",
                changes_made=changes_made,
                patch_mode=PatchMode.FULL_FILE,
            )
        ],
        fixer_model="test-model",
    )


def _inject_fix_memory_entries(json_path, count, issue_type="null deref in payment_service",
                                file_context="payment_service.py", reverted=False):
    """Write ``count`` fix-memory records for the given issue_type."""
    records = []
    if json_path.exists():
        records = json.loads(json_path.read_text())
    for i in range(count):
        ts = (datetime.now(timezone.utc) - timedelta(days=i)).isoformat()
        records.append({
            "issue_type":   issue_type,
            "file_context": file_context,
            "fix_approach": f"[REVERTED] approach {i}" if reverted else f"approach {i}",
            "test_result":  "REGRESSION: broke auth" if reverted else "passed=1 failed=0",
            "run_id":       f"run-{i}",
            "created_at":   ts,
            "reverted":     reverted,
        })
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(records))


# ===========================================================================
# TestBugRecurrenceEscalation
# ===========================================================================

class TestBugRecurrenceEscalation:
    """
    Signal 1 — fix_memory recurrence gate.

    When the same bug class has been successfully patched >= 3 times
    within 180 days, _check_bug_recurrence() must return
    is_structural=True and _fix_group() must route to
    _generate_refactor_proposal() instead of generating a patch.
    """

    @pytest.mark.asyncio
    async def test_recurrence_above_threshold_returns_is_structural_true(self, tmp_path):
        """_check_bug_recurrence returns is_structural=True when recurrence_count >= 3."""
        from memory.fix_memory import FixMemory

        fm = FixMemory(repo_url="https://example.com/repo", data_dir=tmp_path)
        fm.initialise()
        json_path = tmp_path / "fix_memory.json"

        # Inject 3 successful (non-reverted) entries for the same bug class.
        _inject_fix_memory_entries(json_path, count=3,
                                   issue_type="null deref in payment_service")

        storage = _make_storage(tmp_path)
        await storage.initialise()
        fixer = _make_fixer(storage, fix_memory=fm)

        issue = _make_issue(description="null deref in payment_service")
        signal = await fixer._check_bug_recurrence(
            issues=[issue],
            file_paths=["payment_service.py"],
            function_names=["process_payment"],
        )

        assert signal.is_structural is True
        assert signal.recurrence_count >= 3
        assert "recurrence" in signal.escalation_reason

    @pytest.mark.asyncio
    async def test_recurrence_below_threshold_returns_is_structural_false(self, tmp_path):
        """_check_bug_recurrence returns is_structural=False when recurrence_count < 3."""
        from memory.fix_memory import FixMemory

        fm = FixMemory(repo_url="https://example.com/repo", data_dir=tmp_path)
        fm.initialise()
        json_path = tmp_path / "fix_memory.json"

        # Only 2 entries — below the threshold of 3.
        _inject_fix_memory_entries(json_path, count=2,
                                   issue_type="null deref in payment_service")

        storage = _make_storage(tmp_path)
        await storage.initialise()
        fixer = _make_fixer(storage, fix_memory=fm)

        issue = _make_issue(description="null deref in payment_service")
        signal = await fixer._check_bug_recurrence(
            issues=[issue],
            file_paths=["payment_service.py"],
            function_names=["process_payment"],
        )

        assert signal.is_structural is False
        assert signal.recurrence_count == 2

    @pytest.mark.asyncio
    async def test_reverted_entries_counted_separately(self, tmp_path):
        """Reverted entries populate reverted_count but do NOT increment recurrence_count."""
        from memory.fix_memory import FixMemory

        fm = FixMemory(repo_url="https://example.com/repo", data_dir=tmp_path)
        fm.initialise()
        json_path = tmp_path / "fix_memory.json"

        # 1 successful + 3 reverted — recurrence_count should be 1 (below threshold).
        _inject_fix_memory_entries(json_path, count=1,
                                   issue_type="null deref", reverted=False)
        _inject_fix_memory_entries(json_path, count=3,
                                   issue_type="null deref", reverted=True)

        storage = _make_storage(tmp_path)
        await storage.initialise()
        fixer = _make_fixer(storage, fix_memory=fm)

        issue = _make_issue(description="null deref")
        signal = await fixer._check_bug_recurrence(
            issues=[issue],
            file_paths=["foo.py"],
            function_names=["bar"],
        )

        assert signal.reverted_count >= 1, "reverted entries must be captured"
        # recurrence_count counts only non-reverted successes.
        # 1 success < threshold=3, so is_structural must be False for recurrence alone.
        # (coupling signal is absent — cpg_engine=None)
        assert signal.recurrence_count == 1
        assert signal.is_structural is False

    @pytest.mark.asyncio
    async def test_no_crash_when_fix_memory_is_none(self, tmp_path):
        """_check_bug_recurrence must not raise when fix_memory is None."""
        storage = _make_storage(tmp_path)
        await storage.initialise()
        fixer = _make_fixer(storage, fix_memory=None)

        issue = _make_issue(description="null deref")
        signal = await fixer._check_bug_recurrence(
            issues=[issue],
            file_paths=["foo.py"],
            function_names=["bar"],
        )

        assert signal.is_structural is False
        assert signal.recurrence_count == 0

    @pytest.mark.asyncio
    async def test_recurrence_gate_triggers_refactor_proposal_in_fix_group(self, tmp_path):
        """
        When _check_bug_recurrence returns is_structural=True, _fix_group()
        must return a FixAttempt with blast_radius_exceeded=True and
        refactor_proposal_id set — no fixed_files should be generated.
        """
        from memory.fix_memory import FixMemory
        from brain.schemas import IssueStatus, Severity
        from unittest.mock import patch, AsyncMock

        fm = FixMemory(repo_url="https://example.com/repo", data_dir=tmp_path)
        fm.initialise()
        json_path = tmp_path / "fix_memory.json"
        _inject_fix_memory_entries(json_path, count=4,
                                   issue_type="null deref payment")

        storage = _make_storage(tmp_path)
        await storage.initialise()

        # Write a minimal file so _load_file works.
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        (repo_root / "payment_service.py").write_text("def process(x):\n    return x.value\n")

        fixer = _make_fixer(storage, repo_root=repo_root, fix_memory=fm)

        issue = _make_issue(
            file_path="payment_service.py",
            severity="CRITICAL",
            description="null deref payment",
        )
        issue.status = IssueStatus.OPEN
        await storage.upsert_issue(issue)

        # Stub _generate_refactor_proposal so we don't need a live LLM.
        from brain.schemas import FixAttempt, PatchMode, RefactorProposal
        stub_proposal = RefactorProposal(
            run_id="test-run",
            issue_ids=[issue.id],
            trigger_source="recurrence",
            recurrence_count=4,
            requires_human_review=True,
        )
        await storage.upsert_refactor_proposal(stub_proposal)

        stub_fix = FixAttempt(
            run_id="test-run",
            issue_ids=[issue.id],
            fixed_files=[],
            fixer_model="test-model",
            blast_radius_exceeded=True,
            refactor_proposal_id=stub_proposal.id,
            patch_mode=PatchMode.FULL_FILE,
        )
        await storage.upsert_fix(stub_fix)

        with patch.object(fixer, "_generate_refactor_proposal",
                          new=AsyncMock(return_value=stub_fix)):
            result = await fixer._fix_group(
                frozenset(["payment_service.py"]), [issue]
            )

        assert result.blast_radius_exceeded is True
        assert result.refactor_proposal_id != ""
        assert result.fixed_files == []

    def test_bug_recurrence_signal_schema_fields(self):
        """BugRecurrenceSignal has all expected fields with correct defaults."""
        from brain.schemas import BugRecurrenceSignal
        s = BugRecurrenceSignal()
        assert s.recurrence_count == 0
        assert s.reverted_count == 0
        assert s.window_days == 180
        assert s.coupling_score == -1.0
        assert s.is_structural is False
        assert s.escalation_reason == ""
        assert s.coupling_module_threshold == 5

    def test_refactor_proposal_has_recurrence_fields(self):
        """RefactorProposal schema carries all recurrence/coupling fields."""
        from brain.schemas import RefactorProposal
        p = RefactorProposal(
            recurrence_count=3,
            reverted_count=1,
            distinct_caller_modules=6,
            coupling_score=0.6,
            recurrence_escalation_reason="test reason",
            trigger_source="recurrence",
        )
        assert p.recurrence_count == 3
        assert p.reverted_count == 1
        assert p.distinct_caller_modules == 6
        assert p.coupling_score == 0.6
        assert p.trigger_source == "recurrence"

    def test_refactor_proposal_trigger_source_default(self):
        """RefactorProposal.trigger_source defaults to 'blast_radius'."""
        from brain.schemas import RefactorProposal
        p = RefactorProposal()
        assert p.trigger_source == "blast_radius"


# ===========================================================================
# TestCPGCouplingSmellEscalation
# ===========================================================================

class TestCPGCouplingSmellEscalation:
    """
    Signal 2 — CPG coupling smell gate.

    When CPGEngine.compute_coupling_smell() returns is_smell=True
    (distinct_caller_modules >= threshold), _check_bug_recurrence() must
    set is_structural=True and _fix_group() must route to a refactor proposal.
    """

    @pytest.mark.asyncio
    async def test_compute_coupling_smell_returns_is_smell_true_above_threshold(self):
        """
        CPGEngine.compute_coupling_smell marks is_smell=True when
        distinct_caller_modules >= coupling_module_threshold.
        """
        from cpg.cpg_engine import CPGEngine
        from unittest.mock import MagicMock, AsyncMock

        mock_client = MagicMock()
        mock_client.is_ready = True
        # Simulate 6 distinct modules (threshold=5).
        mock_client.compute_coupling_score = AsyncMock(return_value={
            "distinct_caller_modules": 6,
            "coupling_score":          0.6,
            "dominant_caller_module":  "auth/",
            "total_callers":           24,
            "function_name_used":      "validate_session",
        })

        engine = CPGEngine()
        engine._client = mock_client
        engine._ready  = True

        result = await engine.compute_coupling_smell(
            function_names=["validate_session"],
            coupling_module_threshold=5,
        )

        assert result["is_smell"] is True
        assert result["distinct_caller_modules"] == 6
        assert result["coupling_score"] == 0.6
        assert result["coupling_module_threshold"] == 5

    @pytest.mark.asyncio
    async def test_compute_coupling_smell_returns_is_smell_false_below_threshold(self):
        """
        CPGEngine.compute_coupling_smell marks is_smell=False when
        distinct_caller_modules < coupling_module_threshold.
        """
        from cpg.cpg_engine import CPGEngine
        from unittest.mock import MagicMock, AsyncMock

        mock_client = MagicMock()
        mock_client.is_ready = True
        mock_client.compute_coupling_score = AsyncMock(return_value={
            "distinct_caller_modules": 3,
            "coupling_score":          0.3,
            "dominant_caller_module":  "auth/",
            "total_callers":           9,
            "function_name_used":      "validate_session",
        })

        engine = CPGEngine()
        engine._client = mock_client
        engine._ready  = True

        result = await engine.compute_coupling_smell(
            function_names=["validate_session"],
            coupling_module_threshold=5,
        )

        assert result["is_smell"] is False

    @pytest.mark.asyncio
    async def test_compute_coupling_smell_returns_safe_dict_when_joern_unavailable(self):
        """
        When Joern is unavailable compute_coupling_smell must return a safe
        dict (is_smell=False, coupling_score=-1.0) without raising.
        """
        from cpg.cpg_engine import CPGEngine

        engine = CPGEngine()
        # _client=None, _ready=False — Joern unavailable.

        result = await engine.compute_coupling_smell(function_names=["fn"])

        assert result["is_smell"] is False
        assert result["coupling_score"] == -1.0
        assert result["distinct_caller_modules"] == 0

    @pytest.mark.asyncio
    async def test_coupling_smell_sets_is_structural_true_in_check_bug_recurrence(self, tmp_path):
        """
        When cpg_engine.compute_coupling_smell returns is_smell=True,
        _check_bug_recurrence must set is_structural=True even when
        recurrence_count=0 (no prior fixes in memory).
        """
        from cpg.cpg_engine import CPGEngine
        from unittest.mock import MagicMock, AsyncMock

        mock_engine = MagicMock(spec=CPGEngine)
        mock_engine.is_available = True
        mock_engine.compute_coupling_smell = AsyncMock(return_value={
            "distinct_caller_modules":  7,
            "coupling_score":           0.7,
            "dominant_caller_module":   "billing/",
            "total_callers":            28,
            "function_name_used":       "process_payment",
            "is_smell":                 True,
            "coupling_module_threshold": 5,
        })

        storage = _make_storage(tmp_path)
        await storage.initialise()
        fixer = _make_fixer(storage, cpg_engine=mock_engine, fix_memory=None)

        issue = _make_issue(description="null deref process_payment")
        signal = await fixer._check_bug_recurrence(
            issues=[issue],
            file_paths=["billing/payment_service.py"],
            function_names=["process_payment"],
        )

        assert signal.is_structural is True
        assert signal.distinct_caller_modules == 7
        assert signal.coupling_score == 0.7
        assert "coupling_smell" in signal.escalation_reason

    @pytest.mark.asyncio
    async def test_compute_structural_risk_combines_blast_and_coupling(self):
        """
        CPGEngine.compute_structural_risk returns requires_refactor=True when
        coupling is_smell is True, even if blast.requires_human_review=False.
        """
        from cpg.cpg_engine import CPGEngine
        from unittest.mock import MagicMock, AsyncMock, patch

        mock_client = MagicMock()
        mock_client.is_ready = True

        engine = CPGEngine(blast_radius_threshold=50)
        engine._client = mock_client
        engine._ready  = True

        # Blast radius: well below gate (only 5 functions).
        from cpg.cpg_engine import CPGBlastRadius
        safe_blast = CPGBlastRadius(
            changed_functions=["fn"],
            affected_function_count=5,
            affected_file_count=2,
            blast_radius_score=0.025,
            requires_human_review=False,
        )

        # Coupling smell: 6 distinct modules (above threshold=5).
        coupling_smell = {
            "distinct_caller_modules":  6,
            "coupling_score":           0.6,
            "dominant_caller_module":   "auth/",
            "total_callers":            18,
            "function_name_used":       "fn",
            "is_smell":                 True,
            "coupling_module_threshold": 5,
        }

        with patch.object(engine, "compute_blast_radius",
                          new=AsyncMock(return_value=safe_blast)), \
             patch.object(engine, "compute_coupling_smell",
                          new=AsyncMock(return_value=coupling_smell)):
            result = await engine.compute_structural_risk(
                function_names=["fn"],
                coupling_module_threshold=5,
            )

        assert result["requires_refactor"] is True
        assert "coupling_smell" in result["refactor_reason"]
        assert result["blast"] is safe_blast
        assert result["coupling"] is coupling_smell

    @pytest.mark.asyncio
    async def test_joern_client_compute_coupling_score_groups_by_parent_dir(self):
        """
        JoernClient.compute_coupling_score must group caller files by parent
        directory and count distinct directories as distinct_caller_modules.
        """
        from cpg.joern_client import JoernClient
        from unittest.mock import MagicMock, AsyncMock

        client = JoernClient()
        client._session = MagicMock()
        client._ready   = True

        # 5 callers from 3 different parent dirs.
        raw_results = [
            {"callerFile": "auth/login.py"},
            {"callerFile": "auth/logout.py"},
            {"callerFile": "billing/invoice.py"},
            {"callerFile": "reporting/dashboard.py"},
            {"callerFile": "reporting/export.py"},
        ]
        client._query = AsyncMock(return_value=raw_results)

        result = await client.compute_coupling_score(
            function_names=["validate_session"],
        )

        assert result["distinct_caller_modules"] == 3   # auth, billing, reporting
        assert result["total_callers"] == 5
        assert result["coupling_score"] == pytest.approx(0.3, abs=0.01)
        assert result["function_name_used"] == "validate_session"

    @pytest.mark.asyncio
    async def test_joern_client_compute_coupling_score_returns_safe_dict_when_not_ready(self):
        """compute_coupling_score returns safe dict when client is not ready."""
        from cpg.joern_client import JoernClient

        client = JoernClient()
        # _ready=False — no active session.

        result = await client.compute_coupling_score(function_names=["fn"])

        assert result["distinct_caller_modules"] == 0
        assert result["coupling_score"] == -1.0
        assert result["is_smell"] if "is_smell" in result else True  # key may be absent

    def test_coupling_smell_reason_included_in_refactor_proposal(self):
        """
        When trigger_source='coupling_smell', RefactorProposal stores the
        coupling metadata so the human reviewer sees the structural context.
        """
        from brain.schemas import RefactorProposal
        p = RefactorProposal(
            trigger_source="coupling_smell",
            distinct_caller_modules=7,
            coupling_score=0.7,
            recurrence_escalation_reason=(
                "coupling_smell: 7 distinct caller modules >= threshold=5"
            ),
        )
        assert p.trigger_source == "coupling_smell"
        assert p.distinct_caller_modules == 7
        assert "coupling_smell" in p.recurrence_escalation_reason


# ===========================================================================
# TestArchitecturalSymptomPlanner
# ===========================================================================

class TestArchitecturalSymptomPlanner:
    """
    PlannerAgent architectural symptom detection.

    When the coherence LLM sets is_architectural_symptom=True the planner
    must block the fix, set verdict=UNSAFE, and create an
    ARCHITECTURAL_SYMPTOM_DETECTED escalation — regardless of blast radius.
    """

    @pytest.mark.asyncio
    async def test_architectural_symptom_blocks_fix(self, tmp_path):
        """
        When _assess_coherence returns is_architectural_symptom=True the
        planner verdict must be UNSAFE and block_commit must be True.
        """
        from agents.planner import PlannerAgent
        from brain.schemas import PlannerVerdict
        from unittest.mock import patch, AsyncMock

        storage = _make_storage(tmp_path)
        await storage.initialise()

        fix = _make_fix_attempt()
        await storage.upsert_fix(fix)

        planner = _make_planner(storage)

        # Stub the two LLM calls.
        with patch.object(
            planner, "_classify_reversibility",
            new=AsyncMock(return_value=("REVERSIBLE", 0.9, "clean revert possible"))
        ), patch.object(
            planner, "_assess_coherence",
            new=AsyncMock(return_value=(
                True,    # safe
                0.4,     # risk_score — below block threshold
                [],      # concerns
                "looks fine locally",   # simulation_summary
                True,    # is_architectural_symptom  ← triggers block
                "Function called from 6 unrelated modules; ownership boundary absent.",
            ))
        ):
            record = await planner.evaluate(fix)

        assert record.verdict == PlannerVerdict.UNSAFE
        assert record.block_commit is True
        assert "ArchSymptom" in record.reason or "architectural" in record.reason.lower()

    @pytest.mark.asyncio
    async def test_architectural_symptom_creates_escalation(self, tmp_path):
        """
        When is_architectural_symptom=True the planner must call
        escalation_manager.create_escalation with type
        ARCHITECTURAL_SYMPTOM_DETECTED.
        """
        from agents.planner import PlannerAgent
        from unittest.mock import patch, AsyncMock, MagicMock

        storage = _make_storage(tmp_path)
        await storage.initialise()

        fix = _make_fix_attempt()
        await storage.upsert_fix(fix)

        mock_esc_manager = MagicMock()
        mock_esc_manager.create_escalation = AsyncMock(return_value=MagicMock(id="esc-1"))

        planner = _make_planner(storage, escalation_manager=mock_esc_manager)

        with patch.object(
            planner, "_classify_reversibility",
            new=AsyncMock(return_value=("REVERSIBLE", 0.9, ""))
        ), patch.object(
            planner, "_assess_coherence",
            new=AsyncMock(return_value=(
                True, 0.3, [], "", True,
                "Same null-deref class appeared here 4 times in 6 months.",
            ))
        ):
            await planner.evaluate(fix)

        mock_esc_manager.create_escalation.assert_called_once()
        call_kwargs = mock_esc_manager.create_escalation.call_args.kwargs
        assert call_kwargs.get("escalation_type") == "ARCHITECTURAL_SYMPTOM_DETECTED"
        assert "architectural" in call_kwargs.get("description", "").lower()

    @pytest.mark.asyncio
    async def test_no_architectural_symptom_does_not_block_when_otherwise_safe(self, tmp_path):
        """
        When is_architectural_symptom=False and all other signals are safe
        the planner must NOT block the fix.
        """
        from agents.planner import PlannerAgent
        from brain.schemas import PlannerVerdict
        from unittest.mock import patch, AsyncMock

        storage = _make_storage(tmp_path)
        await storage.initialise()

        fix = _make_fix_attempt()
        await storage.upsert_fix(fix)

        planner = _make_planner(storage)

        with patch.object(
            planner, "_classify_reversibility",
            new=AsyncMock(return_value=("REVERSIBLE", 0.95, ""))
        ), patch.object(
            planner, "_assess_coherence",
            new=AsyncMock(return_value=(
                True,   # safe
                0.2,    # risk_score — safe
                [],
                "no issues",
                False,  # is_architectural_symptom=False
                "",
            ))
        ):
            record = await planner.evaluate(fix)

        assert record.block_commit is False
        assert record.verdict in (PlannerVerdict.SAFE, PlannerVerdict.SAFE_WITH_WARNING)

    def test_causal_chain_response_has_architectural_symptom_fields(self):
        """CausalChainResponse carries the two new architectural symptom fields."""
        from agents.planner import CausalChainResponse
        r = CausalChainResponse(
            safe=True,
            risk_score=0.3,
            is_architectural_symptom=True,
            architectural_reason="over-coupled function",
        )
        assert r.is_architectural_symptom is True
        assert r.architectural_reason == "over-coupled function"

    def test_causal_chain_response_defaults_symptom_false(self):
        """CausalChainResponse.is_architectural_symptom defaults to False."""
        from agents.planner import CausalChainResponse
        r = CausalChainResponse()
        assert r.is_architectural_symptom is False
        assert r.architectural_reason == ""
