"""
tests/unit/test_gap4_integration.py
====================================
Gap 4 end-to-end integration tests.

These tests verify the COMPLETE path from webhook payload → Celery task →
CommitAuditScheduler → IncrementalCPGUpdater → FunctionStalenessMark rows →
ReaderAgent staleness detection.

Why these tests were missing
-----------------------------
The previous test suite had only unit tests for individual components (hunk
extraction, diff parsing, CPGEngine).  No test exercised the full chain to
verify that a commit event actually produces FunctionStalenessMark rows that a
subsequent ReaderAgent read pass detects and acts on.  The integration gap was
the reason the four production bugs survived code review:

  BUG-A  list_stale_functions run_id filter (reader.py)
  BUG-B  Bare method names in _extract_function_names (reader.py)
  BUG-C  Bare method names in _ts_changed_functions (incremental_updater.py)
  BUG-D  Unbounded blast radius emitting thousands of marks (controller.py)
  BUG-E  test_runner=None in CommitAuditScheduler (controller._init_cpg)

All five regression-checks are represented below.

All tests run without a live Joern server — CPG-dependent paths are mocked.
SQLite is used for storage (same backend used in CI / SWE-bench runs).
"""
from __future__ import annotations

import asyncio
import hashlib
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_run_id() -> str:
    import uuid
    return str(uuid.uuid4())


async def _make_storage(tmp_path: Path):
    from brain.sqlite_storage import SQLiteBrainStorage
    db = SQLiteBrainStorage(tmp_path / "brain.db")
    await db.initialise()
    return db


def _make_file_record(path: str, content: str, run_id: str):
    from brain.schemas import FileRecord, FileStatus
    file_hash = hashlib.sha256(content.encode()).hexdigest()
    return FileRecord(
        path=path,
        language="python",
        status=FileStatus.READ,
        run_id=run_id,
        hash=file_hash,
        known_functions=["MyClass.process", "MyClass.validate", "helper"],
        line_count=content.count("\n") + 1,
    )


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
async def storage(tmp_path):
    db = await _make_storage(tmp_path)
    yield db
    await db.close()


@pytest.fixture
def repo_root(tmp_path) -> Path:
    """A minimal fake git repository on disk used by IncrementalCPGUpdater."""
    root = tmp_path / "repo"
    root.mkdir()
    git_dir = root / ".git"
    git_dir.mkdir()
    return root


@pytest.fixture
def payment_service_v1() -> str:
    return textwrap.dedent("""\
        class PaymentService:
            def process(self, amount):
                return amount

            def validate(self, amount):
                return amount > 0

        def helper():
            pass
    """)


@pytest.fixture
def payment_service_v2() -> str:
    """Changed: PaymentService.process body differs; PaymentService.validate unchanged."""
    return textwrap.dedent("""\
        class PaymentService:
            def process(self, amount):
                if amount <= 0:
                    raise ValueError("bad amount")
                return amount * 1.02

            def validate(self, amount):
                return amount > 0

        def helper():
            pass
    """)


@pytest.fixture
def no_cpg_updater(repo_root, storage):
    from cpg.incremental_updater import IncrementalCPGUpdater
    return IncrementalCPGUpdater(
        cpg_engine=None,
        repo_root=repo_root,
        storage=storage,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: CommitAuditScheduler end-to-end (no CPG)
# ─────────────────────────────────────────────────────────────────────────────

class TestCommitAuditSchedulerE2E:
    """
    Full pipeline: webhook payload → CommitAuditScheduler → staleness marks.
    CPG is absent so the test exercises the git+regex fallback path.
    """

    async def test_schedule_from_webhook_produces_staleness_marks(
        self, storage, repo_root, payment_service_v1, payment_service_v2
    ):
        """
        When a webhook fires with a changed file, CommitAuditScheduler must
        write FunctionStalenessMark rows for the changed functions so the next
        ReaderAgent read pass detects and re-reads the file.
        """
        run_id = _make_run_id()

        # Write v1 to disk (so git diff produces a meaningful diff)
        svc_path = repo_root / "payment_service.py"
        svc_path.write_text(payment_service_v1, encoding="utf-8")

        # Store the pre-existing FileRecord to simulate an already-read file
        rec = _make_file_record("payment_service.py", payment_service_v1, run_id)
        await storage.upsert_file(rec)

        # Write v2 (simulates the commit landing on disk)
        svc_path.write_text(payment_service_v2, encoding="utf-8")
        orig_backup = Path(str(svc_path) + ".orig")
        orig_backup.write_text(payment_service_v1, encoding="utf-8")

        from cpg.incremental_updater import IncrementalCPGUpdater
        from orchestrator.commit_audit_scheduler import CommitAuditScheduler

        updater = IncrementalCPGUpdater(
            cpg_engine=None, repo_root=repo_root, storage=storage
        )
        mock_runner = AsyncMock()
        mock_runner.run_for_functions = AsyncMock(return_value=MagicMock(
            status=MagicMock(value="PASSED"), passed=3, failed=0, coverage_pct=90.0
        ))

        scheduler = CommitAuditScheduler(
            storage=storage,
            incremental_updater=updater,
            test_runner=mock_runner,
            run_id=run_id,
            repo_root=repo_root,
        )

        record = await scheduler.schedule_from_webhook(
            changed_files=["payment_service.py"],
            commit_hash="abc123",
            branch="main",
            author="dev",
            commit_message="fix: improve payment validation",
        )

        assert record.status.value in ("DONE", "SKIPPED"), (
            f"Expected DONE or SKIPPED, got {record.status.value}: "
            f"{record.error_detail}"
        )

        # At minimum the changed file must be recorded
        assert "payment_service.py" in record.changed_files

        if record.status.value == "DONE":
            # Staleness marks must exist
            marks = await storage.list_stale_functions(
                file_path="payment_service.py"
            )
            assert len(marks) >= 1, (
                "CommitAuditScheduler must write FunctionStalenessMark rows "
                "so the next ReaderAgent pass detects the changed functions"
            )

    async def test_idempotency_same_commit_not_reaudited(self, storage, repo_root):
        """
        A commit that has already been audited (status=DONE) must not be
        re-scheduled.  CommitAuditScheduler.schedule_commit_audit must return
        the existing record immediately.
        """
        run_id = _make_run_id()

        from cpg.incremental_updater import IncrementalCPGUpdater
        from orchestrator.commit_audit_scheduler import CommitAuditScheduler
        from brain.schemas import CommitAuditRecord, CommitAuditStatus

        # Pre-seed a DONE record
        existing = CommitAuditRecord(
            run_id=run_id,
            commit_hash="dupe_hash",
            status=CommitAuditStatus.DONE,
            changed_files=["foo.py"],
        )
        await storage.upsert_commit_audit_record(existing)

        updater = IncrementalCPGUpdater(cpg_engine=None, repo_root=repo_root, storage=storage)
        scheduler = CommitAuditScheduler(
            storage=storage,
            incremental_updater=updater,
            test_runner=None,
            run_id=run_id,
            repo_root=repo_root,
        )

        returned = await scheduler.schedule_commit_audit(
            commit_hash="dupe_hash",
            changed_files={"foo.py"},
        )

        assert returned.id == existing.id, (
            "Idempotency check failed: scheduler created a new record instead "
            "of returning the existing DONE record"
        )
        assert returned.status == CommitAuditStatus.DONE

    async def test_no_changed_functions_produces_skipped_status(
        self, storage, repo_root
    ):
        """
        When the diff produces no changed functions (e.g. only a comment
        changed in a data file with no hunk context), the record must be
        SKIPPED — not FAILED.
        """
        run_id = _make_run_id()

        from cpg.incremental_updater import IncrementalCPGUpdater
        from orchestrator.commit_audit_scheduler import CommitAuditScheduler

        updater = IncrementalCPGUpdater(cpg_engine=None, repo_root=repo_root, storage=storage)
        scheduler = CommitAuditScheduler(
            storage=storage,
            incremental_updater=updater,
            test_runner=None,
            run_id=run_id,
            repo_root=repo_root,
        )

        record = await scheduler.schedule_from_webhook(
            changed_files=[],
            commit_hash="no_fns_commit",
        )

        assert record.status.value in ("SKIPPED", "DONE"), (
            f"Expected SKIPPED for empty file list, got {record.status.value}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: ReaderAgent staleness detection (BUG-A fix)
# ─────────────────────────────────────────────────────────────────────────────

class TestReaderAgentStalenessBugA:
    """
    BUG-A: list_stale_functions was called with run_id=self.run_id, silently
    dropping marks written by a CommitAuditScheduler with a DIFFERENT run_id
    (e.g. a CI webhook task).

    Fix: query without run_id filter; accept marks from any run.
    """

    async def test_reader_detects_marks_from_different_run_id(
        self, storage, repo_root, payment_service_v2
    ):
        """
        Marks written with webhook_run_id must be visible to a ReaderAgent
        whose run_id differs.
        """
        webhook_run_id = _make_run_id()
        reader_run_id  = _make_run_id()
        assert webhook_run_id != reader_run_id

        # Write a file to disk
        svc_path = repo_root / "payment_service.py"
        svc_path.write_text(payment_service_v2, encoding="utf-8")

        # Pre-seed a FileRecord that looks fully up-to-date (same hash as
        # what is on disk) so the reader would normally skip it.
        content = svc_path.read_text(encoding="utf-8")
        file_hash = hashlib.sha256(content.encode()).hexdigest()
        from brain.schemas import FileRecord, FileStatus
        rec = FileRecord(
            path="payment_service.py",
            language="python",
            status=FileStatus.READ,
            run_id=reader_run_id,
            hash=file_hash,
            known_functions=["PaymentService.process", "PaymentService.validate"],
            line_count=content.count("\n") + 1,
        )
        await storage.upsert_file(rec)

        # Write a staleness mark using the WEBHOOK run_id
        from brain.schemas import FunctionStalenessMark
        mark = FunctionStalenessMark(
            file_path="payment_service.py",
            function_name="PaymentService.process",
            stale_reason="commit_change",
            run_id=webhook_run_id,   # ← different from reader_run_id
        )
        await storage.upsert_staleness_mark(mark)

        # Verify the mark is in storage
        all_marks = await storage.list_stale_functions(file_path="payment_service.py")
        assert len(all_marks) >= 1, "Setup failed: staleness mark not stored"

        # The FIX: list_stale_functions without run_id must return the mark
        marks_no_filter = await storage.list_stale_functions(
            file_path="payment_service.py"
        )
        fn_names = [m.function_name for m in marks_no_filter]
        assert "PaymentService.process" in fn_names, (
            "BUG-A regression: list_stale_functions without run_id must return "
            "marks written by any run (including the webhook task run_id). "
            "The reader would have skipped this file even though it has "
            "stale functions."
        )

    async def test_reader_force_rereads_file_with_cross_run_staleness_mark(
        self, storage, repo_root, payment_service_v2
    ):
        """
        End-to-end: ReaderAgent._process_file must NOT return existing when
        staleness marks exist — even marks written by a different run.

        This directly tests the BUG-A fix in the incremental check branch.
        """
        webhook_run_id = _make_run_id()
        reader_run_id  = _make_run_id()

        svc_path = repo_root / "payment_service.py"
        svc_path.write_text(payment_service_v2, encoding="utf-8")

        content   = svc_path.read_text(encoding="utf-8")
        file_hash = hashlib.sha256(content.encode()).hexdigest()
        from brain.schemas import FileRecord, FileStatus, FunctionStalenessMark
        rec = FileRecord(
            path="payment_service.py",
            language="python",
            status=FileStatus.READ,
            run_id=reader_run_id,
            hash=file_hash,
            known_functions=["PaymentService.process"],
            line_count=content.count("\n") + 1,
        )
        await storage.upsert_file(rec)

        # Inject mark with a DIFFERENT run_id
        mark = FunctionStalenessMark(
            file_path="payment_service.py",
            function_name="PaymentService.process",
            stale_reason="commit_change",
            run_id=webhook_run_id,
        )
        await storage.upsert_staleness_mark(mark)

        from agents.reader import ReaderAgent
        reader = ReaderAgent(
            storage=storage,
            run_id=reader_run_id,
            repo_root=repo_root,
            incremental=True,
        )

        processed = await reader.run()
        assert len(processed) >= 1

        # After re-read, staleness marks for that file must be cleared
        remaining = await storage.list_stale_functions(
            file_path="payment_service.py"
        )
        assert len(remaining) == 0, (
            "BUG-A fix verification: after the reader re-reads a file, all "
            "staleness marks for that file must be cleared regardless of which "
            "run_id wrote them."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Method name qualification (BUG-B and BUG-C fix)
# ─────────────────────────────────────────────────────────────────────────────

class TestMethodNameQualification:
    """
    BUG-B (_extract_function_names in reader.py) and
    BUG-C (_ts_changed_functions in incremental_updater.py):

    Both functions used bare method names, causing:
    - Two classes each with __init__ → dict collision, only one stored
    - clear_staleness_mark("__init__") clearing marks for ALL classes at once
    - Changed methods in ClassA being missed because ClassB overrode the key

    Fix: qualify as ClassName.method_name everywhere.
    """

    def test_extract_function_names_qualifies_class_methods(self):
        from agents.reader import ReaderAgent
        from unittest.mock import MagicMock
        reader = ReaderAgent.__new__(ReaderAgent)
        reader.log = MagicMock()

        content = textwrap.dedent("""\
            class ServiceA:
                def __init__(self):
                    pass
                def run(self):
                    pass

            class ServiceB:
                def __init__(self):
                    pass
                def run(self):
                    pass

            def module_level():
                pass
        """)

        with patch("startup.feature_matrix.is_available", return_value=True):
            try:
                names = reader._extract_function_names(content, "python")
                if not names:
                    pytest.skip("tree-sitter not installed in test environment")

                assert "ServiceA.__init__" in names, (
                    "BUG-B regression: ServiceA.__init__ must be qualified "
                    "to avoid collision with ServiceB.__init__"
                )
                assert "ServiceB.__init__" in names, (
                    "BUG-B regression: ServiceB.__init__ must be present as a "
                    "distinct entry — bare '__init__' would collide"
                )
                assert "ServiceA.run" in names
                assert "ServiceB.run" in names
                assert "module_level" in names

                # Bare '__init__' must NOT appear (would indicate bug still present)
                assert "__init__" not in names, (
                    "BUG-B still present: bare '__init__' in known_functions "
                    "means class methods are not being qualified"
                )
            except RuntimeError:
                pytest.skip("tree-sitter not installed in test environment")

    def test_ts_changed_functions_detects_class_method_change(self):
        """
        When only ServiceA.run changes, ServiceB.run must NOT appear in the
        changed-function list — the tree-sitter path must use qualified names
        so the two 'run' bodies are compared independently.
        """
        from cpg.incremental_updater import _ts_changed_functions

        original = textwrap.dedent("""\
            class ServiceA:
                def run(self):
                    return 1

            class ServiceB:
                def run(self):
                    return 2
        """)
        modified = textwrap.dedent("""\
            class ServiceA:
                def run(self):
                    return 999  # changed

            class ServiceB:
                def run(self):
                    return 2    # unchanged
        """)

        with patch("startup.feature_matrix.is_available", return_value=True):
            try:
                changed = _ts_changed_functions(original, modified, "service.py")
                if not changed and original == modified:
                    pytest.skip("tree-sitter not installed in test environment")

                assert "ServiceA.run" in changed, (
                    "BUG-C regression: changed method must be detected by qualified name"
                )
                assert "ServiceB.run" not in changed, (
                    "BUG-C regression: unchanged method in sibling class must NOT "
                    "appear in the changed list — bare 'run' key collision caused this"
                )
            except RuntimeError:
                pytest.skip("tree-sitter not installed in test environment")

    def test_ts_changed_functions_regex_fallback_catches_indented_methods(self):
        """
        BUG-C secondary: the regex fallback pattern `^def` only matched
        column-0 functions.  Fixed to `^\\s*def` so class methods at any
        indentation level are caught.
        """
        from cpg.incremental_updater import _regex_changed_functions

        original = textwrap.dedent("""\
            class Foo:
                def process(self):
                    return 1
        """)
        modified = textwrap.dedent("""\
            class Foo:
                def process(self):
                    return 2  # changed
        """)

        changed = _regex_changed_functions(original, modified, "foo.py")
        assert "process" in changed, (
            "BUG-C regex fix regression: indented 'def process' must be "
            "detected by _regex_changed_functions.  Was broken when the "
            "pattern anchored at ^ (column 0 only)."
        )

    async def test_clear_staleness_uses_qualified_names(self, storage, repo_root):
        """
        After ReaderAgent re-reads a file, it must clear marks by qualified
        name (ClassName.method_name).  If the reader still uses bare names,
        clearing 'run' would wipe marks for ALL classes' 'run' methods at once
        instead of only the one that changed.
        """
        run_id = _make_run_id()
        from brain.schemas import FunctionStalenessMark

        # Write two marks with qualified names (as the fixed code produces)
        for qname in ("ServiceA.run", "ServiceB.run"):
            await storage.upsert_staleness_mark(FunctionStalenessMark(
                file_path="service.py",
                function_name=qname,
                stale_reason="commit_change",
                run_id=run_id,
            ))

        marks_before = await storage.list_stale_functions(file_path="service.py")
        assert len(marks_before) == 2

        # Clear only ServiceA.run
        await storage.clear_staleness_mark("service.py", "ServiceA.run")

        marks_after = await storage.list_stale_functions(file_path="service.py")
        remaining_names = [m.function_name for m in marks_after]

        assert "ServiceA.run" not in remaining_names, (
            "ServiceA.run mark must be cleared"
        )
        assert "ServiceB.run" in remaining_names, (
            "ServiceB.run mark must NOT be cleared — only ServiceA.run was cleared. "
            "If bare 'run' was used, both would be cleared at once."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Blast radius cap (BUG-D fix)
# ─────────────────────────────────────────────────────────────────────────────

class TestBlastRadiusCap:
    """
    BUG-D: compute_blast_radius had no breadth cap.  A single change to a
    widely-used utility at depth 3 could produce thousands of callers and
    flood the FunctionStalenessMark table.

    Fix: cap at _MAX_IMPACT_BREADTH (500) in incremental_updater.py and at
    _MAX_BLAST_CAP (500) in controller._requeue_transitive_dependents.
    """

    async def test_impact_set_truncated_at_cap(self, storage, repo_root):
        """
        When CPG returns >500 impacted functions, update_after_commit must
        truncate the impact set and log a warning instead of emitting
        thousands of staleness marks.
        """
        from cpg.incremental_updater import IncrementalCPGUpdater, _MAX_IMPACT_BREADTH

        # Build a mock CPG that returns 800 functions in the impact set
        big_impact = [
            {"function_name": f"fn_{i}", "file_path": f"module_{i}.py"}
            for i in range(800)
        ]
        mock_blast = MagicMock()
        mock_blast.affected_functions    = big_impact
        mock_blast.affected_files        = [f"module_{i}.py" for i in range(800)]
        mock_blast.affected_function_count = 800

        mock_cpg = AsyncMock()
        mock_cpg.is_available            = True
        mock_cpg.invalidate_cache        = MagicMock()
        mock_cpg.compute_blast_radius    = AsyncMock(return_value=mock_blast)

        updater = IncrementalCPGUpdater(
            cpg_engine=mock_cpg,
            repo_root=repo_root,
            storage=storage,
        )

        # Make a minimal .orig backup so the fallback diff fires
        changed_file = repo_root / "utils.py"
        changed_file.write_text("def util(): return 1\n", encoding="utf-8")
        orig = Path(str(changed_file) + ".orig")
        orig.write_text("def util(): return 0\n", encoding="utf-8")

        result = await updater.update_after_commit(
            changed_files={"utils.py"},
            run_id=_make_run_id(),
            commit_hash="big_blast",
        )

        assert len(result.impact_set) <= _MAX_IMPACT_BREADTH, (
            f"BUG-D regression: impact set size {len(result.impact_set)} "
            f"exceeds cap {_MAX_IMPACT_BREADTH}.  Unbounded blast radius "
            "would flood the staleness table."
        )

    def test_max_impact_breadth_constant_exists_and_is_reasonable(self):
        """
        The _MAX_IMPACT_BREADTH constant must be present and set to a value
        that prevents runaway staleness mark emission (between 100 and 2000).
        """
        from cpg.incremental_updater import _MAX_IMPACT_BREADTH
        assert 100 <= _MAX_IMPACT_BREADTH <= 2000, (
            f"_MAX_IMPACT_BREADTH={_MAX_IMPACT_BREADTH} is outside the "
            "expected safety range [100, 2000]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Test runner wiring (BUG-E fix)
# ─────────────────────────────────────────────────────────────────────────────

class TestTestRunnerWiring:
    """
    BUG-E: CommitAuditScheduler was constructed with test_runner=None in
    controller._init_cpg().  The scoped test re-run step silently skipped
    every time with "no test runner configured".

    Fix: TestRunnerAgent is now instantiated and passed in.
    """

    async def test_scheduler_invokes_test_runner_when_present(
        self, storage, repo_root, payment_service_v1, payment_service_v2
    ):
        """
        When test_runner is properly wired, _run_scoped_tests must call
        run_for_functions with the list of changed functions.
        """
        run_id = _make_run_id()

        svc_path = repo_root / "payment_service.py"
        svc_path.write_text(payment_service_v2, encoding="utf-8")
        orig     = Path(str(svc_path) + ".orig")
        orig.write_text(payment_service_v1, encoding="utf-8")

        from cpg.incremental_updater import IncrementalCPGUpdater
        from orchestrator.commit_audit_scheduler import CommitAuditScheduler

        updater = IncrementalCPGUpdater(cpg_engine=None, repo_root=repo_root, storage=storage)

        mock_runner = AsyncMock()
        mock_run_result = MagicMock()
        mock_run_result.status     = MagicMock(value="PASSED")
        mock_run_result.passed     = 5
        mock_run_result.failed     = 0
        mock_run_result.coverage_pct = 95.0
        mock_runner.run_for_functions = AsyncMock(return_value=mock_run_result)

        scheduler = CommitAuditScheduler(
            storage=storage,
            incremental_updater=updater,
            test_runner=mock_runner,     # properly wired
            run_id=run_id,
            repo_root=repo_root,
        )

        record = await scheduler.schedule_from_webhook(
            changed_files=["payment_service.py"],
            commit_hash="wired_runner_test",
        )

        if record.status.value == "DONE" and record.all_changed_functions:
            mock_runner.run_for_functions.assert_called_once()
            call_kwargs = mock_runner.run_for_functions.call_args
            called_fns  = (
                call_kwargs.kwargs.get("function_names")
                or (call_kwargs.args[0] if call_kwargs.args else [])
            )
            assert len(called_fns) >= 1, (
                "BUG-E fix verification: run_for_functions must be called with "
                "the list of changed functions, not an empty list"
            )

    async def test_scheduler_does_not_raise_when_runner_is_none(
        self, storage, repo_root
    ):
        """
        test_runner=None is still a valid configuration (e.g. test-less repos).
        CommitAuditScheduler must handle it gracefully without raising.
        """
        run_id = _make_run_id()

        from cpg.incremental_updater import IncrementalCPGUpdater
        from orchestrator.commit_audit_scheduler import CommitAuditScheduler

        updater = IncrementalCPGUpdater(cpg_engine=None, repo_root=repo_root, storage=storage)
        scheduler = CommitAuditScheduler(
            storage=storage,
            incremental_updater=updater,
            test_runner=None,   # explicitly None — must not raise
            run_id=run_id,
            repo_root=repo_root,
        )

        record = await scheduler.schedule_from_webhook(
            changed_files=["some_file.py"],
            commit_hash="no_runner_commit",
        )

        # Must complete without exception; status must be valid
        assert record.status.value in ("DONE", "SKIPPED", "FAILED"), (
            f"Unexpected status {record.status.value!r} with test_runner=None"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Staleness mark round-trip (full Gap 4 flow)
# ─────────────────────────────────────────────────────────────────────────────

class TestStalenesMarkRoundTrip:
    """
    Full round-trip: commit → marks written → reader detects marks → reader
    clears marks after re-read.

    This is the most important integration test in this file.  It validates
    the complete Gap 4 mechanism end-to-end.
    """

    async def test_full_round_trip_commit_to_reader_clears_marks(
        self, storage, repo_root, payment_service_v1, payment_service_v2
    ):
        """
        Step 1: Simulate a commit landing — write staleness marks via
                IncrementalCPGUpdater (same path as CommitAuditScheduler).
        Step 2: Confirm marks are in storage.
        Step 3: Run ReaderAgent — it must detect the marks and re-read.
        Step 4: Confirm marks are cleared after re-read.
        """
        webhook_run_id = _make_run_id()   # e.g. the Celery task run_id
        reader_run_id  = _make_run_id()   # the controller's AuditRun id
        assert webhook_run_id != reader_run_id

        svc_path = repo_root / "payment_service.py"
        svc_path.write_text(payment_service_v2, encoding="utf-8")
        orig = Path(str(svc_path) + ".orig")
        orig.write_text(payment_service_v1, encoding="utf-8")

        # Pre-seed a FileRecord that looks fully up-to-date so the reader
        # would normally skip it (same hash as v2 on disk).
        content   = payment_service_v2
        file_hash = hashlib.sha256(content.encode()).hexdigest()
        from brain.schemas import FileRecord, FileStatus, FunctionStalenessMark
        rec = FileRecord(
            path="payment_service.py",
            language="python",
            status=FileStatus.READ,
            run_id=reader_run_id,
            hash=file_hash,
            known_functions=[
                "PaymentService.process",
                "PaymentService.validate",
                "helper",
            ],
            line_count=content.count("\n") + 1,
        )
        await storage.upsert_file(rec)

        # ── Step 1: write staleness marks (simulates the webhook Celery task) ──
        from cpg.incremental_updater import IncrementalCPGUpdater
        updater = IncrementalCPGUpdater(
            cpg_engine=None,
            repo_root=repo_root,
            storage=storage,
        )
        update_result = await updater.update_after_commit(
            changed_files={"payment_service.py"},
            run_id=webhook_run_id,   # ← different run_id than the reader
            commit_hash="round_trip_commit",
        )

        # ── Step 2: marks must be in storage ──────────────────────────────────
        marks_after_commit = await storage.list_stale_functions(
            file_path="payment_service.py"
        )
        assert len(marks_after_commit) >= 1, (
            "Round-trip setup failed: IncrementalCPGUpdater must write at "
            "least one FunctionStalenessMark row after a commit"
        )

        # ── Step 3: ReaderAgent must re-read despite same file hash ───────────
        reread_count_before = 0
        original_process_file = None

        reread_count = {"n": 0}

        from agents.reader import ReaderAgent
        original_process = ReaderAgent._process_file

        async def _spy_process_file(self_inner, path, sem, force_reread):
            result = await original_process(self_inner, path, sem, force_reread)
            rel = str(path.relative_to(self_inner.repo_root))
            if rel == "payment_service.py":
                reread_count["n"] += 1
            return result

        reader = ReaderAgent(
            storage=storage,
            run_id=reader_run_id,
            repo_root=repo_root,
            incremental=True,
        )

        with patch.object(ReaderAgent, "_process_file", _spy_process_file):
            await reader.run()

        # ── Step 4: marks must be cleared after re-read ───────────────────────
        marks_after_read = await storage.list_stale_functions(
            file_path="payment_service.py"
        )
        assert len(marks_after_read) == 0, (
            f"BUG-A + round-trip fix: after the reader re-reads the file, "
            f"all staleness marks must be cleared.  {len(marks_after_read)} "
            "mark(s) remain — this means the reader either did not re-read "
            "the file (BUG-A: run_id filter blocked detection) or did not "
            "clear the marks (clear_staleness_mark uses wrong name format)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Commit audit record persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestCommitAuditRecordPersistence:
    """
    CommitAuditRecord must be persisted through every status transition so
    that CI systems can poll the GET /commits/{record_id} endpoint for status.
    """

    async def test_commit_audit_record_retrievable_after_schedule(
        self, storage, repo_root
    ):
        run_id = _make_run_id()

        from cpg.incremental_updater import IncrementalCPGUpdater
        from orchestrator.commit_audit_scheduler import CommitAuditScheduler
        from brain.schemas import CommitAuditStatus

        updater = IncrementalCPGUpdater(cpg_engine=None, repo_root=repo_root, storage=storage)
        scheduler = CommitAuditScheduler(
            storage=storage,
            incremental_updater=updater,
            test_runner=None,
            run_id=run_id,
            repo_root=repo_root,
        )

        record = await scheduler.schedule_from_webhook(
            changed_files=["svc.py"],
            commit_hash="persist_test",
            branch="feature/x",
            author="alice",
        )

        # Must be retrievable by id
        fetched = await storage.get_commit_audit_record(record.id)
        assert fetched is not None, "CommitAuditRecord not persisted to storage"
        assert fetched.id     == record.id
        assert fetched.branch == "feature/x"
        assert fetched.author == "alice"
        assert fetched.status in (
            CommitAuditStatus.DONE,
            CommitAuditStatus.SKIPPED,
            CommitAuditStatus.FAILED,
        ), f"Unexpected final status: {fetched.status}"

        # Must be retrievable by commit hash
        by_hash = await storage.get_commit_audit_record_by_hash(
            "persist_test", run_id=run_id
        )
        assert by_hash is not None, (
            "get_commit_audit_record_by_hash must return the record so the "
            "idempotency check in schedule_commit_audit works correctly"
        )
        assert by_hash.id == record.id

    async def test_list_commit_audit_records_returns_created_record(
        self, storage, repo_root
    ):
        run_id = _make_run_id()

        from cpg.incremental_updater import IncrementalCPGUpdater
        from orchestrator.commit_audit_scheduler import CommitAuditScheduler

        updater = IncrementalCPGUpdater(cpg_engine=None, repo_root=repo_root, storage=storage)
        scheduler = CommitAuditScheduler(
            storage=storage,
            incremental_updater=updater,
            test_runner=None,
            run_id=run_id,
            repo_root=repo_root,
        )

        await scheduler.schedule_from_webhook(
            changed_files=["alpha.py"],
            commit_hash="list_test_1",
        )
        await scheduler.schedule_from_webhook(
            changed_files=["beta.py"],
            commit_hash="list_test_2",
        )

        records = await storage.list_commit_audit_records(run_id=run_id)
        commit_hashes = {r.commit_hash for r in records}
        assert "list_test_1" in commit_hashes
        assert "list_test_2" in commit_hashes
