"""
tests/integration/test_pipeline.py
====================================
TEST-01 FIX: End-to-end integration test for the Rhodawk stabilization pipeline.

Previously the entire test suite was unit tests where every pipeline call
mocked either the LLM layer (patch("litellm.acompletion")) or Docker. No test
ever ran the actual pipeline against a real repository. A skeptical CTO asking
"does this have integration tests?" would learn the answer was no.

This test runs the full pipeline against a known small open-source Python repo
(urllib3 at a pinned commit, ~15K lines) with:
  - Real SQLite storage (no mocks)
  - Real LLM calls (requires ANTHROPIC_API_KEY or OPENROUTER_API_KEY)
  - CPG disabled (no Joern needed)
  - BoBN disabled (single-model fixer, faster)
  - max_cycles=2 (enough to find at least one issue and attempt a fix)

The test is marked @pytest.mark.integration and @pytest.mark.slow so it is
excluded from default CI runs (which run unit tests only) and triggered
explicitly via:
    pytest -m integration --timeout=300

To run in CI weekly (GitHub Actions example):
    on:
      schedule:
        - cron: "0 3 * * 0"  # Every Sunday 03:00 UTC
    jobs:
      integration:
        steps:
          - run: pytest -m integration --timeout=300

Environment variables required:
    ANTHROPIC_API_KEY  or  OPENROUTER_API_KEY  (at least one)

Environment variables optional:
    RHODAWK_INTEGRATION_REPO_URL    (default: the urllib3 test repo below)
    RHODAWK_INTEGRATION_CYCLES      (default: 2)
    RHODAWK_INTEGRATION_TIMEOUT     (default: 240 seconds)
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

# ── Markers ───────────────────────────────────────────────────────────────────

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
]

# ── Constants ─────────────────────────────────────────────────────────────────

# A small, well-known, pure-Python library pinned to a specific commit so the
# test is deterministic. urllib3 at this commit has ~15K lines of source and
# known lint findings that Rhodawk reliably detects.
_DEFAULT_REPO_URL = "https://github.com/urllib3/urllib3"
_DEFAULT_REPO_COMMIT = "2.2.1"   # tag — small and stable

_REPO_URL     = os.environ.get("RHODAWK_INTEGRATION_REPO_URL",  _DEFAULT_REPO_URL)
_MAX_CYCLES   = int(os.environ.get("RHODAWK_INTEGRATION_CYCLES",  "2"))
_TIMEOUT_S    = int(os.environ.get("RHODAWK_INTEGRATION_TIMEOUT", "240"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _has_llm_key() -> bool:
    """Return True if at least one cloud LLM key is configured."""
    for var in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        val = os.environ.get(var, "")
        if val and not val.startswith("sk-ant-...") and not val.startswith("sk-or-..."):
            return True
    return False


def _clone_repo(dest: Path) -> None:
    """Shallow-clone the target repo at the pinned tag/commit."""
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "--branch", _DEFAULT_REPO_COMMIT,
         _REPO_URL, str(dest)],
        check=True,
        capture_output=True,
        timeout=120,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def repo_root(tmp_path_factory) -> Path:
    """
    Clone the target repository once for the entire integration module.
    Cached at module scope so multiple test functions share the same checkout.
    """
    dest = tmp_path_factory.mktemp("integration_repo")
    try:
        _clone_repo(dest)
    except subprocess.CalledProcessError as exc:
        pytest.skip(
            f"Could not clone {_REPO_URL}: {exc.stderr.decode()[:200]}. "
            "Skipping integration tests (network unavailable)."
        )
    return dest


@pytest.fixture(scope="module")
def db_path(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("integration_db") / "brain.db"


# ── Skip conditions ───────────────────────────────────────────────────────────

def _skip_if_no_llm():
    if not _has_llm_key():
        pytest.skip(
            "No LLM API key configured. Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY "
            "to run integration tests."
        )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFullPipeline:
    """
    End-to-end pipeline tests.

    Each test method builds on the shared repo_root and db_path fixtures so
    the controller only needs to initialise once (the most expensive step).
    """

    @pytest.fixture(scope="class", autouse=True)
    def run_pipeline(self, repo_root: Path, db_path: Path):
        """
        Run StabilizerController.stabilize() once for the whole test class.

        Stores the controller on self so individual test methods can inspect
        its state without re-running the pipeline.
        """
        _skip_if_no_llm()

        from orchestrator.controller import StabilizerConfig, StabilizerController
        from brain.schemas import DomainMode

        config = StabilizerConfig(
            repo_url         = _REPO_URL,
            repo_root        = repo_root,
            domain_mode      = DomainMode.GENERAL,
            max_cycles       = _MAX_CYCLES,
            cost_ceiling_usd = 10.0,      # hard cap for test safety
            # Disable heavy subsystems so the test runs on any CI machine:
            cpg_enabled              = False,
            gap5_bobn_enabled        = False,
            gap5_enabled             = False,
        )

        controller = StabilizerController(config)

        async def _run():
            await controller.initialise()
            await controller.stabilize()

        try:
            asyncio.get_event_loop().run_until_complete(
                asyncio.wait_for(_run(), timeout=_TIMEOUT_S)
            )
        except asyncio.TimeoutError:
            pytest.fail(
                f"Pipeline did not complete within {_TIMEOUT_S}s. "
                "Increase RHODAWK_INTEGRATION_TIMEOUT or reduce the repo size."
            )

        # Attach to the class instance so test methods can inspect it.
        self.__class__._controller = controller
        yield controller

    # ── Assertions ────────────────────────────────────────────────────────────

    def test_run_was_created(self):
        """The controller must produce a persisted AuditRun record."""
        ctrl = self.__class__._controller
        assert ctrl.run is not None, "controller.run is None — initialise() may have failed"
        assert ctrl.run.id, "run.id is empty"

    @pytest.mark.asyncio
    async def test_at_least_one_issue_found(self):
        """
        A real codebase must produce at least one finding after auditing.

        urllib3 is a mature library but Rhodawk's auditor reliably finds
        minor style and type annotation issues even in well-maintained code.
        Zero findings indicates the auditor never ran or the LLM returned empty.
        """
        ctrl = self.__class__._controller
        storage = ctrl._storage
        issues = await storage.list_issues(run_id=ctrl.run.id)
        assert len(issues) > 0, (
            f"No issues found after {_MAX_CYCLES} cycles on {_REPO_URL}. "
            "Either the auditor never ran, the LLM returned empty findings on "
            "every chunk, or the repo has been updated and no longer has detectable issues."
        )

    @pytest.mark.asyncio
    async def test_audit_trail_has_entries(self):
        """
        The DO-178C audit trail must be written during a real run.

        An empty audit trail means AuditorAgent.run() never called
        storage.append_audit_trail(), which would silently void all
        compliance claims.
        """
        ctrl = self.__class__._controller
        storage = ctrl._storage
        trail = await storage.get_audit_trail(ctrl.run.id)
        assert len(trail) > 0, (
            "Audit trail is empty after a full pipeline run. "
            "append_audit_trail() may not be wired into AuditorAgent."
        )

    @pytest.mark.asyncio
    async def test_scores_recorded(self):
        """
        AuditScore records must be written at least once per cycle.

        Missing scores means the convergence detector never fired, which
        means the loop logic in controller._run_deerflow() is broken.
        """
        ctrl = self.__class__._controller
        storage = ctrl._storage
        scores = await storage.get_scores(ctrl.run.id)
        assert len(scores) > 0, (
            "No AuditScore records found. The convergence detector may not "
            "be wired or append_score() is not being called."
        )

    @pytest.mark.asyncio
    async def test_run_completes_with_valid_status(self):
        """
        The run must reach a terminal status (not stay RUNNING forever).

        If the run is still RUNNING after stabilize() returns, the controller's
        cleanup path (update_run_status on normal exit) is broken.
        """
        from brain.schemas import RunStatus
        ctrl = self.__class__._controller
        storage = ctrl._storage
        run = await storage.get_run(ctrl.run.id)
        assert run is not None, "Run record not found in storage after pipeline completion"
        terminal = {
            RunStatus.STABILIZED,
            RunStatus.HALTED,
            RunStatus.FAILED,
            RunStatus.ESCALATED,
            RunStatus.BASELINE_PENDING,
            RunStatus.MAX_CYCLES_REACHED,
        }
        assert run.status in terminal, (
            f"Run status is {run.status.value} after stabilize() returned. "
            "Expected a terminal status. The controller may not be calling "
            "update_run_status() on exit."
        )

    @pytest.mark.asyncio
    async def test_fix_attempted_if_issues_found(self):
        """
        If issues were found, at least one fix attempt must be recorded.

        This verifies the audit → fix pipeline is wired end-to-end:
        AuditorAgent finds issues → FixerAgent generates patches →
        upsert_fix() persists them → list_fixes() returns them.
        """
        ctrl = self.__class__._controller
        storage = ctrl._storage
        issues = await storage.list_issues(run_id=ctrl.run.id)
        if not issues:
            pytest.skip("No issues found — fix pipeline not exercised")

        fixes = await storage.list_fixes(run_id=ctrl.run.id)
        assert len(fixes) > 0, (
            f"{len(issues)} issue(s) found but zero fix attempts recorded. "
            "FixerAgent may not be running or upsert_fix() is broken."
        )

    @pytest.mark.asyncio
    async def test_no_crashed_issues_in_storage(self):
        """
        All persisted issues must be deserializable without AttributeError.

        This catches schema mismatches like the BLOCK-04 bug where
        fix_attempt_count was written but fix_attempts was expected,
        crashing every call to upsert_issue().
        """
        ctrl = self.__class__._controller
        storage = ctrl._storage
        # list_issues() deserializes every row through _row_to_issue().
        # If any row fails, this will raise rather than silently skip.
        try:
            issues = await storage.list_issues(run_id=ctrl.run.id)
        except AttributeError as exc:
            pytest.fail(
                f"AttributeError deserializing issues — likely a schema field "
                f"name mismatch (BLOCK-04 class of bug): {exc}"
            )
        # Spot-check a few fields on the first issue to ensure full round-trip.
        if issues:
            issue = issues[0]
            assert issue.id
            assert issue.file_path
            assert issue.severity is not None
            assert issue.status is not None
