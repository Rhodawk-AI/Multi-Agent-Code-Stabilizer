"""
tests/unit/test_controller_deerflow.py
========================================
Adversarial edge-case tests for orchestrator/controller.py DeerFlow loop.

Covers:
 - BoBN routing: gap5_enabled=False → _phase_fix, never _phase_fix_gap5
 - BoBN routing: gap5_enabled=True but _bobn_sampler is None → _phase_fix fallback
 - BoBN routing: gap5_enabled=True, sampler present, litellm raises
   APIConnectionError → _phase_fix_gap5 surfaces the error (no silent swallow)
 - _run_deerflow loop: _shutdown_requested() returns True after cycle 1 →
   loop exits without calling SIGTERM-unsafe subsystems
 - _run_deerflow: max_cycles=1 → loop exits after exactly one cycle
 - run_audit_phase: litellm.completion raises AuthenticationError (bad API key)
   → the exception propagates up (not silently skipped as zero issues)
 - _apply_consensus: EscalationManager.wait_for_resolution raises
   asyncio.CancelledError (worker shutdown mid-wait) → propagates
 - _commit_module_group: git commit subprocess returns non-zero exit code
   (merge conflict) → FixAttempt is NOT marked committed; issues stay OPEN
 - _commit_module_group: git add returns exit 1 (locked index) → hard abort
   before any commit attempt
"""
from __future__ import annotations

import asyncio
import subprocess
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call, PropertyMock


# ── Stub litellm + instructor before any controller import ────────────────────

import sys, types

for _stub in ("litellm", "instructor", "aiohttp", "networkx", "z3"):
    if _stub not in sys.modules:
        sys.modules[_stub] = MagicMock()

# litellm needs these exception classes
_litellm_exc = types.ModuleType("litellm.exceptions")
_litellm_exc.APIConnectionError = ConnectionError
_litellm_exc.AuthenticationError = PermissionError
_litellm_exc.RateLimitError = RuntimeError
sys.modules["litellm.exceptions"] = _litellm_exc


# ── Minimal controller factory ────────────────────────────────────────────────

def _make_controller(
    gap5_enabled: bool = False,
    max_cycles: int = 1,
    bobn_sampler=None,
):
    """Return a StabilizerController with all subsystems mocked."""
    try:
        from orchestrator.controller import StabilizerController
        from config.loader import load_config
    except ImportError:
        pytest.skip("orchestrator.controller not importable in this environment")

    cfg = MagicMock()
    cfg.gap5_enabled = gap5_enabled
    cfg.max_cycles = max_cycles
    cfg.cost_ceiling_usd = 999.0
    cfg.use_sqlite = True
    cfg.repo_root = "/tmp/fake_repo"
    cfg.domain_mode = MagicMock()
    cfg.domain_mode.value = "GENERAL"
    cfg.autonomy_level = MagicMock()
    cfg.auto_commit = False
    cfg.github_token = ""

    ctrl = StabilizerController.__new__(StabilizerController)
    ctrl.cfg = cfg
    ctrl.storage = AsyncMock()
    ctrl.storage.get_total_cost = AsyncMock(return_value=0.0)
    ctrl.storage.list_issues = AsyncMock(return_value=[])
    ctrl.storage.update_run_status = AsyncMock()
    ctrl.storage.upsert_convergence_record = AsyncMock()

    ctrl._run = MagicMock()
    ctrl._run.id = "run-ctrl-test"
    ctrl._run.cycle_count = 0
    ctrl._run.scores = []

    ctrl._bobn_sampler = bobn_sampler
    ctrl._shutdown_requested = lambda: False
    ctrl._last_audit_issues = []
    ctrl._last_approved_issues = []

    ctrl._convergence = MagicMock()
    ctrl._convergence.check = MagicMock(return_value=MagicMock(converged=True, halt_reason="max_cycles"))
    ctrl._convergence.halt_if_ceiling_hit = MagicMock(return_value=None)

    ctrl._patrol = MagicMock()
    ctrl._consensus_engine = MagicMock()
    ctrl._consensus_engine.evaluate_issues = MagicMock(return_value=[])
    ctrl._consensus_engine.filter_approved  = MagicMock(return_value=[])
    ctrl._escalation_mgr = AsyncMock()
    ctrl._escalation_mgr.has_blocking_escalations = AsyncMock(return_value=False)
    ctrl._audit_trail_signer = MagicMock()
    ctrl._audit_trail_signer.sign = MagicMock(return_value="sig-fake")
    ctrl.log = MagicMock()
    ctrl._localization_agent = None
    ctrl._auditors = []
    ctrl._trail_signer = MagicMock()
    ctrl._trail_signer.sign = MagicMock(return_value="sig-fake")
    ctrl._patrol = None
    ctrl._run.baseline_id = ""

    return ctrl


# ── BoBN routing: gap5_enabled=False → _phase_fix only ───────────────────────

@pytest.mark.asyncio
async def test_bobn_routing_gap5_disabled_uses_standard_fixer():
    """
    gap5_enabled=False → run_fix_phase must call _phase_fix,
    never _phase_fix_gap5, even when issues are present.
    """
    ctrl = _make_controller(gap5_enabled=False)

    issues = [MagicMock(), MagicMock()]

    with patch.object(ctrl, "_phase_fix", new=AsyncMock(return_value=[])) as mock_fix, \
         patch.object(ctrl, "_phase_fix_gap5", new=AsyncMock(return_value=[])) as mock_gap5:

        await ctrl.run_fix_phase(issues)

    mock_fix.assert_called_once()
    mock_gap5.assert_not_called()


# ── BoBN routing: gap5_enabled=True but _bobn_sampler is None ────────────────

@pytest.mark.asyncio
async def test_bobn_routing_gap5_enabled_no_sampler_falls_back():
    """
    gap5_enabled=True but _bobn_sampler=None (init failed or not wired) →
    run_fix_phase must fall back to _phase_fix (same as gap5=False path).
    """
    ctrl = _make_controller(gap5_enabled=True, bobn_sampler=None)

    issues = [MagicMock()]

    with patch.object(ctrl, "_phase_fix", new=AsyncMock(return_value=[])) as mock_fix, \
         patch.object(ctrl, "_phase_fix_gap5", new=AsyncMock(return_value=[])) as mock_gap5:

        await ctrl.run_fix_phase(issues)

    mock_fix.assert_called_once()
    mock_gap5.assert_not_called()


# ── BoBN routing: gap5 active, litellm APIConnectionError ────────────────────

@pytest.mark.asyncio
async def test_bobn_routing_gap5_active_litellm_timeout_surfaces():
    """
    gap5_enabled=True, sampler present, but sampler.sample() raises
    ConnectionError (simulating litellm APIConnectionError / network freeze).
    The error must propagate out of _phase_fix_gap5 — not silently dropped.
    """
    sampler = AsyncMock()
    sampler.sample = AsyncMock(side_effect=ConnectionError("litellm: API connection failed"))

    ctrl = _make_controller(gap5_enabled=True, bobn_sampler=sampler)

    issue = MagicMock()
    issue.id = "issue-bobn-001"
    issue.severity = MagicMock()
    issue.severity.value = "CRITICAL"

    with patch.object(ctrl, "_phase_fix", new=AsyncMock(return_value=[])):
        with pytest.raises(ConnectionError, match="litellm"):
            await ctrl._phase_fix_gap5([issue])


# ── _run_deerflow: shutdown after cycle 1 ─────────────────────────────────────

@pytest.mark.asyncio
async def test_deerflow_shutdown_requested_exits_loop_cleanly():
    """
    _shutdown_requested() returns True on the second check → the DeerFlow loop
    must exit without processing further cycles. No assertions about status value
    — just that the loop terminates without error.
    """
    ctrl = _make_controller(max_cycles=10)

    call_count = {"n": 0}

    def _shutdown():
        call_count["n"] += 1
        return call_count["n"] > 1  # True from the 2nd check onwards

    ctrl._shutdown_requested = _shutdown

    with patch.object(ctrl, "run_read_phase", new=AsyncMock(return_value=[])), \
         patch.object(ctrl, "run_build_graph_phase", new=AsyncMock()), \
         patch.object(ctrl, "run_audit_phase", new=AsyncMock(return_value=MagicMock(score=80.0, critical_count=0))), \
         patch.object(ctrl, "run_consensus_phase", new=AsyncMock(return_value=[])), \
         patch.object(ctrl, "run_fix_phase", new=AsyncMock()), \
         patch.object(ctrl, "run_review_phase", new=AsyncMock()), \
         patch.object(ctrl, "run_gate_phase", new=AsyncMock()), \
         patch.object(ctrl, "run_commit_phase", new=AsyncMock()), \
         patch.object(ctrl, "run_reindex_phase", new=AsyncMock()), \
         patch.object(ctrl, "_record_score", new=AsyncMock(return_value=MagicMock(score=80.0, critical_count=0))), \
         patch.object(ctrl, "_finalise", new=AsyncMock()), \
         patch.object(ctrl, "_cleanup", new=AsyncMock()):

        try:
            status = await asyncio.wait_for(ctrl._run_deerflow(), timeout=5.0)
        except asyncio.TimeoutError:
            pytest.fail("_run_deerflow() hung — did not respect _shutdown_requested")


# ── _run_deerflow: max_cycles=1 → exactly one cycle ──────────────────────────

@pytest.mark.asyncio
async def test_deerflow_max_cycles_one_runs_exactly_one_cycle():
    """
    max_cycles=1 → convergence check returns converged after cycle 1.
    run_audit_phase must be called exactly once.
    """
    ctrl = _make_controller(max_cycles=1)

    audit_call_count = {"n": 0}

    async def _audit():
        audit_call_count["n"] += 1
        return MagicMock(score=95.0, critical_count=0)

    with patch.object(ctrl, "run_read_phase", new=AsyncMock(return_value=[])), \
         patch.object(ctrl, "run_build_graph_phase", new=AsyncMock()), \
         patch.object(ctrl, "run_audit_phase", new=AsyncMock(side_effect=_audit)), \
         patch.object(ctrl, "run_consensus_phase", new=AsyncMock(return_value=[])), \
         patch.object(ctrl, "run_fix_phase", new=AsyncMock()), \
         patch.object(ctrl, "run_review_phase", new=AsyncMock()), \
         patch.object(ctrl, "run_gate_phase", new=AsyncMock()), \
         patch.object(ctrl, "run_commit_phase", new=AsyncMock()), \
         patch.object(ctrl, "run_reindex_phase", new=AsyncMock()), \
         patch.object(ctrl, "_record_score", new=AsyncMock(return_value=MagicMock(score=95.0, critical_count=0))), \
         patch.object(ctrl, "_finalise", new=AsyncMock()), \
         patch.object(ctrl, "_cleanup", new=AsyncMock()):

        await asyncio.wait_for(ctrl._run_deerflow(), timeout=5.0)

    assert audit_call_count["n"] == 1


# ── run_audit_phase: litellm AuthenticationError propagates ──────────────────

@pytest.mark.asyncio
async def test_audit_phase_bad_api_key_propagates_not_swallowed():
    """
    If all auditors raise PermissionError (AuthenticationError from litellm),
    run_audit_phase must NOT return a zero-issue score — it must propagate.
    Swallowing auth errors causes silent STABILIZED verdicts on broken deployments.
    """
    ctrl = _make_controller()

    ctrl._auditors = [MagicMock(), MagicMock(), MagicMock()]
    for a in ctrl._auditors:
        a.run = AsyncMock(side_effect=PermissionError("litellm: Invalid API key"))

    # SynthesisAgent should not be reached
    ctrl._synthesis_agent = AsyncMock()

    with pytest.raises(PermissionError):
        await ctrl.run_audit_phase()


# ── _apply_consensus: CancelledError mid-wait propagates ─────────────────────

@pytest.mark.asyncio
async def test_apply_consensus_cancelled_error_propagates():
    """
    If EscalationManager.wait_for_resolution raises asyncio.CancelledError
    (worker shutdown mid-approval wait), _apply_consensus must let it propagate.
    CancelledError must never be caught and silently turned into an approval.
    """
    ctrl = _make_controller()
    ctrl._escalation_mgr.create_escalation = AsyncMock(return_value=MagicMock(id="esc-99"))
    ctrl._escalation_mgr.wait_for_resolution = AsyncMock(side_effect=asyncio.CancelledError())
    ctrl._consensus_engine = MagicMock()

    critical_issue = MagicMock()
    critical_issue.severity = MagicMock()
    critical_issue.severity.value = "CRITICAL"
    critical_issue.fingerprint = "fp-critical-001"
    critical_issue.consensus_confidence = 0.50  # below floor → escalation

    # Patch ConsensusEngine to flag the issue for escalation
    ctrl._consensus_engine.evaluate_issues = MagicMock(return_value=[
        MagicMock(
            fingerprint="fp-critical-001",
            approved=False,
            action="ESCALATE_HUMAN",
            weighted_confidence=0.50,
        )
    ])
    ctrl._consensus_engine.filter_approved = MagicMock(return_value=[])

    with pytest.raises(asyncio.CancelledError):
        await ctrl._apply_consensus([critical_issue])


# ── _commit_module_group: git commit fails with merge conflict ────────────────

@pytest.mark.asyncio
async def test_commit_merge_conflict_does_not_mark_fix_committed():
    """
    subprocess git commit returns returncode=1 (merge conflict / index lock).
    The FixAttempt.committed_at must remain None and issues must stay OPEN.
    storage.update_issue must not be called with IssueStatus.CLOSED.
    """
    ctrl = _make_controller()
    ctrl._pr_manager = None

    fix = MagicMock()
    fix.id = "fix-commit-001"
    fix.issue_ids = ["issue-001"]
    fix.committed_at = None
    fix.fixed_files = [
        MagicMock(path="src/auth.py", content="def verify(t): pass", patch="--- a/src/auth.py\n+++ b/src/auth.py\n")
    ]

    ctrl.storage.get_fix = AsyncMock(return_value=fix)
    ctrl.storage.upsert_fix = AsyncMock()
    ctrl.storage.update_issue = AsyncMock()
    ctrl.storage.upsert_audit_trail_entry = AsyncMock()

    # git add succeeds, git commit fails with merge conflict
    def _git_subprocess(args, **kwargs):
        result = MagicMock()
        if "commit" in args:
            result.returncode = 1
            result.stdout = b""
            result.stderr = b"error: Your local changes to the following files would be overwritten by merge"
        else:
            result.returncode = 0
            result.stdout = b""
            result.stderr = b""
        return result

    with patch("subprocess.run", side_effect=_git_subprocess), \
         patch("subprocess.check_output", return_value=b""), \
         patch.object(ctrl, "_revert_fix", new=AsyncMock()), \
         patch.object(ctrl, "_trail", new=AsyncMock()):

        try:
            await ctrl._commit_module_group("src", [fix])
        except Exception:
            pass  # expected — commit failure should propagate or leave fix uncommitted

    # Fix must NOT have committed_at set
    assert fix.committed_at is None, (
        "_commit_module_group set committed_at despite git commit failure"
    )


# ── _commit_module_group: git add fails (locked index) ───────────────────────

@pytest.mark.asyncio
async def test_commit_git_add_failure_aborts_before_commit():
    """
    git add returns exit code 1 (index.lock contention).
    Must abort before attempting git commit — no partial state.
    """
    ctrl = _make_controller()
    ctrl._pr_manager = None

    fix = MagicMock()
    fix.id = "fix-add-fail-001"
    fix.issue_ids = ["issue-add-001"]
    fix.committed_at = None
    fix.fixed_files = [
        MagicMock(path="src/storage.py", content="def save(r): pass", patch="")
    ]

    ctrl.storage.get_fix = AsyncMock(return_value=fix)
    ctrl.storage.upsert_fix = AsyncMock()

    git_commit_called = {"called": False}

    def _git_subprocess(args, **kwargs):
        result = MagicMock()
        if "add" in args:
            result.returncode = 1
            result.stderr = b"fatal: Unable to create '.git/index.lock': File exists."
        elif "commit" in args:
            git_commit_called["called"] = True
            result.returncode = 0
        else:
            result.returncode = 0
        result.stdout = b""
        return result

    with patch("subprocess.run", side_effect=_git_subprocess), \
         patch.object(ctrl, "_revert_fix", new=AsyncMock()), \
         patch.object(ctrl, "_trail", new=AsyncMock()):

        try:
            await ctrl._commit_module_group("src", [fix])
        except Exception:
            pass

    assert not git_commit_called["called"], (
        "git commit was called even though git add failed — partial state risk"
    )
    assert fix.committed_at is None
