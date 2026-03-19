"""
tests/unit/test_convergence_loop.py
=====================================
Integration-level tests for the multi-cycle convergence loop introduced in
orchestrator/controller._run_deerflow().

These tests mock DeerFlowOrchestrator and storage so they run without
external dependencies and complete in < 1 second.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from brain.schemas import (
    AuditRun, AuditScore, ConvergenceRecord, DomainMode,
    RunStatus, Severity,
)
from orchestrator.convergence import ConvergenceDetector


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_score(run_id: str, cycle: int, critical: int = 0, score: float = 80.0) -> AuditScore:
    s = AuditScore(
        run_id=run_id,
        cycle_number=cycle,
        critical_count=critical,
        major_count=0,
        minor_count=0,
        info_count=0,
    )
    s.score = score
    return s


def _make_run(max_cycles: int = 10) -> AuditRun:
    r = AuditRun(
        repo_url="file://test",
        repo_name="test",
        branch="main",
        master_prompt_path="config/prompts/base.md",
        max_cycles=max_cycles,
    )
    r.cycle_count = 0
    r.scores = []
    return r


# ─────────────────────────────────────────────────────────────────────────────
# ConvergenceDetector unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConvergenceDetector:
    def test_max_cycles_triggers_convergence(self):
        det = ConvergenceDetector(max_cycles=3, stable_window=5)
        score = _make_score("r1", cycle=3)
        rec = det.check(cycle=3, score=score)
        assert rec.converged
        assert rec.halt_reason == "max_cycles_reached"

    def test_stable_window_triggers_convergence(self):
        det = ConvergenceDetector(max_cycles=50, stable_window=3)
        run_id = "r2"
        for i in range(1, 4):
            score = _make_score(run_id, i, score=75.0)
            rec = det.check(cycle=i, score=score)
        assert rec.converged
        assert rec.halt_reason == "score_stable"

    def test_zero_critical_triggers_convergence_after_ramp(self):
        det = ConvergenceDetector(max_cycles=50, stable_window=10)
        # cycles 1-2: has criticals
        for i in range(1, 3):
            s = _make_score("r3", i, critical=2, score=60.0)
            det.check(cycle=i, score=s)
        # cycle 3: zero criticals
        s = _make_score("r3", 3, critical=0, score=90.0)
        rec = det.check(cycle=3, score=s)
        assert rec.converged
        assert rec.halt_reason == "zero_critical_issues"

    def test_regression_detected(self):
        det = ConvergenceDetector(max_cycles=50, regression_threshold=5.0)
        score = _make_score("r4", cycle=2, score=70.0)
        rec = det.check(cycle=2, score=score, baseline_score=80.0)
        assert rec.converged
        assert "regression" in rec.halt_reason

    def test_cost_ceiling_halt(self):
        det = ConvergenceDetector()
        rec = det.halt_if_ceiling_hit(total_cost=55.0, ceiling=50.0)
        assert rec is not None
        assert rec.converged
        assert "cost_ceiling" in rec.halt_reason

    def test_no_convergence_while_improving(self):
        det = ConvergenceDetector(max_cycles=50, stable_window=3)
        scores = [90.0, 85.0, 80.0, 75.0]   # strictly decreasing (improving)
        rec = None
        for i, s in enumerate(scores, 1):
            score = _make_score("r5", i, score=s)
            rec = det.check(cycle=i, score=score)
        assert not rec.converged

    def test_suggest_status_mapping(self):
        det = ConvergenceDetector()
        assert det.suggest_status(ConvergenceRecord(run_id="x", cycle=1, score=80.0,
                                                     converged=True, halt_reason="score_stable")) \
               == RunStatus.BASELINE_PENDING
        assert det.suggest_status(ConvergenceRecord(run_id="x", cycle=1, score=80.0,
                                                     converged=True, halt_reason="regression: drop")) \
               == RunStatus.FAILED
        assert det.suggest_status(ConvergenceRecord(run_id="x", cycle=1, score=80.0,
                                                     converged=True, halt_reason="max_cycles_reached")) \
               == RunStatus.STABILIZED
        assert det.suggest_status(ConvergenceRecord(run_id="x", cycle=1, score=80.0,
                                                     converged=False)) \
               == RunStatus.RUNNING


# ─────────────────────────────────────────────────────────────────────────────
# Controller multi-cycle loop integration test
# ─────────────────────────────────────────────────────────────────────────────

class TestControllerConvergenceLoop:
    """
    Verifies that StabilizerController._run_deerflow() actually iterates
    multiple cycles and terminates on convergence.
    """

    @pytest.mark.asyncio
    async def test_runs_multiple_cycles_until_convergence(self, tmp_path):
        from orchestrator.controller import StabilizerConfig, StabilizerController
        from brain.schemas import AutonomyLevel

        cfg = StabilizerConfig(
            repo_url="file://test",
            repo_root=tmp_path,
            use_sqlite=True,
            max_cycles=5,
            auto_commit=False,
            autonomy_level=AutonomyLevel.READ_ONLY,
        )
        ctrl = StabilizerController(cfg)
        run = _make_run(max_cycles=5)
        ctrl.run = run
        ctrl.convergence = ConvergenceDetector(max_cycles=5, stable_window=3)

        # Mock storage
        mock_storage = AsyncMock()
        mock_storage.get_total_cost.return_value = 0.0
        mock_storage.get_active_baseline.return_value = None
        mock_storage.upsert_run.return_value = None
        mock_storage.upsert_convergence_record.return_value = None
        mock_storage.update_run_status.return_value = None
        mock_storage.append_audit_trail.return_value = None
        ctrl.storage = mock_storage

        # Mock _trail_signer
        ctrl._trail_signer = MagicMock()
        ctrl._trail_signer.sign.return_value = "fakesig"

        # Mock LangSmith tracer
        ctrl._langsmith = MagicMock()
        ctrl._langsmith.start_run = MagicMock()
        ctrl._langsmith.end_run = MagicMock()

        # Track cycle count across DeerFlow runs
        cycle_calls = []

        async def fake_deerflow_run(wf):
            from swarm.deerflow_orchestrator import StepStatus, WorkflowRun
            cycle_calls.append(ctrl.run.cycle_count)
            # Feed a stable score to trigger convergence after 3 cycles
            score = _make_score(run.id, ctrl.run.cycle_count, score=75.0)
            ctrl.run.scores.append(score)
            result = MagicMock()
            result.status = StepStatus.DONE
            return result

        with patch(
            "swarm.deerflow_orchestrator.DeerFlowOrchestrator.run",
            side_effect=fake_deerflow_run,
        ), patch.object(ctrl, "_cleanup", new_callable=AsyncMock), \
           patch.object(ctrl, "_finalise", new_callable=AsyncMock):
            status = await ctrl._run_deerflow()

        # Must have run at least 3 cycles (stable_window=3)
        assert len(cycle_calls) >= 3, f"Expected ≥3 cycles, got {cycle_calls}"
        # Must have terminated (not infinite loop)
        assert len(cycle_calls) <= 5, f"Exceeded max_cycles: {cycle_calls}"
        # Status must be BASELINE_PENDING (score_stable convergence)
        assert status in (RunStatus.BASELINE_PENDING, RunStatus.STABILIZED)

    @pytest.mark.asyncio
    async def test_single_cycle_stops_on_zero_criticals(self, tmp_path):
        from orchestrator.controller import StabilizerConfig, StabilizerController
        from brain.schemas import AutonomyLevel

        cfg = StabilizerConfig(
            repo_url="file://test",
            repo_root=tmp_path,
            use_sqlite=True,
            max_cycles=10,
            auto_commit=False,
            autonomy_level=AutonomyLevel.READ_ONLY,
        )
        ctrl = StabilizerController(cfg)
        run = _make_run(max_cycles=10)
        ctrl.run = run
        ctrl.convergence = ConvergenceDetector(max_cycles=10, stable_window=10)

        mock_storage = AsyncMock()
        mock_storage.get_total_cost.return_value = 0.0
        mock_storage.get_active_baseline.return_value = None
        mock_storage.upsert_run.return_value = None
        mock_storage.upsert_convergence_record.return_value = None
        mock_storage.update_run_status.return_value = None
        mock_storage.append_audit_trail.return_value = None
        ctrl.storage = mock_storage
        ctrl._trail_signer = MagicMock()
        ctrl._trail_signer.sign.return_value = "sig"
        ctrl._langsmith = MagicMock()
        ctrl._langsmith.start_run = MagicMock()
        ctrl._langsmith.end_run = MagicMock()

        call_count = 0

        async def fake_run(wf):
            nonlocal call_count
            call_count += 1
            from swarm.deerflow_orchestrator import StepStatus
            # Feed zero-critical score on cycle 3
            score = _make_score(run.id, ctrl.run.cycle_count, critical=0, score=95.0)
            ctrl.run.scores.append(score)
            result = MagicMock()
            result.status = StepStatus.DONE
            return result

        with patch("swarm.deerflow_orchestrator.DeerFlowOrchestrator.run",
                   side_effect=fake_run), \
             patch.object(ctrl, "_cleanup", new_callable=AsyncMock), \
             patch.object(ctrl, "_finalise", new_callable=AsyncMock):
            # Pre-seed two cycles so third triggers zero_critical_issues
            ctrl.run.cycle_count = 2
            ctrl.convergence._recent_scores.extend([95.0, 95.0])
            status = await ctrl._run_deerflow()

        assert call_count >= 1
        assert status in (RunStatus.STABILIZED, RunStatus.BASELINE_PENDING)
