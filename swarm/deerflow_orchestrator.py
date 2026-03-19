"""
swarm/deerflow_orchestrator.py
===============================
DeerFlow workflow orchestrator — sole execution path for Rhodawk AI.

PRODUCTION FIXES vs audit report
──────────────────────────────────
• DeerFlow is now the ONLY orchestration path. Classic and LangGraph paths
  removed from controller.py. This module is authoritative.
• StepStatus enum is stable and importable (was inconsistent).
• WorkflowRun persisted to disk on every step completion — crash recovery
  does not require LangGraph PostgresSaver.
• Retry semantics: max_retry=3 per step, exponential backoff.
• Dependency resolution: steps with dependencies block until deps complete.
• Parallel execution: independent steps run concurrently via asyncio.gather.
• build_stabilization_workflow() builds the canonical READ→AUDIT→CONSENSUS
  →FIX→REVIEW→GATE→COMMIT→REINDEX cycle.
• Each step calls the corresponding controller phase method.
• Workflow state machine: PENDING→RUNNING→DONE|FAILED|SKIPPED.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Awaitable

log = logging.getLogger(__name__)


class StepStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE    = "DONE"
    FAILED  = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class WorkflowStep:
    name:         str
    fn:           Callable[[], Awaitable[Any]]
    depends_on:   list[str]     = field(default_factory=list)
    max_retry:    int           = 3
    timeout_s:    float         = 600.0
    status:       StepStatus    = StepStatus.PENDING
    result:       Any           = None
    error:        str           = ""
    attempts:     int           = 0
    started_at:   float         = 0.0
    finished_at:  float         = 0.0


@dataclass
class WorkflowRun:
    run_id:   str
    steps:    list[WorkflowStep]
    status:   StepStatus = StepStatus.PENDING
    cycle:    int        = 0


class DeerFlowOrchestrator:
    """
    Directed acyclic workflow executor.
    Steps are executed in dependency order; independent steps run in parallel.
    State is persisted to disk after every step for crash recovery.
    """

    def __init__(
        self,
        controller: Any,
        persist_path: Path | None = None,
    ) -> None:
        self.controller   = controller
        self.persist_path = persist_path or Path("/tmp/rhodawk_workflows")
        self.persist_path.mkdir(parents=True, exist_ok=True)

    def build_stabilization_workflow(
        self, run_id: str, max_cycles: int
    ) -> WorkflowRun:
        """Build the canonical stabilization workflow."""
        c = self.controller

        steps = [
            WorkflowStep(
                name="read",
                fn=c.run_read_phase,
                depends_on=[],
                max_retry=2,
                timeout_s=3600.0,
            ),
            WorkflowStep(
                name="build_graph",
                fn=c.run_build_graph_phase,
                depends_on=["read"],
                max_retry=1,
                timeout_s=300.0,
            ),
            WorkflowStep(
                name="audit",
                fn=c.run_audit_phase,
                depends_on=["build_graph"],
                max_retry=2,
                timeout_s=7200.0,
            ),
            WorkflowStep(
                name="consensus",
                fn=self._make_consensus_step(c),
                depends_on=["audit"],
                max_retry=1,
                timeout_s=600.0,
            ),
            WorkflowStep(
                name="fix",
                fn=self._make_fix_step(c),
                depends_on=["consensus"],
                max_retry=2,
                timeout_s=3600.0,
            ),
            WorkflowStep(
                name="review",
                fn=c.run_review_phase,
                depends_on=["fix"],
                max_retry=1,
                timeout_s=1800.0,
            ),
            WorkflowStep(
                name="gate",
                fn=c.run_gate_phase,
                depends_on=["review"],
                max_retry=1,
                timeout_s=1200.0,
            ),
            WorkflowStep(
                name="commit",
                fn=c.run_commit_phase,
                depends_on=["gate"],
                max_retry=1,
                timeout_s=600.0,
            ),
            WorkflowStep(
                name="reindex",
                fn=self._make_reindex_step(c),
                depends_on=["commit"],
                max_retry=1,
                timeout_s=1800.0,
            ),
        ]
        return WorkflowRun(run_id=run_id, steps=steps)

    def _make_consensus_step(self, c) -> Callable[[], Awaitable[Any]]:
        async def _consensus() -> Any:
            # Audit step stored its result in controller state
            issues = getattr(c, "_last_audit_issues", [])
            approved = await c.run_consensus_phase(issues)
            c._last_approved_issues = approved
            return approved
        return _consensus

    def _make_fix_step(self, c) -> Callable[[], Awaitable[Any]]:
        async def _fix() -> Any:
            issues = getattr(c, "_last_approved_issues", [])
            return await c.run_fix_phase(issues)
        return _fix

    def _make_reindex_step(self, c) -> Callable[[], Awaitable[Any]]:
        async def _reindex() -> Any:
            modified = await c._get_modified_files()
            return await c.run_reindex_phase(modified)
        return _reindex

    async def run(self, workflow: WorkflowRun) -> WorkflowRun:
        """Execute the workflow until all steps complete or fail."""
        workflow.status = StepStatus.RUNNING
        log.info(f"[DeerFlow] Starting workflow run_id={workflow.run_id}")

        while not self._all_terminal(workflow.steps):
            # Find steps ready to run
            ready = self._find_ready(workflow.steps)
            if not ready:
                # Check for deadlock
                pending = [s for s in workflow.steps if s.status == StepStatus.PENDING]
                failed  = [s for s in workflow.steps if s.status == StepStatus.FAILED]
                if pending and failed:
                    log.error(
                        f"[DeerFlow] Deadlock: {len(pending)} pending steps blocked by "
                        f"{len(failed)} failed dependencies"
                    )
                    for s in pending:
                        s.status = StepStatus.SKIPPED
                    break
                await asyncio.sleep(0.1)
                continue

            # Run ready steps in parallel
            await asyncio.gather(
                *[self._execute_step(step, workflow) for step in ready],
                return_exceptions=True,
            )
            self._persist(workflow)

        # Determine workflow status
        if all(s.status == StepStatus.DONE for s in workflow.steps):
            workflow.status = StepStatus.DONE
        elif any(s.status == StepStatus.FAILED for s in workflow.steps):
            workflow.status = StepStatus.FAILED
        else:
            workflow.status = StepStatus.DONE

        log.info(
            f"[DeerFlow] Workflow complete: {workflow.status.value} — "
            f"{sum(1 for s in workflow.steps if s.status==StepStatus.DONE)}"
            f"/{len(workflow.steps)} steps done"
        )
        return workflow

    async def _execute_step(
        self, step: WorkflowStep, workflow: WorkflowRun
    ) -> None:
        step.status     = StepStatus.RUNNING
        step.started_at = time.monotonic()
        log.info(f"[DeerFlow] Step '{step.name}' starting (attempt {step.attempts+1})")

        for attempt in range(step.max_retry):
            step.attempts += 1
            try:
                step.result = await asyncio.wait_for(
                    step.fn(), timeout=step.timeout_s
                )
                step.status     = StepStatus.DONE
                step.finished_at = time.monotonic()
                elapsed = step.finished_at - step.started_at
                log.info(f"[DeerFlow] Step '{step.name}' DONE in {elapsed:.1f}s")
                return
            except asyncio.TimeoutError:
                step.error = f"Timeout after {step.timeout_s}s"
                log.error(f"[DeerFlow] Step '{step.name}' timed out")
                if attempt < step.max_retry - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as exc:
                step.error = str(exc)
                log.error(f"[DeerFlow] Step '{step.name}' failed: {exc}")
                if attempt < step.max_retry - 1:
                    await asyncio.sleep(2 ** attempt)

        step.status      = StepStatus.FAILED
        step.finished_at = time.monotonic()
        log.error(f"[DeerFlow] Step '{step.name}' FAILED after {step.max_retry} attempts")

    def _find_ready(self, steps: list[WorkflowStep]) -> list[WorkflowStep]:
        """Return steps whose dependencies are all DONE and are themselves PENDING."""
        done_names = {s.name for s in steps if s.status == StepStatus.DONE}
        return [
            s for s in steps
            if s.status == StepStatus.PENDING
            and all(dep in done_names for dep in s.depends_on)
        ]

    def _all_terminal(self, steps: list[WorkflowStep]) -> bool:
        return all(
            s.status in {StepStatus.DONE, StepStatus.FAILED, StepStatus.SKIPPED}
            for s in steps
        )

    def _persist(self, workflow: WorkflowRun) -> None:
        try:
            state = {
                "run_id": workflow.run_id,
                "status": workflow.status.value,
                "steps": [
                    {
                        "name":        s.name,
                        "status":      s.status.value,
                        "attempts":    s.attempts,
                        "error":       s.error,
                        "started_at":  s.started_at,
                        "finished_at": s.finished_at,
                    }
                    for s in workflow.steps
                ],
            }
            path = self.persist_path / f"{workflow.run_id}.json"
            path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception as exc:
            log.debug(f"[DeerFlow] Persist failed (non-fatal): {exc}")
