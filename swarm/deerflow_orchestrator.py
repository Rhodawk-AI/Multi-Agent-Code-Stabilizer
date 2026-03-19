"""
swarm/deerflow_orchestrator.py
==============================
DeerFlow-inspired workflow orchestration for Rhodawk AI.

DeerFlow (https://github.com/bytedance/deer-flow) provides structured
multi-step research and execution workflows.  This module implements
a compatible workflow engine that:

1. Decomposes a stabilization run into a directed workflow graph.
2. Executes each workflow step with the appropriate swarm agent.
3. Handles step retries, partial failures, and step dependencies.
4. Persists workflow state for resumption after crashes.
5. Emits structured events for the dashboard.

DeerFlow-compatible workflow format
─────────────────────────────────────
{
  "id": "wf-<uuid>",
  "name": "Stabilize repo X",
  "steps": [
    {"id": "s1", "type": "read",    "deps": [],       "timeout": 600},
    {"id": "s2", "type": "audit",   "deps": ["s1"],   "timeout": 300},
    {"id": "s3", "type": "fix",     "deps": ["s2"],   "timeout": 600},
    {"id": "s4", "type": "review",  "deps": ["s3"],   "timeout": 300},
    {"id": "s5", "type": "gate",    "deps": ["s4"],   "timeout": 120},
    {"id": "s6", "type": "commit",  "deps": ["s5"],   "timeout": 60},
  ]
}
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine

log = logging.getLogger(__name__)


class StepStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"
    SKIPPED   = "skipped"


@dataclass
class WorkflowStep:
    id:         str
    type:       str
    deps:       list[str]       = field(default_factory=list)
    timeout_s:  int             = 300
    max_retry:  int             = 2
    status:     StepStatus      = StepStatus.PENDING
    result:     Any             = None
    error:      str             = ""
    started_at: datetime | None = None
    ended_at:   datetime | None = None
    attempt:    int             = 0


@dataclass
class Workflow:
    id:         str                 = field(default_factory=lambda: f"wf-{uuid.uuid4().hex[:8]}")
    name:       str                 = ""
    steps:      list[WorkflowStep]  = field(default_factory=list)
    status:     StepStatus          = StepStatus.PENDING
    created_at: datetime            = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    metadata:   dict[str, Any]      = field(default_factory=dict)


EventHandler = Callable[[str, str, Any], Coroutine]  # (workflow_id, event_type, payload)


class DeerFlowOrchestrator:
    """
    Executes Rhodawk AI stabilization workflows using a DeerFlow-compatible
    directed step graph.

    Usage
    ─────
    orchestrator = DeerFlowOrchestrator(controller)
    workflow = orchestrator.build_stabilization_workflow(run_id)
    await orchestrator.run(workflow)
    """

    def __init__(
        self,
        controller:     Any,
        persist_path:   Path | None  = None,
        event_handlers: list[EventHandler] | None = None,
    ) -> None:
        self.controller     = controller
        self.persist_path   = persist_path
        self._handlers      = event_handlers or []
        self._step_registry: dict[str, Callable] = {}
        self._register_steps()

    # ── Step registration ─────────────────────────────────────────────────────

    def _register_steps(self) -> None:
        self._step_registry = {
            "read":    self._step_read,
            "audit":   self._step_audit,
            "fix":     self._step_fix,
            "review":  self._step_review,
            "gate":    self._step_gate,
            "commit":  self._step_commit,
            "test":    self._step_test,
            "verify":  self._step_verify,
        }

    async def _step_read(self, step: WorkflowStep, ctx: dict) -> dict:
        incremental = ctx.get("cycle", 0) > 0
        force_reread = set(ctx.get("modified_files", []))
        await self.controller.run_read_phase(
            incremental=incremental,
            force_reread=force_reread or None,
        )
        await self.controller._build_graph()
        return {"status": "read_complete"}

    async def _step_audit(self, step: WorkflowStep, ctx: dict) -> dict:
        issues = await self.controller._phase_audit()
        score  = await self.controller._record_score(issues)
        return {"issues": len(issues), "score": score.score}

    async def _step_fix(self, step: WorkflowStep, ctx: dict) -> dict:
        issues = await self.controller.storage.list_issues(
            run_id=self.controller.run.id, status="OPEN"
        )
        approved = await self.controller._apply_consensus(issues)
        if approved:
            await self.controller._phase_fix(approved)
        return {"fixed": len(approved)}

    async def _step_review(self, step: WorkflowStep, ctx: dict) -> dict:
        await self.controller._phase_review()
        return {"status": "reviewed"}

    async def _step_gate(self, step: WorkflowStep, ctx: dict) -> dict:
        await self.controller._phase_gate()
        return {"status": "gated"}

    async def _step_commit(self, step: WorkflowStep, ctx: dict) -> dict:
        await self.controller._phase_commit()
        modified = await self.controller._get_modified_files()
        return {"committed": True, "modified": list(modified)}

    async def _step_test(self, step: WorkflowStep, ctx: dict) -> dict:
        # TestRunnerAgent is invoked inline by _phase_commit
        return {"status": "tests_run"}

    async def _step_verify(self, step: WorkflowStep, ctx: dict) -> dict:
        # Formal verification is handled in _phase_gate for critical fixes
        return {"status": "verified"}

    # ── Workflow builder ──────────────────────────────────────────────────────

    def build_stabilization_workflow(self, run_id: str, cycles: int = 50) -> Workflow:
        """Build the standard stabilization workflow."""
        steps = [
            WorkflowStep(id="read",   type="read",   deps=[],          timeout_s=900),
            WorkflowStep(id="audit",  type="audit",  deps=["read"],    timeout_s=600),
            WorkflowStep(id="fix",    type="fix",    deps=["audit"],   timeout_s=900),
            WorkflowStep(id="review", type="review", deps=["fix"],     timeout_s=600),
            WorkflowStep(id="gate",   type="gate",   deps=["review"],  timeout_s=300),
            WorkflowStep(id="commit", type="commit", deps=["gate"],    timeout_s=120),
        ]
        return Workflow(
            name=f"Stabilize run {run_id[:8]}",
            steps=steps,
            metadata={"run_id": run_id, "max_cycles": cycles},
        )

    # ── Execution engine ──────────────────────────────────────────────────────

    async def run(self, workflow: Workflow) -> Workflow:
        """
        Execute the workflow, respecting dependencies and retrying failed steps.

        Returns the workflow with all step statuses updated.
        """
        workflow.status = StepStatus.RUNNING
        ctx: dict = dict(workflow.metadata)
        log.info(f"DeerFlow: starting workflow '{workflow.name}' ({workflow.id})")

        await self._emit(workflow.id, "workflow_started", {"name": workflow.name})

        # Build adjacency for dependency resolution
        step_map = {s.id: s for s in workflow.steps}
        completed: set[str] = set()
        failed:    set[str] = set()

        while True:
            # Find runnable steps (deps all done, not yet run)
            runnable = [
                s for s in workflow.steps
                if s.status == StepStatus.PENDING
                and all(d in completed for d in s.deps)
                and not any(d in failed for d in s.deps)
            ]

            if not runnable:
                break

            # Run runnable steps concurrently
            tasks = [self._execute_step(s, ctx, workflow.id) for s in runnable]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for step, result in zip(runnable, results):
                if isinstance(result, Exception):
                    step.status = StepStatus.FAILED
                    step.error  = str(result)
                    failed.add(step.id)
                    log.error(f"DeerFlow: step '{step.id}' failed: {result}")
                    await self._emit(workflow.id, "step_failed",
                                     {"step": step.id, "error": str(result)})
                else:
                    step.status = StepStatus.DONE
                    step.result = result
                    completed.add(step.id)
                    if isinstance(result, dict):
                        ctx.update(result)
                    await self._emit(workflow.id, "step_done",
                                     {"step": step.id, "result": result})

        all_done = all(
            s.status in (StepStatus.DONE, StepStatus.SKIPPED)
            for s in workflow.steps
        )
        workflow.status = StepStatus.DONE if all_done else StepStatus.FAILED
        await self._emit(workflow.id, "workflow_complete",
                         {"status": workflow.status.value})
        self._persist(workflow)
        return workflow

    async def _execute_step(
        self, step: WorkflowStep, ctx: dict, wf_id: str
    ) -> Any:
        handler = self._step_registry.get(step.type)
        if not handler:
            log.warning(f"DeerFlow: no handler for step type '{step.type}'")
            step.status = StepStatus.SKIPPED
            return None

        step.status     = StepStatus.RUNNING
        step.started_at = datetime.now(tz=timezone.utc)
        step.attempt   += 1

        log.info(f"DeerFlow: executing step '{step.id}' (type={step.type})")
        await self._emit(wf_id, "step_started", {"step": step.id, "type": step.type})

        last_exc: Exception | None = None
        for attempt in range(step.max_retry + 1):
            try:
                result = await asyncio.wait_for(
                    handler(step, ctx),
                    timeout=step.timeout_s,
                )
                step.ended_at = datetime.now(tz=timezone.utc)
                return result
            except asyncio.TimeoutError as exc:
                last_exc = exc
                log.warning(f"DeerFlow: step '{step.id}' timed out (attempt {attempt + 1})")
            except Exception as exc:
                last_exc = exc
                log.warning(
                    f"DeerFlow: step '{step.id}' failed (attempt {attempt + 1}): {exc}"
                )
                if attempt < step.max_retry:
                    await asyncio.sleep(2 ** attempt)

        raise last_exc or RuntimeError(f"Step '{step.id}' failed after retries")

    # ── Event emission ────────────────────────────────────────────────────────

    async def _emit(self, wf_id: str, event_type: str, payload: Any) -> None:
        for handler in self._handlers:
            try:
                await handler(wf_id, event_type, payload)
            except Exception as exc:
                log.debug(f"DeerFlow event handler error: {exc}")

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist(self, workflow: Workflow) -> None:
        if not self.persist_path:
            return
        try:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            path = self.persist_path / f"{workflow.id}.json"
            import dataclasses
            path.write_text(
                json.dumps(dataclasses.asdict(workflow), default=str), encoding="utf-8"
            )
        except Exception as exc:
            log.warning(f"DeerFlow: failed to persist workflow state: {exc}")
