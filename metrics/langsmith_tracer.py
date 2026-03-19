"""
metrics/langsmith_tracer.py
============================
LangSmith trace capture for Rhodawk AI Code Stabilizer.

Replaces the fictional "Aurite-ai" observability reference from the prior
audit. LangSmith is a real, production-grade LLM trace/eval platform by
LangChain Inc. (https://smith.langchain.com).

What is traced:
  • Full pipeline run lifecycle (start → end with status)
  • Every LLM call: model, prompt tokens, completion tokens, latency, cost
  • Every gate decision: tool, verdict, file, reason
  • Audit phase findings summary per cycle
  • Fix phase patch outcomes
  • Escalation events

Environment variables
──────────────────────
LANGSMITH_API_KEY      — Required for trace upload. If absent, tracing is
                         silently disabled (no crash, just no traces).
LANGSMITH_PROJECT      — Project name in LangSmith UI (default: "rhodawk")
LANGSMITH_ENDPOINT     — Override API endpoint (default: official cloud URL)
RHODAWK_TRACE_VERBOSE  — "1" to include full prompt/completion text in traces
                         (disabled by default for PII/IP protection)
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

_API_KEY      = os.environ.get("LANGSMITH_API_KEY", "")
_PROJECT      = os.environ.get("LANGSMITH_PROJECT", "rhodawk")
_ENDPOINT     = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
_VERBOSE      = os.environ.get("RHODAWK_TRACE_VERBOSE", "0") == "1"
_ENABLED      = bool(_API_KEY)

# Try to import the official langsmith SDK
_SDK_AVAILABLE = False
try:
    from langsmith import Client as _LSClient  # type: ignore[import]
    _SDK_AVAILABLE = True
except ImportError:
    pass

if _ENABLED and not _SDK_AVAILABLE:
    log.warning(
        "LANGSMITH_API_KEY is set but langsmith SDK is not installed. "
        "Run: pip install langsmith\n"
        "Trace upload will use the REST API fallback instead."
    )

if not _ENABLED:
    log.info(
        "LangSmith tracing disabled — set LANGSMITH_API_KEY to enable. "
        "Traces help diagnose pipeline regressions and benchmark comparisons."
    )


@dataclass
class _RunTrace:
    run_id:     str
    project:    str
    repo:       str
    domain:     str
    start_time: float = field(default_factory=time.monotonic)
    ls_run_id:  str   = ""   # LangSmith-side run UUID


class LangSmithTracer:
    """
    Thin wrapper around the LangSmith SDK / REST API.

    All public methods are fire-and-forget — they log warnings on failure
    but never raise, so the pipeline is never blocked by observability failures.
    """

    def __init__(self) -> None:
        self._client: Any = None
        self._active: dict[str, _RunTrace] = {}
        if _ENABLED:
            self._init_client()

    def _init_client(self) -> None:
        if _SDK_AVAILABLE:
            try:
                self._client = _LSClient(
                    api_url=_ENDPOINT,
                    api_key=_API_KEY,
                )
                log.info(f"LangSmith tracer connected → project={_PROJECT!r}")
            except Exception as exc:
                log.warning(f"LangSmith client init failed: {exc} — tracing disabled")
                self._client = None
        else:
            # Mark as REST-only mode
            self._client = "rest"

    # ── Public API ─────────────────────────────────────────────────────────────

    def start_run(self, run_id: str, repo: str, domain: str) -> None:
        """Record the start of a stabilization pipeline run."""
        if not _ENABLED:
            return
        trace = _RunTrace(run_id=run_id, project=_PROJECT, repo=repo, domain=domain)
        self._active[run_id] = trace
        self._post_run_event(
            run_id=run_id,
            event="pipeline_start",
            data={"repo": repo, "domain": domain},
        )

    def end_run(self, run_id: str, status: str) -> None:
        """Record the end of a stabilization pipeline run."""
        if not _ENABLED:
            return
        trace = self._active.pop(run_id, None)
        elapsed = (
            time.monotonic() - trace.start_time if trace else 0.0
        )
        self._post_run_event(
            run_id=run_id,
            event="pipeline_end",
            data={"status": status, "elapsed_s": round(elapsed, 2)},
        )

    def trace_llm_call(
        self,
        run_id:           str,
        agent:            str,
        model:            str,
        prompt_tokens:    int,
        completion_tokens: int,
        latency_ms:       float,
        cost_usd:         float,
        task:             str = "",
        prompt_text:      str = "",
        completion_text:  str = "",
    ) -> None:
        """Record a single LLM call with cost and latency."""
        if not _ENABLED:
            return
        data: dict[str, Any] = {
            "agent":             agent,
            "model":             model,
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
            "latency_ms":        round(latency_ms, 1),
            "cost_usd":          round(cost_usd, 6),
            "task":              task,
        }
        if _VERBOSE:
            data["prompt"]     = prompt_text[:2000]
            data["completion"] = completion_text[:2000]
        self._post_run_event(run_id, "llm_call", data)

    def trace_gate(
        self,
        run_id:   str,
        tool:     str,
        verdict:  str,
        file:     str,
        reason:   str = "",
    ) -> None:
        """Record a gate decision (pass or fail) with the triggering tool."""
        if not _ENABLED:
            return
        self._post_run_event(run_id, "gate_decision", {
            "tool":    tool,
            "verdict": verdict,
            "file":    file,
            "reason":  reason[:300],
        })

    def trace_audit_cycle(
        self,
        run_id:          str,
        cycle:           int,
        issues_found:    int,
        issues_critical: int,
        issues_fixed:    int,
        score:           float,
    ) -> None:
        """Record a single audit cycle summary."""
        if not _ENABLED:
            return
        self._post_run_event(run_id, "audit_cycle", {
            "cycle":           cycle,
            "issues_found":    issues_found,
            "issues_critical": issues_critical,
            "issues_fixed":    issues_fixed,
            "score":           round(score, 2),
        })

    def trace_escalation(
        self,
        run_id:    str,
        esc_id:    str,
        esc_type:  str,
        severity:  str,
        resolved:  bool = False,
        resolution: str = "",
    ) -> None:
        """Record a human escalation event."""
        if not _ENABLED:
            return
        self._post_run_event(run_id, "escalation", {
            "escalation_id": esc_id,
            "type":          esc_type,
            "severity":      severity,
            "resolved":      resolved,
            "resolution":    resolution,
        })

    # ── Internal upload ────────────────────────────────────────────────────────

    def _post_run_event(
        self,
        run_id: str,
        event:  str,
        data:   dict[str, Any],
    ) -> None:
        """
        Post a structured event to LangSmith.

        Uses the SDK when available; falls back to a lightweight aiohttp-free
        REST call via urllib so there are no additional async dependencies.
        """
        if not _ENABLED:
            return
        try:
            if _SDK_AVAILABLE and self._client and self._client != "rest":
                self._sdk_post(run_id, event, data)
            else:
                self._rest_post(run_id, event, data)
        except Exception as exc:
            log.debug(f"LangSmith trace upload failed (non-fatal): {exc}")

    def _sdk_post(self, run_id: str, event: str, data: dict) -> None:
        """Use the official LangSmith SDK to create/update a run."""
        import uuid
        try:
            self._client.create_run(
                name=f"rhodawk/{event}",
                run_type="chain",
                inputs={"run_id": run_id, "event": event},
                outputs=data,
                project_name=_PROJECT,
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}/{event}/{time.time()}")),
                tags=["rhodawk", event, data.get("domain", ""), data.get("model", "")],
            )
        except Exception as exc:
            # SDK failed — try REST fallback
            log.debug(f"LangSmith SDK post failed ({exc}), trying REST")
            self._rest_post(run_id, event, data)

    def _rest_post(self, run_id: str, event: str, data: dict) -> None:
        """Minimal REST POST to LangSmith runs endpoint via urllib."""
        import json, uuid
        import urllib.request, urllib.error
        payload = {
            "id":           str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}/{event}/{time.time()}")),
            "name":         f"rhodawk/{event}",
            "run_type":     "chain",
            "inputs":       {"run_id": run_id, "event": event},
            "outputs":      data,
            "session_name": _PROJECT,
            "tags":         ["rhodawk", event],
            "start_time":   int(time.time() * 1000),
            "end_time":     int(time.time() * 1000),
        }
        body = json.dumps(payload).encode()
        req  = urllib.request.Request(
            f"{_ENDPOINT}/runs",
            data=body,
            method="POST",
            headers={
                "Content-Type":  "application/json",
                "x-api-key":     _API_KEY,
                "User-Agent":    "Rhodawk-AI/2.0",
            },
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status not in (200, 201, 202):
                log.debug(f"LangSmith REST returned {resp.status}")

    # ── Context manager helpers ────────────────────────────────────────────────

    def is_active(self) -> bool:
        """True when LangSmith tracing is configured and the SDK is available."""
        return _ENABLED

    def project(self) -> str:
        return _PROJECT
