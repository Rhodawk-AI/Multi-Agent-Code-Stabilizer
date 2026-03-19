"""
metrics/prometheus_exporter.py
================================
Prometheus instrumentation for Rhodawk AI.

Exposes a /metrics endpoint consumable by Prometheus + Grafana.

Metrics exported
─────────────────
• rhodawk_llm_calls_total{agent,model,status}      — LLM call counter
• rhodawk_llm_cost_usd_total{agent,model}           — cumulative cost
• rhodawk_llm_latency_seconds{agent,model}          — histogram
• rhodawk_issues_found_total{severity,domain}       — issues detected
• rhodawk_fixes_generated_total{status}             — fix attempts
• rhodawk_gate_results_total{result}                — gate pass/fail
• rhodawk_test_runs_total{status}                   — post-fix test results
• rhodawk_active_runs                               — gauge: runs in flight
• rhodawk_cycle_duration_seconds                    — histogram: cycle time
• rhodawk_swebench_score                            — gauge: latest benchmark
• rhodawk_cost_ceiling_pct                          — gauge: cost burn %
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

log = logging.getLogger(__name__)

try:
    from prometheus_client import (   # type: ignore[import]
        Counter, Gauge, Histogram, Summary, Info,
        CollectorRegistry, make_asgi_app,
        REGISTRY as _DEFAULT_REGISTRY,
    )
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False
    log.warning(
        "prometheus-client not installed — metrics disabled. "
        "Run: pip install prometheus-client"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Metric definitions
# ──────────────────────────────────────────────────────────────────────────────

class _Stub:
    """No-op stub for all prometheus_client objects."""
    def labels(self, **_):     return self
    def inc(self, *_, **__):   pass
    def dec(self, *_, **__):   pass
    def set(self, *_, **__):   pass
    def observe(self, *_, **__): pass
    def info(self, *_, **__):  pass
    def time(self):
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()


def _make(cls, *args, **kwargs):
    if not _PROM_AVAILABLE:
        return _Stub()
    try:
        return cls(*args, **kwargs)
    except Exception:
        return _Stub()


# LLM metrics
LLM_CALLS = _make(
    Counter, "rhodawk_llm_calls_total",
    "Total LLM calls", ["agent", "model", "status"]
)
LLM_COST = _make(
    Counter, "rhodawk_llm_cost_usd_total",
    "Cumulative LLM cost in USD", ["agent", "model"]
)
LLM_LATENCY = _make(
    Histogram, "rhodawk_llm_latency_seconds",
    "LLM call latency", ["agent", "model"],
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
) if _PROM_AVAILABLE else _Stub()

# Issue metrics
ISSUES_FOUND = _make(
    Counter, "rhodawk_issues_found_total",
    "Issues detected by auditors", ["severity", "domain"]
)

# Fix metrics
FIXES_GENERATED = _make(
    Counter, "rhodawk_fixes_generated_total",
    "Fix attempts generated", ["status"]
)
GATE_RESULTS = _make(
    Counter, "rhodawk_gate_results_total",
    "Static analysis gate results", ["result"]
)
TEST_RUNS = _make(
    Counter, "rhodawk_test_runs_total",
    "Post-fix test run results", ["status"]
)

# Run metrics
ACTIVE_RUNS = _make(
    Gauge, "rhodawk_active_runs",
    "Number of active stabilization runs"
)
CYCLE_DURATION = _make(
    Histogram, "rhodawk_cycle_duration_seconds",
    "Duration of a single stabilization cycle",
    buckets=(30, 60, 120, 300, 600, 1200, 3600),
) if _PROM_AVAILABLE else _Stub()

# Benchmark metrics
SWEBENCH_SCORE = _make(
    Gauge, "rhodawk_swebench_score",
    "Latest SWE-bench Verified score (0.0–1.0)"
)
TERMINAL_BENCH_SCORE = _make(
    Gauge, "rhodawk_terminal_bench_score",
    "Latest Terminal-Bench 2.0 score (0.0–1.0)"
)

# Cost control
COST_CEILING_PCT = _make(
    Gauge, "rhodawk_cost_ceiling_pct",
    "Current spend as a percentage of cost ceiling"
)

# System info
BUILD_INFO = _make(
    Info, "rhodawk_build",
    "Build information"
) if _PROM_AVAILABLE else _Stub()


# ──────────────────────────────────────────────────────────────────────────────
# Convenience helpers
# ──────────────────────────────────────────────────────────────────────────────

def record_llm_call(
    agent: str,
    model: str,
    success: bool,
    cost_usd: float,
    latency_s: float,
) -> None:
    status = "success" if success else "error"
    LLM_CALLS.labels(agent=agent, model=model, status=status).inc()
    if success:
        LLM_COST.labels(agent=agent, model=model).inc(cost_usd)
        LLM_LATENCY.labels(agent=agent, model=model).observe(latency_s)


def record_issue(severity: str, domain: str) -> None:
    ISSUES_FOUND.labels(severity=severity, domain=domain).inc()


def record_fix(status: str) -> None:
    FIXES_GENERATED.labels(status=status).inc()


def record_gate(passed: bool) -> None:
    GATE_RESULTS.labels(result="pass" if passed else "fail").inc()


def record_test_run(status: str) -> None:
    TEST_RUNS.labels(status=status).inc()


def update_cost_pct(spent: float, ceiling: float) -> None:
    if ceiling > 0:
        COST_CEILING_PCT.set(spent / ceiling * 100.0)


@contextmanager
def time_cycle() -> Iterator[None]:
    start = time.monotonic()
    try:
        yield
    finally:
        CYCLE_DURATION.observe(time.monotonic() - start)


def set_build_info(version: str, commit: str = "unknown") -> None:
    try:
        BUILD_INFO.info({"version": version, "commit": commit})
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# ASGI metrics endpoint factory
# ──────────────────────────────────────────────────────────────────────────────

def make_metrics_app():
    """
    Returns an ASGI app that serves /metrics in Prometheus text format.
    Mount it on your FastAPI app::

        from metrics.prometheus_exporter import make_metrics_app
        app.mount("/metrics", make_metrics_app())
    """
    if not _PROM_AVAILABLE:
        async def _stub_app(scope, receive, send):
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            })
            await send({
                "type": "http.response.body",
                "body": b"# prometheus-client not installed\n",
            })
        return _stub_app
    return make_asgi_app()
