"""metrics/prometheus_exporter.py — Prometheus metrics for Rhodawk AI."""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False
    log.info("prometheus_client not installed — metrics disabled")


def _noop(*a, **kw):
    class _Stub:
        def inc(self, *a, **kw): pass
        def dec(self, *a, **kw): pass
        def set(self, *a, **kw): pass
        def observe(self, *a, **kw): pass
        def labels(self, *a, **kw): return self
        def __call__(self, *a, **kw): return self
        def __enter__(self): return self
        def __exit__(self, *a): pass
    return _Stub()


if _PROM_AVAILABLE:
    ACTIVE_RUNS    = Gauge("rhodawk_active_runs", "Active stabilization runs")
    ISSUES_TOTAL   = Counter("rhodawk_issues_total", "Total issues found",
                             ["severity", "domain"])
    FIXES_TOTAL    = Counter("rhodawk_fixes_total", "Total fix attempts",
                             ["status"])
    GATE_RESULTS   = Counter("rhodawk_gate_results_total", "Gate pass/fail",
                             ["result"])
    TEST_RUNS      = Counter("rhodawk_test_runs_total", "Test run outcomes",
                             ["status"])
    CYCLE_DURATION = Histogram("rhodawk_cycle_duration_seconds",
                               "Seconds per stabilization cycle",
                               buckets=[10,30,60,120,300,600,1200,3600])
    COST_PCT       = Gauge("rhodawk_cost_ceiling_pct",
                           "Current cost as percentage of ceiling")
else:
    ACTIVE_RUNS    = _noop()
    ISSUES_TOTAL   = _noop()
    FIXES_TOTAL    = _noop()
    GATE_RESULTS   = _noop()
    TEST_RUNS      = _noop()
    CYCLE_DURATION = _noop()
    COST_PCT       = _noop()


def record_issue(severity: str, domain: str = "GENERAL") -> None:
    try: ISSUES_TOTAL.labels(severity=severity, domain=domain).inc()
    except Exception: pass

def record_fix(status: str) -> None:
    try: FIXES_TOTAL.labels(status=status).inc()
    except Exception: pass

def record_gate(passed: bool) -> None:
    try: GATE_RESULTS.labels(result="pass" if passed else "fail").inc()
    except Exception: pass

def record_test_run(status: str) -> None:
    try: TEST_RUNS.labels(status=status).inc()
    except Exception: pass

def update_cost_pct(cost: float, ceiling: float) -> None:
    try:
        if ceiling > 0:
            COST_PCT.set(cost / ceiling * 100)
    except Exception: pass

class time_cycle:
    def __enter__(self):
        self._start = __import__("time").monotonic()
        return self
    def __exit__(self, *a):
        try: CYCLE_DURATION.observe(__import__("time").monotonic() - self._start)
        except Exception: pass
