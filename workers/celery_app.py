"""
workers/celery_app.py
=====================
Celery distributed worker configuration for Rhodawk AI.

Enables horizontal scaling to 10M+ line codebases by distributing:
• File reading tasks across worker pool
• Independent fix generation per file group
• Parallel static analysis gate checks
• Gap 4: commit-granularity incremental audit tasks

Environment variables
──────────────────────
REDIS_URL           — Celery broker and result backend (default: redis://localhost:6379/0)
RHODAWK_WORKERS     — number of Celery worker processes (default: 4)
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

try:
    from celery import Celery  # type: ignore[import]

    celery_app = Celery(
        "rhodawk",
        broker=_REDIS_URL,
        backend=_REDIS_URL,
        include=["workers.tasks"],
    )

    celery_app.conf.update(
        task_serializer            = "json",
        result_serializer          = "json",
        accept_content             = ["json"],
        timezone                   = "UTC",
        enable_utc                 = True,
        task_track_started         = True,
        task_acks_late             = True,
        worker_prefetch_multiplier = 1,
        task_routes                = {
            "workers.tasks.read_file_task":       {"queue": "read"},
            "workers.tasks.fix_file_task":        {"queue": "fix"},
            "workers.tasks.gate_check_task":      {"queue": "gate"},
            # Gap 4: commit-audit tasks run on a dedicated queue so heavy
            # CPG queries don't block fix generation or file reading.
            "workers.tasks.commit_audit_task":    {"queue": "commit"},
        },
    )

    _CELERY_AVAILABLE = True
    log.info(f"Celery configured with broker: {_REDIS_URL}")

except ImportError:
    celery_app = None  # type: ignore[assignment]
    _CELERY_AVAILABLE = False
    log.info("Celery not installed — distributed workers disabled. Run: pip install celery[redis]")
