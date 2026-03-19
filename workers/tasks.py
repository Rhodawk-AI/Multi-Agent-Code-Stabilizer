"""workers/tasks.py — Celery tasks for distributed stabilization."""
from __future__ import annotations

import asyncio
import logging

log = logging.getLogger(__name__)

try:
    from workers.celery_app import celery_app

    if celery_app:

        @celery_app.task(name="workers.tasks.read_file_task", bind=True, max_retries=3)
        def read_file_task(self, file_path: str, run_id: str, db_path: str) -> dict:
            """Read and analyse a single file in a worker process."""
            try:
                from brain.sqlite_storage import SQLiteBrainStorage
                from agents.reader import ReaderAgent
                from agents.base import AgentConfig
                import asyncio

                async def _run():
                    storage = SQLiteBrainStorage(db_path)
                    await storage.initialise()
                    from pathlib import Path
                    reader = ReaderAgent(
                        storage=storage,
                        run_id=run_id,
                        repo_root=Path(file_path).parent.parent,
                        config=AgentConfig(),
                    )
                    result = await reader._process_file(
                        Path(file_path), file_path, force=True
                    )
                    await storage.close()
                    return result

                return {"status": "ok", "processed": asyncio.run(_run())}
            except Exception as exc:
                self.retry(exc=exc, countdown=2 ** self.request.retries)

        @celery_app.task(name="workers.tasks.gate_check_task", bind=True)
        def gate_check_task(self, file_path: str, content: str, domain_mode: str = "general") -> dict:
            """Run static analysis gate on a file in a worker process."""
            try:
                from sandbox.executor import StaticAnalysisGate
                from pathlib import Path

                async def _run():
                    gate = StaticAnalysisGate(domain_mode=domain_mode)
                    result = await gate.validate(file_path, content)
                    return {"approved": result.approved, "reason": result.rejection_reason}

                return asyncio.run(_run())
            except Exception as exc:
                return {"approved": False, "reason": str(exc)}

except (ImportError, TypeError):
    log.info("Celery tasks not registered — celery not available")
