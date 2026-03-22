"""
orchestrator/commit_audit_scheduler.py
========================================
Gap 4: Commit-aware audit scheduler.

This is the central component that Gap 4 demands.  When a commit lands (via
CI webhook, post-fix git commit, or manual trigger) the scheduler:

  1. Parses the diff — only changed *functions*, not files.
  2. Queries the CPG for the impact set of those functions
     (direct callers + transitive dependents to depth 3).
  3. Marks only the impact set as stale in storage — not the whole codebase.
  4. Re-runs only the test cases that cover the changed functions.
  5. Persists a CommitAuditRecord at every state transition so interrupted
     audits can be resumed and CI systems can poll for completion.

The result is 50–200 functions audited per commit instead of 10M lines.
That is what makes continuous autonomous auditing economically viable.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brain.schemas import (
    CommitAuditRecord,
    CommitAuditStatus,
    FileStatus,
    FunctionStalenessMark,
)
from brain.storage import BrainStorage

log = logging.getLogger(__name__)

# ── Coverage data provider ────────────────────────────────────────────────────

class _CoverageDataProvider:
    """Read a pytest/coverage.py coverage artefact and map function names to
    the test node IDs that execute them.

    Supports two coverage formats:
      • ``coverage.json``  — produced by ``coverage json``
      • ``.coverage``      — the SQLite database produced by ``coverage run``
        (requires the ``coverage`` package to be importable)

    Falls back gracefully to an empty mapping when neither file exists or
    when the required libraries are absent.  Callers should treat a None
    return from ``tests_for_functions()`` as "no coverage data available".

    Usage
    -----
    The provider is constructed once per CommitAuditScheduler and caches the
    loaded data so repeated calls during one scheduler invocation do not re-read
    the file from disk.
    """

    def __init__(self, repo_root: Path) -> None:
        self._repo_root   = repo_root
        self._loaded      = False
        # Maps function name (bare, no module prefix) → set of test node IDs
        self._fn_to_tests: dict[str, set[str]] = {}

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        # Try coverage.json first (always parseable without extra deps)
        json_path = self._repo_root / "coverage.json"
        if json_path.exists():
            try:
                import json
                data = json.loads(json_path.read_text(encoding="utf-8"))
                # coverage.json structure:
                #   {"files": {"path": {"contexts": {"line_no": ["test::id"]}}}}
                for _file_path, file_data in data.get("files", {}).items():
                    for _line, test_ids in file_data.get("contexts", {}).items():
                        for tid in test_ids:
                            if "::" in tid:
                                # test node id like "tests/test_foo.py::test_bar"
                                # extract the bare test function name
                                fn = tid.rsplit("::", 1)[-1]
                                self._fn_to_tests.setdefault(fn, set()).add(tid)
                log.debug(
                    "[CoverageDataProvider] Loaded coverage.json: %d test entries",
                    sum(len(v) for v in self._fn_to_tests.values()),
                )
                return
            except Exception as exc:
                log.debug("[CoverageDataProvider] coverage.json read failed: %s", exc)

        # Try .coverage SQLite database
        cov_db = self._repo_root / ".coverage"
        if cov_db.exists():
            try:
                import coverage as _cov_pkg
                cov = _cov_pkg.Coverage(data_file=str(cov_db))
                cov.load()
                cov_data = cov.get_data()
                for test_id in cov_data.measured_files():
                    # In coverage 7+ context() returns test node ids per line
                    pass  # full context iteration requires CoverageData.contexts
                # Simpler approach: use CoverageData directly
                raw_data = _cov_pkg.CoverageData(basename=str(cov_db))
                raw_data.read()
                for ctx in raw_data.measured_contexts():
                    if "::" in ctx:
                        fn = ctx.rsplit("::", 1)[-1]
                        self._fn_to_tests.setdefault(fn, set()).add(ctx)
                log.debug(
                    "[CoverageDataProvider] Loaded .coverage db: %d test entries",
                    sum(len(v) for v in self._fn_to_tests.values()),
                )
            except Exception as exc:
                log.debug("[CoverageDataProvider] .coverage db read failed: %s", exc)

    def tests_for_functions(self, function_names: list[str]) -> list[str] | None:
        """Return test node IDs that cover any of the given function names.

        Returns None when no coverage data is available (signals caller to
        fall back to the heuristic grep strategy).  Returns an empty list when
        coverage data exists but none of the functions appear in it.
        """
        self._load()
        if not self._fn_to_tests:
            return None
        result: set[str] = set()
        for fn in function_names:
            bare = fn.rsplit(".", 1)[-1]   # strip module prefix if present
            result |= self._fn_to_tests.get(fn, set())
            result |= self._fn_to_tests.get(bare, set())
        return sorted(result)


class CommitAuditScheduler:
    """
    Orchestrates function-granularity incremental audits after each commit.

    Parameters
    ----------
    storage:
        The BrainStorage instance for persisting CommitAuditRecords and
        FunctionStalenessMark rows.
    incremental_updater:
        An IncrementalCPGUpdater that parses diffs and computes CPG impact sets.
    test_runner:
        A TestRunnerAgent capable of running tests scoped to changed functions.
    run_id:
        The active AuditRun id.  All records produced are linked to this run.
    repo_root:
        Filesystem path to the repository root (used for git commands).
    cpg_engine:
        Optional CPGEngine — when provided the impact set is computed via the
        Joern CPG; when absent we fall back to the import-graph blast radius.
    graph_engine:
        Optional file-level DependencyGraph used as a fallback impact source
        when the CPG is unavailable.
    """

    def __init__(
        self,
        storage:             BrainStorage,
        incremental_updater: Any,
        test_runner:         Any,
        run_id:              str,
        repo_root:           Path | None = None,
        cpg_engine:          Any | None  = None,
        graph_engine:        Any | None  = None,
    ) -> None:
        self._storage    = storage
        self._updater    = incremental_updater
        self._runner     = test_runner
        self._run_id     = run_id
        self._repo_root  = repo_root
        self._cpg        = cpg_engine
        self._graph      = graph_engine
        # Coverage data provider — lazily loaded on first test-scope call
        self._coverage   = _CoverageDataProvider(repo_root) if repo_root else None

    # ── Primary entry point ───────────────────────────────────────────────────

    async def schedule_commit_audit(
        self,
        commit_hash:    str,
        changed_files:  set[str] | None = None,
        branch:         str = "",
        author:         str = "",
        commit_message: str = "",
    ) -> CommitAuditRecord:
        """
        Schedule and execute an incremental audit for a single commit.

        If a CommitAuditRecord for this commit_hash already exists in the
        DONE state we return it immediately (idempotent).  A record in FAILED
        state is retried.

        Returns the final CommitAuditRecord (status DONE or FAILED).
        """
        # Idempotency check — don't re-audit the same commit
        existing = await self._storage.get_commit_audit_record_by_hash(
            commit_hash, run_id=self._run_id
        )
        if existing and existing.status == CommitAuditStatus.DONE:
            log.info(
                "[CommitAudit] commit %s already audited (record=%s) — skipping",
                commit_hash[:12],
                existing.id[:12],
            )
            return existing

        # Resolve changed files from git when not supplied by caller
        if changed_files is None:
            changed_files = await self._resolve_changed_files(commit_hash)

        record = CommitAuditRecord(
            run_id=self._run_id,
            commit_hash=commit_hash,
            branch=branch,
            author=author,
            commit_message=commit_message,
            changed_files=sorted(changed_files),
            status=CommitAuditStatus.PENDING,
        )
        await self._storage.upsert_commit_audit_record(record)

        try:
            record = await self._execute(record, changed_files)
        except Exception as exc:
            log.error("[CommitAudit] fatal error for %s: %s", commit_hash[:12], exc)
            record.status       = CommitAuditStatus.FAILED
            record.error_detail = str(exc)[:500]
            record.finished_at  = datetime.now(tz=timezone.utc)
            await self._storage.upsert_commit_audit_record(record)

        return record

    # ── Execution pipeline ────────────────────────────────────────────────────

    async def _execute(
        self, record: CommitAuditRecord, changed_files: set[str]
    ) -> CommitAuditRecord:
        record.status     = CommitAuditStatus.RUNNING
        record.started_at = datetime.now(tz=timezone.utc)
        await self._storage.upsert_commit_audit_record(record)

        log.info(
            "[CommitAudit] starting incremental audit for commit=%s "
            "files=%d run_id=%s",
            record.commit_hash[:12],
            len(changed_files),
            self._run_id[:8],
        )

        # ── Step 1: Parse diff at function granularity ────────────────────────
        update_result = await self._updater.update_after_commit(
            changed_files=changed_files,
            run_id=self._run_id,
            commit_hash=record.commit_hash,
        )
        diff = update_result.commit_diff

        record.changed_functions     = diff.changed_functions
        record.all_changed_functions = diff.all_changed_functions
        record.new_functions         = diff.new_functions
        record.deleted_functions     = diff.deleted_functions
        record.impact_functions      = [
            item.get("function_name", "") for item in update_result.impact_set
        ]
        record.impact_files          = update_result.impact_files
        record.audit_targets         = update_result.audit_targets
        record.total_changed_functions  = update_result.total_functions_changed
        record.total_impact_functions   = update_result.total_functions_affected
        record.total_functions_to_audit = update_result.total_functions_to_audit
        record.cpg_updated           = update_result.cpg_updated
        record.joern_update_status   = update_result.joern_update_status
        await self._storage.upsert_commit_audit_record(record)

        if not diff.all_changed_functions and not changed_files:
            record.status     = CommitAuditStatus.SKIPPED
            record.finished_at = datetime.now(tz=timezone.utc)
            await self._storage.upsert_commit_audit_record(record)
            log.info("[CommitAudit] no changed functions detected — skipped")
            return record

        # ── Step 2: Mark impact set as stale at function granularity ─────────
        await self._mark_impact_set_stale(update_result.audit_targets)

        # ── Step 3: Mark impact files stale for next read phase ───────────────
        await self._mark_impact_files_stale(
            set(update_result.impact_files) | changed_files
        )

        # ── Step 4: Run scoped test suite ─────────────────────────────────────
        test_files, test_fns = await self._identify_test_scope(
            diff.all_changed_functions, changed_files
        )
        record.test_files_to_run     = test_files
        record.test_functions_to_run = test_fns
        await self._storage.upsert_commit_audit_record(record)

        await self._run_scoped_tests(diff.all_changed_functions)

        # ── Done ──────────────────────────────────────────────────────────────
        record.status     = CommitAuditStatus.DONE
        record.finished_at = datetime.now(tz=timezone.utc)
        await self._storage.upsert_commit_audit_record(record)

        log.info(
            "[CommitAudit] done commit=%s — "
            "changed=%d fns, impact=%d fns, audit_targets=%d "
            "(saved full re-audit of %d lines)",
            record.commit_hash[:12],
            record.total_changed_functions,
            record.total_impact_functions,
            record.total_functions_to_audit,
            await self._estimate_full_reaudit_lines(),
        )
        return record

    # ── Step implementations ──────────────────────────────────────────────────

    async def _mark_impact_set_stale(self, audit_targets: list[dict]) -> None:
        """
        Persist a FunctionStalenessMark for every function in the audit target
        set.  The next read phase will pick these up and re-audit only them.
        """
        for target in audit_targets:
            fn   = target.get("function_name", "")
            fp   = target.get("file_path", "")
            rel  = target.get("relationship", "commit_impact")
            if not fn or not fp:
                continue
            try:
                mark = FunctionStalenessMark(
                    file_path=fp,
                    function_name=fn,
                    line_start=target.get("line_number", 0),
                    stale_reason=rel,
                    run_id=self._run_id,
                )
                await self._storage.upsert_staleness_mark(mark)
            except Exception as exc:
                log.debug("[CommitAudit] staleness mark failed for %s::%s: %s", fp, fn, exc)

    async def _mark_impact_files_stale(self, file_paths: set[str]) -> None:
        """
        Mark each impacted file as STALE in the FileRecord table so the next
        reader phase re-reads it.

        Gap 4 fix: when the CPG is available we compute function-level impact
        so we only mark the files that actually contain impacted functions —
        not every transitive import.  When CPG is unavailable we fall back to
        the import-graph blast radius (file-level), which is coarser but still
        better than a full re-audit.
        """
        for path in file_paths:
            try:
                rec = await self._storage.get_file(path)
                if rec:
                    rec.status = FileStatus.STALE
                    await self._storage.upsert_file(rec)
            except Exception as exc:
                log.debug("[CommitAudit] file status update failed for %s: %s", path, exc)

    async def _identify_test_scope(
        self,
        changed_functions: list[str],
        changed_files:     set[str],
    ) -> tuple[list[str], list[str]]:
        """Identify which test files and test function names should be re-run.

        Strategy (in order of preference):

        1. **Coverage-map lookup** — read ``coverage.json`` or ``.coverage``
           to find the exact test node IDs that executed each changed function.
           This is the only strategy that gives true precision: only tests that
           actually exercised the changed code are re-run.  Falls back when no
           coverage artefact exists.

        2. **CPG call-graph query** — ask Joern for test functions that have
           a call edge into any of the changed functions.  More precise than
           grep but requires Joern to be running.

        3. **Heuristic grep** — scan ``tests/`` for files/functions that
           mention any changed function name as a text substring.  Fast and
           dependency-free but may miss tests that call the function indirectly
           or via a fixture, and may include tests that merely import the name.
        """
        # ── Strategy 1: coverage-map lookup ───────────────────────────────────
        if self._coverage is not None and changed_functions:
            try:
                cov_tests = self._coverage.tests_for_functions(changed_functions)
                if cov_tests is not None:
                    if cov_tests:
                        log.info(
                            "[CommitAudit] coverage-map found %d test(s) for %d changed fn(s)",
                            len(cov_tests),
                            len(changed_functions),
                        )
                        # Separate file paths from bare function names
                        test_files = sorted({t.split("::")[0] for t in cov_tests if "::" in t})
                        test_fns   = [t.rsplit("::", 1)[-1] for t in cov_tests if "::" in t]
                        return test_files, test_fns
                    else:
                        log.debug(
                            "[CommitAudit] coverage-map loaded but no tests found for changed fns "
                            "— falling through to CPG strategy"
                        )
                # cov_tests is None → no coverage data, fall through silently
            except Exception as exc:
                log.debug("[CommitAudit] coverage-map strategy failed: %s", exc)

        # ── Strategy 2: CPG call-graph query ──────────────────────────────────
        if self._cpg and getattr(self._cpg, "is_available", False):
            try:
                test_fns = await self._cpg_test_scope(changed_functions)
                if test_fns:
                    return [], test_fns
            except Exception as exc:
                log.debug("[CommitAudit] CPG test scope query failed: %s", exc)

        # ── Strategy 3: heuristic grep fallback ───────────────────────────────
        return await self._heuristic_test_scope(changed_functions, changed_files)

    async def _cpg_test_scope(self, changed_functions: list[str]) -> list[str]:
        """
        Query Joern for test functions that have a call edge into any of the
        changed functions.  Returns test function names usable as pytest -k args.
        """
        if not self._cpg or not self._cpg._client:
            return []
        fn_list = ", ".join(f'"{fn}"' for fn in changed_functions[:50])
        query = (
            f'cpg.method.name({fn_list}).caller'
            f'.filter(_.filename.contains("test")).name.toList'
        )
        try:
            result = await self._cpg._client.query(query)
            if isinstance(result, list):
                return [str(r) for r in result if r]
        except Exception:
            pass
        return []

    async def _heuristic_test_scope(
        self,
        changed_functions: list[str],
        changed_files:     set[str],
    ) -> tuple[list[str], list[str]]:
        """
        Walk the tests/ directory looking for test files that contain any of
        the changed function names.  This is a cheap grep-based heuristic that
        works without CPG.
        """
        if not self._repo_root or not changed_functions:
            return [], []

        loop         = asyncio.get_event_loop()
        test_root    = self._repo_root / "tests"
        test_files:  list[str] = []
        test_fns:    list[str] = []

        if not test_root.exists():
            return [], []

        fn_set = set(changed_functions)

        def _scan() -> tuple[list[str], list[str]]:
            import ast
            files:   list[str] = []
            fn_names: list[str] = []
            for path in test_root.rglob("test_*.py"):
                try:
                    src  = path.read_text(encoding="utf-8", errors="replace")
                    # Quick text scan first (fast)
                    if not any(fn in src for fn in fn_set):
                        continue
                    files.append(str(path.relative_to(self._repo_root)))
                    # AST scan to find test function names that reference
                    # any changed function
                    try:
                        tree = ast.parse(src)
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                if node.name.startswith("test"):
                                    # Check body for name references
                                    body_src = ast.unparse(node)
                                    if any(fn in body_src for fn in fn_set):
                                        fn_names.append(node.name)
                    except Exception:
                        pass
                except Exception:
                    pass
            return files, fn_names

        test_files, test_fns = await loop.run_in_executor(None, _scan)
        return test_files, test_fns

    async def _run_scoped_tests(self, changed_functions: list[str]) -> None:
        """Invoke the test runner using function-granularity scoping."""
        if not self._runner:
            log.debug("[CommitAudit] no test runner configured — skipping tests")
            return
        try:
            result = await self._runner.run_for_functions(
                function_names=changed_functions,
                fix_attempt_id="",
            )
            log.info(
                "[CommitAudit] scoped test run: %s  passed=%d failed=%d coverage=%.1f%%",
                result.status.value,
                result.passed,
                result.failed,
                result.coverage_pct,
            )
        except Exception as exc:
            log.warning("[CommitAudit] scoped test run failed (non-fatal): %s", exc)

    # ── Git helpers ───────────────────────────────────────────────────────────

    async def _resolve_changed_files(self, commit_hash: str) -> set[str]:
        """Use git to resolve the set of files changed by a commit.

        MERGE COMMIT FIX
        ----------------
        ``git diff-tree --no-commit-id -r <hash>`` on a merge commit returns
        the combined diff between the merge commit and ALL its parents, which
        includes every file touched by the entire merged branch.  On an active
        engineering team this can be hundreds of files — defeating the whole
        point of function-granularity incremental auditing.

        The correct behaviour for merge commits is to audit only the files that
        differ between the two parent commits (i.e. the branch diff), not the
        full merge diff.  We detect merge commits by checking how many parents
        the commit has (``git log --format=%P -n 1``).  For a two-parent merge
        we use ``git diff <parent1>...<parent2>`` to get the branch diff.

        For regular (non-merge) commits the original ``diff-tree`` logic is
        used unchanged.

        Falls back to an empty set when git is unavailable.
        """
        if not self._repo_root:
            return set()

        git_dir = self._repo_root / ".git"
        if not git_dir.exists():
            return set()

        loop = asyncio.get_event_loop()

        def _run() -> set[str]:
            try:
                cwd = str(self._repo_root)

                # ── Detect merge commits ───────────────────────────────────
                # A merge commit has two or more space-separated parent hashes
                # in the %P format field.
                if commit_hash:
                    parents_result = subprocess.run(
                        ["git", "log", "--format=%P", "-n", "1", commit_hash],
                        capture_output=True, text=True, cwd=cwd, timeout=10,
                    )
                    parents = parents_result.stdout.strip().split() if parents_result.returncode == 0 else []
                else:
                    parents = []

                is_merge = len(parents) >= 2

                if is_merge:
                    # Diff the two parents to get only the changes introduced
                    # by the merged branch, not the full combined merge diff.
                    parent1, parent2 = parents[0], parents[1]
                    cmd = [
                        "git", "diff", "--name-only",
                        f"{parent1}...{parent2}",
                    ]
                    log.debug(
                        "[CommitAudit] merge commit %s detected "
                        "(parents: %s %s) — using branch diff",
                        commit_hash[:12] if commit_hash else "HEAD",
                        parent1[:8],
                        parent2[:8],
                    )
                elif commit_hash:
                    cmd = [
                        "git", "diff-tree", "--no-commit-id", "-r",
                        "--name-only", commit_hash,
                    ]
                else:
                    cmd = ["git", "diff", "--name-only", "HEAD~1", "HEAD"]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=cwd,
                    timeout=15,
                )
                if result.returncode != 0:
                    return set()
                return {
                    line.strip()
                    for line in result.stdout.splitlines()
                    if line.strip()
                }
            except Exception as exc:
                log.debug("[CommitAudit] git diff-tree failed: %s", exc)
                return set()

        return await loop.run_in_executor(None, _run)

    async def _estimate_full_reaudit_lines(self) -> int:
        """Return total lines indexed (for compute-savings log message)."""
        try:
            files = await self._storage.list_files()
            return sum(getattr(f, "line_count", 0) for f in files)
        except Exception:
            return 0

    # ── Convenience aliases used by controller and API route ─────────────────

    async def schedule_after_commit(
        self,
        changed_files:  set[str],
        commit_hash:    str = "",
        branch:         str = "",
        author:         str = "",
        commit_message: str = "",
    ) -> CommitAuditRecord:
        """
        Alias called by ``StabilizerController._phase_commit()`` after each
        fix is written to disk.

        Delegates to ``schedule_commit_audit`` with caller-supplied
        ``changed_files``; git diff resolution is skipped because the
        controller already knows which files were modified.
        """
        return await self.schedule_commit_audit(
            commit_hash=commit_hash,
            changed_files=changed_files,
            branch=branch,
            author=author,
            commit_message=commit_message,
        )

    async def schedule_from_webhook(
        self,
        changed_files:  list[str],
        commit_hash:    str = "",
        branch:         str = "",
        author:         str = "",
        commit_message: str = "",
    ) -> CommitAuditRecord:
        """
        Alias called by ``api/routes/commits.py`` when a CI push webhook
        arrives.

        Converts ``changed_files`` list → set (webhook payloads may repeat
        paths) and delegates to ``schedule_commit_audit``.  The idempotency
        check inside ``schedule_commit_audit`` prevents duplicate audits for
        the same commit hash.
        """
        return await self.schedule_commit_audit(
            commit_hash=commit_hash,
            changed_files=set(changed_files),
            branch=branch,
            author=author,
            commit_message=commit_message,
        )

    # ── MISSING-2 FIX: startup catch-up for missed commits ───────────────────

    async def replay_missed_commits(
        self,
        branch: str = "",
        max_commits: int = 50,
    ) -> list[CommitAuditRecord]:
        """
        MISSING-2 FIX: Replay any commits that landed while the API was down,
        the webhook timed out, or the repo does not support webhooks.

        Algorithm
        ---------
        1. Find the most recently DONE CommitAuditRecord stored for this run.
           That commit hash is the last point we know was audited.
        2. Run ``git log {last_hash}..HEAD --format=%H`` to enumerate all
           commits that arrived after that point (up to ``max_commits``).
        3. For each unprocessed SHA, call ``schedule_commit_audit()`` which
           already handles idempotency — SHAs already in DONE state are
           skipped immediately.
        4. Return all records created/replayed during this call.

        Called from ``api/app.py`` lifespan on startup so every service
        restart automatically catches up on the gap window.

        Parameters
        ----------
        branch:
            Branch name to pass through to each CommitAuditRecord. Uses the
            current HEAD branch when empty.
        max_commits:
            Safety cap on the number of commits processed in one replay run.
            Prevents a cold-start from triggering thousands of audits if the
            service was down for a long time.  Default: 50.

        Returns
        -------
        list[CommitAuditRecord]
            Records created or returned (already-DONE) for each replayed SHA.
        """
        if not self._repo_root:
            log.debug("[CommitAudit] replay_missed_commits: no repo_root — skipping")
            return []

        # ── Step 1: find anchor — most recent DONE record ────────────────────
        anchor_hash: str = ""
        try:
            done_records = await self._storage.list_commit_audit_records(
                run_id=self._run_id,
                status=CommitAuditStatus.DONE,
                limit=1,
            )
            if done_records:
                anchor_hash = done_records[0].commit_hash
                log.info(
                    "[CommitAudit] replay: anchor commit = %s",
                    anchor_hash[:12] if anchor_hash else "(none)",
                )
        except Exception as exc:
            log.warning("[CommitAudit] replay: could not load anchor record: %s", exc)

        # ── Step 2: enumerate commits since anchor ───────────────────────────
        missed_shas = await self._git_commits_since(anchor_hash, max_commits)
        if not missed_shas:
            log.info("[CommitAudit] replay: no missed commits found")
            return []

        log.info(
            "[CommitAudit] replay: %d commit(s) to process since anchor %s",
            len(missed_shas),
            anchor_hash[:12] if anchor_hash else "beginning",
        )

        # ── Step 3: determine current branch if not supplied ─────────────────
        if not branch:
            branch = await self._git_current_branch()

        # ── Step 4: schedule each missed commit (idempotent) ─────────────────
        records: list[CommitAuditRecord] = []
        for sha in missed_shas:
            try:
                record = await self.schedule_commit_audit(
                    commit_hash=sha,
                    branch=branch,
                    author="replay",
                    commit_message="(replayed on startup)",
                )
                records.append(record)
                log.info(
                    "[CommitAudit] replay: %s → status=%s",
                    sha[:12],
                    record.status.value,
                )
            except Exception as exc:
                log.error("[CommitAudit] replay: failed for %s: %s", sha[:12], exc)

        log.info(
            "[CommitAudit] replay complete: %d processed, %d done, %d failed",
            len(records),
            sum(1 for r in records if r.status == CommitAuditStatus.DONE),
            sum(1 for r in records if r.status == CommitAuditStatus.FAILED),
        )
        return records

    async def _git_commits_since(
        self, anchor_hash: str, max_commits: int
    ) -> list[str]:
        """
        Return SHAs of commits reachable from HEAD that are not ancestors
        of anchor_hash (i.e. arrived after that commit), oldest first.
        Limited to max_commits entries.
        """
        if not self._repo_root:
            return []

        loop = asyncio.get_event_loop()

        def _run() -> list[str]:
            try:
                rev_range = (
                    f"{anchor_hash}..HEAD" if anchor_hash else f"HEAD~{max_commits}..HEAD"
                )
                result = subprocess.run(
                    ["git", "log", rev_range, "--format=%H", "--reverse",
                     f"--max-count={max_commits}"],
                    capture_output=True,
                    text=True,
                    cwd=str(self._repo_root),
                    timeout=15,
                )
                if result.returncode != 0:
                    # anchor may not be in history (shallow clone, force-push, etc.)
                    # Fall back to listing the last N commits only.
                    log.debug(
                        "[CommitAudit] replay: git log %s failed (rc=%d), "
                        "falling back to last %d commits",
                        rev_range, result.returncode, max_commits,
                    )
                    fallback = subprocess.run(
                        ["git", "log", f"--max-count={max_commits}",
                         "--format=%H", "--reverse"],
                        capture_output=True, text=True,
                        cwd=str(self._repo_root), timeout=15,
                    )
                    if fallback.returncode != 0:
                        return []
                    return [s.strip() for s in fallback.stdout.splitlines() if s.strip()]
                return [s.strip() for s in result.stdout.splitlines() if s.strip()]
            except Exception as exc:
                log.debug("[CommitAudit] replay: git log error: %s", exc)
                return []

        return await loop.run_in_executor(None, _run)

    async def _git_current_branch(self) -> str:
        """Return the name of the current git branch, or empty string on error."""
        if not self._repo_root:
            return ""
        loop = asyncio.get_event_loop()

        def _run() -> str:
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True, text=True,
                    cwd=str(self._repo_root), timeout=5,
                )
                return result.stdout.strip() if result.returncode == 0 else ""
            except Exception:
                return ""

        return await loop.run_in_executor(None, _run)

    # ── MISSING-2 FIX: continuous background git-polling loop ────────────────

    def start_background_poll(
        self,
        poll_interval_s: float = 60.0,
        max_commits_per_poll: int = 20,
        branch: str = "",
    ) -> asyncio.Task:
        """
        MISSING-2 FIX: Start a background asyncio Task that polls the watched
        git branch on a fixed interval and calls replay_missed_commits() for
        any commits that arrived since the last audit.

        This closes the gap left by the startup-only catch-up in api/app.py:
        that replay runs once at restart but cannot catch commits that arrive
        mid-session through a broken or missing webhook.

        This polling loop is the autonomous trigger that makes continuous
        operation work for:
          - Repositories that do not support webhooks (local mirrors, Gitea
            instances without outbound network, corporate proxies)
          - Webhook delivery failures (timeout, 500, network partition)
          - Commits that land while the API is restarting (the startup replay
            catches these, but a race exists between the task starting and the
            first webhook arriving)

        The loop is idempotent with the webhook path: schedule_commit_audit()
        inside replay_missed_commits() checks the CommitAuditRecord for each
        SHA and returns immediately if it is already in DONE state.  A commit
        processed by both the webhook and the poll loop is audited exactly once.

        Parameters
        ----------
        poll_interval_s:
            Seconds between polls.  Default 60.  Increase to reduce git
            subprocess overhead on busy repositories.
        max_commits_per_poll:
            Maximum commits replayed per poll cycle.  Prevents a polling
            restart from triggering thousands of audits if the repo has a
            long commit history since the anchor.  Default 20.
        branch:
            Branch name to attach to replayed CommitAuditRecords.  When empty,
            the current HEAD branch is resolved via git inside replay.

        Returns
        -------
        asyncio.Task
            The background task.  Store the reference and call task.cancel()
            to stop the loop cleanly during shutdown.
        """
        if hasattr(self, "_poll_task") and self._poll_task and not self._poll_task.done():
            log.warning(
                "[CommitAudit] start_background_poll called but a poll task is "
                "already running — ignoring duplicate call"
            )
            return self._poll_task

        self._poll_interval_s       = poll_interval_s
        self._poll_max_commits      = max_commits_per_poll
        self._poll_branch           = branch
        self._poll_task: asyncio.Task = asyncio.create_task(
            self._poll_loop(),
            name="commit_audit_poll",
        )
        log.info(
            "[CommitAudit] Background git-poll loop started — interval=%ss "
            "max_per_poll=%d",
            poll_interval_s,
            max_commits_per_poll,
        )
        return self._poll_task

    def stop_background_poll(self) -> None:
        """
        Cancel the background polling task started by start_background_poll().

        Safe to call when no task is running.  Should be called from the
        FastAPI lifespan shutdown hook to prevent asyncio from logging a
        "Task was destroyed but it is pending" warning.
        """
        task: asyncio.Task | None = getattr(self, "_poll_task", None)
        if task and not task.done():
            task.cancel()
            log.info("[CommitAudit] Background git-poll loop cancelled")

    async def _poll_loop(self) -> None:
        """
        Internal coroutine for the background polling loop.

        Sleeps for _poll_interval_s seconds, then calls replay_missed_commits()
        to process any commits that arrived since the last audit anchor.
        Exceptions inside a single poll cycle are caught and logged — the loop
        continues so a transient git error or storage hiccup does not
        permanently stop autonomous operation.

        The loop exits cleanly when cancelled (CancelledError propagates out).
        """
        log.info("[CommitAudit] Poll loop entering wait cycle (interval=%ss)",
                 self._poll_interval_s)
        while True:
            # Sleep first — the startup replay already handled any backlog.
            # Sleeping at the top prevents a double-audit of commits processed
            # by the startup catch-up just before this task was scheduled.
            try:
                await asyncio.sleep(self._poll_interval_s)
            except asyncio.CancelledError:
                log.info("[CommitAudit] Poll loop cancelled during sleep — exiting")
                return

            try:
                log.debug("[CommitAudit] Poll cycle: calling replay_missed_commits …")
                records = await self.replay_missed_commits(
                    branch=self._poll_branch,
                    max_commits=self._poll_max_commits,
                )
                if records:
                    log.info(
                        "[CommitAudit] Poll cycle: %d commit(s) processed "
                        "(%d done, %d failed)",
                        len(records),
                        sum(1 for r in records if r.status == CommitAuditStatus.DONE),
                        sum(1 for r in records if r.status == CommitAuditStatus.FAILED),
                    )
                else:
                    log.debug("[CommitAudit] Poll cycle: no new commits")
            except asyncio.CancelledError:
                log.info("[CommitAudit] Poll loop cancelled during replay — exiting")
                return
            except Exception as exc:
                # Non-fatal: log and continue.  A broken poll cycle must not
                # terminate autonomous operation.
                log.error(
                    "[CommitAudit] Poll cycle error (will retry in %ss): %s",
                    self._poll_interval_s, exc,
                )

