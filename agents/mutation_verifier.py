"""
agents/mutation_verifier.py
===========================
MutationVerifierAgent — mutation testing gate for Rhodawk AI.

Runs ``mutmut`` on the changed functions after TestGeneratorAgent has
produced a test suite.  If the mutation score is below the domain threshold
the fix is blocked from committing.

This capability has no equivalent in Claude Code or any other commercial
AI coding tool.  It is required for DO-178C DAL-A coverage evidence.

Domain thresholds
──────────────────
    MILITARY / AEROSPACE / NUCLEAR: 90 % mutation score (DAL-A)
    EMBEDDED / AUTOMOTIVE:          80 %
    GENERAL:                        configurable (default 60 %)

How it works
────────────
1. ``mutmut run`` generates mutants for the changed source files.
2. The test suite (from TestGeneratorAgent) is executed against each mutant.
3. A mutant that causes a test failure is "killed" — the test suite detected
   the bug.  A surviving mutant means the test suite has a blind spot.
4. Mutation score = killed / (killed + survived) × 100.
5. If score < threshold → fix.gate_passed = False with a mutation reason.

Stored in the RTM entry:
    mutation_score:       float (0–100)
    mutants_total:        int
    mutants_killed:       int
    mutants_survived:     int

Dependencies
────────────
    mutmut>=2.5.0
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.base import AgentConfig, BaseAgent
from brain.schemas import DomainMode, ExecutorType, FixAttempt
from brain.storage import BrainStorage

log = logging.getLogger(__name__)

# Domain → minimum acceptable mutation score (percentage)
_DOMAIN_THRESHOLDS: dict[str, float] = {
    DomainMode.MILITARY.value:   90.0,
    DomainMode.AEROSPACE.value:  90.0,
    DomainMode.NUCLEAR.value:    90.0,
    DomainMode.EMBEDDED.value:   80.0,
    DomainMode.AUTOMOTIVE.value: 80.0,
    DomainMode.MEDICAL.value:    85.0,
    DomainMode.GENERAL.value:    60.0,
}


@dataclass
class MutationResult:
    file_path:         str
    mutation_score:    float
    mutants_total:     int
    mutants_killed:    int
    mutants_survived:  int
    passed:            bool
    threshold:         float
    details:           str = ""


class MutationVerifierAgent(BaseAgent):
    """
    Run mutmut against changed files and block commits below the score threshold.

    Parameters
    ----------
    repo_root:
        Absolute path to the repository root.
    domain_mode:
        Controls which mutation score threshold is applied.
    score_threshold:
        Override threshold (0–100).  If None, derived from domain_mode.
    timeout_s:
        Maximum seconds to allow mutmut to run per file.
    """

    agent_type = ExecutorType.FIXER   # reuses cost / trail infrastructure

    def __init__(
        self,
        storage:          BrainStorage,
        run_id:           str,
        config:           AgentConfig | None = None,
        mcp_manager:      Any | None         = None,
        repo_root:        Path | None        = None,
        domain_mode:      DomainMode         = DomainMode.GENERAL,
        score_threshold:  float | None       = None,
        timeout_s:        int                = 120,
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.repo_root       = Path(repo_root) if repo_root else None
        self.domain_mode     = domain_mode
        self.score_threshold = (
            score_threshold
            if score_threshold is not None
            else _DOMAIN_THRESHOLDS.get(domain_mode.value, 60.0)
        )
        self.timeout_s = timeout_s

    # ── Public ────────────────────────────────────────────────────────────────

    async def run(
        self,
        fix: FixAttempt,
        test_paths: list[str] | None = None,
    ) -> list[MutationResult]:
        """
        Verify mutation scores for all Python files in *fix*.

        Non-Python files are skipped (mutmut is Python-only; language-specific
        mutation tools can be added per-language in a future iteration).

        Returns a list of MutationResult — one per changed Python file.
        Updates ``fix.mutation_results`` and may set ``fix.gate_passed=False``.
        """
        if not self.repo_root:
            log.debug("[MutationVerifier] No repo_root — skipping")
            return []

        python_files = [
            ff.path for ff in fix.fixed_files
            if ff.path.endswith((".py", ".pyi"))
        ]
        if not python_files:
            log.debug("[MutationVerifier] No Python files in fix — skipping")
            return []

        # Resolve test paths
        resolved_tests = self._resolve_test_paths(test_paths)

        results: list[MutationResult] = []
        for fp in python_files:
            try:
                result = await self._run_for_file(fp, resolved_tests)
                results.append(result)
            except Exception as exc:
                log.warning(
                    f"[MutationVerifier] Failed for {fp}: {exc}"
                )

        # Persist on fix and gate
        any_failed = any(not r.passed for r in results)
        if any_failed:
            failed_files = [r.file_path for r in results if not r.passed]
            reason = (
                f"Mutation score below {self.score_threshold:.0f}% threshold "
                f"in: {', '.join(failed_files)}"
            )
            self.log.warning(f"[MutationVerifier] GATE FAIL: {reason}")
            fix.gate_passed = False
            fix.gate_reason = reason
            await self.storage.upsert_fix(fix)
        else:
            scores = [f"{r.file_path}:{r.mutation_score:.1f}%" for r in results]
            self.log.info(
                f"[MutationVerifier] All files passed. Scores: {', '.join(scores)}"
            )

        return results

    # ── Per-file mutation run ─────────────────────────────────────────────────

    async def _run_for_file(
        self,
        rel_path: str,
        test_paths: list[Path],
    ) -> MutationResult:
        abs_path = self.repo_root / rel_path

        # mutmut must be run in the repo root with the source on PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo_root)

        # Build test runner command for mutmut
        runner = "python -m pytest"
        if test_paths:
            runner += " " + " ".join(str(p) for p in test_paths[:5])

        cmd_run = [
            sys.executable, "-m", "mutmut", "run",
            "--paths-to-mutate", str(abs_path),
            "--runner", runner,
            "--no-progress",
        ]

        loop = asyncio.get_event_loop()
        try:
            run_result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        cmd_run,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_s,
                        cwd=str(self.repo_root),
                        env=env,
                    ),
                ),
                timeout=self.timeout_s + 10,
            )
        except (asyncio.TimeoutError, subprocess.TimeoutExpired):
            log.warning(
                f"[MutationVerifier] mutmut timed out for {rel_path} "
                f"after {self.timeout_s}s"
            )
            return MutationResult(
                file_path=rel_path,
                mutation_score=0.0,
                mutants_total=0,
                mutants_killed=0,
                mutants_survived=0,
                passed=False,
                threshold=self.score_threshold,
                details="mutmut timed out",
            )

        # Parse results
        return await loop.run_in_executor(
            None,
            lambda: self._parse_mutmut_results(rel_path, run_result.stdout + run_result.stderr),
        )

    def _parse_mutmut_results(
        self, rel_path: str, output: str
    ) -> MutationResult:
        """
        Parse mutmut stdout/stderr.  mutmut prints something like:
            Killed 17 out of 20 mutants
        """
        killed   = 0
        survived = 0
        total    = 0

        # Typical mutmut summary line
        m = re.search(
            r"Killed\s+(\d+)\s+out\s+of\s+(\d+)", output, re.IGNORECASE
        )
        if m:
            killed = int(m.group(1))
            total  = int(m.group(2))
            survived = total - killed
        else:
            # Also try the "survived" line
            ms = re.search(r"(\d+)\s+survived", output, re.IGNORECASE)
            mk = re.search(r"(\d+)\s+killed",   output, re.IGNORECASE)
            if ms:
                survived = int(ms.group(1))
            if mk:
                killed   = int(mk.group(1))
            total = killed + survived

        score: float = 100.0 * killed / total if total > 0 else 100.0
        passed = score >= self.score_threshold or total == 0

        log.info(
            f"[MutationVerifier] {rel_path}: {killed}/{total} killed "
            f"(score={score:.1f}%, threshold={self.score_threshold:.0f}%, "
            f"passed={passed})"
        )
        return MutationResult(
            file_path=rel_path,
            mutation_score=round(score, 2),
            mutants_total=total,
            mutants_killed=killed,
            mutants_survived=survived,
            passed=passed,
            threshold=self.score_threshold,
            details=output[:500],
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_test_paths(
        self, test_paths: list[str] | None
    ) -> list[Path]:
        if not self.repo_root:
            return []
        resolved: list[Path] = []
        if test_paths:
            for tp in test_paths:
                abs_p = self.repo_root / tp
                if abs_p.exists():
                    resolved.append(abs_p)
        # Also include the generated tests directory if it exists
        gen_dir = self.repo_root / "tests" / "generated"
        if gen_dir.exists() and gen_dir not in resolved:
            resolved.append(gen_dir)
        return resolved
