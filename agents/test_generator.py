"""
agents/test_generator.py
========================
TestGeneratorAgent — automated test generation for Rhodawk AI.

Sits between FixerAgent and ReviewerAgent in the pipeline.  For each
approved fix, it generates unit tests targeting the changed functions and
stores them alongside the fix in the brain.

Two complementary engines
──────────────────────────
1. **pynguin** — search-based automatic test generation.  Given a Python
   module and a target function, pynguin generates a test suite that
   achieves high branch coverage automatically.  Supports export in
   pytest format.

2. **Hypothesis** — property-based testing driven by the LLM.  The model
   describes invariants / properties; Hypothesis searches for
   counterexamples.  Works for Python; for other languages the LLM
   generates hand-written tests instead.

For non-Python files (C, C++, JS, Go, …) the agent falls back to LLM-only
test generation using the existing call_llm_structured infrastructure.

Output
──────
Generated tests are written to ``<repo_root>/tests/generated/`` and their
paths + coverage data are stored in ``FixAttempt.generated_test_paths`` (a
new field we add to the schema in this PR).  The RTM entry for each fixed
function is updated with the test case IDs so MC/DC coverage evidence is
traceable.

Dependencies
────────────
    pynguin>=0.36.0
    hypothesis>=6.100.0
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agents.base import AgentConfig, BaseAgent
from brain.schemas import ExecutorType, FixAttempt, FixedFile
from brain.storage import BrainStorage

log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 60      # pynguin budget per module
_MAX_FUNCTIONS     = 10      # cap test-gen to top-N changed functions


# ─────────────────────────────────────────────────────────────────────────────
# LLM response model — fallback for non-Python or when pynguin fails
# ─────────────────────────────────────────────────────────────────────────────

class GeneratedTestSuite(BaseModel):
    file_path:       str              = Field(description="Relative path of the file under test")
    test_file_path:  str              = Field(description="Where to write the test file, e.g. tests/generated/test_foo.py")
    test_code:       str              = Field(description="Complete runnable test file content")
    functions_covered: list[str]      = Field(default_factory=list)
    notes:           str              = ""


class GeneratedTestBatch(BaseModel):
    suites: list[GeneratedTestSuite] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class TestGeneratorAgent(BaseAgent):
    """
    Generate unit tests for every function touched by a FixAttempt.

    Parameters
    ----------
    storage, run_id, config, mcp_manager:
        Standard agent constructor args.
    repo_root:
        Absolute path to the repository root.
    pynguin_timeout:
        Budget in seconds for pynguin per module (default 60 s).
    use_hypothesis:
        Whether to emit Hypothesis @given strategies alongside pynguin tests.
    """

    agent_type = ExecutorType.FIXER     # reuses FIXER token budget tracking

    def __init__(
        self,
        storage:          BrainStorage,
        run_id:           str,
        config:           AgentConfig | None = None,
        mcp_manager:      Any | None         = None,
        repo_root:        Path | None        = None,
        pynguin_timeout:  int                = _DEFAULT_TIMEOUT_S,
        use_hypothesis:   bool               = True,
    ) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.repo_root       = Path(repo_root) if repo_root else None
        self.pynguin_timeout = pynguin_timeout
        self.use_hypothesis  = use_hypothesis

    async def run(self, fix: FixAttempt) -> list[str]:
        """
        Generate tests for all files in *fix*.

        Returns the list of generated test file paths (relative to repo_root).
        """
        generated: list[str] = []
        for ff in fix.fixed_files:
            lang = self._detect_language(ff.path)
            try:
                paths = await self._generate_for_file(ff, lang)
                generated.extend(paths)
            except Exception as exc:
                self.log.warning(
                    f"[TestGenerator] Failed for {ff.path}: {exc}"
                )

        # Persist generated test paths on the fix record
        fix.generated_test_paths = generated
        await self.storage.upsert_fix(fix)

        self.log.info(
            f"[TestGenerator] Generated {len(generated)} test files "
            f"for fix {fix.id[:12]}"
        )
        return generated

    async def generate_for_issue(self, issue: Any) -> list[str]:
        """
        Generate a minimal reproduction test for an Issue that has no fail_tests.

        Called by the controller before BoBN scoring so every issue gets at least
        one correctness signal regardless of how it was sourced.

        Returns a list of pytest node IDs / test function names, or [] on failure.
        """
        file_path = getattr(issue, "file_path", "") or ""
        description = getattr(issue, "description", "") or ""
        line_start = getattr(issue, "line_start", 0) or 0

        if not file_path:
            return []

        lang = self._detect_language(file_path)

        # Read source file content if repo_root is available
        content = ""
        if self.repo_root:
            abs_path = self.repo_root / file_path
            try:
                content = abs_path.read_text(encoding="utf-8", errors="replace")
                # Trim to a window around the issue line
                lines = content.splitlines()
                start = max(0, line_start - 20)
                end   = min(len(lines), line_start + 40)
                content = "\n".join(lines[start:end])
            except OSError:
                content = ""

        prompt = (
            f"## Issue\n{description}\n\n"
            f"## File\nPath: {file_path}  Language: {lang}  Line: {line_start}\n"
            + (f"```\n{content}\n```\n\n" if content else "")
            + "## Task\n"
            "Write the minimal pytest test(s) that would FAIL on the buggy code "
            "and PASS after a correct fix. Return only test function names (one per line), "
            "no code. Format: test_<description>"
        )

        class _TestNames(BaseModel):
            names: list[str] = Field(default_factory=list)

        try:
            result = await self.call_llm_structured(
                prompt=prompt,
                response_model=_TestNames,
                system=(
                    "You are a test engineer. Return only a JSON object with "
                    "a 'names' list of pytest function names that reproduce the issue."
                ),
            )
            names = [n.strip() for n in result.names if n.strip().startswith("test_")]
            return names[:5]  # cap at 5 test names per issue
        except Exception as exc:
            self.log.debug("[TestGenerator] generate_for_issue failed for %s: %s", file_path, exc)
            return []

    # ── Per-file dispatch ─────────────────────────────────────────────────────

    async def _generate_for_file(
        self, ff: FixedFile, lang: str
    ) -> list[str]:
        if lang == "python":
            return await self._generate_python(ff)
        else:
            return await self._generate_via_llm(ff, lang)

    # ── Python: pynguin + hypothesis ─────────────────────────────────────────

    async def _generate_python(self, ff: FixedFile) -> list[str]:
        if not self.repo_root:
            return await self._generate_via_llm(ff, "python")

        abs_path = self.repo_root / ff.path
        if not abs_path.exists():
            return await self._generate_via_llm(ff, "python")

        out_dir = self.repo_root / "tests" / "generated"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Try pynguin in a subprocess
        pynguin_result = await self._run_pynguin(abs_path, out_dir)

        if pynguin_result:
            # Optionally augment with Hypothesis strategies via LLM
            if self.use_hypothesis:
                await self._augment_with_hypothesis(ff, pynguin_result, out_dir)
            return [pynguin_result]

        # pynguin failed — fall back to LLM
        return await self._generate_via_llm(ff, "python")

    async def _run_pynguin(
        self, abs_path: Path, out_dir: Path
    ) -> str | None:
        """
        Run pynguin in a subprocess.  Returns the relative path of the
        generated test file or None on failure.
        """
        try:
            import pynguin  # type: ignore — just a version check
        except ImportError:
            log.debug("pynguin not installed — skipping")
            return None

        module_name = abs_path.stem
        cmd = [
            sys.executable, "-m", "pynguin",
            "--project-path", str(abs_path.parent),
            "--module-name", module_name,
            "--output-path", str(out_dir),
            "--maximum-search-time", str(self.pynguin_timeout),
            "--assertion-generation", "MUTATION",
            "-v",
        ]
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=self.pynguin_timeout + 10,
                        cwd=str(abs_path.parent),
                    ),
                ),
                timeout=self.pynguin_timeout + 15,
            )
        except (asyncio.TimeoutError, subprocess.TimeoutExpired):
            log.warning(f"[TestGenerator] pynguin timed out for {abs_path.name}")
            return None

        if result.returncode != 0:
            log.debug(
                f"[TestGenerator] pynguin exit {result.returncode}: "
                f"{result.stderr[:300]}"
            )
            return None

        # pynguin writes test_<module>.py
        candidate = out_dir / f"test_{abs_path.stem}.py"
        if candidate.exists() and self.repo_root:
            try:
                return str(candidate.relative_to(self.repo_root))
            except ValueError:
                return str(candidate)
        return None

    async def _augment_with_hypothesis(
        self,
        ff: FixedFile,
        test_path: str,
        out_dir: Path,
    ) -> None:
        """
        Ask the LLM to add Hypothesis @given tests for the changed functions.
        Appends the generated strategies to the existing pynguin test file.
        """
        content = ff.content or ""
        changed_fns = self._extract_changed_functions(ff)[:_MAX_FUNCTIONS]
        if not changed_fns:
            return

        prompt = (
            "## File Under Test\n"
            f"Path: {ff.path}\n"
            f"```python\n{content[:4000]}\n```\n\n"
            "## Changed Functions\n"
            + "\n".join(f"- {fn}" for fn in changed_fns)
            + "\n\n"
            "## Task\n"
            "Write Hypothesis property-based tests for the changed functions above.\n"
            "Requirements:\n"
            "1. Use `from hypothesis import given, strategies as st`.\n"
            "2. Each test must have a meaningful property docstring.\n"
            "3. Use `@given` decorators with appropriate strategies.\n"
            "4. Do NOT duplicate tests from pynguin — focus on invariants.\n"
            "5. Return ONLY the test code as a Python code block.\n"
        )
        try:
            resp = await self.call_llm_raw(
                prompt=prompt,
                system=(
                    "You are a test engineer specialising in property-based "
                    "testing with Hypothesis.  Return only valid Python code."
                ),
            )
            hyp_code = _extract_code_block(resp)
            if hyp_code and self.repo_root:
                full_path = self.repo_root / test_path
                if full_path.exists():
                    with full_path.open("a", encoding="utf-8") as fh:
                        fh.write(
                            "\n\n# ── Hypothesis property-based tests "
                            "(generated by Antagonist) ──\n"
                        )
                        fh.write(hyp_code)
        except Exception as exc:
            log.debug(f"[TestGenerator] Hypothesis augmentation failed: {exc}")

    # ── LLM-only test generation (non-Python or pynguin fallback) ────────────

    async def _generate_via_llm(
        self, ff: FixedFile, lang: str
    ) -> list[str]:
        content   = ff.content or ""
        patch     = ff.patch   or ""
        changed   = self._extract_changed_functions(ff)[:_MAX_FUNCTIONS]

        prompt = (
            f"## File Under Test\n"
            f"Path: {ff.path}  Language: {lang}\n"
            f"```\n{content[:5000]}\n```\n\n"
            + (f"## Patch Applied\n```diff\n{patch[:2000]}\n```\n\n"
               if patch else "")
            + (f"## Changed Functions\n"
               + "\n".join(f"- {fn}" for fn in changed)
               + "\n\n" if changed else "")
            + "## Task\n"
            "Generate a comprehensive unit test file for the changed functions.\n"
            "Requirements:\n"
            f"1. Use the standard test framework for {lang} "
            f"(pytest for Python, JUnit for Java, googletest for C/C++, "
            f"Jest for JS/TS, go test for Go).\n"
            "2. Cover: happy path, boundary values, error conditions.\n"
            "3. Include at least one test per changed function.\n"
            "4. Do not import internal symbols that are not exported.\n"
            "5. Each test must have a descriptive name explaining what it verifies.\n"
        )

        try:
            batch = await self.call_llm_structured(
                prompt=prompt,
                response_model=GeneratedTestBatch,
                system=(
                    "You are a senior test engineer.  Generate complete, "
                    "runnable test files.  Return structured JSON only."
                ),
            )
        except Exception as exc:
            log.warning(f"[TestGenerator] LLM call failed for {ff.path}: {exc}")
            return []

        written: list[str] = []
        for suite in batch.suites:
            if not suite.test_code.strip():
                continue
            rel_path = suite.test_file_path or self._default_test_path(ff.path, lang)
            if self.repo_root:
                abs_out = self.repo_root / rel_path
                abs_out.parent.mkdir(parents=True, exist_ok=True)
                try:
                    abs_out.write_text(suite.test_code, encoding="utf-8")
                    written.append(rel_path)
                except OSError as exc:
                    log.warning(f"[TestGenerator] Could not write {rel_path}: {exc}")
        return written

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _detect_language(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        return {
            ".py": "python", ".pyi": "python",
            ".c": "c", ".h": "c",
            ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
            ".hpp": "cpp", ".hxx": "cpp",
            ".js": "javascript", ".mjs": "javascript",
            ".ts": "typescript", ".tsx": "typescript",
            ".rs": "rust", ".go": "go",
            ".java": "java",
        }.get(ext, "unknown")

    def _extract_changed_functions(self, ff: FixedFile) -> list[str]:
        """Parse the patch / diff to find which function names were changed."""
        import re
        names: list[str] = []
        text = ff.patch or ""
        # Look for function definitions in added lines
        for line in text.splitlines():
            if not line.startswith("+"):
                continue
            stripped = line[1:].strip()
            m = re.match(
                r"(?:def |async def |function |void |int |char )\s*(\w+)\s*\(",
                stripped,
            )
            if m:
                names.append(m.group(1))
        return list(dict.fromkeys(names))  # deduplicate preserving order

    @staticmethod
    def _default_test_path(file_path: str, lang: str) -> str:
        stem = Path(file_path).stem
        ext  = {
            "python":     ".py",
            "javascript": ".test.js",
            "typescript": ".test.ts",
            "go":         "_test.go",
            "rust":       "_test.rs",
            "java":       "Test.java",
        }.get(lang, ".py")
        return f"tests/generated/test_{stem}{ext}"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_code_block(text: str) -> str:
    """Pull the first ```...``` block from an LLM response."""
    import re
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()
