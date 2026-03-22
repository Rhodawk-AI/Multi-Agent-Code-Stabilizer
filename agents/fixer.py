from __future__ import annotations
import asyncio
import difflib
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field
from agents.base import AgentConfig, BaseAgent, wrap_content
from brain.schemas import BugRecurrenceSignal, ExecutorType, FixAttempt, FixedFile, IssueStatus, PatchMode, RefactorProposal, Severity
from brain.storage import BrainStorage
from verification.independence_enforcer import extract_model_family
try:
    from cpg.context_selector import CPGContextSelector, ContextSlice
    from cpg.program_slicer import ProgramSlicer
    _CPG_AVAILABLE = True
except ImportError:
    _CPG_AVAILABLE = False
log = logging.getLogger(__name__)
_DEFAULT_SURGICAL_THRESHOLD = 2000
_AST_REWRITE_THRESHOLD = 500

# ── Gap 3 proactive architectural smell thresholds ────────────────────────────
_RECURRENCE_ESCALATION_THRESHOLD = 3
_COUPLING_MODULE_THRESHOLD = 5

import re as _re_mod

_NORM_LINE_PATTERNS = [
    _re_mod.compile(r'\bat (line|col|column)\s+\d+\b', _re_mod.IGNORECASE),
    _re_mod.compile(r'(?<!\w)(?:L\s*)?\d{1,6}(?!\w)'),
    _re_mod.compile(r'(?:[a-zA-Z0-9_.\-/\\]+\.(?:py|c|cpp|h|hpp|js|ts|go|rs|java))\s*:?\s*'),
    _re_mod.compile(r'\bin (?:function|method|class)\s+\w+', _re_mod.IGNORECASE),
]

class FixedFileFullResponse(BaseModel):
    path: str
    content: str = Field(description='COMPLETE corrected file content')
    issues_resolved: list[str]
    changes_made: str = Field(description='Bullet-point summary of every change')
    diff_summary: str = Field(description='1-2 sentence plain-English summary')
    confidence: float = Field(ge=0.0, le=1.0, default=0.85)

class FixedFilePatchResponse(BaseModel):
    path: str
    patch: str = Field(description="A valid unified diff patch in the format produced by 'diff -u original.c fixed.c'. Must start with '--- ' and '+++ ' headers. Must apply cleanly with 'patch -p0'.")
    issues_resolved: list[str]
    changes_made: str = Field(description='Bullet-point summary of every change')
    diff_summary: str = Field(description='1-2 sentence plain-English summary')
    confidence: float = Field(ge=0.0, le=1.0, default=0.85)
    lines_changed: int = 0

class FixResponse(BaseModel):
    fixed_files: list[FixedFileFullResponse] = Field(default_factory=list)
    overall_notes: str = ''

class PatchResponse(BaseModel):
    patched_files: list[FixedFilePatchResponse] = Field(default_factory=list)
    overall_notes: str = ''

class FixerAgent(BaseAgent):
    agent_type = ExecutorType.FIXER

    def __init__(self, storage: BrainStorage, run_id: str, config: AgentConfig | None=None, mcp_manager: Any | None=None, repo_root: Path | None=None, graph_engine: Any | None=None, vector_brain: Any | None=None, surgical_patch_threshold: int=_DEFAULT_SURGICAL_THRESHOLD, repo_map: Any | None=None, hybrid_retriever: Any | None=None, fix_memory: Any | None=None, cpg_engine: Any | None=None, cpg_context_selector: Any | None=None, program_slicer: Any | None=None, escalation_manager: Any | None=None, blast_radius_threshold: int=50) -> None:
        super().__init__(storage, run_id, config, mcp_manager)
        self.repo_root = repo_root
        self.graph_engine = graph_engine
        self.vector_brain = vector_brain
        self.surgical_patch_threshold = surgical_patch_threshold
        self.repo_map = repo_map
        self.hybrid_retriever = hybrid_retriever
        self.fix_memory = fix_memory
        self.cpg_engine = cpg_engine
        self.cpg_context_selector = cpg_context_selector
        self.program_slicer = program_slicer
        self.escalation_manager = escalation_manager
        self.blast_radius_threshold = blast_radius_threshold

    @staticmethod
    def _normalize_bug_class(description: str) -> str:
        """Return a location-stripped, lowercased bug-class key for fix_memory."""
        text = description[:200]
        for pat in _NORM_LINE_PATTERNS:
            text = pat.sub(' ', text)
        text = _re_mod.sub(r'\s+', ' ', text).strip().lower()
        return text[:80]

    async def run(self, **kwargs: Any) -> list[FixAttempt]:
        issues = await self.storage.list_issues(run_id=self.run_id, status=IssueStatus.APPROVED.value)
        if not issues:
            issues = await self.storage.list_issues(run_id=self.run_id, status=IssueStatus.OPEN.value)
        groups = self._group_issues(issues)
        batches = await self._non_overlapping_batches(groups)
        created: list[FixAttempt] = []
        for batch in batches:
            batch_results = await asyncio.gather(*[self._fix_group(group_key, group_issues) for group_key, group_issues in batch.items()], return_exceptions=True)
            for result in batch_results:
                if isinstance(result, FixAttempt):
                    created.append(result)
                elif isinstance(result, Exception):
                    self.log.error(f'Fix group failed: {result}')
            await self.check_cost_ceiling()
        return created

    def _group_issues(self, issues) -> dict[frozenset[str], list]:
        groups: dict[frozenset[str], list] = {}
        for issue in issues:
            if issue.fix_attempts >= issue.max_fix_attempts:
                continue
            key = frozenset(issue.fix_requires_files or [issue.file_path])
            groups.setdefault(key, []).append(issue)
        return groups

    async def _non_overlapping_batches(self, groups: dict[frozenset[str], list]) -> list[dict[frozenset[str], list]]:
        group_symbols: dict[frozenset[str], set[str]] = {}
        for key in groups:
            syms = set()
            for fpath in key:
                syms |= await self._extract_file_symbols(fpath)
            group_symbols[key] = syms
        batches: list[dict[frozenset[str], list]] = []
        remaining = dict(groups)
        while remaining:
            batch: dict[frozenset[str], list] = {}
            batch_files: set[str] = set()
            batch_symbols: set[str] = set()
            for key, issues in list(remaining.items()):
                key_files = set(key)
                key_symbols = group_symbols.get(key, set())
                if not key_files & batch_files and (not key_symbols & batch_symbols):
                    batch[key] = issues
                    batch_files |= key_files
                    batch_symbols |= key_symbols
                    del remaining[key]
            if not batch:
                key, issues = next(iter(remaining.items()))
                batch[key] = issues
                del remaining[key]
            batches.append(batch)
        return batches

    async def _extract_file_symbols(self, file_path: str) -> set[str]:
        try:
            from startup.feature_matrix import is_available
            if not is_available('tree_sitter_language_pack'):
                return set()
            content = await self._load_file(file_path)
            if not content:
                return set()
            ext = Path(file_path).suffix.lower()
            lang_map = {'.py': 'python', '.c': 'c', '.h': 'c', '.cpp': 'cpp', '.cc': 'cpp', '.hpp': 'cpp', '.js': 'javascript', '.ts': 'typescript', '.rs': 'rust', '.go': 'go'}
            lang = lang_map.get(ext)
            if not lang:
                return set()
            from tree_sitter_language_pack import get_parser
            parser = get_parser(lang)
            tree = parser.parse(content.encode())
            symbols: set[str] = set()

            def _walk(node) -> None:
                if node.type in {'function_definition', 'function_declaration', 'method_definition', 'function_item', 'class_definition', 'struct_item', 'struct_specifier', 'enum_specifier', 'typedef_declaration'}:
                    for child in node.children:
                        if child.type in {'identifier', 'name', 'field_identifier'}:
                            symbols.add(child.text.decode(errors='replace'))
                for child in node.children:
                    _walk(child)
            _walk(tree.root_node)
            return symbols
        except Exception as exc:
            self.log.debug(f'_extract_file_symbols({file_path}): {exc}')
            return set()

    async def _fix_group(self, file_key: frozenset[str], issues: list) -> FixAttempt:
        file_paths = list(file_key)
        has_critical = any((i.severity == Severity.CRITICAL for i in issues))
        model = self.config.critical_fix_model if has_critical and self.config else self.config.primary_model if self.config else ''
        file_contents: dict[str, str] = {}
        patch_modes: dict[str, PatchMode] = {}
        for fp in file_paths:
            content = await self._load_file(fp)
            file_contents[fp] = content
            line_count = content.count('\n') if content else 0
            patch_modes[fp] = self._select_patch_mode(fp, line_count)
        needs_patch = any((m == PatchMode.UNIFIED_DIFF for m in patch_modes.values()))
        needs_ast = any((m == PatchMode.AST_REWRITE for m in patch_modes.values()))
        issue_summary = '\n'.join((f'- [{i.severity.value}] {i.description} ({i.file_path}:{i.line_start})' for i in issues[:10]))
        repo_map_text = self._get_repo_map_context(file_paths)
        memory_examples = await self._get_memory_examples(issues)

        all_symbols: list[str] = []
        for fp in file_paths:
            syms = await self._extract_file_symbols(fp)
            all_symbols.extend(sorted(syms)[:20])

        recurrence_signal = await self._check_bug_recurrence(
            issues=issues,
            file_paths=file_paths,
            function_names=all_symbols or file_paths,
            window_days=180,
        )
        if recurrence_signal.is_structural:
            stub_blast = None
            if self.cpg_engine and self.cpg_engine.is_available:
                try:
                    stub_blast = await self.cpg_engine.compute_blast_radius(
                        function_names=all_symbols or file_paths,
                        file_paths=file_paths,
                        depth=3,
                    )
                except Exception as exc:
                    self.log.debug(f'[Fixer] recurrence gate blast compute failed: {exc}')

            if stub_blast is None:
                from cpg.cpg_engine import CPGBlastRadius
                stub_blast = CPGBlastRadius(
                    changed_functions=all_symbols or file_paths,
                    blast_radius_score=recurrence_signal.coupling_score
                    if recurrence_signal.coupling_score >= 0 else 0.0,
                    requires_human_review=True,
                )

            has_recurrence = recurrence_signal.recurrence_count >= _RECURRENCE_ESCALATION_THRESHOLD
            has_coupling   = recurrence_signal.distinct_caller_modules >= _COUPLING_MODULE_THRESHOLD
            if has_recurrence and has_coupling:
                trigger = 'combined'
            elif has_recurrence:
                trigger = 'recurrence'
            else:
                trigger = 'coupling_smell'

            self.log.warning(
                f'[Fixer] Gap 3 proactive gate ({trigger}): aborting patch, '
                f'generating refactor proposal. reason={recurrence_signal.escalation_reason}'
            )
            return await self._generate_refactor_proposal(
                issues, file_paths, stub_blast,
                forward_impact_context=recurrence_signal.escalation_reason,
                model=model,
                recurrence_signal=recurrence_signal,
                trigger_source=trigger,
            )

        forward_impact_context, blast = await self._get_forward_impact_context(file_paths, issues)
        if blast is not None and blast.requires_human_review:
            self.log.warning(f'[Fixer] Gap 3 gate: blast_radius={blast.affected_function_count} functions across {blast.affected_file_count} files — aborting patch, generating refactor proposal')
            return await self._generate_refactor_proposal(issues, file_paths, blast, forward_impact_context, model, recurrence_signal=None, trigger_source='blast_radius')
        cpg_context = await self._get_cpg_context(issues)
        file_context = await self._build_file_context(file_paths, file_contents, patch_modes)
        vector_context = await self._get_vector_context(issues)
        MAX_FEEDBACK_ROUNDS = 3
        last_result = None
        last_test_output = ''
        test_passed = False
        for feedback_round in range(1, MAX_FEEDBACK_ROUNDS + 1):
            prompt_extra = ''
            if last_test_output:
                prompt_extra = f'\n\n## Previous Attempt Failed — Test Output\nYour previous fix was applied but the following tests failed.\nAnalyze the failures and produce a corrected patch.\n\n```\n{last_test_output[:3000]}\n```\n'
            if needs_patch:
                last_result = await self._generate_patch_fix(issue_summary + prompt_extra, file_context, vector_context, model, file_paths, repo_map_text=repo_map_text, memory_examples=memory_examples, cpg_context=cpg_context, forward_impact_context=forward_impact_context)
            else:
                last_result = await self._generate_full_fix(issue_summary + prompt_extra, file_context, vector_context, model, file_paths, repo_map_text=repo_map_text, memory_examples=memory_examples, cpg_context=cpg_context, forward_impact_context=forward_impact_context)
            test_passed, last_test_output = await self._probe_candidate(last_result, file_contents, file_paths)
            self.log.info(f'[Fixer] feedback_round={feedback_round}/{MAX_FEEDBACK_ROUNDS} test_passed={test_passed}')
            if test_passed:
                break

        if not test_passed and self.fix_memory and last_result is not None:
            await self._store_failure_memory(issues, file_paths, last_result, last_test_output)
        result = last_result
        if needs_ast and isinstance(result, FixResponse):
            result = await self._apply_ast_rewrites(result, file_contents, patch_modes)
        fixed_files: list[FixedFile] = []
        if isinstance(result, PatchResponse):
            for pfr in result.patched_files:
                original = file_contents.get(pfr.path, '')
                orig_hash = hashlib.sha256(original.encode()).hexdigest()
                ff = FixedFile(path=pfr.path, content='', patch=pfr.patch, patch_mode=PatchMode.UNIFIED_DIFF, changes_made=pfr.changes_made, diff_summary=pfr.diff_summary, confidence=pfr.confidence, original_hash=orig_hash, lines_changed=pfr.lines_changed or pfr.patch.count('\n+'))
                fixed_files.append(ff)
        elif isinstance(result, FixResponse):
            for ffr in result.fixed_files:
                original = file_contents.get(ffr.path, '')
                orig_hash = hashlib.sha256(original.encode()).hexdigest()
                new_hash = hashlib.sha256(ffr.content.encode()).hexdigest()
                if orig_hash == new_hash:
                    self.log.warning(f'Fix for {ffr.path} produced no changes — skipping')
                    continue
                diff = list(difflib.unified_diff(original.splitlines(), ffr.content.splitlines(), lineterm=''))
                pm = patch_modes.get(ffr.path, PatchMode.FULL_FILE)
                ff = FixedFile(path=ffr.path, content=ffr.content, patch='\n'.join(diff), patch_mode=pm, changes_made=ffr.changes_made, diff_summary=ffr.diff_summary, confidence=ffr.confidence, original_hash=orig_hash, patched_hash=new_hash, lines_changed=sum((1 for l in diff if l.startswith(('+', '-')))))
                fixed_files.append(ff)
        fix = FixAttempt(run_id=self.run_id, issue_ids=[i.id for i in issues], fixed_files=fixed_files, fixer_model=model, fixer_model_family=extract_model_family(model), patch_mode=PatchMode.UNIFIED_DIFF if needs_patch else PatchMode.AST_REWRITE if needs_ast else PatchMode.FULL_FILE)
        await self.storage.upsert_fix(fix)
        if fixed_files and self.fix_memory:
            await self._store_fix_memory(issues, fixed_files, fix.id)
        for issue in issues:
            issue.fix_attempts += 1
            issue.status = IssueStatus.FIX_GENERATED
            await self.storage.upsert_issue(issue)
        return fix

    async def _check_bug_recurrence(
        self,
        issues:         list,
        file_paths:     list[str],
        function_names: list[str],
        window_days:    int = 180,
    ) -> BugRecurrenceSignal:
        """Gap 3 — Proactive Architectural Smell Detection.

        BUG-1 FIX: Previously called the sync self.fix_memory.retrieve() at
        line 422 inside an async context (running event loop).  That caused
        _retrieve_federated() to detect loop.is_running() == True and always
        return [] for the federated pull, silently giving zero federated
        augmentation on every recurrence check.

        Fix: always call retrieve_async() and await it so the full federated
        augmentation path executes correctly without the nested-loop workaround.
        """
        signal = BugRecurrenceSignal(
            window_days=window_days,
            coupling_module_threshold=_COUPLING_MODULE_THRESHOLD,
        )

        # ── Signal 1: recurrence check ────────────────────────────────────────
        if self.fix_memory:
            try:
                raw_query = ' '.join(i.description[:100] for i in issues[:3])
                query = self._normalize_bug_class(raw_query)
                # BUG-1 FIX: use retrieve_async() so federated augmentation is
                # live in this async context.  The sync retrieve() detected
                # loop.is_running() == True and silently returned [] for
                # federated patterns on every call from this method.
                if hasattr(self.fix_memory, 'retrieve_async'):
                    entries = await self.fix_memory.retrieve_async(
                        query, n=20, max_age_days=window_days
                    )
                else:
                    entries = self.fix_memory.retrieve(
                        query, n=20, max_age_days=window_days
                    )
                successful = [
                    e for e in entries
                    if not getattr(e, 'fix_approach', '').startswith('[REVERTED]')
                ]
                reverted = [
                    e for e in entries
                    if getattr(e, 'fix_approach', '').startswith('[REVERTED]')
                ]
                signal.recurrence_count = len(successful)
                signal.reverted_count   = len(reverted)

                ctx_counts: dict[str, int] = {}
                for e in entries:
                    fc = getattr(e, 'file_context', '') or ''
                    if fc:
                        ctx_counts[fc] = ctx_counts.get(fc, 0) + 1
                if ctx_counts:
                    signal.dominant_file_context = max(
                        ctx_counts, key=ctx_counts.__getitem__
                    )

                if signal.recurrence_count >= _RECURRENCE_ESCALATION_THRESHOLD:
                    signal.is_structural = True
                    signal.escalation_reason = (
                        f'recurrence: this bug class has been patched '
                        f'{signal.recurrence_count} time(s) in the last '
                        f'{window_days} days'
                        + (
                            f' ({signal.reverted_count} revert(s))'
                            if signal.reverted_count else ''
                        )
                        + f'. Dominant file context: {signal.dominant_file_context!r}.'
                        f' A structural fix is required.'
                    )
                    self.log.warning(
                        f'[Fixer] Gap 3 recurrence gate: '
                        f'recurrence_count={signal.recurrence_count} '
                        f'>= threshold={_RECURRENCE_ESCALATION_THRESHOLD} '
                        f'— escalating to refactor proposal'
                    )
            except Exception as exc:
                self.log.debug(f'_check_bug_recurrence (fix_memory): {exc}')

        # ── Signal 2: CPG coupling smell ──────────────────────────────────────
        if self.cpg_engine and self.cpg_engine.is_available:
            try:
                coupling = await self.cpg_engine.compute_coupling_smell(
                    function_names=function_names,
                    coupling_module_threshold=_COUPLING_MODULE_THRESHOLD,
                )
                signal.coupling_score          = coupling.get('coupling_score', -1.0)
                signal.distinct_caller_modules = coupling.get('distinct_caller_modules', 0)

                if coupling.get('is_smell', False):
                    dominant = coupling.get('dominant_caller_module', '')
                    coupling_reason = (
                        f'coupling_smell: the changed function(s) are called from '
                        f'{signal.distinct_caller_modules} distinct module(s) '
                        f'(threshold={_COUPLING_MODULE_THRESHOLD}), '
                        f'indicating no ownership boundary exists '
                        + (f'(dominant caller module: {dominant!r}). ' if dominant else '. ')
                        + 'A patch is locally correct but structurally unsound.'
                    )
                    if signal.is_structural:
                        signal.escalation_reason += ' | ' + coupling_reason
                    else:
                        signal.is_structural      = True
                        signal.escalation_reason  = coupling_reason
                    self.log.warning(
                        f'[Fixer] Gap 3 coupling smell gate: '
                        f'distinct_modules={signal.distinct_caller_modules} '
                        f'>= threshold={_COUPLING_MODULE_THRESHOLD} '
                        f'coupling_score={signal.coupling_score:.4f} '
                        f'— escalating to refactor proposal'
                    )
            except Exception as exc:
                self.log.debug(f'_check_bug_recurrence (coupling): {exc}')

        return signal

    def _get_repo_map_context(self, target_files: list[str]) -> str:
        if not self.repo_map:
            return ''
        try:
            return self.repo_map.generate(target_files=target_files, max_tokens=2048)
        except Exception as exc:
            self.log.debug(f'_get_repo_map_context: {exc}')
            return ''

    async def _get_cpg_context(self, issues: list) -> str:
        if not (self.cpg_context_selector and issues):
            return ''
        try:
            ctx = await self.cpg_context_selector.select_context_for_issues(issues=issues, max_lines=3000)
            if not ctx.context_text:
                return ''
            # ARCH-1: emit WARNING when CPG fallback is active so operators can
            # see that causal context is degraded rather than silently missing it.
            if ctx.source != 'cpg':
                self.log.warning(
                    '[Fixer] CPG context source=%s (not full Joern CPG) — '
                    'cross-file causal relationships may be missed. '
                    'Start Joern with: docker-compose up joern',
                    ctx.source,
                )
            total_info = f'total_functions={ctx.total_functions} files={ctx.total_files} source={ctx.source}'
            self.log.info(f'[Fixer] CPG context: {total_info}')
            return ctx.context_text
        except Exception as exc:
            self.log.debug(f'_get_cpg_context: {exc}')
            return ''

    async def _get_forward_impact_context(self, file_paths: list[str], issues: list) -> tuple[str, Any]:
        if not (self.cpg_engine and self.cpg_engine.is_available):
            return ('', None)
        try:
            function_names: list[str] = []
            for fp in file_paths:
                syms = await self._extract_file_symbols(fp)
                module_prefix = (
                    Path(fp)
                    .with_suffix('')
                    .as_posix()
                    .replace('/', '.')
                )
                for sym in sorted(syms)[:20]:
                    function_names.append(sym)
                    qualified = f'{module_prefix}.{sym}'
                    if qualified not in function_names:
                        function_names.append(qualified)

            if not function_names:
                function_names = [
                    Path(fp).with_suffix('').as_posix().replace('/', '.')
                    for fp in file_paths
                ]

            blast = await self.cpg_engine.compute_blast_radius(function_names=function_names, file_paths=file_paths, depth=3)
            if not blast.affected_functions and not blast.importing_modules:
                return ('', blast)

            lines: list[str] = [
                f'Blast radius: **{blast.affected_function_count} functions** across '
                f'**{blast.affected_file_count} files** '
                f'(score={blast.blast_radius_score:.4f})',
                '',
            ]

            by_file: dict[str, list[dict]] = {}
            for fn in blast.affected_functions:
                fp = fn.get('file_path', '<unknown>')
                by_file.setdefault(fp, []).append(fn)
            for fp, fns in sorted(by_file.items()):
                lines.append(f'### `{fp}`')
                for fn in fns[:8]:
                    name = fn.get('function_name', '?')
                    line = fn.get('line_number', 0)
                    rel = fn.get('relationship', '')
                    lines.append(f'  - L{line}  `{name}` [{rel}]')
                lines.append('')

            if blast.test_files_affected:
                lines.append(f'**Test files that cover changed code** ({len(blast.test_files_affected)}):')
                for tf in blast.test_files_affected[:10]:
                    lines.append(f'  - `{tf}`')
                lines.append('')

            if blast.importing_modules:
                lines.append(
                    f'**Import-only references** — files that import a changed symbol '
                    f'without calling any changed function ({blast.importing_module_count}):'
                )
                lines.append(
                    '_These files are broken by type changes, constant renames, and '
                    'signature shifts even though the call graph does not reach them._'
                )
                for im in blast.importing_modules[:20]:
                    lines.append(f'  - `{im}`')
                lines.append('')

            context_text = '\n'.join(lines)
            self.log.info(
                f'[Fixer] Gap 3 forward impact: {blast.affected_function_count} fns '
                f'+ {blast.importing_module_count} import-only modules '
                f'requires_human_review={blast.requires_human_review}'
            )
            return (context_text, blast)
        except Exception as exc:
            self.log.debug(f'_get_forward_impact_context: {exc}')
            return ('', None)

    async def _generate_refactor_proposal(self, issues: list, file_paths: list[str], blast: Any, forward_impact_context: str, model: str, recurrence_signal: 'BugRecurrenceSignal | None' = None, trigger_source: str = 'blast_radius') -> FixAttempt:
        from pydantic import BaseModel as _BM, Field as _F

        class _RefactorResponse(_BM):
            affected_components: list[str] = _F(default_factory=list)
            proposed_refactoring: str = ''
            migration_steps: list[str] = _F(default_factory=list)
            estimated_scope: str = ''
            risks: list[str] = _F(default_factory=list)
            recommendation: str = ''
        issue_summary = '\n'.join((f'- [{i.severity.value}] {i.description} ({i.file_path}:{i.line_start})' for i in issues[:10]))
        prompt = f'## Issues Identified\n{issue_summary}\n\n## Files Being Fixed\n' + '\n'.join((f'- `{fp}`' for fp in file_paths)) + f"\n\n## Forward Impact (CPG blast radius = {blast.affected_function_count} functions)\n{forward_impact_context}\n\n## Task\nThe blast radius of a direct patch exceeds the safe threshold (50 downstream functions). You MUST NOT generate a patch. Instead, produce a structured refactor proposal that:\n1. Lists every component that would be affected by changing the involved    function signatures or contracts.\n2. Proposes a safe migration strategy (adapter pattern, deprecation shim,    versioned API, feature flag, or incremental rollout).\n3. Lists concrete migration steps in order.\n4. Estimates scope (e.g. '3-5 engineering days, 12 files').\n5. Lists risks of each approach.\n6. Provides a clear recommendation.\n\nReturn structured JSON only."
        try:
            resp = await self.call_llm_structured(prompt=prompt, response_model=_RefactorResponse, system='You are a principal architect producing a refactor proposal. Be specific about affected components and concrete about migration steps. Do not generate any code patches — only architecture guidance.', model_override=model)
        except Exception as exc:
            self.log.error(f'[Fixer] _generate_refactor_proposal LLM call failed: {exc}')
            resp = _RefactorResponse(proposed_refactoring='LLM generation failed — manual review required.', recommendation='Route to senior engineer for manual refactor planning.')
        proposal = RefactorProposal(run_id=self.run_id, issue_ids=[i.id for i in issues], changed_functions=blast.changed_functions, affected_function_count=blast.affected_function_count, affected_file_count=blast.affected_file_count, test_files_affected=blast.test_files_affected, blast_radius_score=blast.blast_radius_score, importing_modules=blast.importing_modules, importing_module_count=blast.importing_module_count, affected_components=resp.affected_components, proposed_refactoring=resp.proposed_refactoring, migration_steps=resp.migration_steps, estimated_scope=resp.estimated_scope, risks=resp.risks, recommendation=resp.recommendation, requires_human_review=True, trigger_source=trigger_source, recurrence_count=recurrence_signal.recurrence_count if recurrence_signal else 0, reverted_count=recurrence_signal.reverted_count if recurrence_signal else 0, distinct_caller_modules=recurrence_signal.distinct_caller_modules if recurrence_signal else 0, coupling_score=recurrence_signal.coupling_score if recurrence_signal else -1.0, recurrence_escalation_reason=recurrence_signal.escalation_reason if recurrence_signal else '')
        if hasattr(self.storage, 'upsert_refactor_proposal'):
            await self.storage.upsert_refactor_proposal(proposal)
        escalation_id = ''
        if self.escalation_manager:
            try:
                esc = await self.escalation_manager.create_escalation(escalation_type='BLAST_RADIUS_EXCEEDED', description=f'Fix blast radius exceeds safe threshold: {blast.affected_function_count} downstream functions across {blast.affected_file_count} files. A direct patch is globally unsound. Refactor proposal {proposal.id[:12]} requires human review before any code changes are committed.', issue_ids=[i.id for i in issues], severity=Severity.CRITICAL)
                escalation_id = esc.id
                proposal.escalation_id = escalation_id
                if hasattr(self.storage, 'upsert_refactor_proposal'):
                    await self.storage.upsert_refactor_proposal(proposal)
            except Exception as exc:
                self.log.error(f'[Fixer] Failed to create blast-radius escalation: {exc}')
        fix = FixAttempt(run_id=self.run_id, issue_ids=[i.id for i in issues], fixed_files=[], fixer_model=model, fixer_model_family=extract_model_family(model), patch_mode=PatchMode.FULL_FILE, blast_radius_exceeded=True, refactor_proposal_id=proposal.id)
        await self.storage.upsert_fix(fix)
        for issue in issues:
            issue.fix_attempts += 1
            issue.status = IssueStatus.FIX_GENERATED
            await self.storage.upsert_issue(issue)
        self.log.warning(f'[Fixer] Refactor proposal created: proposal_id={proposal.id[:12]} escalation_id={(escalation_id[:12] if escalation_id else "none")} blast={blast.affected_function_count} fns')
        return fix

    async def _get_memory_examples(self, issues: list) -> str:
        if not self.fix_memory:
            return ''
        try:
            raw_query = ' '.join((i.description[:100] for i in issues[:3]))
            query = self._normalize_bug_class(raw_query)
            if hasattr(self.fix_memory, 'retrieve_async'):
                entries = await self.fix_memory.retrieve_async(query, n=3, max_age_days=180)
            else:
                entries = self.fix_memory.retrieve(query, n=3, max_age_days=180)
            return self.fix_memory.format_as_few_shot(entries)
        except Exception as exc:
            self.log.debug(f'_get_memory_examples: {exc}')
            return ''

    async def _store_fix_memory(self, issues: list, fixed_files: list[FixedFile], fix_id: str) -> None:
        if not self.fix_memory:
            return
        try:
            raw_type  = issues[0].description[:80] if issues else 'unknown'
            issue_type = self._normalize_bug_class(raw_type)
            file_context = ', '.join((ff.path for ff in fixed_files[:3]))
            fix_approach = '; '.join((ff.diff_summary[:60] for ff in fixed_files[:3] if ff.diff_summary))
            self.fix_memory.store_success(issue_type=issue_type, file_context=file_context, fix_approach=fix_approach, test_result='gate_passed=True', run_id=self.run_id)
        except Exception as exc:
            self.log.debug(f'_store_fix_memory: {exc}')
        try:
            await self._report_federated_usage(issues, success=True)
        except Exception as exc:
            self.log.debug(f'_store_fix_memory federated_usage: {exc}')

    async def _store_failure_memory(self, issues: list, file_paths: list[str], last_result: Any, test_output: str) -> None:
        if not self.fix_memory:
            return
        try:
            raw_type   = issues[0].description[:80] if issues else 'unknown'
            issue_type = self._normalize_bug_class(raw_type)
            file_context = ', '.join(file_paths[:3])
            if isinstance(last_result, PatchResponse) and last_result.patched_files:
                fix_approach = '; '.join(
                    pf.diff_summary[:80] for pf in last_result.patched_files[:3] if pf.diff_summary
                ) or 'patch generated (no diff_summary)'
            elif isinstance(last_result, FixResponse) and last_result.fixed_files:
                fix_approach = '; '.join(
                    ff.diff_summary[:80] for ff in last_result.fixed_files[:3] if ff.diff_summary
                ) or 'full-file rewrite (no diff_summary)'
            else:
                fix_approach = 'unknown approach'
            failure_reason = test_output[:400] if test_output else 'all probe rounds failed — no test output captured'
            self.fix_memory.store_failure(
                issue_type=issue_type,
                file_context=file_context,
                fix_approach=fix_approach,
                failure_reason=failure_reason,
                run_id=self.run_id,
            )
            self.log.info(
                f'[Fixer] Gap 3.A: stored failed approach to revert memory '
                f'issue_type={issue_type!r:.60} files={file_context!r:.80}'
            )
        except Exception as exc:
            self.log.debug(f'_store_failure_memory: {exc}')
        try:
            await self._report_federated_usage(issues, success=False)
        except Exception as exc:
            self.log.debug(f'_store_failure_memory federated_usage: {exc}')

    async def _report_federated_usage(self, issues: list, success: bool) -> None:
        if not self.fix_memory:
            return
        if not hasattr(self.fix_memory, 'record_federated_usage'):
            return
        if getattr(self.fix_memory, '_federated_store', None) is None:
            return
        try:
            raw_query = ' '.join(i.description[:100] for i in issues[:3])
            query = self._normalize_bug_class(raw_query)
            if hasattr(self.fix_memory, 'retrieve_async'):
                entries = await self.fix_memory.retrieve_async(query, n=10, max_age_days=180)
            else:
                entries = self.fix_memory.retrieve(query, n=10, max_age_days=180)
            for entry in entries:
                if entry.fix_approach.startswith('[FEDERATED]'):
                    fp = entry.id
                    if len(fp) == 64:
                        self.fix_memory.record_federated_usage(
                            fingerprint=fp,
                            success=success,
                        )
                    else:
                        self.log.debug(
                            f'_report_federated_usage: skipping entry with '
                            f'short fingerprint id={fp!r} (legacy entry)'
                        )
        except Exception as exc:
            self.log.debug(f'_report_federated_usage: {exc}')

    async def _apply_ast_rewrites(self, result: 'FixResponse', original_contents: dict[str, str], patch_modes: dict[str, PatchMode]) -> 'FixResponse':
        from sandbox.ast_rewrite import get_rewriter, ASTRewriteInstruction, RewriteOp
        rewriter = get_rewriter()
        updated_files: list[FixedFileFullResponse] = []
        for ffr in result.fixed_files:
            if patch_modes.get(ffr.path) != PatchMode.AST_REWRITE:
                updated_files.append(ffr)
                continue
            if not ffr.path.endswith(('.py', '.pyi')):
                updated_files.append(ffr)
                continue
            original = original_contents.get(ffr.path, '')
            try:
                import libcst as cst
                cst.parse_module(ffr.content)
                updated_files.append(ffr)
            except Exception as exc:
                self.log.warning(f'[Fixer] AST_REWRITE: libcst parse failed for {ffr.path} — falling back to FULL_FILE: {exc}')
                updated_files.append(ffr)
        return FixResponse(fixed_files=updated_files, overall_notes=result.overall_notes)

    def _select_patch_mode(self, file_path: str, line_count: int) -> PatchMode:
        if line_count >= self.surgical_patch_threshold:
            return PatchMode.UNIFIED_DIFF
        is_python = file_path.endswith(('.py', '.pyi'))
        if is_python and line_count >= _AST_REWRITE_THRESHOLD:
            return PatchMode.AST_REWRITE
        return PatchMode.FULL_FILE

    async def _generate_full_fix(self, issue_summary: str, file_context: str, vector_context: str, model: str, file_paths: list[str], repo_map_text: str='', memory_examples: str='', cpg_context: str='', forward_impact_context: str='') -> FixResponse:
        prompt_parts = []
        if repo_map_text:
            prompt_parts.append(repo_map_text)
        if memory_examples:
            prompt_parts.append(memory_examples)
        if cpg_context:
            prompt_parts.append(f'## Causal Context (Code Property Graph Backward Slice)\n**CRITICAL**: These are the functions that CAUSED the bug — not just similar-looking code. You MUST read and understand ALL of them before generating any fix. A fix that ignores the causal chain will introduce regressions.\n\n{cpg_context}')
        if forward_impact_context:
            prompt_parts.append(f'## Forward Impact (Functions That Will Break If Signatures Change)\n**HARD CONSTRAINT**: The following functions call or depend on the code you are fixing. Your fix MUST preserve all existing call signatures, return types, and contracts. If you must change a signature, you MUST also update every caller listed below in the same patch.\n\n{forward_impact_context}')
        prompt_parts += [f'## Issues to Fix\n{issue_summary}', f'## Files\n{file_context}']
        if vector_context:
            prompt_parts.append(f'## Semantically Similar Code (for reference)\n{vector_context}')
        prompt_parts.append('## Instructions\nReturn the COMPLETE corrected content for EVERY file listed. Do not truncate. Do not omit unchanged sections. Fix ONLY the listed issues. Make NO other changes. If a change is not directly required to fix an issue, do not make it.')
        prompt = '\n\n'.join(prompt_parts)
        return await self.call_llm_structured(prompt=prompt, response_model=FixResponse, system=self._fix_system_prompt(), model_override=model)

    async def _generate_patch_fix(self, issue_summary: str, file_context: str, vector_context: str, model: str, file_paths: list[str], repo_map_text: str='', memory_examples: str='', cpg_context: str='', forward_impact_context: str='') -> PatchResponse:
        prompt_parts = []
        if repo_map_text:
            prompt_parts.append(repo_map_text)
        if memory_examples:
            prompt_parts.append(memory_examples)
        if cpg_context:
            prompt_parts.append(f'## Causal Context (Code Property Graph Backward Slice)\n**CRITICAL**: These are the functions that CAUSED the bug — not just similar-looking code. Your patch MUST respect the causal chain. A patch that fixes the symptom without understanding the cause will regress.\n\n{cpg_context}')
        if forward_impact_context:
            prompt_parts.append(f'## Forward Impact (Functions That Will Break If Signatures Change)\n**HARD CONSTRAINT**: Your patch MUST preserve all call signatures, return types, and contracts for the functions below. If you must change a signature, include hunks for every affected call site.\n\n{forward_impact_context}')
        prompt_parts += [f'## Issues to Fix\n{issue_summary}', f'## Files (LARGE — surgical patch required)\n{file_context}']
        if vector_context:
            prompt_parts.append(f'## Semantically Similar Code\n{vector_context}')
        prompt_parts.append("## Instructions — UNIFIED DIFF MODE\nThese files are large. Return ONLY the changed lines as a unified diff.\nFormat:\n```\n--- a/path/to/file.c\n+++ b/path/to/file.c\n@@ -line,count +line,count @@\n context line\n-removed line\n+added line\n```\nRules:\n- 3 lines of unchanged context before and after each change\n- Fix ONLY the listed issues\n- Make NO other changes\n- The patch must apply cleanly with 'patch -p0'\n")
        prompt = '\n\n'.join(prompt_parts)
        return await self.call_llm_structured(prompt=prompt, response_model=PatchResponse, system=self._fix_system_prompt(), model_override=model)

    def _fix_system_prompt(self) -> str:
        return 'You are a principal software engineer generating precise, minimal fixes for identified code issues. Rules:\n1. Fix ONLY the reported issue. Do not refactor, improve, or reorganize.\n2. Preserve all existing comments, formatting conventions, and structure.\n3. Do not add logging, assertions, or tests unless the issue requires it.\n4. Prefer the simplest correct fix over a clever one.\n5. If fixing a security issue (buffer overflow, injection, UAF), apply the    standard safe pattern for the language — do not invent novel patterns.\n6. Every change must be directly traceable to a specific listed issue.\n7. Output structured JSON only — no prose explanation outside the JSON fields.'

    async def _probe_candidate(self, result: Any, original_contents: dict[str, str], file_paths: list[str]) -> tuple[bool, str]:
        if not self.repo_root:
            return (True, '')
        import tempfile, shutil as _shutil
        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix='rhodawk_probe_')
            tmp_root = Path(tmp_dir)
            for fp in file_paths:
                dest = tmp_root / fp
                dest.parent.mkdir(parents=True, exist_ok=True)
                original = original_contents.get(fp, '')
                dest.write_text(original, encoding='utf-8')
            if isinstance(result, PatchResponse):
                for pfr in result.patched_files:
                    dest = tmp_root / pfr.path
                    if dest.exists() and pfr.patch:
                        import subprocess as _sp
                        r = _sp.run(['patch', '--forward', '-p0', str(dest)], input=pfr.patch, capture_output=True, text=True, timeout=30)
                        if r.returncode != 0:
                            return (False, f'patch apply failed: {r.stderr[:500]}')
            elif isinstance(result, FixResponse):
                for ffr in result.fixed_files:
                    dest = tmp_root / ffr.path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(ffr.content, encoding='utf-8')
            from agents.test_runner import TestRunnerAgent
            from brain.schemas import FixAttempt, FixedFile, PatchMode
            import hashlib as _hl
            probe_fixed_files = [FixedFile(path=fp, content=(tmp_root / fp).read_text(encoding='utf-8', errors='replace') if (tmp_root / fp).exists() else '', patch='', patch_mode=PatchMode.FULL_FILE, changes_made='probe', diff_summary='probe') for fp in file_paths]
            probe_fix = FixAttempt(run_id=self.run_id, issue_ids=[], fixed_files=probe_fixed_files, fixer_model=self.config.model if self.config else '')
            tr = TestRunnerAgent(storage=self.storage, run_id=self.run_id, repo_root=tmp_root, config=self.config)
            tres = await tr.run_after_fix(probe_fix)
            from brain.schemas import TestRunStatus
            passed = tres.status in (TestRunStatus.PASSED, TestRunStatus.NO_TESTS)
            output = f'passed={tres.passed} failed={tres.failed}\n{tres.output[:2000]}'
            return (passed, output)
        except Exception as exc:
            self.log.debug(f'_probe_candidate failed (non-fatal): {exc}')
            return (True, '')
        finally:
            if tmp_dir:
                try:
                    _shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

    async def _build_file_context(self, file_paths: list[str], file_contents: dict[str, str], patch_modes: dict[str, PatchMode]) -> str:
        parts: list[str] = []
        for fp in file_paths:
            content = file_contents.get(fp, '')
            mode = patch_modes.get(fp, PatchMode.FULL_FILE)
            lines = content.count('\n')
            if mode == PatchMode.UNIFIED_DIFF:
                skeleton = self._extract_skeleton(content)
                parts.append(f'### {fp} ({lines} lines — SURGICAL PATCH MODE)\nSkeleton (function signatures and key structure):\n{wrap_content(skeleton)}\n')
            elif mode == PatchMode.AST_REWRITE:
                parts.append(f'### {fp} ({lines} lines — AST_REWRITE MODE)\nReturn the complete corrected file; libcst will validate syntax.\n{wrap_content(content)}\n')
            else:
                parts.append(f'### {fp}\n{wrap_content(content)}\n')
        return '\n'.join(parts)

    def _extract_skeleton(self, content: str) -> str:
        lines = content.splitlines()
        result: list[str] = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if any((kw in stripped for kw in ['def ', 'class ', 'void ', 'int ', 'char ', 'static ', 'inline ', 'struct ', 'enum ', 'typedef ', '#define ', '#include ', 'namespace ', 'template '])):
                result.append(f'L{i:5d}: {line}')
            elif stripped in ('{', '}', '};', 'end'):
                result.append(f'L{i:5d}: {line}')
        return '\n'.join(result[:200])

    async def _get_vector_context(self, issues) -> str:
        if self.hybrid_retriever and self.hybrid_retriever.is_available:
            try:
                query = ' '.join((i.description[:100] for i in issues[:3]))
                results = self.hybrid_retriever.find_similar_to_issue(query, n=6)
                parts = [f'[{r.file_path}:{r.line_start}-{r.line_end}] {r.summary}' for r in results]
                return '\n'.join(parts)
            except Exception as exc:
                self.log.debug(f'HybridRetriever context failed: {exc}')
        if self.vector_brain and self.vector_brain.is_available:
            try:
                query = ' '.join((i.description[:100] for i in issues[:3]))
                results = self.vector_brain.find_similar_to_issue(query, n=5)
                parts = [f'[{r.file_path}:{r.line_start}-{r.line_end}] {r.summary}' for r in results]
                return '\n'.join(parts)
            except Exception as exc:
                self.log.debug(f'Vector context failed: {exc}')
        return ''

    async def _load_file(self, file_path: str) -> str:
        if self.repo_root:
            try:
                from sandbox.executor import validate_path_within_root
                validate_path_within_root(file_path, self.repo_root)
                p = (self.repo_root / file_path).resolve()
                if p.exists():
                    return p.read_text(encoding='utf-8', errors='replace')
            except Exception:
                pass
        if self.mcp:
            try:
                return await self.mcp.read_file(file_path)
            except Exception:
                pass
        return ''

    async def run_with_patch(
        self,
        issues:      list,
        patch:       str,
        patch_model: str,
        patch_meta:  dict | None = None,
    ) -> list[FixAttempt]:
        if not patch or not patch.strip():
            self.log.warning('[run_with_patch] Empty patch — falling back to standard run()')
            return await self.run()

        fixed_files = self._parse_unified_diff_to_fixed_files(patch, patch_model)

        import json as _json
        meta = patch_meta or {}
        extra_notes = _json.dumps({
            'bobn_candidate_id':    meta.get('bobn_candidate_id', ''),
            'bobn_composite_score': meta.get('bobn_composite_score', 0.0),
            'bobn_test_score':      meta.get('bobn_test_score', 0.0),
            'bobn_n_candidates':    meta.get('bobn_n_candidates', 0),
            'attack_confidence':    meta.get('attack_confidence', 0.5),
        }, separators=(',', ':'))

        fix = FixAttempt(
            run_id             = self.run_id,
            issue_ids          = [i.id for i in issues],
            fixed_files        = fixed_files,
            fixer_model        = patch_model,
            fixer_model_family = extract_model_family(patch_model),
            patch_mode         = PatchMode.UNIFIED_DIFF,
            extra_notes        = extra_notes,
        )
        await self.storage.upsert_fix(fix)

        for issue in issues:
            issue.fix_attempts += 1
            issue.status = IssueStatus.FIX_GENERATED
            await self.storage.upsert_issue(issue)

        self.log.info(
            f'[run_with_patch] Persisted BoBN winner '
            f'fix_id={fix.id[:12]} '
            f'files={len(fixed_files)} '
            f'candidate={meta.get("bobn_candidate_id", "?")}'
        )
        return [fix]

    def _parse_unified_diff_to_fixed_files(
        self,
        patch:       str,
        patch_model: str,
    ) -> list[FixedFile]:
        import re as _re

        if not patch or not patch.strip():
            return []

        fixed_files: list[FixedFile] = []
        file_blocks = _re.split(r'(?=^--- )', patch, flags=_re.MULTILINE)

        for block in file_blocks:
            if not block.strip():
                continue

            old_match = _re.match(r'^--- (.+)', block)
            if not old_match:
                continue

            new_match = _re.search(r'^\+\+\+ (.+)', block, _re.MULTILINE)
            if not new_match:
                continue

            raw_path = new_match.group(1).strip()
            path = _re.sub(r'^[ab]/', '', raw_path)
            path = path.split('\t')[0].strip()

            if not path or path == '/dev/null':
                continue

            lines      = block.splitlines()
            added      = sum(1 for l in lines if l.startswith('+') and not l.startswith('+++'))
            removed    = sum(1 for l in lines if l.startswith('-') and not l.startswith('---'))
            lines_changed = added + removed

            fixed_files.append(FixedFile(
                path          = path,
                content       = '',
                patch         = block,
                patch_mode    = PatchMode.UNIFIED_DIFF,
                changes_made  = f'BoBN patch from {patch_model}',
                diff_summary  = f'BoBN adversarial winner: {added}+/{removed}- lines',
                lines_changed = lines_changed,
            ))

        return fixed_files
