"""
discovery/full_pipeline.py
============================
Autonomous end-to-end security research pipeline.

This is the top-level orchestrator that turns Rhodawk into a fully autonomous
security research system:

    Discover repos  →  Clone  →  Multi-tool audit  →  CVE correlation
    →  Fix generation  →  Patch verification  →  Security advisory
    →  Pull request  →  Track disclosure deadline

Execution modes
───────────────
HUNT mode:
    Autonomously discovers high-value repos, clones them, runs the full
    analysis pipeline, and files PRs for every valid finding.

TARGETED mode:
    Runs the full pipeline on a single pre-specified repo. Used for
    bug bounty submissions or continuous monitoring of a known target.

PATROL mode:
    Monitors a list of repos for new commits and re-audits only the
    changed files (using the CPG incremental updater).

Usage
─────
    from discovery.full_pipeline import AutonomousPipeline

    pipeline = AutonomousPipeline.from_env()
    await pipeline.run(
        mode="hunt",
        language="c",
        min_stars=500,
        max_repos=20,
        duration_h=8,
    )
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    # Discovery
    github_token:    str   = ""
    gitlab_token:    str   = ""
    sourcegraph_token: str = ""
    # Scanning
    run_codeql:      bool  = True
    run_trufflehog:  bool  = True
    run_nuclei:      bool  = True
    run_infer:       bool  = True
    run_afl:         bool  = False  # off by default — requires compiled targets
    run_semgrep:     bool  = True
    run_mariana_trench: bool = False  # Android only
    # Correlation
    nvd_api_key:     str   = ""
    correlate_cves:  bool  = True
    # Disclosure
    create_advisories: bool = True
    create_prs:      bool  = True
    draft_prs:       bool  = True   # always draft — human reviews before merge
    disclosure_days: int   = 90
    # Targets
    work_dir:        str   = ""
    max_repos:       int   = 10
    min_stars:       int   = 500
    language:        str   = "c"
    strategy:        str   = "language_sweep"
    # Rhodawk core
    primary_model:   str   = "openai/Qwen/Qwen2.5-Coder-32B-Instruct"
    use_sqlite:      bool  = True
    domain_mode:     str   = "general"


class AutonomousPipeline:
    """
    End-to-end autonomous security research pipeline.

    Combines repo discovery, multi-tool static analysis, LLM-powered
    auditing, patch generation, CVE correlation, and security PR creation
    into a single autonomous loop.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._work_dir = config.work_dir or tempfile.mkdtemp(prefix="rhodawk_hunt_")

    @classmethod
    def from_env(cls) -> "AutonomousPipeline":
        """Create from environment variables — convenient for CI/CD."""
        return cls(PipelineConfig(
            github_token     = os.environ.get("GITHUB_TOKEN", ""),
            gitlab_token     = os.environ.get("GITLAB_TOKEN", ""),
            sourcegraph_token= os.environ.get("SOURCEGRAPH_TOKEN", ""),
            nvd_api_key      = os.environ.get("RHODAWK_NVD_API_KEY", ""),
            primary_model    = os.environ.get("RHODAWK_PRIMARY_MODEL",
                                              "openai/Qwen/Qwen2.5-Coder-32B-Instruct"),
            work_dir         = os.environ.get("RHODAWK_WORK_DIR", ""),
            run_afl          = os.environ.get("RHODAWK_RUN_AFL", "0") == "1",
            run_mariana_trench=os.environ.get("RHODAWK_RUN_MARIANA", "0") == "1",
        ))

    async def run(
        self,
        mode:      str = "targeted",
        repo_url:  str = "",
        language:  str = "",
        min_stars: int = 0,
        max_repos: int = 0,
        duration_h:float = 4.0,
    ) -> list[dict]:
        """
        Run the autonomous pipeline.

        Parameters
        ----------
        mode:
            "hunt" | "targeted" | "patrol"
        repo_url:
            Required for targeted mode.
        language:
            Target language for hunt mode (c, python, java, ...).
        min_stars:
            Minimum repo stars for hunt mode.
        max_repos:
            Max repos to process in hunt mode.
        duration_h:
            Max pipeline duration in hours.

        Returns list of summary dicts for all findings and PRs created.
        """
        import asyncio
        deadline = asyncio.get_event_loop().time() + duration_h * 3600
        cfg = self.config
        language  = language  or cfg.language
        max_repos = max_repos or cfg.max_repos
        min_stars = min_stars or cfg.min_stars

        results: list[dict] = []

        if mode == "targeted" and repo_url:
            result = await self._process_repo(repo_url)
            results.append(result)

        elif mode == "hunt":
            from discovery.repo_hunter import RepoHunter
            hunter = RepoHunter(
                github_token      = cfg.github_token,
                gitlab_token      = cfg.gitlab_token,
                sourcegraph_token = cfg.sourcegraph_token,
            )
            targets = await hunter.hunt(
                strategy  = cfg.strategy,
                language  = language,
                min_stars = min_stars,
                max_repos = max_repos,
            )
            log.info("[Pipeline] Hunt mode: %d targets queued", len(targets))

            for target in targets:
                if asyncio.get_event_loop().time() > deadline:
                    log.info("[Pipeline] Duration limit reached — stopping")
                    break
                # Clone target
                local_path = await hunter.clone(
                    target,
                    dest_dir=os.path.join(self._work_dir, "repos"),
                )
                if not local_path:
                    continue
                result = await self._process_repo(
                    repo_url    = target.clone_url,
                    local_path  = local_path,
                    repo_owner  = target.owner,
                    repo_name   = target.name,
                )
                results.append(result)

        elif mode == "patrol":
            log.info("[Pipeline] Patrol mode not yet implemented — use targeted mode")

        log.info(
            "[Pipeline] Complete. Processed %d repos, %d total findings, %d PRs",
            len(results),
            sum(r.get("finding_count", 0) for r in results),
            sum(r.get("pr_count", 0) for r in results),
        )
        return results

    async def _process_repo(
        self,
        repo_url:   str,
        local_path: str = "",
        repo_owner: str = "",
        repo_name:  str = "",
    ) -> dict:
        """
        Run the full pipeline against a single repository.

        Returns a summary dict with finding_count, pr_count, advisories.
        """
        cfg    = self.config
        result = {
            "repo_url":     repo_url,
            "finding_count": 0,
            "pr_count":      0,
            "advisories":    [],
            "errors":        [],
        }

        # ── 1. Determine local path ────────────────────────────────────────────
        if not local_path:
            from discovery.repo_hunter import RepoHunter
            hunter    = RepoHunter(github_token=cfg.github_token)
            # Parse owner/name from URL
            parts     = repo_url.rstrip("/").split("/")
            tmp_owner = parts[-2] if len(parts) >= 2 else "unknown"
            tmp_name  = parts[-1] if parts else "repo"
            from discovery.repo_hunter import RepoTarget
            target = RepoTarget(
                owner       = tmp_owner,
                name        = tmp_name,
                clone_url   = repo_url,
            )
            local_path = await hunter.clone(
                target, dest_dir=os.path.join(self._work_dir, "repos")
            )
        if not local_path:
            result["errors"].append(f"Clone failed for {repo_url}")
            return result

        repo_path = Path(local_path)
        if not repo_owner:
            parts     = repo_url.rstrip("/").split("/")
            repo_owner= parts[-2] if len(parts) >= 2 else "unknown"
            repo_name = parts[-1] if parts else "repo"

        log.info("[Pipeline] Processing %s/%s at %s", repo_owner, repo_name, local_path)

        # ── 2. Run Rhodawk core pipeline (StabilizerController) ────────────────
        rhodawk_issues: list[Any] = []
        rhodawk_fixes:  list[Any] = []
        try:
            rhodawk_issues, rhodawk_fixes = await self._run_rhodawk_core(
                repo_url   = repo_url,
                local_path = local_path,
            )
            log.info(
                "[Pipeline] Rhodawk: %d issues, %d fixes for %s/%s",
                len(rhodawk_issues), len(rhodawk_fixes), repo_owner, repo_name,
            )
        except Exception as exc:
            log.error("[Pipeline] Rhodawk core failed for %s: %s", repo_url, exc)
            result["errors"].append(str(exc))

        # ── 3. Run extra tool servers ──────────────────────────────────────────
        extra_findings: list[dict] = []

        if cfg.run_trufflehog:
            try:
                from tools.servers.trufflehog_server import trufflehog_scan
                th = await trufflehog_scan(local_path, scan_history=True)
                extra_findings.extend(th)
                log.info("[Pipeline] TruffleHog: %d secrets", len(th))
            except Exception as exc:
                log.warning("[Pipeline] TruffleHog failed: %s", exc)

        if cfg.run_codeql:
            try:
                from tools.servers.codeql_server import codeql_scan_repo
                cq = await codeql_scan_repo(local_path)
                extra_findings.extend(cq)
                log.info("[Pipeline] CodeQL: %d findings", len(cq))
            except Exception as exc:
                log.warning("[Pipeline] CodeQL failed: %s", exc)

        if cfg.run_nuclei:
            try:
                from tools.servers.nuclei_server import nuclei_scan_repo
                nu = await nuclei_scan_repo(local_path)
                extra_findings.extend(nu)
                log.info("[Pipeline] Nuclei: %d findings", len(nu))
            except Exception as exc:
                log.warning("[Pipeline] Nuclei failed: %s", exc)

        if cfg.run_infer:
            try:
                from tools.servers.infer_server import infer_scan
                inf = await infer_scan(local_path)
                extra_findings.extend(inf)
                log.info("[Pipeline] Infer: %d findings", len(inf))
            except Exception as exc:
                log.warning("[Pipeline] Infer failed: %s", exc)

        if cfg.run_afl:
            try:
                from tools.servers.afl_server import discover_and_fuzz
                crashes = await discover_and_fuzz(local_path, duration_per_target_s=120)
                extra_findings.extend(crashes)
                log.info("[Pipeline] AFL++: %d crashes", len(crashes))
            except Exception as exc:
                log.warning("[Pipeline] AFL++ failed: %s", exc)

        result["finding_count"] = len(rhodawk_issues) + len(extra_findings)

        # ── 4. CVE correlation ─────────────────────────────────────────────────
        cve_map: dict[str, Any] = {}
        if cfg.correlate_cves and cfg.nvd_api_key:
            try:
                from intelligence.osv_correlator import OSVCorrelator
                correlator = OSVCorrelator(nvd_api_key=cfg.nvd_api_key)
                await correlator.initialise()
                try:
                    # Also scan SBOM if available
                    sbom_path = repo_path / "sbom.json"
                    if sbom_path.exists():
                        import json as _json
                        sbom = _json.loads(sbom_path.read_text())
                        sbom_vulns = await correlator.scan_sbom(sbom)
                        for sv in sbom_vulns:
                            for cve in sv.get("cves", []):
                                cve_map[cve.cve_id] = cve
                finally:
                    await correlator.close()
            except Exception as exc:
                log.warning("[Pipeline] CVE correlation failed: %s", exc)

        # ── 5. Security advisories + PRs ──────────────────────────────────────
        if (cfg.create_prs or cfg.create_advisories) and rhodawk_fixes:
            advisory_engine = None
            if cfg.create_advisories and cfg.github_token:
                from disclosure.security_advisory import SecurityAdvisory
                advisory_engine = SecurityAdvisory(
                    github_token = cfg.github_token,
                    repo_owner   = repo_owner,
                    repo_name    = repo_name,
                    domain_mode  = cfg.domain_mode,
                )

            from orchestrator.patch_transaction import PatchTransaction
            txn = PatchTransaction(repo_root=repo_path)

            for fix in rhodawk_fixes[:10]:  # cap PRs per repo
                if not fix.fixed_files:
                    continue
                try:
                    # Verify patch before PR
                    verify_result = await txn.apply_and_verify(
                        fixed_files = fix.fixed_files,
                        run_id      = fix.run_id,
                    )
                    if not verify_result.ok:
                        log.warning(
                            "[Pipeline] Patch gate FAILED for fix %s: %s",
                            fix.id[:8], verify_result.failure_reason,
                        )
                        continue

                    # Create advisory
                    advisory = None
                    if advisory_engine:
                        # Get the issues for this fix
                        issue = None
                        if fix.issue_ids and rhodawk_issues:
                            issue = next(
                                (i for i in rhodawk_issues
                                 if i.id in fix.issue_ids), None
                            )
                        if issue:
                            cve_match = cve_map.get(
                                getattr(issue, "cve_id", ""), None
                            )
                            advisory = await advisory_engine.create_private_advisory(
                                issue     = issue,
                                fix       = fix,
                                cve_match = cve_match,
                            )
                            if advisory.ghsa_id:
                                result["advisories"].append(advisory.ghsa_id)

                    # Create PR
                    if cfg.create_prs and cfg.github_token:
                        import uuid
                        branch = f"rhodawk/{fix.id[:8]}-{uuid.uuid4().hex[:4]}"
                        pr_url = await advisory_engine.create_security_pr(
                            branch_name  = branch,
                            fixed_files  = fix.fixed_files,
                            issue        = issue if issue else _stub_issue(fix),
                            fix          = fix,
                            advisory     = advisory,
                            cve_match    = cve_match if advisory else None,
                            gate_output  = verify_result.gate_output,
                        ) if advisory_engine else ""
                        if pr_url:
                            result["pr_count"] += 1
                            log.info("[Pipeline] PR created: %s", pr_url)

                except Exception as exc:
                    log.error("[Pipeline] Fix processing failed: %s", exc)

        return result

    async def _run_rhodawk_core(
        self, repo_url: str, local_path: str
    ) -> tuple[list[Any], list[Any]]:
        """
        Run the full StabilizerController pipeline on a single repo.
        Returns (issues, fixes).
        """
        from orchestrator.controller import StabilizerController, StabilizerConfig
        from pathlib import Path

        cfg    = self.config
        db_path = os.path.join(self._work_dir, "db",
                               local_path.replace("/", "_") + ".db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        sc_config = StabilizerConfig(
            repo_url       = repo_url,
            repo_root      = Path(local_path),
            primary_model  = cfg.primary_model,
            use_sqlite     = True,
            max_cycles     = 5,    # cap cycles per repo in hunt mode
            run_semgrep    = cfg.run_semgrep,
            auto_commit    = False,  # never auto-commit in hunt mode
            github_token   = cfg.github_token,
        )
        sc_config.use_sqlite = True

        # Override db path
        import os as _os
        _os.environ.setdefault("RHODAWK_DB_PATH", db_path)

        controller = StabilizerController(sc_config)
        await controller.initialise()
        try:
            await controller.run()
        finally:
            if controller.storage:
                issues = await controller.storage.list_issues()
                fixes  = await controller.storage.list_fixes()
                await controller.storage.close()
                return issues, fixes
        return [], []


def _stub_issue(fix: Any) -> Any:
    """Create a minimal issue stub when the real issue isn't available."""
    from types import SimpleNamespace
    from brain.schemas import Severity
    return SimpleNamespace(
        id          = fix.issue_ids[0] if fix.issue_ids else "unknown",
        description = "Security fix",
        severity    = Severity.HIGH,
        file_path   = fix.fixed_files[0].path if fix.fixed_files else "unknown",
        line_start  = 0,
    )
