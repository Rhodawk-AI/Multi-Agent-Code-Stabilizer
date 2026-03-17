from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from github import Github, GithubException
from github.Repository import Repository

from brain.schemas import FixAttempt

log = logging.getLogger(__name__)


class PRManager:

    def __init__(
        self,
        token: str,
        repo_url: str,
        branch_prefix: str = "stabilizer",
        base_branch: str = "main",
    ) -> None:
        self._gh = Github(token)
        self._repo_url = repo_url
        self.branch_prefix = branch_prefix
        self.base_branch = base_branch
        self._repo: Repository | None = None

    def _get_repo(self) -> Repository:
        if self._repo is None:
            url = self._repo_url.replace("https://", "").replace("http://", "")
            if url.startswith("github.com/"):
                url = url[len("github.com/"):]
            url = url.rstrip("/").removesuffix(".git")
            self._repo = self._gh.get_repo(url)
        return self._repo

    async def create_pr_for_fix(
        self,
        attempt: FixAttempt,
        run_id: str,
        cycle: int,
        repo_root: Path,
    ) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._create_pr_sync,
            attempt, run_id, cycle, repo_root,
        )

    def _create_pr_sync(
        self,
        attempt: FixAttempt,
        run_id: str,
        cycle: int,
        repo_root: Path,
    ) -> str:
        repo = self._get_repo()
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
        branch_name = f"{self.branch_prefix}/run-{run_id[:8]}-cycle-{cycle}-{ts}"

        base_ref = repo.get_git_ref(f"heads/{self.base_branch}")
        base_sha = base_ref.object.sha

        try:
            repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base_sha)
            log.info(f"Created branch: {branch_name}")
        except GithubException as exc:
            if "already exists" in str(exc):
                log.warning(f"Branch {branch_name} already exists, proceeding")
            else:
                raise

        for ff in attempt.fixed_files:
            self._commit_file(repo, branch_name, ff.path, ff.content, ff.changes_made)

        pr_body = self._build_pr_body(attempt, run_id, cycle)
        pr = repo.create_pull(
            title=f"[OpenMOSS] Cycle {cycle} — Fix {len(attempt.issue_ids)} issue(s)",
            body=pr_body,
            head=branch_name,
            base=self.base_branch,
            draft=False,
        )

        try:
            repo.get_label("openmoss")
        except GithubException:
            repo.create_label("openmoss", "0075ca", "OpenMOSS automated fix")
        try:
            pr.add_to_labels("openmoss")
        except Exception:
            pass

        log.info(f"PR created: {pr.html_url}")
        return pr.html_url

    def _commit_file(
        self,
        repo: Repository,
        branch: str,
        path: str,
        content: str,
        message_suffix: str,
    ) -> None:
        commit_message = (
            f"fix(openmoss): {path}\n\n"
            f"{message_suffix[:500]}\n\n"
            "Automated fix by OpenMOSS autonomous stabilizer."
        )
        try:
            existing = repo.get_contents(path, ref=branch)
            repo.update_file(
                path=path,
                message=commit_message,
                content=content,
                sha=existing.sha,  # type: ignore[union-attr]
                branch=branch,
            )
        except GithubException as exc:
            if exc.status == 404:
                repo.create_file(
                    path=path,
                    message=commit_message,
                    content=content,
                    branch=branch,
                )
            else:
                raise

    def _build_pr_body(self, attempt: FixAttempt, run_id: str, cycle: int) -> str:
        files_changed = "\n".join(
            f"- `{ff.path}` (+{ff.line_count}/-{ff.original_line_count} lines, {ff.diff_summary})"
            for ff in attempt.fixed_files
        )
        issues_fixed = "\n".join(f"- {iid}" for iid in attempt.issue_ids)
        changes = "\n".join(
            f"**{ff.path}**: {ff.changes_made}" for ff in attempt.fixed_files
        )
        ts = datetime.now(tz=timezone.utc).isoformat()
        return (
            f"## OpenMOSS Autonomous Stabilizer Fix\n\n"
            f"**Run ID:** `{run_id}`  \n"
            f"**Cycle:** {cycle}  \n"
            f"**Fix Attempt:** `{attempt.id}`  \n"
            f"**Timestamp:** {ts}\n\n"
            f"---\n\n"
            f"### Issues Resolved\n{issues_fixed}\n\n"
            f"### Files Changed\n{files_changed}\n\n"
            f"### Changes Made\n{changes}\n\n"
            f"---\n\n"
            f"### Automated Review\n"
            f"Score: {attempt.reviewer_confidence:.2f}  \n"
            f"Verdict: {attempt.reviewer_verdict.value if attempt.reviewer_verdict else 'N/A'}  \n"
            f"Notes: {attempt.reviewer_reason or 'None'}\n\n"
            f"---\n\n"
            f"> ⚠️ Generated by OpenMOSS. Review carefully before merging.\n"
            f"> Load-bearing files require manual review.\n"
        )
