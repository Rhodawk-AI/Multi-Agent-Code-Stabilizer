"""
github_integration/pr_manager.py
GitHub integration for OpenMOSS.
Creates branches, commits complete fixed files, opens PRs with full audit trail.
Never force-pushes. Never touches main directly.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from github import Github, GithubException
from github.Repository import Repository

from brain.schemas import FixAttempt

log = logging.getLogger(__name__)


class PRManager:
    """
    Manages all GitHub interactions for OpenMOSS.
    One PR per fix batch. Branch per cycle.
    """

    def __init__(
        self,
        token: str,
        repo_url: str,
        branch_prefix: str = "stabilizer",
        base_branch: str = "main",
    ) -> None:
        self._gh      = Github(token)
        self._repo_url = repo_url
        self.branch_prefix = branch_prefix
        self.base_branch   = base_branch
        self._repo: Repository | None = None

    def _get_repo(self) -> Repository:
        if self._repo is None:
            # Extract owner/repo from URL
            # Handles: https://github.com/owner/repo, github.com/owner/repo, owner/repo
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
        """
        Create a GitHub branch, commit all fixed files, open a PR.
        Returns the PR URL.
        This is a synchronous operation wrapped for async callers.
        """
        import asyncio
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
        """Synchronous GitHub operations."""
        repo = self._get_repo()
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        branch_name = f"{self.branch_prefix}/run-{run_id[:8]}-cycle-{cycle}-{timestamp}"

        # Get base branch SHA
        base_ref = repo.get_git_ref(f"heads/{self.base_branch}")
        base_sha = base_ref.object.sha

        # Create branch
        try:
            repo.create_git_ref(
                ref=f"refs/heads/{branch_name}",
                sha=base_sha,
            )
            log.info(f"Created branch: {branch_name}")
        except GithubException as exc:
            if "already exists" in str(exc):
                log.warning(f"Branch {branch_name} already exists, proceeding")
            else:
                raise

        # Commit each fixed file
        for ff in attempt.fixed_files:
            self._commit_file(repo, branch_name, ff.path, ff.content, ff.changes_made)

        # Build PR body
        pr_body = self._build_pr_body(attempt, run_id, cycle)

        # Create PR
        pr = repo.create_pull(
            title=f"[OpenMOSS] Cycle {cycle} — Fix {len(attempt.issue_ids)} issue(s)",
            body=pr_body,
            head=branch_name,
            base=self.base_branch,
            draft=False,
        )

        # Add labels
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
        """Create or update a file on the branch."""
        commit_message = (
            f"fix(openmoss): {path}\n\n"
            f"{message_suffix[:500]}\n\n"
            "Automated fix by OpenMOSS autonomous stabilizer."
        )
        try:
            # Try to get existing file for its SHA
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
                # New file
                repo.create_file(
                    path=path,
                    message=commit_message,
                    content=content,
                    branch=branch,
                )
            else:
                raise

    def _build_pr_body(self, attempt: FixAttempt, run_id: str, cycle: int) -> str:
        files_changed = "\n".join(f"- `{ff.path}` ({ff.line_count} lines)" for ff in attempt.fixed_files)
        issues_fixed = "\n".join(f"- {iid}" for iid in attempt.issue_ids)
        changes = "\n".join(
            f"**{ff.path}**: {ff.changes_made}"
            for ff in attempt.fixed_files
        )

        return f"""## OpenMOSS Autonomous Stabilizer Fix

**Run ID:** `{run_id}`
**Cycle:** {cycle}
**Fix Attempt:** `{attempt.id}`
**Timestamp:** {datetime.utcnow().isoformat()}

---

### Issues Resolved
{issues_fixed}

### Files Changed
{files_changed}

### Changes Made
{changes}

---

### Reviewer Confidence
Score: {attempt.reviewer_confidence:.2f}
Verdict: {attempt.reviewer_verdict.value if attempt.reviewer_verdict else "N/A"}
Notes: {attempt.reviewer_reason or "None"}

---

> ⚠️ This PR was generated by [OpenMOSS](https://github.com/openmoss/openmoss).
> Review carefully before merging. All fixes have passed automated review.
> Load-bearing files require manual review before merge.
"""
