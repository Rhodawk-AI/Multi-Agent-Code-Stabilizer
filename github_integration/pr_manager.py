"""github_integration/pr_manager.py — GitHub PR manager for Rhodawk AI."""
from __future__ import annotations
import asyncio, json, logging, shutil, subprocess, urllib.request, urllib.error
from pathlib import Path
log = logging.getLogger(__name__)


class PRManager:
    def __init__(
        self,
        token:        str,
        repo_url:     str,
        base_branch:  str = "main",
        branch_prefix: str = "stabilizer",
    ) -> None:
        self.token         = token
        self.repo_url      = repo_url.rstrip("/")
        self.base_branch   = base_branch
        self.branch_prefix = branch_prefix
        parts              = self.repo_url.split("/")
        self.owner         = parts[-2] if len(parts) >= 2 else ""
        self.repo          = parts[-1] if len(parts) >= 1 else ""

    async def create_pr(
        self,
        branch_name: str,
        files:       list[tuple[str, str]],
        title:       str,
        body:        str,
    ) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._create_pr_sync, branch_name, files, title, body
        )

    def _push_branch(self, branch_name: str) -> bool:
        """
        ADD-4 FIX: Push current HEAD to the remote branch before opening a PR.

        Previously _create_pr_sync() called the GitHub PR API with
        `head: branch_name` but never pushed the branch first.  A branch that
        does not exist on the remote causes the GitHub API to return:
            422 Unprocessable Entity: head sha ... is not a valid sha
        The error was silently swallowed by the broad `except Exception` block,
        returning "" with no indication of whether GitHub mode was unconfigured
        or the push simply hadn't happened.

        Returns True if the push succeeded (or git is unavailable — callers
        decide whether to proceed).  Logs the specific failure mode so
        operators can distinguish misconfigured git remote from API errors.
        """
        if not shutil.which("git"):
            log.warning("[pr_manager] git not found on PATH — skipping branch push")
            return False

        push_cmd = ["git", "push", "origin", f"HEAD:{branch_name}", "--force-with-lease"]
        try:
            result = subprocess.run(
                push_cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                log.error(
                    "[pr_manager] Branch push failed (returncode=%d): %s",
                    result.returncode,
                    (result.stderr or result.stdout).strip()[:500],
                )
                return False
            log.info("[pr_manager] Pushed HEAD to remote branch '%s'", branch_name)
            return True
        except subprocess.TimeoutExpired:
            log.error("[pr_manager] Branch push timed out after 60 s")
            return False
        except Exception as exc:
            log.error("[pr_manager] Branch push error: %s", exc)
            return False

    def _create_pr_sync(
        self, branch_name: str, files: list[tuple[str, str]],
        title: str, body: str
    ) -> str:
        if not self.token or not self.owner:
            return ""

        # ADD-4 FIX: push the branch before opening the PR.
        # Without this, the GitHub API returns 422 because the branch does not
        # exist on the remote.  If the push fails, abort and return a structured
        # failure indicator so the controller can log the specific failure mode
        # rather than silently treating it as "GitHub mode not configured".
        push_ok = self._push_branch(branch_name)
        if not push_ok:
            log.error(
                "[pr_manager] Aborting PR creation for branch '%s' — "
                "branch push failed.  Fix the git remote / credentials and retry.",
                branch_name,
            )
            return ""

        try:
            api_url = (
                f"https://api.github.com/repos/{self.owner}/{self.repo}/pulls"
            )
            payload = json.dumps({
                "title": title, "body": body,
                "head": branch_name, "base": self.base_branch,
            }).encode()
            req = urllib.request.Request(
                api_url, data=payload, method="POST",
                headers={
                    "Authorization": f"token {self.token}",
                    "Content-Type":  "application/json",
                    "Accept":        "application/vnd.github.v3+json",
                    "User-Agent":    "Rhodawk-AI/2.0",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                pr_url = data.get("html_url", "")
                log.info(f"[pr_manager] PR created: {pr_url}")
                return pr_url
        except urllib.error.HTTPError as exc:
            body_text = ""
            try:
                body_text = exc.read().decode(errors="replace")[:500]
            except Exception:
                pass
            log.error(
                "[pr_manager] PR creation failed: HTTP %d — %s",
                exc.code, body_text,
            )
            return ""
        except Exception as exc:
            log.error("[pr_manager] PR creation failed: %s", exc)
            return ""
