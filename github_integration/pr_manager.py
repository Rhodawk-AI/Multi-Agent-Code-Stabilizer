"""github_integration/pr_manager.py — GitHub PR manager for Rhodawk AI."""
from __future__ import annotations
import asyncio, json, logging, urllib.request, urllib.error
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

    def _create_pr_sync(
        self, branch_name: str, files: list[tuple[str, str]],
        title: str, body: str
    ) -> str:
        if not self.token or not self.owner:
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
        except Exception as exc:
            log.error(f"[pr_manager] PR creation failed: {exc}")
            return ""
