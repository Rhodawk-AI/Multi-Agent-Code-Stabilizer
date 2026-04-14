"""
discovery/repo_hunter.py
==========================
Autonomous repository discovery and targeting engine.

The hunter finds high-value repositories across GitHub, GitLab, and
Sourcegraph, scores them by attack surface, and queues them for the
Rhodawk pipeline. This is what turns Rhodawk from a tool you point at
a repo into a self-directed vulnerability research system.

Targeting strategies
────────────────────
1. LANGUAGE_SWEEP    — all repos in a language above a stars threshold
2. DEPENDENCY_AUDIT  — repos that depend on a package with a known CVE
3. PATTERN_HUNT      — repos matching code patterns (grep.app / Sourcegraph)
4. ORGANISATION_SWEEP— all public repos in a GitHub org / GitLab group
5. TRENDING          — GitHub trending (high visibility, fast triage ROI)
6. BUGBOUNTY         — repos linked from HackerOne / Bugcrowd programmes

Scoring heuristics (higher = higher priority)
──────────────────────────────────────────────
• Stars / forks  → community trust signal
• Recent commits → active maintenance, fixes will be merged
• Open issues    → known pain points
• Language risk  → C/C++ > Java > Python > TypeScript
• File patterns  → Makefiles, .proto, CMakeLists.txt → larger attack surface
• Dependency age → stale lockfiles = higher CVE exposure

Public API
──────────
    hunter = RepoHunter(github_token="ghp_...", gitlab_token="glpat_...")
    targets = await hunter.hunt(strategy="dependency_audit",
                                 package="log4j",
                                 language="java",
                                 max_repos=50)
    for t in targets:
        print(t.clone_url, t.priority_score)
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

_LANG_RISK: dict[str, float] = {
    "c":          1.0,
    "c++":        1.0,
    "rust":       0.5,
    "go":         0.6,
    "java":       0.75,
    "kotlin":     0.65,
    "swift":      0.65,
    "python":     0.55,
    "javascript": 0.6,
    "typescript": 0.5,
    "php":        0.85,
    "ruby":       0.5,
    "solidity":   0.95,   # smart contracts — extreme
}

_HIGH_VALUE_FILE_PATTERNS = [
    "Makefile", "CMakeLists.txt", "*.proto", "*.thrift",
    "docker-compose.yml", "Dockerfile", "*.yaml", ".github/workflows",
    "AndroidManifest.xml", "*.plist", "Package.swift",
]


@dataclass
class RepoTarget:
    """A discovered repository ready to feed into the Rhodawk pipeline."""
    owner:           str
    name:            str
    clone_url:       str
    default_branch:  str           = "main"
    language:        str           = ""
    stars:           int           = 0
    forks:           int           = 0
    open_issues:     int           = 0
    size_kb:         int           = 0
    last_pushed_at:  str           = ""
    description:     str           = ""
    topics:          list[str]     = field(default_factory=list)
    priority_score:  float         = 0.0
    discovery_reason: str          = ""
    # Set by pipeline after clone
    local_path:      str           = ""


class RepoHunter:
    """
    Autonomous multi-source repository discovery engine.

    Parameters
    ----------
    github_token:
        Personal access token with repo:read scope.
    gitlab_token:
        GitLab personal access token (optional).
    sourcegraph_token:
        Sourcegraph access token for code-pattern search (optional).
    max_concurrent_requests:
        Concurrency cap for API calls.
    """

    def __init__(
        self,
        github_token:            str = "",
        gitlab_token:            str = "",
        sourcegraph_token:       str = "",
        max_concurrent_requests: int = 8,
    ) -> None:
        self._gh_token  = github_token  or os.environ.get("GITHUB_TOKEN", "")
        self._gl_token  = gitlab_token  or os.environ.get("GITLAB_TOKEN", "")
        self._sg_token  = sourcegraph_token or os.environ.get("SOURCEGRAPH_TOKEN", "")
        self._sem       = asyncio.Semaphore(max_concurrent_requests)
        self._session: Any = None   # aiohttp.ClientSession — lazy

    # ── Public entry point ────────────────────────────────────────────────────

    async def hunt(
        self,
        strategy:     str   = "language_sweep",
        language:     str   = "c",
        min_stars:    int   = 100,
        max_repos:    int   = 50,
        package:      str   = "",
        pattern:      str   = "",
        org:          str   = "",
        **kwargs: Any,
    ) -> list[RepoTarget]:
        """
        Discover and rank repositories matching the given strategy.

        Parameters
        ----------
        strategy:
            One of: language_sweep | dependency_audit | pattern_hunt |
                    organisation_sweep | trending | bugbounty
        language:
            Primary language filter (GitHub language name, lowercase).
        min_stars:
            Minimum stars for language_sweep / trending strategies.
        max_repos:
            Maximum number of targets to return (sorted by priority).
        package:
            Package name for dependency_audit strategy.
        pattern:
            Code pattern (Sourcegraph query) for pattern_hunt strategy.
        org:
            GitHub org or GitLab group for organisation_sweep strategy.
        """
        import aiohttp
        async with aiohttp.ClientSession() as session:
            self._session = session
            raw: list[RepoTarget] = []
            strat = strategy.lower()
            if strat == "language_sweep":
                raw = await self._language_sweep(language, min_stars, max_repos * 3)
            elif strat == "dependency_audit":
                raw = await self._dependency_audit(package, language, max_repos * 3)
            elif strat == "pattern_hunt":
                raw = await self._pattern_hunt(pattern, max_repos * 3)
            elif strat == "organisation_sweep":
                raw = await self._org_sweep(org, max_repos * 3)
            elif strat == "trending":
                raw = await self._trending(language, max_repos * 3)
            elif strat == "bugbounty":
                raw = await self._bugbounty_targets(language, max_repos * 3)
            else:
                log.warning("[RepoHunter] Unknown strategy %r — using language_sweep", strategy)
                raw = await self._language_sweep(language, min_stars, max_repos * 3)

        # Score and rank
        for t in raw:
            t.priority_score = self._score(t)
        raw.sort(key=lambda t: t.priority_score, reverse=True)
        result = raw[:max_repos]
        log.info(
            "[RepoHunter] strategy=%s → %d candidates → %d selected "
            "(top score=%.2f)",
            strategy, len(raw), len(result),
            result[0].priority_score if result else 0.0,
        )
        return result

    # ── GitHub search strategies ──────────────────────────────────────────────

    async def _gh_search(self, query: str, per_page: int = 100) -> list[dict]:
        """Run a GitHub code/repo search and return raw repo items."""
        if not self._gh_token:
            log.warning("[RepoHunter] GITHUB_TOKEN not set — GitHub search disabled")
            return []
        headers = {
            "Authorization": f"Bearer {self._gh_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        url = "https://api.github.com/search/repositories"
        params = {"q": query, "per_page": per_page, "sort": "stars", "order": "desc"}
        async with self._sem:
            try:
                async with self._session.get(url, headers=headers, params=params, timeout=30) as r:
                    if r.status == 403:
                        log.warning("[RepoHunter] GitHub rate-limited — backing off 60 s")
                        await asyncio.sleep(60)
                        return []
                    if r.status != 200:
                        log.warning("[RepoHunter] GitHub search HTTP %d", r.status)
                        return []
                    data = await r.json()
                    return data.get("items", [])
            except Exception as exc:
                log.warning("[RepoHunter] GitHub search error: %s", exc)
                return []

    async def _language_sweep(
        self, language: str, min_stars: int, limit: int
    ) -> list[RepoTarget]:
        query = f"language:{language} stars:>{min_stars} archived:false"
        items = await self._gh_search(query, per_page=min(100, limit))
        return [self._gh_item_to_target(i, f"language_sweep:{language}") for i in items]

    async def _dependency_audit(
        self, package: str, language: str, limit: int
    ) -> list[RepoTarget]:
        """Find repos that import a known-vulnerable package."""
        if not package:
            return []
        # GitHub dependency graph API (requires repo:read + Contents permission)
        # Fallback: search for the package name in typical manifest files
        query = f"{package!r} in:file language:{language} archived:false"
        items = await self._gh_search(query, per_page=min(100, limit))
        return [self._gh_item_to_target(i, f"dependency_audit:{package}") for i in items]

    async def _pattern_hunt(self, pattern: str, limit: int) -> list[RepoTarget]:
        """
        Use Sourcegraph's search API to find repos containing a code pattern.

        Sourcegraph is far more powerful than GitHub code search for semantic
        patterns — it supports structural search (comby), regex, and symbol search
        across 2M+ public repos.
        """
        if not pattern:
            return []
        if self._sg_token:
            return await self._sourcegraph_search(pattern, limit)
        # Fall back to GitHub code search
        items = await self._gh_search(f"{pattern} archived:false", per_page=min(100, limit))
        return [self._gh_item_to_target(i, f"pattern_hunt:{pattern[:40]}") for i in items]

    async def _sourcegraph_search(self, pattern: str, limit: int) -> list[RepoTarget]:
        """Query Sourcegraph GraphQL search API."""
        query = """
        query Search($query: String!) {
          search(query: $query, version: V3) {
            results {
              repositories { name url defaultBranch { name } }
            }
          }
        }
        """
        headers = {"Authorization": f"token {self._sg_token}"}
        payload = {"query": query, "variables": {"query": f"{pattern} count:{limit}"}}
        try:
            async with self._session.post(
                "https://sourcegraph.com/.api/graphql",
                json=payload, headers=headers, timeout=30
            ) as r:
                if r.status != 200:
                    return []
                data = await r.json()
                repos = (
                    data.get("data", {})
                        .get("search", {})
                        .get("results", {})
                        .get("repositories", [])
                )
                targets = []
                for repo in repos[:limit]:
                    name_parts = repo.get("name", "/").split("/")
                    targets.append(RepoTarget(
                        owner       = name_parts[-2] if len(name_parts) >= 2 else "",
                        name        = name_parts[-1],
                        clone_url   = f"https://{repo.get('url', '')}",
                        default_branch = repo.get("defaultBranch", {}).get("name", "main"),
                        discovery_reason = f"sourcegraph:{pattern[:40]}",
                    ))
                return targets
        except Exception as exc:
            log.debug("[RepoHunter] Sourcegraph error: %s", exc)
            return []

    async def _org_sweep(self, org: str, limit: int) -> list[RepoTarget]:
        """All public repos in a GitHub organisation."""
        if not org or not self._gh_token:
            return []
        headers = {
            "Authorization": f"Bearer {self._gh_token}",
            "Accept": "application/vnd.github+json",
        }
        url = f"https://api.github.com/orgs/{org}/repos"
        params = {"type": "public", "per_page": min(100, limit), "sort": "pushed"}
        async with self._sem:
            try:
                async with self._session.get(url, headers=headers, params=params, timeout=30) as r:
                    if r.status != 200:
                        return []
                    items = await r.json()
                    return [self._gh_item_to_target(i, f"org_sweep:{org}") for i in items]
            except Exception as exc:
                log.debug("[RepoHunter] org_sweep error: %s", exc)
                return []

    async def _trending(self, language: str, limit: int) -> list[RepoTarget]:
        """GitHub trending repos (high visibility, fixes merged quickly)."""
        query = f"language:{language} pushed:>2024-01-01 archived:false stars:>500"
        items = await self._gh_search(query, per_page=min(100, limit))
        return [self._gh_item_to_target(i, f"trending:{language}") for i in items]

    async def _bugbounty_targets(self, language: str, limit: int) -> list[RepoTarget]:
        """
        Repos from organisations with active bug bounty programmes.

        Known high-value bug bounty orgs: mozilla, apache, google, microsoft,
        torvalds (linux kernel), kubernetes, istio, envoyproxy, etc.
        These organisations pay for CVEs and merge fixes rapidly.
        """
        bb_orgs = [
            "mozilla", "apache", "kubernetes", "istio", "envoyproxy",
            "openssl", "curl", "openssh", "nodejs", "python",
            "chromium", "android", "google", "microsoft", "facebook",
        ]
        tasks = [self._org_sweep(org, limit // len(bb_orgs) + 1) for org in bb_orgs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        targets = []
        for r in results:
            if isinstance(r, list):
                targets.extend(r)
        # Filter by language if specified
        if language:
            targets = [t for t in targets if t.language.lower() == language.lower()]
        return targets[:limit]

    # ── Cloning ───────────────────────────────────────────────────────────────

    async def clone(self, target: RepoTarget, dest_dir: str) -> str:
        """
        Shallow-clone a repo target into dest_dir/<owner>_<name>.

        Returns the local path or empty string on failure.
        Shallow clone (depth=1) is sufficient for static analysis and minimises
        disk/bandwidth. For git history secret scanning, pass depth=0.
        """
        import subprocess, os as _os
        local_name = f"{target.owner}_{target.name}"
        dest = _os.path.join(dest_dir, local_name)
        if _os.path.exists(dest):
            log.debug("[RepoHunter] Already cloned: %s", dest)
            target.local_path = dest
            return dest
        cmd = [
            "git", "clone", "--depth=1",
            f"--branch={target.default_branch}",
            target.clone_url, dest,
        ]
        loop = asyncio.get_event_loop()
        try:
            proc = await loop.run_in_executor(
                None,
                lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=120),
            )
            if proc.returncode != 0:
                log.warning(
                    "[RepoHunter] Clone failed %s: %s",
                    target.clone_url, proc.stderr[:200],
                )
                return ""
            target.local_path = dest
            log.info("[RepoHunter] Cloned %s → %s", target.clone_url, dest)
            return dest
        except Exception as exc:
            log.warning("[RepoHunter] Clone error %s: %s", target.clone_url, exc)
            return ""

    # ── Conversion and scoring ────────────────────────────────────────────────

    @staticmethod
    def _gh_item_to_target(item: dict, reason: str) -> RepoTarget:
        owner = item.get("owner", {}).get("login", "") or ""
        return RepoTarget(
            owner           = owner,
            name            = item.get("name", ""),
            clone_url       = item.get("clone_url", ""),
            default_branch  = item.get("default_branch", "main"),
            language        = (item.get("language") or "").lower(),
            stars           = item.get("stargazers_count", 0),
            forks           = item.get("forks_count", 0),
            open_issues     = item.get("open_issues_count", 0),
            size_kb         = item.get("size", 0),
            last_pushed_at  = item.get("pushed_at", ""),
            description     = (item.get("description") or "")[:200],
            topics          = item.get("topics", []),
            discovery_reason= reason,
        )

    @staticmethod
    def _score(t: RepoTarget) -> float:
        """
        Heuristic priority score [0, 10].

        Higher = more likely to yield high-severity, mergeable findings.
        """
        score = 0.0
        # Language risk
        score += _LANG_RISK.get(t.language, 0.4) * 3.0
        # Community size (log-scaled — very large repos have diminishing returns)
        import math
        score += min(2.0, math.log10(max(1, t.stars)) * 0.5)
        score += min(1.0, math.log10(max(1, t.forks)) * 0.4)
        # Active maintenance — recent push = fix will be merged
        if t.last_pushed_at:
            try:
                from datetime import datetime, timezone
                pushed = datetime.fromisoformat(
                    t.last_pushed_at.rstrip("Z")
                ).replace(tzinfo=timezone.utc)
                age_days = (datetime.now(timezone.utc) - pushed).days
                if age_days < 30:
                    score += 1.5
                elif age_days < 180:
                    score += 0.8
            except Exception:
                pass
        # Open issues — proxy for "people care about bugs here"
        score += min(1.0, t.open_issues / 1000)
        # High-value topics
        hv_topics = {"security", "cryptography", "authentication", "networking",
                     "blockchain", "embedded", "kernel", "firmware", "iot"}
        if hv_topics & set(t.topics):
            score += 1.0
        return round(score, 3)
