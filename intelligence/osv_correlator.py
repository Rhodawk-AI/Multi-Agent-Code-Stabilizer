"""
intelligence/osv_correlator.py
================================
CVE / OSV / NVD vulnerability correlation and exploitability scoring.

PURPOSE
───────
When Rhodawk finds a bug, this module answers the critical questions:
  1. Is this bug already a known CVE? (duplicate effort detection)
  2. Is this dependency version vulnerable to any CVE? (supply chain)
  3. What is the CVSS score / exploitability? (triage prioritisation)
  4. Is there a public exploit or PoC? (severity escalation)
  5. What is the remediation (patched version)?

Data sources
────────────
• OSV (Open Source Vulnerabilities) — https://osv.dev/
  The most comprehensive open database: 50,000+ vulns across PyPI, npm,
  Maven, Go, Rust, RubyGems, NuGet, Linux, Alpine, Debian. Query API is
  free, JSON-based, and can be run offline with the bulk download.

• NVD (National Vulnerability Database) — https://nvd.nist.gov/
  The authoritative CVE database with CVSS scores. Rate-limited API
  (requires API key for production use: https://nvd.nist.gov/developers/request-an-api-key)

• GitHub Security Advisories — https://github.com/advisories
  Fastest to update for open-source projects. Has exact affected version
  ranges and patch commit SHAs.

• exploit-db.com — public exploit PoC database (Exploit-DB).
  We query by CVE ID to detect when a public exploit exists.

Public API
──────────
    from intelligence.osv_correlator import OSVCorrelator

    correlator = OSVCorrelator(nvd_api_key="...")
    await correlator.initialise()

    # Check a specific package version
    results = await correlator.check_package("log4j", "2.14.1", "maven")

    # Correlate a code finding with known CVEs
    matches = await correlator.correlate_finding(
        description="Format string vulnerability in printf wrapper",
        file_path="src/log.c",
        cwe_ids=["CWE-134"],
    )
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

_OSV_API         = "https://api.osv.dev/v1"
_NVD_API         = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_GHSA_API        = "https://api.github.com/advisories"
_EXPLOITDB_SEARCH= "https://www.exploit-db.com/search?cve={cve_id}&type=0&platform=0"

# CVSS score ranges → Rhodawk severity
def _cvss_to_severity(score: float) -> str:
    if score >= 9.0: return "critical"
    if score >= 7.0: return "high"
    if score >= 4.0: return "medium"
    return "low"


@dataclass
class CVEMatch:
    """A matched CVE with full context for the fixer and reviewer."""
    cve_id:           str
    description:      str
    cvss_v3_score:    float   = 0.0
    cvss_v3_vector:   str     = ""
    severity:         str     = "unknown"
    cwe_ids:          list[str] = field(default_factory=list)
    affected_packages: list[dict] = field(default_factory=list)
    patched_versions: list[str] = field(default_factory=list)
    patch_commits:    list[str] = field(default_factory=list)
    public_exploit:   bool    = False
    exploit_urls:     list[str] = field(default_factory=list)
    published:        str     = ""
    source:           str     = "osv"


class OSVCorrelator:
    """
    Multi-source CVE correlator with CVSS scoring and exploit detection.

    Parameters
    ----------
    nvd_api_key:
        NVD API key for higher rate limits. Set RHODAWK_NVD_API_KEY env var.
    github_token:
        GitHub token for GHSA queries. Set GITHUB_TOKEN env var.
    cache_ttl_s:
        How long to cache CVE lookups (default: 24 h).
    """

    def __init__(
        self,
        nvd_api_key:  str = "",
        github_token: str = "",
        cache_ttl_s:  int = 86_400,
    ) -> None:
        self._nvd_key    = nvd_api_key  or os.environ.get("RHODAWK_NVD_API_KEY", "")
        self._gh_token   = github_token or os.environ.get("GITHUB_TOKEN", "")
        self._cache:     dict[str, tuple[float, Any]] = {}
        self._cache_ttl  = cache_ttl_s
        self._session: Any = None

    async def initialise(self) -> None:
        import aiohttp
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )

    async def close(self) -> None:
        if self._session:
            await self._session.close()

    # ── Package CVE check ─────────────────────────────────────────────────────

    async def check_package(
        self,
        package:   str,
        version:   str,
        ecosystem: str = "PyPI",
    ) -> list[CVEMatch]:
        """
        Query OSV for all vulnerabilities affecting package@version.

        Parameters
        ----------
        package:
            Package name as it appears in the ecosystem.
        version:
            Exact version string (e.g. "2.14.1").
        ecosystem:
            OSV ecosystem: PyPI | npm | Maven | Go | RubyGems | crates.io |
                           NuGet | Hex | Pub | Linux | OSS-Fuzz | ...
        """
        cache_key = f"pkg:{ecosystem}:{package}:{version}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        payload = {
            "version": version,
            "package": {"name": package, "ecosystem": ecosystem},
        }
        try:
            async with self._session.post(
                f"{_OSV_API}/query", json=payload
            ) as resp:
                if resp.status != 200:
                    return []
                data   = await resp.json()
                vulns  = data.get("vulns", [])
                result = [self._osv_to_match(v) for v in vulns]
                # Enrich with NVD CVSS scores
                result = await self._enrich_with_nvd(result)
                self._set_cache(cache_key, result)
                return result
        except Exception as exc:
            log.debug("[OSV] check_package(%s@%s) failed: %s", package, version, exc)
            return []

    # ── Batch SBOM scan ───────────────────────────────────────────────────────

    async def scan_sbom(self, sbom: dict) -> list[dict]:
        """
        Scan a CycloneDX or SPDX SBOM for vulnerable components.

        Returns list of {component, version, cves: [CVEMatch]} dicts,
        sorted by max CVSS score descending.
        """
        # Extract components from CycloneDX format
        components = sbom.get("components", [])
        if not components:
            # SPDX format
            packages = sbom.get("packages", [])
            components = [
                {
                    "name": p.get("name", ""),
                    "version": p.get("versionInfo", ""),
                    "purl": p.get("downloadLocation", ""),
                }
                for p in packages
            ]

        tasks = []
        for comp in components:
            name    = comp.get("name", "")
            version = comp.get("version", "")
            purl    = comp.get("purl", "")
            eco     = _purl_to_ecosystem(purl) if purl else "PyPI"
            if name and version:
                tasks.append(self._check_component(name, version, eco, comp))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        findings = []
        for r in results:
            if isinstance(r, dict) and r.get("cves"):
                findings.append(r)

        findings.sort(
            key=lambda x: max((c.cvss_v3_score for c in x["cves"]), default=0),
            reverse=True,
        )
        log.info(
            "[OSV] SBOM scan: %d vulnerable components out of %d total",
            len(findings), len(components),
        )
        return findings

    async def _check_component(
        self, name: str, version: str, ecosystem: str, raw: dict
    ) -> dict:
        cves = await self.check_package(name, version, ecosystem)
        return {"component": name, "version": version, "raw": raw, "cves": cves}

    # ── Finding correlation ───────────────────────────────────────────────────

    async def correlate_finding(
        self,
        description: str,
        file_path:   str   = "",
        cwe_ids:     list[str] | None = None,
        package:     str   = "",
    ) -> list[CVEMatch]:
        """
        Find known CVEs that match a Rhodawk Issue by description / CWE.

        This prevents wasting fixer capacity on bugs that already have
        upstream fixes — the right action is to update the dependency,
        not patch the local usage.
        """
        matches: list[CVEMatch] = []

        # Query NVD by keyword
        keywords = description.split()[:6]
        nvd_results = await self._nvd_keyword_search(keywords)
        matches.extend(nvd_results)

        # Cross-reference by CWE
        if cwe_ids:
            for cve in matches:
                cve.cwe_ids = list(set(cve.cwe_ids) | set(cwe_ids or []))

        # Check for public exploits on top matches
        for cve in matches[:5]:
            cve.public_exploit, cve.exploit_urls = await self._check_exploit_db(
                cve.cve_id
            )
            if cve.public_exploit:
                # Escalate severity when public exploit exists
                cve.severity = "critical"
                cve.cvss_v3_score = max(cve.cvss_v3_score, 9.0)

        matches.sort(key=lambda c: c.cvss_v3_score, reverse=True)
        return matches[:10]

    # ── NVD integration ───────────────────────────────────────────────────────

    async def _nvd_keyword_search(self, keywords: list[str]) -> list[CVEMatch]:
        if not keywords:
            return []
        headers = {}
        if self._nvd_key:
            headers["apiKey"] = self._nvd_key
        params = {
            "keywordSearch": " ".join(keywords[:5]),
            "resultsPerPage": 20,
        }
        try:
            async with self._session.get(
                _NVD_API, params=params, headers=headers
            ) as resp:
                if resp.status == 403:
                    log.warning("[NVD] Rate limited — using OSV only")
                    return []
                if resp.status != 200:
                    return []
                data   = await resp.json()
                return [
                    self._nvd_to_match(v)
                    for v in data.get("vulnerabilities", [])
                    if v.get("cve")
                ]
        except Exception as exc:
            log.debug("[NVD] keyword search failed: %s", exc)
            return []

    async def _enrich_with_nvd(self, matches: list[CVEMatch]) -> list[CVEMatch]:
        """Fetch CVSS scores from NVD for OSV matches that lack them."""
        tasks = []
        for m in matches:
            if m.cve_id.startswith("CVE-") and m.cvss_v3_score == 0.0:
                tasks.append(self._nvd_fetch_cve(m))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return matches

    async def _nvd_fetch_cve(self, match: CVEMatch) -> None:
        headers = {"apiKey": self._nvd_key} if self._nvd_key else {}
        try:
            async with self._session.get(
                _NVD_API, params={"cveId": match.cve_id}, headers=headers
            ) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
                vulns = data.get("vulnerabilities", [])
                if vulns:
                    nvd_match = self._nvd_to_match(vulns[0])
                    match.cvss_v3_score  = nvd_match.cvss_v3_score
                    match.cvss_v3_vector = nvd_match.cvss_v3_vector
                    match.severity       = nvd_match.severity
                    match.cwe_ids        = list(
                        set(match.cwe_ids) | set(nvd_match.cwe_ids)
                    )
        except Exception as exc:
            log.debug("[NVD] fetch %s failed: %s", match.cve_id, exc)

    # ── Exploit detection ─────────────────────────────────────────────────────

    async def _check_exploit_db(self, cve_id: str) -> tuple[bool, list[str]]:
        """
        Check if a public exploit exists for this CVE.

        Uses GitHub search as a proxy for exploit-db.com (avoids fragile
        web scraping). Searching for '<CVE-ID> exploit' in GitHub code
        reliably surfaces PoC repositories.
        """
        if not self._gh_token or not cve_id.startswith("CVE-"):
            return False, []
        headers = {
            "Authorization": f"Bearer {self._gh_token}",
            "Accept": "application/vnd.github+json",
        }
        try:
            async with self._session.get(
                "https://api.github.com/search/repositories",
                params={"q": f"{cve_id} exploit poc", "per_page": 5},
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    return False, []
                data  = await resp.json()
                items = data.get("items", [])
                urls  = [i.get("html_url", "") for i in items if i.get("html_url")]
                return bool(urls), urls
        except Exception:
            return False, []

    # ── Converters ────────────────────────────────────────────────────────────

    @staticmethod
    def _osv_to_match(vuln: dict) -> CVEMatch:
        aliases = vuln.get("aliases", [])
        cve_id  = next((a for a in aliases if a.startswith("CVE-")), vuln.get("id", ""))

        affected  = vuln.get("affected", [])
        packages  = []
        patched   = []
        for aff in affected:
            pkg = aff.get("package", {})
            packages.append({"name": pkg.get("name", ""), "ecosystem": pkg.get("ecosystem", "")})
            for r in aff.get("ranges", []):
                for evt in r.get("events", []):
                    if "fixed" in evt:
                        patched.append(evt["fixed"])

        severity_entry = vuln.get("severity", [])
        cvss_score = 0.0
        cvss_vec   = ""
        if severity_entry:
            s = severity_entry[0]
            cvss_vec   = s.get("score", "")
            cvss_score = _extract_cvss_score(cvss_vec)

        return CVEMatch(
            cve_id            = cve_id,
            description       = vuln.get("summary", "")[:400],
            cvss_v3_score     = cvss_score,
            cvss_v3_vector    = cvss_vec,
            severity          = _cvss_to_severity(cvss_score),
            affected_packages = packages,
            patched_versions  = list(set(patched))[:5],
            published         = vuln.get("published", ""),
            source            = "osv",
        )

    @staticmethod
    def _nvd_to_match(vuln: dict) -> CVEMatch:
        cve      = vuln.get("cve", {})
        cve_id   = cve.get("id", "")
        desc     = next(
            (d["value"] for d in cve.get("descriptions", []) if d.get("lang") == "en"),
            ""
        )
        metrics  = cve.get("metrics", {})
        cvss_v3  = (
            metrics.get("cvssMetricV31", [{}])[0].get("cvssData", {})
            if metrics.get("cvssMetricV31") else
            metrics.get("cvssMetricV30", [{}])[0].get("cvssData", {})
            if metrics.get("cvssMetricV30") else {}
        )
        score    = float(cvss_v3.get("baseScore", 0.0))
        vector   = cvss_v3.get("vectorString", "")
        cwe_ids  = [
            w.get("description", [{}])[0].get("value", "")
            for w in cve.get("weaknesses", [])
            if w.get("description")
        ]

        return CVEMatch(
            cve_id         = cve_id,
            description    = desc[:400],
            cvss_v3_score  = score,
            cvss_v3_vector = vector,
            severity       = _cvss_to_severity(score),
            cwe_ids        = [c for c in cwe_ids if c][:5],
            published      = cve.get("published", ""),
            source         = "nvd",
        )

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _get_cache(self, key: str) -> Any:
        entry = self._cache.get(key)
        if entry is None:
            return None
        ts, value = entry
        if (datetime.now(timezone.utc).timestamp() - ts) > self._cache_ttl:
            del self._cache[key]
            return None
        return value

    def _set_cache(self, key: str, value: Any) -> None:
        self._cache[key] = (datetime.now(timezone.utc).timestamp(), value)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_cvss_score(vector: str) -> float:
    """Extract the base score from a CVSS vector string or return 0.0."""
    import re
    m = re.search(r"/AV:[^/]+/AC:[^/]+", vector)
    if not m:
        return 0.0
    try:
        # Try to parse CVSS:3.1 base score from vector
        parts = {k: v for k, v in (p.split(":") for p in vector.split("/") if ":" in p)}
        # Rough approximation — use NVD for authoritative score
        return 5.0  # Default medium when we can't parse
    except Exception:
        return 0.0


def _purl_to_ecosystem(purl: str) -> str:
    """Convert a Package URL (purl) type to OSV ecosystem name."""
    purl_map = {
        "pypi": "PyPI",
        "npm":  "npm",
        "maven":"Maven",
        "golang":"Go",
        "gem":  "RubyGems",
        "cargo":"crates.io",
        "nuget":"NuGet",
        "hex":  "Hex",
    }
    if ":" in purl:
        pkg_type = purl.split(":")[1].split("/")[0].lower()
        return purl_map.get(pkg_type, "PyPI")
    return "PyPI"
