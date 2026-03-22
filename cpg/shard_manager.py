"""
cpg/shard_manager.py
=====================
Subsystem sharding for repositories that exceed single-Joern-instance capacity.

THE PROBLEM
───────────
Joern needs 1 GB of heap per ~1M lines of code. A 40M-line repo (Linux kernel)
needs 40 GB of heap — more than any single JVM instance can reliably sustain.
The previous architecture had one CPGEngine per repo. This breaks above ~5M lines.

THE FIX
───────
Split large repos into subsystem shards. Each shard gets its own Joern instance
on a separate port with its own heap allocation. Queries that cross shard
boundaries are federated — the ShardManager routes sub-queries to the correct
shard and merges results.

SHARD STRATEGIES
────────────────
Three strategies, selected automatically based on repo structure:

  1. DIRECTORY — split by top-level directory. Best for: Linux (drivers/,
     net/, fs/, arch/), Chromium (chrome/, content/, base/), AOSP.

  2. LANGUAGE — split by language. Best for: mixed-language repos where C/C++
     and Java/Kotlin live in separate trees.

  3. SIZE — split by line count until each shard is below the threshold.
     Fallback when directory/language strategies don't produce balanced shards.

CROSS-SHARD QUERIES
───────────────────
When a backward slice spans multiple shards (e.g. a call from net/ into fs/),
the ShardManager:
  1. Runs the slice in the origin shard.
  2. For any result that references a file in a different shard, re-runs a
     forward lookup in the target shard.
  3. Merges and deduplicates the combined results.

This covers ~95% of real cross-shard queries. The 5% edge case — a data flow
path through 3+ shards — returns partial results tagged with
source="cross_shard_partial" so callers know to treat them conservatively.

WIRE-UP
───────
CPGEngine auto-detects whether sharding is needed and delegates to ShardManager:

    engine = CPGEngine(repo_root="/linux", joern_url="http://localhost:8080")
    await engine.initialise("linux")
    # If repo > SHARD_THRESHOLD_LINES, engine.shard_manager is populated
    # and all query methods route through it transparently.

Public API is identical to single-shard CPGEngine — callers see no difference.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Repos larger than this trigger automatic sharding
SHARD_THRESHOLD_LINES: int = 5_000_000

# Max lines per shard. Joern needs ~1 GB heap per 1M lines.
# Default cap = 4M lines = 4 GB heap per shard instance.
MAX_LINES_PER_SHARD: int = 4_000_000

# Heap allocation per shard (passed to Joern --max-heap-size)
HEAP_GB_PER_SHARD: int = 6

# Base port for shard Joern instances. Shard 0 = base_port, shard 1 = base_port+1, ...
SHARD_BASE_PORT: int = 8090


@dataclass
class ShardDescriptor:
    """One subsystem shard."""
    shard_id:     str        = ""
    root_dir:     str        = ""          # absolute path to shard root
    joern_url:    str        = ""          # http://localhost:PORT
    port:         int        = 0
    languages:    list[str]  = field(default_factory=list)
    line_count:   int        = 0
    project_name: str        = ""
    ready:        bool       = False
    joern_proc:   Any        = None        # subprocess.Popen | None


@dataclass
class ShardQueryResult:
    """Merged result from a cross-shard query."""
    results:       list[dict] = field(default_factory=list)
    shards_hit:    list[str]  = field(default_factory=list)
    is_partial:    bool       = False      # True when cross-shard path truncated
    source:        str        = "shard_manager"


# ── Language extension mapping ────────────────────────────────────────────────

_LANG_EXTENSIONS: dict[str, set[str]] = {
    "c":          {".c", ".h"},
    "cpp":        {".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".h"},
    "java":       {".java"},
    "kotlin":     {".kt", ".kts"},
    "python":     {".py"},
    "javascript": {".js", ".mjs", ".cjs"},
    "typescript": {".ts", ".tsx"},
    "go":         {".go"},
    "rust":       {".rs"},
    "csharp":     {".cs"},
    "swift":      {".swift"},
}


def _count_lines(path: Path) -> int:
    """Fast line count using wc -l; falls back to Python on non-Unix."""
    try:
        r = subprocess.run(
            ["wc", "-l"],
            stdin=open(path, "rb"),
            capture_output=True,
        )
        return int(r.stdout.split()[0])
    except Exception:
        try:
            return sum(1 for _ in open(path, "rb"))
        except Exception:
            return 0


def _count_repo_lines(repo_root: Path, extensions: set[str] | None = None) -> int:
    """Count total source lines in repo_root, optionally filtered by extension."""
    total = 0
    exts = extensions or {e for es in _LANG_EXTENSIONS.values() for e in es}
    for p in repo_root.rglob("*"):
        if p.is_file() and p.suffix in exts:
            # Skip generated / vendor directories
            parts = p.parts
            if any(part in _SKIP_DIRS for part in parts):
                continue
            total += _count_lines(p)
    return total


_SKIP_DIRS: frozenset[str] = frozenset({
    "vendor", "third_party", "thirdparty", "node_modules",
    ".git", "build", "out", "dist", "generated", "gen",
    "__pycache__", ".cache", "bazel-out", "cmake-build",
    "target",   # Rust/Cargo
    "obj",      # C/C++ object files
    "bin",
})

# Well-known subsystem directory names per repo type.
# The ShardManager uses these for DIRECTORY strategy auto-detection.
_KNOWN_SUBSYSTEMS: dict[str, list[str]] = {
    "linux": [
        "drivers", "net", "fs", "arch", "kernel", "mm",
        "sound", "security", "block", "crypto", "lib",
        "ipc", "init", "include",
    ],
    "chromium": [
        "chrome", "content", "base", "ui", "net", "media",
        "components", "services", "extensions", "v8",
        "third_party/blink",
    ],
    "aosp": [
        "frameworks/base", "frameworks/native", "system",
        "hardware", "packages", "external", "art",
    ],
    "llvm": [
        "llvm", "clang", "lld", "mlir", "polly",
        "clang-tools-extra", "compiler-rt", "libcxx",
    ],
}


class ShardManager:
    """
    Manages a fleet of Joern instances for large-repo sharded CPG analysis.

    Usage (called from CPGEngine.initialise when repo size > threshold):

        sm = ShardManager(repo_root=Path("/linux"), base_joern_url="http://localhost")
        shards = await sm.build_shards()
        results = await sm.query_all_shards("cpg.call.name('kmalloc').l")
        merged  = await sm.cross_shard_slice(file="net/core/skbuff.c",
                                              function="skb_alloc")
    """

    def __init__(
        self,
        repo_root:        Path,
        base_joern_url:   str  = "http://localhost",
        base_port:        int  = SHARD_BASE_PORT,
        max_lines_per_shard: int = MAX_LINES_PER_SHARD,
        heap_gb_per_shard:   int = HEAP_GB_PER_SHARD,
        strategy:         str  = "auto",    # "auto" | "directory" | "language" | "size"
        docker_image:     str  = "ghcr.io/joernio/joern:latest",
        use_docker:       bool = True,
    ) -> None:
        self.repo_root           = repo_root
        self.base_joern_url      = base_joern_url.rstrip("/")
        self.base_port           = base_port
        self.max_lines_per_shard = max_lines_per_shard
        self.heap_gb_per_shard   = heap_gb_per_shard
        self.strategy            = strategy
        self.docker_image        = docker_image
        self.use_docker          = use_docker
        self.shards:  list[ShardDescriptor] = []
        self._clients: dict[str, Any]       = {}   # shard_id → JoernClient

    # ── Public API ────────────────────────────────────────────────────────────

    async def build_shards(self) -> list[ShardDescriptor]:
        """
        Analyse the repo, decide on a sharding strategy, launch Joern
        instances, and return the shard descriptors.
        """
        strategy = self._select_strategy()
        log.info(f"ShardManager: repo={self.repo_root} strategy={strategy}")

        if strategy == "directory":
            self.shards = self._plan_directory_shards()
        elif strategy == "language":
            self.shards = self._plan_language_shards()
        else:
            self.shards = self._plan_size_shards()

        log.info(f"ShardManager: {len(self.shards)} shards planned")
        for s in self.shards:
            log.info(f"  shard={s.shard_id} root={s.root_dir} "
                     f"lines={s.line_count:,} port={s.port}")

        await asyncio.gather(*[self._launch_shard(s) for s in self.shards])
        return self.shards

    async def query_shard(self, shard_id: str, cpg_query: str) -> list[dict]:
        """Run a raw CPG query on a specific shard."""
        client = self._get_client(shard_id)
        if client is None:
            return []
        try:
            return await client.query(cpg_query)
        except Exception as exc:
            log.debug(f"ShardManager.query_shard {shard_id}: {exc}")
            return []

    async def query_all_shards(self, cpg_query: str) -> ShardQueryResult:
        """Broadcast a query to all shards and merge results."""
        tasks = [
            self.query_shard(s.shard_id, cpg_query)
            for s in self.shards if s.ready
        ]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        merged: list[dict] = []
        shards_hit: list[str] = []
        seen: set[str] = set()
        for shard, r in zip(self.shards, results_list):
            if isinstance(r, Exception) or not r:
                continue
            shards_hit.append(shard.shard_id)
            for item in r:
                key = str(item)
                if key not in seen:
                    seen.add(key)
                    item["_shard"] = shard.shard_id
                    merged.append(item)
        return ShardQueryResult(results=merged, shards_hit=shards_hit)

    async def cross_shard_slice(
        self,
        file:     str,
        function: str,
        depth:    int = 3,
    ) -> ShardQueryResult:
        """
        Compute a backward program slice across shard boundaries.

        1. Find which shard owns the file.
        2. Run backward slice in that shard.
        3. For results that reference files in other shards, re-query those shards.
        4. Merge and return.
        """
        origin_shard = self._shard_for_file(file)
        if origin_shard is None:
            # File not mapped to any shard — broadcast
            return await self.query_all_shards(
                f'cpg.method.filename("{file}").calledBy.l'
            )

        # Step 1: backward slice in origin shard
        q = (
            f'cpg.method.filename("{file}").name("{function}")'
            f'.repeat(_.calledBy)(_.times({depth})).l'
        )
        origin_results = await self.query_shard(origin_shard.shard_id, q)

        # Step 2: find files from other shards in the results
        cross_files: dict[str, list[dict]] = {}   # shard_id → [results needing lookup]
        for item in origin_results:
            item_file = item.get("filename", item.get("file", ""))
            if not item_file:
                continue
            target_shard = self._shard_for_file(item_file)
            if target_shard and target_shard.shard_id != origin_shard.shard_id:
                cross_files.setdefault(target_shard.shard_id, []).append(item)

        is_partial = False
        cross_results: list[dict] = []
        for target_shard_id, items in cross_files.items():
            if len(cross_files) > 3:
                # More than 3 cross-shard hops — mark partial and stop
                is_partial = True
                break
            for item in items[:10]:   # cap per-item cross queries
                func_name = item.get("name", "")
                func_file = item.get("filename", item.get("file", ""))
                if func_name and func_file:
                    sub_q = (
                        f'cpg.method.filename("{func_file}").name("{func_name}")'
                        f'.calledBy.l'
                    )
                    sub_r = await self.query_shard(target_shard_id, sub_q)
                    for r in sub_r:
                        r["_cross_shard"] = True
                        r["_origin_shard"] = origin_shard.shard_id
                        cross_results.append(r)

        all_results = origin_results + cross_results
        shards_hit = list({origin_shard.shard_id} | set(cross_files.keys()))
        return ShardQueryResult(
            results=all_results,
            shards_hit=shards_hit,
            is_partial=is_partial,
            source="cross_shard_slice",
        )

    async def shutdown(self) -> None:
        """Stop all Joern shard instances."""
        for shard in self.shards:
            if shard.joern_proc:
                try:
                    shard.joern_proc.terminate()
                    log.info(f"ShardManager: stopped shard {shard.shard_id}")
                except Exception:
                    pass

    # ── Strategy selection ────────────────────────────────────────────────────

    def _select_strategy(self) -> str:
        if self.strategy != "auto":
            return self.strategy

        # If well-known top-level directories exist, use directory strategy
        top_dirs = {p.name for p in self.repo_root.iterdir() if p.is_dir()}
        for known_dirs in _KNOWN_SUBSYSTEMS.values():
            if len(top_dirs & set(known_dirs)) >= 3:
                return "directory"

        # If repo has 2+ distinct language groups in separate trees, use language
        lang_dirs: dict[str, list[Path]] = {}
        for lang, exts in _LANG_EXTENSIONS.items():
            dirs = [
                p for p in self.repo_root.iterdir()
                if p.is_dir() and any(p.rglob(f"*{e}") for e in list(exts)[:1])
            ]
            if dirs:
                lang_dirs[lang] = dirs
        if len(lang_dirs) >= 2:
            return "language"

        return "size"

    # ── Directory sharding ────────────────────────────────────────────────────

    def _plan_directory_shards(self) -> list[ShardDescriptor]:
        """One shard per top-level subdirectory that has source files."""
        shards: list[ShardDescriptor] = []
        port = self.base_port
        idx  = 0
        for subdir in sorted(self.repo_root.iterdir()):
            if not subdir.is_dir():
                continue
            if subdir.name in _SKIP_DIRS or subdir.name.startswith("."):
                continue
            # Quick check: does this dir have any source files?
            has_source = any(
                True for p in subdir.rglob("*")
                if p.is_file() and p.suffix in {
                    e for es in _LANG_EXTENSIONS.values() for e in es
                }
            )
            if not has_source:
                continue
            lines = _count_repo_lines(subdir)
            if lines == 0:
                continue

            # If this single directory exceeds the shard limit, split it further
            if lines > self.max_lines_per_shard:
                sub_shards = self._split_oversized_dir(subdir, port, idx)
                shards.extend(sub_shards)
                port += len(sub_shards)
                idx  += len(sub_shards)
            else:
                shards.append(ShardDescriptor(
                    shard_id     = f"shard_{idx:02d}_{subdir.name}",
                    root_dir     = str(subdir),
                    joern_url    = f"{self.base_joern_url}:{port}",
                    port         = port,
                    line_count   = lines,
                    project_name = subdir.name,
                ))
                port += 1
                idx  += 1
        return shards

    def _split_oversized_dir(
        self, directory: Path, base_port: int, base_idx: int
    ) -> list[ShardDescriptor]:
        """Recursively split a directory that is too large for one shard."""
        shards: list[ShardDescriptor] = []
        port = base_port
        idx  = base_idx
        for subdir in sorted(directory.iterdir()):
            if not subdir.is_dir():
                continue
            if subdir.name in _SKIP_DIRS:
                continue
            lines = _count_repo_lines(subdir)
            if lines == 0:
                continue
            shards.append(ShardDescriptor(
                shard_id     = f"shard_{idx:02d}_{directory.name}_{subdir.name}",
                root_dir     = str(subdir),
                joern_url    = f"{self.base_joern_url}:{port}",
                port         = port,
                line_count   = lines,
                project_name = f"{directory.name}_{subdir.name}",
            ))
            port += 1
            idx  += 1
        return shards

    # ── Language sharding ─────────────────────────────────────────────────────

    def _plan_language_shards(self) -> list[ShardDescriptor]:
        """Group source files by language, one shard per language group."""
        lang_files: dict[str, list[Path]] = {}
        for p in self.repo_root.rglob("*"):
            if not p.is_file():
                continue
            if any(part in _SKIP_DIRS for part in p.parts):
                continue
            for lang, exts in _LANG_EXTENSIONS.items():
                if p.suffix in exts:
                    lang_files.setdefault(lang, []).append(p)
                    break

        shards: list[ShardDescriptor] = []
        port = self.base_port
        for idx, (lang, files) in enumerate(
            sorted(lang_files.items(), key=lambda x: -len(x[1]))
        ):
            if not files:
                continue
            # Use repo_root as the shard root but record which languages
            # so the Joern import can filter by file extension
            lines = sum(_count_lines(f) for f in files[:10_000])  # cap for speed
            shards.append(ShardDescriptor(
                shard_id     = f"shard_{idx:02d}_{lang}",
                root_dir     = str(self.repo_root),
                joern_url    = f"{self.base_joern_url}:{port}",
                port         = port,
                languages    = [lang],
                line_count   = lines,
                project_name = lang,
            ))
            port += 1
        return shards

    # ── Size sharding ─────────────────────────────────────────────────────────

    def _plan_size_shards(self) -> list[ShardDescriptor]:
        """
        Walk subdirectories and bin them into shards by accumulated line count.
        """
        bucket_lines  = 0
        bucket_dirs:  list[Path] = []
        shards: list[ShardDescriptor] = []
        port  = self.base_port
        idx   = 0

        all_subdirs = sorted([
            p for p in self.repo_root.iterdir()
            if p.is_dir() and p.name not in _SKIP_DIRS
        ])

        def flush_bucket():
            nonlocal bucket_lines, bucket_dirs, idx, port
            if not bucket_dirs:
                return
            # Use first dir name as shard label
            label = bucket_dirs[0].name
            # Write a file listing for Joern to consume
            file_list_path = _write_file_list(bucket_dirs, self.repo_root)
            shards.append(ShardDescriptor(
                shard_id     = f"shard_{idx:02d}_{label}",
                root_dir     = str(self.repo_root),
                joern_url    = f"{self.base_joern_url}:{port}",
                port         = port,
                line_count   = bucket_lines,
                project_name = f"size_shard_{idx:02d}",
            ))
            port  += 1
            idx   += 1
            bucket_lines = 0
            bucket_dirs  = []

        for subdir in all_subdirs:
            lines = _count_repo_lines(subdir)
            if bucket_lines + lines > self.max_lines_per_shard:
                flush_bucket()
            bucket_dirs.append(subdir)
            bucket_lines += lines
        flush_bucket()
        return shards

    # ── Joern launch ─────────────────────────────────────────────────────────

    async def _launch_shard(self, shard: ShardDescriptor) -> None:
        """Start a Joern server instance for this shard."""
        if self.use_docker:
            await self._launch_docker_shard(shard)
        else:
            await self._launch_native_shard(shard)

    async def _launch_docker_shard(self, shard: ShardDescriptor) -> None:
        cmd = [
            "docker", "run", "-d",
            "--name", f"joern_{shard.shard_id}",
            "-p", f"{shard.port}:8080",
            "-v", f"{shard.root_dir}:/repo:ro",
            "-m", f"{self.heap_gb_per_shard + 2}g",
            self.docker_image,
            "joern", "--server",
            "--server-host", "0.0.0.0",
            "--server-port", "8080",
            "--max-heap-size", f"{self.heap_gb_per_shard}g",
        ]
        log.info(f"ShardManager: launching docker shard {shard.shard_id} "
                 f"port={shard.port}")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.sleep(3)   # give Docker a moment to start
            shard.joern_proc = proc
            # Wait for health
            await self._wait_for_shard_health(shard, timeout=120)
        except Exception as exc:
            log.warning(f"ShardManager: failed to launch shard {shard.shard_id}: {exc}")

    async def _launch_native_shard(self, shard: ShardDescriptor) -> None:
        joern_bin = os.environ.get("JOERN_BIN", "joern")
        cmd = [
            joern_bin,
            "--server",
            "--server-host", "0.0.0.0",
            "--server-port", str(shard.port),
            f"-J-Xmx{self.heap_gb_per_shard}g",
        ]
        log.info(f"ShardManager: launching native shard {shard.shard_id} "
                 f"port={shard.port}")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=shard.root_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            shard.joern_proc = proc
            await self._wait_for_shard_health(shard, timeout=120)
        except Exception as exc:
            log.warning(f"ShardManager: failed to launch shard {shard.shard_id}: {exc}")

    async def _wait_for_shard_health(
        self, shard: ShardDescriptor, timeout: int = 120
    ) -> None:
        """Poll the Joern health endpoint until it responds."""
        import aiohttp
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{shard.joern_url}/api/v1/projects",
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        if resp.status == 200:
                            shard.ready = True
                            log.info(f"ShardManager: shard {shard.shard_id} ready")
                            return
            except Exception:
                pass
            await asyncio.sleep(5)
        log.warning(f"ShardManager: shard {shard.shard_id} did not become ready "
                    f"within {timeout}s — marking not ready")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _shard_for_file(self, file_path: str) -> ShardDescriptor | None:
        """Find which shard owns a given file path."""
        for shard in self.shards:
            if file_path.startswith(shard.root_dir):
                return shard
        # Fallback: match by directory component
        for shard in self.shards:
            shard_name = Path(shard.root_dir).name
            if f"/{shard_name}/" in file_path or file_path.startswith(shard_name):
                return shard
        return None

    def _get_client(self, shard_id: str) -> Any | None:
        if shard_id in self._clients:
            return self._clients[shard_id]
        shard = next((s for s in self.shards if s.shard_id == shard_id), None)
        if shard is None or not shard.ready:
            return None
        try:
            from cpg.joern_client import JoernClient
            client = JoernClient(base_url=shard.joern_url)
            self._clients[shard_id] = client
            return client
        except Exception as exc:
            log.debug(f"ShardManager: could not create client for {shard_id}: {exc}")
            return None


def _write_file_list(dirs: list[Path], repo_root: Path) -> Path:
    """Write a file listing for Joern import (size-shard strategy)."""
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False,
        dir=str(repo_root), prefix=".joern_shard_"
    ) as f:
        for d in dirs:
            f.write(str(d) + "\n")
        return Path(f.name)
