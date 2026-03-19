"""
swarm/hf_skills.py
===================
HuggingFace Skills integration for the Rhodawk AI swarm.

https://github.com/huggingface/smolagents

Provides the swarm with reusable, pre-built agent skills:
• WebSearchSkill      — DuckDuckGo search
• CodeExecSkill       — safe Python execution
• DocRetrievalSkill   — semantic document retrieval
• SummarizationSkill  — long-context summarization
• FileAnalysisSkill   — file type detection, binary analysis

Skills are invoked by swarm agents as tool calls via the ToolHive MCP layer
OR directly via the Python API when lower latency is needed.

Usage::

    registry = SkillRegistry()
    result = await registry.run("web_search", {"query": "Python memory leak detection"})
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine

log = logging.getLogger(__name__)

_SMOLAGENTS_AVAILABLE = False
try:
    import smolagents  # type: ignore[import]
    _SMOLAGENTS_AVAILABLE = True
    log.info(f"smolagents {smolagents.__version__} available")
except ImportError:
    log.info(
        "smolagents not installed — HF Skills using MCP fallback. "
        "Run: pip install smolagents"
    )

SkillFn = Callable[..., Coroutine[Any, Any, Any]]


# ──────────────────────────────────────────────────────────────────────────────
# Individual skill implementations
# ──────────────────────────────────────────────────────────────────────────────

async def web_search_skill(query: str, max_results: int = 5) -> list[dict]:
    """Search the web. Uses smolagents DuckDuckGoSearchTool if available."""
    # Delegate to HF Skills MCP server via ToolHive
    try:
        from tools.toolhive import ToolHive
        th = ToolHive(enabled_tools=["hf_skills"])
        result = await th.call("hf_skills", "tools/call", {
            "name": "web_search",
            "arguments": {"query": query, "max_results": max_results},
        })
        if result:
            return result
    except Exception:
        pass

    # Direct fallback
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1}
            )
            data = r.json()
            return [
                {"title": item.get("Text","")[:80], "url": item.get("FirstURL","")}
                for item in data.get("RelatedTopics", [])[:max_results]
                if isinstance(item, dict) and "Text" in item
            ]
    except Exception as exc:
        log.debug(f"web_search_skill fallback failed: {exc}")
        return []


async def code_execution_skill(code: str, timeout: int = 30) -> dict:
    """Execute Python code in a sandbox."""
    from tools.servers.huggingface_skills_server import code_execution
    return await code_execution(code, timeout)


async def summarization_skill(content: str, max_words: int = 200) -> str:
    """Summarize long content."""
    try:
        import litellm
        from models.router import get_router
        model = get_router().primary_model("triage")
        resp = await litellm.acompletion(
            model=model,
            messages=[{"role": "user",
                        "content": f"Summarize in {max_words} words:\n\n{content[:4000]}"}],
            max_tokens=int(max_words * 1.5),
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        return f"Summarization failed: {exc}"


async def doc_retrieval_skill(query: str, n: int = 5) -> list[dict]:
    """Retrieve relevant code chunks from the vector memory."""
    try:
        from memory.openviking import OpenVikingDB
        db = OpenVikingDB()
        db.initialise()
        results = db.search(query, n=n)
        return [
            {"file": r.file_path, "lines": f"{r.line_start}-{r.line_end}",
             "summary": r.summary, "score": r.score}
            for r in results
        ]
    except Exception as exc:
        log.debug(f"doc_retrieval_skill failed: {exc}")
        return []


async def file_analysis_skill(file_path: str) -> dict:
    """Analyze a file: type, size, encoding, binary/text."""
    from pathlib import Path
    import mimetypes
    p = Path(file_path)
    if not p.exists():
        return {"error": f"File not found: {file_path}"}
    mime, _ = mimetypes.guess_type(str(p))
    size    = p.stat().st_size
    is_binary = False
    try:
        p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        is_binary = True
    return {
        "file_path": file_path,
        "size_bytes": size,
        "mime_type":  mime or "application/octet-stream",
        "is_binary":  is_binary,
        "extension":  p.suffix,
        "lines":      len(p.read_text(errors="replace").splitlines()) if not is_binary else 0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Skill registry
# ──────────────────────────────────────────────────────────────────────────────

class SkillRegistry:
    """
    Central registry for all HuggingFace Skills.

    Agents call skills by name:

        registry = SkillRegistry()
        result = await registry.run("web_search", {"query": "..."})
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillFn] = {
            "web_search":       web_search_skill,
            "code_execution":   code_execution_skill,
            "summarization":    summarization_skill,
            "doc_retrieval":    doc_retrieval_skill,
            "file_analysis":    file_analysis_skill,
        }

    def register(self, name: str, fn: SkillFn) -> None:
        """Register a custom skill."""
        self._skills[name] = fn
        log.info(f"SkillRegistry: registered skill '{name}'")

    async def run(self, skill_name: str, arguments: dict) -> Any:
        """Execute a skill by name."""
        fn = self._skills.get(skill_name)
        if not fn:
            raise ValueError(f"Unknown skill: {skill_name}. Available: {list(self._skills)}")
        return await fn(**arguments)

    def list_skills(self) -> list[str]:
        return list(self._skills.keys())

    def is_available(self, skill_name: str) -> bool:
        return skill_name in self._skills


# Module-level singleton
_registry: SkillRegistry | None = None


def get_skill_registry() -> SkillRegistry:
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry
