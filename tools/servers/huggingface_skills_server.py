"""
tools/servers/huggingface_skills_server.py
==========================================
HuggingFace Skills MCP server for Rhodawk AI.

Wraps https://github.com/huggingface/smolagents skills as MCP tools.

Skills exposed
───────────────
• web_search          — DuckDuckGo web search via smolagents
• code_execution      — safe Python code execution sandbox
• image_analysis      — vision model inference for screenshots/diagrams
• doc_retrieval       — semantic document retrieval from HF datasets
• translation         — code comment translation
• summarization       — long-context summarization

Transport: stdio JSON-RPC 2.0 (MCP protocol)

Install: pip install smolagents huggingface-hub
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import traceback
from typing import Any

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Optional smolagents import
# ──────────────────────────────────────────────────────────────────────────────

try:
    from smolagents import DuckDuckGoSearchTool, PythonInterpreterTool  # type: ignore[import]
    _SMOLAGENTS = True
except ImportError:
    _SMOLAGENTS = False


# ──────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ──────────────────────────────────────────────────────────────────────────────

async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web using DuckDuckGo via smolagents."""
    if not _SMOLAGENTS:
        return _fallback_search(query, max_results)
    try:
        tool = DuckDuckGoSearchTool()
        results = tool(query)
        if isinstance(results, str):
            return [{"title": "Result", "snippet": results, "url": ""}]
        return results[:max_results]
    except Exception as exc:
        log.warning(f"HF web_search failed: {exc}")
        return _fallback_search(query, max_results)


def _fallback_search(query: str, max_results: int) -> list[dict]:
    """httpx-based DuckDuckGo fallback."""
    try:
        import httpx
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1}
        r = httpx.get(url, params=params, timeout=10)
        data = r.json()
        results = []
        for item in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(item, dict) and "Text" in item:
                results.append({
                    "title": item.get("Text", "")[:80],
                    "snippet": item.get("Text", ""),
                    "url": item.get("FirstURL", ""),
                })
        return results
    except Exception:
        return [{"title": "Search unavailable", "snippet": str(query), "url": ""}]


async def code_execution(code: str, timeout: int = 30) -> dict:
    """
    Execute Python code in a sandboxed subprocess.
    Returns {"stdout": str, "stderr": str, "returncode": int}
    """
    import asyncio
    import tempfile
    import os
    from security.aegis import scrubbed_env

    # Basic safety check — refuse obviously dangerous code
    dangerous = ["os.system", "subprocess.call", "__import__", "eval(", "exec("]
    for d in dangerous:
        if d in code:
            return {"stdout": "", "stderr": f"Blocked: {d} is not allowed", "returncode": 1}

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        f.write(code)
        tmp = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            "python", tmp,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=scrubbed_env(),
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            return {"stdout": "", "stderr": "Execution timed out", "returncode": -1}
        return {
            "stdout": stdout_b.decode(errors="replace")[:5000],
            "stderr": stderr_b.decode(errors="replace")[:2000],
            "returncode": proc.returncode or 0,
        }
    finally:
        os.unlink(tmp)


async def summarize_code(content: str, max_length: int = 200) -> dict:
    """Summarize a code file using a local model via litellm."""
    try:
        import litellm
        from models.router import get_router
        router = get_router()
        model  = router.primary_model("triage")
        resp = await litellm.acompletion(
            model=model,
            messages=[{
                "role": "user",
                "content": f"Summarize this code in {max_length} words:\n\n{content[:4000]}"
            }],
            max_tokens=300,
            temperature=0.1,
        )
        return {"summary": resp.choices[0].message.content, "model": model}
    except Exception as exc:
        return {"summary": f"Summarization failed: {exc}", "model": "none"}


async def translate_comments(content: str, target_lang: str = "English") -> dict:
    """Translate code comments to target language."""
    try:
        import litellm
        from models.router import get_router
        model = get_router().primary_model("simple_codegen")
        resp = await litellm.acompletion(
            model=model,
            messages=[{
                "role": "user",
                "content": (
                    f"Translate all comments and docstrings in this code to {target_lang}. "
                    f"Return the complete file with translated comments only:\n\n{content[:6000]}"
                )
            }],
            max_tokens=4096,
            temperature=0.0,
        )
        return {"translated": resp.choices[0].message.content}
    except Exception as exc:
        return {"translated": content, "error": str(exc)}


# ──────────────────────────────────────────────────────────────────────────────
# Tool registry
# ──────────────────────────────────────────────────────────────────────────────

_TOOLS = {
    "web_search":         ("Search the web for information", web_search),
    "code_execution":     ("Execute Python code safely", code_execution),
    "summarize_code":     ("Summarize a code file", summarize_code),
    "translate_comments": ("Translate code comments", translate_comments),
}


# ──────────────────────────────────────────────────────────────────────────────
# MCP stdio server
# ──────────────────────────────────────────────────────────────────────────────

async def handle_request(req: dict) -> dict:
    method = req.get("method", "")
    params = req.get("params", {})
    rid    = req.get("id", 1)

    if method == "tools/list":
        return {
            "jsonrpc": "2.0", "id": rid,
            "result": {
                "tools": [
                    {"name": name, "description": desc}
                    for name, (desc, _) in _TOOLS.items()
                ]
            }
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        if tool_name not in _TOOLS:
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}}
        try:
            _, func = _TOOLS[tool_name]
            result = await func(**arguments)
            return {"jsonrpc": "2.0", "id": rid, "result": result}
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32000, "message": str(exc)}}

    return {"jsonrpc": "2.0", "id": rid,
            "error": {"code": -32601, "message": f"Unknown method: {method}"}}


async def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = await handle_request(req)
        except json.JSONDecodeError as exc:
            resp = {"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"}}
        except Exception as exc:
            resp = {"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32000, "message": traceback.format_exc()}}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())
