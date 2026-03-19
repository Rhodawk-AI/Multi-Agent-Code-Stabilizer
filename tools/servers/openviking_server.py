"""
tools/servers/openviking_server.py
===================================
Vector memory MCP server — wraps Qdrant (via HelixDB) for Rhodawk AI.

PREVIOUS PHANTOM: "OpenViking" (https://github.com/lalalune/open-viking)
is not a real installable Python package.  The canonical backend is
Qdrant via the HelixDB adapter.  The `import open_viking` hook is
retained for forward-compatibility only.

This server wraps the real Qdrant/HelixDB stack and exposes it as an
MCP tool providing:
• store     — persist a code chunk with embedding
• search    — semantic nearest-neighbour search
• delete    — remove vectors by file path
• stats     — collection statistics

Transport: stdio JSON-RPC 2.0
"""
from __future__ import annotations

import asyncio
import json
import sys
import os
import logging
from typing import Any

log = logging.getLogger(__name__)

# Try OpenViking first, fall back to our HelixDB (Qdrant-backed)
_BACKEND: Any = None


def _init_backend():
    global _BACKEND
    if _BACKEND is not None:
        return

    try:
        import open_viking  # type: ignore[import]
        _BACKEND = open_viking.VectorDB(
            url=os.environ.get("OPENVIKING_URL", "http://localhost:6333"),
            collection="rhodawk_openviking",
        )
        _BACKEND.ensure_collection()
        log.info("OpenViking native backend initialised")
        return
    except ImportError:
        pass

    # Fall back to HelixDB (Qdrant)
    try:
        from memory.helixdb import HelixDB
        helix = HelixDB(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
        helix.initialise()
        _BACKEND = _HelixAdapter(helix)
        log.info("OpenViking: using HelixDB (Qdrant) backend")
    except Exception as exc:
        log.warning(f"OpenViking: all backends failed: {exc}")
        _BACKEND = _StubBackend()


class _HelixAdapter:
    """Adapts HelixDB to the OpenViking interface."""
    def __init__(self, helix):
        self._h = helix

    def store(self, doc_id: str, file_path: str, content: str, summary: str,
              line_start: int = 0, line_end: int = 0, language: str = "unknown") -> bool:
        from memory.helixdb import HelixDocument
        self._h.index(HelixDocument(
            id=doc_id, file_path=file_path, line_start=line_start, line_end=line_end,
            language=language, content=content, summary=summary,
        ))
        return True

    def search(self, query: str, n: int = 10, file_filter: str | None = None) -> list[dict]:
        results = self._h.search(query, n=n, file_filter=file_filter)
        return [{"id": r.id, "file_path": r.file_path, "line_start": r.line_start,
                 "line_end": r.line_end, "language": r.language, "summary": r.summary,
                 "score": r.score} for r in results]

    def delete(self, file_path: str) -> bool:
        self._h.delete_file(file_path)
        return True

    def stats(self) -> dict:
        return self._h.stats()


class _StubBackend:
    def store(self, **_) -> bool: return False
    def search(self, query: str, n: int = 10, **_) -> list[dict]: return []
    def delete(self, file_path: str) -> bool: return False
    def stats(self) -> dict: return {"available": False}


# ──────────────────────────────────────────────────────────────────────────────
# Tool handlers
# ──────────────────────────────────────────────────────────────────────────────

async def handle_store(arguments: dict) -> dict:
    _init_backend()
    ok = _BACKEND.store(
        doc_id     = arguments.get("doc_id", ""),
        file_path  = arguments.get("file_path", ""),
        content    = arguments.get("content", ""),
        summary    = arguments.get("summary", ""),
        line_start = arguments.get("line_start", 0),
        line_end   = arguments.get("line_end", 0),
        language   = arguments.get("language", "unknown"),
    )
    return {"stored": ok}


async def handle_search(arguments: dict) -> list[dict]:
    _init_backend()
    return _BACKEND.search(
        query       = arguments.get("query", ""),
        n           = arguments.get("n", 10),
        file_filter = arguments.get("file_filter"),
    )


async def handle_delete(arguments: dict) -> dict:
    _init_backend()
    ok = _BACKEND.delete(arguments.get("file_path", ""))
    return {"deleted": ok}


async def handle_stats(_: dict) -> dict:
    _init_backend()
    return _BACKEND.stats()


_HANDLERS = {
    "store":  handle_store,
    "search": handle_search,
    "delete": handle_delete,
    "stats":  handle_stats,
}


async def handle_request(req: dict) -> dict:
    method = req.get("method", "")
    params = req.get("params", {})
    rid    = req.get("id", 1)

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "tools": [{"name": k, "description": f"OpenViking {k}"} for k in _HANDLERS]
        }}

    if method == "tools/call":
        tool  = params.get("name", "")
        args  = params.get("arguments", {})
        handler = _HANDLERS.get(tool)
        if not handler:
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool}"}}
        try:
            result = await handler(args)
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
            req  = json.loads(line)
            resp = await handle_request(req)
        except Exception as exc:
            resp = {"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": str(exc)}}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())
