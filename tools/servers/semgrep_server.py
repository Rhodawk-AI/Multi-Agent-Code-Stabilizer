"""Semgrep MCP server — runs semgrep as an MCP tool."""
import json, sys, asyncio, subprocess, tempfile, os

async def semgrep_scan(file_path: str, content: str) -> list[dict]:
    suffix = os.path.splitext(file_path)[1] or ".py"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w") as f:
        f.write(content)
        tmp = f.name
    try:
        r = subprocess.run(
            ["semgrep", "--config=auto", "--json", tmp],
            capture_output=True, text=True, timeout=30
        )
        data = json.loads(r.stdout or "{}")
        return [{"rule": f["check_id"], "line": f["start"]["line"], "msg": f["extra"]["message"]}
                for f in data.get("results", [])]
    except Exception:
        return []
    finally:
        os.unlink(tmp)

async def main():
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try:
            req = json.loads(line)
            p = req.get("params", {}).get("arguments", {})
            results = await semgrep_scan(p.get("file_path","tmp.py"), p.get("content",""))
            sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":req.get("id",1),"result":results}) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":1,"error":str(e)}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    asyncio.run(main())
