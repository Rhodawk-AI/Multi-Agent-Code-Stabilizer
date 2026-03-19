"""SBOM generation MCP server using CycloneDX."""
import json, sys, asyncio, subprocess, os

async def generate_sbom(repo_path: str) -> dict:
    try:
        r = subprocess.run(
            ["cyclonedx-py", "environment", "--output-format", "json"],
            cwd=repo_path, capture_output=True, text=True, timeout=60
        )
        return {"sbom": r.stdout[:5000], "status": "ok" if r.returncode == 0 else "error"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

async def main():
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try:
            req = json.loads(line)
            p = req.get("params", {}).get("arguments", {})
            result = await generate_sbom(p.get("repo_path", "."))
            sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":req.get("id",1),"result":result}) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":1,"error":str(e)}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    asyncio.run(main())
