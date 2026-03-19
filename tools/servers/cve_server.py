"""CVE lookup MCP server stub — queries NVD API."""
import json, sys, asyncio, httpx

async def search_cve(keywords: list[str], limit: int = 10) -> list[dict]:
    url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    query = " ".join(keywords[:3])
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(url, params={"keywordSearch": query, "resultsPerPage": limit})
            data = r.json()
            return [
                {"id": v["cve"]["id"], "description": v["cve"]["descriptions"][0]["value"][:200]}
                for v in data.get("vulnerabilities", [])
            ]
    except Exception:
        return []

async def main():
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try:
            req = json.loads(line)
            params = req.get("params", {}).get("arguments", {})
            results = await search_cve(params.get("keywords", []))
            sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":req.get("id",1),"result":results}) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":1,"error":str(e)}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    asyncio.run(main())
