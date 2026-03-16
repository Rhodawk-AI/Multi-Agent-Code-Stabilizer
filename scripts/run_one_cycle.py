"""
scripts/run_one_cycle.py
Manually trigger a single audit + fix cycle without the full loop.
Useful for debugging and testing a specific repo state.

Usage:
    python scripts/run_one_cycle.py /path/to/repo https://github.com/owner/repo
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from orchestrator.controller import StabilizerConfig, StabilizerController


async def run_single_cycle(repo_path: Path, repo_url: str) -> None:
    cfg = StabilizerConfig(
        repo_url=repo_url,
        repo_root=repo_path,
        master_prompt_path=Path("config/prompts/base.md"),
        github_token=os.getenv("GITHUB_TOKEN", ""),
        max_cycles=1,
        auto_commit=False,
    )
    ctrl = StabilizerController(cfg)
    await ctrl.initialise()
    await ctrl._phase_read()
    score = await ctrl._phase_audit()
    print(f"\nAudit complete:")
    print(f"  CRITICAL: {score.critical_count}")
    print(f"  MAJOR:    {score.major_count}")
    print(f"  MINOR:    {score.minor_count}")
    print(f"  Score:    {score.score}")
    await ctrl.storage.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_one_cycle.py <repo_path> <repo_url>")
        sys.exit(1)
    asyncio.run(run_single_cycle(Path(sys.argv[1]), sys.argv[2]))
