from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from orchestrator.controller import StabilizerConfig, StabilizerController
from brain.schemas import AutonomyLevel


async def main(repo_path: Path, repo_url: str) -> None:
    cfg = StabilizerConfig(
        repo_url=repo_url,
        repo_root=repo_path,
        master_prompt_path=repo_path / "config" / "prompts" / "base.md",
        github_token=os.getenv("GITHUB_TOKEN", ""),
        primary_model=os.getenv("RHODAWK_AI_CODE_STABILIZER_MODEL", "claude-sonnet-4-20250514"),
        max_cycles=1,
        cost_ceiling_usd=10.0,
        auto_commit=False,
        autonomy_level=AutonomyLevel.PROPOSE_ONLY,
    )
    ctrl = StabilizerController(cfg)
    await ctrl.initialise()
    status = await ctrl.stabilize()
    print(f"Cycle complete. Status: {status.value}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/run_one_cycle.py /path/to/repo https://github.com/owner/repo")
        sys.exit(1)
    asyncio.run(main(Path(sys.argv[1]), sys.argv[2]))
