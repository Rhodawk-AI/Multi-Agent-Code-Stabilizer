"""
scripts/bootstrap_brain.py
Bootstrap the agent brain on an existing repo before a full stabilize run.
Use this on large repos to pre-index everything cheaply before expensive audits.

Usage:
    python scripts/bootstrap_brain.py /path/to/repo [--model claude-haiku-4-5-20251001]
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.cli import bootstrap

if __name__ == "__main__":
    import typer
    typer.run(bootstrap)
