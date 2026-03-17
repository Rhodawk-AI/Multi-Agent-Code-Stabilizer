"""
run.py
OpenMOSS — top-level entry point.

This file was referenced throughout the audit prompt as the bootstrap root
but was completely absent from the codebase. Every import trace starting
from "run.py" led to a dead end.

This file is intentionally minimal — it delegates to the CLI which owns
all subcommands. Keeping logic here thin means the CLI remains the single
source of truth for argument parsing and command dispatch.

Usage:
    python run.py stabilize https://github.com/org/repo --path /local/repo
    python run.py audit     https://github.com/org/repo --path /local/repo
    python run.py bootstrap /local/repo
    python run.py status    /local/repo
    python run.py serve

Or install the package and use the openmoss entrypoint:
    pip install -e .
    openmoss stabilize ...
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the package root is on the path when run as a script
_root = Path(__file__).parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.cli import app

if __name__ == "__main__":
    app()
