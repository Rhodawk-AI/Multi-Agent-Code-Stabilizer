from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.cli import app

if __name__ == "__main__":
    app()
