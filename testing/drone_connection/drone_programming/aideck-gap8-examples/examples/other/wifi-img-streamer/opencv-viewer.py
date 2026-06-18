#!/usr/bin/env python3
"""Compatibility entry point for the original Bitcraze viewer path.

Use `python main.py` from the repository root for the project GUI.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from testing.drone_connection.drone_AI_deck_main import main


if __name__ == "__main__":
    if "--cli" not in sys.argv:
        sys.argv.insert(1, "--cli")
    main()
