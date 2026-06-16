from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def find_cfclient() -> str | None:
    executable = shutil.which("cfclient")
    if executable is not None:
        return executable

    python_dir = Path(sys.executable).resolve().parent
    local_executable = python_dir / "cfclient.exe"
    if local_executable.exists():
        return str(local_executable)

    return None


def main() -> int:
    executable = find_cfclient()
    if executable is None:
        print("cfclient is not installed.")
        print("Run this first:")
        print("  pip install -r requirements.txt")
        return 1

    subprocess.Popen([executable], cwd=Path.cwd())
    print("Crazyflie Client started.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
