#!/usr/bin/env bash
set -euo pipefail
GAME_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$GAME_DIR/.." && pwd)"
cd "$GAME_DIR"

# Prefer a virtual environment at the repo root. Do not resolve the python
# symlink: resolving it can fall back to the system interpreter and lose the
# installed packages. Check the Windows `Scripts/` layout too, so this script
# also works from Git Bash / MSYS2 on Windows.
if [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
elif [ -x "$GAME_DIR/.venv/bin/python" ]; then
    PYTHON="$GAME_DIR/.venv/bin/python"
elif [ -f "$PROJECT_DIR/.venv/Scripts/python.exe" ]; then
    PYTHON="$PROJECT_DIR/.venv/Scripts/python.exe"
elif [ -f "$GAME_DIR/.venv/Scripts/python.exe" ]; then
    PYTHON="$GAME_DIR/.venv/Scripts/python.exe"
else
    # Respect an explicitly provided interpreter, otherwise use python3 when
    # available and fall back to python (the usual name on Windows).
    if [ -z "${PYTHON:-}" ]; then
        if command -v python3 >/dev/null 2>&1; then
            PYTHON=python3
        else
            PYTHON=python
        fi
    fi
fi

exec "$PYTHON" CaptchSparrowApp.py "$@"
