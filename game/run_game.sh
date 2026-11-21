#!/usr/bin/env bash
set -euo pipefail
GAME_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$GAME_DIR/.." && pwd)"
cd "$GAME_DIR"

# Prefer a virtual environment at the repo root. Do not resolve the python
# symlink: resolving it can fall back to the system interpreter and lose the
# installed packages.
if [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
elif [ -x "$GAME_DIR/.venv/bin/python" ]; then
    PYTHON="$GAME_DIR/.venv/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

exec "$PYTHON" CaptchSparrowApp.py "$@"
