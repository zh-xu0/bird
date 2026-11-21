#!/usr/bin/env bash
# Optional convenience wrapper. The unified launcher is ../game_launch.py, which
# works on Linux, macOS and Windows and selects the project .venv (or the
# current Python) automatically.
set -euo pipefail
GAME_DIR="$(cd "$(dirname "$0")" && pwd)"
LAUNCHER="$GAME_DIR/../game_launch.py"

if [ -z "${PYTHON:-}" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    else
        PYTHON=python
    fi
fi

exec "$PYTHON" "$LAUNCHER" "$@"
