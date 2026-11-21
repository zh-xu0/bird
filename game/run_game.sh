#!/usr/bin/env bash
# Launch the Bird CAPTCHA game on Linux/macOS.
#
# Interpreter preference (mirrors the Windows run_game.bat):
#   1. BIRD_PYTHON (or PYTHON) environment override
#   2. a portable Python next to the project (../Python or ../../Python)
#   3. the project's .venv (repository root, then game/)
#   4. python3 / python from PATH
#
# The script cd's into the game folder, so dataset arguments are relative to
# game/ (for example: --dataset birds_dataset.zip or --dataset ../dataset/test).
set -uo pipefail

GAME_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$GAME_DIR/.." && pwd)"

find_python() {
    # Explicit overrides, for machines where Python is installed elsewhere.
    for override in "${BIRD_PYTHON:-}" "${PYTHON:-}"; do
        if [ -z "$override" ]; then
            continue
        fi
        if [ -x "$override" ] || command -v "$override" >/dev/null 2>&1; then
            printf '%s\n' "$override"
            return 0
        fi
        printf 'warning: %s was not found; ignoring it\n' "$override" >&2
    done

    # Portable Python, matching the Windows launcher's convention.
    for candidate in \
        "$PROJECT_DIR/../../Python/bin/python3" \
        "$PROJECT_DIR/../../Python/python3" \
        "$PROJECT_DIR/../../Python/bin/python" \
        "$PROJECT_DIR/../../Python/python" \
        "$PROJECT_DIR/../Python/bin/python3" \
        "$PROJECT_DIR/../Python/python3" \
        "$PROJECT_DIR/../Python/bin/python" \
        "$PROJECT_DIR/../Python/python"
    do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    # Project virtual environments. Do not resolve the python symlink: it
    # points at the system interpreter, which may not have the packages.
    for candidate in \
        "$PROJECT_DIR/.venv/bin/python" \
        "$GAME_DIR/.venv/bin/python"
    do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    if command -v python3 >/dev/null 2>&1; then
        printf 'python3\n'
    else
        printf 'python\n'
    fi
}

PYTHON="$(find_python)"
if [ ! -x "$PYTHON" ] && ! command -v "$PYTHON" >/dev/null 2>&1; then
    printf 'error: no usable Python found.\n' >&2
    printf 'Install Python 3.11+, then run: python3 -m pip install -r requirements.txt\n' >&2
    printf 'or create the project .venv described in the README.\n' >&2
    exit 1
fi

cd "$GAME_DIR"
"$PYTHON" CaptchSparrowApp.py "$@"
status=$?

# Keep the window open on failure when launched from a file manager.
if [ "$status" -ne 0 ] && [ -t 0 ]; then
    printf '\nThe game exited with status %s.\n' "$status"
    read -r -p "Press Enter to close..." _ || true
fi

exit "$status"
