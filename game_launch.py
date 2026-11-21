#!/usr/bin/env python3
"""Unified launcher for the Bird CAPTCHA game (Windows, Linux and macOS).

The launcher works with whichever Python runs it:

* If that interpreter has the game dependencies, the game starts immediately.
* If it does not, and the repository has a ``.venv``, game_launch.py re-executes
  itself with the venv's Python automatically. No activation step is needed.
* If there is no usable venv, it prints setup instructions.

Usage (from the repository root):
    python game_launch.py
    python3 game_launch.py
    python game_launch.py --dataset birds_dataset.zip
    python game_launch.py --no-venv

``--no-venv`` ignores the project virtual environment and uses the interpreter
running game_launch.py. All other arguments are passed to
``game/CaptchSparrowApp.py`` (``--dataset``, ``--split``).

Set ``BIRD_PYTHON`` to use a specific interpreter, for example:
    BIRD_PYTHON=/path/to/python python game_launch.py
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
GAME_DIR = PROJECT_DIR / "game"
GAME_APP = GAME_DIR / "CaptchSparrowApp.py"
REEXEC_ENV = "BIRD_LAUNCH_REEXEC"
REQUIRED_MODULES = (
    "tkinter",
    "PIL",
    "numpy",
    "sklearn",
    "joblib",
    "torch",
    "torchvision",
)


def _venv_python() -> Path | None:
    """Return the project's venv interpreter, in either OS layout."""
    if os.name == "nt":
        candidates = (
            PROJECT_DIR / ".venv" / "Scripts" / "python.exe",
            GAME_DIR / ".venv" / "Scripts" / "python.exe",
        )
    else:
        candidates = (
            PROJECT_DIR / ".venv" / "bin" / "python",
            GAME_DIR / ".venv" / "bin" / "python",
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _resolve_python(value: str) -> Path | None:
    """Resolve a path or a command name (e.g. 'python3') to an executable."""
    path = Path(value).expanduser()
    if path.is_file():
        return path
    found = shutil.which(value)
    return Path(found) if found else None


def _same_interpreter(first: Path, second: Path) -> bool:
    # Compare lexical absolute paths, not symlink targets: a venv's
    # bin/python is normally a symlink to the system interpreter, so
    # samefile() would wrongly treat the system Python and the venv as equal.
    return os.path.normcase(os.path.abspath(first)) == os.path.normcase(
        os.path.abspath(second)
    )


def _missing_modules() -> list[str]:
    missing: list[str] = []
    for name in REQUIRED_MODULES:
        try:
            if name == "tkinter":
                # find_spec can succeed while the _tkinter extension is absent,
                # so import it for a real check.
                import tkinter  # noqa: F401

                found = True
            else:
                found = importlib.util.find_spec(name) is not None
        except (ImportError, AttributeError, ValueError):
            found = False
        if not found:
            missing.append(name)
    return missing


def _print_missing(missing: list[str]) -> None:
    requirements = PROJECT_DIR / "requirements.txt"
    python = sys.executable
    print("The game is missing required Python packages:\n")
    print("  " + ", ".join(missing) + "\n")
    print("Usage: python game_launch.py [--dataset PATH] [--split {test,train}] [--no-venv]\n")
    print("Option 1 - virtual environment (recommended):")
    if os.name == "nt":
        print('  py -3 -m venv .venv')
        print(r"  .venv\Scripts\activate")
    else:
        print("  python3 -m venv .venv")
        print("  source .venv/bin/activate")
    print(f'  python -m pip install -r "{requirements}"')
    print("  python game_launch.py\n")
    print("Option 2 - install into the Python you are using (no venv):")
    print(f'  "{python}" -m pip install -r "{requirements}"')
    print(f'  "{python}" game_launch.py --no-venv\n')

    if "tkinter" in missing:
        print("Tkinter is a system package on Linux:")
        print("  Arch/CachyOS:     sudo pacman -S tk")
        print("  Debian/Ubuntu:    sudo apt install python3-tk")
        print("  macOS Homebrew:   brew install python-tk")
        print("  Windows/macOS:    use the python.org installer (enable tcl/tk)\n")
    print(
        "On Linux distributions that mark the system Python as externally "
        "managed (PEP 668, e.g. Arch/CachyOS), pip may refuse Option 2; use "
        "the virtual environment instead."
    )


def _launch(game_args: list[str]) -> None:
    sys.path.insert(0, str(GAME_DIR))
    # Make argparse inside the app show the app's own name and receive only
    # game arguments (launcher flags were removed above).
    sys.argv = [str(GAME_APP), *game_args]
    from CaptchSparrowApp import main as game_main

    game_main()


def main() -> None:
    argv = sys.argv[1:]
    no_venv = "--no-venv" in argv
    game_args = [argument for argument in argv if argument != "--no-venv"]

    if not no_venv and not os.environ.get(REEXEC_ENV):
        override = os.environ.get("BIRD_PYTHON")
        if override:
            target = _resolve_python(override)
            if target is None:
                print(
                    f"warning: BIRD_PYTHON={override!r} was not found; "
                    "using the current interpreter",
                    file=sys.stderr,
                )
        else:
            target = _venv_python()

        if target is not None and not _same_interpreter(
            target, Path(sys.executable)
        ):
            environment = os.environ.copy()
            environment[REEXEC_ENV] = "1"
            command = [str(target), str(Path(__file__).resolve()), *argv]
            print(f"Using project environment: {target}", flush=True)
            try:
                completed = subprocess.run(command, env=environment)
            except OSError as exc:
                print(f"Could not start {target}: {exc}", file=sys.stderr)
                raise SystemExit(1) from exc
            raise SystemExit(completed.returncode)

    missing = _missing_modules()
    if missing:
        _print_missing(missing)
        raise SystemExit(1)

    _launch(game_args)


if __name__ == "__main__":
    main()
