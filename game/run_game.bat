@echo off
rem Launch the Bird CAPTCHA game on Windows (cmd.exe / double-click).
rem Mirrors run_game.sh: prefer the virtual environment at the repo root.
setlocal

set "GAME_DIR=%~dp0"
set "PROJECT_DIR=%GAME_DIR%.."

if exist "%PROJECT_DIR%\.venv\Scripts\python.exe" (
    set "PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe"
) else if exist "%GAME_DIR%.venv\Scripts\python.exe" (
    set "PYTHON=%GAME_DIR%.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

pushd "%GAME_DIR%"
"%PYTHON%" CaptchSparrowApp.py %*
set "GAME_EXIT=%ERRORLEVEL%"
popd

rem Keep the window open on failure when launched by double-click.
if not "%GAME_EXIT%"=="0" pause

exit /b %GAME_EXIT%
