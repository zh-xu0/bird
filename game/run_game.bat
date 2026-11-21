rem This is for no venv python machines (Jonathan)
@echo off
rem Launch the Bird CAPTCHA game on Windows (cmd.exe / double-click).
setlocal

set "GAME_DIR=%~dp0"
set "PROJECT_DIR=%GAME_DIR%.."
set "PYTHON="

rem 1. Prefer standalone portable Python in PortableVS\Python (works on any PC)
for /f "delims=" %%P in ('dir /b /s "%PROJECT_DIR%\..\..\Python\python.exe" 2^>nul') do (
    set "PYTHON=%%P"
    goto :run_game
)

rem 2. Fall back to repo-local .venv or system python
if exist "%PROJECT_DIR%\.venv\Scripts\python.exe" (
    set "PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe"
) else if exist "%GAME_DIR%.venv\Scripts\python.exe" (
    set "PYTHON=%GAME_DIR%.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

:run_game
pushd "%GAME_DIR%"
"%PYTHON%" CaptchSparrowApp.py %*
set "GAME_EXIT=%ERRORLEVEL%"
popd

rem Keep the window open on failure when launched by double-click.
if not "%GAME_EXIT%"=="0" pause

exit /b %GAME_EXIT%