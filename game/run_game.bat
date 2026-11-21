@echo off
rem Optional convenience wrapper. The unified launcher is ..\launch.py, which
rem works on Windows, Linux and macOS and selects the project .venv (or the
rem current Python) automatically.
setlocal

set "LAUNCHER=%~dp0..\launch.py"

where py >nul 2>nul
if %ERRORLEVEL%==0 (
    py -3 "%LAUNCHER%" %*
) else (
    python "%LAUNCHER%" %*
)
set "GAME_EXIT=%ERRORLEVEL%"

if not "%GAME_EXIT%"=="0" pause
exit /b %GAME_EXIT%
