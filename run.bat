@echo off
REM One-shot launcher for Windows. Installs deps on first run, then serves
REM the dashboard at http://127.0.0.1:8080.
setlocal EnableDelayedExpansion
cd /d "%~dp0"

REM --- locate a working Python interpreter -----------------------------------
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if "%PY%"=="" (
  where python >nul 2>&1 && set "PY=python"
)
if "%PY%"=="" (
  where python3 >nul 2>&1 && set "PY=python3"
)
if "%PY%"=="" (
  echo ---------------------------------------------------------------
  echo Python was not found on PATH.
  echo.
  echo 1. Install Python 3.10 or newer from https://www.python.org/downloads/
  echo 2. During install, CHECK "Add python.exe to PATH".
  echo 3. Close this window, open a new Command Prompt, and run run.bat again.
  echo ---------------------------------------------------------------
  pause
  exit /b 1
)

REM Sanity check that the interpreter actually runs (handles the Windows
REM app-execution-alias that pretends Python exists and then fails).
%PY% -c "import sys; print(sys.version)" >nul 2>&1
if errorlevel 1 (
  echo ---------------------------------------------------------------
  echo Your 'python' command is the Microsoft Store alias - real Python
  echo isn't installed. Install from https://www.python.org/downloads/
  echo and make sure you tick "Add python.exe to PATH".
  echo ---------------------------------------------------------------
  pause
  exit /b 1
)

REM --- first-run dependency install ------------------------------------------
if not exist ".installed" (
  echo Installing dependencies...
  %PY% -m pip install --upgrade pip || exit /b 1
  %PY% -m pip install -r requirements.txt || exit /b 1
  %PY% -m pip install -e . || exit /b 1
  type nul > .installed
)

set PYTHONPATH=src
echo.
echo ============================================================
echo Starting Kalshibot at http://127.0.0.1:8080
echo Leave this window open while you use the site.
echo Press Ctrl+C here to stop.
echo ============================================================
echo.
%PY% -m kalshibot.cli web --port 8080 --auto-backtest %*
echo.
echo The server exited. Scroll up to see any error messages.
pause
