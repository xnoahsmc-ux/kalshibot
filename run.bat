@echo off
REM One-shot launcher for Windows. Installs deps on first run, then serves
REM the dashboard at http://127.0.0.1:8080.
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found on PATH. Install Python 3.10+ from python.org and retry.
  exit /b 1
)

if not exist ".installed" (
  echo Installing dependencies...
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt || exit /b 1
  python -m pip install -e . || exit /b 1
  type nul > .installed
)

set PYTHONPATH=src
python -m kalshibot.cli web --port 8080 --auto-backtest %*
