@echo off
REM Phone-friendly launcher. Same as run.bat but binds to all network
REM interfaces so your phone (on the same WiFi) can hit the dashboard.
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if "%PY%"=="" (
  where python >nul 2>&1 && set "PY=python"
)
if "%PY%"=="" (
  echo Python was not found. Install from python.org and retry.
  pause
  exit /b 1
)

if not exist ".installed" (
  echo Installing dependencies...
  %PY% -m pip install --upgrade pip
  %PY% -m pip install -r requirements.txt
  %PY% -m pip install -e .
  type nul > .installed
)

set PYTHONPATH=src
echo.
echo ============================================================
echo  Kalshi Bot — phone-accessible mode
echo  Your phone (on the same WiFi) can open this site at:
echo.
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /R "IPv4.Address"') do (
  for /f "tokens=* delims= " %%b in ("%%a") do echo     http://%%b:8080
)
echo.
echo  Leave this window open. Ctrl+C to stop.
echo ============================================================
echo.
%PY% -m kalshibot.cli setup-live --host 0.0.0.0 --port 8080 %*
echo.
echo The server exited.
pause
