#!/usr/bin/env bash
# One-shot launcher for macOS / Linux. Installs deps on first run, then
# serves the dashboard at http://127.0.0.1:8080.
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found on PATH. Install Python 3.10+ and retry." >&2
  exit 1
fi

if [ ! -f .installed ]; then
  echo "Installing dependencies..."
  python3 -m pip install --upgrade pip
  python3 -m pip install -r requirements.txt
  python3 -m pip install -e .
  touch .installed
fi

export PYTHONPATH=src
python3 -m kalshibot.cli web --port 8080 --auto-backtest "$@"
