"""Shared in-process state for the web app: latest backtest report, running
backtest jobs, and live trader handle.
"""
from __future__ import annotations

import pickle
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..analytics import BacktestReport


@dataclass
class JobStatus:
    id: str
    kind: str
    started: float
    finished: float | None = None
    progress: float = 0.0
    message: str = "pending"
    error: str | None = None
    result_ref: str | None = None


class AppState:
    def __init__(self, report_path: Path = Path("backtest_report.pkl")) -> None:
        self.report_path = report_path
        self._report: BacktestReport | None = None
        self._jobs: dict[str, JobStatus] = {}
        self._lock = threading.Lock()
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if self.report_path.exists():
            try:
                self._report = pickle.loads(self.report_path.read_bytes())
            except Exception:
                self._report = None

    @property
    def report(self) -> BacktestReport | None:
        return self._report

    def set_report(self, report: BacktestReport) -> None:
        with self._lock:
            self._report = report
            self.report_path.write_bytes(pickle.dumps(report))

    def register_job(self, kind: str) -> JobStatus:
        jid = f"{kind}-{int(time.time()*1000)}"
        js = JobStatus(id=jid, kind=kind, started=time.time(),
                       message="starting")
        with self._lock:
            self._jobs[jid] = js
        return js

    def update_job(self, jid: str, **kw: Any) -> None:
        with self._lock:
            if jid not in self._jobs:
                return
            for k, v in kw.items():
                setattr(self._jobs[jid], k, v)

    def job(self, jid: str) -> JobStatus | None:
        return self._jobs.get(jid)

    def jobs(self) -> list[JobStatus]:
        return sorted(self._jobs.values(), key=lambda j: -j.started)
