"""Load v2 config from config.yaml + .env without taking a hard YAML
dependency. Falls back to a tiny parser if PyYAML isn't installed."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False


@dataclass
class UniverseCfg:
    max_spread_cents: int = 4
    min_volume_usd_24h: float = 500.0
    min_time_to_resolution_minutes: int = 60
    max_time_to_resolution_minutes: int = 20160
    min_open_interest: int = 100


@dataclass
class SignalsCfg:
    min_edge_cents: int = 3
    min_confidence: float = 0.55
    weights: dict[str, float] = field(default_factory=lambda: {
        "mispricing": 1.0, "microstructure": 0.6, "external_anchor": 1.4})


@dataclass
class ExecutionCfg:
    passive_inset_cents: int = 1
    escalate_after_seconds: int = 30
    take_if_edge_cents: int = 5
    watchdog_timeout_seconds: int = 60
    rate_limit_qps: int = 8


@dataclass
class RiskCfg:
    kelly_fraction: float = 0.25
    hard_max_per_contract_usd: float = 25.0
    daily_loss_limit_pct: float = 3.0
    max_concurrent_positions: int = 12
    max_per_genre: int = 4
    max_same_series: int = 3
    halt_flag_path: str = "data/HALT"


@dataclass
class StorageCfg:
    snapshots_db: str = "data/markets.db"
    paper_ledger: str = "data/paper_ledger.jsonl"
    live_journal: str = "data/journal.jsonl"


@dataclass
class BacktestCfg:
    fee_cents_per_trade: int = 7
    slippage_cents: int = 0
    queue_position_assumption: float = 0.5


@dataclass
class MonitoringCfg:
    log_dir: str = "logs"
    cli_dashboard: bool = True
    discord_webhook_url: str = ""
    slack_webhook_url: str = ""


@dataclass
class V2Config:
    universe: UniverseCfg = field(default_factory=UniverseCfg)
    signals: SignalsCfg = field(default_factory=SignalsCfg)
    execution: ExecutionCfg = field(default_factory=ExecutionCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)
    storage: StorageCfg = field(default_factory=StorageCfg)
    backtest: BacktestCfg = field(default_factory=BacktestCfg)
    monitoring: MonitoringCfg = field(default_factory=MonitoringCfg)
    bankroll_usd: float = 1000.0


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text()
    if _HAVE_YAML:
        return yaml.safe_load(text) or {}
    # Minimal YAML parser sufficient for our flat-with-one-level-of-nesting config.
    out: dict = {}
    cur: dict = out
    indent_stack: list[tuple[int, dict]] = [(0, out)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        while indent_stack and indent_stack[-1][0] > indent:
            indent_stack.pop()
        cur = indent_stack[-1][1]
        if ":" in line:
            key, _, val = line.lstrip().partition(":")
            val = val.strip()
            if not val:
                cur[key] = {}
                indent_stack.append((indent + 2, cur[key]))
            else:
                if val.replace(".", "", 1).replace("-", "", 1).isdigit():
                    v: Any = float(val) if "." in val else int(val)
                elif val.lower() in ("true", "false"):
                    v = val.lower() == "true"
                elif val.startswith('"') and val.endswith('"'):
                    v = val[1:-1]
                else:
                    v = val
                cur[key] = v
    return out


def load(path: str = "config.yaml") -> V2Config:
    raw = _read_yaml(Path(path))

    def _merge(dc, key):
        section = raw.get(key, {}) or {}
        for k, v in section.items():
            if hasattr(dc, k):
                setattr(dc, k, v)

    cfg = V2Config()
    _merge(cfg.universe, "universe")
    _merge(cfg.signals, "signals")
    _merge(cfg.execution, "execution")
    _merge(cfg.risk, "risk")
    _merge(cfg.storage, "storage")
    _merge(cfg.backtest, "backtest")
    _merge(cfg.monitoring, "monitoring")
    if "weights" in raw.get("signals", {}):
        cfg.signals.weights = dict(raw["signals"]["weights"])
    cfg.bankroll_usd = float(os.getenv("KALSHIBOT_BANKROLL_USD",
                                         cfg.bankroll_usd) or cfg.bankroll_usd)
    return cfg
