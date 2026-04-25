"""Outbound alerts. Three backends in priority order:

  1. Twilio SMS (if KALSHIBOT_TWILIO_SID + TOKEN + FROM are set)
  2. Email-to-SMS gateway (if KALSHIBOT_NOTIFY_EMAIL is set; uses SMTP env)
  3. Local log file (always works) - shown on the website Alerts panel

Set destination phone via KALSHIBOT_PHONE (digits only, e.g. 4087637531).
"""
from __future__ import annotations

import json
import os
import smtplib
import threading
import time
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from pathlib import Path

import requests


_LOG_PATH = Path("data/alerts.jsonl")
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_LOCK = threading.Lock()


@dataclass
class AlertResult:
    ok: bool
    backend: str
    message: str
    sent_at: float = field(default_factory=time.time)


def _log(record: dict) -> None:
    with _LOCK, _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def recent_alerts(limit: int = 50) -> list[dict]:
    if not _LOG_PATH.exists():
        return []
    out: list[dict] = []
    try:
        with _LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return out[-limit:][::-1]


def _send_twilio(body: str, to_number: str) -> AlertResult | None:
    sid = os.getenv("KALSHIBOT_TWILIO_SID")
    token = os.getenv("KALSHIBOT_TWILIO_TOKEN")
    sender = os.getenv("KALSHIBOT_TWILIO_FROM")
    if not (sid and token and sender):
        return None
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    try:
        r = requests.post(url, auth=(sid, token), timeout=10,
                           data={"From": sender, "To": to_number, "Body": body})
        if r.status_code in (200, 201):
            return AlertResult(True, "twilio", f"Twilio queued message {r.json().get('sid','')}")
        return AlertResult(False, "twilio", f"Twilio error {r.status_code}: {r.text[:200]}")
    except Exception as e:
        return AlertResult(False, "twilio", f"Twilio exception: {e}")


def _send_email_to_sms(body: str) -> AlertResult | None:
    """Use any SMTP server to email a phone-carrier gateway.
    Set KALSHIBOT_NOTIFY_EMAIL to e.g. 4087637531@txt.att.net (carrier-specific)."""
    host = os.getenv("KALSHIBOT_SMTP_HOST")
    port = int(os.getenv("KALSHIBOT_SMTP_PORT", "587"))
    user = os.getenv("KALSHIBOT_SMTP_USER")
    pwd = os.getenv("KALSHIBOT_SMTP_PASS")
    to = os.getenv("KALSHIBOT_NOTIFY_EMAIL")
    if not (host and to):
        return None
    msg = MIMEText(body)
    msg["Subject"] = "Kalshi Bot"
    msg["From"] = user or "kalshibot@local"
    msg["To"] = to
    try:
        with smtplib.SMTP(host, port, timeout=12) as s:
            s.starttls()
            if user and pwd:
                s.login(user, pwd)
            s.send_message(msg)
        return AlertResult(True, "email_sms", f"Emailed {to}")
    except Exception as e:
        return AlertResult(False, "email_sms", f"SMTP exception: {e}")


def send_alert(body: str, *, kind: str = "info") -> AlertResult:
    to_number = os.getenv("KALSHIBOT_PHONE", "")
    record = {"ts": time.time(), "kind": kind, "body": body, "backends": []}

    # Try Twilio
    if to_number:
        formatted = to_number if to_number.startswith("+") else f"+1{to_number}"
        r = _send_twilio(body, formatted)
        if r is not None:
            record["backends"].append({"name": r.backend, "ok": r.ok, "msg": r.message})
            if r.ok:
                _log(record); return r

    # Try Email-to-SMS
    r = _send_email_to_sms(body)
    if r is not None:
        record["backends"].append({"name": r.backend, "ok": r.ok, "msg": r.message})
        if r.ok:
            _log(record); return r

    # Log-only fallback
    record["backends"].append({"name": "log", "ok": True, "msg": "Saved to data/alerts.jsonl"})
    _log(record)
    return AlertResult(True, "log", "Saved to local log (no SMS backend configured)")
