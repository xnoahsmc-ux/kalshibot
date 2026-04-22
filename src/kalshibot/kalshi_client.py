from __future__ import annotations

import base64
import datetime as dt
import time
from pathlib import Path
from typing import Any

import requests

from .config import Config


class KalshiAuthError(RuntimeError):
    pass


class KalshiClient:
    """Thin wrapper over Kalshi's REST API.

    Supports the modern API-key + RSA-PSS signing flow. Falls back to the
    legacy email/password login when only those are configured.
    """

    def __init__(self, cfg: Config, session: requests.Session | None = None) -> None:
        self.cfg = cfg
        self.s = session or requests.Session()
        self._token: str | None = None
        self._private_key = None
        self._key_path = cfg.api_private_key_path

    def _ensure_key(self) -> None:
        if self._private_key is not None or not self._key_path:
            return
        # Imported lazily so machines without OpenSSL/cffi can still backtest.
        from cryptography.hazmat.primitives import serialization  # noqa: WPS433
        self._private_key = serialization.load_pem_private_key(
            Path(self._key_path).read_bytes(), password=None
        )

    # ---- auth ----
    def _sign(self, ts_ms: str, method: str, path: str) -> str:
        self._ensure_key()
        if self._private_key is None:
            raise KalshiAuthError("API key auth requires KALSHI_API_PRIVATE_KEY_PATH")
        from cryptography.hazmat.primitives import hashes  # noqa: WPS433
        from cryptography.hazmat.primitives.asymmetric import padding  # noqa: WPS433
        msg = f"{ts_ms}{method.upper()}{path}".encode()
        sig = self._private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode()

    def _login_legacy(self) -> None:
        if not (self.cfg.email and self.cfg.password):
            raise KalshiAuthError("No credentials configured")
        r = self.s.post(
            f"{self.cfg.base_url}/login",
            json={"email": self.cfg.email, "password": self.cfg.password},
            timeout=15,
        )
        r.raise_for_status()
        self._token = r.json()["token"]

    def _headers(self, method: str, path: str) -> dict[str, str]:
        if self.cfg.api_key_id and self._key_path:
            ts = str(int(time.time() * 1000))
            return {
                "KALSHI-ACCESS-KEY": self.cfg.api_key_id,
                "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, path),
                "Content-Type": "application/json",
            }
        if self._token is None and (self.cfg.email or self.cfg.password):
            self._login_legacy()
        if self._token:
            return {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}
        return {"Content-Type": "application/json"}

    # ---- core ----
    def _request(self, method: str, path: str, **kw: Any) -> Any:
        url = f"{self.cfg.base_url}{path}"
        r = self.s.request(method, url, headers=self._headers(method, path), timeout=20, **kw)
        if r.status_code == 401 and self._token is not None:
            self._token = None
            r = self.s.request(method, url, headers=self._headers(method, path), timeout=20, **kw)
        r.raise_for_status()
        return r.json() if r.text else {}

    # ---- public market data ----
    def get_markets(self, limit: int = 100, status: str = "open", cursor: str | None = None) -> dict:
        params = {"limit": limit, "status": status}
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/markets", params=params)

    def get_market(self, ticker: str) -> dict:
        return self._request("GET", f"/markets/{ticker}")

    def get_orderbook(self, ticker: str, depth: int = 50) -> dict:
        return self._request("GET", f"/markets/{ticker}/orderbook", params={"depth": depth})

    def get_trades(self, ticker: str, limit: int = 1000, cursor: str | None = None) -> dict:
        params: dict[str, Any] = {"ticker": ticker, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/markets/trades", params=params)

    def get_candles(
        self,
        series_ticker: str,
        ticker: str,
        start: dt.datetime,
        end: dt.datetime,
        period_minutes: int = 1,
    ) -> dict:
        params = {
            "ticker": ticker,
            "start_ts": int(start.timestamp()),
            "end_ts": int(end.timestamp()),
            "period_interval": period_minutes,
        }
        return self._request("GET", f"/series/{series_ticker}/markets/{ticker}/candlesticks", params=params)

    # ---- portfolio ----
    def get_balance(self) -> dict:
        return self._request("GET", "/portfolio/balance")

    def get_positions(self, limit: int = 200) -> dict:
        return self._request("GET", "/portfolio/positions", params={"limit": limit})

    def place_order(
        self,
        ticker: str,
        side: str,                   # "yes" | "no"
        action: str,                 # "buy" | "sell"
        count: int,
        type_: str = "limit",
        price_cents: int | None = None,
        client_order_id: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": type_,
            "client_order_id": client_order_id or f"kbot-{int(time.time()*1000)}",
        }
        if type_ == "limit":
            if price_cents is None:
                raise ValueError("price_cents required for limit orders")
            body["yes_price" if side == "yes" else "no_price"] = price_cents
        return self._request("POST", "/portfolio/orders", json=body)

    def cancel_order(self, order_id: str) -> dict:
        return self._request("DELETE", f"/portfolio/orders/{order_id}")
