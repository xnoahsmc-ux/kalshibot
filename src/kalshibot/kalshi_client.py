from __future__ import annotations

import base64
import datetime as dt
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from .config import Config


class KalshiAuthError(RuntimeError):
    pass


class KalshiAPIError(RuntimeError):
    def __init__(self, status: int, url: str, body: str) -> None:
        super().__init__(f"{status} on {url}: {body[:400]}")
        self.status = status
        self.url = url
        self.body = body


class KalshiClient:
    """Thin wrapper over Kalshi's REST API.

    Signs with RSA-PSS (SHA256) using the full request path (including the
    /trade-api/v2 prefix) per Kalshi's spec. Base URL defaults to the
    production elections host; override via `KALSHI_ENV=demo` for the
    sandbox or via KALSHI_BASE_URL for a custom endpoint.
    """

    def __init__(self, cfg: Config, session: requests.Session | None = None) -> None:
        self.cfg = cfg
        self.s = session or requests.Session()
        self._token: str | None = None
        self._private_key = None
        self._key_path = cfg.api_private_key_path

        # Split base_url into scheme+host and the API path prefix that must be
        # included in the signing material.
        parts = urlsplit(cfg.base_url.rstrip("/"))
        self._origin = f"{parts.scheme}://{parts.netloc}"
        self._api_prefix = parts.path or ""

    # ---- key material ----
    def _ensure_key(self) -> None:
        if self._private_key is not None or not self._key_path:
            return
        from cryptography.hazmat.primitives import serialization  # noqa: WPS433
        key_path = Path(self._key_path)
        if not key_path.is_absolute():
            # Resolve relative to the current working directory so
            # `KALSHI_API_PRIVATE_KEY_PATH=secrets/key.pem` works from anywhere.
            key_path = Path.cwd() / key_path
        if not key_path.exists():
            raise KalshiAuthError(
                f"Private key not found: {key_path}. "
                "Set KALSHI_API_PRIVATE_KEY_PATH to the PEM file."
            )
        self._private_key = serialization.load_pem_private_key(
            key_path.read_bytes(), password=None
        )

    # ---- signing ----
    def _sign(self, ts_ms: str, method: str, signing_path: str) -> str:
        self._ensure_key()
        if self._private_key is None:
            raise KalshiAuthError("API key auth requires KALSHI_API_PRIVATE_KEY_PATH")
        from cryptography.hazmat.primitives import hashes  # noqa: WPS433
        from cryptography.hazmat.primitives.asymmetric import padding  # noqa: WPS433
        msg = f"{ts_ms}{method.upper()}{signing_path}".encode()
        sig = self._private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode()

    def _signing_path_for(self, path: str) -> str:
        """Kalshi signs the full URL path including any API-version prefix,
        but without query string."""
        full = f"{self._api_prefix}{path}"
        # Drop query-string if accidentally included (we always pass params via requests)
        return full.split("?", 1)[0]

    # ---- auth ----
    def _login_legacy(self) -> None:
        if not (self.cfg.email and self.cfg.password):
            raise KalshiAuthError("No credentials configured (set API key or email/password)")
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
                "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, self._signing_path_for(path)),
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        if self._token is None and (self.cfg.email or self.cfg.password):
            self._login_legacy()
        if self._token:
            return {"Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"}
        return {"Content-Type": "application/json", "Accept": "application/json"}

    # ---- core ----
    def _request(self, method: str, path: str, **kw: Any) -> Any:
        url = f"{self.cfg.base_url}{path}"
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                r = self.s.request(method, url, headers=self._headers(method, path),
                                    timeout=20, **kw)
                if r.status_code == 401 and self._token is not None:
                    self._token = None
                    r = self.s.request(method, url, headers=self._headers(method, path),
                                        timeout=20, **kw)
                if r.status_code >= 400:
                    raise KalshiAPIError(r.status_code, url, r.text or "")
                return r.json() if r.text else {}
            except KalshiAPIError as e:
                if e.status in (502, 503, 504):
                    last_exc = e
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
            except requests.RequestException as e:
                last_exc = e
                time.sleep(1.5 * (attempt + 1))
        raise last_exc if last_exc else RuntimeError("request failed")

    # ---- introspection ----
    def auth_check(self) -> dict:
        """Verify credentials by hitting a lightweight auth-required endpoint
        plus an unauthenticated one. Returns a dict describing what worked.
        """
        out: dict[str, Any] = {
            "base_url": self.cfg.base_url,
            "api_key_id": self.cfg.api_key_id,
            "has_private_key": bool(self._key_path),
        }
        # Public endpoint first - if this fails, networking is the issue
        try:
            page = self.get_markets(limit=1, status="open")
            out["public_ok"] = True
            out["markets_visible"] = len(page.get("markets", []))
        except Exception as e:
            out["public_ok"] = False
            out["public_error"] = str(e)
            return out
        # Auth'd endpoint - proves signing works
        try:
            bal = self.get_balance()
            out["auth_ok"] = True
            out["balance"] = bal
        except Exception as e:
            out["auth_ok"] = False
            out["auth_error"] = str(e)
        return out

    # ---- public market data ----
    def get_exchange_status(self) -> dict:
        return self._request("GET", "/exchange/status")

    def get_series(self, series_ticker: str) -> dict:
        return self._request("GET", f"/series/{series_ticker}")

    def get_markets(self, limit: int = 100, status: str = "open",
                     cursor: str | None = None, event_ticker: str | None = None,
                     series_ticker: str | None = None) -> dict:
        params: dict[str, Any] = {"limit": limit, "status": status}
        if cursor:
            params["cursor"] = cursor
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        return self._request("GET", "/markets", params=params)

    def get_events(self, limit: int = 100, status: str = "open",
                    cursor: str | None = None) -> dict:
        params: dict[str, Any] = {"limit": limit, "status": status}
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/events", params=params)

    def get_market(self, ticker: str) -> dict:
        return self._request("GET", f"/markets/{ticker}")

    def get_orderbook(self, ticker: str, depth: int = 50) -> dict:
        return self._request("GET", f"/markets/{ticker}/orderbook",
                              params={"depth": depth})

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
            "start_ts": int(start.timestamp()),
            "end_ts": int(end.timestamp()),
            "period_interval": period_minutes,
        }
        path = f"/series/{series_ticker}/markets/{ticker}/candlesticks"
        return self._request("GET", path, params=params)

    # ---- portfolio ----
    def get_balance(self) -> dict:
        return self._request("GET", "/portfolio/balance")

    def get_positions(self, limit: int = 200) -> dict:
        return self._request("GET", "/portfolio/positions", params={"limit": limit})

    def get_orders(self, limit: int = 100, status: str | None = None) -> dict:
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return self._request("GET", "/portfolio/orders", params=params)

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
