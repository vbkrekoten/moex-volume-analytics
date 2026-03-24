"""Fetch daily trading history for individual securities from MOEX ISS API."""

from __future__ import annotations

import time
from datetime import date

import requests


ISS_BASE = "https://iss.moex.com/iss"
CANDLE_LIMIT = 500  # ISS returns max 500 rows per request


def fetch_security_history(
    ticker: str,
    date_from: str | date,
    date_to: str | date,
    market: str = "shares",
    engine: str = "stock",
    pause: float = 0.25,
) -> list[dict]:
    """Fetch daily candles for a single security.

    Returns list of dicts with keys: trade_date, ticker, value_rub, volume_shares.
    Uses interval=24 (daily) candles endpoint with pagination.
    """
    url = f"{ISS_BASE}/engines/{engine}/markets/{market}/securities/{ticker}/candles.json"
    all_rows: list[dict] = []
    start = 0

    while True:
        params = {
            "from": str(date_from),
            "till": str(date_to),
            "interval": 24,
            "iss.meta": "off",
            "start": start,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            break

        candles = data.get("candles", {})
        columns = candles.get("columns", [])
        rows = candles.get("data", [])

        if not rows:
            break

        for row in rows:
            rec = dict(zip(columns, row))
            all_rows.append({
                "trade_date": rec["begin"][:10],
                "ticker": ticker,
                "value_rub": rec.get("value", 0) or 0,
                "volume_shares": int(rec.get("volume", 0) or 0),
            })

        if len(rows) < CANDLE_LIMIT:
            break
        start += CANDLE_LIMIT
        time.sleep(pause)

    return all_rows


def fetch_multiple_tickers(
    tickers: list[str],
    date_from: str | date,
    date_to: str | date,
    pause: float = 0.3,
    progress_callback=None,
) -> list[dict]:
    """Fetch daily history for multiple tickers.

    Args:
        tickers: list of ticker symbols
        date_from, date_to: date range
        pause: seconds between tickers to respect rate limits
        progress_callback: optional callable(ticker, i, total)

    Returns combined list of dicts.
    """
    all_rows: list[dict] = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(ticker, i, total)

        rows = fetch_security_history(ticker, date_from, date_to)
        all_rows.extend(rows)

        if i < total - 1:
            time.sleep(pause)

    return all_rows
