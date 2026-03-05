"""Fetch daily OHLC history for MOEX indices (IMOEX, RGBI, RVI)."""

import time
from datetime import date

import pandas as pd
import requests

BASE_URL = "https://iss.moex.com/iss/history/engines/stock/markets/index/securities"
TIMEOUT = 30
PAGE_SIZE = 100

TICKERS = ["IMOEX", "RGBI", "RVI"]


def fetch_index_history(ticker: str, date_from: date, date_to: date,
                        delay: float = 0.1) -> pd.DataFrame:
    """Fetch daily OHLC + capitalization for one index ticker."""
    url = f"{BASE_URL}/{ticker}.json"
    all_rows = []
    start = 0

    while True:
        params = {
            "from": date_from.strftime("%Y-%m-%d"),
            "till": date_to.strftime("%Y-%m-%d"),
            "iss.meta": "off",
            "history.columns": "TRADEDATE,OPEN,HIGH,LOW,CLOSE,CAPITALIZATION",
            "iss.only": "history",
            "start": start,
        }
        r = requests.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()

        history = data.get("history", {})
        columns = history.get("columns", [])
        raw = history.get("data", [])

        if not raw:
            break

        for row in raw:
            rec = dict(zip(columns, row))
            all_rows.append({
                "trade_date": rec["TRADEDATE"],
                "ticker": ticker,
                "open_val": float(rec["OPEN"]) if rec.get("OPEN") else None,
                "high_val": float(rec["HIGH"]) if rec.get("HIGH") else None,
                "low_val": float(rec["LOW"]) if rec.get("LOW") else None,
                "close_val": float(rec["CLOSE"]) if rec["CLOSE"] else None,
                "capitalization": float(rec["CAPITALIZATION"]) if rec.get("CAPITALIZATION") else None,
            })

        start += len(raw)
        if len(raw) < PAGE_SIZE:
            break
        time.sleep(delay)

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


def fetch_all_indices(date_from: date, date_to: date,
                      delay: float = 0.1,
                      progress_callback=None) -> pd.DataFrame:
    """Fetch history for all tracked indices."""
    frames = []
    for i, ticker in enumerate(TICKERS):
        df = fetch_index_history(ticker, date_from, date_to, delay)
        frames.append(df)
        if progress_callback:
            progress_callback((i + 1) / len(TICKERS))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
