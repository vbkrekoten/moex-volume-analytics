"""Fetch daily trading turnovers from MOEX ISS by engine/market."""

import time
from datetime import date, timedelta

import pandas as pd
import requests

# Mapping from (engine, market) to normalized instrument_class
MARKET_CLASS_MAP = {
    ("stock", "shares"): "shares",
    ("stock", "bonds"): "bonds",
    ("stock", "ndm"): "bonds",  # negotiated deals in bonds
    ("stock", "repo"): "repo",
    ("stock", "ccp"): "repo",   # CCP repo
    ("stock", "gcc"): "repo",   # GCC repo
    ("currency", "selt"): "currency",
    ("currency", "otc"): "currency",
    ("futures", "forts"): "futures",
    ("futures", "options"): "options",
}

BASE_URL = "https://iss.moex.com/iss"
TIMEOUT = 30


def _fetch_turnovers_for_date(dt: date) -> list[dict]:
    """Fetch turnover breakdown for a single date from all engines."""
    date_str = dt.strftime("%Y-%m-%d")
    rows = []

    # Fetch stock engine sub-markets (shares, bonds, repo)
    for engine in ("stock", "currency", "futures"):
        url = f"{BASE_URL}/engines/{engine}/turnovers.json"
        params = {
            "iss.meta": "off",
            "is_tonight_session": 0,
            "date": date_str,
            "iss.only": "turnovers",
        }
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json()
        except Exception:
            continue

        turnovers = data.get("turnovers", {})
        columns = turnovers.get("columns", [])
        raw_data = turnovers.get("data", [])

        for row in raw_data:
            rec = dict(zip(columns, row))
            market = rec.get("MARKET", "").lower() if rec.get("MARKET") else ""
            if not market:
                # Try NAME column as fallback
                market = rec.get("NAME", "").lower()

            key = (engine, market)
            instrument_class = MARKET_CLASS_MAP.get(key)
            if not instrument_class:
                continue

            value_rub = rec.get("VALTODAY") or rec.get("VALUE") or 0
            num_trades = rec.get("NUMTRADES") or 0

            # Skip zero-value rows
            if not value_rub:
                continue

            rows.append({
                "trade_date": date_str,
                "engine": engine,
                "market": market,
                "instrument_class": instrument_class,
                "value_rub": float(value_rub),
                "num_trades": int(num_trades),
            })

    return rows


def fetch_turnovers(date_from: date, date_to: date,
                    delay: float = 0.1,
                    progress_callback=None) -> pd.DataFrame:
    """
    Fetch daily turnovers for a date range.
    Iterates through each trading day (Mon-Fri), calls 3 engine endpoints.
    """
    all_rows = []
    current = date_from
    total_days = (date_to - date_from).days
    processed = 0

    while current <= date_to:
        # Skip weekends
        if current.weekday() < 5:
            rows = _fetch_turnovers_for_date(current)
            all_rows.extend(rows)
            time.sleep(delay)

        current += timedelta(days=1)
        processed += 1

        if progress_callback and total_days > 0:
            progress_callback(processed / total_days)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return df
