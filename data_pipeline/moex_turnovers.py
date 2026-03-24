"""Fetch daily trading turnovers from MOEX ISS by engine/market.

For most markets the engine-level /turnovers.json endpoint is used (one row per
market). The NDM market is an exception: it mixes equity and bond boards, so we
fetch board-level turnovers and classify each board individually.
"""

import logging
import time
from datetime import date, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Mapping from (engine, market) to normalized instrument_class ──
# Used for engine-level turnovers where one row = one market.
MARKET_CLASS_MAP = {
    ("stock", "shares"): "shares",
    ("stock", "foreignshares"): "shares",   # foreign equities / ADRs on MOEX
    ("stock", "bonds"): "bonds",
    # NDM is handled separately at board level — see NDM_BOARD_CLASS_MAP
    ("stock", "repo"): "repo",
    ("stock", "ccp"): "repo",   # CCP repo
    ("stock", "gcc"): "repo",   # GCC repo
    ("currency", "selt"): "currency",
    ("currency", "otc"): "currency",
    ("futures", "forts"): "futures",
    ("futures", "options"): "options",
}

# ── NDM board → instrument_class ──
# Boards whose title contains equity-related keywords go to "shares";
# everything else in NDM goes to "bonds".
NDM_EQUITY_BOARDS = {
    # РПС / РПС с ЦК by equity (shares, ETF, funds, DRs)
    "PSEQ", "PTEQ",    # Акции и ДР
    "PSDE", "PTDE",    # Акции Д
    "PSES", "PTES",    # А2-Акции и паи
    "PSNE", "PTNE",    # Акции, паи и ДР внесписочные
    "PSNL", "PTNL",    # Б-Акции и паи
    "PSLV", "PTLV",    # В-Акции и ДР
    "PSLI", "PTLI",    # И-Акции
    "PSSE", "PTSE",    # Акции и ДР (EUR)
    "PSIF", "PTIF",    # Паи
    "PSFD", "PTFD",    # Паи (USD)
    "PSFE", "PTFE",    # Паи (EUR)
    "PSTH", "PTTH",    # Паи (HKD)
    "PSTY", "PTTY",    # Паи (CNY)
    "PSTF", "PTTF",    # ETF
    "PSTD", "PTTD",    # ETF (USD)
    "PSTE", "PTTE",    # ETF (EUR)
    "PSTC", "PTTC",    # ETC
    # ПИР (shares for qualified investors)
    "PSPI", "PTPI",    # Акции ПИР
    "PSPD", "PTPD",    # Акции ПИР (USD)
    "PSPE", "PTPE",    # Акции ПИР (EUR)
    "PSPH", "PTPH",    # Акции ПИР (HKD)
    "PSPY", "PTPY",    # Акции ПИР (CNY)
}

BASE_URL = "https://iss.moex.com/iss"
TIMEOUT = 30


def _fetch_ndm_board_turnovers(dt: date) -> list[dict]:
    """Fetch NDM turnovers at board level and classify equity vs bonds."""
    date_str = dt.strftime("%Y-%m-%d")
    url = f"{BASE_URL}/engines/stock/markets/ndm/turnovers.json"
    params = {
        "iss.meta": "off",
        "is_tonight_session": 0,
        "date": date_str,
    }
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning("MOEX NDM board turnovers error for %s: %s", date_str, e)
        return []

    turnovers = data.get("turnovers", {})
    columns = turnovers.get("columns", [])
    raw_data = turnovers.get("data", [])

    # Accumulate by instrument_class (shares vs bonds)
    accum: dict[str, dict] = {}  # instrument_class -> {value_rub, num_trades}
    for row in raw_data:
        rec = dict(zip(columns, row))
        board_id = rec.get("BOARDID", "")
        value_rub = rec.get("VALTODAY") or rec.get("VALUE") or 0
        num_trades = rec.get("NUMTRADES") or 0
        if not value_rub:
            continue

        instrument_class = "shares" if board_id in NDM_EQUITY_BOARDS else "bonds"
        if instrument_class not in accum:
            accum[instrument_class] = {"value_rub": 0.0, "num_trades": 0}
        accum[instrument_class]["value_rub"] += float(value_rub)
        accum[instrument_class]["num_trades"] += int(num_trades)

    rows = []
    for instrument_class, vals in accum.items():
        if vals["value_rub"] > 0:
            # Use distinct market names so upsert conflict key works
            # (trade_date, engine, market) must be unique per row
            market_name = "ndm_equity" if instrument_class == "shares" else "ndm"
            rows.append({
                "trade_date": date_str,
                "engine": "stock",
                "market": market_name,
                "instrument_class": instrument_class,
                "value_rub": vals["value_rub"],
                "num_trades": vals["num_trades"],
            })
    return rows


def _fetch_turnovers_for_date(dt: date) -> list[dict]:
    """Fetch turnover breakdown for a single date from all engines."""
    date_str = dt.strftime("%Y-%m-%d")
    rows = []

    # Engine-level turnovers (one row per market)
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
        except Exception as e:
            logger.warning("MOEX API error for %s/%s: %s", engine, date_str, e)
            continue

        turnovers = data.get("turnovers", {})
        columns = turnovers.get("columns", [])
        raw_data = turnovers.get("data", [])

        for row in raw_data:
            rec = dict(zip(columns, row))
            market = rec.get("MARKET", "").lower() if rec.get("MARKET") else ""
            if not market:
                market = rec.get("NAME", "").lower()

            # Skip NDM — handled separately at board level
            if engine == "stock" and market == "ndm":
                continue

            key = (engine, market)
            instrument_class = MARKET_CLASS_MAP.get(key)
            if not instrument_class:
                continue

            value_rub = rec.get("VALTODAY") or rec.get("VALUE") or 0
            num_trades = rec.get("NUMTRADES") or 0

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

    # Board-level NDM turnovers (split equity vs bonds)
    rows.extend(_fetch_ndm_board_turnovers(dt))

    return rows


def fetch_turnovers(date_from: date, date_to: date,
                    delay: float = 0.1,
                    progress_callback=None) -> pd.DataFrame:
    """
    Fetch daily turnovers for a date range.
    Iterates through each trading day (Mon-Fri), calls 3 engine endpoints
    + 1 NDM market-level endpoint.
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
