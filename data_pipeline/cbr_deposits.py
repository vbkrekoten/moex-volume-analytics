"""Fetch household bank deposits data from CBR Data Service API.

Source: Bank of Russia (cbr.ru) — Monetary Statistics / M2 Structure.
Returns monthly volumes of household deposits in billions RUB.
"""

import logging
from datetime import date

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Note: use cbr.ru without 'www' — www.cbr.ru returns 501 on this endpoint
CBR_API_URL = "https://cbr.ru/dataservice/dataNew"


def fetch_household_deposits(
    date_from: date = date(2010, 1, 1),
    date_to: date | None = None,
) -> pd.DataFrame:
    """
    Fetch household deposit volumes from CBR Data Service API.

    Returns a DataFrame with columns: period_date, indicator, value.
    The 'indicator' column is 'HH_DEPOSITS' — total household deposits
    (checking + term + FX deposits of individuals).

    Data is monthly, in billions RUB.
    """
    if date_to is None:
        date_to = date.today()

    # m2_ids=22 — Term/other deposits of households (primary component)
    # Note: m2_ids=16 (checking) and m2_ids=26 (FX) return empty results
    params = {
        "categoryId": 5,
        "y1": date_from.year,
        "y2": date_to.year,
        "i_ids": "7",
        "m2_ids": "22",
    }

    try:
        resp = requests.get(
            CBR_API_URL,
            params=params,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.error("Failed to fetch CBR deposits data: %s", e)
        return pd.DataFrame()
    except ValueError as e:
        logger.error("Failed to parse CBR deposits JSON: %s", e)
        return pd.DataFrame()

    rows = data.get("RowData", [])
    if not rows:
        logger.warning("No deposit data returned from CBR API")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logger.info("Fetched %d raw deposit rows from CBR", len(df))

    # CBR API returns 'date' and 'obs_val' fields
    df["date"] = pd.to_datetime(df["date"])
    df["obs_val"] = pd.to_numeric(df["obs_val"], errors="coerce")

    # Aggregate the three deposit components by date (sum them)
    monthly = (
        df.groupby("date")["obs_val"]
        .sum()
        .reset_index()
    )
    monthly.columns = ["period_date", "value"]

    # Filter by date range
    monthly = monthly[
        (monthly["period_date"] >= pd.Timestamp(date_from))
        & (monthly["period_date"] <= pd.Timestamp(date_to))
    ].copy()

    # Format for vol_macro-compatible structure
    monthly["period_date"] = monthly["period_date"].dt.strftime("%Y-%m-%d")
    monthly["indicator"] = "HH_DEPOSITS"

    logger.info(
        "Processed %d monthly household deposit records (%.1f — %.1f bln RUB)",
        len(monthly),
        monthly["value"].min() if not monthly.empty else 0,
        monthly["value"].max() if not monthly.empty else 0,
    )

    return monthly[["period_date", "indicator", "value"]]
