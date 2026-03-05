"""Fetch daily exchange rates from the Central Bank of Russia XML API."""

import xml.etree.ElementTree as ET
from datetime import date

import pandas as pd
import requests

CBR_URL = "https://www.cbr.ru/scripts/XML_dynamic.asp"
TIMEOUT = 30

# CBR internal currency codes
CURRENCY_CODES = {
    "USD/RUB": "R01235",
    "CNY/RUB": "R01375",
}


def _fetch_cbr_rates(pair: str, date_from: date, date_to: date) -> list[dict]:
    """Fetch daily rates for one currency pair from CBR XML API."""
    code = CURRENCY_CODES.get(pair)
    if not code:
        raise ValueError(f"Unknown pair: {pair}")

    params = {
        "date_req1": date_from.strftime("%d/%m/%Y"),
        "date_req2": date_to.strftime("%d/%m/%Y"),
        "VAL_NM_RQ": code,
    }

    r = requests.get(CBR_URL, params=params, timeout=TIMEOUT)
    r.raise_for_status()

    # CBR returns windows-1251 encoded XML
    root = ET.fromstring(r.content)
    rows = []

    for record in root.findall("Record"):
        dt_str = record.attrib.get("Date", "")  # DD.MM.YYYY
        nominal = record.findtext("Nominal", "1")
        value_str = record.findtext("Value", "0")

        # CBR uses comma as decimal separator
        value = float(value_str.replace(",", "."))
        nom = float(nominal.replace(",", "."))
        rate = value / nom

        # Convert date format
        parts = dt_str.split(".")
        if len(parts) == 3:
            iso_date = f"{parts[2]}-{parts[1]}-{parts[0]}"
        else:
            continue

        rows.append({
            "trade_date": iso_date,
            "pair": pair,
            "rate": round(rate, 4),
        })

    return rows


def fetch_all_currencies(date_from: date, date_to: date,
                         progress_callback=None) -> pd.DataFrame:
    """Fetch USD/RUB and CNY/RUB rates for the given period."""
    all_rows = []
    pairs = list(CURRENCY_CODES.keys())

    for i, pair in enumerate(pairs):
        rows = _fetch_cbr_rates(pair, date_from, date_to)
        all_rows.extend(rows)
        if progress_callback:
            progress_callback((i + 1) / len(pairs))

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
