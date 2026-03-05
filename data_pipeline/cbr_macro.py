"""Fetch M2 money supply and key rate from the Central Bank of Russia."""

import xml.etree.ElementTree as ET
from datetime import date, datetime

import pandas as pd
import requests

TIMEOUT = 30


def _soap_request(method: str, params_xml: str) -> bytes:
    """Send a SOAP request to CBR DailyInfo web service."""
    url = "https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx"
    envelope = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                   xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                   xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
      <soap:Body>
        <{method} xmlns="http://web.cbr.ru/">
          {params_xml}
        </{method}>
      </soap:Body>
    </soap:Envelope>"""

    headers = {
        "Content-Type": "text/xml; charset=utf-8",
        "SOAPAction": f"http://web.cbr.ru/{method}",
    }
    r = requests.post(url, data=envelope.encode("utf-8"),
                      headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.content


def fetch_key_rate(date_from: date, date_to: date) -> pd.DataFrame:
    """Fetch CBR key rate history via SOAP API."""
    params_xml = (
        f"<fromDate>{date_from.isoformat()}</fromDate>"
        f"<ToDate>{date_to.isoformat()}</ToDate>"
    )
    content = _soap_request("KeyRate", params_xml)
    root = ET.fromstring(content)

    rows = []
    # Tags are plain (no namespace): KR > DT, Rate
    for kr in root.iter("KR"):
        dt_el = kr.find("DT")
        rate_el = kr.find("Rate")
        if dt_el is not None and rate_el is not None and dt_el.text and rate_el.text:
            dt_str = dt_el.text.split("T")[0]
            rate = float(rate_el.text)
            rows.append({
                "period_date": dt_str,
                "indicator": "KEY_RATE",
                "value": rate,
            })

    if rows:
        # Deduplicate: keep only dates where rate changes (step function)
        df = pd.DataFrame(rows).sort_values("period_date")
        df = df.drop_duplicates(subset=["value"], keep="first")
        return df

    return pd.DataFrame()


def fetch_m2(date_from: date, date_to: date) -> pd.DataFrame:
    """
    Fetch M2 money supply from CBR SOAP API (mrrf7D — weekly M2 data).
    Raw values appear to be in 100s of billions RUB.
    """
    params_xml = (
        f"<fromDate>{date_from.isoformat()}</fromDate>"
        f"<ToDate>{date_to.isoformat()}</ToDate>"
    )
    content = _soap_request("mrrf7D", params_xml)
    root = ET.fromstring(content)

    rows = []
    # Tags are plain: mr > D0, val
    for mr in root.iter("mr"):
        dt_el = mr.find("D0")
        val_el = mr.find("val")
        if dt_el is not None and val_el is not None and dt_el.text and val_el.text:
            dt_str = dt_el.text.split("T")[0]
            # mrrf7D returns M2 in 100s of billions RUB; convert to trillions
            val = float(val_el.text)
            rows.append({
                "period_date": dt_str,
                "indicator": "M2",
                "value": round(val / 10, 2),  # trillions RUB
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_all_macro(date_from: date, date_to: date,
                    progress_callback=None) -> pd.DataFrame:
    """Fetch key rate and M2 money supply."""
    frames = []

    try:
        kr = fetch_key_rate(date_from, date_to)
        if not kr.empty:
            frames.append(kr)
    except Exception as e:
        print(f"Warning: failed to fetch key rate: {e}")
    if progress_callback:
        progress_callback(0.5)

    try:
        m2 = fetch_m2(date_from, date_to)
        if not m2.empty:
            frames.append(m2)
    except Exception as e:
        print(f"Warning: failed to fetch M2: {e}")
    if progress_callback:
        progress_callback(1.0)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
