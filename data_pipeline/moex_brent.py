"""Read Brent oil prices from existing brent_history table in Supabase."""

import pandas as pd

from data_pipeline.db import get_client


def fetch_brent_from_db() -> pd.DataFrame:
    """
    Read all Brent oil prices from the brent_history table.
    Returns DataFrame with columns: trade_date, close.
    """
    client = get_client()
    all_data: list[dict] = []
    offset = 0
    page_size = 1000
    while True:
        resp = (
            client.table("brent_history")
            .select("trade_date,close")
            .order("trade_date")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        if not resp.data:
            break
        all_data.extend(resp.data)
        if len(resp.data) < page_size:
            break
        offset += page_size

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df
