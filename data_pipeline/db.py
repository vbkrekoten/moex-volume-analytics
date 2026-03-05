"""Supabase client and upsert helpers for vol_* tables."""

import os
from typing import Any

from supabase import create_client, Client

_client: Client | None = None

# Unique constraint columns for each table
CONFLICT_COLS = {
    "vol_daily_turnovers": "trade_date,engine,market",
    "vol_index_history": "trade_date,ticker",
    "vol_currency_rates": "trade_date,pair",
    "vol_macro": "period_date,indicator",
    "vol_weekly_volumes": "week_start,instrument_class",
    "vol_weekly_factors": "week_start,factor_name",
}


def get_client() -> Client:
    """Return a Supabase client, using env vars or Streamlit secrets."""
    global _client
    if _client is not None:
        return _client

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")

    # Try Streamlit secrets as fallback
    if not url or not key:
        try:
            import streamlit as st
            url = url or st.secrets.get("SUPABASE_URL", "")
            key = key or st.secrets.get("SUPABASE_KEY", "")
        except Exception:
            pass

    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")

    _client = create_client(url, key)
    return _client


def upsert_rows(client: Client, table: str, rows: list[dict[str, Any]],
                batch_size: int = 500) -> int:
    """Upsert rows into a Supabase table in batches. Returns total upserted."""
    if not rows:
        return 0

    conflict = CONFLICT_COLS.get(table, "")
    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        client.table(table).upsert(batch, on_conflict=conflict).execute()
        total += len(batch)
    return total


def max_date(client: Client, table: str, date_col: str = "trade_date") -> str | None:
    """Return the latest date in the table, or None if empty."""
    resp = (
        client.table(table)
        .select(date_col)
        .order(date_col, desc=True)
        .limit(1)
        .execute()
    )
    if resp.data:
        return resp.data[0][date_col]
    return None


def row_count(client: Client, table: str) -> int:
    """Return approximate row count."""
    resp = client.table(table).select("id", count="exact").limit(0).execute()
    return resp.count or 0
