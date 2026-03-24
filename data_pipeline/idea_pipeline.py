"""Pipeline: load investment ideas, fetch security history, compute impact, upsert."""

from __future__ import annotations

import os
from datetime import timedelta

import pandas as pd

from data_pipeline.db import get_client, upsert_rows
from data_pipeline.moex_security_history import fetch_multiple_tickers
from analytics.event_study import compute_abnormal_volume


# Estimation window needs ~120 trading days before event.
# 120 trading days ≈ 170 calendar days.  Add buffer.
CALENDAR_BUFFER_BEFORE = 200
# Event window goes +5 trading days ≈ +8 calendar days.  Add buffer.
CALENDAR_BUFFER_AFTER = 15


def load_ideas_from_xlsx(path: str) -> pd.DataFrame:
    """Load investment ideas registry from Excel file.

    Supports two formats:
    - New (header at row 1): columns id, analyst, ticker, date_start, link
    - Old (header at row 0): columns ticker, idea_date

    Returns DataFrame with columns: ticker, idea_date (str YYYY-MM-DD), source.
    """
    # Try new format first (row 0 is title, row 1 is header)
    df = pd.read_excel(path, header=1)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "date_start" in df.columns and "analyst" in df.columns:
        # New format: analyst -> source, date_start -> idea_date
        df = df.rename(columns={"date_start": "idea_date", "analyst": "source"})
        df["source"] = df["source"].astype(str).str.strip()
    elif "idea_date" not in df.columns:
        # Fallback: old format (header at row 0)
        df = pd.read_excel(path)
        df.columns = [str(c).strip().lower() for c in df.columns]

    df["idea_date"] = pd.to_datetime(df["idea_date"], dayfirst=True).dt.strftime("%Y-%m-%d")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    # Filter out non-standard tickers (contain spaces — not equity symbols)
    df = df[~df["ticker"].str.contains(r"\s", na=False)]

    if "source" not in df.columns:
        df["source"] = None

    return df[["ticker", "idea_date", "source"]].drop_duplicates(
        subset=["ticker", "idea_date"]
    )


def _ideas_to_rows(ideas_df: pd.DataFrame) -> list[dict]:
    """Convert ideas DataFrame to list of dicts for Supabase upsert."""
    rows = []
    for _, row in ideas_df.iterrows():
        r = {
            "ticker": row["ticker"],
            "idea_date": row["idea_date"],
            "idea_time": "12:00:00",
        }
        if row.get("source"):
            r["source"] = row["source"]
        rows.append(r)
    return rows


def _compute_date_ranges(ideas_df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Compute required date range per ticker (union of all estimation+event windows).

    Returns {ticker: (date_from, date_to)}.
    """
    ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for _, row in ideas_df.iterrows():
        ticker = row["ticker"]
        idea_dt = pd.Timestamp(row["idea_date"])
        start = idea_dt - timedelta(days=CALENDAR_BUFFER_BEFORE)
        end = idea_dt + timedelta(days=CALENDAR_BUFFER_AFTER)
        if ticker not in ranges:
            ranges[ticker] = (start, end)
        else:
            prev_start, prev_end = ranges[ticker]
            ranges[ticker] = (min(prev_start, start), max(prev_end, end))

    return {
        t: (s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"))
        for t, (s, e) in ranges.items()
    }


def run_idea_impact_pipeline(
    client=None,
    xlsx_path: str | None = None,
    progress_callback=None,
) -> dict:
    """Run the full idea impact pipeline.

    Steps:
    1. Load ideas from xlsx (first run) or Supabase
    2. Upsert ideas into investment_ideas table
    3. Fetch per-ticker daily history from MOEX ISS
    4. Upsert security history cache
    5. Compute abnormal volume for each idea
    6. Upsert impact results

    Returns summary dict.
    """
    if client is None:
        client = get_client()

    # --- Step 1: Load ideas ---
    if xlsx_path and os.path.exists(xlsx_path):
        ideas_df = load_ideas_from_xlsx(xlsx_path)
        _log(progress_callback, f"Loaded {len(ideas_df)} ideas from xlsx")
    else:
        resp = client.table("investment_ideas").select("ticker,idea_date,source").execute()
        ideas_df = pd.DataFrame(resp.data)
        if ideas_df.empty:
            _log(progress_callback, "No ideas in DB and no xlsx provided")
            return {"ideas": 0, "computed": 0}
        _log(progress_callback, f"Loaded {len(ideas_df)} ideas from Supabase")

    # --- Step 2: Upsert ideas ---
    idea_rows = _ideas_to_rows(ideas_df)
    n_ideas = upsert_rows(client, "investment_ideas", idea_rows)
    _log(progress_callback, f"Upserted {n_ideas} ideas")

    # --- Step 3: Fetch security history ---
    date_ranges = _compute_date_ranges(ideas_df)
    tickers = sorted(date_ranges.keys())

    # Check what we already have cached
    existing_tickers = set()
    try:
        resp = (
            client.table("idea_security_history")
            .select("ticker")
            .limit(1000)
            .execute()
        )
        existing_tickers = {r["ticker"] for r in resp.data}
    except Exception:
        pass

    # Fetch only tickers not yet cached (or all on first run)
    tickers_to_fetch = [t for t in tickers if t not in existing_tickers]
    if not tickers_to_fetch:
        # Still fetch if we have ideas but check date coverage
        _log(progress_callback, f"All {len(tickers)} tickers already cached")
    else:
        _log(progress_callback, f"Fetching history for {len(tickers_to_fetch)} tickers from MOEX ISS")

    all_history: list[dict] = []
    for t in tickers_to_fetch:
        d_from, d_to = date_ranges[t]
        rows = []
        try:
            from data_pipeline.moex_security_history import fetch_security_history
            rows = fetch_security_history(t, d_from, d_to)
        except Exception as e:
            _log(progress_callback, f"  Warning: {t} failed: {e}")
        all_history.extend(rows)
        _log(progress_callback, f"  {t}: {len(rows)} days")

    # --- Step 4: Upsert security history ---
    if all_history:
        n_hist = upsert_rows(client, "idea_security_history", all_history)
        _log(progress_callback, f"Upserted {n_hist} security history rows")

    # --- Step 5: Load full history from Supabase for computation ---
    _log(progress_callback, "Loading security history from Supabase...")
    history_records = []
    for t in tickers:
        resp = (
            client.table("idea_security_history")
            .select("trade_date,ticker,value_rub")
            .eq("ticker", t)
            .order("trade_date")
            .execute()
        )
        history_records.extend(resp.data)

    hist_df = pd.DataFrame(history_records)
    if hist_df.empty:
        _log(progress_callback, "No security history available")
        return {"ideas": len(ideas_df), "computed": 0}

    hist_df["trade_date"] = pd.to_datetime(hist_df["trade_date"])

    # --- Step 6: Compute abnormal volume for each idea ---
    _log(progress_callback, "Computing abnormal volumes...")
    impact_results: list[dict] = []
    skipped = 0

    for i, (_, idea) in enumerate(ideas_df.iterrows()):
        ticker = idea["ticker"]
        idea_date = idea["idea_date"]

        ticker_hist = hist_df[hist_df["ticker"] == ticker].set_index("trade_date")["value_rub"]

        if ticker_hist.empty:
            skipped += 1
            continue

        result = compute_abnormal_volume(
            ticker_hist,
            idea_date,
            est_window=(-120, -6),
            evt_window=(-1, 5),
        )

        if result is None:
            skipped += 1
            continue

        result["ticker"] = ticker
        result["idea_date"] = idea_date
        source = idea.get("source")
        if source:
            result["source"] = source
        impact_results.append(result)

    _log(progress_callback, f"Computed {len(impact_results)} impacts, skipped {skipped}")

    # --- Step 7: Upsert impact results ---
    if impact_results:
        # Convert to Supabase-friendly format (JSONB fields stay as dicts)
        n_impact = upsert_rows(client, "idea_impact_results", impact_results)
        _log(progress_callback, f"Upserted {n_impact} impact results")

    return {
        "ideas": len(ideas_df),
        "computed": len(impact_results),
        "skipped": skipped,
    }


def _log(callback, msg: str):
    """Log message via callback or print."""
    if callback:
        callback(msg)
    else:
        print(f"  [idea_pipeline] {msg}")
