"""Orchestrate full data pipeline: fetch, aggregate, compute factors, store."""

import sys
import os
from datetime import date, datetime, timedelta

import pandas as pd

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_pipeline.db import get_client, upsert_rows, max_date
from data_pipeline.moex_turnovers import fetch_turnovers
from data_pipeline.moex_indices import fetch_all_indices
from data_pipeline.cbr_currencies import fetch_all_currencies
from data_pipeline.cbr_macro import fetch_all_macro
from data_pipeline.cpi_data import load_cpi
from data_pipeline.moex_brent import fetch_brent_from_db
from data_pipeline.cbr_deposits import fetch_household_deposits
from data_pipeline.moex_intraday import fetch_intraday_candles, compute_realized_volatility
from data_pipeline.aggregator import to_weekly_volumes, forward_fill_monthly_to_weekly
from analytics.factors import compute_index_factors, compute_currency_factors
from analytics.daily_factors import compute_all_daily_factors


DATE_FROM = date(2018, 1, 1)


def fetch_all_rows(client, table: str) -> pd.DataFrame:
    """Fetch all rows from a Supabase table, handling the 1000-row default limit."""
    all_data = []
    offset = 0
    page_size = 1000
    while True:
        resp = (
            client.table(table)
            .select("*")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        if not resp.data:
            break
        all_data.extend(resp.data)
        if len(resp.data) < page_size:
            break
        offset += page_size
    return pd.DataFrame(all_data) if all_data else pd.DataFrame()


def run_full_pipeline(progress_callback=None):
    """
    Run the complete data pipeline:
    1. Fetch raw daily data from MOEX ISS and CBR
    2. Store in Supabase
    3. Compute weekly aggregates and factors
    4. Store weekly data
    """
    client = get_client()

    def update_progress(stage: str, pct: float):
        if progress_callback:
            progress_callback(stage, pct)

    date_to = date.today()

    # --- Step 1: MOEX turnovers ---
    update_progress("Загрузка оборотов MOEX...", 0.0)
    last_turnover = max_date(client, "vol_daily_turnovers")
    turnover_from = (
        datetime.strptime(last_turnover, "%Y-%m-%d").date() + timedelta(days=1)
        if last_turnover
        else DATE_FROM
    )

    if turnover_from <= date_to:
        turnovers_df = fetch_turnovers(
            turnover_from, date_to, delay=0.1,
            progress_callback=lambda p: update_progress("Загрузка оборотов MOEX...", p * 0.25),
        )
        if not turnovers_df.empty:
            upsert_rows(client, "vol_daily_turnovers", turnovers_df.to_dict("records"))

    # --- Step 2: Index history ---
    update_progress("Загрузка индексов (IMOEX, RGBI, RVI)...", 0.25)
    last_index = max_date(client, "vol_index_history")
    index_from = (
        datetime.strptime(last_index, "%Y-%m-%d").date() + timedelta(days=1)
        if last_index
        else DATE_FROM
    )

    if index_from <= date_to:
        index_df = fetch_all_indices(
            index_from, date_to, delay=0.1,
            progress_callback=lambda p: update_progress("Загрузка индексов...", 0.25 + p * 0.15),
        )
        if not index_df.empty:
            upsert_rows(client, "vol_index_history", index_df.to_dict("records"))

    # --- Step 3: CBR exchange rates ---
    update_progress("Загрузка курсов валют (ЦБР)...", 0.40)
    last_rate = max_date(client, "vol_currency_rates")
    rate_from = (
        datetime.strptime(last_rate, "%Y-%m-%d").date() + timedelta(days=1)
        if last_rate
        else DATE_FROM
    )

    if rate_from <= date_to:
        rates_df = fetch_all_currencies(
            rate_from, date_to,
            progress_callback=lambda p: update_progress("Загрузка курсов валют...", 0.40 + p * 0.10),
        )
        if not rates_df.empty:
            upsert_rows(client, "vol_currency_rates", rates_df.to_dict("records"))

    # --- Step 4: Macro (key rate, M2, CPI) ---
    update_progress("Загрузка макроданных...", 0.50)
    macro_df = fetch_all_macro(
        DATE_FROM, date_to,
        progress_callback=lambda p: update_progress("Загрузка макроданных...", 0.50 + p * 0.10),
    )

    # Add CPI from bundled CSV
    cpi_df = load_cpi()
    if not cpi_df.empty:
        macro_df = pd.concat([macro_df, cpi_df], ignore_index=True)

    if not macro_df.empty:
        upsert_rows(client, "vol_macro", macro_df.to_dict("records"))

    # --- Step 5: Compute weekly volumes ---
    update_progress("Вычисление недельных агрегатов...", 0.60)

    # Fetch all daily turnovers from DB (paginated)
    turnovers_full = fetch_all_rows(client, "vol_daily_turnovers")

    if not turnovers_full.empty:
        weekly_vol = to_weekly_volumes(turnovers_full)
        if not weekly_vol.empty:
            upsert_rows(client, "vol_weekly_volumes", weekly_vol.to_dict("records"))

    # --- Step 6: Compute weekly factors ---
    update_progress("Вычисление факторов...", 0.75)

    # Fetch all index data (paginated)
    index_full = fetch_all_rows(client, "vol_index_history")

    # Fetch all currency data (paginated)
    rates_full = fetch_all_rows(client, "vol_currency_rates")

    factor_frames = []

    # Index-derived factors (volatility, ADX, trend direction, index level, market cap)
    if not index_full.empty:
        idx_factors = compute_index_factors(index_full)
        factor_frames.append(idx_factors)

    # Currency factors (USD/RUB, CNY/RUB)
    if not rates_full.empty:
        cur_factors = compute_currency_factors(rates_full)
        factor_frames.append(cur_factors)

    # Macro factors (CPI, M2, key rate) - forward-fill monthly to weekly
    macro_full = fetch_all_rows(client, "vol_macro")

    if not macro_full.empty and not turnovers_full.empty:
        # Get weekly dates from volume data
        turnovers_full["trade_date"] = pd.to_datetime(turnovers_full["trade_date"])
        turnovers_full["week_start"] = turnovers_full["trade_date"] - pd.to_timedelta(
            turnovers_full["trade_date"].dt.dayofweek, unit="D"
        )
        weekly_dates = sorted(turnovers_full["week_start"].dt.strftime("%Y-%m-%d").unique())

        macro_factors = forward_fill_monthly_to_weekly(macro_full, weekly_dates)
        factor_frames.append(macro_factors)

    if factor_frames:
        all_factors = pd.concat(factor_frames, ignore_index=True)
        if not all_factors.empty:
            upsert_rows(client, "vol_weekly_factors", all_factors.to_dict("records"))

    # --- Step 7: Fetch intraday candles & compute Realized Volatility ---
    update_progress("Загрузка внутридневных данных IMOEX...", 0.80)

    rv_series = None
    try:
        candles_df = fetch_intraday_candles(
            DATE_FROM, date_to, interval=10, delay=0.05,
            progress_callback=lambda p: update_progress(
                "Загрузка внутридневных данных IMOEX...", 0.80 + p * 0.05,
            ),
        )
        if not candles_df.empty:
            rv_series = compute_realized_volatility(candles_df, annualize=True)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Failed to fetch intraday candles: %s", e)

    # --- Step 8: Compute daily factors ---
    update_progress("Вычисление дневных факторов...", 0.85)

    # Re-fetch full daily tables for factor computation
    if turnovers_full.empty:
        turnovers_full = fetch_all_rows(client, "vol_daily_turnovers")
    if index_full.empty:
        index_full = fetch_all_rows(client, "vol_index_history")
    if rates_full.empty:
        rates_full = fetch_all_rows(client, "vol_currency_rates")
    if macro_full.empty:
        macro_full = fetch_all_rows(client, "vol_macro")

    brent_df = fetch_brent_from_db()

    # Fetch household deposits from CBR
    try:
        deposits_df = fetch_household_deposits(DATE_FROM)
        if not deposits_df.empty:
            upsert_rows(client, "vol_macro", deposits_df.to_dict("records"))
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Failed to fetch CBR deposits: %s", e)
        deposits_df = pd.DataFrame()

    daily_factors = compute_all_daily_factors(
        index_df=index_full,
        currency_df=rates_full,
        macro_df=macro_full,
        brent_df=brent_df,
        turnovers_df=turnovers_full,
        deposits_df=deposits_df,
        rv_series=rv_series,
    )
    if not daily_factors.empty:
        upsert_rows(client, "vol_daily_factors", daily_factors.to_dict("records"))

    update_progress("Готово!", 1.0)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    def print_progress(stage, pct):
        print(f"  [{pct:.0%}] {stage}")

    print("Starting full data pipeline...")
    run_full_pipeline(progress_callback=print_progress)
    print("Pipeline complete.")
