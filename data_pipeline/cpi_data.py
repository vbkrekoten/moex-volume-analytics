"""Load CPI (Consumer Price Index) data from bundled CSV file."""

import os

import pandas as pd

# Path to bundled CPI data
CPI_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "cpi_russia.csv")


def load_cpi() -> pd.DataFrame:
    """
    Load Russian CPI data from CSV.
    Expected CSV columns: period_date, value (CPI index, base=100).
    Returns DataFrame with columns: period_date, indicator, value.
    """
    if not os.path.exists(CPI_CSV):
        return pd.DataFrame(columns=["period_date", "indicator", "value"])

    df = pd.read_csv(CPI_CSV)

    if "period_date" not in df.columns or "value" not in df.columns:
        return pd.DataFrame(columns=["period_date", "indicator", "value"])

    df["period_date"] = pd.to_datetime(df["period_date"]).dt.strftime("%Y-%m-%d")

    # Compute YoY inflation: (CPI_current / CPI_12months_ago - 1) * 100
    df = df.sort_values("period_date").reset_index(drop=True)
    df["cpi_yoy"] = (df["value"] / df["value"].shift(12) - 1) * 100

    rows = []
    for _, row in df.iterrows():
        # Store the raw CPI index
        rows.append({
            "period_date": row["period_date"],
            "indicator": "CPI_INDEX",
            "value": row["value"],
        })
        # Store YoY if available
        if pd.notna(row["cpi_yoy"]):
            rows.append({
                "period_date": row["period_date"],
                "indicator": "CPI_YOY",
                "value": round(row["cpi_yoy"], 2),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()
