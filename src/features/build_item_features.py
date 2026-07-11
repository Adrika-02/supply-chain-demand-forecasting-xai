"""
Feature engineering for store x item (SKU x location) demand forecasting.

Mirrors the leakage-safe pattern in src/features/build_features.py: every
lag and rolling-window feature is computed on values strictly before the row
being scored (`.shift(1)` before any rolling window, grouped by (Store,
Item) rather than Store alone), and the store-item target encoding uses an
expanding mean shifted by one row.
"""
from pathlib import Path

import pandas as pd

from src.utils.db_utils import run_query

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
LAG_DAYS = (7, 14, 28)
ROLLING_WINDOWS = (7, 28)


def load_raw() -> pd.DataFrame:
    df = run_query("SELECT date AS Date, store AS Store, item AS Item, sales AS Sales FROM item_sales")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store", "Item", "Date"]).reset_index(drop=True)
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["DayOfWeek"] = df["Date"].dt.weekday + 1
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)
    return df


def add_lag_features(df: pd.DataFrame, lags: tuple = LAG_DAYS) -> pd.DataFrame:
    grouped = df.groupby(["Store", "Item"])["Sales"]
    for lag in lags:
        df[f"SalesLag{lag}"] = grouped.shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: tuple = ROLLING_WINDOWS) -> pd.DataFrame:
    shifted = df.groupby(["Store", "Item"])["Sales"].shift(1)
    keys = [df["Store"], df["Item"]]
    for window in windows:
        df[f"RollingMean{window}"] = shifted.groupby(keys).transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df[f"RollingStd{window}"] = shifted.groupby(keys).transform(
            lambda s: s.rolling(window, min_periods=1).std()
        )
    return df


def add_target_encoding(df: pd.DataFrame) -> pd.DataFrame:
    shifted_sales = df.groupby(["Store", "Item"])["Sales"].shift(1)
    df["StoreItemAvgSalesExpanding"] = (
        shifted_sales.groupby([df["Store"], df["Item"]])
        .expanding()
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    return df


def build_feature_table() -> pd.DataFrame:
    df = load_raw()
    df = add_date_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_target_encoding(df)
    df = df.sort_values(["Store", "Item", "Date"]).reset_index(drop=True)
    return df


def save_features(df: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "item_features.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    features = build_feature_table()
    out_path = save_features(features)
    print(f"Feature table shape: {features.shape}")
    print(f"Columns: {list(features.columns)}")
    print(f"NaN counts (top 10):\n{features.isna().sum().sort_values(ascending=False).head(10)}")
    print(f"Saved to {out_path}")
