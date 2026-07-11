"""
Feature engineering for SKU-category-node demand forecasting.

Builds date, lag, rolling-window, promotion-interaction, and store-encoding
features from `sales_enriched`, and writes the result to
data/processed/features.parquet for the modeling step.

Leakage guards:
  - `Customers` is contemporaneous with `Sales` (recorded on the same day, not
    knowable ahead of a forecast horizon) so it is dropped from the feature
    set entirely -- it must never be used as a model input.
  - All lag and rolling-window features are computed on values *strictly
    before* the row being scored (rolling stats use `.shift(1)` before the
    window), so no row ever sees its own or a future day's sales.
  - The store-level target encoding uses an expanding mean shifted by one
    row, so a store's historical average never includes the current day.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.db_utils import run_query

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
LAG_DAYS = (7, 14, 30)
ROLLING_WINDOWS = (7, 30)

PROMO_INTERVAL_MONTHS = {
    "Jan,Apr,Jul,Oct": {1, 4, 7, 10},
    "Feb,May,Aug,Nov": {2, 5, 8, 11},
    "Mar,Jun,Sept,Dec": {3, 6, 9, 12},
}


def load_raw() -> pd.DataFrame:
    df = run_query(
        """
        SELECT Store, Date, DayOfWeek, Sales, Open, Promo, StateHoliday,
               SchoolHoliday, StoreType, Assortment, CompetitionDistance,
               CompetitionOpenSinceMonth, CompetitionOpenSinceYear,
               Promo2, Promo2SinceWeek, Promo2SinceYear, PromoInterval
        FROM sales_enriched
        """
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    return df


def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)
    df["IsHoliday"] = (df["StateHoliday"] != "0").astype(int)
    return df


def add_holiday_proximity_features(df: pd.DataFrame) -> pd.DataFrame:
    def _distances(group: pd.DataFrame) -> pd.DataFrame:
        dates = group["Date"].values.astype("datetime64[D]")
        holiday_dates = dates[group["IsHoliday"].values == 1]
        if len(holiday_dates) == 0:
            group["DaysToHoliday"] = 9999
            group["DaysSinceHoliday"] = 9999
            return group

        idx = np.searchsorted(holiday_dates, dates)
        next_idx = np.clip(idx, 0, len(holiday_dates) - 1)
        prev_idx = np.clip(idx - 1, 0, len(holiday_dates) - 1)

        days_to = (holiday_dates[next_idx] - dates).astype("timedelta64[D]").astype(int)
        days_to = np.where(idx >= len(holiday_dates), 9999, days_to)
        days_since = (dates - holiday_dates[prev_idx]).astype("timedelta64[D]").astype(int)
        days_since = np.where(idx == 0, 9999, days_since)

        group["DaysToHoliday"] = np.clip(days_to, 0, None)
        group["DaysSinceHoliday"] = np.clip(days_since, 0, None)
        return group

    df = df.groupby("Store", group_keys=False).apply(_distances)
    return df


def add_lag_features(df: pd.DataFrame, lags: tuple = LAG_DAYS) -> pd.DataFrame:
    grouped = df.groupby("Store")["Sales"]
    for lag in lags:
        df[f"SalesLag{lag}"] = grouped.shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: tuple = ROLLING_WINDOWS) -> pd.DataFrame:
    shifted = df.groupby("Store")["Sales"].shift(1)
    for window in windows:
        df[f"RollingMean{window}"] = shifted.groupby(df["Store"]).transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df[f"RollingStd{window}"] = shifted.groupby(df["Store"]).transform(
            lambda s: s.rolling(window, min_periods=1).std()
        )
    return df


def add_promo_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df["PromoWeekend"] = df["Promo"] * df["IsWeekend"]
    df["PromoDayOfWeek"] = df["Promo"] * df["DayOfWeek"]
    df["PromoSchoolHoliday"] = df["Promo"] * df["SchoolHoliday"]
    return df


def add_store_encodings(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, columns=["StoreType", "Assortment"], prefix=["StoreType", "Assortment"])

    df = df.sort_values(["Store", "Date"])
    shifted_sales = df.groupby("Store")["Sales"].shift(1)
    df["StoreAvgSalesExpanding"] = (
        shifted_sales.groupby(df["Store"]).expanding().mean().reset_index(level=0, drop=True)
    )

    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())

    comp_open = pd.to_datetime(
        dict(
            year=df["CompetitionOpenSinceYear"].fillna(df["Year"]),
            month=df["CompetitionOpenSinceMonth"].fillna(df["Month"]),
            day=1,
        ),
        errors="coerce",
    )
    months_since_comp = (df["Date"].dt.to_period("M") - comp_open.dt.to_period("M")).apply(
        lambda x: x.n if pd.notnull(x) else np.nan
    )
    df["MonthsSinceCompetition"] = months_since_comp.clip(lower=0).fillna(0)
    df["HasCompetitionInfo"] = df["CompetitionOpenSinceYear"].notna().astype(int)

    df["IsPromo2Month"] = df.apply(
        lambda row: int(
            row["Promo2"] == 1
            and isinstance(row["PromoInterval"], str)
            and row["Month"] in PROMO_INTERVAL_MONTHS.get(row["PromoInterval"], set())
        ),
        axis=1,
    )
    df["Promo2"] = df["Promo2"].fillna(0).astype(int)

    df = df.drop(
        columns=[
            "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear",
            "Promo2SinceWeek",
            "Promo2SinceYear",
            "PromoInterval",
            "StateHoliday",
        ]
    )
    return df


def build_feature_table() -> pd.DataFrame:
    df = load_raw()
    df = clean_outliers(df)
    df = add_date_features(df)
    df = add_holiday_proximity_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_promo_interactions(df)
    df = add_store_encodings(df)
    df = df.drop(columns=["Open"])
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    return df


def save_features(df: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "features.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    features = build_feature_table()
    out_path = save_features(features)
    print(f"Feature table shape: {features.shape}")
    print(f"Columns: {list(features.columns)}")
    print(f"NaN counts (top 10):\n{features.isna().sum().sort_values(ascending=False).head(10)}")
    print(f"Saved to {out_path}")
