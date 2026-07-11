"""
Recursive multi-step demand forecasting using the production XGBoost model.

Tree models predict one day at a time, so forecasting N days ahead requires
feeding each day's prediction back in as the "actual" for the next day's lag
and rolling-window features -- the standard approach for multi-step
tree-based forecasting. Calendar features are computed deterministically;
recurring holidays (e.g. Christmas) are inferred from historical month/day
patterns. Promo flags for future (unobserved) dates default to that store's
historical same-weekday promo frequency unless overridden, which also powers
"what-if" scenario planning in the dashboard (force promotions on/off).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
FEATURES_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.parquet"
DROP_COLS = ["Date", "Sales"]

STORE_TYPE_COLS = ["StoreType_a", "StoreType_b", "StoreType_c", "StoreType_d"]
ASSORTMENT_COLS = ["Assortment_a", "Assortment_b", "Assortment_c"]


def load_best_model():
    return joblib.load(MODELS_DIR / "best_model.joblib")


def load_feature_frame() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)


def _infer_holiday_month_days(df: pd.DataFrame) -> set:
    holiday_dates = df.loc[df["IsHoliday"] == 1, "Date"]
    return set(zip(holiday_dates.dt.month, holiday_dates.dt.day))


def _build_future_row(history_sales: pd.Series, date: pd.Timestamp, promo: int, static: dict, holiday_month_days: set) -> dict:
    is_holiday = int((date.month, date.day) in holiday_month_days)
    row = {
        "Store": static["Store"],
        "DayOfWeek": date.isoweekday(),
        "Promo": promo,
        "SchoolHoliday": static["SchoolHoliday"],
        "CompetitionDistance": static["CompetitionDistance"],
        "Promo2": static["Promo2"],
        "Year": date.year,
        "Month": date.month,
        "Quarter": (date.month - 1) // 3 + 1,
        "WeekOfYear": int(date.isocalendar().week),
        "IsWeekend": int(date.isoweekday() in (6, 7)),
        "IsMonthStart": int(date.is_month_start),
        "IsMonthEnd": int(date.is_month_end),
        "IsHoliday": is_holiday,
        "DaysToHoliday": 0 if is_holiday else 9999,
        "DaysSinceHoliday": 0 if is_holiday else 9999,
        "SalesLag7": history_sales.iloc[-7] if len(history_sales) >= 7 else history_sales.mean(),
        "SalesLag14": history_sales.iloc[-14] if len(history_sales) >= 14 else history_sales.mean(),
        "SalesLag30": history_sales.iloc[-30] if len(history_sales) >= 30 else history_sales.mean(),
        "RollingMean7": history_sales.tail(7).mean(),
        "RollingStd7": history_sales.tail(7).std(ddof=0),
        "RollingMean30": history_sales.tail(30).mean(),
        "RollingStd30": history_sales.tail(30).std(ddof=0),
        "PromoWeekend": promo * int(date.isoweekday() in (6, 7)),
        "PromoDayOfWeek": promo * date.isoweekday(),
        "PromoSchoolHoliday": promo * static["SchoolHoliday"],
        "StoreAvgSalesExpanding": history_sales.mean(),
        "MonthsSinceCompetition": static["MonthsSinceCompetition"],
        "HasCompetitionInfo": static["HasCompetitionInfo"],
        "IsPromo2Month": int(static["Promo2"] == 1 and date.month in static.get("promo2_months", set())),
    }
    for col in STORE_TYPE_COLS + ASSORTMENT_COLS:
        row[col] = static[col]
    return row


def forecast_store(
    store_id: int,
    horizon_days: int,
    promo_override: int | None = None,
    df: pd.DataFrame | None = None,
    model=None,
) -> pd.DataFrame:
    """Recursively forecast `horizon_days` beyond the store's last known date.

    If promo_override is None, uses that store's historical same-weekday
    promo frequency (rounded) as the default; pass 0 or 1 to force
    promotions off/on for the whole horizon (what-if scenario planning).
    """
    if df is None:
        df = load_feature_frame()
    if model is None:
        model = load_best_model()

    store_df = df[df["Store"] == store_id].sort_values("Date").reset_index(drop=True)
    if store_df.empty:
        raise ValueError(f"No data for Store {store_id}")

    last_row = store_df.iloc[-1]
    static = {
        "Store": store_id,
        "SchoolHoliday": 0,
        "CompetitionDistance": last_row["CompetitionDistance"],
        "Promo2": last_row["Promo2"],
        "MonthsSinceCompetition": last_row["MonthsSinceCompetition"],
        "HasCompetitionInfo": last_row["HasCompetitionInfo"],
    }
    for col in STORE_TYPE_COLS + ASSORTMENT_COLS:
        static[col] = last_row[col]

    promo_by_weekday = (
        store_df.groupby(store_df["Date"].dt.weekday + 1)["Promo"].mean().round().to_dict()
    )
    holiday_month_days = _infer_holiday_month_days(df)

    history_sales = store_df["Sales"].copy()
    last_date = store_df["Date"].max()
    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    predictions = []
    for i in range(1, horizon_days + 1):
        date = last_date + pd.Timedelta(days=i)
        promo = promo_override if promo_override is not None else int(promo_by_weekday.get(date.isoweekday(), 0))
        row = _build_future_row(history_sales, date, promo, static, holiday_month_days)
        X_row = pd.DataFrame([row])[feature_cols]
        pred = float(model.predict(X_row)[0])
        predictions.append({"Date": date, "Store": store_id, "PredictedSales": pred})
        history_sales = pd.concat([history_sales, pd.Series([pred])], ignore_index=True)

    return pd.DataFrame(predictions)


def compute_residual_quantiles(model=None, df=None, lower_q: float = 0.1, upper_q: float = 0.9) -> tuple:
    """Empirical residual-based interval from the global holdout's actual
    errors (not a per-store conditional variance model, but a fast, honest
    approximation for confidence bands on forward forecasts)."""
    from src.models.config import get_cutoff_date

    if model is None:
        model = load_best_model()
    if df is None:
        df = load_feature_frame()
    df = df.dropna()
    cutoff = get_cutoff_date()
    test = df[df["Date"] >= cutoff]
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    preds = model.predict(test[feature_cols])
    residuals = test["Sales"].values - preds
    return float(np.quantile(residuals, lower_q)), float(np.quantile(residuals, upper_q))


def batch_forecast_all_stores(horizon_days: int, df: pd.DataFrame | None = None, model=None) -> pd.DataFrame:
    """Vectorized recursive forecast for every store at once.

    Builds all stores' next-day feature rows together and predicts in a
    single batched call per horizon day, instead of looping store-by-store
    (avoids ~1,115x the fixed per-call overhead of the naive per-store
    approach) -- the only tractable way to score the full network live in a
    dashboard.
    """
    if df is None:
        df = load_feature_frame()
    if model is None:
        model = load_best_model()

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    holiday_month_days = _infer_holiday_month_days(df)

    store_states = {}
    for store_id, store_df in df.sort_values("Date").groupby("Store"):
        if store_df.empty:
            continue
        last_row = store_df.iloc[-1]
        static = {
            "Store": store_id,
            "SchoolHoliday": 0,
            "CompetitionDistance": last_row["CompetitionDistance"],
            "Promo2": last_row["Promo2"],
            "MonthsSinceCompetition": last_row["MonthsSinceCompetition"],
            "HasCompetitionInfo": last_row["HasCompetitionInfo"],
        }
        for col in STORE_TYPE_COLS + ASSORTMENT_COLS:
            static[col] = last_row[col]
        promo_by_weekday = store_df.groupby(store_df["Date"].dt.weekday + 1)["Promo"].mean().round().to_dict()
        store_states[store_id] = {
            "history": store_df["Sales"].reset_index(drop=True),
            "static": static,
            "promo_by_weekday": promo_by_weekday,
            "last_date": store_df["Date"].max(),
        }

    all_predictions = {store_id: [] for store_id in store_states}

    for day in range(1, horizon_days + 1):
        rows = []
        store_ids_this_round = []
        for store_id, state in store_states.items():
            date = state["last_date"] + pd.Timedelta(days=day)
            promo = int(state["promo_by_weekday"].get(date.weekday() + 1, 0))
            row = _build_future_row(state["history"], date, promo, state["static"], holiday_month_days)
            rows.append(row)
            store_ids_this_round.append(store_id)
        batch_df = pd.DataFrame(rows)[feature_cols]
        preds = model.predict(batch_df)
        for store_id, pred in zip(store_ids_this_round, preds):
            all_predictions[store_id].append(float(pred))
            store_states[store_id]["history"] = pd.concat(
                [store_states[store_id]["history"], pd.Series([float(pred)])], ignore_index=True
            )

    records = []
    for store_id, preds in all_predictions.items():
        last_date = store_states[store_id]["last_date"]
        for i, pred in enumerate(preds, start=1):
            records.append({"Date": last_date + pd.Timedelta(days=i), "Store": store_id, "PredictedSales": pred})
    return pd.DataFrame(records)


if __name__ == "__main__":
    preds = forecast_store(store_id=262, horizon_days=30)
    print(preds)
    lo, hi = compute_residual_quantiles()
    print(f"\nEmpirical 10-90% residual band: [{lo:.1f}, {hi:.1f}]")
