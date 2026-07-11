"""
Vectorized recursive multi-step forecasting for the store x item (SKU x
location) demand grain, plus a real statistical seasonality test per item.

Mirrors the recursive-batch approach in src/models/forecast.py (feed each
day's prediction back in as history for the next day's lag/rolling
features), scoped to the store x item dataset used by the Product-Level
Forecasting dashboard page.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
FEATURES_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "item_features.parquet"
DROP_COLS = ["Date", "Sales"]


def load_item_model():
    return joblib.load(MODELS_DIR / "item_xgboost.joblib")


def load_item_feature_frame() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)


def _build_future_row(history_sales: pd.Series, date: pd.Timestamp, store: int, item: int, expanding_mean: float) -> dict:
    weekday = date.weekday() + 1
    return {
        "Store": store,
        "Item": item,
        "Year": date.year,
        "Month": date.month,
        "Quarter": (date.month - 1) // 3 + 1,
        "DayOfWeek": weekday,
        "WeekOfYear": int(date.isocalendar().week),
        "IsWeekend": int(weekday in (6, 7)),
        "IsMonthStart": int(date.is_month_start),
        "IsMonthEnd": int(date.is_month_end),
        "SalesLag7": history_sales.iloc[-7] if len(history_sales) >= 7 else history_sales.mean(),
        "SalesLag14": history_sales.iloc[-14] if len(history_sales) >= 14 else history_sales.mean(),
        "SalesLag28": history_sales.iloc[-28] if len(history_sales) >= 28 else history_sales.mean(),
        "RollingMean7": history_sales.tail(7).mean(),
        "RollingStd7": history_sales.tail(7).std(ddof=0),
        "RollingMean28": history_sales.tail(28).mean(),
        "RollingStd28": history_sales.tail(28).std(ddof=0),
        "StoreItemAvgSalesExpanding": expanding_mean,
    }


def batch_forecast_all_pairs(horizon_days: int, df: pd.DataFrame | None = None, model=None) -> pd.DataFrame:
    """Vectorized recursive forecast for every (Store, Item) pair at once --
    builds all pairs' next-day feature rows together and predicts in a single
    batched call per horizon day, avoiding 500x the per-call overhead of
    looping pair-by-pair."""
    if df is None:
        df = load_item_feature_frame()
    if model is None:
        model = load_item_model()

    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    pair_states = {}
    for (store, item), grp in df.sort_values("Date").groupby(["Store", "Item"]):
        if grp.empty:
            continue
        pair_states[(store, item)] = {
            "history": grp["Sales"].reset_index(drop=True),
            "last_date": grp["Date"].max(),
            "expanding_mean": float(grp["Sales"].mean()),
            "expanding_n": len(grp),
        }

    all_predictions = {key: [] for key in pair_states}

    for day in range(1, horizon_days + 1):
        rows = []
        keys_this_round = []
        for (store, item), state in pair_states.items():
            date = state["last_date"] + pd.Timedelta(days=day)
            row = _build_future_row(state["history"], date, store, item, state["expanding_mean"])
            rows.append(row)
            keys_this_round.append((store, item))
        batch_df = pd.DataFrame(rows)[feature_cols]
        preds = np.clip(model.predict(batch_df), 0, None)
        for (store, item), pred in zip(keys_this_round, preds):
            pred = float(pred)
            all_predictions[(store, item)].append(pred)
            state = pair_states[(store, item)]
            state["history"] = pd.concat([state["history"], pd.Series([pred])], ignore_index=True)
            n = state["expanding_n"]
            state["expanding_mean"] = (state["expanding_mean"] * n + pred) / (n + 1)
            state["expanding_n"] = n + 1

    records = []
    for (store, item), preds in all_predictions.items():
        last_date = pair_states[(store, item)]["last_date"]
        for i, pred in enumerate(preds, start=1):
            records.append({
                "Date": last_date + pd.Timedelta(days=i),
                "Store": store,
                "Item": item,
                "PredictedSales": pred,
            })
    return pd.DataFrame(records)


def seasonality_test(df: pd.DataFrame, store: int, item: int) -> dict:
    """Kruskal-Wallis test of whether daily sales differ significantly across
    calendar months for this store-item pair -- a real, data-derived answer
    to "is this product seasonal", not a heuristic guess."""
    pair = df[(df["Store"] == store) & (df["Item"] == item)]
    groups = [g["Sales"].values for _, g in pair.groupby("Month")]
    stat, p_value = stats.kruskal(*groups)
    monthly_avg = pair.groupby("Month")["Sales"].mean()
    peak_month = int(monthly_avg.idxmax())
    trough_month = int(monthly_avg.idxmin())
    seasonal_swing_pct = float((monthly_avg.max() - monthly_avg.min()) / monthly_avg.mean() * 100)
    return {
        "is_seasonal": bool(p_value < 0.05),
        "p_value": float(p_value),
        "kruskal_stat": float(stat),
        "peak_month": peak_month,
        "trough_month": trough_month,
        "seasonal_swing_pct": seasonal_swing_pct,
    }


if __name__ == "__main__":
    _df = load_item_feature_frame()
    _preds = batch_forecast_all_pairs(horizon_days=14, df=_df)
    print(_preds.head())
    print(seasonality_test(_df, store=1, item=1))
