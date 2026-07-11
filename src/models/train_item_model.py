"""
Trains a single global XGBoost regressor on the store x item demand grain.

Only one model is trained here (unlike the 4-model bake-off used to select
the Rossmann production model) -- this second dataset exists to demonstrate
genuine SKU x location forecasting and reorder-point math on the
Product-Level Forecasting dashboard page, not to repeat the full
model-selection exercise a second time.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features.build_item_features import build_feature_table

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
TEST_DAYS = 90

FEATURE_COLS = [
    "Store", "Item", "Year", "Month", "Quarter", "DayOfWeek", "WeekOfYear",
    "IsWeekend", "IsMonthStart", "IsMonthEnd",
    "SalesLag7", "SalesLag14", "SalesLag28",
    "RollingMean7", "RollingStd7", "RollingMean28", "RollingStd28",
    "StoreItemAvgSalesExpanding",
]


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def train_and_evaluate() -> dict:
    df = build_feature_table()
    df = df.dropna(subset=FEATURE_COLS)

    cutoff = df["Date"].max() - pd.Timedelta(days=TEST_DAYS)
    train = df[df["Date"] <= cutoff]
    test = df[df["Date"] > cutoff]

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[FEATURE_COLS], train["Sales"])

    preds = np.clip(model.predict(test[FEATURE_COLS]), 0, None)

    metrics = {
        "MAE": float(mean_absolute_error(test["Sales"], preds)),
        "RMSE": float(mean_squared_error(test["Sales"], preds) ** 0.5),
        "MAPE": mape(test["Sales"].values, preds),
        "test_days": TEST_DAYS,
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_store_item_pairs": int(df.groupby(["Store", "Item"]).ngroups),
        "n_stores": int(df["Store"].nunique()),
        "n_items": int(df["Item"].nunique()),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "item_xgboost.joblib")
    (REPORTS_DIR / "item_model_metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics


if __name__ == "__main__":
    result = train_and_evaluate()
    print(json.dumps(result, indent=2))
