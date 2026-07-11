"""
XGBoost Regressor -- global demand model trained across all 1,115
distribution nodes at once using the engineered feature table. This is the
project's production candidate: gradient boosting typically captures the
nonlinear promo/holiday/lag interactions in retail demand better than a
single-tree-type Random Forest, and SHAP's TreeExplainer is exact for it.
"""
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBRegressor

from src.models.config import get_cutoff_date
from src.models.train_random_forest import DROP_COLS, FEATURES_PATH, MODELS_DIR, save_model


def load_global_split(test_days: int | None = None):
    df = pd.read_parquet(FEATURES_PATH).dropna()
    cutoff = get_cutoff_date() if test_days is None else pd.Timestamp(df["Date"].max()) - pd.Timedelta(days=test_days)
    train = df[df["Date"] < cutoff]
    test = df[df["Date"] >= cutoff]
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    return train[feature_cols], train["Sales"], test[feature_cols], test["Sales"], test[["Store", "Date"]]


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> XGBRegressor:
    params = dict(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    params.update(kwargs)
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    from src.models.evaluate import score

    X_train, y_train, X_test, y_test, meta = load_global_split()
    model = train_xgboost(X_train, y_train)
    preds = model.predict(X_test)
    print(f"XGBoost (global, n_test={len(y_test)}):", score(y_test, preds))
    path = save_model(model, "xgboost_global")
    print(f"Saved to {path}")
