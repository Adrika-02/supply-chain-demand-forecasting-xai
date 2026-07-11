"""
Random Forest Regressor (scikit-learn) -- global demand model trained across
all 1,115 distribution nodes at once using the engineered feature table.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.models.config import get_cutoff_date

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
FEATURES_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.parquet"
DROP_COLS = ["Date", "Sales"]


def load_global_split(test_days: int | None = None):
    df = pd.read_parquet(FEATURES_PATH).dropna()
    cutoff = get_cutoff_date() if test_days is None else pd.Timestamp(df["Date"].max()) - pd.Timedelta(days=test_days)
    train = df[df["Date"] < cutoff]
    test = df[df["Date"] >= cutoff]
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    return train[feature_cols], train["Sales"], test[feature_cols], test["Sales"], test[["Store", "Date"]]


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> RandomForestRegressor:
    params = dict(n_estimators=200, max_depth=16, min_samples_leaf=3, n_jobs=-1, random_state=42)
    params.update(kwargs)
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model


def save_model(model, name: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


if __name__ == "__main__":
    from src.models.evaluate import score

    X_train, y_train, X_test, y_test, meta = load_global_split()
    model = train_random_forest(X_train, y_train)
    preds = model.predict(X_test)
    print(f"Random Forest (global, n_test={len(y_test)}):", score(y_test, preds))
    path = save_model(model, "random_forest_global")
    print(f"Saved to {path}")
