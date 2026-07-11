"""
MAE / RMSE / MAPE evaluation, model comparison tables, and best-model
selection for production.

Two comparison tables are produced:
  - `model_comparison_single_store`: ARIMA, Prophet, Random Forest, and
    XGBoost evaluated on the identical date range for BENCHMARK_STORE only
    (an apples-to-apples bake-off -- ARIMA/Prophet don't scale to one model
    per store, so this is the only fair way to compare all four).
  - `model_comparison_global`: Random Forest and XGBoost evaluated across
    all 1,115 distribution nodes on the same holdout window -- this is what
    actually gets deployed, and what the resume-bullet accuracy numbers
    come from.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def score(y_true, y_pred) -> dict:
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "MAPE": mape(y_true, y_pred)}


def save_comparison(results: dict, filename: str) -> pd.DataFrame:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results).T
    df.to_csv(REPORTS_DIR / f"{filename}.csv")
    (REPORTS_DIR / f"{filename}.json").write_text(json.dumps(results, indent=2))
    return df


def run_full_comparison() -> None:
    from src.models.config import BENCHMARK_STORE, get_cutoff_date
    from src.models.train_arima import load_store_series, split_series, train_arima
    from src.models.train_prophet import REGRESSORS, load_store_prophet_df, split_df, train_prophet
    from src.models.train_random_forest import load_global_split as rf_split
    from src.models.train_random_forest import save_model, train_random_forest
    from src.models.train_xgboost import load_global_split as xgb_split
    from src.models.train_xgboost import train_xgboost

    cutoff = get_cutoff_date()

    print(f"=== Training global Random Forest and XGBoost (cutoff={cutoff.date()}) ===")
    X_train, y_train, X_test, y_test, meta_test = rf_split()
    rf_model = train_random_forest(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    save_model(rf_model, "random_forest_global")

    Xg_train, yg_train, Xg_test, yg_test, metag_test = xgb_split()
    xgb_model = train_xgboost(Xg_train, yg_train)
    xgb_preds = xgb_model.predict(Xg_test)
    save_model(xgb_model, "xgboost_global")

    global_results = {
        "Random Forest": score(y_test, rf_preds),
        "XGBoost": score(yg_test, xgb_preds),
    }
    global_df = save_comparison(global_results, "model_comparison_global")
    print("\n--- Global (all 1,115 nodes) model comparison ---")
    print(global_df.to_string())

    print(f"\n=== Single-store benchmark (Store {BENCHMARK_STORE}) ===")
    series = load_store_series(BENCHMARK_STORE)
    train_s, test_s = split_series(series, cutoff)
    arima_preds = train_arima(train_s, len(test_s))
    arima_preds.index = test_s.index
    arima_metrics = score(test_s.values, arima_preds.values)

    prophet_df = load_store_prophet_df(BENCHMARK_STORE)
    train_p, test_p = split_df(prophet_df, cutoff)
    prophet_model = train_prophet(train_p)
    forecast = prophet_model.predict(test_p[["ds"] + REGRESSORS])
    prophet_metrics = score(test_p["y"].values, forecast["yhat"].values)

    store_mask = meta_test["Store"] == BENCHMARK_STORE
    rf_store_metrics = score(y_test[store_mask], rf_preds[store_mask.values])

    store_mask_g = metag_test["Store"] == BENCHMARK_STORE
    xgb_store_metrics = score(yg_test[store_mask_g], xgb_preds[store_mask_g.values])

    single_store_results = {
        "ARIMA": arima_metrics,
        "Prophet": prophet_metrics,
        "Random Forest": rf_store_metrics,
        "XGBoost": xgb_store_metrics,
    }
    single_df = save_comparison(single_store_results, "model_comparison_single_store")
    print(f"\n--- Single-store (Store {BENCHMARK_STORE}) model comparison ---")
    print(single_df.to_string())

    best_name = min(global_results, key=lambda k: global_results[k]["MAPE"])
    best_file = "random_forest_global" if best_name == "Random Forest" else "xgboost_global"
    shutil.copy(MODELS_DIR / f"{best_file}.joblib", MODELS_DIR / "best_model.joblib")

    selection = {
        "selected_model": best_name,
        "reason": (
            f"{best_name} achieved the lowest global MAPE "
            f"({global_results[best_name]['MAPE']:.2f}%) across all 1,115 distribution nodes "
            "on the holdout window, and scales to a single production model serving the "
            "full network -- unlike ARIMA/Prophet, which require one model per node."
        ),
        "global_metrics": global_results,
        "single_store_benchmark_store": BENCHMARK_STORE,
        "single_store_metrics": single_store_results,
        "test_holdout_start": str(cutoff.date()),
    }
    (REPORTS_DIR / "best_model_selection.json").write_text(json.dumps(selection, indent=2))
    print(f"\n=== Best model for production: {best_name} ===")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    run_full_comparison()
