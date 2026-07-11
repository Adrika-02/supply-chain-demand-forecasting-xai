"""
Business impact quantification: forecast accuracy lift over a naive
seasonal baseline, estimated inventory cost savings from reduced overstock,
and projected revenue-equivalent demand at a 500-store FMCG manufacturer
scale.

The Rossmann dataset has no inventory, cost, or margin fields, so dollar and
percentage figures that aren't directly measurable from it use clearly
labeled, published FMCG/retail supply-chain industry benchmarks rather than
fabricated precision. Every other figure here (accuracy lift, revenue
run-rate) is computed from actual model output and the real dataset.
"""
import json
from pathlib import Path

import pandas as pd

from src.models.evaluate import score
from src.models.train_xgboost import load_global_split

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
FEATURES_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.parquet"

CURRENT_STORE_COUNT = 1115
TARGET_STORE_COUNT = 500

# Documented industry-benchmark assumptions (NOT derived from Rossmann data,
# which has no inventory/cost fields):
INVENTORY_TO_SALES_RATIO = 0.10  # typical FMCG/retail inventory value as a share of annual revenue
OVERSTOCK_REDUCTION_LOW = 0.20  # published lower bound for AI-driven demand forecasting
OVERSTOCK_REDUCTION_HIGH = 0.30  # published upper bound for AI-driven demand forecasting


def naive_seasonal_baseline_metrics() -> dict:
    """7-day seasonal naive forecast (demand = same weekday last week, using
    the already-computed SalesLag7 feature), scored on the identical global
    holdout window as the production model -- the fairest naive baseline
    given the strong weekly seasonality confirmed in EDA."""
    _, _, X_test, y_test, _ = load_global_split()
    naive_preds = X_test["SalesLag7"]
    return score(y_test, naive_preds)


def annual_revenue_run_rate(store_count: int) -> float:
    """Full-year revenue-equivalent demand run-rate, extrapolated from the
    complete cleaned dataset's average daily per-store demand (not just the
    6-week holdout window, which is summer-seasonal and skews low)."""
    df = pd.read_parquet(FEATURES_PATH)
    avg_daily_demand_per_store = df["Sales"].mean()
    return float(avg_daily_demand_per_store * 365 * store_count)


def compute_business_impact() -> dict:
    best_model_selection = json.loads((REPORTS_DIR / "best_model_selection.json").read_text())
    model_mape = best_model_selection["global_metrics"]["XGBoost"]["MAPE"]

    naive_metrics = naive_seasonal_baseline_metrics()
    naive_mape = naive_metrics["MAPE"]
    accuracy_improvement_pct = (naive_mape - model_mape) / naive_mape * 100

    annual_revenue_current = annual_revenue_run_rate(CURRENT_STORE_COUNT)
    annual_revenue_target = annual_revenue_run_rate(TARGET_STORE_COUNT)

    inventory_value_target = annual_revenue_target * INVENTORY_TO_SALES_RATIO
    savings_low = inventory_value_target * OVERSTOCK_REDUCTION_LOW
    savings_high = inventory_value_target * OVERSTOCK_REDUCTION_HIGH
    savings_base = (savings_low + savings_high) / 2

    return {
        "forecast_accuracy": {
            "naive_seasonal_baseline_mae": naive_metrics["MAE"],
            "naive_seasonal_baseline_rmse": naive_metrics["RMSE"],
            "naive_seasonal_baseline_mape": naive_mape,
            "xgboost_model_mape": model_mape,
            "relative_mape_improvement_pct": accuracy_improvement_pct,
            "note": "Naive baseline = demand on the same weekday one week prior (SalesLag7), the fairest naive comparator given the strong weekly seasonality confirmed in EDA. Both scored on the identical holdout window (last 6 weeks, cutoff 2015-06-19).",
        },
        "network_scale": {
            "current_store_count": CURRENT_STORE_COUNT,
            "target_store_count": TARGET_STORE_COUNT,
            "annual_revenue_run_rate_current_network": annual_revenue_current,
            "annual_revenue_run_rate_target_network": annual_revenue_target,
        },
        "inventory_cost_savings": {
            "assumptions": {
                "inventory_to_sales_ratio": INVENTORY_TO_SALES_RATIO,
                "overstock_reduction_range": [OVERSTOCK_REDUCTION_LOW, OVERSTOCK_REDUCTION_HIGH],
                "note": (
                    "Inventory-to-sales ratio and overstock-reduction range are published "
                    "FMCG/retail supply-chain industry benchmarks, not derived from the "
                    "Rossmann dataset (which has no inventory/cost fields). The annual "
                    "revenue run-rate they're applied to IS derived from actual model/data output."
                ),
            },
            "estimated_inventory_value_at_target_scale": inventory_value_target,
            "estimated_annual_savings_low_20pct": savings_low,
            "estimated_annual_savings_high_30pct": savings_high,
            "estimated_annual_savings_base_25pct": savings_base,
        },
    }


if __name__ == "__main__":
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = compute_business_impact()
    (REPORTS_DIR / "business_impact_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
