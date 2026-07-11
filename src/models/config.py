"""
Shared modeling configuration.

TEST_DAYS defines the holdout forecast horizon (last 6 weeks of the
network's history, roughly matching a procurement re-forecast cycle).
BENCHMARK_STORE is the single distribution node used for the apples-to-apples
ARIMA vs Prophet vs Random Forest vs XGBoost comparison -- classical
univariate models don't scale to fitting one model per store, so all four
models are compared on one representative high-volume node, evaluated on the
identical date range. Random Forest and XGBoost are additionally evaluated
as global, all-store production models (see reports/model_comparison_global).
"""
import pandas as pd

from src.utils.db_utils import run_query

TEST_DAYS = 42
BENCHMARK_STORE = 262


def get_cutoff_date(test_days: int = TEST_DAYS) -> pd.Timestamp:
    max_date = run_query("SELECT MAX(Date) AS max_date FROM sales_enriched")["max_date"].iloc[0]
    return pd.Timestamp(max_date) - pd.Timedelta(days=test_days)
