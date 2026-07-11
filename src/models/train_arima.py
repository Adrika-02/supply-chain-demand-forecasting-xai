"""
ARIMA/SARIMAX baseline (statsmodels) for one representative distribution node.

Classical univariate models don't scale to fitting one model per store, so
this baseline is fit on BENCHMARK_STORE's own daily series and evaluated on
the same holdout window used for every other model (see config.py). A
seasonal order of period=7 captures the strong weekly cycle seen in EDA.
"""
import warnings

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models.config import BENCHMARK_STORE, get_cutoff_date
from src.utils.db_utils import run_query


def load_store_series(store_id: int = BENCHMARK_STORE) -> pd.Series:
    df = run_query(
        "SELECT Date, Sales FROM sales_enriched WHERE Store = ? AND Open = 1 AND Sales > 0 ORDER BY Date",
        params=(store_id,),
    )
    df["Date"] = pd.to_datetime(df["Date"])
    series = df.set_index("Date")["Sales"].asfreq("D").interpolate()
    return series


def split_series(series: pd.Series, cutoff: pd.Timestamp):
    return series[series.index < cutoff], series[series.index >= cutoff]


def train_arima(train: pd.Series, steps: int) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
    return fitted.get_forecast(steps=steps).predicted_mean


if __name__ == "__main__":
    from src.models.evaluate import score

    cutoff = get_cutoff_date()
    series = load_store_series()
    train, test = split_series(series, cutoff)
    preds = train_arima(train, len(test))
    preds.index = test.index
    print(f"Store {BENCHMARK_STORE} ARIMA (n_test={len(test)}):", score(test.values, preds.values))
