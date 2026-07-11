"""
Facebook Prophet baseline for the same representative distribution node used
by the ARIMA baseline, with Promo / holiday / school-holiday as regressors so
it can capture the same promotional and calendar effects the ML models see.
"""
import pandas as pd
from prophet import Prophet

from src.models.config import BENCHMARK_STORE, get_cutoff_date
from src.utils.db_utils import run_query

REGRESSORS = ["Promo", "IsHoliday", "SchoolHoliday"]


def load_store_prophet_df(store_id: int = BENCHMARK_STORE) -> pd.DataFrame:
    df = run_query(
        """
        SELECT Date, Sales, Promo, StateHoliday, SchoolHoliday
        FROM sales_enriched
        WHERE Store = ? AND Open = 1 AND Sales > 0
        ORDER BY Date
        """,
        params=(store_id,),
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df["IsHoliday"] = (df["StateHoliday"] != "0").astype(int)
    return df.rename(columns={"Date": "ds", "Sales": "y"})[["ds", "y"] + REGRESSORS]


def split_df(df: pd.DataFrame, cutoff: pd.Timestamp):
    return df[df["ds"] < cutoff].copy(), df[df["ds"] >= cutoff].copy()


def train_prophet(train: pd.DataFrame) -> Prophet:
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    for regressor in REGRESSORS:
        model.add_regressor(regressor)
    model.fit(train)
    return model


if __name__ == "__main__":
    from src.models.evaluate import score

    cutoff = get_cutoff_date()
    df = load_store_prophet_df()
    train, test = split_df(df, cutoff)
    model = train_prophet(train)
    forecast = model.predict(test[["ds"] + REGRESSORS])
    preds = forecast["yhat"].values
    print(f"Store {BENCHMARK_STORE} Prophet (n_test={len(test)}):", score(test["y"].values, preds))
