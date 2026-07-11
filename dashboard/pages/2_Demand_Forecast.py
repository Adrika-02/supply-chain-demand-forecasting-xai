"""Demand Forecast -- store-level forecast vs. actual, forward forecasts with what-if promo toggle."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.helpers import feature_columns, inject_theme_css, load_features, load_model, store_list, style_fig
from src.models.config import get_cutoff_date
from src.models.forecast import compute_residual_quantiles, forecast_store

st.set_page_config(page_title="Demand Forecast", page_icon="📦", layout="wide")
inject_theme_css()
st.title("Demand Forecast")

df = load_features()
model = load_model()
stores = store_list(df)


@st.cache_data
def cached_residual_band(_model, _df):
    return compute_residual_quantiles(model=_model, df=_df)


@st.cache_data
def cached_forward_forecast(_model, _df, store_id: int, horizon: int, promo_override):
    return forecast_store(store_id, horizon, promo_override=promo_override, df=_df, model=_model)


@st.cache_data
def cached_history_predictions(_model, _df, store_id: int):
    store_df = _df[_df["Store"] == store_id].sort_values("Date").dropna()
    feat_cols = feature_columns(store_df)
    preds = _model.predict(store_df[feat_cols])
    out = store_df[["Date", "Sales"]].copy()
    out["Predicted"] = preds
    return out


col_a, col_b = st.columns([1, 2])
with col_a:
    selected_store = st.selectbox("Distribution node (Store)", stores, index=stores.index(262) if 262 in stores else 0)
with col_b:
    store_hist = df[df["Store"] == selected_store]
    min_date, max_date = store_hist["Date"].min(), store_hist["Date"].max()
    date_range = st.slider(
        "Date range (historical view)",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(max(min_date, max_date - pd.Timedelta(days=180)).to_pydatetime(), max_date.to_pydatetime()),
    )

st.divider()
st.subheader(f"Store {selected_store}: Forecast vs. Actual")

cutoff = get_cutoff_date()
hist_preds = cached_history_predictions(model, df, selected_store)
mask = (hist_preds["Date"] >= pd.Timestamp(date_range[0])) & (hist_preds["Date"] <= pd.Timestamp(date_range[1]))
view = hist_preds[mask]

lo_q, hi_q = cached_residual_band(model, df)

fig = go.Figure()
fig.add_trace(go.Scatter(x=view["Date"], y=view["Sales"], name="Actual", mode="lines", line=dict(color="#94a3b8")))
fig.add_trace(go.Scatter(x=view["Date"], y=view["Predicted"], name="Model Prediction", mode="lines", line=dict(color="#22d3ee")))
if cutoff >= pd.Timestamp(date_range[0]):
    fig.add_vline(x=cutoff, line_dash="dash", line_color="#64748b", annotation_text="holdout starts")
fig.update_layout(hovermode="x unified", yaxis_title="Demand (revenue-equivalent)", legend=dict(orientation="h"))
st.plotly_chart(style_fig(fig), use_container_width=True)
st.caption(
    "Left of the dashed line is in-sample (model has seen this data during training); right of it is the "
    "holdout window used for the accuracy metrics reported elsewhere in this project."
)

st.divider()
st.subheader("Forward Forecast")

col1, col2 = st.columns([1, 1])
with col1:
    horizon = st.radio("Forecast horizon", [30, 60, 90], horizontal=True, index=0)
with col2:
    promo_choice = st.radio(
        "Promotion assumption (what-if planning)",
        ["Historical pattern", "Force promotions ON", "Force promotions OFF"],
        horizontal=True,
    )
promo_override = {"Historical pattern": None, "Force promotions ON": 1, "Force promotions OFF": 0}[promo_choice]

forecast_df = cached_forward_forecast(model, df, selected_store, horizon, promo_override)
forecast_df["Lower"] = (forecast_df["PredictedSales"] + lo_q).clip(lower=0)
forecast_df["Upper"] = forecast_df["PredictedSales"] + hi_q

recent_actual = hist_preds.tail(30)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=recent_actual["Date"], y=recent_actual["Sales"], name="Recent Actual", line=dict(color="#94a3b8")))
fig2.add_trace(
    go.Scatter(
        x=pd.concat([forecast_df["Date"], forecast_df["Date"][::-1]]),
        y=pd.concat([forecast_df["Upper"], forecast_df["Lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(34,211,238,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Empirical 10-90% interval",
        showlegend=True,
    )
)
fig2.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["PredictedSales"], name=f"{horizon}-Day Forecast", line=dict(color="#f59e0b")))
fig2.update_layout(hovermode="x unified", yaxis_title="Demand (revenue-equivalent)", legend=dict(orientation="h"))
st.plotly_chart(style_fig(fig2), use_container_width=True)
st.caption(
    "Forecast is generated recursively (each day's prediction feeds the next day's lag/rolling features), "
    "the standard approach for multi-step tree-based forecasting. The confidence band is an empirical "
    "10th-90th percentile residual interval from the holdout window, not a per-store conditional variance model."
)

st.dataframe(
    forecast_df[["Date", "PredictedSales", "Lower", "Upper"]].rename(
        columns={"PredictedSales": "Forecast", "Lower": "Lower (10%)", "Upper": "Upper (90%)"}
    ),
    use_container_width=True,
    hide_index=True,
)
