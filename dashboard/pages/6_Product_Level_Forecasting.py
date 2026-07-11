"""Product-Level Forecasting -- genuine SKU x location forecasting on a second dataset."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.helpers import inject_theme_css, load_json_report, style_fig
from src.models.item_forecast import (
    batch_forecast_all_pairs,
    load_item_feature_frame,
    load_item_model,
    seasonality_test,
)
from src.models.reorder import compute_reorder_point

st.set_page_config(page_title="Product-Level Forecasting", page_icon="🏷️", layout="wide")
inject_theme_css()
st.title("Product-Level Forecasting (Store x Item)")
st.caption(
    "Pages 1-5 use the Rossmann distribution network (1,115 stores, no SKU field -- Store doubles as "
    "the demand-category proxy throughout). This page uses a second, genuinely SKU x location dataset "
    "-- Kaggle's Store Item Demand Forecasting Challenge (10 stores x 50 items, daily unit sales "
    "2013-2017) -- to answer the question a real product-level forecast requires: demand and reorder "
    "sizing for one specific item at one specific store, plus a real statistical test of whether that "
    "item is seasonal."
)

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@st.cache_data
def cached_item_features() -> pd.DataFrame:
    return load_item_feature_frame()


@st.cache_resource
def cached_item_model():
    return load_item_model()


@st.cache_data
def cached_item_metrics() -> dict:
    return load_json_report("item_model_metrics.json")


@st.cache_data
def cached_batch_forecast(_model, _df, horizon: int) -> pd.DataFrame:
    return batch_forecast_all_pairs(horizon, df=_df, model=_model)


df = cached_item_features()
model = cached_item_model()
metrics = cached_item_metrics()

st.subheader("Model Accuracy (Real Holdout Evaluation)")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("MAPE", f"{metrics.get('MAPE', 0):.2f}%")
with col2:
    st.metric("MAE", f"{metrics.get('MAE', 0):.2f} units/day")
with col3:
    st.metric("RMSE", f"{metrics.get('RMSE', 0):.2f} units/day")
with col4:
    st.metric("Store x Item Pairs", metrics.get("n_store_item_pairs", 0))
st.caption(
    f"Evaluated on the last {metrics.get('test_days', 0)} days, held out across all "
    f"{metrics.get('n_stores', 0)} stores x {metrics.get('n_items', 0)} items "
    f"({metrics.get('n_test_rows', 0):,} test rows, {metrics.get('n_train_rows', 0):,} training rows)."
)

st.divider()

st.subheader("Select a Product")
col_store, col_item, col_horizon = st.columns(3)
with col_store:
    store = st.selectbox("Store", sorted(df["Store"].unique()))
with col_item:
    item = st.selectbox("Item", sorted(df["Item"].unique()))
with col_horizon:
    horizon = st.radio("Forecast horizon (days)", [7, 14, 30], horizontal=True, index=1)

pair_df = df[(df["Store"] == store) & (df["Item"] == item)].sort_values("Date")

with st.spinner(f"Forecasting all {df.groupby(['Store', 'Item']).ngroups} store x item pairs..."):
    all_forecasts = cached_batch_forecast(model, df, horizon)
pair_forecast = all_forecasts[(all_forecasts["Store"] == store) & (all_forecasts["Item"] == item)]

st.subheader(f"Store {store} / Item {item} -- {horizon}-Day Forward Forecast")
fig_fc = go.Figure()
recent = pair_df.tail(90)
fig_fc.add_trace(go.Scatter(x=recent["Date"], y=recent["Sales"], name="Actual (last 90 days)", line=dict(color="#94a3b8")))
fig_fc.add_trace(go.Scatter(x=pair_forecast["Date"], y=pair_forecast["PredictedSales"], name="Forecast", line=dict(color="#f59e0b", dash="dash")))
fig_fc.update_layout(xaxis_title="Date", yaxis_title="Units Sold / Day", hovermode="x unified")
st.plotly_chart(style_fig(fig_fc), use_container_width=True)

st.divider()

st.subheader("Is This Product Seasonal?")
season = seasonality_test(df, store, item)
col_a, col_b = st.columns([1, 1])
with col_a:
    st.caption(
        "Kruskal-Wallis test across the 12 calendar months of this store-item pair's full sales "
        "history -- a real statistical test, not a heuristic guess."
    )
    if season["is_seasonal"]:
        st.success(
            f"**Seasonal** (Kruskal-Wallis p = {season['p_value']:.2e}, well below 0.05). Demand swings "
            f"**{season['seasonal_swing_pct']:.0f}%** between the peak month "
            f"({MONTH_NAMES[season['peak_month'] - 1]}) and trough month "
            f"({MONTH_NAMES[season['trough_month'] - 1]})."
        )
    else:
        st.info(
            f"**Not clearly seasonal** (Kruskal-Wallis p = {season['p_value']:.3f}, above 0.05) -- "
            "month-to-month demand differences aren't statistically significant for this item at this store."
        )
with col_b:
    monthly = pair_df.groupby(pair_df["Date"].dt.month)["Sales"].mean().reindex(range(1, 13))
    fig_month = px.bar(
        x=MONTH_NAMES, y=monthly.values,
        labels={"x": "Month", "y": "Avg Units Sold / Day"},
        title="Average Daily Demand by Month",
    )
    st.plotly_chart(style_fig(fig_month), use_container_width=True)

st.divider()

st.subheader("Reorder Point / Safety Stock for This Product")
st.caption(
    "Same formula as the Procurement Recommendations page (Reorder Point = Avg Daily Demand x Lead "
    "Time + Z(service level) x Demand Std Dev x sqrt(Lead Time)), computed here from this specific "
    "store-item pair's own recent daily rate -- and in real physical units, since this dataset's "
    "`sales` field is a literal unit count rather than Rossmann's revenue-equivalent figure."
)
col_lt, col_sl = st.columns(2)
with col_lt:
    lead_time_days = st.number_input("Lead time (days)", min_value=1, max_value=60, value=7, key="item_lt")
with col_sl:
    service_level_pct = st.select_slider(
        "Target service level", options=[90, 95, 97, 99, 99.5], value=95,
        format_func=lambda x: f"{x}%", key="item_sl",
    )

recent_mean = pair_df["Sales"].tail(28).mean()
recent_std = pair_df["Sales"].tail(28).std()
result = compute_reorder_point(
    avg_daily_demand=recent_mean,
    demand_std=recent_std if pd.notna(recent_std) else 0.0,
    lead_time_days=lead_time_days,
    service_level=service_level_pct / 100,
)

col_r1, col_r2, col_r3 = st.columns(3)
with col_r1:
    st.metric("Avg Daily Demand (last 28 days)", f"{recent_mean:.1f} units")
with col_r2:
    st.metric("Safety Stock", f"{result['safety_stock']:.0f} units")
with col_r3:
    st.metric("Reorder Point", f"{result['reorder_point']:.0f} units")
