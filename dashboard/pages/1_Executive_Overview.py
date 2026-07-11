"""Executive Overview -- network-wide KPIs, demand trend, distribution-segment heatmap."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import plotly.express as px
import streamlit as st

from dashboard.utils.helpers import (
    fmt_currency,
    fmt_pct,
    load_features,
    load_json_report,
    store_type_of,
)
from src.utils.live_data import fetch_live_weather

st.set_page_config(page_title="Executive Overview", page_icon="📦", layout="wide")
st.title("Executive Overview")

df = load_features()
impact = load_json_report("business_impact_summary.json")
best_model = load_json_report("best_model_selection.json")

df["StoreTypeLabel"] = df.apply(store_type_of, axis=1)

st.subheader("Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_demand = df["Sales"].sum()
    st.metric("Total Historical Demand", fmt_currency(total_demand))
with col2:
    st.metric(
        "Production Model MAPE",
        fmt_pct(best_model.get("global_metrics", {}).get("XGBoost", {}).get("MAPE", 0)),
    )
with col3:
    top_store = df.groupby("Store")["Sales"].sum().idxmax()
    st.metric("Top-Demand Store", f"Store {top_store}")
with col4:
    st.metric(
        "Network Revenue Run-Rate (current)",
        fmt_currency(impact.get("network_scale", {}).get("annual_revenue_run_rate_current_network", 0)),
    )

st.divider()

st.subheader("Demand Trend")
daily = df.groupby("Date", as_index=False)["Sales"].sum()
fig_trend = px.line(
    daily,
    x="Date",
    y="Sales",
    title="Total Network Demand Over Time",
    labels={"Sales": "Total Daily Demand (revenue-equivalent)"},
)
fig_trend.update_layout(hovermode="x unified")
st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Demand Heatmap: Store Format x Month")
    st.caption(
        "Rossmann does not publish store geography, so StoreType (store format) is used as "
        "a distribution-segment proxy in place of literal geographic region."
    )
    heat = (
        df.groupby(["StoreTypeLabel", "Month"], as_index=False)["Sales"]
        .mean()
        .pivot(index="StoreTypeLabel", columns="Month", values="Sales")
    )
    fig_heat = px.imshow(
        heat,
        labels=dict(x="Month", y="Store Type", color="Avg Daily Demand"),
        aspect="auto",
        color_continuous_scale="Teal",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with col_right:
    st.subheader("Top 10 Stores by Total Demand")
    top10 = (
        df.groupby("Store", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
        .head(10)
    )
    top10["Sales"] = top10["Sales"].round(0)
    top10["Store"] = top10["Store"].astype(str)
    fig_top = px.bar(top10, x="Store", y="Sales", labels={"Sales": "Total Demand (revenue-equivalent)"})
    fig_top.update_xaxes(type="category")
    st.plotly_chart(fig_top, use_container_width=True)

st.divider()

st.subheader("Live Market Signals")
st.caption(
    "Temperature and precipitation are established exogenous drivers of FMCG footfall and demand "
    "(e.g., beverage/ice-cream demand rises with heat; foot traffic drops in heavy rain). Rossmann's "
    "sales history stops in 2015 and carries no live feed, so this panel instead pulls genuinely live "
    "current weather (Open-Meteo API, refreshed every 15 minutes) for Rossmann's core German market -- "
    "the kind of real-time external signal a demand planner monitors alongside the model's forecast."
)


@st.cache_data(ttl=900)
def cached_live_weather():
    return fetch_live_weather()


if st.button("Refresh live data now"):
    cached_live_weather.clear()

try:
    weather_df = cached_live_weather()
    st.dataframe(weather_df, use_container_width=True, hide_index=True)
    st.caption("Observation times are reported live by Open-Meteo in the Europe/Berlin timezone.")
except Exception as exc:
    st.warning(f"Live weather feed temporarily unavailable: {exc}")
