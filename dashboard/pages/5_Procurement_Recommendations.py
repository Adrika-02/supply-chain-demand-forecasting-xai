"""Procurement Recommendations -- auto-generated reorder suggestions, understock risk flags, CSV export."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.utils.helpers import assortment_of, load_features, load_model, store_type_of
from src.models.forecast import batch_forecast_all_stores

st.set_page_config(page_title="Procurement Recommendations", page_icon="📦", layout="wide")
st.title("Procurement Recommendations")
st.caption(
    "Reorder signal derived from a 7-day-ahead demand forecast vs. each store's recent 7-day demand rate. "
    "Rossmann has no literal SKU inventory/stock-on-hand data, so this is a demand-signal-based allocation "
    "recommendation (framed in % adjustment terms), not a live stock-tracking system."
)

HORIZON = 7
RISK_THRESHOLD_PCT = 15.0

df = load_features()
model = load_model()


@st.cache_data
def cached_batch_forecast(_model, _df, horizon: int):
    return batch_forecast_all_stores(horizon, df=_df, model=_model)


with st.spinner(f"Scoring {df['Store'].nunique():,} distribution nodes ({HORIZON}-day forecast)..."):
    forecast_df = cached_batch_forecast(model, df, HORIZON)

forecast_totals = forecast_df.groupby("Store", as_index=False)["PredictedSales"].sum().rename(
    columns={"PredictedSales": "Forecasted7Day"}
)

last_rows = df.sort_values("Date").groupby("Store").tail(1).copy()
last_rows["Recent7DayRate"] = last_rows["RollingMean7"] * 7
last_rows["StoreTypeLabel"] = last_rows.apply(store_type_of, axis=1)
last_rows["AssortmentLabel"] = last_rows.apply(assortment_of, axis=1)

rec = last_rows[["Store", "StoreTypeLabel", "AssortmentLabel", "Recent7DayRate"]].merge(
    forecast_totals, on="Store"
)
rec["DeviationPct"] = (rec["Forecasted7Day"] - rec["Recent7DayRate"]) / rec["Recent7DayRate"] * 100


def classify(pct: float) -> str:
    if pct > RISK_THRESHOLD_PCT:
        return "Understock Risk"
    if pct < -RISK_THRESHOLD_PCT:
        return "Overstock Risk"
    return "On Track"


def recommend(pct: float) -> str:
    if pct > RISK_THRESHOLD_PCT:
        return f"Increase allocation ~{pct:.0f}% -- demand surge expected"
    if pct < -RISK_THRESHOLD_PCT:
        return f"Reduce allocation ~{abs(pct):.0f}% -- demand pullback expected"
    return "Maintain current allocation"


rec["RiskFlag"] = rec["DeviationPct"].apply(classify)
rec["Recommendation"] = rec["DeviationPct"].apply(recommend)
rec = rec.sort_values("DeviationPct", ascending=False)

st.subheader("Network Summary")
counts = rec["RiskFlag"].value_counts()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Understock Risk", int(counts.get("Understock Risk", 0)))
with col2:
    st.metric("Overstock Risk", int(counts.get("Overstock Risk", 0)))
with col3:
    st.metric("On Track", int(counts.get("On Track", 0)))

fig_dist = px.histogram(
    rec, x="DeviationPct", color="RiskFlag", nbins=50,
    labels={"DeviationPct": "Forecast Deviation from Recent Rate (%)"},
    title="Network-Wide Demand Deviation Distribution",
)
st.plotly_chart(fig_dist, use_container_width=True)

st.divider()
st.subheader("Store-Level Recommendations")

risk_filter = st.multiselect(
    "Filter by risk flag", ["Understock Risk", "Overstock Risk", "On Track"],
    default=["Understock Risk", "Overstock Risk"],
)
filtered = rec[rec["RiskFlag"].isin(risk_filter)] if risk_filter else rec

display_cols = {
    "Store": "Store",
    "StoreTypeLabel": "Store Type",
    "AssortmentLabel": "Assortment",
    "Recent7DayRate": "Recent 7-Day Demand",
    "Forecasted7Day": "Forecasted 7-Day Demand",
    "DeviationPct": "Deviation %",
    "RiskFlag": "Risk Flag",
    "Recommendation": "Recommendation",
}
display_df = filtered[list(display_cols.keys())].rename(columns=display_cols)
display_df["Recent 7-Day Demand"] = display_df["Recent 7-Day Demand"].round(0)
display_df["Forecasted 7-Day Demand"] = display_df["Forecasted 7-Day Demand"].round(0)
display_df["Deviation %"] = display_df["Deviation %"].round(1)

st.dataframe(display_df, use_container_width=True, hide_index=True)

csv = display_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Procurement Recommendations (CSV)",
    data=csv,
    file_name="procurement_recommendations.csv",
    mime="text/csv",
)
