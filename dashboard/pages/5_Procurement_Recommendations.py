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
from src.models.reorder import compute_reorder_point

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

rec = last_rows[
    ["Store", "StoreTypeLabel", "AssortmentLabel", "Recent7DayRate", "RollingMean7", "RollingStd7"]
].merge(forecast_totals, on="Store")
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
st.subheader("Reorder Point / Safety Stock Calculator")
st.caption(
    "Reorder Point = (Avg Daily Demand x Lead Time) + Safety Stock, where Safety Stock = "
    "Z(service level) x Demand Std Dev x sqrt(Lead Time). Demand and its volatility come from each "
    "store's own recent daily rate (RollingMean7 / RollingStd7) -- real, model-derived numbers. Lead "
    "time and service level are configurable assumptions below (Rossmann has no real supplier lead-time "
    "data). This is a *target stocking level* to cover lead-time demand at your chosen service level, "
    "not \"how many more units to order right now\" -- that would additionally require real current "
    "on-hand inventory, which doesn't exist in this dataset."
)

col_lt, col_sl, col_up = st.columns(3)
with col_lt:
    lead_time_days = st.number_input("Lead time (days)", min_value=1, max_value=60, value=7)
with col_sl:
    service_level_pct = st.select_slider(
        "Target service level", options=[90, 95, 97, 99, 99.5], value=95, format_func=lambda x: f"{x}%"
    )
with col_up:
    unit_price = st.number_input(
        "Optional: unit price (to convert to physical units)",
        min_value=0.0, value=0.0, step=0.5,
        help="Rossmann has no real per-SKU price. Leave at 0 to keep the reorder point in "
        "revenue-equivalent terms; enter a price to see it converted to physical units.",
    )

reorder = rec.apply(
    lambda row: compute_reorder_point(
        avg_daily_demand=row["RollingMean7"],
        demand_std=row["RollingStd7"] if pd.notna(row["RollingStd7"]) else 0.0,
        lead_time_days=lead_time_days,
        service_level=service_level_pct / 100,
        unit_price=unit_price if unit_price > 0 else None,
    ),
    axis=1,
)
rec["SafetyStock"] = reorder.apply(lambda r: r["safety_stock"])
rec["ReorderPoint"] = reorder.apply(lambda r: r["reorder_point"])
if unit_price > 0:
    rec["ReorderPointUnits"] = reorder.apply(lambda r: r["reorder_point_units"])

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
    "SafetyStock": f"Safety Stock ({lead_time_days}d lead, {service_level_pct}% SL)",
    "ReorderPoint": "Reorder Point",
}
if unit_price > 0:
    display_cols["ReorderPointUnits"] = "Reorder Point (units)"

display_df = filtered[list(display_cols.keys())].rename(columns=display_cols)
for col in display_df.columns:
    if display_df[col].dtype.kind in "fc":
        display_df[col] = display_df[col].round(1 if "%" in col else 0)

st.dataframe(display_df, use_container_width=True, hide_index=True)

csv = display_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Procurement Recommendations (CSV)",
    data=csv,
    file_name="procurement_recommendations.csv",
    mime="text/csv",
)
