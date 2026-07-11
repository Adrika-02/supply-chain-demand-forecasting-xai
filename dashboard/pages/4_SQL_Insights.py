"""SQL Insights -- run pre-written SQL queries against the demand warehouse, promo ROI view."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import plotly.express as px
import streamlit as st

from dashboard.utils.helpers import load_json_report
from src.data.sql_queries import QUERY_LIBRARY
from src.utils.db_utils import run_query

st.set_page_config(page_title="SQL Insights", page_icon="📦", layout="wide")
st.title("SQL Insights")
st.caption("Pre-written SQL queries against `db/supply_chain.db`, run live -- not hardcoded results.")

st.subheader("Query Library")
query_name = st.selectbox("Choose a query", list(QUERY_LIBRARY.keys()))
with st.expander("View SQL"):
    st.code(QUERY_LIBRARY[query_name], language="sql")

result_df = run_query(QUERY_LIBRARY[query_name])
st.dataframe(result_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Top Performing Stores")
top_stores = run_query(QUERY_LIBRARY["Top 20 highest-demand nodes"])
fig_top = px.bar(
    top_stores.head(10),
    x="Store",
    y="total_demand",
    color="StoreType",
    labels={"total_demand": "Total Demand (revenue-equivalent)"},
    title="Top 10 Stores by Total Demand",
)
fig_top.update_xaxes(type="category")
st.plotly_chart(fig_top, use_container_width=True)
st.dataframe(top_stores, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Promotional ROI Analysis")
promo_df = run_query(QUERY_LIBRARY["Demand: promotional vs non-promotional"])
eda_summary = load_json_report("eda_summary.json")
promo_stats = eda_summary.get("promo_impact", {})

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Avg Daily Demand -- Promo", f"{promo_stats.get('promo_mean', 0):,.0f}")
with col2:
    st.metric("Avg Daily Demand -- No Promo", f"{promo_stats.get('non_promo_mean', 0):,.0f}")
with col3:
    st.metric("Promotional Lift", f"+{promo_stats.get('pct_lift', 0):.1f}%")

st.caption(
    f"Welch's t-test: t={promo_stats.get('t_statistic', 0):.1f}, "
    f"p={promo_stats.get('p_value', 0):.2e}, "
    f"Cohen's d={promo_stats.get('cohens_d', 0):.2f} "
    f"({'statistically significant' if promo_stats.get('significant_at_0.05') else 'not significant'} at α=0.05)."
)

fig_promo = px.bar(
    promo_df,
    x="period_type",
    y="avg_daily_demand",
    color="period_type",
    labels={"avg_daily_demand": "Avg Daily Demand", "period_type": ""},
    title="Promotional vs. Non-Promotional Average Daily Demand",
)
st.plotly_chart(fig_promo, use_container_width=True)
