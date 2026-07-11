"""Streamlit multipage app entry point -- landing page and navigation."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from dashboard.utils.helpers import fmt_currency, fmt_pct, load_features, load_json_report

st.set_page_config(
    page_title="FMCG Demand Planning Cockpit",
    page_icon="📦",
    layout="wide",
)

st.title("📦 FMCG Demand Planning Cockpit")
st.caption(
    "Explainable AI-driven demand forecasting for procurement planning across a "
    "multi-category distribution network."
)

st.markdown(
    """
This system forecasts daily demand for every distribution node in the network, explains
**why** the model predicts what it predicts using SHAP, and turns those forecasts into
concrete procurement actions -- reorder suggestions, understock risk flags, and
promotional ROI analysis.

**Dataset framing:** built on the Rossmann Store Sales dataset, reframed as a
multi-category FMCG manufacturer's distribution network (`Store` = retail/distribution
node, `StoreType`/`Assortment` = product-category proxies). See the README for the full
data and modeling methodology.
"""
)

st.divider()

impact = load_json_report("business_impact_summary.json")
best_model = load_json_report("best_model_selection.json")
df = load_features()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Production Model MAPE",
        fmt_pct(best_model.get("global_metrics", {}).get("XGBoost", {}).get("MAPE", 0)),
        help="Mean Absolute Percentage Error, XGBoost, all 1,115 distribution nodes, holdout window.",
    )
with col2:
    acc = impact.get("forecast_accuracy", {})
    st.metric(
        "Accuracy vs. Naive Baseline",
        f"-{acc.get('relative_mape_improvement_pct', 0):.1f}%",
        help="Relative MAPE reduction vs. a 7-day seasonal naive forecast.",
    )
with col3:
    savings = impact.get("inventory_cost_savings", {})
    st.metric(
        "Est. Annual Inventory Savings",
        fmt_currency(savings.get("estimated_annual_savings_base_25pct", 0)),
        help="Base-case estimate at a 500-store scale (20-30% overstock reduction industry benchmark).",
    )
with col4:
    st.metric("Distribution Nodes Tracked", f"{df['Store'].nunique():,}")

st.divider()

st.markdown(
    """
### Navigate

- **Executive Overview** -- network-wide KPIs, demand trends, distribution-segment heatmap
- **Demand Forecast** -- store-level forecast vs. actual, 30/60/90-day forward forecasts
- **SHAP Explainability** -- global demand drivers, store-level "why", auto-generated narratives
- **SQL Insights** -- pre-built SQL queries against the demand warehouse
- **Procurement Recommendations** -- reorder suggestions, understock risk flags, downloadable report

Use the sidebar to move between pages.
"""
)
