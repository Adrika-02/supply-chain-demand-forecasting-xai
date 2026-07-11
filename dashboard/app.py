"""Streamlit multipage app entry point -- landing page and navigation."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from dashboard.utils.helpers import fmt_currency, fmt_pct, inject_theme_css, load_features, load_json_report

st.set_page_config(
    page_title="FMCG Demand Planning Cockpit",
    page_icon="📦",
    layout="wide",
)
inject_theme_css()

# -- Animated hero: rotating network hub, radiating "distribution nodes", glow title --
# Pure inline SVG + CSS (no chart/image library) since this is a decorative network
# diagram, not data -- keeping it out of Plotly avoids paying for a chart engine to draw
# six circles and some lines.
HERO_HTML = """
<style>
.hero-wrap { position: relative; padding: 8px 0 4px; text-align: center; overflow: hidden; }
.hero-eyebrow {
    display: inline-block; font-size: 12px; font-weight: 700; letter-spacing: 0.14em;
    color: #5ed29c; text-transform: uppercase; margin-bottom: 10px;
    animation: fade-in-up 0.6s ease both;
}
.hero-title {
    font-size: clamp(30px, 4.5vw, 52px); font-weight: 800; line-height: 1.08; margin: 0 0 10px;
    background: linear-gradient(135deg, #f8fafc 20%, #67e8f9 55%, #5ed29c 90%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    text-shadow: 0 0 40px rgba(34,211,238,0.18);
    animation: fade-in-up 0.7s ease both; animation-delay: 0.08s;
}
.hero-sub {
    max-width: 720px; margin: 0 auto; color: #94a3b8; font-size: 15px; line-height: 1.6;
    animation: fade-in-up 0.7s ease both; animation-delay: 0.16s;
}

.hub-orbit { position: relative; width: 340px; height: 340px; margin: 22px auto 6px; }
.hub-ring {
    position: absolute; inset: 0; border-radius: 50%;
    border: 1px dashed rgba(34,211,238,0.28);
    animation: spin 26s linear infinite;
}
.hub-ring.inner { inset: 38px; border-color: rgba(94,210,156,0.28); animation-duration: 18s; animation-direction: reverse; }
.hub-core {
    position: absolute; top: 50%; left: 50%; width: 78px; height: 78px; margin: -39px 0 0 -39px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 30%, rgba(255,255,255,0.9), rgba(34,211,238,0.55) 45%, rgba(10,14,20,0.9) 75%);
    box-shadow: 0 0 40px 6px rgba(34,211,238,0.45), inset 0 0 18px rgba(255,255,255,0.35);
    animation: pulse-core 3.2s ease-in-out infinite;
    display: flex; align-items: center; justify-content: center; font-size: 28px;
}
@keyframes pulse-core {
    0%, 100% { transform: scale(1); box-shadow: 0 0 40px 6px rgba(34,211,238,0.45), inset 0 0 18px rgba(255,255,255,0.35); }
    50%      { transform: scale(1.06); box-shadow: 0 0 56px 10px rgba(34,211,238,0.6), inset 0 0 22px rgba(255,255,255,0.45); }
}
.hub-node {
    position: absolute; width: 14px; height: 14px; border-radius: 50%;
    background: #67e8f9; box-shadow: 0 0 14px 3px rgba(103,232,249,0.7);
    animation: node-pulse 2.4s ease-in-out infinite;
}
.hub-node.green { background: #5ed29c; box-shadow: 0 0 14px 3px rgba(94,210,156,0.7); }
@keyframes node-pulse {
    0%, 100% { transform: scale(0.85); opacity: 0.75; }
    50%      { transform: scale(1.15); opacity: 1; }
}
.hub-node.n1 { top: 4%;  left: 50%;  margin-left: -7px; animation-delay: 0s; }
.hub-node.n2 { top: 26%; left: 92%;  animation-delay: 0.3s; }
.hub-node.n3 { top: 74%; left: 92%;  animation-delay: 0.6s; }
.hub-node.n4 { top: 96%; left: 50%;  margin-left: -7px; animation-delay: 0.9s; }
.hub-node.n5 { top: 74%; left: 4%;   animation-delay: 1.2s; }
.hub-node.n6 { top: 26%; left: 4%;   animation-delay: 1.5s; }

.hub-spokes { position: absolute; inset: 0; }
.hub-spokes svg { width: 100%; height: 100%; }
.hub-spokes line { stroke: url(#spokeGrad); stroke-width: 1; opacity: 0.5; }

@keyframes fade-in-up { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
</style>

<div class="hero-wrap">
  <div class="hero-eyebrow">AI-Powered Supply Chain Intelligence</div>
  <div class="hero-title">FMCG Demand Planning Cockpit</div>
  <p class="hero-sub">Explainable AI-driven demand forecasting for procurement planning across a
  multi-category distribution network -- every forecast comes with a real, SHAP-derived reason why.</p>

  <div class="hub-orbit">
    <div class="hub-spokes">
      <svg viewBox="0 0 340 340">
        <defs>
          <linearGradient id="spokeGrad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="#22d3ee"/>
            <stop offset="100%" stop-color="#5ed29c"/>
          </linearGradient>
        </defs>
        <line x1="170" y1="170" x2="170" y2="14"/>
        <line x1="170" y1="170" x2="308" y2="92"/>
        <line x1="170" y1="170" x2="308" y2="252"/>
        <line x1="170" y1="170" x2="170" y2="326"/>
        <line x1="170" y1="170" x2="32"  y2="252"/>
        <line x1="170" y1="170" x2="32"  y2="92"/>
      </svg>
    </div>
    <div class="hub-ring"></div>
    <div class="hub-ring inner"></div>
    <div class="hub-node n1"></div>
    <div class="hub-node green n2"></div>
    <div class="hub-node n3"></div>
    <div class="hub-node green n4"></div>
    <div class="hub-node n5"></div>
    <div class="hub-node green n6"></div>
    <div class="hub-core">📦</div>
  </div>
</div>
"""
st.markdown(HERO_HTML, unsafe_allow_html=True)

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

- **Executive Overview** -- network-wide KPIs, demand trends, distribution-segment heatmap, live weather signal
- **Demand Forecast** -- store-level forecast vs. actual, 30/60/90-day forward forecasts
- **SHAP Explainability** -- global demand drivers, store-level "why", auto-generated narratives
- **SQL Insights** -- pre-built SQL queries against the demand warehouse
- **Procurement Recommendations** -- reorder suggestions, understock risk flags, downloadable report
- **Product Level Forecasting** -- genuine SKU x store forecasting, seasonality testing, and reorder sizing on a second dataset (Kaggle Store Item Demand Forecasting Challenge)

Use the sidebar to move between pages.
"""
)
