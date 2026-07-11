"""SHAP Explainability -- global feature importance, store-level 'why', auto-generated narratives."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

from dashboard.utils.helpers import STORE_TYPE_LABELS, inject_theme_css, load_features, load_json_report, load_model, store_list, style_fig
from src.explainability.shap_explainer import (
    FEATURE_LABELS,
    explain_single_prediction,
    generate_local_narrative,
    generate_segment_narrative,
)

st.set_page_config(page_title="SHAP Explainability", page_icon="📦", layout="wide")
inject_theme_css()
st.title("SHAP Explainability")
st.caption("Built with shap.TreeExplainer on the production XGBoost model -- fully functional against live predictions.")

df = load_features().dropna().reset_index(drop=True)
model = load_model()
DROP_COLS = ["Date", "Sales"]
feature_cols = [c for c in df.columns if c not in DROP_COLS]


@st.cache_resource
def cached_explainer(_model):
    # No background sample needed: TreeExplainer reads XGBoost's tree
    # structure directly, so this stays cheap regardless of network size --
    # unlike caching a multi-thousand-row SHAP sample in memory.
    return shap.TreeExplainer(_model)


explainer = cached_explainer(model)

st.subheader("Global Feature Importance")
importance = load_json_report("shap_feature_importance.json")
if importance:
    importance_df = pd.DataFrame(
        {"Feature": list(importance.keys()), "Mean |SHAP value|": list(importance.values())}
    ).tail(15)
    fig_imp = px.bar(
        importance_df,
        x="Mean |SHAP value|",
        y="Feature",
        orientation="h",
        title="Top 15 Demand Drivers (Network-Wide)",
    )
    st.plotly_chart(style_fig(fig_imp), use_container_width=True)
    st.caption(
        "Precomputed from a 5,000-row network sample (see reports/shap_feature_importance.json) "
        "rather than recomputed live, to keep the deployed app's memory footprint small."
    )
else:
    st.info("Run `python -m src.explainability.shap_explainer` to generate the global importance report.")

st.divider()
st.subheader("Store-Level Explanation")

stores = store_list(df)
col1, col2 = st.columns([1, 1])
with col1:
    selected_store = st.selectbox("Distribution node (Store)", stores, index=stores.index(262) if 262 in stores else 0)
with col2:
    store_dates = sorted(df.loc[df["Store"] == selected_store, "Date"].dt.date.unique(), reverse=True)
    selected_date = st.selectbox("Date", store_dates)

shap_row, X_row, row_meta = explain_single_prediction(explainer, df, feature_cols, selected_store, str(selected_date))

plt.close("all")
with plt.style.context("dark_background"):
    shap.plots.waterfall(shap_row[0], show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, fig.get_size_inches()[1] + 1)
    fig.patch.set_facecolor("#0e1520")
    for ax in fig.axes:
        ax.set_facecolor("#0e1520")
    st.pyplot(fig, use_container_width=True)
plt.close("all")

st.markdown("**Auto-generated business narrative:**")
narrative = generate_local_narrative(shap_row, feature_cols)
for line in narrative.split("\n"):
    st.markdown(f"> {line}")

st.divider()
st.subheader("Segment Narrative")
st.caption("Aggregate SHAP contribution of a chosen feature, averaged across a store-format x quarter segment.")

col3, col4, col5 = st.columns(3)
store_type_options = {
    STORE_TYPE_LABELS["A"]: "StoreType_a",
    STORE_TYPE_LABELS["B"]: "StoreType_b",
    STORE_TYPE_LABELS["C"]: "StoreType_c",
    STORE_TYPE_LABELS["D"]: "StoreType_d",
}
feature_options = {FEATURE_LABELS.get(f, f): f for f in feature_cols}
with col3:
    segment_label = st.selectbox("Store format segment", list(store_type_options.keys()))
with col4:
    quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=3)
with col5:
    feature_label = st.selectbox("Feature", list(feature_options.keys()), index=list(feature_options.values()).index("Promo"))

if st.button("Generate narrative"):
    segment_text = generate_segment_narrative(
        explainer,
        df,
        feature_cols,
        feature_options[feature_label],
        store_type_options[segment_label],
        1,
        segment_label,
        quarter=quarter,
        sample_size=500,
    )
    st.success(segment_text)
