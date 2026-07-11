"""SHAP Explainability -- global feature importance, store-level 'why', auto-generated narratives."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

from dashboard.utils.helpers import load_features, load_model, store_list
from src.explainability.shap_explainer import (
    FEATURE_LABELS,
    compute_shap_values,
    explain_single_prediction,
    generate_local_narrative,
    generate_segment_narrative,
)

st.set_page_config(page_title="SHAP Explainability", page_icon="📦", layout="wide")
st.title("SHAP Explainability")
st.caption("Built with shap.TreeExplainer on the production XGBoost model -- fully functional against live predictions.")

df = load_features().dropna().reset_index(drop=True)
model = load_model()
DROP_COLS = ["Date", "Sales"]
feature_cols = [c for c in df.columns if c not in DROP_COLS]


@st.cache_resource
def cached_explainer_and_sample(_model, _df, sample_size: int = 3000):
    X = _df[feature_cols]
    explainer, shap_values, X_sample = compute_shap_values(_model, X, sample_size=sample_size)
    return explainer, shap_values, X_sample


explainer, shap_values, X_sample = cached_explainer_and_sample(model, df)

st.subheader("Global Feature Importance")
mean_abs = np.abs(shap_values.values).mean(axis=0)
importance_df = pd.DataFrame(
    {
        "Feature": [FEATURE_LABELS.get(f, f) for f in feature_cols],
        "Mean |SHAP value|": mean_abs,
    }
).sort_values("Mean |SHAP value|", ascending=True).tail(15)
fig_imp = px.bar(
    importance_df,
    x="Mean |SHAP value|",
    y="Feature",
    orientation="h",
    title="Top 15 Demand Drivers (Network-Wide)",
)
st.plotly_chart(fig_imp, use_container_width=True)

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
shap.plots.waterfall(shap_row[0], show=False)
fig = plt.gcf()
fig.set_size_inches(10, fig.get_size_inches()[1] + 1)
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
store_type_options = {"Store Type A": "StoreType_a", "Store Type B": "StoreType_b", "Store Type C": "StoreType_c", "Store Type D": "StoreType_d"}
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
    )
    st.success(segment_text)
