"""
SHAP-based explainability for the production XGBoost demand model.

Provides:
  - Global feature importance (SHAP summary/beeswarm plot)
  - SHAP dependence plots for the top 3 global demand drivers
  - Local waterfall explanations for a single store/date prediction
  - Auto-generated business narrative text translating SHAP contributions
    into procurement-planning language (e.g. "Promotions increased
    predicted demand by 34% for Store Type A in Q4")
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
FEATURES_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.parquet"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DROP_COLS = ["Date", "Sales"]

FEATURE_LABELS = {
    "Promo": "Promotional activity",
    "IsHoliday": "Public/bank holiday",
    "SchoolHoliday": "School holiday",
    "DayOfWeek": "Day of week",
    "Year": "Year",
    "Month": "Month",
    "Quarter": "Quarter",
    "WeekOfYear": "Week of year",
    "IsWeekend": "Weekend timing",
    "IsMonthStart": "Start-of-month timing",
    "IsMonthEnd": "End-of-month timing",
    "DaysToHoliday": "Proximity to upcoming holiday",
    "DaysSinceHoliday": "Time since last holiday",
    "SalesLag7": "Demand 7 days prior",
    "SalesLag14": "Demand 14 days prior",
    "SalesLag30": "Demand 30 days prior",
    "RollingMean7": "Recent 7-day demand trend",
    "RollingStd7": "7-day demand volatility",
    "RollingMean30": "Recent 30-day demand trend",
    "RollingStd30": "30-day demand volatility",
    "PromoWeekend": "Weekend promotion interaction",
    "PromoDayOfWeek": "Promotion timing within week",
    "PromoSchoolHoliday": "Promotion during school holiday",
    "CompetitionDistance": "Distance to nearest competitor",
    "MonthsSinceCompetition": "Competitor tenure nearby",
    "HasCompetitionInfo": "Competitor data availability",
    "Promo2": "Recurring promotion enrollment",
    "IsPromo2Month": "Active recurring-promotion month",
    "StoreAvgSalesExpanding": "Store's historical average demand",
    "StoreType_a": "Store Type A format",
    "StoreType_b": "Store Type B format",
    "StoreType_c": "Store Type C format",
    "StoreType_d": "Store Type D format",
    "Assortment_a": "Basic assortment breadth",
    "Assortment_b": "Extra assortment breadth",
    "Assortment_c": "Extended assortment breadth",
    "Store": "Distribution node identity",
}


def load_model_and_data():
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    df = pd.read_parquet(FEATURES_PATH).dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    return model, df, feature_cols


def compute_shap_values(model, X: pd.DataFrame, sample_size: int = 5000, random_state: int = 42):
    X_sample = X.sample(sample_size, random_state=random_state) if len(X) > sample_size else X
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return explainer, shap_values, X_sample


def plot_global_summary(shap_values, X_sample) -> Path:
    plt.close("all")
    shap.summary_plot(shap_values, X_sample, show=False)
    path = FIGURES_DIR / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return path


def top_features_by_importance(shap_values, feature_names, top_n: int = 3) -> list:
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return [feature_names[i] for i in order[:top_n]]


def plot_dependence_plots(shap_values, X_sample, feature_names, top_n: int = 3):
    top_feats = top_features_by_importance(shap_values, feature_names, top_n)
    paths = []
    for feat in top_feats:
        plt.close("all")
        shap.dependence_plot(feat, shap_values.values, X_sample, show=False)
        path = FIGURES_DIR / f"shap_dependence_{feat}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close("all")
        paths.append(path)
    return paths, top_feats


def explain_single_prediction(explainer, df: pd.DataFrame, feature_cols: list, store: int, date: str):
    row = df[(df["Store"] == store) & (df["Date"] == pd.Timestamp(date))]
    if row.empty:
        raise ValueError(f"No data for Store {store} on {date}")
    X_row = row[feature_cols]
    return explainer(X_row), X_row, row


def plot_waterfall(shap_row, index: int = 0, filename: str = "shap_waterfall_example.png") -> Path:
    plt.close("all")
    shap.plots.waterfall(shap_row[index], show=False)
    fig = plt.gcf()
    fig.set_size_inches(11, fig.get_size_inches()[1] + 1)
    path = FIGURES_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return path


def generate_local_narrative(shap_row, feature_cols: list, top_n: int = 3) -> str:
    values = shap_row.values[0]
    base_value = shap_row.base_values[0]
    predicted = base_value + values.sum()
    contributions = sorted(zip(feature_cols, values), key=lambda x: abs(x[1]), reverse=True)[:top_n]

    lines = [f"Predicted demand (revenue-equivalent): {predicted:,.0f} (baseline expectation: {base_value:,.0f})."]
    for feat, val in contributions:
        label = FEATURE_LABELS.get(feat, feat)
        direction = "increased" if val > 0 else "decreased"
        pct = abs(val) / base_value * 100 if base_value else 0
        lines.append(f"- {label} {direction} predicted demand by {pct:.1f}% ({val:+,.0f}).")
    return "\n".join(lines)


def generate_segment_narrative(
    explainer,
    df: pd.DataFrame,
    feature_cols: list,
    feature: str,
    segment_col: str,
    segment_value,
    segment_label: str,
    quarter: int | None = None,
    sample_size: int = 3000,
    random_state: int = 42,
) -> str:
    """Aggregate business narrative, e.g. 'Promotions increased predicted demand
    by 34% for Store Type A in Q4.'"""
    mask = df[segment_col] == segment_value
    if quarter is not None:
        mask &= df["Quarter"] == quarter
    subset = df[mask]
    if subset.empty:
        return f"No data available for {segment_label}{f' in Q{quarter}' if quarter else ''}."
    if len(subset) > sample_size:
        subset = subset.sample(sample_size, random_state=random_state)

    X_subset = subset[feature_cols]
    shap_vals = explainer(X_subset)
    feat_idx = feature_cols.index(feature)
    avg_contribution = shap_vals.values[:, feat_idx].mean()
    avg_base = shap_vals.base_values.mean()
    pct = avg_contribution / avg_base * 100 if avg_base else 0
    direction = "increased" if avg_contribution > 0 else "decreased"
    label = FEATURE_LABELS.get(feature, feature)
    quarter_label = f" in Q{quarter}" if quarter else ""
    return f"{label} {direction} predicted demand by {abs(pct):.1f}% for {segment_label}{quarter_label}."


def save_global_importance(shap_values, feature_cols: list) -> Path:
    """Persist mean(|SHAP|) per feature to JSON so the dashboard can render
    the global importance chart without holding a multi-thousand-row SHAP
    sample in memory at runtime (keeps the deployed app's memory footprint
    small on Streamlit Cloud's free tier)."""
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    importance = {
        FEATURE_LABELS.get(f, f): float(v)
        for f, v in sorted(zip(feature_cols, mean_abs), key=lambda x: x[1])
    }
    path = REPORTS_DIR / "shap_feature_importance.json"
    path.write_text(json.dumps(importance, indent=2))
    return path


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model, df, feature_cols = load_model_and_data()
    X = df[feature_cols]

    explainer, shap_values, X_sample = compute_shap_values(model, X)
    summary_path = plot_global_summary(shap_values, X_sample)
    print(f"Saved global summary plot to {summary_path}")

    importance_path = save_global_importance(shap_values, feature_cols)
    print(f"Saved global feature importance to {importance_path}")

    dep_paths, top_feats = plot_dependence_plots(shap_values, X_sample, feature_cols)
    print(f"Top 3 global drivers: {top_feats}")
    for p in dep_paths:
        print(f"Saved dependence plot to {p}")

    example_row = df.sort_values("Date").iloc[-1]
    shap_row, X_row, row_meta = explain_single_prediction(
        explainer, df, feature_cols, int(example_row["Store"]), str(example_row["Date"].date())
    )
    wf_path = plot_waterfall(shap_row)
    print(f"Saved waterfall plot to {wf_path}")
    narrative = generate_local_narrative(shap_row, feature_cols)
    print("\nLocal narrative example:\n" + narrative)

    segment_text = generate_segment_narrative(
        explainer, df, feature_cols, "Promo", "StoreType_a", 1, "Store Type A", quarter=4
    )
    print("\nSegment narrative example:\n" + segment_text)
