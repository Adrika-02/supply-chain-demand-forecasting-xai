"""Shared formatting, caching, and data-loading helpers for dashboard pages."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT_DIR / "data" / "processed" / "features.parquet"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
DROP_COLS = ["Date", "Sales"]


@st.cache_data
def load_features() -> pd.DataFrame:
    # Downcast float64/int64 -> float32/smallest-int: roughly halves the
    # ~850K-row table's memory footprint, which matters on Streamlit Cloud's
    # free tier (1GB RAM). XGBoost predicts fine on float32 input.
    df = pd.read_parquet(FEATURES_PATH)
    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    int_cols = df.select_dtypes(include="int64").columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODELS_DIR / "best_model.joblib")


@st.cache_data
def load_json_report(filename: str) -> dict:
    path = REPORTS_DIR / filename
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_data
def load_csv_report(filename: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS_DIR / filename, index_col=0)


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in DROP_COLS]


def store_list(df: pd.DataFrame) -> list:
    return sorted(df["Store"].unique().tolist())


def fmt_currency(value: float) -> str:
    if abs(value) >= 1e9:
        return f"€{value / 1e9:,.2f}B"
    if abs(value) >= 1e6:
        return f"€{value / 1e6:,.2f}M"
    if abs(value) >= 1e3:
        return f"€{value / 1e3:,.1f}K"
    return f"€{value:,.0f}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}%"


def store_type_of(row: pd.Series) -> str:
    for letter in ("a", "b", "c", "d"):
        if row.get(f"StoreType_{letter}", 0) == 1:
            return letter.upper()
    return "Unknown"


def assortment_of(row: pd.Series) -> str:
    for letter in ("a", "b", "c"):
        if row.get(f"Assortment_{letter}", 0) == 1:
            return letter.upper()
    return "Unknown"


# -- Dark "AI supply chain" visual theme -------------------------------------------------
# Injected as raw CSS (Streamlit has no first-class theming API for gradients, glow, or
# keyframe animation) rather than a Python plotting/styling library, since every element
# being styled here -- metric cards, sidebar nav, buttons, background -- is a Streamlit-
# generated DOM node identified by its `data-testid`, not something we render ourselves.
_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(ellipse 900px 500px at 15% -10%, rgba(34,211,238,0.16), transparent 60%),
        radial-gradient(ellipse 700px 500px at 90% 10%, rgba(94,210,156,0.12), transparent 60%),
        linear-gradient(180deg, #0a0e14 0%, #0a0e14 100%);
    background-attachment: fixed;
}

/* faint animated network grid across the page */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(148,163,184,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148,163,184,0.05) 1px, transparent 1px);
    background-size: 48px 48px;
    animation: grid-drift 40s linear infinite;
}
@keyframes grid-drift {
    from { background-position: 0 0, 0 0; }
    to   { background-position: 96px 48px, 48px 96px; }
}

h1, h2, h3 { color: #e2e8f0 !important; letter-spacing: -0.01em; }
h1 { text-shadow: 0 0 24px rgba(34,211,238,0.25); }

/* glass metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 14px;
    padding: 16px 18px;
    backdrop-filter: blur(6px);
    box-shadow: inset 0 1px 1px rgba(255,255,255,0.06), 0 8px 24px rgba(0,0,0,0.25);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    animation: fade-in-up 0.6s ease both;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: rgba(34,211,238,0.4);
    box-shadow: inset 0 1px 1px rgba(255,255,255,0.08), 0 12px 28px rgba(34,211,238,0.12);
}
[data-testid="stMetricValue"] { color: #67e8f9 !important; text-shadow: 0 0 18px rgba(34,211,238,0.35); }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }

@keyframes fade-in-up {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* buttons */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, #22d3ee, #5ed29c);
    color: #06111a;
    font-weight: 700;
    border: none;
    border-radius: 999px;
    padding: 0.55em 1.6em;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 4px 18px rgba(34,211,238,0.25);
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 26px rgba(34,211,238,0.4);
}

/* sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(17,24,35,0.98), rgba(10,14,20,0.98));
    border-right: 1px solid rgba(148,163,184,0.1);
}
[data-testid="stSidebarNav"] a { transition: color 0.2s ease, background 0.2s ease; border-radius: 8px; }
[data-testid="stSidebarNav"] a:hover { color: #22d3ee !important; background: rgba(34,211,238,0.08); }

/* dataframes / tables */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid rgba(148,163,184,0.12); }

hr { border-color: rgba(148,163,184,0.15) !important; }
</style>
"""


def inject_theme_css() -> None:
    """Injects the shared dark 'AI supply chain' theme (gradients, glow, glass cards,
    subtle animation) once per page. Call at the top of every page's script."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


DARK_PLOTLY_COLORWAY = ["#22d3ee", "#5ed29c", "#f59e0b", "#a78bfa", "#f472b6", "#60a5fa", "#fb923c"]


def style_fig(fig):
    """Applies the dark theme to a Plotly figure: transparent background (so the page's
    gradient shows through), muted gridlines, and a cyan/green colorway consistent with
    the rest of the UI. Call on every figure right before st.plotly_chart."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        colorway=DARK_PLOTLY_COLORWAY,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=48),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.2)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.2)")
    return fig
