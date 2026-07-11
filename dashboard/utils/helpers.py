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
    return pd.read_parquet(FEATURES_PATH)


@st.cache_data
def load_sales_enriched() -> pd.DataFrame:
    from src.utils.db_utils import run_query

    df = run_query("SELECT * FROM sales_enriched")
    df["Date"] = pd.to_datetime(df["Date"])
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
