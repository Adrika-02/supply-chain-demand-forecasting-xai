"""
Exploratory Data Analysis for the FMCG demand-forecasting pipeline.

Reads from the `sales_enriched` SQLite view and produces:
  - Sales distribution by store type (A-D)
  - Seasonal decomposition (trend / seasonality / residual) of aggregate demand
  - Promotion impact analysis with a Welch's t-test for statistical significance
  - Holiday effect quantification (t-test + effect size)
  - Correlation matrix of numeric demand-driving features

All figures are saved as PNGs to reports/figures/. Numeric findings (means,
p-values, effect sizes) are saved to reports/eda_summary.json so later steps
(business narrative, README) can quote real numbers instead of guesses.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

from src.utils.db_utils import run_query

FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
SUMMARY_PATH = Path(__file__).resolve().parents[1] / "reports" / "eda_summary.json"

sns.set_theme(style="whitegrid")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / pooled_std


def load_open_days() -> pd.DataFrame:
    df = run_query(
        """
        SELECT Store, Date, DayOfWeek, Sales, Customers, Promo, StateHoliday,
               SchoolHoliday, StoreType, Assortment, CompetitionDistance, Promo2
        FROM sales_enriched
        WHERE Open = 1
        """
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df["IsHoliday"] = (df["StateHoliday"] != "0").astype(int)
    return df


def plot_sales_distribution_by_store_type(df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 6))
    order = sorted(df["StoreType"].dropna().unique())
    sns.boxplot(data=df, x="StoreType", y="Sales", order=order, showfliers=False)
    plt.title("Daily Demand Distribution by Store Format (Type)")
    plt.xlabel("Store Type")
    plt.ylabel("Daily Sales (Demand)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sales_distribution_by_store_type.png", dpi=150)
    plt.close()


def plot_seasonal_decomposition(df: pd.DataFrame) -> dict:
    daily = df.groupby("Date")["Sales"].sum().asfreq("D").interpolate()
    decomposition = seasonal_decompose(daily, model="additive", period=7)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    decomposition.observed.plot(ax=axes[0], title="Observed Total Daily Demand")
    decomposition.trend.plot(ax=axes[1], title="Trend")
    decomposition.seasonal.plot(ax=axes[2], title="Weekly Seasonality")
    decomposition.resid.plot(ax=axes[3], title="Residual")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "seasonal_decomposition.png", dpi=150)
    plt.close()

    return {
        "trend_start": float(decomposition.trend.dropna().iloc[0]),
        "trend_end": float(decomposition.trend.dropna().iloc[-1]),
        "seasonal_amplitude": float(decomposition.seasonal.max() - decomposition.seasonal.min()),
        "residual_std": float(decomposition.resid.dropna().std()),
    }


def analyze_promo_impact(df: pd.DataFrame) -> dict:
    promo_sales = df.loc[df["Promo"] == 1, "Sales"].values
    non_promo_sales = df.loc[df["Promo"] == 0, "Sales"].values

    t_stat, p_value = stats.ttest_ind(promo_sales, non_promo_sales, equal_var=False)
    effect_size = cohens_d(promo_sales, non_promo_sales)
    pct_lift = (promo_sales.mean() - non_promo_sales.mean()) / non_promo_sales.mean() * 100

    labels = ["Non-Promotional", "Promotional"]
    plt.figure(figsize=(7, 6))
    sns.barplot(
        x=labels,
        y=[non_promo_sales.mean(), promo_sales.mean()],
        hue=labels,
        palette=["#94a3b8", "#0E7C7B"],
        legend=False,
    )
    plt.title(f"Promotion Impact on Daily Demand (+{pct_lift:.1f}%, p={p_value:.2e})")
    plt.ylabel("Mean Daily Sales")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "promo_impact.png", dpi=150)
    plt.close()

    return {
        "promo_mean": float(promo_sales.mean()),
        "non_promo_mean": float(non_promo_sales.mean()),
        "pct_lift": float(pct_lift),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(effect_size),
        "significant_at_0.05": bool(p_value < 0.05),
    }


def analyze_holiday_effect(df: pd.DataFrame) -> dict:
    holiday_sales = df.loc[df["IsHoliday"] == 1, "Sales"].values
    regular_sales = df.loc[df["IsHoliday"] == 0, "Sales"].values

    t_stat, p_value = stats.ttest_ind(holiday_sales, regular_sales, equal_var=False)
    effect_size = cohens_d(holiday_sales, regular_sales)
    pct_diff = (holiday_sales.mean() - regular_sales.mean()) / regular_sales.mean() * 100

    labels = ["Regular Day", "Holiday"]
    plt.figure(figsize=(7, 6))
    sns.barplot(
        x=labels,
        y=[regular_sales.mean(), holiday_sales.mean()],
        hue=labels,
        palette=["#94a3b8", "#d97706"],
        legend=False,
    )
    plt.title(f"Holiday Effect on Daily Demand ({pct_diff:+.1f}%, p={p_value:.2e})")
    plt.ylabel("Mean Daily Sales")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "holiday_effect.png", dpi=150)
    plt.close()

    return {
        "holiday_mean": float(holiday_sales.mean()),
        "regular_mean": float(regular_sales.mean()),
        "pct_diff": float(pct_diff),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(effect_size),
        "significant_at_0.05": bool(p_value < 0.05),
    }


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    corr_df = df.copy()
    corr_df["StoreTypeCode"] = corr_df["StoreType"].astype("category").cat.codes
    corr_df["AssortmentCode"] = corr_df["Assortment"].astype("category").cat.codes

    numeric_cols = [
        "Sales", "Customers", "Promo", "IsHoliday", "SchoolHoliday",
        "DayOfWeek", "CompetitionDistance", "Promo2", "StoreTypeCode", "AssortmentCode",
    ]
    corr = corr_df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True)
    plt.title("Correlation Matrix of Demand-Driving Features")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=150)
    plt.close()


def run_eda() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_open_days()

    plot_sales_distribution_by_store_type(df)
    decomposition_summary = plot_seasonal_decomposition(df)
    promo_summary = analyze_promo_impact(df)
    holiday_summary = analyze_holiday_effect(df)
    plot_correlation_matrix(df)

    summary = {
        "seasonal_decomposition": decomposition_summary,
        "promo_impact": promo_summary,
        "holiday_effect": holiday_summary,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nFigures saved to {FIGURES_DIR}")
    print(f"Summary saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    run_eda()
