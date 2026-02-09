# %% [markdown]
# # Revenue Forecasting & Policy Impact Analysis
#
# This notebook analyzes daily revenue for a global payments platform.
# The goal is to model historical revenue, estimate the impact of a potential
# policy / tariff change, and forecast revenue forward to support business decisions.

# %% [markdown]
# ## 0) Imports, settings, paths

# %%
# Sanity check: print environment info
import os, sys, time, platform

print("python:", sys.version.split()[0])
print("executable:", sys.executable)
print("pid:", os.getpid())
print("platform:", platform.platform())
print("cwd:", os.getcwd())
time.sleep(0.1)
print("kernel ok")


# %%
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 140)
pd.set_option("display.max_rows", 200)

sns.set_theme(style="whitegrid", palette="muted")
sns.set_context("notebook")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("Python:", sys.version)

# %%
# Paths
# If running as a notebook (no __file__), fallback to cwd.
try:
    ANALYSIS_DIR = Path(__file__).resolve().parent
except NameError:
    ANALYSIS_DIR = Path.cwd()

PROJECT_ROOT = ANALYSIS_DIR.parent
RAW_DIR = PROJECT_ROOT / "raw_data"
CLEAN_DIR = PROJECT_ROOT / "clean_data"
OUTPUT_DIR = PROJECT_ROOT / "output"

for d in [RAW_DIR, CLEAN_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("RAW_DIR:", RAW_DIR)
print("CLEAN_DIR:", CLEAN_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)


# %%
df = pd.read_csv(
    RAW_DIR / "airwallex_revenue_forecast_synth.csv"
)  # <- replace if needed
df.head()

# %% [markdown]
# ## 3) Dataset Overview / Quick EDA
#
# Goal:
# - Understand structure, types, missingness, duplicates, weird values
# - Check treatment/control sizing and quick conversion rates
# - Check whether missingness differs by group


# %%
def df_overview(df: pd.DataFrame, n: int = 5) -> None:
    print("Shape:", df.shape)
    display(df.head(n))  # type: ignore
    print("\nDtypes:\n", df.dtypes)
    print(
        "\nMissingness (%):\n",
        (df.isna().mean() * 100).sort_values(ascending=False).head(30),
    )
    print("\nDuplicate rows:", df.duplicated().sum())


df_overview(df)

# %% [markdown]
# ## Dataset
#
# **Granularity:** Daily time series
# **Time span:** ~2.5 years
#
# **Columns:**
# - `date` (datetime): calendar date
# - `revenue_usd` (float): total daily revenue in USD
# - `marketing_index` (float): proxy for marketing intensity
# - `fx_volatility_index` (float): proxy for FX market conditions

# %% [markdown]
# ## Target Variable
#
# **Primary target:** `revenue_usd`
#
# Depending on modeling choice, revenue may be modeled:
# - in raw USD terms, or
# - on a log scale to stabilize variance and model multiplicative effects
# ## Problem Statement
#
# Revenue is influenced by:
# - long-term growth trends
# - weekly and annual seasonality
# - external drivers such as marketing activity and FX conditions
#
# In addition, a potential policy or tariff change may have introduced
# a structural shift in revenue at some point in the historical data.
#
# The task is to disentangle these effects and produce reliable forecasts.

# %% [markdown]
# ## Analysis Goals
#
# 1. Identify and model trend and seasonality in revenue
# 2. Incorporate external drivers into the forecasting model
# 3. Estimate the direction and magnitude of any policy-related impact
# 4. Generate forward-looking revenue forecasts with uncertainty
# 5. Communicate results clearly for business decision-making
#
# ## What Success Looks Like
#
# **Technical:**
# - Reasonable forecast accuracy on held-out future periods
# - Well-calibrated prediction intervals
#
# **Business:**
# - Errors within an acceptable tolerance (≈3–5% at medium horizons)
# - Clear explanation of assumptions and risks
# - Actionable insights for planning and scenario analysis

# %%

def prepare_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    revenue_col: str = "revenue_usd",
    add_log: bool = True,
    log_col: str = "revenue_usd_log",
    log_mode: str = "auto",   # "log", "log1p", or "auto" (but avoid auto once training starts)
    t0: pd.Timestamp | str | None = None,  # fixed origin for t
    t_col: str = "t",
    weekend_col: str = "is_weekend",
    period_days: float = 365.25,
    harmonics: int = 2,
    tau: pd.Timestamp | str | None = None,  # break date
    break_col: str = "break_term",
    drop_duplicate_dates: bool = True,
    check_gaps: bool = False,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Prepare a clean, reproducible feature set for time-series modelling.

    Adds (by default):
      - log revenue (log_col)
      - time index in days since t0 (t_col)
      - weekend indicator (weekend_col)
      - Fourier seasonality terms: sin_1y_1, cos_1y_1, ..., sin_1y_k, cos_1y_k
      - optional hinge break term: break_col = max(0, t - t_tau)

    Forecast safety:
      - If t0 is provided, t is pinned to that origin for both training and future dates.
      - If tau is provided, break term is computed relative to that same t0.
      - log_mode should be fixed ("log" or "log1p") once training starts.
    """
    d = df.copy() if copy else df

    # --- Validate and standardize date column ---
    if date_col not in d.columns:
        raise ValueError(f"Missing required column: {date_col}")

    d[date_col] = pd.to_datetime(d[date_col]).dt.normalize()
    d = d.sort_values(date_col)

    if drop_duplicate_dates:
        d = d.drop_duplicates(subset=[date_col], keep="first")

    # --- Time origin (t0) ---
    if t0 is None:
        t0_ts = d[date_col].min()
    else:
        t0_ts = pd.to_datetime(t0).normalize()

    # --- Time index (days since t0) ---
    d[t_col] = (d[date_col] - t0_ts).dt.total_seconds() / (24 * 3600)

    # --- Weekend flag ---
    d[weekend_col] = d[date_col].dt.dayofweek >= 5

    # --- Optional break term (hinge / sharp slope break) ---
    if tau is not None:
        
        tau_ts = pd.to_datetime(tau).normalize()
        if tau_ts < t0_ts:
            raise ValueError("tau must be on or after t0")
        t_tau = (tau_ts - t0_ts).total_seconds() / (24 * 3600)
        d[break_col] = np.maximum(0.0, d[t_col] - t_tau)

    # --- Log revenue (variance stabilization / multiplicative effects) ---
    if add_log:
        if revenue_col not in d.columns:
            raise ValueError(f"Missing required column for log transform: {revenue_col}")

        if log_mode not in {"log", "log1p", "auto"}:
            raise ValueError(f"log_mode must be one of 'log', 'log1p', 'auto' (got {log_mode!r})")

        if log_mode == "auto":
            # OK for exploration, but NOT for final training/forecasting
            use_log1p = (d[revenue_col] <= 0).any()
        else:
            use_log1p = (log_mode == "log1p")

        if use_log1p:
            if (d[revenue_col] < 0).any():
                raise ValueError("Negative revenue values found; log/log1p undefined.")
            d[log_col] = np.log1p(d[revenue_col])
        else:
            if (d[revenue_col] <= 0).any():
                raise ValueError("Non-positive revenue values found; use log1p or clean data.")
            d[log_col] = np.log(d[revenue_col])

    # --- Fourier seasonality terms (annual) ---
    for k in range(1, harmonics + 1):
        angle = 2.0 * np.pi * k * d[t_col] / period_days
        d[f"sin_1y_{k}"] = np.sin(angle)
        d[f"cos_1y_{k}"] = np.cos(angle)

    # --- Sanity checks ---
    if not d[date_col].is_monotonic_increasing:
        raise ValueError("Dates are not sorted/monotonic after preparation (unexpected).")

    # Optional: gaps check (off by default to avoid noisy prints)
    if check_gaps:
        expected = pd.date_range(d[date_col].min(), d[date_col].max(), freq="D")
        missing_dates = expected.difference(d[date_col])
        if len(missing_dates) > 0:
            print(f"Warning: {len(missing_dates)} missing dates (series has gaps).")
    # Note - | here means set union rather than bitwise or. 
    expected_fourier = {f"sin_1y_{k}" for k in range(1, harmonics + 1)} | {
        f"cos_1y_{k}" for k in range(1, harmonics + 1)
    }
    actual_fourier = {c for c in d.columns if c.startswith(("sin_1y_", "cos_1y_"))}

    missing = expected_fourier - actual_fourier
    extra = actual_fourier - expected_fourier
    if missing:
        raise ValueError(f"Missing Fourier columns: {missing}")
    if extra:
        print(f"Warning: extra Fourier columns present: {extra}")

    fourier_cols = sorted(actual_fourier)
    if not np.isfinite(d[fourier_cols].to_numpy()).all():
        raise ValueError("Non-finite values found in Fourier features (unexpected).")

    return d

# %%
# list(range(0, 3))
state = {
    "t0": pd.to_datetime(df["date"]).min().normalize(),
    "tau": None,
    "log_mode": "log",
}


# %%
# Fit a regression model predicting USD from date, day of week, and
# month of the year. This will help quantify the effects.
import statsmodels.api as sm
from statsmodels.formula.api import ols



# %% [markdown]
### Detecting an Unknown Trend Break (Single Structural Break)

## Objective
# We want to test whether there is evidence of a structural change in the
# revenue growth rate (e.g. due to a policy or regulatory change), without
# assuming in advance when (or if) such a change occurred.

## Baseline model
# We begin with a well-specified baseline model that already explains most
# systematic variation in revenue:
# - Linear time trend
# - Weekend effects
# - Annual seasonality (Fourier terms)
#
# Any detected break should therefore represent new structure not already
# captured by trend or seasonality.

## Break model (slope change)
# We allow the slope of the time trend to change at an unknown breakpoint τ
# using a hinge (ramp) function:
#
#   hinge_t(τ) = max(0, t − τ)
#
# The regression becomes:
#   log(revenue_t) = baseline + β·t + δ·hinge_t(τ) + error
#
## Interpretation:
# - β: pre-break growth rate
# - β + δ: post-break growth rate
# - δ: change in growth rate after the break

## Estimating the break date
# Because τ is unknown, we perform a grid search over candidate break dates:
# - Exclude the first and last N days to ensure data on both sides
# - Evaluate candidate τ values at regular intervals (e.g. weekly)
# - Fit the model for each τ and record a penalised fit metric (BIC)

## Model selection
# The breakpoint τ that minimises BIC is selected.
# BIC is used to penalise unnecessary complexity and reduce the risk of
# detecting spurious breaks driven by noise.

## Outputs
# This procedure yields:
# - An estimated break date
# - Pre- and post-break growth rates
# - Statistical evidence for a change in slope
# - Diagnostic plots (BIC vs τ, fitted piecewise trend)

## Rationale
# This approach is interpretable, conservative, and well-suited to identifying
# potential policy-driven changes in growth without overfitting.


# %%

# Define a series of functions to perform and plot grid search finding a single breakpoint
# in the trend.

# -----------------------------
# Single-break grid search (slope break) with a hinge term
# y ~ baseline + delta * max(0, t - tau)
# -----------------------------


def single_break_grid_search(
    df,
    date_col="date",
    y_col="revenue_usd_log",
    t_col="t",
    baseline_terms=None,
    min_side_days=120,  # exclude first/last N days as candidate breaks
    step_days=7,  # evaluate candidate break every 7 days (speeds up)
    criterion="bic",  # "bic" (default) or "aic" or "rss"
):
    """
    Grid-search an unknown single breakpoint tau for a *slope change* model.

    Model form:
      y = baseline + beta * t + delta * hinge_tau + error
    where:
      hinge_tau = max(0, t - t_tau)

    Returns:
      results_df: one row per candidate tau with criterion + params
      best_tau_date: selected break date (Timestamp)
      best_model: fitted statsmodels regression for the best tau
    """

    df = prepare_features(df)

    # Ensure time index exists
    # if t_col not in df.columns:
    #     df[t_col] = (df[date_col] - df[date_col].min()).dt.days.astype(float)

    # Default baseline: your current spec (adjust if your column names differ)
    if baseline_terms is None:
        baseline_terms = [
            f"{t_col}",
            "is_weekend",
            "sin_1y_1",
            "cos_1y_1",
            "sin_1y_2",
            "cos_1y_2",
        ]

    # Build the core formula without hinge (we'll add hinge_tau each iteration)
    base_formula = f"{y_col} ~ " + " + ".join(baseline_terms)

    # Candidate tau dates: exclude edges so we have data on both sides
    dates = df[date_col]
    min_date = dates.min() + pd.Timedelta(days=min_side_days)
    max_date = dates.max() - pd.Timedelta(days=min_side_days)

    # Use a spaced grid of candidate dates
    cand_dates = pd.date_range(min_date, max_date, freq=f"{step_days}D")
    if len(cand_dates) < 5:
        raise ValueError(
            "Not enough candidate break dates. Reduce min_side_days or step_days."
        )

    rows = []

    for tau_date in cand_dates:
        # tau in t-units (days since start)
        tau_t = float(df.loc[df[date_col] <= tau_date, t_col].max())

        # Hinge term: (t - tau_t) if t > tau_t else 0
        hinge = np.maximum(0.0, df[t_col].to_numpy() - tau_t)
        df["_hinge"] = hinge

        # Fit model with hinge
        m = ols(base_formula + " + _hinge", data=df).fit()

        # Choose objective
        if criterion == "bic":
            score = m.bic
        elif criterion == "aic":
            score = m.aic
        elif criterion == "rss":
            score = np.sum(m.resid**2)
        else:
            raise ValueError("criterion must be one of: 'bic', 'aic', 'rss'")

        beta_pre = m.params.get(t_col, np.nan)
        delta = m.params.get("_hinge", np.nan)
        beta_post = beta_pre + delta

        rows.append(
            {
                "tau_date": tau_date,
                "tau_t": tau_t,
                criterion: score,
                "beta_pre": beta_pre,
                "delta_slope": delta,
                "beta_post": beta_post,
                "p_delta": m.pvalues.get("_hinge", np.nan),
                "r2": m.rsquared,
            }
        )

    results_df = pd.DataFrame(rows).sort_values(criterion).reset_index(drop=True)

    # Best tau
    best_tau_date = results_df.loc[0, "tau_date"]
    best_tau_t = results_df.loc[0, "tau_t"]

    # Refit best model (cleanly) and return it
    df["_hinge"] = np.maximum(0.0, df[t_col].to_numpy() - best_tau_t)
    best_model = ols(base_formula + " + _hinge", data=df).fit()

    return results_df, best_tau_date, best_model


# -----------------------------
# Helper: plot criterion over tau dates + optional fitted trend overlay
# -----------------------------


def plot_break_search(results_df, criterion="bic"):
    plt.figure(figsize=(14, 5))
    plt.plot(results_df["tau_date"], results_df[criterion], marker="o", linewidth=1)
    best_idx = results_df[criterion].idxmin()
    best_tau = results_df.loc[best_idx, "tau_date"]
    plt.axvline(best_tau, linestyle="--")
    plt.title(f"Single-break grid search: {criterion.upper()} vs candidate break date")
    plt.xlabel("Candidate break date (tau)")
    plt.ylabel(criterion.upper())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %%
# Run the single-break grid search and plot results
# Set break_col to _hinge to match what we set it to in the grid search
df = prepare_features(df, **state, break_col = '_hinge')
from statsmodels.formula.api import ols
results_df, best_tau_date, best_model = single_break_grid_search(
    df=df,
    min_side_days=120,  # you can try 90 or 150 as sensitivity checks
    step_days=7,  # 7 = weekly grid; 1 = daily grid (slower)
    criterion="bic",
)

print("Best break date (BIC):", best_tau_date.date())
print(best_model.summary().tables[1])  # coefficient table

plot_break_search(results_df, criterion="bic")

# best_tau_t = (pd.to_datetime(best_tau_date).normalize() - state["t0"]).days
# Update the state dict with the best tau_date so next time we run prepare_features we'll 
# compute a break column
state["tau"] = best_tau_date

# %%
baseline_model = ols(
    "revenue_usd_log ~ t + is_weekend + sin_1y_1 + cos_1y_1 + sin_1y_2 + cos_1y_2",
    data=df,
).fit()

print("Baseline BIC:", baseline_model.bic)
print("Break model BIC:", best_model.bic)
print("ΔBIC (baseline − break):", baseline_model.bic - best_model.bic)

# %% 
results_df.head()

# %%
# We use the definition that a change of <2 in BIC is not meaningful 
# evidence (bayes factor of ~3:1 or less). See e.g. Kass & Raftery (1995)
best_bic = results_df['bic'].min()
mask = results_df['bic'] <= best_bic + 2.0
earliest_date = results_df.loc[mask, 'tau_date'].min()
latest_date = results_df.loc[mask, 'tau_date'].max()
print(f"BIC within 2 of best ({best_bic:.2f}) for break dates between {earliest_date.date()} and {latest_date.date()}")

# %%
def overlay_baseline_vs_break(
    df,
    baseline_model,
    break_model,
    *,
    state: dict,
    date_col: str = "date",
    y_col: str = "revenue_usd_log",
    title: str = "Fitted log revenue: baseline vs single-break model",
    break_col: str = "_hinge",
):
    """
    Plot actual y plus fitted values from:
      - baseline_model (no break_term)
      - break_model (with break_term)
    Assumes `state` contains at least t0/log_mode and (optionally) tau.
    """

    # --- Build features WITHOUT break for baseline predictions ---
    state_no_tau = dict(state)
    state_no_tau["tau"] = None
    d_base = prepare_features(df, **state_no_tau)

    # Baseline predictions (must match the model's design columns)
    d_base["_yhat_base"] = baseline_model.predict(d_base)

    # --- Build features WITH break for break-model predictions ---
    d_break = prepare_features(df, **state, break_col=break_col)

    # Break predictions (only valid if tau is set)
    if state.get("tau") is None:
        raise ValueError("state['tau'] is None; cannot plot break model overlay.")

    if break_col not in d_break.columns:
        raise ValueError(f"Expected break column {break_col!r} not found in features.")

    d_break["_yhat_break"] = break_model.predict(d_break)

    # --- Plot ---
    plt.figure(figsize=(14, 6))
    plt.plot(d_break[date_col], d_break[y_col], alpha=0.25, label="Actual (log revenue)")
    plt.plot(d_base[date_col], d_base["_yhat_base"], linewidth=2, label="Fitted: baseline")

    plt.plot(
        d_break[date_col],
        d_break["_yhat_break"],
        linewidth=2,
        alpha=0.6,
        label="Fitted: break (piecewise slope)",
    )

    tau_date = state["tau"]
    plt.axvline(tau_date, linestyle="--", label=f"Break: {pd.to_datetime(tau_date).date()}")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Log revenue")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # return d  # handy if you want to inspect the fitted columns


# %%
best_idx = results_df['bic'].idxmin()
best_tau_t = results_df.loc[best_idx, "tau_t"]
overlay_baseline_vs_break(
    df,
    baseline_model,
    break_model = best_model,
    state=state,
    title="Fitted log revenue: baseline vs single-break slope model",
)

# %%
def add_smooth_break_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    t_col: str = "t",
    center_date,
    width_days: float,
    prefix: str = "_sb",   # "smooth break"
    copy: bool = True,
) -> tuple[pd.DataFrame, str]:
    """
    Add smooth-break features for a gradual slope change centered at `center_date`.

    We build a smooth step s(t) = sigmoid((t - tau)/w), then a smooth "hinge-like"
    term: (t - tau) * s(t). This behaves like:
      - ~0 before tau
      - ~ (t - tau) after tau
      - smooth transition around tau controlled by width_days

    Returns
    -------
    (d, feature_col_name)
    """
    d = df.copy() if copy else df

    if date_col not in d.columns:
        raise ValueError(f"Missing '{date_col}' in df.")
    if t_col not in d.columns:
        raise ValueError(f"Missing '{t_col}' in df.")

    tau = pd.to_datetime(center_date)

    # Ensure we have a numeric "days since tau" aligned with your t_col units.
    # Here we assume t_col is in DAYS (as in your prepare_features).
    # We'll compute dt in days using timestamps to be safe.
    dt_days = (pd.to_datetime(d[date_col]) - tau).dt.total_seconds() / (24 * 3600)

    w = float(width_days)
    if not np.isfinite(w) or w <= 0:
        raise ValueError("width_days must be a positive finite number.")

    # Smooth step (sigmoid). Clip argument to avoid overflow.
    z = np.clip(dt_days.to_numpy() / w, -60, 60)
    s = 1.0 / (1.0 + np.exp(-z))

    # Smooth hinge-like term controlling slope change
    feat_col = f"{prefix}_term"
    d[feat_col] = dt_days.to_numpy() * s

    if not np.isfinite(d[feat_col].to_numpy()).all():
        raise ValueError("Non-finite values in smooth-break feature (unexpected).")

    return d, feat_col


def smooth_break_grid_search(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    y_col: str = "revenue_usd_log",
    t_col: str = "t",
    baseline_terms: list[str] | None = None,
    min_side_days: int = 120,
    step_days: int = 7,
    width_days_grid: list[int] = (14, 28, 56, 84),
    criterion: str = "bic",  # "bic" | "aic" | "rss"
    feature_prefix: str = "_sb",
) -> tuple[pd.DataFrame, pd.Timestamp, int, object]:
    """
    2D grid search for a smooth/gradual slope change:
      - center_date (tau) on a date grid
      - width_days in width_days_grid

    Model:
      y ~ baseline_terms + smooth_break_feature
    where smooth_break_feature = (t - tau) * sigmoid((t - tau)/width)

    Returns
    -------
    results_df : DataFrame with columns:
        - tau_date (center date)
        - width_days
        - bic / aic / rss (depending on criterion)
        - hinge_coef (coefficient on smooth break term)
    best_tau : Timestamp
    best_width : int
    best_model : fitted statsmodels result
    """
    if baseline_terms is None:
        baseline_terms = [
            "is_weekend",
            "sin_1y_1",
            "cos_1y_1",
            "sin_1y_2",
            "cos_1y_2",
        ]

    d = df.copy()
    
    d = prepare_features(d, date_col=date_col, t_col=t_col)

    if y_col not in d.columns:
        raise ValueError(f"Missing '{y_col}' in df.")
    if t_col not in d.columns:
        raise ValueError(f"Missing '{t_col}' in df.")

    # Candidate tau dates (exclude edges)
    start = d[date_col].min() + pd.Timedelta(days=min_side_days)
    end = d[date_col].max() - pd.Timedelta(days=min_side_days)
    if start >= end:
        raise ValueError("min_side_days too large for the span of the data.")

    tau_grid = pd.date_range(start, end, freq=f"{step_days}D")

    rows = []
    best_val = np.inf
    best_tau = None
    best_w = None
    best_model = None
    
    

    # Build baseline RHS once as a string
    rhs_base = " + ".join([t_col] + baseline_terms) if baseline_terms else t_col

    for tau in tau_grid:
        for w in width_days_grid:
            dd, feat_col = add_smooth_break_features(
                d,
                date_col=date_col,
                t_col=t_col,
                center_date=tau,
                width_days=w,
                prefix=feature_prefix,
                copy=True,
            )

            formula = f"{y_col} ~ {rhs_base} + {feat_col}"
            m = ols(formula, data=dd).fit()

            if criterion == "bic":
                val = float(m.bic)
            elif criterion == "aic":
                val = float(m.aic)
            elif criterion == "rss":
                val = float(np.sum(m.resid ** 2))
            else:
                raise ValueError("criterion must be one of {'bic','aic','rss'}")

            coef = float(m.params.get(feat_col, np.nan))
            se = float(m.bse.get(feat_col, np.nan))
            tstat = coef / se if np.isfinite(coef) and np.isfinite(se) and se != 0 else np.nan

            rows.append(
                {
                    "tau_date": tau,
                    "width_days": int(w),
                    criterion: val,
                    "smooth_coef": coef,
                    "smooth_se": se,
                    "smooth_t": tstat,
                }
            )

            if val < best_val:
                best_val = val
                best_tau = tau
                best_w = int(w)
                best_model = m

    results_df = pd.DataFrame(rows).sort_values([criterion, "tau_date", "width_days"]).reset_index(drop=True)
    return results_df, pd.to_datetime(best_tau), best_w, best_model
# %%
# Define plot_smooth_break_search function that plots one bic line for each w value

from statsmodels.formula.api import ols

def plot_smooth_break_search(results_df, criterion="bic"):
    plt.figure(figsize=(14, 6))
    for w in results_df["width_days"].unique():
        subset = results_df[results_df["width_days"] == w]
        plt.plot(
            subset["tau_date"],
            subset[criterion],
            marker="o",
            linewidth=1,
            label=f"Width {w} days",
        )
    best_idx = results_df[criterion].idxmin()
    best_tau = results_df.loc[best_idx, "tau_date"]
    plt.axvline(best_tau, linestyle="--", color="black", label="Best break date")
    plt.title(f"Smooth-break grid search: {criterion.upper()} vs candidate break date")
    plt.xlabel("Candidate break date (tau)")
    plt.ylabel(criterion.upper())
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
# Run the smooth-break grid search and plot results

results_sb_df, best_sb_tau, best_sb_w, best_sb_model = smooth_break_grid_search(
    df=df,
    min_side_days=120,
    step_days=7,
    width_days_grid=[14, 28, 56, 84],
    criterion="bic",
)

print("Best smooth break date (BIC):", best_sb_tau.date())
print("Best smooth break width (days):", best_sb_w)
print(best_sb_model.summary().tables[1])  # coefficient table
# Wait - do we have a plot_break_search function that works here?
plot_smooth_break_search(results_sb_df, criterion="bic")
# %%
# sanity checking we have the same obs and outcome for the sharp and smooth models. Bic is 
# very different though. 
print(best_model.nobs, best_model.model.endog_names, best_model.bic)
print(best_sb_model.nobs, best_sb_model.model.endog_names, best_sb_model.bic)
print(best_sb_model.model.exog_names)
# %% [markdown]
## Sharp vs smooth break comparison
##### We considered both a sharp (instantaneous) and smooth (logistic) change in trend.
##### Both models identify a break at approximately the same date.
##### Model comparison via BIC shows no meaningful improvement from allowing a smooth transition, and within the smooth family, narrower transition windows are preferred.
##### This indicates that while a change in growth rate is supported by the data, the transition width is not identified, and an instantaneous break provides the most parsimonious representation.

# %%
# Explore residual autocorrelation to see if our model capture everything.
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

resid = best_model.resid

plot_acf(resid, lags=40)
plot_pacf(resid, lags=40)

# %%

# Lock in feature and outcome cols, split the data, prepare for forecasting
df_feat = prepare_features(df, **state)

y_col = "revenue_usd_log"

feature_cols = [
    "t",
    "is_weekend",
    "sin_1y_1", "cos_1y_1",
    "sin_1y_2", "cos_1y_2",
    "break_term",
]

X_train = df_feat[feature_cols]
y_train = df_feat[y_col]

# We have to manually add a constant column for statsmodels or else the 
# model will be fited to pass a line through (0, 0) which is very bad in log space. 
X_train = sm.add_constant(X_train.astype(float), has_constant="add")

X_train = X_train.astype(float)   # bool -> 0/1
y_train = y_train.astype(float)

print("X dtypes:\n", X_train.dtypes)
print("y dtype:", y_train.dtype)

print("Any NaNs in X?", X_train.isna().any().any())
print("Any NaNs in y?", y_train.isna().any())

print("Any inf in X?", np.isinf(X_train.to_numpy()).any())
print("Any inf in y?", np.isinf(y_train.to_numpy()).any())

# %%
# Fit the final model. We don't have to specify a model formula, because we've split the X and
# y dataframes already.
import statsmodels.api as sm

model = sm.OLS(y_train, X_train)
results = model.fit()

print(results.summary())
# %%

# Create future dates dataframe
last_date = df["date"].max()
horizon = 366

future_df = pd.DataFrame({
    "date": pd.date_range(
        last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
    )
})
# %%
future_feat = prepare_features(
    future_df,
    **state,
    add_log=False,
)


X_future = future_feat[feature_cols].astype(float)
X_future = sm.add_constant(X_future.astype(float), has_constant="add")

assert list(X_future.columns) == list(X_train.columns)
print("X and X future columns aligned")
X_future.head()

# %% 
print("X_train cols:", list(X_train.columns))
print("X_future cols:", list(X_future.columns))
print("Same columns?", list(X_train.columns) == list(X_future.columns))
print("Train shape:", X_train.shape, "Future shape:", X_future.shape)

feature_cols = X_train.columns.tolist()
X_future = X_future.reindex(columns=feature_cols).astype(float)

if X_future.isna().any().any():
    raise ValueError("Missing feature columns in X_future after reindex.")

# %%
# Anlytic forecast

# Use our best model to predict X_future inputs and get predictions
pred = results.get_prediction(X_future)
pred_df = pred.summary_frame(alpha=0.2)  # 80% intervals
pred_df.head()
# %%
# Plot the forecast in log space 
plt.figure(figsize=(14, 6))
# Plot the actual data 
plt.plot(df_feat["date"], df_feat[y_col], label="Historical log revenue", alpha=0.5)
# Plot the forecast with prediction intervals
plt.plot(future_feat["date"], pred_df["mean"], label="Forecast log revenue", color="orange")

plt.fill_between(
    future_feat["date"],
    pred_df["obs_ci_lower"],
    pred_df["obs_ci_upper"],
    color="orange",
    alpha=0.3,
    label="80% Prediction Interval",
)
# %%
# Plot within sample model predictions as an extra layer of sanity check. 
df_feat["_yhat"] = results.predict(X_train)
plt.plot(df_feat["date"], df_feat[y_col], alpha=0.3)
plt.plot(df_feat["date"], df_feat["_yhat"], linewidth=2)

# %%
df_feat["_yhat_in_sample"] = results.predict(X_train)

plt.plot(df_feat["date"], df_feat[y_col], alpha=0.25, label="Actual")
plt.plot(df_feat["date"], df_feat["_yhat_in_sample"], lw=2, label="Fitted mean")
plt.plot(future_feat["date"], pred_df["mean"], lw=2, label="Forecast mean")
plt.legend()

# %%
# Test that errors are symmetric
down_a = pred_df["mean"] - pred_df["obs_ci_lower"]
up_a   = pred_df["obs_ci_upper"] - pred_df["mean"]

print(down_a.mean(), up_a.mean(), np.max(np.abs(down_a - up_a)))

# Now doing the bootstrap version of prediction rather than the analytic 
# version. 
from sklearn.utils import resample

def bootstrap_predictions(model, X_train, y_train, X_future, n_bootstrap=1000, confidence_level=0.8):
    """
    Generate bootstrap prediction intervals
    
    Parameters:
    - model: fitted model with coef_ and predict() methods
    - X_train, y_train: training data
    - X_future: future features to predict
    - n_bootstrap: number of bootstrap iterations
    - confidence_level: confidence level for intervals
    """
    
    # Get model parameters
    beta = model.params.values
    
    # Calculate residuals for noise sampling
    y_pred_train = model.predict(X_train)
    residuals = y_train - y_pred_train
    residual_std = np.std(residuals)
    
    # Check for statsmodels method instead
    if hasattr(model, 'cov_params'):
        param_cov = model.cov_params().values  # statsmodels
    elif hasattr(model, 'covariance_matrix_'):
        param_cov = model.covariance_matrix_   # sklearn (if available)
    else:
        param_cov = estimate_param_covariance(model, X_train, y_train, n_samples=100)
        
    # Prepare arrays for storage
    n_future = len(X_future)
    all_predictions = np.zeros((n_bootstrap, n_future))
    
    # Add intercept column to X_future if needed
    if X_future.shape[1] == len(beta) - 1:
        X_future_with_intercept = np.column_stack([np.ones(len(X_future)), X_future])
    else:
        X_future_with_intercept = X_future.copy()
    
    # Bootstrap iterations
    for i in range(n_bootstrap):
        
        # 1. Sample beta coefficients from multivariate normal
        if param_cov is not None:
            beta_sample = np.random.multivariate_normal(beta, param_cov)
        else:
            # Print a warning
            print("Warning: No parameter covariance matrix available. Using exact beta weights.")
            beta_sample = beta  # Use original if no covariance available
        
        # 2. Make predictions with sampled parameters
        # Matrix multiplication of X_future parameter values and sampled beta weights.
        pred_mean = X_future_with_intercept @ beta_sample
        
        # 3. Add residual noise
        # sample from a gaussian distribution with mean 0 and SD of the residuals
        residual_noise = np.random.normal(0, residual_std, n_future)
        
        # 4. Final prediction with uncertainty
        all_predictions[i, :] = pred_mean + residual_noise
    
    return all_predictions

def estimate_param_covariance(model, X, y, n_samples=100):
    """Estimate parameter covariance via bootstrap if not available"""
    n_params = len(model.coef_) + 1  # +1 for intercept
    param_samples = np.zeros((n_samples, n_params))
    
    for i in range(n_samples):
        X_boot, y_boot = resample(X, y, random_state=i)
        model_class = type(model)              # Get the class
        model_params = model.get_params()      # Get parameters dict
        # This ensures we use the same model type and hyperparameters from our main model
        model_boot = model_class(**model_params)  # Create new instance
        model_boot = model_boot.fit(X_boot, y_boot)  # Fit it
        # The above four lines can be done in this one, it's just harder to parse. 
        # model_boot = type(model)(**model.get_params()).fit(X_boot, y_boot)
        param_samples[i, :] = np.append(model_boot.intercept_, model_boot.coef_)
    
    return np.cov(param_samples.T)
# %%
# Usage
n_bootstrap = 1000
all_preds = bootstrap_predictions(
    model.fit(), X_train, y_train, X_future, 
    n_bootstrap=n_bootstrap, 
    confidence_level=0.8
)

# %%
# Create DataFrame with all paths
future_dates = future_feat["date"]  # Adjust this to your date column
bootstrap_df = pd.DataFrame(
    all_preds.T,  # Transpose so dates are rows
    index=future_dates,
    columns=[f"bootstrap_{i}" for i in range(n_bootstrap)]
)

CI = .8
# Calculate summary statistics
alpha = (1 - CI) / 2  # For 80% CI
bootstrap_summary = pd.DataFrame({
    'date': future_dates,
    'mean': np.mean(all_preds, axis=0),
    'obs_ci_lower': np.quantile(all_preds, alpha, axis=0),
    'obs_ci_upper': np.quantile(all_preds, 1-alpha, axis=0),
    'std': np.std(all_preds, axis=0)
})

# Plot the results
plt.figure(figsize=(14, 8))

# Plot historical data
plt.plot(df_feat["date"], df_feat[y_col], label="Historical", alpha=0.7)

# Plot some bootstrap paths (sample for visibility)
for i in range(0, n_bootstrap, n_bootstrap//20):  # Plot every 20th path
    plt.plot(future_dates, all_preds[i, :], alpha=0.1, color='gray', linewidth=0.5)

# Plot bootstrap mean and CI
plt.plot(bootstrap_summary['date'], bootstrap_summary['mean'], 
         label="Bootstrap Mean", color="red", linewidth=2)

plt.fill_between(
    bootstrap_summary['date'],
    bootstrap_summary['obs_ci_lower'],
    bootstrap_summary['obs_ci_upper'],
    color="red", alpha=0.3, label="80% Bootstrap CI"
)

plt.legend()
plt.title("Bootstrap Prediction Intervals")
plt.show()

# %%
down = bootstrap_summary['mean'] - bootstrap_summary['obs_ci_lower']
up   = bootstrap_summary['obs_ci_upper'] - bootstrap_summary['mean']

print("avg down:", down.mean(), "avg up:", up.mean())
print("max abs diff:", np.max(np.abs(down - up)))
# %%
# Doing it in raw revenue space now by exponentiating the log forecasts
# Transform each individual bootstrap path to revenue space
revenue_predictions = np.exp(all_preds)

alpha = (1 - 0.8) / 2  # For 80% CI
revenue_summary = pd.DataFrame({
    'date': future_dates,
    'mean': np.mean(revenue_predictions, axis=0),           # Mean in revenue space
    'obs_ci_lower': np.quantile(revenue_predictions, alpha, axis=0),      # 10th percentile
    'obs_ci_upper': np.quantile(revenue_predictions, 1-alpha, axis=0),    # 90th percentile
    'std': np.std(revenue_predictions, axis=0)              # Std in revenue space
})

# Create DataFrame with all revenue paths (transformed)
revenue_bootstrap_df = pd.DataFrame(
    revenue_predictions.T,  # Transpose so dates are rows
    index=future_dates,
    columns=[f"bootstrap_{i}" for i in range(n_bootstrap)]
)

# %%
# Plot the results in revenue space
plt.figure(figsize=(14, 8))

# Plot historical data (transform from log if needed)
plt.plot(df_feat["date"], np.exp(df_feat[y_col]), label="Historical Revenue", alpha=0.7)

# Plot some bootstrap revenue paths (sample for visibility)
for i in range(0, n_bootstrap, n_bootstrap//20):  # Plot every 20th path
    plt.plot(future_dates, revenue_predictions[i, :], alpha=0.1, color='gray', linewidth=0.5)

# Plot bootstrap mean and CI in revenue space
plt.plot(revenue_summary['date'], revenue_summary['mean'], 
         label="Bootstrap Mean Revenue", color="red", linewidth=2)

plt.fill_between(
    revenue_summary['date'],
    revenue_summary['obs_ci_lower'],
    revenue_summary['obs_ci_upper'],
    color="red", alpha=0.3, label="80% Bootstrap CI"
)

# Format Y-axis to show millions
from matplotlib.ticker import FuncFormatter

def millions_formatter(x, pos):
    return f'{x/1e6:.1f}M'

plt.gca().yaxis.set_major_formatter(FuncFormatter(millions_formatter))

plt.legend()
plt.title("Bootstrap Prediction Intervals - Revenue Space")
plt.ylabel("Revenue (Millions)")
plt.show()

print("Revenue bootstrap summary:")
print(revenue_summary.head())

# %%
# This cell defines all the functions needed to evaluate the MAE of different
# FX volatility windows using a rolling-origin time-series cross-validation approach.


# ----------------------------
# 1) Helper: rolling-origin split generator
# ----------------------------
# Note: this is a generator function that yields train/test indices for each fold.
# Basically if step days is set to 14, you'll get a new fold (train + test data) every
# 14 days. The training data is all data up to the origin date, and the test data
# is the next horizon_days after the origin date. We dont do overlapping training windows
# because we want to simulate real forecasting as closely as possible.
def rolling_origin_splits(
    df,
    date_col="date",
    min_train_days=365,  # must have at least this much training history
    step_days=14,  # move forecast origin forward by this many days each fold
    horizon_days=30,  # predict next H days
):
    """
    Yields (train_idx, test_idx, origin_date) for a rolling-origin backtest.

    - Train uses all data up to origin_date (expanding window).
    - Test is the next 'horizon_days' after origin_date.

    Why: This respects temporal ordering and approximates real forecasting.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    dates = df[date_col]

    start_date = dates.min() + pd.Timedelta(days=min_train_days)
    last_possible_origin = dates.max() - pd.Timedelta(days=horizon_days)

    origin = start_date
    while origin <= last_possible_origin:
        train_idx = df.index[dates <= origin]
        test_idx = df.index[
            (dates > origin) & (dates <= origin + pd.Timedelta(days=horizon_days))
        ]
        if len(test_idx) > 0 and len(train_idx) > 0:
            yield train_idx, test_idx, origin
        origin += pd.Timedelta(days=step_days)


# %%

def backtest_final_model(
    df,
    *,
    state,                        # contains t0, tau, log_mode etc
    feature_cols_no_const,        # ["t","is_weekend","sin_1y_1",...,"break_term"]
    date_col="date",
    y_log_col="revenue_usd_log",
    y_usd_col="revenue_usd",
    horizon_days=30,
    min_train_days=365,
    step_days=14,
    alpha=0.2,                    # 80% PI
):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    rows = []

    for fold_i, (train_idx, test_idx, origin) in enumerate(
        rolling_origin_splits(
            df, date_col=date_col,
            min_train_days=min_train_days,
            step_days=step_days,
            horizon_days=horizon_days,
        )
    ):
        train_raw = df.loc[train_idx].copy()
        test_raw  = df.loc[test_idx].copy()

        # --- Feature engineering (forecast-safe) ---
        train_feat = prepare_features(train_raw, **state, add_log=True)
        test_feat  = prepare_features(test_raw,  **state, add_log=True)

        # --- Build X/y ---
        X_train = train_feat[feature_cols_no_const].astype(float)
        y_train = train_feat[y_log_col].astype(float)

        X_test = test_feat[feature_cols_no_const].astype(float)
        y_test_log = test_feat[y_log_col].astype(float).to_numpy()
        y_test_usd = test_feat[y_usd_col].astype(float).to_numpy()

        # Add intercept (critical)
        X_train = sm.add_constant(X_train, has_constant="add")
        X_test  = sm.add_constant(X_test,  has_constant="add")

        # Fit
        res = sm.OLS(y_train, X_train).fit()

        # Predict + PI in log space
        pred = res.get_prediction(X_test)
        sf = pred.summary_frame(alpha=alpha)

        mu_log = sf["mean"].to_numpy()
        lo_log = sf["obs_ci_lower"].to_numpy()
        hi_log = sf["obs_ci_upper"].to_numpy()

        # --- Point forecasts in $ ---
        # exp(mean log) is median of lognormal. Stakeholders usually accept it.
        yhat_usd = np.exp(mu_log)
        lo_usd = np.exp(lo_log)
        hi_usd = np.exp(hi_log)

        # --- Metrics: log space ---
        ae_log = np.abs(y_test_log - mu_log)
        mae_log = ae_log.mean()
        rmse_log = np.sqrt(np.mean((y_test_log - mu_log) ** 2))

        # Interpret log MAE as a typical % error factor
        typical_pct_err = np.expm1(mae_log)  # approx typical relative error

        # --- Metrics: $ space ---
        ae_usd = np.abs(y_test_usd - yhat_usd)
        mae_usd = ae_usd.mean()

        # WAPE is stakeholder-friendly
        wape = ae_usd.sum() / np.maximum(y_test_usd.sum(), 1e-12)

        # Interval calibration (log)
        covered = np.mean((y_test_log >= lo_log) & (y_test_log <= hi_log))
        avg_width_log = np.mean(hi_log - lo_log)

        # Interval calibration ($)
        covered_usd = np.mean((y_test_usd >= lo_usd) & (y_test_usd <= hi_usd))
        avg_width_usd = np.mean(hi_usd - lo_usd)

        rows.append({
            "fold": fold_i,
            "origin_date": origin,
            "n_train": len(train_feat),
            "n_test": len(test_feat),
            "mae_log": mae_log,
            "rmse_log": rmse_log,
            "typical_pct_err": typical_pct_err,   # ~ “typical % error”
            "mae_usd": mae_usd,
            "wape": wape,
            "pi80_coverage_log": covered,
            "pi80_width_log": avg_width_log,
            "pi80_coverage_usd": covered_usd,
            "pi80_width_usd": avg_width_usd,
        })

    folds = pd.DataFrame(rows)

    summary = {
        "folds": len(folds),
        "horizon_days": horizon_days,
        "mae_log_mean": folds["mae_log"].mean(),
        "mae_log_p90": folds["mae_log"].quantile(0.9),
        "typical_pct_err_mean": folds["typical_pct_err"].mean(),
        "wape_mean": folds["wape"].mean(),
        "wape_p90": folds["wape"].quantile(0.9),
        "pi80_coverage_log": folds["pi80_coverage_log"].mean(),
        "pi80_coverage_usd": folds["pi80_coverage_usd"].mean(),
    }

    return folds, pd.Series(summary)
# %%

backtest_results, backtest_summary = backtest_final_model(
    df=df,
    state=state,
    feature_cols_no_const=feature_cols,
    date_col="date",
    y_log_col="revenue_usd_log",
    y_usd_col="revenue_usd",
    horizon_days=90,
    min_train_days=365,
    step_days=14,
    alpha=0.2,
)
# %%
print("=== Backtest summary ===")
print(backtest_summary)
# %%
