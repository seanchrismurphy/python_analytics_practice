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
# Use seaborn to plot revenue over time highlighting seasonality
# We see a slow upward trend with yearly seasonality peaking around
# March and bottoming out around August each year.
plt.figure(figsize=(14, 6))
df = prepare_features(df, **state, add_log=True)
sns.lineplot(data=df, x="date", y="revenue_usd_log", marker="o")
plt.title("Daily Revenue Over Time")
plt.xlabel("Date")
plt.ylabel("Revenue (USD)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# %%
# Calculate a day of week variable and visualise weekly seasonality
# Clear pattern with a drop over the weekend and flat throughout the week.
df["day_of_week"] = df["date"].dt.day_name()
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df,
    x="day_of_week",
    y="revenue_usd_log",
    order=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
)
plt.title("Weekly Seasonality in Revenue")
plt.xlabel("Day of Week")
plt.ylabel("Revenue (USD)")
plt.tight_layout()
plt.show()

# %%
# Calculate day of the month variable and visualise monthly seasonality
# No clear seasonality by day of month, but some variability.
df["day_of_month"] = df["date"].dt.day
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="day_of_month", y="revenue_usd_log")
plt.title("Monthly Seasonality in Revenue")
plt.xlabel("Day of Month")
plt.ylabel("Revenue (USD)")
plt.tight_layout()
plt.show()
# %%
# Fit a regression model predicting USD from date, day of week, and
# month of the year. This will help quantify the effects.
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Create a variable for weekend vs weekday, as categorical not int

df = prepare_features(df)

model = ols("revenue_usd_log ~ t + is_weekend + sin_1y_1 + cos_1y_1", data=df).fit()

print(model.summary())

# %%
# Visualise model residuals to check for patterns. Date on the
# x-axis and residuals on the y-axis.
df["residuals"] = model.resid
plt.figure(figsize=(14, 6))
sns.lineplot(data=df, x="date", y="residuals", marker="o")
plt.axhline(0, color="red", linestyle="--")
plt.title("Model Residuals Over Time")
plt.xlabel("Date")
plt.ylabel("Residuals")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Visualise model residuals against day-of-year to check for seasonal patterns.
df["day_of_year"] = df["date"].dt.dayofyear
df["residuals"] = model.resid
plt.figure(figsize=(14, 6))
sns.lineplot(data=df, x="day_of_year", y="residuals", marker="o")
plt.axhline(0, color="red", linestyle="--")
plt.title("Model Residuals Over Time")
plt.xlabel("day-of-year")
plt.ylabel("Residuals")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# %%
# Now, we're going to fit the same model but with a second harmonic

# fit
model2 = ols(
    "revenue_usd_log ~ t + is_weekend + sin_1y_1 + cos_1y_1 + sin_1y_2 + cos_1y_2",
    data=df,
).fit()

print(model2.summary())

# %%
# Visualise model residuals to check for patterns. Date on the
# x-axis and residuals on the y-axis.
df["residuals2"] = model2.resid
plt.figure(figsize=(14, 6))
sns.lineplot(data=df, x="date", y="residuals2", marker="o")
plt.axhline(0, color="red", linestyle="--")
plt.title("Model Residuals Over Time")
plt.xlabel("Date")
plt.ylabel("Residuals")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
df["marketing_c"] = df["marketing_index"] - df["marketing_index"].mean()
df["fxvol_c"] = df["fx_volatility_index"] - df["fx_volatility_index"].mean()

df["marketing_squared"] = df["marketing_c"] ** 2
df["fxvol_squared"] = df["fxvol_c"] ** 2

model3 = ols(
    "revenue_usd_log ~ t + marketing_c + marketing_squared"
    " + fxvol_c + fxvol_squared"
    " + is_weekend + sin_1y_1 + cos_1y_1 + sin_1y_2 + cos_1y_2",
    data=df,
).fit()


print(model3.summary())
# %%
# Visualise model residuals to check for patterns. Date on the
# x-axis and residuals on the y-axis.
df["residuals3"] = model3.resid
plt.figure(figsize=(14, 6))
sns.lineplot(data=df, x="date", y="residuals3", marker="o")
plt.axhline(0, color="red", linestyle="--")
plt.title("Model Residuals Over Time")
plt.xlabel("Date")
plt.ylabel("Residuals")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Creating rolling windows for marketing and fx volatility.
# Marketing 7, 14 and 28 day windows ending at T-1
# Volatility 3, 7, and 14 day windows ending at T-1
df = df.sort_values("date")
for w in [3, 7, 14]:
    df[f"marketing_index_{w}d"] = (
        df["marketing_index"].shift(1).rolling(window=w, min_periods=w).mean()
    )
for w in [7, 14, 28]:
    df[f"fx_volatility_index_{w}d"] = (
        df["fx_volatility_index"].shift(1).rolling(window=w, min_periods=w).mean()
    )

    df["marketing_index_lag1"] = df["marketing_index"].shift(1)
    df["fx_volatility_index_lag1"] = df["fx_volatility_index"].shift(1)
    df["revenue_usd_log_lag1"] = df["revenue_usd_log"].shift(1)
# %%

model4 = ols(
    "revenue_usd_log ~ t +" " + is_weekend + sin_1y_1 + cos_1y_1 + sin_1y_2 + cos_1y_2",
    data=df,
).fit()


print(model4.summary())

# %%
# Now, we plot the residuals from model 3 and model 4 against each other to compare and
# see the effect of removing marketing from the model.

df = df.copy()
df["residuals3"] = model3.resid
df["residuals4"] = model4.resid

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="residuals3", y="residuals4")
plt.axhline(0, color="red", linestyle="--")
plt.axvline(0, color="red", linestyle="--")
plt.title("Model 3 Residuals vs Model 4 Residuals")
plt.xlabel("Model 3 Residuals")
plt.ylabel("Model 4 Residuals")
plt.tight_layout()
plt.show()

# %%
df_long = df.melt(
    id_vars="date",
    value_vars=["residuals3", "residuals4"],
    var_name="model",
    value_name="residual",
)

g = sns.FacetGrid(df_long, row="model", height=3.5, aspect=4, sharex=True, sharey=True)

g.map_dataframe(sns.lineplot, x="date", y="residual", alpha=0.7)
g.map_dataframe(lambda data, **k: plt.axhline(0, color="black", linestyle="--"))

g.set_axis_labels("Date", "Residuals (log revenue)")
g.fig.suptitle("Residuals Over Time: With vs Without Marketing", y=1.02)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
model5 = ols(
    "revenue_usd_log ~ t + fx_volatility_index_14d"
    " + is_weekend + sin_1y_1 + cos_1y_1 + sin_1y_2 + cos_1y_2",
    data=df,
).fit()


print(model5.summary())

# %% [markdown]
# ## Rolling-Origin Time-Series Backtest
#
# This backtest evaluates time-series regression models using:
#
# **Feature Variants:**
# - Multiple window lengths (e.g., fx_volatility_index_7d, 14d, 28d)
#
# **Evaluation Metrics:**
# - MAE (Mean Absolute Error) on fixed forecast horizons
#

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


# ----------------------------
# 2) Helper: bootstrap coefficient stability (within a fold)
# ----------------------------
# basically, for each training fold, we estimate the stabilitiy of coefficient estimates
# by resampling rows with replacement and refitting the model multiple times.
# This gives a sense of how sensitive coefficient estimates are to sampling noise.
def bootstrap_params(train_df, formula, coef_names, n_boot=300, seed=0):
    """
    Bootstrap the regression coefficients by resampling *rows* from the training set.

    Why: This gives a sense of how sensitive coefficient estimates are to sampling noise
         (conditional on the model spec and the training window).

    Returns: dict {coef_name: np.array of bootstrapped estimates}
    """
    rng = np.random.default_rng(seed)
    n = len(train_df)
    out = {c: np.empty(n_boot) for c in coef_names}

    # Fit once to ensure formula is viable on this fold (optional sanity)
    _ = ols(formula, data=train_df).fit()

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)  # sample rows with replacement
        boot_df = train_df.iloc[idx]
        m = ols(formula, data=boot_df).fit()
        for c in coef_names:
            out[c][b] = m.params.get(c, np.nan)
    return out


# ----------------------------
# 3) Core evaluator: compare multiple FX windows
# ----------------------------
def evaluate_fx_windows(
    df,
    fx_feature_cols,  # e.g. ["fx_volatility_index_7d", "fx_volatility_index_14d", ...]
    target_col="revenue_usd_log",
    date_col="date",
    weekend_col="is_weekend",  # bool or 0/1; if bool, statsmodels handles it
    horizon_days=30,
    min_train_days=365,
    step_days=14,
    n_boot=200,  # bootstrap reps per fold (set to 0 to skip)
):
    """
    For each fx feature variant, run rolling-origin CV and compute:
      - MAE across folds
      - coefficient stability:
          * across-fold std of beta_fx and beta_t
          * within-fold bootstrap CI widths (optional)

    Returns:
      - fold_results: per-fold metrics
      - summary: per-feature aggregate metrics
    """
    df = df.sort_values(date_col).copy()

    # Basic time index t (days since start) if not present
    if "t" not in df.columns:
        df["t"] = (df[date_col] - df[date_col].min()).dt.days.astype(float)

    # Ensure seasonal Fourier terms exist (2 harmonics)
    needed = ["sin_1y_1", "cos_1y_1", "sin_1y_2", "cos_1y_2"]
    if not set(needed).issubset(df.columns):
        df = prepare_features(df)

    all_fold_rows = []

    for fx_col in fx_feature_cols:
        # Regression formula for this variant:
        # (You can add/remove controls here as needed; keep fixed across variants)
        formula = (
            f"{target_col} ~ t + {fx_col}"
            f" + {weekend_col} + sin_1y_1 + cos_1y_1 + sin_1y_2 + cos_1y_2"
        )

        # Rolling-origin folds
        splits = list(
            rolling_origin_splits(
                df,
                date_col=date_col,
                min_train_days=min_train_days,
                step_days=step_days,
                horizon_days=horizon_days,
            )
        )

        for fold_i, (train_idx, test_idx, origin) in enumerate(splits):
            train_df = df.loc[train_idx].copy()
            test_df = df.loc[test_idx].copy()

            # Drop rows with NA in any variables required by the model.
            # (Rolling features create NAs early on; this prevents statsmodels errors.)
            model_vars = [
                "t",
                fx_col,
                weekend_col,
                "sin_1y_1",
                "cos_1y_1",
                "sin_1y_2",
                "cos_1y_2",
                target_col,
            ]
            train_df = train_df.dropna(subset=model_vars)
            test_df = test_df.dropna(subset=model_vars)

            # If not enough data after dropping NAs, skip this fold
            if len(train_df) < 30 or len(test_df) < 1:
                continue

            # Fit on training
            m = ols(formula, data=train_df).fit()

            # Predict on test horizon
            pred = m.predict(test_df)
            y_true = test_df[target_col].to_numpy()
            y_pred = np.asarray(pred)

            # MAE on the horizon
            mae = np.mean(np.abs(y_true - y_pred))

            # Record coefficients we care about:
            beta_t = m.params.get("t", np.nan)
            beta_fx = m.params.get(fx_col, np.nan)

            row = {
                "fx_feature": fx_col,
                "fold": fold_i,
                "origin_date": origin,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "mae": mae,
                "beta_t": beta_t,
                "beta_fx": beta_fx,
            }

            # Optional: within-fold bootstrap stability (CI width)
            # Note: This is a sampling-stability check, NOT a time-stability check.
            if n_boot and n_boot > 0:
                boots = bootstrap_params(
                    train_df=train_df,
                    formula=formula,
                    coef_names=["t", fx_col],
                    n_boot=n_boot,
                    seed=fold_i,  # deterministic-ish per fold
                )
                # 90% bootstrap interval width as a stability summary
                bt = boots["t"]
                bfx = boots[fx_col]
                row["boot_ci90_width_t"] = np.nanpercentile(bt, 95) - np.nanpercentile(
                    bt, 5
                )
                row["boot_ci90_width_fx"] = np.nanpercentile(
                    bfx, 95
                ) - np.nanpercentile(bfx, 5)

            all_fold_rows.append(row)

    fold_results = pd.DataFrame(all_fold_rows)

    # ----------------------------
    # 4) Aggregate summaries per FX window
    # ----------------------------
    if fold_results.empty:
        raise ValueError(
            "No folds were evaluated. Check NA handling, date ranges, and window features."
        )

    summary = (
        fold_results.groupby("fx_feature")
        .agg(
            folds=("fold", "count"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            beta_t_mean=("beta_t", "mean"),
            beta_t_std=("beta_t", "std"),  # fold-to-fold stability of trend
            beta_fx_mean=("beta_fx", "mean"),
            beta_fx_std=("beta_fx", "std"),  # fold-to-fold stability of FX elasticity
            boot_ci90_width_t_mean=("boot_ci90_width_t", "mean"),
            boot_ci90_width_fx_mean=("boot_ci90_width_fx", "mean"),
        )
        .reset_index()
        .sort_values("mae_mean")
    )

    return fold_results, summary


# %%
# ----------------------------
# 5) Example usage
# ----------------------------
# Choose the FX variants you want to compare
fx_variants = [
    "fx_volatility_index_7d",
    "fx_volatility_index_14d",
    "fx_volatility_index_28d",
]

# Run evaluation
fold_results, summary = evaluate_fx_windows(
    df=df,
    fx_feature_cols=fx_variants,
    target_col="revenue_usd_log",
    date_col="date",
    weekend_col="is_weekend",
    horizon_days=30,  # evaluate 30-day ahead performance
    min_train_days=365,  # first origin after 1y training history
    step_days=14,  # new origin every 2 weeks
    n_boot=200,  # set 0 to skip bootstrap stability
)

print("=== Summary (lower MAE is better; also check beta stability) ===")
print(summary)

print("\n=== Per-fold sample (useful for plotting MAE over time) ===")
print(fold_results.head())

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


# %%# Run the single-break grid search and plot results
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
    d_break = prepare_features(df, **state)

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

# %%
# Fit the final model. We don't have to specify a model formula, because we've split the X and
# y dataframes already.
import statsmodels.api as sm

model = sm.OLS(y_train, X_train)
results = model.fit()

print(results.summary())
# %%
print(X_train.dtypes)
# %%
