# %% [markdown]
# # Project Analysis Notebook (Python Script w/ Jupyter Cells)
#
# **Goal:** (1–2 sentences)
#
# **Data sources (raw_data/):**
# - ...
#
# **Key questions / hypotheses:**
# 1. ...
# 2. ...
#
# **Notes / decisions log:**
# - YYYY-MM-DD: ...

# %%
# --- Imports & global settings ---
from __future__ import annotations

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

# Seaborn styling
sns.set_theme(style="whitegrid", palette="muted")
sns.set_context("notebook")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("Python:", sys.version)

# %% [markdown]
# ## 0) Paths
#
# Assumes this file lives in: `project/analysis/your_file.py`

# %%
ANALYSIS_DIR = Path(__file__).resolve().parent
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

# %% [markdown]
# ## 1) Load data
#
# Keep this section minimal + reproducible:
# - define file paths
# - read raw data
# - light parsing only (dates/categories)

# %%
# --- Load raw data ---
# Replace with your actual file(s)
# df = pd.read_csv(RAW_DIR / "your_file.csv")

df = pd.read_csv(RAW_DIR / "checkout_ab_synth_v1.csv")  # <- replace

# %%
df.head()

# %% [markdown]
# ## 2) Dataset overview / quick EDA
#
# Goal: understand structure, types, missingness, duplicates, weird values.

# %%
def df_overview(df: pd.DataFrame, n: int = 5) -> None:
    print("Shape:", df.shape)
    display(df.head(n))  # type: ignore
    print("\nDtypes:\n", df.dtypes)
    print("\nMissingness (%):\n", (df.isna().mean() * 100).sort_values(ascending=False).head(30))
    print("\nDuplicate rows:", df.duplicated().sum())

# %%
df_overview(df)

# %%
def col_profile(df: pd.DataFrame, max_unique: int = 50) -> pd.DataFrame:
    out = []
    for c in df.columns:
        s = df[c]
        n_unique = int(s.nunique(dropna=True))
        out.append(
            {
                "col": c,
                "dtype": str(s.dtype),
                "n": int(s.shape[0]),
                "n_missing": int(s.isna().sum()),
                "missing_pct": round(float(s.isna().mean() * 100), 2),
                "n_unique": n_unique,
                "example_values": s.dropna().unique()[:5].tolist() if n_unique <= max_unique else [],
            }
        )
    return pd.DataFrame(out).sort_values(["missing_pct", "n_unique"], ascending=[False, False])

# %%
profile = col_profile(df)
profile.head(30)

# %% [markdown]
# ### Missingness Visualization

# %%
def plot_missingness(df: pd.DataFrame, top_n: int = 30) -> None:
    """
    Visualize missing data patterns.
    Shows bar chart of missingness % for columns with any missing values.
    """
    missing = df.isna().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False).head(top_n)
    
    if missing.empty:
        print("No missing values detected in dataset.")
        return
    
    plt.figure(figsize=(10, max(6, len(missing) * 0.3)))
    sns.barplot(x=missing.values, y=missing.index, orient="h", palette="rocket")
    plt.xlabel("Missing %")
    plt.ylabel("Column")
    plt.title(f"Missing Data by Column (top {len(missing)})")
    plt.tight_layout()
    plt.show()

# %%
plot_missingness(df)
# save_fig("missingness_overview")

# %%
def plot_missingness_heatmap(df: pd.DataFrame, sample_n: Optional[int] = None) -> None:
    """
    Heatmap showing missing data patterns across rows.
    Useful for identifying systematic missingness.
    
    Args:
        df: Input dataframe
        sample_n: If provided, randomly sample this many rows for visualization
    """
    cols_with_missing = df.columns[df.isna().any()].tolist()
    
    if not cols_with_missing:
        print("No missing values detected in dataset.")
        return
    
    plot_df = df[cols_with_missing]
    
    if sample_n and len(plot_df) > sample_n:
        plot_df = plot_df.sample(n=sample_n, random_state=RANDOM_SEED)
    
    plt.figure(figsize=(min(12, len(cols_with_missing) * 0.5), 8))
    sns.heatmap(plot_df.isna(), cbar=True, yticklabels=False, cmap="viridis")
    plt.title("Missing Data Pattern (Yellow = Missing)")
    plt.xlabel("Column")
    plt.ylabel("Row Index")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# %%
plot_missingness_heatmap(df, sample_n=1000)
# save_fig("missingness_heatmap")

# %%
def missingness_by_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    cols_with_missing = df.columns[df.isna().any()].tolist()

    if not cols_with_missing:
        return pd.DataFrame()

    records = []
    for g, gdf in df.groupby(group):
        miss_pct = gdf[cols_with_missing].isna().mean() * 100
        for col, pct in miss_pct.items():
            records.append(
                {
                    "group": g,
                    "column": col,
                    "missing_pct": pct,
                }
            )

    return pd.DataFrame(records)

def plot_missingness_groups(df: pd.DataFrame, group: str) -> None:
    plot_df = missingness_by_group(df, group)

    if plot_df.empty:
        print("No missing values detected in dataset.")
        return

    plt.figure(figsize=(min(14, plot_df["column"].nunique() * 0.6), 6))
    sns.barplot(
        data=plot_df,
        x="column",
        y="missing_pct",
        hue="group",
        palette="viridis",
    )
    plt.title("Missing Data Percentage by Group")
    plt.xlabel("Column")
    plt.ylabel("Missing %")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# %%
plot_missingness_groups(df, group = 'group')
# save_fig("missingness_heatmap")

# %%
df['missing_checkout'] = df['checkout_seconds'].isna()

# Start with a t-test, but it's not actually appropriate for a 0-1 binary output. 
from scipy import stats
stats.ttest_ind(
    df.loc[df['group'] == 'control', 'missing_checkout'],
    df.loc[df['group'] == 'treatment', 'missing_checkout'],
    equal_var = False
)

# %% 
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

def logit_with_marginal_effects(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    reference: str,
    compare: str | None = None,
) -> dict:
    """
    Fits logistic regression outcome ~ C(group, Treatment(reference=...)),
    returns model + marginal effects + nice printable summary stats.

    outcome: binary 0/1
    group: categorical with >=2 levels
    reference: baseline group label
    compare: optional comparison group label (e.g., "treatment"). If None and
             there are exactly 2 groups, it will infer the non-reference group.
    """
    d = df[[outcome, group]].dropna().copy()

    # Ensure outcome is 0/1 numeric
    d[outcome] = d[outcome].astype(int)

    levels = pd.unique(d[group])
    if reference not in levels:
        raise ValueError(f"reference='{reference}' not found in {group} levels: {levels}")

    # Choose compare group
    if compare is None:
        other_levels = [x for x in levels if x != reference]
        if len(other_levels) != 1:
            raise ValueError(
                f"compare not provided and found {len(other_levels)} non-reference levels: {other_levels}. "
                "Pass compare= explicitly."
            )
        compare = other_levels[0]

    # Fit model
    formula = f"{outcome} ~ C({group}, Treatment(reference='{reference}'))"
    model = smf.logit(formula, data=d).fit(disp=False)

    # Average marginal effects (AME)
    mfx = model.get_margeff(at="overall")
    mfx_df = mfx.summary_frame()

    # Predicted probabilities for each group (holding nothing fixed; just set group)
    def pred_prob_for(level: str) -> float:
        tmp = pd.DataFrame({group: [level]})
        return float(model.predict(tmp)[0])

    p_ref = pred_prob_for(reference)
    p_cmp = pred_prob_for(compare)

    # Absolute difference in probability (percentage points)
    diff_pp = (p_cmp - p_ref) * 100

    # Relative difference vs baseline probability (careful: not AME; just a readable relative lift)
    rel_pct = (p_cmp / p_ref - 1) * 100 if p_ref > 0 else np.nan

    # Odds ratio + CI for the compare coefficient (if present)
    coef_name = f"C({group}, Treatment(reference='{reference}'))[T.{compare}]"
    if coef_name in model.params.index:
        log_odds = model.params[coef_name]
        or_ = float(np.exp(log_odds))
        ci = model.conf_int().loc[coef_name].to_numpy()
        or_ci = tuple(np.exp(ci))
        p_value = float(model.pvalues[coef_name])
    else:
        # This happens if compare is not represented as a single dummy (rare edge cases)
        or_, or_ci, p_value = np.nan, (np.nan, np.nan), np.nan

    return {
        "data": d,
        "model": model,
        "marginal_effects": mfx,
        "marginal_effects_table": mfx_df,
        "reference": reference,
        "compare": compare,
        "p_ref": p_ref,
        "p_cmp": p_cmp,
        "diff_pp": diff_pp,
        "rel_pct": rel_pct,
        "odds_ratio": or_,
        "odds_ratio_ci": or_ci,
        "p_value": p_value,
        "coef_name": coef_name,
    }


def print_logit_summary(result: dict, outcome_label: str | None = None) -> None:
    ref = result["reference"]
    cmp = result["compare"]
    p_ref = result["p_ref"]
    p_cmp = result["p_cmp"]
    diff_pp = result["diff_pp"]
    rel_pct = result["rel_pct"]
    or_ = result["odds_ratio"]
    or_lo, or_hi = result["odds_ratio_ci"]
    pval = result["p_value"]
    coef_name = result["coef_name"]

    outcome_text = outcome_label or "the outcome"

    # Marginal-effects table row for the group dummy if it exists
    mfx_tbl = result["marginal_effects_table"]
    me_text = ""
    if coef_name in mfx_tbl.index:
        me = float(mfx_tbl.loc[coef_name, "dy/dx"]) * 100  # percentage points
        me_se = float(mfx_tbl.loc[coef_name, "Std. Err."]) * 100
        me_p = float(mfx_tbl.loc[coef_name, "Pr(>|z|)"])
        me_text = f"AME: {me:+.2f} pp (SE {me_se:.2f}, p={me_p:.3g})."
    else:
        me_text = "AME: (could not find group contrast row in marginal effects table)."

    print(f"\nLogistic regression: {cmp} vs {ref}")
    print(f"Predicted P({outcome_text} | {ref}) = {p_ref:.3%}")
    print(f"Predicted P({outcome_text} | {cmp}) = {p_cmp:.3%}")

    # Statistically safest headline statement
    print(f"\nHeadline (probability): {cmp} is {diff_pp:+.2f} percentage points "
          f"{'higher' if diff_pp >= 0 else 'lower'} than {ref}.")

    # Optional “% more likely” relative phrasing (use with care)
    if np.isfinite(rel_pct):
        print(f"Relative phrasing: {cmp} is {rel_pct:+.1f}% "
              f"{'more' if rel_pct >= 0 else 'less'} likely than {ref} "
              f"(relative to {ref}’s baseline probability).")

    print(f"\nOdds ratio (logit coef): OR={or_:.3f} (95% CI {or_lo:.3f}–{or_hi:.3f}), p={pval:.3g}")
    print(me_text)

# %%
res = logit_with_marginal_effects(
    df,
    outcome="missing_checkout",
    group="group",
    reference="control",
    compare="treatment"   # optional if exactly 2 groups
)
 
# res['marginal_effects_table'].head()
print_logit_summary(res, outcome_label="missing_checkout")

# No evidence of differential missingness between groups.

# %% [markdown]
# ## 3) Univariate exploration
#
# - Numeric distributions
# - Categorical breakdowns

# %%
def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return pd.DataFrame()
    return num.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T

# %%
numeric_summary(df).head(30)

# %%
def plot_numeric_hist(df: pd.DataFrame, col: str, bins: int = 40, kde: bool = True) -> None:
    """
    Plot histogram with optional KDE overlay using seaborn.
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x=col, bins=bins, kde=kde, stat="count")
    plt.title(f"Distribution: {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# %%
plot_numeric_hist(df, "checkout_seconds")
save_fig("some_numeric_col_distribution")

# %%
def value_counts(df: pd.DataFrame, col: str, n: int = 20) -> pd.DataFrame:
    vc = df[col].value_counts(dropna=False).head(n)
    return vc.rename_axis(col).reset_index(name="count")

# %%
value_counts(df, "some_category_col", n=30)

# %%
def plot_categorical_counts(df: pd.DataFrame, col: str, top_n: int = 20, horizontal: bool = True) -> None:
    """
    Bar plot of categorical value counts using seaborn.
    """
    vc = df[col].value_counts(dropna=False).head(top_n)
    
    if horizontal:
        plt.figure(figsize=(10, max(6, len(vc) * 0.4)))
        sns.barplot(x=vc.values, y=vc.index, orient="h", palette="viridis")
        plt.xlabel("Count")
        plt.ylabel(col)
    else:
        plt.figure(figsize=(max(10, len(vc) * 0.5), 6))
        sns.barplot(x=vc.index, y=vc.values, palette="viridis")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
    
    plt.title(f"Top {len(vc)} Values: {col}")
    plt.tight_layout()
    plt.show()

# %%
plot_categorical_counts(df, "group", top_n=15)
save_fig("some_category_col_counts")

# %% [markdown]
# ### Outlier Detection

# %%
def detect_outliers_iqr(df: pd.DataFrame, col: str, multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR method.
    
    Args:
        df: Input dataframe
        col: Column name to check
        multiplier: IQR multiplier (1.5 = standard, 3.0 = extreme)
    
    Returns:
        Boolean series indicating outliers
    """
    s = df[col].dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
    
    print(f"Column: {col}")
    print(f"  Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
    print(f"  Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
    print(f"  Outliers detected: {outliers.sum()} ({outliers.mean()*100:.2f}%)")
    
    return outliers

# %%
outliers = detect_outliers_iqr(df, "checkout_seconds", multiplier = 3)
df[outliers].head()

# %%
def detect_outliers_zscore(df: pd.DataFrame, col: str, threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers using z-score method.
    
    Args:
        df: Input dataframe
        col: Column name to check
        threshold: Z-score threshold (typically 2.5 or 3.0)
    
    Returns:
        Boolean series indicating outliers
    """
    s = df[col].dropna()
    mean = s.mean()
    std = s.std()
    
    z_scores = np.abs((df[col] - mean) / std)
    outliers = z_scores > threshold
    
    print(f"Column: {col}")
    print(f"  Mean: {mean:.2f}, Std: {std:.2f}")
    print(f"  Threshold: {threshold} standard deviations")
    print(f"  Outliers detected: {outliers.sum()} ({outliers.mean()*100:.2f}%)")
    
    return outliers

# %%
df['checkout_seconds_log'] = np.log1p(df['checkout_seconds'])
outliers_z = detect_outliers_zscore(df, "checkout_seconds_log", threshold=3.0)
df[outliers_z].sort_values('checkout_seconds_log', ascending = False).head()

# %%
def plot_outliers_boxplot(df: pd.DataFrame, col: str, by: Optional[str] = None, top_n: int = 15) -> None:
    """
    Boxplot to visualize outliers using seaborn.
    
    Args:
        df: Input dataframe
        col: Numeric column to plot
        by: Optional categorical column to group by
        top_n: If 'by' is provided, show only top N categories
    """
    if by:
        top_categories = df[by].value_counts().head(top_n).index
        plot_df = df[df[by].isin(top_categories)]
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=plot_df, y=by, x=col, orient="h", palette="Set2")
        plt.title(f"Outlier Detection: {col} by {by}")
        plt.xlabel(col)
        plt.ylabel(by)
    else:
        plt.figure(figsize=(10, 4))
        sns.boxplot(data=df, x=col, palette="Set2")
        plt.title(f"Outlier Detection: {col}")
        plt.xlabel(col)
    
    plt.tight_layout()
    plt.show()

# %%
plot_outliers_boxplot(df, "checkout_seconds", by = 'group')
save_fig("outliers by group")

 # %%
# plot_outliers_boxplot(df_clean, "some_numeric_col", by="category", top_n=10)
# save_fig("some_numeric_col_outliers_by_category")

# %%
def plot_outliers_scatter(df: pd.DataFrame, x_col: str, y_col: str, highlight_col: Optional[str] = None) -> None:
    """
    Scatter plot to identify multivariate outliers using seaborn.
    
    Args:
        df: Input dataframe
        x_col: X-axis column
        y_col: Y-axis column
        highlight_col: Optional column to color points by
    """
    plt.figure(figsize=(10, 6))
    
    if highlight_col:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=highlight_col, alpha=0.6, s=50)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col, alpha=0.6, s=50)
    
    plt.title(f"Scatter: {x_col} vs {y_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.show()

# %%
# plot_outliers_scatter(df_clean, "feature_1", "feature_2", highlight_col="category")
# save_fig("feature_1_vs_feature_2_scatter")

# %%
def outlier_summary(df: pd.DataFrame, method: str = "iqr", multiplier: float = 1.5, threshold: float = 3.0) -> pd.DataFrame:
    """
    Generate summary of outliers across all numeric columns.
    
    Args:
        df: Input dataframe
        method: "iqr" or "zscore"
        multiplier: IQR multiplier (if method="iqr")
        threshold: Z-score threshold (if method="zscore")
    
    Returns:
        DataFrame with outlier counts and percentages per column
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    results = []
    
    for col in numeric_cols:
        if method == "iqr":
            s = df[col].dropna()
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            outliers = (df[col] < lower) | (df[col] > upper)
        else:  # zscore
            s = df[col].dropna()
            z_scores = np.abs((df[col] - s.mean()) / s.std())
            outliers = z_scores > threshold
        
        results.append({
            "column": col,
            "n_outliers": int(outliers.sum()),
            "pct_outliers": float(outliers.mean() * 100),
            "method": method,
        })
    
    return pd.DataFrame(results).sort_values("n_outliers", ascending=False)

# %%
# outlier_summary(df_clean, method="iqr", multiplier=1.5)

# %% [markdown]
# ## 4) Bivariate / multivariate exploration
#
# - Relationships, correlations, group comparisons, time trends

# %%
def corr_table(df: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return pd.DataFrame()
    return num.corr(method=method)

# %%
# corr_table(df)

# %%
def plot_correlation_heatmap(df: pd.DataFrame, method: str = "spearman", annot: bool = True, cmap: str = "coolwarm") -> None:
    """
    Correlation heatmap using seaborn.
    """
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        print("Need at least 2 numeric columns for correlation.")
        return
    
    corr = num.corr(method=method)
    
    plt.figure(figsize=(max(10, len(corr.columns) * 0.7), max(8, len(corr.columns) * 0.6)))
    sns.heatmap(corr, annot=annot, fmt=".2f", cmap=cmap, center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f"Correlation Matrix ({method.capitalize()})")
    plt.tight_layout()
    plt.show()

# %%
# plot_correlation_heatmap(df)
# save_fig("correlation_heatmap")

# %%
def group_summary(df: pd.DataFrame, group_col: str, target_col: str) -> pd.DataFrame:
    return (
        df.groupby(group_col)[target_col]
        .agg(["count", "mean", "median", "std"])
        .sort_values("count", ascending=False)
        .reset_index()
    )

# %%
group_summary(df, "group", "checkout_seconds_log").head(20)

# %% [markdown]
# ## 5) Cleaning & validation
#
# Keep cleaning explicit. Add lightweight assertions after major steps.

# %%
df_clean = df  # <- replace with real cleaning steps

# Examples:
# df_clean = df_clean.drop_duplicates()
# df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")
# df_clean["category"] = df_clean["category"].astype("category")

# %%
def validate(df: pd.DataFrame) -> None:
    # Edit these to match your dataset’s invariants
    # assert df["id"].notna().all(), "Missing ids found"
    # assert (df["amount"] >= 0).all(), "Negative amounts detected"
    pass

# %%
# validate(df_clean)

# %% [markdown]
# ## 6) Feature engineering
#
# Produce an analysis/modelling table, ideally separate from raw columns once stable.

# %%
df_feat = df_clean  # <- transform into features

# Examples:
df_feat["log_checkout_seconds"] = np.log1p(df_feat["checkout_seconds"])
# df_feat["is_weekend"] = df_feat["date"].dt.dayofweek >= 5

# %% [markdown]
# ## 7) Plotting helpers + saving outputs
#
# You said no separate figures folder, so we save images into `output/`.

# %%
def save_fig(name: str) -> Path:
    path = OUTPUT_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print("Saved:", path)
    return path

# %%
def boxplot_by_group(df: pd.DataFrame, group_col: str, value_col: str, top_n: int = 15, showfliers: bool = True) -> None:
    """
    Grouped boxplot using seaborn.
    """
    top = df[group_col].value_counts().head(top_n).index
    plot_df = df[df[group_col].isin(top)][[group_col, value_col]].dropna()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=plot_df, y=group_col, x=value_col, orient="h", palette="Set3", showfliers=showfliers)
    plt.title(f"{value_col} by {group_col} (top {top_n})")
    plt.xlabel(value_col)
    plt.ylabel(group_col)
    plt.tight_layout()
    plt.show()

# %%
# boxplot_by_group(df_feat, "segment", "amount", showfliers=False)
# save_fig("amount_by_segment_boxplot")

# %%
def plot_scatter_with_regression(df: pd.DataFrame, x_col: str, y_col: str, hue: Optional[str] = None) -> None:
    """
    Scatter plot with regression line using seaborn.
    """
    plt.figure(figsize=(10, 6))
    
    if hue:
        sns.lmplot(data=df, x=x_col, y=y_col, hue=hue, height=6, aspect=1.5, scatter_kws={"alpha": 0.5})
    else:
        sns.regplot(data=df, x=x_col, y=y_col, scatter_kws={"alpha": 0.5})
        plt.title(f"{y_col} vs {x_col}")
    
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.show()

# %%
# plot_scatter_with_regression(df_feat, "feature_1", "feature_2")
# save_fig("feature_1_vs_feature_2_regression")

# %% [markdown]
# ## 8) Analysis / modelling
#
# Put the “real work” here: metrics, hypothesis tests, baselines, models.

# %%

model = smf.ols(
    "log_checkout_seconds ~ C(group, Treatment(reference='control'))",
    data=df_feat
).fit(cov_type="HC3")

model.summary()

# %%

def bootstrap_median_diff(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    control_label: str = "control",
    treatment_label: str = "treatment",
    n_boot: int = 20_000,
    ci: float = 0.95,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    x = df.loc[df[group] == control_label, outcome].dropna().to_numpy()
    y = df.loc[df[group] == treatment_label, outcome].dropna().to_numpy()

    if len(x) < 2 or len(y) < 2:
        raise ValueError("Need at least 2 observations per group after dropping NA.")

    # Observed statistic
    delta_obs = np.median(y) - np.median(x)

    # Bootstrap resamples (resample within each group)
    deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        deltas[b] = np.median(yb) - np.median(xb)

    alpha = 1 - ci
    lo = np.quantile(deltas, alpha / 2)
    hi = np.quantile(deltas, 1 - alpha / 2)

    # Two-sided p-value: how often bootstrap deltas are as or more extreme than 0
    # (common heuristic; see note below)
    p_two_sided = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())

    return {
        "delta_obs": float(delta_obs),
        "ci_level": ci,
        "ci": (float(lo), float(hi)),
        "p_two_sided": float(p_two_sided),
        "control_median": float(np.median(x)),
        "treatment_median": float(np.median(y)),
        "n_control": int(len(x)),
        "n_treatment": int(len(y)),
    }

res = bootstrap_median_diff(
    df,
    outcome="checkout_seconds",
    group="group",
    control_label="control",
    treatment_label="treatment",
)

print(
    f"Median(control) = {res['control_median']:.2f}, "
    f"Median(treatment) = {res['treatment_median']:.2f}\n"
    f"Median diff (treat - control) = {res['delta_obs']:.2f} seconds\n"
    f"{int(res['ci_level']*100)}% CI: [{res['ci'][0]:.2f}, {res['ci'][1]:.2f}]\n"
    f"Approx p-value (two-sided): {res['p_two_sided']:.4f}"
)



# %% [markdown]
# ## 9) Save artefacts
#
# Save cleaned/feature data + key outputs to `clean_data/` and `output/`.

# %%
# Examples:
# df_clean.to_parquet(CLEAN_DIR / "df_clean.parquet", index=False)
# df_feat.to_parquet(CLEAN_DIR / "df_features.parquet", index=False)

# %% [markdown]
# ## 10) Writeup / conclusions
#
# **What did you find?**
# - ...
#
# **So what?**
# - ...
#
# **Limitations / next steps**
# - ...
#
# **Actions / recommendations**
# - ...

