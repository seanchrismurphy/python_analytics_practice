# %% [markdown]
# # A/B Test Analysis: Conversion with Potential Heterogeneous Effects
#
# This notebook analyzes a randomized A/B experiment measuring the impact of a treatment on user conversion.
#
# Goals:
# - Estimate the average treatment effect (ATE)
# - Check balance, missingness, and data sanity
# - Use both unadjusted and model-based approaches
# - Explore potential heterogeneous treatment effects (HTE)
# - Make a ship/no-ship recommendation with caveats

# %% [markdown]
# ## 0) Imports, settings, paths

# %%
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

# %% [markdown]
# ## 1) Background & Context
#
# A product experiment was run in which users were randomly assigned to either control or treatment.
#
# Outcome:
# - `converted` (binary 0/1)
#
# Covariates:
# - `tenure_days` (numeric; some missing)
# - `device` (mobile/desktop)
# - `region` (NA/EU/APAC)
#
# Objective:
# - Determine whether treatment changes conversion
# - Assess whether effects differ across segments (HTE)
# - Provide a ship/no-ship recommendation

# %% [markdown]
# ## 2) Load data

# %%
df = pd.read_csv(RAW_DIR / "ab_conversion_hte_v1.csv")  # <- replace if needed
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
    print("\nMissingness (%):\n", (df.isna().mean() * 100).sort_values(ascending=False).head(30))
    print("\nDuplicate rows:", df.duplicated().sum())

df_overview(df)

# %% 
df.isna().mean(axis = 0)

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

profile = col_profile(df)
profile.head(30)

# %% 
df['device'].value_counts()

# %%
# Quick group sizing & conversion sanity check
# First pass, it doesn't look like there's been any treatment effects. 
(df.groupby("treatment")["converted"]
   .agg(n="count", conv_rate="mean")
   .assign(conv_rate=lambda x: (x["conv_rate"] * 100).round(2))
)

# %% [markdown]
# ## 3.5) Exploratory tests / helper functions (keep utilities here)
#
# Use this section for any helper functions used across multiple later sections:
# - plotting
# - missingness checks
# - proportion tests
# - logistic regression helpers
# - bootstrap CIs

# %%
def save_fig(name: str) -> Path:
    path = OUTPUT_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print("Saved:", path)
    return path

# %%
def plot_missingness(df: pd.DataFrame, top_n: int = 30) -> None:
    missing = (df.isna().mean() * 100)
    missing = missing[missing > 0].sort_values(ascending=False).head(top_n)

    if missing.empty:
        print("No missing values detected.")
        return

    plt.figure(figsize=(10, max(6, len(missing) * 0.3)))
    sns.barplot(x=missing.values, y=missing.index, orient="h")
    plt.xlabel("Missing %")
    plt.ylabel("Column")
    plt.title(f"Missing Data by Column (top {len(missing)})")
    plt.tight_layout()
    plt.show()

# %%
def missingness_by_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    cols_with_missing = df.columns[df.isna().any()].tolist()
    if not cols_with_missing:
        return pd.DataFrame()

    records = []
    for g, gdf in df.groupby(group):
        miss_pct = gdf[cols_with_missing].isna().mean() * 100
        for col, pct in miss_pct.items():
            records.append({"group": g, "column": col, "missing_pct": float(pct)})
    return pd.DataFrame(records)

# %%
def plot_missingness_groups(df: pd.DataFrame, group: str) -> None:
    plot_df = missingness_by_group(df, group)
    if plot_df.empty:
        print("No missing values detected.")
        return

    plt.figure(figsize=(min(14, plot_df["column"].nunique() * 0.8), 6))
    sns.barplot(data=plot_df, x="column", y="missing_pct", hue="group")
    plt.title("Missing Data Percentage by Group")
    plt.xlabel("Column")
    plt.ylabel("Missing %")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# %%
def diff_in_proportions_wald_ci(p1: float, p0: float, n1: int, n0: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Wald CI for difference in proportions: (p1 - p0) ± z * sqrt(p1(1-p1)/n1 + p0(1-p0)/n0)
    """
    from scipy.stats import norm

    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    z = norm.ppf(1 - alpha / 2)
    diff = p1 - p0
    return float(diff - z * se), float(diff + z * se)

# %%
def ztest_proportions(success1: int, n1: int, success0: int, n0: int) -> float:
    """
    Two-sided z-test for difference in proportions (large-sample).
    Returns p-value.
    """
    from scipy.stats import norm

    p1 = success1 / n1
    p0 = success0 / n0
    p_pool = (success1 + success0) / (n1 + n0)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n0))
    z = (p1 - p0) / se
    pval = 2 * (1 - norm.cdf(abs(z)))
    return float(pval)

# %%
def bootstrap_diff_in_means(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Bootstrap CI for mean difference (y - x).
    """
    rng = np.random.default_rng(seed)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    deltas = np.empty(n_boot)
    for b in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        deltas[b] = yb.mean() - xb.mean()

    alpha = 1 - ci
    lo = np.quantile(deltas, alpha / 2)
    hi = np.quantile(deltas, 1 - alpha / 2)

    return {"delta_obs": float(y.mean() - x.mean()), "ci": (float(lo), float(hi))}

# %%
def prep_features_for_logit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal, explicit feature prep for modeling.
    - Keep it simple: one-hot encode categoricals, handle missing tenure.
    You can iterate later.
    """
    d = df.copy()

    # Example missingness handling: add indicator + impute with median
    d["tenure_missing"] = d["tenure_days"].isna().astype(int)
    d["tenure_days_imputed"] = d["tenure_days"].fillna(d["tenure_days"].median())

    # One-hot encode categoricals
    d = pd.get_dummies(d, columns=["device", "region"], drop_first=True)

    return d

# %% 
# Missingness doesn't appear to meaningfully differ by treatment group
plot_missingness_groups(df, 'treatment')

# %% [markdown]
# ## Basic data cleaning 
# Goal:
# - Handle missingness (if needed)
# - Prepare features for modeling

# %%
df_clean = df.copy()
# Example: impute tenure_days missingness with median + indicator
df_clean["tenure_missing"] = df_clean["tenure_days"].isna().astype(int)
df_clean["tenure_days_imputed"] = df_clean["tenure_days"].fillna(df_clean["tenure_days"].median())
# Convert string variables to categorical 
df_clean["device"] = df_clean["device"].astype("category")
df_clean["region"] = df_clean["region"].astype("category")
df_clean["region"] = df_clean["region"].cat.add_categories("Missing")
df_clean["region"] = df_clean["region"].fillna("Missing")

# %% [markdown]
# ## 4) Baseline A/B Readout (Unadjusted)
#
# Goal:
# - Conversion rate by group
# - Absolute lift + relative lift
# - p-value + CI for difference in proportions

# %% 
import statsmodels.formula.api as smf

model = smf.logit(
    "converted ~ C(treatment, Treatment(reference=0))",
    data=df
).fit()

model.summary()

# %%
# Get the predicted probabilities for each group (probability went from X to Y)
# This is the absolute conversion rate in each group according to the model
pred = (
    df.assign(pred=model.predict(df))
      .groupby("treatment")["pred"]
      .mean() 
      .round(4) * 100 
)

pred

# %%
# Get the marginal effects (treatment effect in pp)
# This shows us the percentage change. 
coef_name = "C(treatment, Treatment(reference=0))[T.1]"
mfx = model.get_margeff(at="overall")
mfx.summary()

mfx_df = mfx.summary_frame()

me = mfx_df.loc[coef_name, "dy/dx"]
se_me = mfx_df.loc[coef_name, "Std. Err."]
ci_lo = mfx_df.loc[coef_name, "Conf. Int. Low"]
ci_hi = mfx_df.loc[coef_name, "Cont. Int. Hi."]
p_me = mfx_df.loc[coef_name, "Pr(>|z|)"]

print(
    f"Treatment effect: {me*100:+.2f} pp "
    f"(95% CI {ci_lo*100:+.2f} to {ci_hi*100:+.2f}), "
    f"p={p_me:.4f}"
)


# %%
# Fill in analysis here (use helper functions above)
# - table of conversion rates
# - compute absolute/relative lift
# - z-test p-value
# - CI for absolute difference

# %% [markdown]
# ## 5) Model-Based Analysis
#
# Goal:
# - Logistic regression estimating treatment effect
# - Interpret effect in business terms (probability / lift)
# - Compare to unadjusted result

# %%
# Logistic regression with formula interface and robust SEs
import statsmodels.formula.api as smf


def fit_logit_formula(
    df: pd.DataFrame,
    formula: str,
    treatment_var: str = "treatment",
    robust_se: str | None = "HC3",
    dropna: bool = True,
) -> dict:
    """
    Fit logistic regression using a statsmodels formula, with optional robust SEs.

    Parameters
    ----------
    df : DataFrame
    formula : str
        e.g. "converted ~ treatment + tenure_days + C(device) + C(region)"
        interactions: "converted ~ treatment * C(device)" or "treatment:C(device)"
    treatment_var : str
        Name of the 0/1 treatment variable used for summary outputs.
    robust_se : str | None
        e.g. "HC3", "HC1", "HC0" or None for default (model-based) SEs.
    dropna : bool
        If True, drops rows with NA in any variables used in formula.

    Returns
    -------
    dict with keys: model, model_robust, data_used, formula, robust_cov, etc.
    """
    # Build design via formula; statsmodels will handle categoricals with C(...)
    d = df.copy()
    if dropna:
        # statsmodels will drop NAs internally too, but we keep the actual used df for reporting
        d = d.dropna(subset=_vars_in_formula(formula))

    if robust_se:
        model = smf.logit(formula, data=d).fit(
        disp=False,
        cov_type=robust_se
    )
    else:
        model = smf.logit(formula, data=d).fit(disp=False)

    model_rob = model
    # AME (may fail for some specifications; keep optional)
    ame_table = None
    try:
        mfx = model.get_margeff(at="overall")
        ame_table = mfx.summary_frame()
    except Exception:
        pass

    return {
        "data_used": d,
        "formula": formula,
        "model": model,
        "model_robust": model_rob,
        "robust_cov": robust_se,
        "treatment_var": treatment_var,
        "ame_table": ame_table,
        "n": int(model.nobs),
    }


def summarize_treatment_from_formula(
    res: dict,
    treatment_var: str | None = None,
    label: str = "conversion",
    at: str = "mean",
    control_value: int = 0,
    treatment_value: int = 1,
) -> dict:
    """
    Produce a readable treatment summary from a fitted formula logit model.

    - Odds ratio + CI for the *main* treatment coefficient (if present)
    - Predicted P(outcome) for control vs treatment at a reference covariate profile

    Notes
    -----
    With interactions, the "main effect" coefficient for treatment is the effect
    at the *reference levels* of categorical variables (and at zero for centered numerics).
    The predicted-probability lift computed here also uses that same reference profile
    (by default: mean numeric covariates + modal categorical levels).
    """
    model = res["model"]
    model_rob = res["model_robust"]
    d = res["data_used"]

    treatment_var = treatment_var or res.get("treatment_var", "treatment")

    # Build a baseline "reference row" for prediction
    x_ref = _make_reference_row_for_formula(model, d, treatment_var, at=at)
    x0 = x_ref.copy()
    x1 = x_ref.copy()
    x0[treatment_var] = control_value
    x1[treatment_var] = treatment_value

    p0 = float(model.predict(x0)[0])
    p1 = float(model.predict(x1)[0])
    diff_pp = (p1 - p0) * 100
    rel_pct = (p1 / p0 - 1) * 100 if p0 > 0 else np.nan

    # OR/CI for main treatment term, if it exists in params
    if treatment_var in model_rob.params.index:
        log_odds = float(model_rob.params[treatment_var])
        se = float(model_rob.bse[treatment_var])
        pval = float(model_rob.pvalues[treatment_var])

        or_ = float(np.exp(log_odds))
        ci_lo = float(np.exp(log_odds - 1.96 * se))
        ci_hi = float(np.exp(log_odds + 1.96 * se))
    else:
        # If you used something like C(treatment) or renamed it, you can pass treatment_var accordingly
        or_, ci_lo, ci_hi, pval = np.nan, np.nan, np.nan, np.nan

    summary = {
        "n": res["n"],
        "robust_cov": res["robust_cov"],
        "formula": res["formula"],
        "p_control": p0,
        "p_treatment": p1,
        "diff_pp": float(diff_pp),
        "rel_pct": float(rel_pct),
        "odds_ratio_main": or_,
        "odds_ratio_ci_main": (ci_lo, ci_hi),
        "p_value_main": pval,
        "reference_row": x_ref,
        "note": (
            "With interactions, the 'main' treatment coefficient is conditional on reference levels; "
            "prefer reporting predicted lifts by segment."
        ),
    }
    return summary


def print_treatment_formula_summary(s: dict, label: str = "conversion") -> None:
    print("\nLogit (formula) treatment summary")
    print("-" * 36)
    print(f"N used: {s['n']}")
    if s["robust_cov"]:
        print(f"SEs: robust ({s['robust_cov']})")
    else:
        print("SEs: standard")
    print(f"Formula: {s['formula']}")

    print(f"\nPredicted P({label}) at reference profile:")
    print(f"  Control:   {s['p_control']:.3%}")
    print(f"  Treatment: {s['p_treatment']:.3%}")

    direction = "higher" if s["diff_pp"] >= 0 else "lower"
    print(f"\nLift (Treatment - Control): {s['diff_pp']:+.2f} pp ({direction})")
    if np.isfinite(s["rel_pct"]):
        print(f"Relative lift: {s['rel_pct']:+.1f}%")

    lo, hi = s["odds_ratio_ci_main"]
    if np.isfinite(s["odds_ratio_main"]):
        print(f"\nMain-effect OR: {s['odds_ratio_main']:.3f} (95% CI {lo:.3f}–{hi:.3f}), p={s['p_value_main']:.3g}")
        print("Note:", s["note"])
    else:
        print("\nMain-effect OR: (treatment main coefficient not found; check treatment_var / formula encoding)")
        print("Note:", s["note"])


# --- internal helpers ---

def _vars_in_formula(formula: str) -> list[str]:
    """
    Best-effort parse of variable tokens in a formula for dropna.
    Handles 'C(var)' and basic operators. Not a full Patsy parser.
    """
    import re
    rhs = formula.split("~", 1)[1]
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rhs)
    # Remove reserved words from patsy / pythonish tokens
    reserved = {"C", "T", "I", "np", "log", "exp"}
    out = [t for t in tokens if t not in reserved]
    # Include outcome too
    lhs = formula.split("~", 1)[0].strip()
    return [lhs] + sorted(set(out))


def _make_reference_row_for_formula(model, df: pd.DataFrame, treatment_var: str, at: str = "mean") -> pd.DataFrame:
    """
    Create a single-row DataFrame with values for all variables used in the formula.
    - numeric: mean (or median)
    - categorical/object: mode
    - treatment: left as-is; caller overwrites to 0/1
    """
    # Pull variable names from the dataframe columns; for formula terms, we just need columns referenced.
    # We'll use the subset of df columns that appear in the formula tokens as a pragmatic approach.
    used = _vars_in_formula(model.model.formula)
    used = [c for c in used if c in df.columns]

    row = {}
    for col in used:
        if col == treatment_var:
            row[col] = 0
            continue

        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            row[col] = float(s.median()) if at == "median" else float(s.mean())
        else:
            # mode fallback
            m = s.mode(dropna=True)
            row[col] = m.iloc[0] if len(m) else s.dropna().iloc[0]

    return pd.DataFrame([row])


# %% 
res = fit_logit_formula(df_clean, "converted ~ treatment + tenure_days_imputed + tenure_missing + C(device) + C(region)", robust_se="HC3")
s = summarize_treatment_from_formula(res)
print_treatment_formula_summary(s)


# %% 
# Defining a function to make it easy to see logistic interaction effects

def conditional_treatment_effects(
    res: dict,
    by: str,
    treatment_var: str = "treatment",
    control_value: int = 0,
    treatment_value: int = 1,
    at: str = "mean",
) -> pd.DataFrame:
    """
    Compute conditional treatment effects by a categorical variable
    using predicted probabilities.
    """
    model = res["model"]
    d = res["data_used"]

    levels = d[by].dropna().unique()
    rows = []

    for lvl in levels:
        base = _make_reference_row_for_formula(model, d, treatment_var, at=at)
        base[by] = lvl

        x0 = base.copy()
        x1 = base.copy()
        x0[treatment_var] = control_value
        x1[treatment_var] = treatment_value

        p0 = float(model.predict(x0)[0])
        p1 = float(model.predict(x1)[0])

        rows.append({
            by: lvl,
            "p_control": p0,
            "p_treatment": p1,
            "diff_pp": (p1 - p0) * 100,
            "rel_pct": (p1 / p0 - 1) * 100 if p0 > 0 else np.nan,
        })

    return pd.DataFrame(rows).sort_values("diff_pp", ascending=False)

# %% 

res = fit_logit_formula(
    df_clean,
    "converted ~ treatment * C(region) + tenure_days_imputed + tenure_missing + C(device)",
    robust_se="HC3",
)
hte_device = conditional_treatment_effects(res, by="region")
hte_device

# %% 
# Get bootstrapped confidencce intervals for interaction effects by region

def bootstrap_conditional_effect(
    df: pd.DataFrame,
    formula: str,
    by: str,
    n_boot: int = 1000,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    effects = []

    for _ in range(n_boot):
        b = df.sample(frac=1, replace=True, random_state=rng.integers(1e9))
        m = smf.logit(formula, data=b).fit(disp=False)

        for lvl in b[by].dropna().unique():
            ref = _make_reference_row_for_formula(m, b, "treatment")
            ref[by] = lvl

            x0 = ref.copy()
            x1 = ref.copy()
            x0["treatment"] = 0
            x1["treatment"] = 1

            effects.append({
                by: lvl,
                "diff_pp": (m.predict(x1)[0] - m.predict(x0)[0]) * 100,
            })

    return pd.DataFrame(effects)

# %%
# Bootstrapping CIs for conditional effects by device and region
# Neither device nor region show significant heterogeneity here.
boot = bootstrap_conditional_effect(
    df_clean,
    "converted ~ treatment * C(region) + tenure_days_imputed + tenure_missing + C(device)",
    by="region",
)

boot.groupby("region")["diff_pp"].quantile([0.025, 0.5, 0.975])

# %% 
# Defining a function to measure and visualise continuous interactions

def conditional_effects_continuous(
    res: dict,
    x: str,
    treatment_var: str = "treatment",
    grid: np.ndarray | None = None,
):
    model = res["model"]
    d = res["data_used"]

    if grid is None:
        grid = np.quantile(d[x], np.linspace(0.05, 0.95, 20))

    rows = []
    for val in grid:
        base = _make_reference_row_for_formula(model, d, treatment_var)
        base[x] = val

        x0 = base.copy()
        x1 = base.copy()
        x0[treatment_var] = 0
        x1[treatment_var] = 1

        p0 = float(model.predict(x0)[0])
        p1 = float(model.predict(x1)[0])

        rows.append({
            x: val,
            "diff_pp": (p1 - p0) * 100,
            "p_control": p0,
            "p_treatment": p1,
        })

    return pd.DataFrame(rows)

# %% 
# Continuous interaction by tenure_days_imputed

df_clean["tenure_days_imputed_centered"] = df_clean["tenure_days_imputed"] - df_clean["tenure_days_imputed"].mean()
res = fit_logit_formula(
    df_clean,
    "converted ~ treatment * tenure_days_imputed_centered + tenure_missing + C(region) + C(device)",
    robust_se="HC3",
)

ce = conditional_effects_continuous(res, x="tenure_days_imputed_centered")

plt.figure(figsize=(8, 5))
plt.plot(ce["tenure_days_imputed_centered"], ce["diff_pp"])
plt.axhline(0, color="black", lw=1)
plt.xlabel("Tenure (days)")
plt.ylabel("Treatment effect (pp)")
plt.title("Conditional treatment effect by tenure")
plt.show()


# %% 
# Adding condition effects bootstrapping for continuous variables

def bootstrap_conditional_effect_curve(
    df: pd.DataFrame,
    formula: str,
    x: str,
    grid: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    curves = []

    for _ in range(n_boot):
        b = df.sample(frac=1, replace=True, random_state=rng.integers(1e9))
        m = smf.logit(formula, data=b).fit(disp=False)

        for val in grid:
            ref = _make_reference_row_for_formula(m, b, treatment_var="treatment")
            ref[x] = val

            x0 = ref.copy()
            x1 = ref.copy()
            x0["treatment"] = 0
            x1["treatment"] = 1

            diff_pp = (m.predict(x1)[0] - m.predict(x0)[0]) * 100
            curves.append({"x": val, "diff_pp": diff_pp})

    return pd.DataFrame(curves)

# %% 
# Run bootstrapping across a grid of tenure values
# and plot 95% CI around conditional effect curve
grid = np.linspace(df_clean["tenure_days_imputed_centered"].quantile(0.05),
                   df_clean["tenure_days_imputed_centered"].quantile(0.95),
                   30)

boot = bootstrap_conditional_effect_curve(
    df_clean,
    "converted ~ treatment * tenure_days_imputed_centered + tenure_missing + C(device) + C(region)",
    x="tenure_days_imputed_centered",
    grid=grid,
)

ci = (
    boot.groupby("x")["diff_pp"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.5: "median", 0.025: "lo", 0.975: "hi"})
)

plt.figure(figsize=(8, 5))
plt.plot(ci["x"], ci["median"], label="Treatment effect")
plt.fill_between(ci["x"], ci["lo"], ci["hi"], alpha=0.2)
plt.axhline(0, color="black", lw=1)
plt.xlabel("Tenure (days)")
plt.ylabel("Treatment effect (pp)")
plt.title("Conditional treatment effect by tenure (95% CI)")
plt.legend()
plt.show()


# %%
# Fill in modeling here:
# - prepare model matrix (prep_features_for_logit)
# - fit logit (statsmodels or sklearn)
# - interpret treatment coefficient (marginal effects or predicted probabilities)

# %% [markdown]
# ## 6) Robustness & Assumption Checks
#
# Goal:
# - Check sensitivity to modeling / prep choices
# - Sanity-check predicted probabilities
# - Possibly run a bootstrap or stratified check

# %%
# Fill in robustness checks here

# %% [markdown]
# ## 7) Stretch: Treatment Effect Heterogeneity
#
# Goal:
# - Investigate whether treatment effect varies by:
#   - tenure bucket
#   - device
#   - region
#
# Approaches:
# - Interaction terms (treatment * segment)
# - Stratified models
# - Plots of uplift by segment

# %%
# Fill in HTE exploration here

# %% [markdown]
# ## 8) Risk Assessment
#
# Goal:
# - Identify whether some important segment might be harmed
# - Explain how ATE could be misleading
# - Recommend guardrails / follow-up experiments

# %%
# Fill in risk notes here

# %% [markdown]
# ## 9) Final Recommendation
#
# Deliver:
# - Clear ship / no-ship / iterate recommendation
# - Key numbers (ATE + CI)
# - Any HTE findings
# - Caveats and next steps (brief)

# %%
# Fill in final writeup here
