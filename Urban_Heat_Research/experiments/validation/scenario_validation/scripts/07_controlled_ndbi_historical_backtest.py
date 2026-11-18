from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[4]

BASE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "kaduwela_preprocessed_base.csv"
)

VALIDATION_DIR = (
    ROOT
    / "experiments"
    / "validation"
    / "scenario_validation"
)

OUTPUT_DIR = VALIDATION_DIR / "outputs"
FIGURE_DIR = VALIDATION_DIR / "figures"

STEP6_PATH = (
    OUTPUT_DIR
    / "06_direction_matched_grid_results.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

GRID = "grid_id"
YEAR = "year"

EPS = 1e-12

CONTROL_VARIABLES = [
    "d_green",
    "d_ndwi",
    "d_albedo",
    "d_nightlights",
]


# ============================================================
# HELPERS
# ============================================================

def safe_corr(a, b, method="pearson"):

    a = pd.Series(
        np.asarray(a, dtype=float)
    )

    b = pd.Series(
        np.asarray(b, dtype=float)
    )

    if a.nunique() < 2 or b.nunique() < 2:
        return np.nan

    return float(
        a.corr(
            b,
            method=method,
        )
    )


def sign_agreement(a, b):

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    valid = (
        np.isfinite(a)
        & np.isfinite(b)
        & (np.abs(a) > EPS)
        & (np.abs(b) > EPS)
    )

    if valid.sum() == 0:
        return np.nan, 0

    same_sign = (
        np.sign(a[valid])
        == np.sign(b[valid])
    )

    return (
        float(
            same_sign.mean() * 100.0
        ),
        int(valid.sum()),
    )


def metrics(
    observed,
    modeled,
):

    observed = np.asarray(
        observed,
        dtype=float,
    )

    modeled = np.asarray(
        modeled,
        dtype=float,
    )

    sign_pct, sign_n = sign_agreement(
        observed,
        modeled,
    )

    return {
        "pearson_r":
            safe_corr(
                observed,
                modeled,
                "pearson",
            ),

        "spearman_rho":
            safe_corr(
                observed,
                modeled,
                "spearman",
            ),

        "sign_agreement_pct":
            sign_pct,

        "sign_comparison_n":
            sign_n,

        "mae_c":
            float(
                np.mean(
                    np.abs(
                        observed - modeled
                    )
                )
            ),

        "rmse_c":
            float(
                np.sqrt(
                    np.mean(
                        (
                            observed - modeled
                        ) ** 2
                    )
                )
            ),
    }


def residualize(
    values,
    nuisance_X,
):

    model = sm.OLS(
        np.asarray(
            values,
            dtype=float,
        ),
        nuisance_X,
    ).fit()

    return (
        np.asarray(
            model.resid,
            dtype=float,
        ),
        model,
    )


# ============================================================
# LOAD
# ============================================================

print("\n============================================================")
print("SCENARIO VALIDATION 07")
print("CONTROLLED NDBI HISTORICAL BACKTEST")
print("============================================================\n")

if not BASE_PATH.exists():
    raise FileNotFoundError(
        f"Historical base file missing:\n{BASE_PATH}"
    )

if not STEP6_PATH.exists():
    raise FileNotFoundError(
        f"Step-6 results missing:\n{STEP6_PATH}"
    )


base = pd.read_csv(
    BASE_PATH,
    dtype={GRID: str},
    low_memory=False,
)

step6 = pd.read_csv(
    STEP6_PATH,
    dtype={GRID: str},
    low_memory=False,
)


base[GRID] = (
    base[GRID]
    .astype(str)
    .str.strip()
)

base[YEAR] = pd.to_numeric(
    base[YEAR],
    errors="raise",
).astype(int)


required = [
    GRID,
    YEAR,
    "green_mask",
    "NDBI",
    "NDWI",
    "Albedo",
    "NightLights",
    "LST_C",
]

missing = [
    c for c in required
    if c not in base.columns
]

if missing:
    raise RuntimeError(
        f"Historical base missing: {missing}"
    )


# ============================================================
# HISTORICAL YEAR-TO-YEAR CHANGES
# ============================================================

base = base.sort_values(
    [GRID, YEAR]
).copy()


for col in [
    "green_mask",
    "NDBI",
    "NDWI",
    "Albedo",
    "NightLights",
]:

    base[
        f"prev_{col}"
    ] = (
        base.groupby(GRID)[col]
        .shift(1)
    )


base["d_green"] = (
    base["green_mask"]
    - base["prev_green_mask"]
)

base["d_ndbi"] = (
    base["NDBI"]
    - base["prev_NDBI"]
)

base["d_ndwi"] = (
    base["NDWI"]
    - base["prev_NDWI"]
)

base["d_albedo"] = (
    base["Albedo"]
    - base["prev_Albedo"]
)

base["d_nightlights"] = (
    base["NightLights"]
    - base["prev_NightLights"]
)


changes = (
    base[
        [
            GRID,
            YEAR,
            "d_green",
            "d_ndbi",
            "d_ndwi",
            "d_albedo",
            "d_nightlights",
        ]
    ]
    .rename(
        columns={
            YEAR: "target_year",
        }
    )
)


# ============================================================
# NDBI REDUCTION CASES ONLY
# ============================================================

ndbi = (
    step6[
        step6[
            "scenario_family"
        ] == "BUILT_NDBI"
    ]
    .copy()
)


ndbi = ndbi.merge(
    changes,
    on=[
        GRID,
        "target_year",
    ],
    how="left",
    validate="one_to_one",
)


print(
    f"Direction-matched NDBI rows: "
    f"{len(ndbi):,}"
)


if ndbi[
    CONTROL_VARIABLES
].isna().any().any():

    raise RuntimeError(
        "Missing control changes after merge."
    )


# Confirm these are really NDBI reductions
if not (
    ndbi[
        "actual_treatment_change"
    ] < 0
).all():

    raise RuntimeError(
        "Step-6 NDBI subset contains "
        "non-reduction cases."
    )


# ============================================================
# RAW RESULT - PRESERVE STEP 6 EVIDENCE
# ============================================================

observed_raw = (
    ndbi[
        "observed_spatial_anomaly_change_c"
    ]
    .to_numpy(dtype=float)
)

modeled_raw = (
    ndbi[
        "modeled_pseudo_intervention_response_c"
    ]
    .to_numpy(dtype=float)
)


raw_metrics = metrics(
    observed_raw,
    modeled_raw,
)


# ============================================================
# BUILD NUISANCE DESIGN MATRIX
# ============================================================

controls = (
    ndbi[
        CONTROL_VARIABLES
    ]
    .apply(
        pd.to_numeric,
        errors="raise",
    )
    .copy()
)


# Standardize controls only for numerical stability.
# This does NOT change residualization meaning.
for col in CONTROL_VARIABLES:

    std = float(
        controls[col].std(ddof=0)
    )

    if std <= EPS:
        raise RuntimeError(
            f"Control {col} has zero variance."
        )

    controls[col] = (
        controls[col]
        - controls[col].mean()
    ) / std


# Add target-year fixed effect
year_dummy = pd.get_dummies(
    ndbi["target_year"]
    .astype(str),
    prefix="year",
    drop_first=True,
    dtype=float,
)


nuisance_X = pd.concat(
    [
        controls.reset_index(drop=True),
        year_dummy.reset_index(drop=True),
    ],
    axis=1,
)


nuisance_X = sm.add_constant(
    nuisance_X,
    has_constant="add",
)


# ============================================================
# FRISCH-WAUGH-LOVELL STYLE RESIDUALIZATION
# ============================================================

observed_resid, observed_nuisance_model = (
    residualize(
        observed_raw,
        nuisance_X,
    )
)


modeled_resid, modeled_nuisance_model = (
    residualize(
        modeled_raw,
        nuisance_X,
    )
)


controlled_metrics = metrics(
    observed_resid,
    modeled_resid,
)


# ============================================================
# ASSOCIATION TEST AFTER CONTROL
# ============================================================

association_X = sm.add_constant(
    modeled_resid,
    has_constant="add",
)


association_model = sm.OLS(
    observed_resid,
    association_X,
).fit(
    cov_type="HC3"
)


controlled_beta = float(
    association_model.params[1]
)

controlled_p = float(
    association_model.pvalues[1]
)

controlled_r2 = float(
    association_model.rsquared
)


# ============================================================
# STRONG NDBI REDUCTIONS
# ============================================================

abs_ndbi_change = np.abs(
    ndbi[
        "actual_treatment_change"
    ].to_numpy(dtype=float)
)


q75 = float(
    np.quantile(
        abs_ndbi_change,
        0.75,
    )
)


strong_mask = (
    abs_ndbi_change >= q75
)


strong_metrics = metrics(
    observed_resid[
        strong_mask
    ],
    modeled_resid[
        strong_mask
    ],
)


# ============================================================
# PER-YEAR CONTROLLED CHECK
# ============================================================

year_rows = []


for target_year in sorted(
    ndbi[
        "target_year"
    ].unique()
):

    mask = (
        ndbi[
            "target_year"
        ].to_numpy()
        == target_year
    )

    current = metrics(
        observed_resid[mask],
        modeled_resid[mask],
    )

    year_rows.append(
        {
            "target_year":
                target_year,

            "rows":
                int(mask.sum()),

            **current,
        }
    )


year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# MAIN SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "analysis":
                "RAW_DIRECTION_MATCHED",

            "rows":
                len(ndbi),

            **raw_metrics,

            "association_beta":
                np.nan,

            "association_p_value":
                np.nan,

            "association_r2":
                np.nan,
        },

        {
            "analysis":
                "CONTROLLED_RESIDUALIZED",

            "rows":
                len(ndbi),

            **controlled_metrics,

            "association_beta":
                controlled_beta,

            "association_p_value":
                controlled_p,

            "association_r2":
                controlled_r2,
        },

        {
            "analysis":
                "CONTROLLED_STRONG_NDBI_REDUCTIONS",

            "rows":
                int(
                    strong_mask.sum()
                ),

            **strong_metrics,

            "association_beta":
                np.nan,

            "association_p_value":
                np.nan,

            "association_r2":
                np.nan,
        },
    ]
)


# ============================================================
# SAVE GRID-LEVEL RESULTS
# ============================================================

ndbi[
    "observed_controlled_residual_c"
] = observed_resid

ndbi[
    "modeled_controlled_residual_c"
] = modeled_resid

ndbi[
    "strong_ndbi_reduction_q75"
] = strong_mask


summary_path = (
    OUTPUT_DIR
    / "07_controlled_ndbi_backtest_summary.csv"
)

year_path = (
    OUTPUT_DIR
    / "07_controlled_ndbi_backtest_by_year.csv"
)

grid_path = (
    OUTPUT_DIR
    / "07_controlled_ndbi_grid_results.csv"
)


summary.to_csv(
    summary_path,
    index=False,
)

year_df.to_csv(
    year_path,
    index=False,
)

ndbi.to_csv(
    grid_path,
    index=False,
)


# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)

ax.scatter(
    observed_resid,
    modeled_resid,
    alpha=0.35,
    s=12,
)

ax.axhline(
    0,
    linewidth=1,
)

ax.axvline(
    0,
    linewidth=1,
)

ax.set_xlabel(
    "Controlled Observed Heat Response (°C)"
)

ax.set_ylabel(
    "Controlled Modeled NDBI Response (°C)"
)

ax.set_title(
    "Controlled Historical NDBI Backtest"
)

fig.tight_layout()

figure_path = (
    FIGURE_DIR
    / "07_controlled_ndbi_backtest.png"
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# TERMINAL
# ============================================================

print("\n============================================================")
print("RAW vs CONTROLLED NDBI RESULTS")
print("============================================================")

print(
    summary.to_string(
        index=False
    )
)


print("\n============================================================")
print("CONTROLLED RESULTS BY YEAR")
print("============================================================")

print(
    year_df.to_string(
        index=False
    )
)


print("\n============================================================")
print("CONTROL VARIABLES")
print("============================================================")

print(
    "Removed concurrent variation from:"
)

for col in CONTROL_VARIABLES:
    print(f" - {col}")

print(
    " - target-year fixed effect"
)


print("\nIMPORTANT:")
print(
    "The RAW Step-6 result is preserved."
)

print(
    "The controlled result is an additional "
    "diagnostic, not a replacement."
)

print(
    "If controlled agreement improves, this suggests "
    "that simultaneous changes in other environmental "
    "variables partly obscured the isolated NDBI response."
)

print(
    "If it remains weak, NDBI scenario effects should "
    "continue to be reported cautiously."
)


print("\nSaved:")
print(summary_path)
print(year_path)
print(grid_path)
print(figure_path)

print(
    "\nCONTROLLED NDBI HISTORICAL "
    "BACKTEST COMPLETE."
)