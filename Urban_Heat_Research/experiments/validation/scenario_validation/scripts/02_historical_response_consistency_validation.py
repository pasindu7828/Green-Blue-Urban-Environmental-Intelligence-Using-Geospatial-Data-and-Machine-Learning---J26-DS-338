from pathlib import Path

import numpy as np
import pandas as pd
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

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "validation"
    / "scenario_validation"
    / "outputs"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

GRID = "grid_id"
YEAR = "year"
TARGET = "LST_C"

EXPECTED_GRIDS = 2194
ALL_YEARS = list(range(2015, 2026))
EXCLUDE_2017_YEARS = [
    y for y in ALL_YEARS
    if y != 2017
]

REQUIRED = [
    GRID,
    YEAR,
    TARGET,
    "green_mask",
    "NDBI",
    "NDWI",
    "Albedo",
    "NightLights",
]


# ============================================================
# LOAD DATA
# ============================================================

print("\n============================================================")
print("SCENARIO VALIDATION 02")
print("HISTORICAL RESPONSE CONSISTENCY VALIDATION")
print("============================================================\n")

df = pd.read_csv(BASE_PATH, low_memory=False)

missing = [
    col for col in REQUIRED
    if col not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )

df = df[REQUIRED].copy()

df[GRID] = df[GRID].astype(str).str.strip()

df[YEAR] = pd.to_numeric(
    df[YEAR],
    errors="raise",
).astype(int)

df = df[
    df[YEAR].between(2015, 2025)
].copy()

print(f"Rows         : {len(df):,}")
print(f"Unique grids : {df[GRID].nunique():,}")
print(f"Years        : {sorted(df[YEAR].unique())}")


# ============================================================
# TWO-WAY DEMEANING
# ============================================================

def two_way_demean(data, columns):

    result = pd.DataFrame(
        index=data.index
    )

    for column in columns:

        x = pd.to_numeric(
            data[column],
            errors="raise",
        )

        grid_mean = (
            x.groupby(data[GRID])
            .transform("mean")
        )

        year_mean = (
            x.groupby(data[YEAR])
            .transform("mean")
        )

        grand_mean = x.mean()

        result[column] = (
            x
            - grid_mean
            - year_mean
            + grand_mean
        )

    return result


# ============================================================
# TWFE MODEL
# ============================================================

def fit_twfe(
    treatment,
    controls,
    years,
    label,
):

    sub = (
        df[
            df[YEAR].isin(years)
        ]
        .copy()
        .sort_values([GRID, YEAR])
        .reset_index(drop=True)
    )

    expected_rows = (
        EXPECTED_GRIDS
        * len(years)
    )

    if len(sub) != expected_rows:
        raise RuntimeError(
            f"{label}: expected "
            f"{expected_rows:,} rows, "
            f"found {len(sub):,}"
        )

    columns = [
        TARGET,
        treatment,
        *controls,
    ]

    transformed = two_way_demean(
        sub,
        columns,
    )

    y = transformed[
        TARGET
    ].to_numpy(dtype=float)

    X = transformed[
        [
            treatment,
            *controls,
        ]
    ].copy()

    rank = np.linalg.matrix_rank(
        X.to_numpy(dtype=float)
    )

    if rank != X.shape[1]:
        raise RuntimeError(
            f"{label}: rank-deficient model."
        )

    fit = sm.OLS(
        y,
        X,
        hasconst=False,
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups":
                sub[GRID]
                .astype(str)
                .to_numpy(),
            "use_correction":
                True,
        },
    )

    beta = float(
        fit.params[treatment]
    )

    se = float(
        fit.bse[treatment]
    )

    p = float(
        fit.pvalues[treatment]
    )

    ci = fit.conf_int().loc[
        treatment
    ]

    return {
        "model": label,
        "treatment": treatment,
        "rows": int(fit.nobs),
        "beta_c_per_unit": beta,
        "clustered_se": se,
        "p_value": p,
        "ci95_lower": float(ci.iloc[0]),
        "ci95_upper": float(ci.iloc[1]),
        "sign":
            "POSITIVE"
            if beta > 0
            else "NEGATIVE",
        "within_r2":
            float(fit.rsquared),
    }


# ============================================================
# RUN THREE SCENARIO RESPONSE CHECKS
# ============================================================

results = []

# -------------------------
# GREEN COVER
# -------------------------

green_controls = [
    "NDBI",
    "NDWI",
    "Albedo",
    "NightLights",
]

results.append(
    fit_twfe(
        treatment="green_mask",
        controls=green_controls,
        years=ALL_YEARS,
        label="GREEN_ALL_2015_2025",
    )
)

results.append(
    fit_twfe(
        treatment="green_mask",
        controls=green_controls,
        years=EXCLUDE_2017_YEARS,
        label="GREEN_EXCLUDE_2017",
    )
)


# -------------------------
# BUILT-UP / NDBI
# -------------------------

ndbi_controls = [
    "green_mask",
    "NDWI",
    "Albedo",
    "NightLights",
]

results.append(
    fit_twfe(
        treatment="NDBI",
        controls=ndbi_controls,
        years=ALL_YEARS,
        label="NDBI_ALL_2015_2025",
    )
)

results.append(
    fit_twfe(
        treatment="NDBI",
        controls=ndbi_controls,
        years=EXCLUDE_2017_YEARS,
        label="NDBI_EXCLUDE_2017",
    )
)


# -------------------------
# BLUE SPACE / NDWI
# -------------------------

ndwi_controls = [
    "green_mask",
    "NDBI",
    "Albedo",
    "NightLights",
]

results.append(
    fit_twfe(
        treatment="NDWI",
        controls=ndwi_controls,
        years=ALL_YEARS,
        label="NDWI_ALL_2015_2025",
    )
)

results.append(
    fit_twfe(
        treatment="NDWI",
        controls=ndwi_controls,
        years=EXCLUDE_2017_YEARS,
        label="NDWI_EXCLUDE_2017",
    )
)


# ============================================================
# SUMMARIZE
# ============================================================

result_df = pd.DataFrame(results)

result_df[
    "response_per_standard_change_c"
] = np.nan

# Green: +10 percentage points = +0.10
green_mask = (
    result_df["treatment"]
    == "green_mask"
)

result_df.loc[
    green_mask,
    "response_per_standard_change_c",
] = (
    result_df.loc[
        green_mask,
        "beta_c_per_unit",
    ]
    * 0.10
)

# NDBI +0.05
ndbi_mask = (
    result_df["treatment"]
    == "NDBI"
)

result_df.loc[
    ndbi_mask,
    "response_per_standard_change_c",
] = (
    result_df.loc[
        ndbi_mask,
        "beta_c_per_unit",
    ]
    * 0.05
)

# NDWI +0.05
ndwi_mask = (
    result_df["treatment"]
    == "NDWI"
)

result_df.loc[
    ndwi_mask,
    "response_per_standard_change_c",
] = (
    result_df.loc[
        ndwi_mask,
        "beta_c_per_unit",
    ]
    * 0.05
)


# ============================================================
# SIGN STABILITY
# ============================================================

stability_rows = []

for treatment in [
    "green_mask",
    "NDBI",
    "NDWI",
]:

    sub = result_df[
        result_df["treatment"]
        == treatment
    ]

    signs = sub["sign"].tolist()

    stable = (
        len(set(signs)) == 1
    )

    stability_rows.append(
        {
            "treatment": treatment,
            "full_period_sign":
                signs[0],
            "exclude_2017_sign":
                signs[1],
            "sign_stable":
                stable,
        }
    )

stability_df = pd.DataFrame(
    stability_rows
)


# ============================================================
# INTERPRETATION
# ============================================================

interpretation = []

for _, row in stability_df.iterrows():

    treatment = row["treatment"]
    sign = row["full_period_sign"]

    if treatment == "NDBI":

        meaning = (
            "Positive historical association means "
            "higher NDBI is associated with higher LST; "
            "therefore an NDBI reduction is "
            "cooling-compatible."
            if sign == "POSITIVE"
            else
            "Historical direction does not support "
            "the expected NDBI-reduction cooling response."
        )

    elif treatment == "NDWI":

        meaning = (
            "Negative historical association means "
            "higher NDWI is associated with lower LST; "
            "therefore an NDWI increase is "
            "cooling-compatible."
            if sign == "NEGATIVE"
            else
            "Historical direction does not support "
            "the expected NDWI-increase cooling response."
        )

    else:

        meaning = (
            "Positive historical green-cover association "
            "indicates that increased green cover was not "
            "associated with cooling in this observational "
            "panel after controls. The result must be "
            "reported without forcing a cooling direction."
            if sign == "POSITIVE"
            else
            "Negative historical green-cover association "
            "is cooling-compatible, but remains "
            "observational rather than causal."
        )

    interpretation.append(
        {
            "treatment": treatment,
            "sign_stable":
                row["sign_stable"],
            "interpretation": meaning,
        }
    )

interpretation_df = pd.DataFrame(
    interpretation
)


# ============================================================
# SAVE
# ============================================================

result_path = (
    OUTPUT_DIR
    / "02_twfe_historical_response_results.csv"
)

stability_path = (
    OUTPUT_DIR
    / "02_response_sign_stability.csv"
)

interpretation_path = (
    OUTPUT_DIR
    / "02_response_interpretation.csv"
)

result_df.to_csv(
    result_path,
    index=False,
)

stability_df.to_csv(
    stability_path,
    index=False,
)

interpretation_df.to_csv(
    interpretation_path,
    index=False,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n============================================================")
print("TWFE HISTORICAL RESPONSE RESULTS")
print("============================================================")

print(
    result_df[
        [
            "model",
            "treatment",
            "beta_c_per_unit",
            "response_per_standard_change_c",
            "p_value",
            "sign",
            "within_r2",
        ]
    ].to_string(index=False)
)

print("\n============================================================")
print("SIGN STABILITY")
print("============================================================")

print(
    stability_df.to_string(
        index=False
    )
)

print("\n============================================================")
print("INTERPRETATION")
print("============================================================")

for _, row in interpretation_df.iterrows():

    print(
        f"\n{row['treatment']}: "
        f"{row['interpretation']}"
    )


print("\n============================================================")
print("VALIDATION STATUS")
print("============================================================")

for _, row in stability_df.iterrows():

    status = (
        "PASS"
        if row["sign_stable"]
        else "REVIEW"
    )

    print(
        f"{status} - "
        f"{row['treatment']} response sign stability"
    )

print("\nIMPORTANT:")
print(
    "These are historical observational within-grid "
    "associations, not causal intervention proof."
)

print("\nSaved:")
print(result_path)
print(stability_path)
print(interpretation_path)

print(
    "\nSCENARIO RESPONSE CONSISTENCY "
    "VALIDATION COMPLETE."
)