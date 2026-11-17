from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[4]

SCENARIO_DIR = ROOT / "outputs" / "scenarios"

VALIDATION_DIR = (
    ROOT
    / "experiments"
    / "validation"
    / "scenario_validation"
)

OUTPUT_DIR = VALIDATION_DIR / "outputs"
FIGURE_DIR = VALIDATION_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


GREEN_PATH = (
    SCENARIO_DIR
    / "29a_v2_persistent_green_scenarios_2026_2030.csv"
)

BUILT_PATH = (
    SCENARIO_DIR
    / "29b_built_up_intensity_scenarios_2026_2030.csv"
)

COMBINED_PATH = (
    SCENARIO_DIR
    / "29c_combined_green_built_scenarios_2026_2030.csv"
)

BLUE_PATH = (
    SCENARIO_DIR
    / "29d_blue_space_ndwi_scenarios_2026_2030.csv"
)

TWFE_PATH = (
    OUTPUT_DIR
    / "02_twfe_historical_response_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

GRID = "grid_id"
YEAR = "year"

EXPECTED_GRIDS = 2194
YEARS = list(range(2026, 2031))

BASE_ROWS = EXPECTED_GRIDS * len(YEARS)

EXPECTED_SINGLE_FAMILY_ROWS = BASE_ROWS * 3
EXPECTED_COMBINED_ROWS = BASE_ROWS * 9

TOL = 1e-9
DIRECTION_TOL = 1e-10


# ============================================================
# HELPERS
# ============================================================

def require_file(path, label):

    if not path.exists():

        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def require_columns(df, columns, label):

    missing = [
        c for c in columns
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"{label} missing columns: {missing}"
        )


def finite_check(df, columns):

    values = (
        df[columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    return bool(
        np.isfinite(values).all()
    )


def max_abs(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        return np.nan

    return float(
        np.max(np.abs(values))
    )


def classify_direction(delta):

    delta = np.asarray(
        delta,
        dtype=float,
    )

    result = np.full(
        len(delta),
        "no_change",
        dtype=object,
    )

    result[
        delta < -DIRECTION_TOL
    ] = "cooling"

    result[
        delta > DIRECTION_TOL
    ] = "warming"

    return result


audit_rows = []


def add_audit(
    family,
    check,
    passed,
    details,
):

    audit_rows.append(
        {
            "scenario_family": family,
            "check": check,
            "status":
                "PASS"
                if bool(passed)
                else "REVIEW",
            "details": str(details),
        }
    )


def base_structure_checks(
    df,
    family,
    expected_rows,
    scenarios_per_grid_year,
):

    add_audit(
        family,
        "row_count",
        len(df) == expected_rows,
        f"actual={len(df):,}; "
        f"expected={expected_rows:,}",
    )

    add_audit(
        family,
        "unique_grid_count",
        df[GRID].nunique()
        == EXPECTED_GRIDS,
        f"actual={df[GRID].nunique():,}; "
        f"expected={EXPECTED_GRIDS:,}",
    )

    actual_years = sorted(
        pd.to_numeric(
            df[YEAR],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    add_audit(
        family,
        "forecast_years",
        actual_years == YEARS,
        f"actual={actual_years}; "
        f"expected={YEARS}",
    )

    key_duplicates = int(
        df.duplicated(
            subset=[
                GRID,
                YEAR,
                "scenario_id",
            ]
        ).sum()
    )

    add_audit(
        family,
        "unique_grid_year_scenario_key",
        key_duplicates == 0,
        f"duplicate_rows={key_duplicates}",
    )

    counts = (
        df.groupby(
            [GRID, YEAR]
        )["scenario_id"]
        .nunique()
    )

    correct_counts = bool(
        (
            counts
            == scenarios_per_grid_year
        ).all()
    )

    add_audit(
        family,
        "scenario_count_per_grid_year",
        correct_counts,
        f"expected_each="
        f"{scenarios_per_grid_year}; "
        f"min={int(counts.min())}; "
        f"max={int(counts.max())}",
    )


def direction_agreement(
    expected_sign,
    actual_value,
):

    if abs(actual_value) <= TOL:
        actual_sign = "ZERO"
    elif actual_value > 0:
        actual_sign = "POSITIVE"
    else:
        actual_sign = "NEGATIVE"

    return (
        actual_sign == expected_sign,
        actual_sign,
    )


# ============================================================
# LOAD FILES
# ============================================================

print("\n============================================================")
print("SCENARIO VALIDATION 03")
print("SCENARIO OUTPUT ROBUSTNESS & CONSISTENCY VALIDATION")
print("============================================================\n")

for path, label in [
    (GREEN_PATH, "Green scenarios"),
    (BUILT_PATH, "Built-up scenarios"),
    (COMBINED_PATH, "Combined scenarios"),
    (BLUE_PATH, "Blue-space scenarios"),
    (TWFE_PATH, "Step-2 TWFE validation"),
]:

    require_file(
        path,
        label,
    )


green = pd.read_csv(
    GREEN_PATH,
    low_memory=False,
)

built = pd.read_csv(
    BUILT_PATH,
    low_memory=False,
)

combined = pd.read_csv(
    COMBINED_PATH,
    low_memory=False,
)

blue = pd.read_csv(
    BLUE_PATH,
    low_memory=False,
)

twfe = pd.read_csv(
    TWFE_PATH,
    low_memory=False,
)


for df in [
    green,
    built,
    combined,
    blue,
]:

    df[GRID] = (
        df[GRID]
        .astype(str)
        .str.strip()
    )

    df[YEAR] = pd.to_numeric(
        df[YEAR],
        errors="raise",
    ).astype(int)


print(f"Green rows    : {len(green):,}")
print(f"Built rows    : {len(built):,}")
print(f"Combined rows : {len(combined):,}")
print(f"Blue rows     : {len(blue):,}")


# ============================================================
# 1. GREEN SCENARIO VALIDATION
# ============================================================

print("\n============================================================")
print("1. GREEN SCENARIO OUTPUT CHECKS")
print("============================================================")

green_required = [
    GRID,
    YEAR,
    "scenario_id",
    "requested_green_increase_pp",
    "baseline_green_mask",
    "scenario_green_mask",
    "actual_green_increase_pp",
    "baseline_predicted_lst_c",
    "scenario_predicted_lst_c",
    "cooling_benefit_c",
    "scenario_effect_direction",
]

require_columns(
    green,
    green_required,
    "Green scenarios",
)

base_structure_checks(
    green,
    "GREEN",
    EXPECTED_SINGLE_FAMILY_ROWS,
    3,
)

green_numeric = [
    "requested_green_increase_pp",
    "baseline_green_mask",
    "scenario_green_mask",
    "actual_green_increase_pp",
    "baseline_predicted_lst_c",
    "scenario_predicted_lst_c",
    "cooling_benefit_c",
]

add_audit(
    "GREEN",
    "finite_numeric_outputs",
    finite_check(
        green,
        green_numeric,
    ),
    f"rows={len(green):,}",
)

green_change_error = max_abs(
    (
        green["scenario_green_mask"]
        - green["baseline_green_mask"]
    )
    - (
        green["actual_green_increase_pp"]
        / 100.0
    )
)

add_audit(
    "GREEN",
    "green_change_identity",
    green_change_error <= TOL,
    f"max_error={green_change_error:.12e}",
)

green_bounds_ok = bool(
    (
        green["scenario_green_mask"]
        >= -TOL
    ).all()
    and (
        green["scenario_green_mask"]
        <= 1.0 + TOL
    ).all()
    and (
        green["actual_green_increase_pp"]
        >= -TOL
    ).all()
    and (
        green["actual_green_increase_pp"]
        <= (
            green[
                "requested_green_increase_pp"
            ]
            + TOL
        )
    ).all()
)

add_audit(
    "GREEN",
    "intervention_bounds",
    green_bounds_ok,
    (
        f"scenario_green_min="
        f"{green['scenario_green_mask'].min():.6f}; "
        f"max="
        f"{green['scenario_green_mask'].max():.6f}"
    ),
)

green_delta = (
    green["scenario_predicted_lst_c"]
    - green["baseline_predicted_lst_c"]
)

green_benefit_error = max_abs(
    green["cooling_benefit_c"]
    + green_delta
)

add_audit(
    "GREEN",
    "cooling_benefit_identity",
    green_benefit_error <= TOL,
    f"max_error_c="
    f"{green_benefit_error:.12e}",
)

green_direction_expected = classify_direction(
    green_delta
)

green_direction_match = bool(
    (
        green[
            "scenario_effect_direction"
        ].astype(str)
        == green_direction_expected
    ).all()
)

add_audit(
    "GREEN",
    "effect_direction_labels",
    green_direction_match,
    "Checked against scenario LST minus baseline LST.",
)


# ============================================================
# 2. BUILT-UP / NDBI VALIDATION
# ============================================================

print("\n============================================================")
print("2. BUILT-UP SCENARIO OUTPUT CHECKS")
print("============================================================")

built_required = [
    GRID,
    YEAR,
    "scenario_id",
    "scenario_level",
    "requested_ndbi_reduction",
    "baseline_ndbi",
    "scenario_ndbi",
    "actual_ndbi_reduction",
    "baseline_predicted_lst_c",
    "scenario_predicted_lst_c",
    "modeled_lst_change_c",
    "cooling_benefit_c",
    "scenario_effect_direction",
]

require_columns(
    built,
    built_required,
    "Built-up scenarios",
)

base_structure_checks(
    built,
    "BUILT_NDBI",
    EXPECTED_SINGLE_FAMILY_ROWS,
    3,
)

built_numeric = [
    "requested_ndbi_reduction",
    "baseline_ndbi",
    "scenario_ndbi",
    "actual_ndbi_reduction",
    "baseline_predicted_lst_c",
    "scenario_predicted_lst_c",
    "modeled_lst_change_c",
    "cooling_benefit_c",
]

add_audit(
    "BUILT_NDBI",
    "finite_numeric_outputs",
    finite_check(
        built,
        built_numeric,
    ),
    f"rows={len(built):,}",
)

built_change_error = max_abs(
    (
        built["baseline_ndbi"]
        - built["scenario_ndbi"]
    )
    - built["actual_ndbi_reduction"]
)

add_audit(
    "BUILT_NDBI",
    "ndbi_reduction_identity",
    built_change_error <= TOL,
    f"max_error="
    f"{built_change_error:.12e}",
)

built_bounds_ok = bool(
    (
        built["actual_ndbi_reduction"]
        >= -TOL
    ).all()
    and (
        built["actual_ndbi_reduction"]
        <= (
            built[
                "requested_ndbi_reduction"
            ]
            + TOL
        )
    ).all()
    and (
        built["scenario_ndbi"]
        <= (
            built["baseline_ndbi"]
            + TOL
        )
    ).all()
)

add_audit(
    "BUILT_NDBI",
    "intervention_bounds",
    built_bounds_ok,
    (
        f"actual_reduction_min="
        f"{built['actual_ndbi_reduction'].min():.6f}; "
        f"max="
        f"{built['actual_ndbi_reduction'].max():.6f}"
    ),
)

built_delta = (
    built["scenario_predicted_lst_c"]
    - built["baseline_predicted_lst_c"]
)

built_delta_error = max_abs(
    built["modeled_lst_change_c"]
    - built_delta
)

add_audit(
    "BUILT_NDBI",
    "modeled_lst_change_identity",
    built_delta_error <= TOL,
    f"max_error_c="
    f"{built_delta_error:.12e}",
)

built_benefit_error = max_abs(
    built["cooling_benefit_c"]
    + built_delta
)

add_audit(
    "BUILT_NDBI",
    "cooling_benefit_identity",
    built_benefit_error <= TOL,
    f"max_error_c="
    f"{built_benefit_error:.12e}",
)

built_direction_expected = classify_direction(
    built_delta
)

built_direction_match = bool(
    (
        built[
            "scenario_effect_direction"
        ].astype(str)
        == built_direction_expected
    ).all()
)

add_audit(
    "BUILT_NDBI",
    "effect_direction_labels",
    built_direction_match,
    "Checked against modeled LST change.",
)


# ============================================================
# 3. BLUE / NDWI VALIDATION
# ============================================================

print("\n============================================================")
print("3. BLUE-SPACE SCENARIO OUTPUT CHECKS")
print("============================================================")

blue_required = [
    GRID,
    YEAR,
    "scenario_id",
    "scenario_level",
    "requested_ndwi_increase",
    "baseline_ndwi",
    "scenario_ndwi",
    "actual_ndwi_increase",
    "baseline_predicted_lst_c",
    "scenario_predicted_lst_c",
    "modeled_lst_change_c",
    "cooling_benefit_c",
    "scenario_effect_direction",
]

require_columns(
    blue,
    blue_required,
    "Blue-space scenarios",
)

base_structure_checks(
    blue,
    "BLUE_NDWI",
    EXPECTED_SINGLE_FAMILY_ROWS,
    3,
)

blue_numeric = [
    "requested_ndwi_increase",
    "baseline_ndwi",
    "scenario_ndwi",
    "actual_ndwi_increase",
    "baseline_predicted_lst_c",
    "scenario_predicted_lst_c",
    "modeled_lst_change_c",
    "cooling_benefit_c",
]

add_audit(
    "BLUE_NDWI",
    "finite_numeric_outputs",
    finite_check(
        blue,
        blue_numeric,
    ),
    f"rows={len(blue):,}",
)

blue_change_error = max_abs(
    (
        blue["scenario_ndwi"]
        - blue["baseline_ndwi"]
    )
    - blue["actual_ndwi_increase"]
)

add_audit(
    "BLUE_NDWI",
    "ndwi_increase_identity",
    blue_change_error <= TOL,
    f"max_error="
    f"{blue_change_error:.12e}",
)

blue_bounds_ok = bool(
    (
        blue["actual_ndwi_increase"]
        >= -TOL
    ).all()
    and (
        blue["actual_ndwi_increase"]
        <= (
            blue[
                "requested_ndwi_increase"
            ]
            + TOL
        )
    ).all()
    and (
        blue["scenario_ndwi"]
        >= (
            blue["baseline_ndwi"]
            - TOL
        )
    ).all()
)

add_audit(
    "BLUE_NDWI",
    "intervention_bounds",
    blue_bounds_ok,
    (
        f"actual_increase_min="
        f"{blue['actual_ndwi_increase'].min():.6f}; "
        f"max="
        f"{blue['actual_ndwi_increase'].max():.6f}"
    ),
)

blue_delta = (
    blue["scenario_predicted_lst_c"]
    - blue["baseline_predicted_lst_c"]
)

blue_delta_error = max_abs(
    blue["modeled_lst_change_c"]
    - blue_delta
)

add_audit(
    "BLUE_NDWI",
    "modeled_lst_change_identity",
    blue_delta_error <= TOL,
    f"max_error_c="
    f"{blue_delta_error:.12e}",
)

blue_benefit_error = max_abs(
    blue["cooling_benefit_c"]
    + blue_delta
)

add_audit(
    "BLUE_NDWI",
    "cooling_benefit_identity",
    blue_benefit_error <= TOL,
    f"max_error_c="
    f"{blue_benefit_error:.12e}",
)

blue_direction_expected = classify_direction(
    blue_delta
)

blue_direction_match = bool(
    (
        blue[
            "scenario_effect_direction"
        ].astype(str)
        == blue_direction_expected
    ).all()
)

add_audit(
    "BLUE_NDWI",
    "effect_direction_labels",
    blue_direction_match,
    "Checked against modeled LST change.",
)


# ============================================================
# 4. COMBINED SCENARIO VALIDATION
# ============================================================

print("\n============================================================")
print("4. COMBINED SCENARIO OUTPUT CHECKS")
print("============================================================")

combined_required = [
    GRID,
    YEAR,
    "scenario_id",
    "green_increase_pp_requested",
    "green_increase_pp_actual",
    "built_up_level",
    "actual_ndbi_reduction",
    "baseline_green_mask",
    "scenario_green_mask",
    "baseline_ndbi",
    "scenario_ndbi",
    "baseline_predicted_lst_c",
    "combined_scenario_predicted_lst_c",
    "combined_modeled_lst_change_c",
    "combined_cooling_benefit_c",
    "combined_effect_direction",
    "green_only_modeled_lst_change_c",
    "built_only_modeled_lst_change_c",
    "additive_expected_lst_change_c",
    "model_non_additivity_c",
]

require_columns(
    combined,
    combined_required,
    "Combined scenarios",
)

base_structure_checks(
    combined,
    "COMBINED",
    EXPECTED_COMBINED_ROWS,
    9,
)

combined_numeric = [
    "green_increase_pp_requested",
    "green_increase_pp_actual",
    "actual_ndbi_reduction",
    "baseline_green_mask",
    "scenario_green_mask",
    "baseline_ndbi",
    "scenario_ndbi",
    "baseline_predicted_lst_c",
    "combined_scenario_predicted_lst_c",
    "combined_modeled_lst_change_c",
    "combined_cooling_benefit_c",
    "green_only_modeled_lst_change_c",
    "built_only_modeled_lst_change_c",
    "additive_expected_lst_change_c",
    "model_non_additivity_c",
]

add_audit(
    "COMBINED",
    "finite_numeric_outputs",
    finite_check(
        combined,
        combined_numeric,
    ),
    f"rows={len(combined):,}",
)

combined_green_error = max_abs(
    (
        combined["scenario_green_mask"]
        - combined["baseline_green_mask"]
    )
    - (
        combined[
            "green_increase_pp_actual"
        ]
        / 100.0
    )
)

add_audit(
    "COMBINED",
    "green_change_identity",
    combined_green_error <= TOL,
    f"max_error="
    f"{combined_green_error:.12e}",
)

combined_ndbi_error = max_abs(
    (
        combined["baseline_ndbi"]
        - combined["scenario_ndbi"]
    )
    - combined["actual_ndbi_reduction"]
)

add_audit(
    "COMBINED",
    "ndbi_change_identity",
    combined_ndbi_error <= TOL,
    f"max_error="
    f"{combined_ndbi_error:.12e}",
)

combined_delta = (
    combined[
        "combined_scenario_predicted_lst_c"
    ]
    - combined[
        "baseline_predicted_lst_c"
    ]
)

combined_delta_error = max_abs(
    combined[
        "combined_modeled_lst_change_c"
    ]
    - combined_delta
)

add_audit(
    "COMBINED",
    "combined_lst_change_identity",
    combined_delta_error <= TOL,
    f"max_error_c="
    f"{combined_delta_error:.12e}",
)

combined_benefit_error = max_abs(
    combined[
        "combined_cooling_benefit_c"
    ]
    + combined_delta
)

add_audit(
    "COMBINED",
    "combined_cooling_identity",
    combined_benefit_error <= TOL,
    f"max_error_c="
    f"{combined_benefit_error:.12e}",
)

additive_reference = (
    combined[
        "green_only_modeled_lst_change_c"
    ]
    + combined[
        "built_only_modeled_lst_change_c"
    ]
)

additive_error = max_abs(
    combined[
        "additive_expected_lst_change_c"
    ]
    - additive_reference
)

add_audit(
    "COMBINED",
    "additive_reference_identity",
    additive_error <= TOL,
    f"max_error_c="
    f"{additive_error:.12e}",
)

nonadd_error = max_abs(
    combined[
        "model_non_additivity_c"
    ]
    - (
        combined[
            "combined_modeled_lst_change_c"
        ]
        - combined[
            "additive_expected_lst_change_c"
        ]
    )
)

add_audit(
    "COMBINED",
    "non_additivity_identity",
    nonadd_error <= TOL,
    f"max_error_c="
    f"{nonadd_error:.12e}",
)

combined_direction_expected = classify_direction(
    combined_delta
)

combined_direction_match = bool(
    (
        combined[
            "combined_effect_direction"
        ].astype(str)
        == combined_direction_expected
    ).all()
)

add_audit(
    "COMBINED",
    "effect_direction_labels",
    combined_direction_match,
    "Checked against combined modeled LST change.",
)


# ============================================================
# 5. HISTORICAL DIRECTIONAL AGREEMENT
# ============================================================

print("\n============================================================")
print("5. HISTORICAL TWFE vs SCENARIO DIRECTION AGREEMENT")
print("============================================================")

full_twfe = twfe[
    twfe["model"].astype(str)
    .str.contains(
        "ALL_2015_2025",
        regex=False,
    )
].copy()


def get_beta(treatment):

    row = full_twfe[
        full_twfe["treatment"]
        == treatment
    ]

    if len(row) != 1:

        raise RuntimeError(
            f"Could not uniquely find TWFE "
            f"result for {treatment}."
        )

    return float(
        row.iloc[0]["beta_c_per_unit"]
    )


green_beta = get_beta(
    "green_mask"
)

ndbi_beta = get_beta(
    "NDBI"
)

ndwi_beta = get_beta(
    "NDWI"
)


# Green intervention is an INCREASE.
# Cooling benefit = - modeled LST change.
expected_green_benefit_sign = (
    "NEGATIVE"
    if green_beta > 0
    else "POSITIVE"
)

# NDBI intervention is a REDUCTION.
# Positive historical beta means reducing NDBI
# should be cooling-compatible.
expected_built_benefit_sign = (
    "POSITIVE"
    if ndbi_beta > 0
    else "NEGATIVE"
)

# NDWI intervention is an INCREASE.
# Negative historical beta means increasing NDWI
# should be cooling-compatible.
expected_blue_benefit_sign = (
    "POSITIVE"
    if ndwi_beta < 0
    else "NEGATIVE"
)


agreement_rows = []

for family, expected_sign, mean_benefit in [
    (
        "GREEN",
        expected_green_benefit_sign,
        float(
            green[
                "cooling_benefit_c"
            ].mean()
        ),
    ),
    (
        "BUILT_NDBI",
        expected_built_benefit_sign,
        float(
            built[
                "cooling_benefit_c"
            ].mean()
        ),
    ),
    (
        "BLUE_NDWI",
        expected_blue_benefit_sign,
        float(
            blue[
                "cooling_benefit_c"
            ].mean()
        ),
    ),
]:

    agreement, actual_sign = (
        direction_agreement(
            expected_sign,
            mean_benefit,
        )
    )

    agreement_rows.append(
        {
            "scenario_family": family,
            "historically_expected_cooling_benefit_sign":
                expected_sign,
            "scenario_mean_cooling_benefit_c":
                mean_benefit,
            "scenario_mean_sign":
                actual_sign,
            "direction_agreement":
                agreement,
            "status":
                "AGREE"
                if agreement
                else "REVIEW",
        }
    )


agreement_df = pd.DataFrame(
    agreement_rows
)

print(
    agreement_df.to_string(
        index=False
    )
)


# ============================================================
# 6. MONOTONICITY DIAGNOSTIC
# ============================================================

print("\n============================================================")
print("6. SCENARIO INTENSITY MONOTONICITY DIAGNOSTIC")
print("============================================================")


def monotonicity_percent(
    df,
    level_column,
    ordered_levels,
    benefit_column,
):

    pivot = (
        df.pivot_table(
            index=[GRID, YEAR],
            columns=level_column,
            values=benefit_column,
            aggfunc="first",
        )
    )

    missing = [
        level
        for level in ordered_levels
        if level not in pivot.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing levels in monotonicity "
            f"check: {missing}"
        )

    values = pivot[
        ordered_levels
    ].to_numpy(dtype=float)

    monotonic = (
        np.diff(
            values,
            axis=1,
        )
        >= -TOL
    ).all(axis=1)

    return (
        float(
            monotonic.mean()
            * 100.0
        ),
        int(
            monotonic.sum()
        ),
        int(
            len(monotonic)
        ),
    )


green_monotonic = (
    monotonicity_percent(
        green,
        "requested_green_increase_pp",
        [5, 10, 20],
        "cooling_benefit_c",
    )
)

built_monotonic = (
    monotonicity_percent(
        built,
        "scenario_level",
        ["LOW", "MODERATE", "HIGH"],
        "cooling_benefit_c",
    )
)

blue_monotonic = (
    monotonicity_percent(
        blue,
        "scenario_level",
        ["LOW", "MODERATE", "HIGH"],
        "cooling_benefit_c",
    )
)


monotonic_df = pd.DataFrame(
    [
        {
            "scenario_family": "GREEN",
            "monotonic_non_decreasing_cooling_pct":
                green_monotonic[0],
            "monotonic_cases":
                green_monotonic[1],
            "total_grid_year_cases":
                green_monotonic[2],
            "interpretation":
                "Diagnostic only; monotonic cooling is not forced.",
        },
        {
            "scenario_family": "BUILT_NDBI",
            "monotonic_non_decreasing_cooling_pct":
                built_monotonic[0],
            "monotonic_cases":
                built_monotonic[1],
            "total_grid_year_cases":
                built_monotonic[2],
            "interpretation":
                "Diagnostic only; nonlinear model responses are retained.",
        },
        {
            "scenario_family": "BLUE_NDWI",
            "monotonic_non_decreasing_cooling_pct":
                blue_monotonic[0],
            "monotonic_cases":
                blue_monotonic[1],
            "total_grid_year_cases":
                blue_monotonic[2],
            "interpretation":
                "Diagnostic only; nonlinear model responses are retained.",
        },
    ]
)

print(
    monotonic_df.to_string(
        index=False
    )
)


# ============================================================
# 7. OUTPUT SUMMARY
# ============================================================

summary_frames = []


def make_summary(
    df,
    family,
    level_columns,
    benefit_column,
):

    temp = df.copy()

    temp["_effect"] = np.where(
        temp[benefit_column] > DIRECTION_TOL,
        "cooling",
        np.where(
            temp[benefit_column]
            < -DIRECTION_TOL,
            "warming",
            "no_change",
        ),
    )

    grouped = (
        temp.groupby(
            level_columns,
            as_index=False,
        )
        .agg(
            grid_year_rows=(
                GRID,
                "size",
            ),
            mean_cooling_benefit_c=(
                benefit_column,
                "mean",
            ),
            median_cooling_benefit_c=(
                benefit_column,
                "median",
            ),
            min_cooling_benefit_c=(
                benefit_column,
                "min",
            ),
            max_cooling_benefit_c=(
                benefit_column,
                "max",
            ),
        )
    )

    effect_counts = (
        temp.groupby(
            level_columns
            + ["_effect"]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reset_index()
    )

    for col in [
        "cooling",
        "warming",
        "no_change",
    ]:
        if col not in effect_counts:
            effect_counts[col] = 0

    result = grouped.merge(
        effect_counts,
        on=level_columns,
        how="left",
    )

    result[
        "scenario_family"
    ] = family

    result[
        "cooling_pct"
    ] = (
        result["cooling"]
        / result["grid_year_rows"]
        * 100.0
    )

    result[
        "warming_pct"
    ] = (
        result["warming"]
        / result["grid_year_rows"]
        * 100.0
    )

    result[
        "no_change_pct"
    ] = (
        result["no_change"]
        / result["grid_year_rows"]
        * 100.0
    )

    return result


summary_frames.append(
    make_summary(
        green,
        "GREEN",
        [
            "requested_green_increase_pp"
        ],
        "cooling_benefit_c",
    )
)

summary_frames.append(
    make_summary(
        built,
        "BUILT_NDBI",
        ["scenario_level"],
        "cooling_benefit_c",
    )
)

summary_frames.append(
    make_summary(
        blue,
        "BLUE_NDWI",
        ["scenario_level"],
        "cooling_benefit_c",
    )
)

summary_frames.append(
    make_summary(
        combined,
        "COMBINED",
        [
            "green_increase_pp_requested",
            "built_up_level",
        ],
        "combined_cooling_benefit_c",
    )
)

summary_df = pd.concat(
    summary_frames,
    ignore_index=True,
    sort=False,
)


# ============================================================
# 8. SAVE TABLES
# ============================================================

audit_df = pd.DataFrame(
    audit_rows
)

audit_path = (
    OUTPUT_DIR
    / "03_scenario_output_robustness_audit.csv"
)

agreement_path = (
    OUTPUT_DIR
    / "03_historical_scenario_direction_agreement.csv"
)

monotonic_path = (
    OUTPUT_DIR
    / "03_scenario_monotonicity_diagnostic.csv"
)

summary_path = (
    OUTPUT_DIR
    / "03_scenario_output_summary.csv"
)

audit_df.to_csv(
    audit_path,
    index=False,
)

agreement_df.to_csv(
    agreement_path,
    index=False,
)

monotonic_df.to_csv(
    monotonic_path,
    index=False,
)

summary_df.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# 9. FIGURE - MEAN COOLING BENEFIT
# ============================================================

figure_rows = []

for pp in [5, 10, 20]:

    sub = green[
        green[
            "requested_green_increase_pp"
        ] == pp
    ]

    figure_rows.append(
        {
            "label":
                f"Green +{pp}pp",
            "mean_benefit":
                sub[
                    "cooling_benefit_c"
                ].mean(),
        }
    )


for level in [
    "LOW",
    "MODERATE",
    "HIGH",
]:

    sub = built[
        built[
            "scenario_level"
        ] == level
    ]

    figure_rows.append(
        {
            "label":
                f"Built {level.title()}",
            "mean_benefit":
                sub[
                    "cooling_benefit_c"
                ].mean(),
        }
    )


for level in [
    "LOW",
    "MODERATE",
    "HIGH",
]:

    sub = blue[
        blue[
            "scenario_level"
        ] == level
    ]

    figure_rows.append(
        {
            "label":
                f"Blue {level.title()}",
            "mean_benefit":
                sub[
                    "cooling_benefit_c"
                ].mean(),
        }
    )


figure_df = pd.DataFrame(
    figure_rows
)

fig, ax = plt.subplots(
    figsize=(11, 6)
)

ax.bar(
    figure_df["label"],
    figure_df["mean_benefit"],
)

ax.axhline(
    0,
    linewidth=1,
)

ax.set_ylabel(
    "Mean Cooling Benefit (°C)"
)

ax.set_title(
    "Scenario Validation - Mean Modeled Cooling Benefit"
)

ax.tick_params(
    axis="x",
    rotation=35,
)

fig.tight_layout()

benefit_figure_path = (
    FIGURE_DIR
    / "03_mean_modeled_cooling_benefit.png"
)

fig.savefig(
    benefit_figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 10. FIGURE - MONOTONICITY DIAGNOSTIC
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.bar(
    monotonic_df[
        "scenario_family"
    ],
    monotonic_df[
        "monotonic_non_decreasing_cooling_pct"
    ],
)

ax.set_ylim(
    0,
    100,
)

ax.set_ylabel(
    "Monotonic Grid-Year Cases (%)"
)

ax.set_title(
    "Scenario Intensity Monotonicity Diagnostic"
)

fig.tight_layout()

monotonic_figure_path = (
    FIGURE_DIR
    / "03_scenario_monotonicity_diagnostic.png"
)

fig.savefig(
    monotonic_figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FINAL TERMINAL OUTPUT
# ============================================================

print("\n============================================================")
print("STRUCTURAL / FORMULA AUDIT")
print("============================================================")

print(
    audit_df.to_string(
        index=False
    )
)


print("\n============================================================")
print("OVERALL AUDIT STATUS")
print("============================================================")

review_count = int(
    (
        audit_df["status"]
        == "REVIEW"
    ).sum()
)

if review_count == 0:

    print(
        "PASS - All structural, arithmetic, "
        "formula and direction-label checks passed."
    )

else:

    print(
        f"REVIEW - {review_count} audit "
        f"check(s) require investigation."
    )


print("\nHistorical directional agreement:")
print(
    agreement_df.to_string(
        index=False
    )
)

print("\nMonotonicity diagnostic:")
print(
    monotonic_df.to_string(
        index=False
    )
)

print("\nIMPORTANT:")
print(
    "Monotonic cooling is NOT required or forced. "
    "The percentages are diagnostic evidence only."
)

print(
    "Historical directional agreement is "
    "observational consistency evidence, "
    "not causal proof."
)


print("\nSaved outputs:")
print(audit_path)
print(agreement_path)
print(monotonic_path)
print(summary_path)

print("\nSaved figures:")
print(benefit_figure_path)
print(monotonic_figure_path)

print(
    "\nSCENARIO OUTPUT ROBUSTNESS "
    "VALIDATION COMPLETE."
)