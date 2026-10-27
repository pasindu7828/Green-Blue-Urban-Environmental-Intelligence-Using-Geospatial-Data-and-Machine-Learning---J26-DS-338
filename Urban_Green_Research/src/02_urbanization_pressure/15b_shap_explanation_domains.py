"""
STEP 15B - SHAP EXPLANATION DOMAIN SUMMARIZATION

Research Component:
Urban Green Transition, Urbanization Pressure and Conservation Intelligence

Purpose:
1. Convert technical Step 15A SHAP feature names into understandable domains.
2. Create global domain-level SHAP summaries.
3. Create threshold-flagged domain summaries.
4. Create per-grid local domain explanations.
5. Create dashboard-friendly top reasons for every scored grid cell.

IMPORTANT:
- No model training.
- No model scoring.
- No threshold tuning.
- No pressure score modification.
- No SHAP recalculation.
- SHAP values remain raw XGBoost margin contributions.
- Explanations describe model behaviour, not proven real-world causality.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SHAP_VALUES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "xai"
    / "15a_2025_inference_shap_values.csv"
)

LOCAL_EXPLANATIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "xai"
    / "15a_local_top_shap_explanations.csv"
)

GLOBAL_IMPORTANCE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "15a_global_shap_importance.csv"
)

FLAGGED_GLOBAL_IMPORTANCE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "15a_flagged_global_shap_importance.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "urbanization_pressure"
    / "xai"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. OUTPUTS
# =============================================================================

FEATURE_DOMAIN_MAP_CSV = (
    REPORT_DIR
    / "15b_feature_explanation_domain_map.csv"
)

GLOBAL_DOMAIN_CSV = (
    REPORT_DIR
    / "15b_global_shap_domain_importance.csv"
)

FLAGGED_DOMAIN_CSV = (
    REPORT_DIR
    / "15b_flagged_shap_domain_importance.csv"
)

LOCAL_DOMAIN_CSV = (
    OUTPUT_DIR
    / "15b_local_domain_explanations.csv"
)

DASHBOARD_EXPLANATIONS_CSV = (
    OUTPUT_DIR
    / "15b_dashboard_grid_explanations.csv"
)

QA_CSV = (
    REPORT_DIR
    / "15b_explanation_domain_qa.csv"
)

SUMMARY_CSV = (
    REPORT_DIR
    / "15b_explanation_domain_summary.csv"
)


# =============================================================================
# 3. EXPECTED VALUES
# =============================================================================

EXPECTED_CELLS = 4620

EXPECTED_FEATURES = 55

EXPECTED_FLAGGED_CELLS = 255

EXPECTED_LOCAL_TOP_FEATURES = 8

EXPECTED_ORIGIN_YEAR = 2025

EXPECTED_PRESSURE_PERIOD = "2026-2028"


# =============================================================================
# 4. EXPLANATION DOMAINS
# =============================================================================

DOMAIN_ORDER = [
    "Local Built-up Condition",
    "Recent Land-cover Change",
    "Neighbourhood Urban Context",
    "Local Green / Land-cover Context",
    "Terrain",
]


# =============================================================================
# 5. HUMAN-READABLE FEATURE LABELS
# =============================================================================

HUMAN_FEATURE_LABELS = {

    # -------------------------------------------------------------------------
    # GREEN COVER
    # -------------------------------------------------------------------------

    "green_fraction_t_minus_2":
        "Green-cover fraction two years before origin",

    "green_fraction_t_minus_1":
        "Green-cover fraction one year before origin",

    "green_fraction_t":
        "Green-cover fraction at origin",

    "green_mean_3yr":
        "Mean green-cover fraction over 3 years",

    "green_min_3yr":
        "Minimum green-cover fraction over 3 years",

    "green_max_3yr":
        "Maximum green-cover fraction over 3 years",

    "green_change_1yr":
        "1-year green-cover change",

    "green_change_2yr":
        "2-year green-cover change",

    "green_range_3yr":
        "3-year green-cover variability",

    # -------------------------------------------------------------------------
    # BUILT-UP
    # -------------------------------------------------------------------------

    "built_fraction_t_minus_2":
        "Built-up fraction two years before origin",

    "built_fraction_t_minus_1":
        "Built-up fraction one year before origin",

    "built_fraction_t":
        "Built-up fraction at origin",

    "built_mean_3yr":
        "Mean built-up fraction over 3 years",

    "built_min_3yr":
        "Minimum built-up fraction over 3 years",

    "built_max_3yr":
        "Maximum built-up fraction over 3 years",

    "built_change_1yr":
        "1-year built-up change",

    "built_change_2yr":
        "2-year built-up change",

    "built_range_3yr":
        "3-year built-up variability",

    # -------------------------------------------------------------------------
    # OPEN / BARE LAND
    # -------------------------------------------------------------------------

    "open_bare_fraction_t_minus_2":
        "Open/bare-land fraction two years before origin",

    "open_bare_fraction_t_minus_1":
        "Open/bare-land fraction one year before origin",

    "open_bare_fraction_t":
        "Open/bare-land fraction at origin",

    "open_bare_mean_3yr":
        "Mean open/bare-land fraction over 3 years",

    "open_bare_min_3yr":
        "Minimum open/bare-land fraction over 3 years",

    "open_bare_change_1yr":
        "1-year open/bare-land change",

    "open_bare_change_2yr":
        "2-year open/bare-land change",

    "open_bare_range_3yr":
        "3-year open/bare-land variability",

    # -------------------------------------------------------------------------
    # WATER
    # -------------------------------------------------------------------------

    "water_fraction_t_minus_2":
        "Water fraction two years before origin",

    "water_fraction_t_minus_1":
        "Water fraction one year before origin",

    "water_fraction_t":
        "Water fraction at origin",

    "water_mean_3yr":
        "Mean water fraction over 3 years",

    "water_min_3yr":
        "Minimum water fraction over 3 years",

    "water_change_1yr":
        "1-year water change",

    "water_change_2yr":
        "2-year water change",

    "water_range_3yr":
        "3-year water variability",

    # -------------------------------------------------------------------------
    # BUILT / GREEN CHANGE RELATIONSHIPS
    # -------------------------------------------------------------------------

    "built_growth_vs_green_loss_1yr":
        "1-year built-up growth relative to green-cover loss",

    "built_growth_vs_green_loss_2yr":
        "2-year built-up growth relative to green-cover loss",

    # -------------------------------------------------------------------------
    # 8-NEIGHBOUR CONTEXT
    # -------------------------------------------------------------------------

    "nbr8_built_fraction_t_minus_2":
        "Nearby built-up fraction two years before origin (8 neighbours)",

    "nbr8_built_fraction_t_minus_1":
        "Nearby built-up fraction one year before origin (8 neighbours)",

    "nbr8_built_fraction_t":
        "Nearby built-up fraction at origin (8 neighbours)",

    "nbr8_built_mean_3yr":
        "Mean nearby built-up fraction over 3 years (8 neighbours)",

    "nbr8_built_change_1yr":
        "1-year nearby built-up change (8 neighbours)",

    "nbr8_built_change_2yr":
        "2-year nearby built-up change (8 neighbours)",

    "nbr8_green_fraction_t":
        "Nearby green-cover fraction at origin (8 neighbours)",

    "nbr8_green_change_2yr":
        "2-year nearby green-cover change (8 neighbours)",

    # -------------------------------------------------------------------------
    # 24-NEIGHBOUR CONTEXT
    # -------------------------------------------------------------------------

    "nbr24_built_fraction_t":
        "Nearby built-up fraction at origin (24 neighbours)",

    "nbr24_built_mean_3yr":
        "Mean nearby built-up fraction over 3 years (24 neighbours)",

    "nbr24_built_change_2yr":
        "2-year nearby built-up change (24 neighbours)",

    "nbr24_green_fraction_t":
        "Nearby green-cover fraction at origin (24 neighbours)",

    # -------------------------------------------------------------------------
    # TERRAIN
    # -------------------------------------------------------------------------

    "elevation_mean_m":
        "Mean elevation (m)",

    "elevation_min_m":
        "Minimum elevation (m)",

    "elevation_max_m":
        "Maximum elevation (m)",

    "elevation_range_m":
        "Elevation range (m)",

    "slope_mean_deg":
        "Mean slope (degrees)",

    "slope_std_deg":
        "Slope variability (degrees)",

    "slope_max_deg":
        "Maximum slope (degrees)",
}


# =============================================================================
# 6. HELPERS
# =============================================================================

def relative_path(path: Path) -> str:
    """
    Return project-relative path where possible.
    """

    try:

        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:

        return str(path)


def assign_domain(feature_name):
    """
    Assign each frozen predictor to an understandable explanation domain.

    Rule order matters.

    Neighbourhood variables remain in the neighbourhood domain even
    when they contain temporal-change terms.
    """

    name = feature_name.lower()

    # -------------------------------------------------------------------------
    # Neighbourhood spatial context
    # -------------------------------------------------------------------------

    if (
        name.startswith("nbr8_")
        or
        name.startswith("nbr24_")
    ):

        return "Neighbourhood Urban Context"

    # -------------------------------------------------------------------------
    # Terrain
    # -------------------------------------------------------------------------

    if (
        name.startswith("elevation_")
        or
        name.startswith("slope_")
    ):

        return "Terrain"

    # -------------------------------------------------------------------------
    # Recent local temporal changes / variability
    # -------------------------------------------------------------------------

    if (
        "_change_" in name
        or
        "_range_" in name
        or
        "growth_vs_green_loss" in name
    ):

        return "Recent Land-cover Change"

    # -------------------------------------------------------------------------
    # Current / historical local built-up condition
    # -------------------------------------------------------------------------

    if name.startswith("built_"):

        return "Local Built-up Condition"

    # -------------------------------------------------------------------------
    # Green, open/bare and water condition
    # -------------------------------------------------------------------------

    if (
        name.startswith("green_")
        or
        name.startswith("open_bare_")
        or
        name.startswith("water_")
    ):

        return "Local Green / Land-cover Context"

    raise RuntimeError(
        f"No explanation domain rule exists for feature: "
        f"{feature_name}"
    )


def humanize_feature_name(feature_name):
    """
    Convert technical feature name into a clear dashboard label.
    """

    if feature_name in HUMAN_FEATURE_LABELS:

        return HUMAN_FEATURE_LABELS[
            feature_name
        ]

    # Safe fallback.
    return (
        feature_name
        .replace(
            "_",
            " "
        )
        .strip()
        .capitalize()
    )


def shap_direction(value):
    """
    Interpret the sign of the SHAP contribution.

    SHAP values are raw model-margin contributions.
    """

    if value > 0:

        return "INCREASES_PRESSURE"

    if value < 0:

        return "DECREASES_PRESSURE"

    return "NEUTRAL"


# =============================================================================
# 7. START
# =============================================================================

print("=" * 100)
print("STEP 15B - SHAP EXPLANATION DOMAIN SUMMARIZATION")
print("=" * 100)


for required_file in [
    SHAP_VALUES_FILE,
    LOCAL_EXPLANATIONS_FILE,
    GLOBAL_IMPORTANCE_FILE,
    FLAGGED_GLOBAL_IMPORTANCE_FILE,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{required_file}"
        )


# =============================================================================
# 8. LOAD STEP 15A OUTPUTS
# =============================================================================

shap_df = pd.read_csv(
    SHAP_VALUES_FILE
)

local_df = pd.read_csv(
    LOCAL_EXPLANATIONS_FILE
)

global_df = pd.read_csv(
    GLOBAL_IMPORTANCE_FILE
)

flagged_global_df = pd.read_csv(
    FLAGGED_GLOBAL_IMPORTANCE_FILE
)


# =============================================================================
# 9. BASIC INPUT QA
# =============================================================================

if len(
    shap_df
) != EXPECTED_CELLS:

    raise RuntimeError(
        f"Expected {EXPECTED_CELLS:,} SHAP rows, "
        f"found {len(shap_df):,}."
    )


if shap_df[
    "grid_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate grid IDs detected in SHAP output."
    )


if len(
    global_df
) != EXPECTED_FEATURES:

    raise RuntimeError(
        f"Expected {EXPECTED_FEATURES} global features, "
        f"found {len(global_df)}."
    )


if len(
    flagged_global_df
) != EXPECTED_FEATURES:

    raise RuntimeError(
        f"Expected {EXPECTED_FEATURES} flagged-global features, "
        f"found {len(flagged_global_df)}."
    )


expected_local_rows = (
    EXPECTED_CELLS
    *
    EXPECTED_LOCAL_TOP_FEATURES
)


if len(
    local_df
) != expected_local_rows:

    raise RuntimeError(
        f"Expected {expected_local_rows:,} local explanation rows, "
        f"found {len(local_df):,}."
    )


if not (
    shap_df[
        "origin_year"
    ]
    ==
    EXPECTED_ORIGIN_YEAR
).all():

    raise RuntimeError(
        "Unexpected origin year in Step 15A SHAP output."
    )


if not (
    shap_df[
        "pressure_period"
    ]
    .astype(
        str
    )
    ==
    EXPECTED_PRESSURE_PERIOD
).all():

    raise RuntimeError(
        "Unexpected pressure period in Step 15A SHAP output."
    )


flagged_count_input = int(
    shap_df[
        "threshold_flag"
    ]
    .astype(
        int
    )
    .sum()
)


if flagged_count_input != EXPECTED_FLAGGED_CELLS:

    raise RuntimeError(
        f"Expected {EXPECTED_FLAGGED_CELLS} flagged cells, "
        f"found {flagged_count_input}."
    )


print(
    f"\nSHAP cells             : "
    f"{len(shap_df):,}"
)

print(
    f"Frozen predictors      : "
    f"{len(global_df)}"
)

print(
    f"Threshold-flagged      : "
    f"{flagged_count_input:,}"
)

print(
    f"Origin                 : "
    f"{EXPECTED_ORIGIN_YEAR}"
)

print(
    f"Pressure period        : "
    f"{EXPECTED_PRESSURE_PERIOD}"
)


# =============================================================================
# 10. CREATE FEATURE-DOMAIN MAP
# =============================================================================

feature_names = (
    global_df[
        "feature_name"
    ]
    .astype(
        str
    )
    .tolist()
)


if len(
    set(
        feature_names
    )
) != EXPECTED_FEATURES:

    raise RuntimeError(
        "Global feature list contains duplicates."
    )


feature_domain_df = pd.DataFrame(
    {
        "feature_name": (
            feature_names
        )
    }
)


feature_domain_df[
    "explanation_domain"
] = (
    feature_domain_df[
        "feature_name"
    ]
    .apply(
        assign_domain
    )
)


feature_domain_df[
    "human_readable_feature"
] = (
    feature_domain_df[
        "feature_name"
    ]
    .apply(
        humanize_feature_name
    )
)


domain_order_lookup = {
    domain: index
    for index, domain in enumerate(
        DOMAIN_ORDER,
        start=1,
    )
}


feature_domain_df[
    "domain_order"
] = (
    feature_domain_df[
        "explanation_domain"
    ]
    .map(
        domain_order_lookup
    )
)


if feature_domain_df[
    "explanation_domain"
].isna().any():

    raise RuntimeError(
        "Some features were not assigned to an explanation domain."
    )


if feature_domain_df[
    "human_readable_feature"
].isna().any():

    raise RuntimeError(
        "Some features do not have human-readable labels."
    )


print("\n" + "=" * 100)
print("FEATURE DOMAIN COUNTS")
print("=" * 100)


print(
    feature_domain_df[
        "explanation_domain"
    ]
    .value_counts()
    .to_string()
)


# =============================================================================
# 11. GLOBAL DOMAIN IMPORTANCE
# =============================================================================

global_with_domain = global_df.merge(
    feature_domain_df,
    on="feature_name",
    how="left",
    validate="one_to_one",
)


global_domain_df = (
    global_with_domain.groupby(
        "explanation_domain",
        as_index=False,
    )
    .agg(
        mean_absolute_shap_sum=(
            "mean_absolute_shap_raw_margin",
            "sum",
        ),
        feature_importance_percent_sum=(
            "importance_percent",
            "sum",
        ),
        mean_signed_shap_sum=(
            "mean_signed_shap_raw_margin",
            "sum",
        ),
        feature_count=(
            "feature_name",
            "count",
        ),
    )
)


global_absolute_total = float(
    global_domain_df[
        "mean_absolute_shap_sum"
    ]
    .sum()
)


if global_absolute_total <= 0:

    raise RuntimeError(
        "Global SHAP domain importance total is zero."
    )


global_domain_df[
    "absolute_domain_importance_percent"
] = (
    global_domain_df[
        "mean_absolute_shap_sum"
    ]
    /
    global_absolute_total
    *
    100.0
)


global_domain_df = (
    global_domain_df.sort_values(
        "absolute_domain_importance_percent",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


global_domain_df[
    "global_domain_rank"
] = (
    np.arange(
        1,
        len(
            global_domain_df
        )
        +
        1,
    )
)


# =============================================================================
# 12. THRESHOLD-FLAGGED DOMAIN IMPORTANCE
# =============================================================================

flagged_with_domain = (
    flagged_global_df.merge(
        feature_domain_df,
        on="feature_name",
        how="left",
        validate="one_to_one",
    )
)


flagged_domain_df = (
    flagged_with_domain.groupby(
        "explanation_domain",
        as_index=False,
    )
    .agg(
        mean_absolute_shap_sum=(
            "mean_absolute_shap_raw_margin",
            "sum",
        ),
        feature_importance_percent_sum=(
            "importance_percent",
            "sum",
        ),
        mean_signed_shap_sum=(
            "mean_signed_shap_raw_margin",
            "sum",
        ),
        feature_count=(
            "feature_name",
            "count",
        ),
    )
)


flagged_absolute_total = float(
    flagged_domain_df[
        "mean_absolute_shap_sum"
    ]
    .sum()
)


if flagged_absolute_total <= 0:

    raise RuntimeError(
        "Flagged-cell SHAP domain importance total is zero."
    )


flagged_domain_df[
    "absolute_domain_importance_percent"
] = (
    flagged_domain_df[
        "mean_absolute_shap_sum"
    ]
    /
    flagged_absolute_total
    *
    100.0
)


flagged_domain_df = (
    flagged_domain_df.sort_values(
        "absolute_domain_importance_percent",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


flagged_domain_df[
    "flagged_domain_rank"
] = (
    np.arange(
        1,
        len(
            flagged_domain_df
        )
        +
        1,
    )
)


# =============================================================================
# 13. IDENTIFY FULL SHAP FEATURE MATRIX
# =============================================================================

shap_columns = [
    column
    for column in shap_df.columns
    if column.startswith(
        "shap__"
    )
]


if len(
    shap_columns
) != EXPECTED_FEATURES:

    raise RuntimeError(
        f"Expected {EXPECTED_FEATURES} SHAP feature columns, "
        f"found {len(shap_columns)}."
    )


expected_shap_columns = {
    f"shap__{feature_name}"
    for feature_name in feature_names
}


if set(
    shap_columns
) != expected_shap_columns:

    raise RuntimeError(
        "SHAP feature columns do not match the 55 frozen predictors."
    )


# =============================================================================
# 14. CREATE DOMAIN -> SHAP COLUMN MAP
# =============================================================================

domain_feature_map = {
    domain: []
    for domain in DOMAIN_ORDER
}


for feature_name in feature_names:

    domain = assign_domain(
        feature_name
    )

    domain_feature_map[
        domain
    ].append(
        f"shap__{feature_name}"
    )


for domain in DOMAIN_ORDER:

    if len(
        domain_feature_map[
            domain
        ]
    ) == 0:

        raise RuntimeError(
            f"No SHAP features assigned to domain: "
            f"{domain}"
        )


# =============================================================================
# 15. CREATE PER-CELL DOMAIN SHAP EXPLANATIONS
# =============================================================================

print("\nCreating per-grid domain explanations...")


local_domain_records = []


for _, row in shap_df.iterrows():

    domain_values = {}


    for domain in DOMAIN_ORDER:

        columns = (
            domain_feature_map[
                domain
            ]
        )


        domain_array = (
            row[
                columns
            ]
            .to_numpy(
                dtype=float
            )
        )


        signed_sum = float(
            domain_array.sum()
        )


        absolute_sum = float(
            np.abs(
                domain_array
            ).sum()
        )


        domain_values[
            domain
        ] = {
            "signed": (
                signed_sum
            ),
            "absolute": (
                absolute_sum
            ),
        }


    ranked_domains = sorted(
        DOMAIN_ORDER,
        key=lambda domain:
        domain_values[
            domain
        ][
            "absolute"
        ],
        reverse=True,
    )


    for domain_rank, domain in enumerate(
        ranked_domains,
        start=1,
    ):

        signed_value = float(
            domain_values[
                domain
            ][
                "signed"
            ]
        )


        local_domain_records.append(
            {
                "grid_id": (
                    row[
                        "grid_id"
                    ]
                ),
                "origin_year": int(
                    row[
                        "origin_year"
                    ]
                ),
                "pressure_period": (
                    row[
                        "pressure_period"
                    ]
                ),
                "urbanization_pressure_score": float(
                    row[
                        "urbanization_pressure_score"
                    ]
                ),
                "pressure_percentile": float(
                    row[
                        "pressure_percentile"
                    ]
                ),
                "pressure_rank": int(
                    row[
                        "pressure_rank"
                    ]
                ),
                "pressure_category": (
                    row[
                        "pressure_category"
                    ]
                ),
                "threshold_flag": int(
                    row[
                        "threshold_flag"
                    ]
                ),
                "domain_rank": (
                    domain_rank
                ),
                "explanation_domain": (
                    domain
                ),
                "signed_shap_raw_margin": (
                    signed_value
                ),
                "absolute_shap_sum": float(
                    domain_values[
                        domain
                    ][
                        "absolute"
                    ]
                ),
                "direction": (
                    shap_direction(
                        signed_value
                    )
                ),
            }
        )


local_domain_df = pd.DataFrame(
    local_domain_records
)


# =============================================================================
# 16. ENRICH LOCAL TOP-FEATURE EXPLANATIONS
# =============================================================================

enriched_local_df = (
    local_df.merge(
        feature_domain_df[
            [
                "feature_name",
                "explanation_domain",
                "human_readable_feature",
            ]
        ],
        on="feature_name",
        how="left",
        validate="many_to_one",
    )
)


if enriched_local_df[
    "explanation_domain"
].isna().any():

    raise RuntimeError(
        "Some local SHAP explanations could not be assigned to a domain."
    )


if enriched_local_df[
    "human_readable_feature"
].isna().any():

    raise RuntimeError(
        "Some local SHAP explanations lack a readable feature label."
    )


# =============================================================================
# 17. CREATE DASHBOARD-FRIENDLY GRID EXPLANATIONS
# =============================================================================

print(
    "Creating dashboard-ready grid explanations..."
)


dashboard_records = []


for grid_id, grid_domain_df in (
    local_domain_df.groupby(
        "grid_id",
        sort=False,
    )
):

    grid_domain_df = (
        grid_domain_df.sort_values(
            "domain_rank"
        )
        .reset_index(
            drop=True
        )
    )


    grid_feature_df = (
        enriched_local_df[
            enriched_local_df[
                "grid_id"
            ]
            ==
            grid_id
        ]
        .sort_values(
            "local_importance_rank"
        )
        .reset_index(
            drop=True
        )
    )


    if len(
        grid_domain_df
    ) != len(
        DOMAIN_ORDER
    ):

        raise RuntimeError(
            f"Unexpected domain count for grid "
            f"{grid_id}."
        )


    if len(
        grid_feature_df
    ) != EXPECTED_LOCAL_TOP_FEATURES:

        raise RuntimeError(
            f"Unexpected top-feature count for grid "
            f"{grid_id}."
        )


    first_domain = (
        grid_domain_df.iloc[
            0
        ]
    )

    second_domain = (
        grid_domain_df.iloc[
            1
        ]
    )

    third_domain = (
        grid_domain_df.iloc[
            2
        ]
    )


    top_feature_1 = (
        grid_feature_df.iloc[
            0
        ]
    )

    top_feature_2 = (
        grid_feature_df.iloc[
            1
        ]
    )

    top_feature_3 = (
        grid_feature_df.iloc[
            2
        ]
    )


    dashboard_records.append(
        {
            "grid_id": (
                grid_id
            ),
            "origin_year": int(
                first_domain[
                    "origin_year"
                ]
            ),
            "pressure_period": (
                first_domain[
                    "pressure_period"
                ]
            ),
            "urbanization_pressure_score": float(
                first_domain[
                    "urbanization_pressure_score"
                ]
            ),
            "pressure_percentile": float(
                first_domain[
                    "pressure_percentile"
                ]
            ),
            "pressure_rank": int(
                first_domain[
                    "pressure_rank"
                ]
            ),
            "pressure_category": (
                first_domain[
                    "pressure_category"
                ]
            ),
            "threshold_flag": int(
                first_domain[
                    "threshold_flag"
                ]
            ),

            # -----------------------------------------------------------------
            # Dominant explanation domains
            # -----------------------------------------------------------------

            "dominant_domain": (
                first_domain[
                    "explanation_domain"
                ]
            ),
            "dominant_domain_direction": (
                first_domain[
                    "direction"
                ]
            ),
            "dominant_domain_signed_shap": float(
                first_domain[
                    "signed_shap_raw_margin"
                ]
            ),

            "second_domain": (
                second_domain[
                    "explanation_domain"
                ]
            ),
            "second_domain_direction": (
                second_domain[
                    "direction"
                ]
            ),
            "second_domain_signed_shap": float(
                second_domain[
                    "signed_shap_raw_margin"
                ]
            ),

            "third_domain": (
                third_domain[
                    "explanation_domain"
                ]
            ),
            "third_domain_direction": (
                third_domain[
                    "direction"
                ]
            ),
            "third_domain_signed_shap": float(
                third_domain[
                    "signed_shap_raw_margin"
                ]
            ),

            # -----------------------------------------------------------------
            # Top individual model feature 1
            # -----------------------------------------------------------------

            "top_feature_1": (
                top_feature_1[
                    "human_readable_feature"
                ]
            ),
            "top_feature_1_technical_name": (
                top_feature_1[
                    "feature_name"
                ]
            ),
            "top_feature_1_domain": (
                top_feature_1[
                    "explanation_domain"
                ]
            ),
            "top_feature_1_direction": (
                top_feature_1[
                    "direction"
                ]
            ),
            "top_feature_1_value": float(
                top_feature_1[
                    "feature_value"
                ]
            ),
            "top_feature_1_shap": float(
                top_feature_1[
                    "shap_value_raw_margin"
                ]
            ),

            # -----------------------------------------------------------------
            # Top individual model feature 2
            # -----------------------------------------------------------------

            "top_feature_2": (
                top_feature_2[
                    "human_readable_feature"
                ]
            ),
            "top_feature_2_technical_name": (
                top_feature_2[
                    "feature_name"
                ]
            ),
            "top_feature_2_domain": (
                top_feature_2[
                    "explanation_domain"
                ]
            ),
            "top_feature_2_direction": (
                top_feature_2[
                    "direction"
                ]
            ),
            "top_feature_2_value": float(
                top_feature_2[
                    "feature_value"
                ]
            ),
            "top_feature_2_shap": float(
                top_feature_2[
                    "shap_value_raw_margin"
                ]
            ),

            # -----------------------------------------------------------------
            # Top individual model feature 3
            # -----------------------------------------------------------------

            "top_feature_3": (
                top_feature_3[
                    "human_readable_feature"
                ]
            ),
            "top_feature_3_technical_name": (
                top_feature_3[
                    "feature_name"
                ]
            ),
            "top_feature_3_domain": (
                top_feature_3[
                    "explanation_domain"
                ]
            ),
            "top_feature_3_direction": (
                top_feature_3[
                    "direction"
                ]
            ),
            "top_feature_3_value": float(
                top_feature_3[
                    "feature_value"
                ]
            ),
            "top_feature_3_shap": float(
                top_feature_3[
                    "shap_value_raw_margin"
                ]
            ),

            # -----------------------------------------------------------------
            # Interpretation warning
            # -----------------------------------------------------------------

            "explanation_scope": (
                "MODEL_EXPLANATION_NOT_CAUSAL_CLAIM"
            ),

            "shap_output_space": (
                "RAW_XGBOOST_MARGIN"
            ),
        }
    )


dashboard_df = pd.DataFrame(
    dashboard_records
)


dashboard_df = (
    dashboard_df.sort_values(
        "pressure_rank"
    )
    .reset_index(
        drop=True
    )
)


# =============================================================================
# 18. QA
# =============================================================================

all_features_mapped = bool(
    len(
        feature_domain_df
    )
    ==
    EXPECTED_FEATURES
    and
    feature_domain_df[
        "explanation_domain"
    ]
    .notna()
    .all()
)


all_features_have_readable_names = bool(
    feature_domain_df[
        "human_readable_feature"
    ]
    .notna()
    .all()
)


expected_domain_rows = (
    EXPECTED_CELLS
    *
    len(
        DOMAIN_ORDER
    )
)


local_domain_count_valid = (
    len(
        local_domain_df
    )
    ==
    expected_domain_rows
)


dashboard_row_count_valid = (
    len(
        dashboard_df
    )
    ==
    EXPECTED_CELLS
)


dashboard_unique_grid_ids = bool(
    not dashboard_df[
        "grid_id"
    ]
    .duplicated()
    .any()
)


flagged_dashboard_count = int(
    dashboard_df[
        "threshold_flag"
    ]
    .astype(
        int
    )
    .sum()
)


flagged_dashboard_count_valid = (
    flagged_dashboard_count
    ==
    EXPECTED_FLAGGED_CELLS
)


global_domain_percent_sum = float(
    global_domain_df[
        "absolute_domain_importance_percent"
    ]
    .sum()
)


global_domain_sum_valid = bool(
    np.isclose(
        global_domain_percent_sum,
        100.0,
        atol=1e-8,
    )
)


flagged_domain_percent_sum = float(
    flagged_domain_df[
        "absolute_domain_importance_percent"
    ]
    .sum()
)


flagged_domain_sum_valid = bool(
    np.isclose(
        flagged_domain_percent_sum,
        100.0,
        atol=1e-8,
    )
)


no_missing_dashboard_explanations = bool(
    dashboard_df[
        [
            "dominant_domain",
            "second_domain",
            "third_domain",
            "top_feature_1",
            "top_feature_2",
            "top_feature_3",
        ]
    ]
    .notna()
    .all()
    .all()
)


origin_year_valid = bool(
    (
        dashboard_df[
            "origin_year"
        ]
        ==
        EXPECTED_ORIGIN_YEAR
    )
    .all()
)


pressure_period_valid = bool(
    (
        dashboard_df[
            "pressure_period"
        ]
        .astype(
            str
        )
        ==
        EXPECTED_PRESSURE_PERIOD
    )
    .all()
)


pressure_scores_unchanged = bool(
    np.allclose(
        dashboard_df[
            "urbanization_pressure_score"
        ]
        .to_numpy(
            dtype=float
        ),
        shap_df.sort_values(
            "pressure_rank"
        )[
            "urbanization_pressure_score"
        ]
        .to_numpy(
            dtype=float
        ),
        rtol=0.0,
        atol=1e-12,
    )
)


model_retrained = False

model_rescored = False

threshold_changed = False

shap_recalculated = False

causal_claim_made = False

probability_contribution_claim_made = False


overall_status = (
    "PASS"
    if all(
        [
            all_features_mapped,
            all_features_have_readable_names,
            local_domain_count_valid,
            dashboard_row_count_valid,
            dashboard_unique_grid_ids,
            flagged_dashboard_count_valid,
            global_domain_sum_valid,
            flagged_domain_sum_valid,
            no_missing_dashboard_explanations,
            origin_year_valid,
            pressure_period_valid,
            pressure_scores_unchanged,
            not model_retrained,
            not model_rescored,
            not threshold_changed,
            not shap_recalculated,
            not causal_claim_made,
            not probability_contribution_claim_made,
        ]
    )
    else
    "REVIEW_REQUIRED"
)


# =============================================================================
# 19. QA REPORT
# =============================================================================

qa_df = pd.DataFrame(
    [
        {
            "frozen_features": (
                EXPECTED_FEATURES
            ),
            "all_features_mapped_to_domain": (
                all_features_mapped
            ),
            "all_features_have_readable_names": (
                all_features_have_readable_names
            ),
            "explanation_domains": (
                len(
                    DOMAIN_ORDER
                )
            ),
            "local_domain_rows": (
                len(
                    local_domain_df
                )
            ),
            "local_domain_row_count_valid": (
                local_domain_count_valid
            ),
            "dashboard_rows": (
                len(
                    dashboard_df
                )
            ),
            "dashboard_row_count_valid": (
                dashboard_row_count_valid
            ),
            "dashboard_grid_ids_unique": (
                dashboard_unique_grid_ids
            ),
            "flagged_dashboard_cells": (
                flagged_dashboard_count
            ),
            "flagged_count_valid": (
                flagged_dashboard_count_valid
            ),
            "global_domain_importance_sum_valid": (
                global_domain_sum_valid
            ),
            "flagged_domain_importance_sum_valid": (
                flagged_domain_sum_valid
            ),
            "dashboard_explanations_complete": (
                no_missing_dashboard_explanations
            ),
            "origin_year_valid": (
                origin_year_valid
            ),
            "pressure_period_valid": (
                pressure_period_valid
            ),
            "pressure_scores_unchanged": (
                pressure_scores_unchanged
            ),
            "model_retrained": (
                model_retrained
            ),
            "model_rescored": (
                model_rescored
            ),
            "threshold_changed": (
                threshold_changed
            ),
            "shap_recalculated": (
                shap_recalculated
            ),
            "causal_claim_made": (
                causal_claim_made
            ),
            "probability_contribution_claim_made": (
                probability_contribution_claim_made
            ),
            "overall_step15b_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 20. SUMMARY REPORT
# =============================================================================

summary_df = pd.DataFrame(
    [
        {
            "stage": (
                "SHAP explanation domain summarization"
            ),
            "origin_year": (
                EXPECTED_ORIGIN_YEAR
            ),
            "pressure_period": (
                EXPECTED_PRESSURE_PERIOD
            ),
            "explained_cells": (
                EXPECTED_CELLS
            ),
            "explanation_domains": (
                len(
                    DOMAIN_ORDER
                )
            ),
            "domain_names": (
                ";".join(
                    DOMAIN_ORDER
                )
            ),
            "threshold_flagged_cells": (
                flagged_dashboard_count
            ),
            "dashboard_ready_explanations": (
                dashboard_row_count_valid
            ),
            "explanations_are_causal_claims": (
                False
            ),
            "shap_values_are_probability_contributions": (
                False
            ),
            "model_or_threshold_changed": (
                False
            ),
            "overall_step15b_status": (
                overall_status
            ),
        }
    ]
)


# =============================================================================
# 21. SAVE OUTPUTS
# =============================================================================

feature_domain_df.to_csv(
    FEATURE_DOMAIN_MAP_CSV,
    index=False,
)


global_domain_df.to_csv(
    GLOBAL_DOMAIN_CSV,
    index=False,
)


flagged_domain_df.to_csv(
    FLAGGED_DOMAIN_CSV,
    index=False,
)


local_domain_df.to_csv(
    LOCAL_DOMAIN_CSV,
    index=False,
)


dashboard_df.to_csv(
    DASHBOARD_EXPLANATIONS_CSV,
    index=False,
)


qa_df.to_csv(
    QA_CSV,
    index=False,
)


summary_df.to_csv(
    SUMMARY_CSV,
    index=False,
)


# =============================================================================
# 22. TERMINAL OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("GLOBAL EXPLANATION DOMAINS")
print("=" * 100)


print(
    global_domain_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("THRESHOLD-FLAGGED EXPLANATION DOMAINS")
print("=" * 100)


print(
    flagged_domain_df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("TOP PRESSURE CELL - DASHBOARD EXPLANATION")
print("=" * 100)


print(
    dashboard_df.head(
        1
    ).to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STEP 15B QA")
print("=" * 100)


print(
    f"All 55 features mapped                 : "
    f"{all_features_mapped}"
)

print(
    f"All features have readable labels      : "
    f"{all_features_have_readable_names}"
)

print(
    f"Local domain rows valid                : "
    f"{local_domain_count_valid}"
)

print(
    f"Dashboard rows = 4,620                 : "
    f"{dashboard_row_count_valid}"
)

print(
    f"Dashboard grid IDs unique              : "
    f"{dashboard_unique_grid_ids}"
)

print(
    f"Flagged count remains 255              : "
    f"{flagged_dashboard_count_valid}"
)

print(
    f"Global domains sum to 100%             : "
    f"{global_domain_sum_valid}"
)

print(
    f"Flagged domains sum to 100%            : "
    f"{flagged_domain_sum_valid}"
)

print(
    f"Dashboard explanations complete        : "
    f"{no_missing_dashboard_explanations}"
)

print(
    f"Origin year remains 2025               : "
    f"{origin_year_valid}"
)

print(
    f"Pressure period remains 2026-2028      : "
    f"{pressure_period_valid}"
)

print(
    f"Pressure scores unchanged              : "
    f"{pressure_scores_unchanged}"
)

print(
    f"Model retrained                        : "
    f"{model_retrained}"
)

print(
    f"Model rescored                         : "
    f"{model_rescored}"
)

print(
    f"Threshold changed                      : "
    f"{threshold_changed}"
)

print(
    f"SHAP recalculated                      : "
    f"{shap_recalculated}"
)

print(
    f"Causal claim made                      : "
    f"{causal_claim_made}"
)

print(
    f"Probability contribution claim made    : "
    f"{probability_contribution_claim_made}"
)

print(
    f"Overall Step 15B status                : "
    f"{overall_status}"
)


print("\nGenerated outputs:")


for output_file in [
    FEATURE_DOMAIN_MAP_CSV,
    GLOBAL_DOMAIN_CSV,
    FLAGGED_DOMAIN_CSV,
    LOCAL_DOMAIN_CSV,
    DASHBOARD_EXPLANATIONS_CSV,
    QA_CSV,
    SUMMARY_CSV,
]:

    print(
        f" - {relative_path(output_file)}"
    )


print("\nIMPORTANT:")

print(
    "1. Step 15B does not alter the production model or urbanization-pressure scores."
)

print(
    "2. The 55 technical predictors are grouped into five understandable explanation domains."
)

print(
    "3. Dominant domain means the domain with the largest absolute SHAP contribution for that grid cell."
)

print(
    "4. Positive signed SHAP pushes the model toward higher urbanization pressure."
)

print(
    "5. Negative signed SHAP pushes the model toward lower urbanization pressure."
)

print(
    "6. SHAP contributions are in the raw XGBoost margin space, not percentage probabilities."
)

print(
    "7. SHAP explanations describe model behaviour and must not be presented as proven causality."
)

print(
    "8. The dashboard-ready CSV contains one explanation record for every one of the 4,620 scored cells."
)

print(
    "9. After this step passes, the XAI stage can be frozen before moving to Development Suitability / GIS-MCDA-AHP."
)

print("=" * 100)