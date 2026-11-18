from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, kendalltau


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[4]

PRIORITY_DIR = ROOT / "outputs" / "priority"
REPORT_DIR = ROOT / "outputs" / "reports"

VALIDATION_DIR = (
    ROOT
    / "experiments"
    / "validation"
    / "priority_validation"
)

OUTPUT_DIR = VALIDATION_DIR / "outputs"
FIGURE_DIR = VALIDATION_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


MATRIX_PATH = (
    PRIORITY_DIR
    / "31d_priority_decision_matrix_normalized.csv"
)

METADATA_PATH = (
    PRIORITY_DIR
    / "31d_priority_criteria_metadata.csv"
)

WEIGHTS_PATH = (
    PRIORITY_DIR
    / "31e_critic_weights.csv"
)

BASELINE_RANKING_PATH = (
    PRIORITY_DIR
    / "32_topsis_priority_ranking.csv"
)

STEP31D_AUDIT_PATH = (
    REPORT_DIR
    / "31d_priority_decision_matrix_audit.csv"
)

STEP31E_AUDIT_PATH = (
    REPORT_DIR
    / "31e_critic_weighting_audit.csv"
)

STEP32_AUDIT_PATH = (
    REPORT_DIR
    / "32_topsis_priority_ranking_audit.csv"
)


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_GRIDS = 2194
EXPECTED_CRITERIA = 6

TOL = 1e-12

TOP10PCT_K = int(
    np.ceil(
        EXPECTED_GRIDS * 0.10
    )
)

TOP25PCT_K = int(
    np.ceil(
        EXPECTED_GRIDS * 0.25
    )
)

# Additional stronger sensitivity specifically for
# C6 = Modeled Intervention Benefit.
C6_RETAINED_FRACTIONS = [
    0.75,
    0.50,
    0.25,
    0.00,
]


# ============================================================
# HELPERS
# ============================================================

def require_file(path, label):

    if not path.exists():

        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def require_columns(
    df,
    columns,
    label,
):

    missing = [
        col
        for col in columns
        if col not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"{label} missing columns: "
            f"{missing}"
        )


def require_all_pass(
    path,
    label,
):

    require_file(
        path,
        label,
    )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    require_columns(
        df,
        ["status"],
        label,
    )

    status = (
        df["status"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if not status.eq(
        "PASS"
    ).all():

        raise RuntimeError(
            f"{label} is not fully PASS."
        )


def normalize_nonnegative_weights(
    values,
):

    """
    Allow zero only for intentional
    criterion-removal scenarios.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    if not np.isfinite(
        values
    ).all():

        raise RuntimeError(
            "Weight vector contains "
            "non-finite values."
        )

    if (
        values < 0
    ).any():

        raise RuntimeError(
            "Negative weight found."
        )

    total = float(
        values.sum()
    )

    if total <= TOL:

        raise RuntimeError(
            "Weight sum is zero."
        )

    return (
        values / total
    )


def critic_weights(
    matrix_df,
):

    """
    Reproduce Step-31E CRITIC method:

    information_j
        = population_std_j
          * sum_k(1 - Pearson_corr_jk)

    weight_j
        = information_j / total_information
    """

    X = (
        matrix_df
        .astype(float)
    )

    corr = X.corr(
        method="pearson"
    )

    if corr.isna().any().any():

        raise RuntimeError(
            "CRITIC correlation matrix "
            "contains NaN."
        )

    std_population = X.std(
        axis=0,
        ddof=0,
    )

    conflict = (
        1.0 - corr
    ).sum(axis=1)

    information = (
        std_population
        * conflict
    )

    information = information.clip(
        lower=0.0
    )

    total = float(
        information.sum()
    )

    if total <= TOL:

        raise RuntimeError(
            "Invalid CRITIC information."
        )

    return (
        information
        / total
    ).to_numpy(
        dtype=float
    )


def topsis(
    matrix_array,
    grid_ids,
    weights,
):

    """
    Same Step-32 TOPSIS formulation.

    Step-31D matrix is already:
        min-max normalized,
        higher = stronger priority.

    Therefore no second normalization.
    """

    weights = (
        normalize_nonnegative_weights(
            weights
        )
    )

    weighted = (
        matrix_array
        * weights.reshape(
            1,
            -1,
        )
    )

    ideal_best = np.max(
        weighted,
        axis=0,
    )

    ideal_worst = np.min(
        weighted,
        axis=0,
    )

    d_plus = np.sqrt(
        np.sum(
            (
                weighted
                - ideal_best
            ) ** 2,
            axis=1,
        )
    )

    d_minus = np.sqrt(
        np.sum(
            (
                weighted
                - ideal_worst
            ) ** 2,
            axis=1,
        )
    )

    denominator = (
        d_plus
        + d_minus
    )

    if (
        denominator <= TOL
    ).any():

        raise RuntimeError(
            "TOPSIS denominator "
            "zero/near-zero."
        )

    score = (
        d_minus
        / denominator
    )

    if not np.isfinite(
        score
    ).all():

        raise RuntimeError(
            "Non-finite TOPSIS score."
        )

    score = np.clip(
        score,
        0.0,
        1.0,
    )

    order_df = pd.DataFrame(
        {
            "row_index":
                np.arange(
                    len(grid_ids)
                ),

            "grid_id":
                np.asarray(
                    grid_ids,
                    dtype=str,
                ),

            "score":
                score,
        }
    )

    order_df = (
        order_df
        .sort_values(
            [
                "score",
                "grid_id",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )
    )

    ranks = np.empty(
        len(grid_ids),
        dtype=int,
    )

    ranks[
        order_df[
            "row_index"
        ].to_numpy(dtype=int)
    ] = np.arange(
        1,
        len(grid_ids) + 1,
    )

    return (
        score,
        ranks,
    )


def overlap_fraction(
    baseline_rank,
    scenario_rank,
    k,
):

    baseline_set = set(
        np.flatnonzero(
            baseline_rank <= k
        )
    )

    scenario_set = set(
        np.flatnonzero(
            scenario_rank <= k
        )
    )

    return float(
        len(
            baseline_set
            & scenario_set
        )
        / len(
            baseline_set
        )
    )


def rank_metrics(
    baseline_rank,
    scenario_rank,
):

    abs_shift = np.abs(
        scenario_rank
        - baseline_rank
    )

    spearman = float(
        spearmanr(
            baseline_rank,
            scenario_rank,
        ).statistic
    )

    kendall = float(
        kendalltau(
            baseline_rank,
            scenario_rank,
            variant="b",
        ).statistic
    )

    rank1_index = int(
        np.argmin(
            baseline_rank
        )
    )

    return {
        "spearman_rho":
            spearman,

        "kendall_tau_b":
            kendall,

        "top10_overlap_pct":
            100.0
            * overlap_fraction(
                baseline_rank,
                scenario_rank,
                10,
            ),

        "top50_overlap_pct":
            100.0
            * overlap_fraction(
                baseline_rank,
                scenario_rank,
                50,
            ),

        "top100_overlap_pct":
            100.0
            * overlap_fraction(
                baseline_rank,
                scenario_rank,
                100,
            ),

        "top10pct_overlap_pct":
            100.0
            * overlap_fraction(
                baseline_rank,
                scenario_rank,
                TOP10PCT_K,
            ),

        "top25pct_overlap_pct":
            100.0
            * overlap_fraction(
                baseline_rank,
                scenario_rank,
                TOP25PCT_K,
            ),

        "mean_abs_rank_shift":
            float(
                np.mean(
                    abs_shift
                )
            ),

        "median_abs_rank_shift":
            float(
                np.median(
                    abs_shift
                )
            ),

        "p95_abs_rank_shift":
            float(
                np.quantile(
                    abs_shift,
                    0.95,
                )
            ),

        "max_abs_rank_shift":
            int(
                np.max(
                    abs_shift
                )
            ),

        "grids_shift_gt25":
            int(
                np.sum(
                    abs_shift > 25
                )
            ),

        "grids_shift_gt50":
            int(
                np.sum(
                    abs_shift > 50
                )
            ),

        "grids_shift_gt100":
            int(
                np.sum(
                    abs_shift > 100
                )
            ),

        "baseline_rank1_new_rank":
            int(
                scenario_rank[
                    rank1_index
                ]
            ),
    }


# ============================================================
# LOAD
# ============================================================

print(
    "\n============================================================"
)

print(
    "PRIORITY VALIDATION 03"
)

print(
    "CRITERION REMOVAL + C6 SENSITIVITY"
)

print(
    "============================================================\n"
)


for path, label in [
    (
        MATRIX_PATH,
        "Step 31D matrix",
    ),
    (
        METADATA_PATH,
        "Step 31D metadata",
    ),
    (
        WEIGHTS_PATH,
        "Step 31E weights",
    ),
    (
        BASELINE_RANKING_PATH,
        "Step 32 ranking",
    ),
]:

    require_file(
        path,
        label,
    )


require_all_pass(
    STEP31D_AUDIT_PATH,
    "Step 31D audit",
)

require_all_pass(
    STEP31E_AUDIT_PATH,
    "Step 31E audit",
)

require_all_pass(
    STEP32_AUDIT_PATH,
    "Step 32 audit",
)


matrix_df = pd.read_csv(
    MATRIX_PATH,
    dtype={
        "grid_id": str,
    },
    low_memory=False,
)

metadata = pd.read_csv(
    METADATA_PATH,
    low_memory=False,
)

weights_df = pd.read_csv(
    WEIGHTS_PATH,
    low_memory=False,
)

baseline_df = pd.read_csv(
    BASELINE_RANKING_PATH,
    dtype={
        "grid_id": str,
    },
    low_memory=False,
)


# ============================================================
# VALIDATE
# ============================================================

require_columns(
    metadata,
    [
        "criterion_order",
        "criterion_code",
        "decision_dimension",
        "final_field_name",
    ],
    "Metadata",
)


require_columns(
    weights_df,
    [
        "criterion_order",
        "criterion_code",
        "final_field_name",
        "critic_weight",
    ],
    "Weights",
)


require_columns(
    baseline_df,
    [
        "grid_id",
        "topsis_priority_score",
        "topsis_rank",
    ],
    "Baseline ranking",
)


metadata = (
    metadata
    .sort_values(
        "criterion_order"
    )
    .reset_index(drop=True)
)


weights_df = (
    weights_df
    .sort_values(
        "criterion_order"
    )
    .reset_index(drop=True)
)


if len(
    metadata
) != EXPECTED_CRITERIA:

    raise RuntimeError(
        "Expected six criteria."
    )


fields = (
    metadata[
        "final_field_name"
    ]
    .astype(str)
    .tolist()
)


codes = (
    metadata[
        "criterion_code"
    ]
    .astype(str)
    .tolist()
)


dimensions = (
    metadata[
        "decision_dimension"
    ]
    .astype(str)
    .tolist()
)


require_columns(
    matrix_df,
    [
        "grid_id",
        *fields,
    ],
    "Decision matrix",
)


matrix_df["grid_id"] = (
    matrix_df[
        "grid_id"
    ]
    .astype(str)
    .str.strip()
)


if len(
    matrix_df
) != EXPECTED_GRIDS:

    raise RuntimeError(
        f"Expected {EXPECTED_GRIDS} grids."
    )


if (
    matrix_df[
        "grid_id"
    ].nunique()
    != EXPECTED_GRIDS
):

    raise RuntimeError(
        "Unique-grid count mismatch."
    )


matrix_array = (
    matrix_df[
        fields
    ]
    .apply(
        pd.to_numeric,
        errors="raise",
    )
    .to_numpy(
        dtype=float
    )
)


if not np.isfinite(
    matrix_array
).all():

    raise RuntimeError(
        "Decision matrix contains "
        "non-finite values."
    )


base_weights = (
    weights_df
    .set_index(
        "final_field_name"
    )[
        "critic_weight"
    ]
    .reindex(
        fields
    )
    .to_numpy(
        dtype=float
    )
)


base_weights = (
    normalize_nonnegative_weights(
        base_weights
    )
)


# ============================================================
# REPRODUCE BASELINE
# ============================================================

baseline_df["grid_id"] = (
    baseline_df[
        "grid_id"
    ]
    .astype(str)
    .str.strip()
)


baseline_lookup = (
    baseline_df
    .set_index(
        "grid_id"
    )
)


saved_rank = (
    baseline_lookup
    .loc[
        matrix_df[
            "grid_id"
        ],
        "topsis_rank",
    ]
    .to_numpy(
        dtype=int
    )
)


saved_score = (
    baseline_lookup
    .loc[
        matrix_df[
            "grid_id"
        ],
        "topsis_priority_score",
    ]
    .to_numpy(
        dtype=float
    )
)


calc_score, calc_rank = (
    topsis(
        matrix_array,
        matrix_df[
            "grid_id"
        ].to_numpy(
            dtype=str
        ),
        base_weights,
    )
)


max_score_difference = float(
    np.max(
        np.abs(
            calc_score
            - saved_score
        )
    )
)


exact_rank_match = bool(
    np.array_equal(
        calc_rank,
        saved_rank,
    )
)


print(
    f"Baseline max score difference : "
    f"{max_score_difference:.12e}"
)

print(
    f"Baseline exact rank match     : "
    f"{exact_rank_match}"
)


if max_score_difference > 1e-10:

    raise RuntimeError(
        "Baseline score "
        "reproduction failed."
    )


if not exact_rank_match:

    raise RuntimeError(
        "Baseline rank "
        "reproduction failed."
    )


print(
    "PASS - Step-32 baseline "
    "reproduced exactly."
)


baseline_rank1_grid = str(
    matrix_df.loc[
        np.argmin(
            saved_rank
        ),
        "grid_id",
    ]
)


print(
    f"Baseline Rank-1 grid : "
    f"{baseline_rank1_grid}"
)


# ============================================================
# CRITERION REMOVAL TESTS
# ============================================================

print(
    "\n============================================================"
)

print(
    "1. LEAVE-ONE-CRITERION-OUT SENSITIVITY"
)

print(
    "============================================================"
)


scenario_rows = []
scenario_rank_store = {}


for removed_index in range(
    EXPECTED_CRITERIA
):

    removed_code = (
        codes[
            removed_index
        ]
    )

    removed_dimension = (
        dimensions[
            removed_index
        ]
    )

    removed_field = (
        fields[
            removed_index
        ]
    )


    # --------------------------------------------------------
    # Method A:
    # Remove criterion and renormalize
    # original CRITIC weights.
    # --------------------------------------------------------

    fixed_weights = (
        base_weights.copy()
    )

    fixed_weights[
        removed_index
    ] = 0.0

    fixed_weights = (
        normalize_nonnegative_weights(
            fixed_weights
        )
    )


    fixed_score, fixed_rank = (
        topsis(
            matrix_array,
            matrix_df[
                "grid_id"
            ].to_numpy(
                dtype=str
            ),
            fixed_weights,
        )
    )


    fixed_metrics = rank_metrics(
        saved_rank,
        fixed_rank,
    )


    scenario_id_fixed = (
        f"REMOVE_{removed_code}_"
        f"RENORMALIZED"
    )


    scenario_rank_store[
        scenario_id_fixed
    ] = fixed_rank


    scenario_rows.append(
        {
            "scenario_id":
                scenario_id_fixed,

            "scenario_family":
                "LEAVE_ONE_OUT",

            "removed_criterion":
                removed_code,

            "removed_dimension":
                removed_dimension,

            "removed_field":
                removed_field,

            "weight_method":
                "BASELINE_WEIGHTS_RENORMALIZED",

            **fixed_metrics,
        }
    )


    # --------------------------------------------------------
    # Method B:
    # Remove criterion and recompute CRITIC
    # using only remaining five criteria.
    # --------------------------------------------------------

    remaining_indices = [
        i
        for i in range(
            EXPECTED_CRITERIA
        )
        if i != removed_index
    ]


    remaining_fields = [
        fields[i]
        for i in remaining_indices
    ]


    reduced_critic_weights = (
        critic_weights(
            matrix_df[
                remaining_fields
            ]
        )
    )


    recomputed_full_weights = (
        np.zeros(
            EXPECTED_CRITERIA,
            dtype=float,
        )
    )


    for local_index, original_index in enumerate(
        remaining_indices
    ):

        recomputed_full_weights[
            original_index
        ] = (
            reduced_critic_weights[
                local_index
            ]
        )


    recomputed_full_weights = (
        normalize_nonnegative_weights(
            recomputed_full_weights
        )
    )


    recomputed_score, recomputed_rank = (
        topsis(
            matrix_array,
            matrix_df[
                "grid_id"
            ].to_numpy(
                dtype=str
            ),
            recomputed_full_weights,
        )
    )


    recomputed_metrics = (
        rank_metrics(
            saved_rank,
            recomputed_rank,
        )
    )


    scenario_id_recomputed = (
        f"REMOVE_{removed_code}_"
        f"CRITIC_RECOMPUTED"
    )


    scenario_rank_store[
        scenario_id_recomputed
    ] = recomputed_rank


    scenario_rows.append(
        {
            "scenario_id":
                scenario_id_recomputed,

            "scenario_family":
                "LEAVE_ONE_OUT",

            "removed_criterion":
                removed_code,

            "removed_dimension":
                removed_dimension,

            "removed_field":
                removed_field,

            "weight_method":
                "CRITIC_RECOMPUTED_ON_REMAINING_5",

            **recomputed_metrics,
        }
    )


removal_df = pd.DataFrame(
    scenario_rows
)


print(
    removal_df[
        [
            "removed_criterion",
            "removed_dimension",
            "weight_method",
            "spearman_rho",
            "kendall_tau_b",
            "top50_overlap_pct",
            "top100_overlap_pct",
            "top10pct_overlap_pct",
            "mean_abs_rank_shift",
            "p95_abs_rank_shift",
            "baseline_rank1_new_rank",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# C6 SPECIFIC DOWN-WEIGHTING
# ============================================================

print(
    "\n============================================================"
)

print(
    "2. C6 INTERVENTION-BENEFIT SPECIFIC SENSITIVITY"
)

print(
    "============================================================"
)


if "C6" not in codes:

    raise RuntimeError(
        "Could not find C6."
    )


c6_index = codes.index(
    "C6"
)


c6_rows = []


# Baseline row
baseline_metrics = (
    rank_metrics(
        saved_rank,
        saved_rank,
    )
)


c6_rows.append(
    {
        "c6_retained_fraction":
            1.00,

        "c6_effective_weight":
            base_weights[
                c6_index
            ],

        **baseline_metrics,
    }
)


for fraction in (
    C6_RETAINED_FRACTIONS
):

    candidate = (
        base_weights.copy()
    )

    candidate[
        c6_index
    ] = (
        base_weights[
            c6_index
        ]
        * fraction
    )


    candidate = (
        normalize_nonnegative_weights(
            candidate
        )
    )


    current_score, current_rank = (
        topsis(
            matrix_array,
            matrix_df[
                "grid_id"
            ].to_numpy(
                dtype=str
            ),
            candidate,
        )
    )


    current_metrics = (
        rank_metrics(
            saved_rank,
            current_rank,
        )
    )


    c6_rows.append(
        {
            "c6_retained_fraction":
                fraction,

            "c6_effective_weight":
                candidate[
                    c6_index
                ],

            **current_metrics,
        }
    )


c6_df = pd.DataFrame(
    c6_rows
)


print(
    c6_df[
        [
            "c6_retained_fraction",
            "c6_effective_weight",
            "spearman_rho",
            "kendall_tau_b",
            "top10_overlap_pct",
            "top50_overlap_pct",
            "top100_overlap_pct",
            "top10pct_overlap_pct",
            "mean_abs_rank_shift",
            "p95_abs_rank_shift",
            "baseline_rank1_new_rank",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# C6 REMOVAL GRID-LEVEL DETAIL
# ============================================================

c6_removed_rank = (
    scenario_rank_store[
        "REMOVE_C6_RENORMALIZED"
    ]
)


c6_rank_shift = (
    c6_removed_rank
    - saved_rank
)


c6_grid_df = pd.DataFrame(
    {
        "grid_id":
            matrix_df[
                "grid_id"
            ].astype(str),

        "baseline_rank":
            saved_rank,

        "rank_without_c6":
            c6_removed_rank,

        "rank_shift_without_c6":
            c6_rank_shift,

        "abs_rank_shift_without_c6":
            np.abs(
                c6_rank_shift
            ),

        "baseline_top10":
            saved_rank <= 10,

        "baseline_top50":
            saved_rank <= 50,

        "baseline_top100":
            saved_rank <= 100,

        "without_c6_top10":
            c6_removed_rank <= 10,

        "without_c6_top50":
            c6_removed_rank <= 50,

        "without_c6_top100":
            c6_removed_rank <= 100,
    }
)


c6_grid_df = (
    c6_grid_df
    .sort_values(
        "baseline_rank"
    )
    .reset_index(drop=True)
)


# ============================================================
# OVERALL SUMMARY
# ============================================================

c6_remove_fixed = (
    removal_df[
        (
            removal_df[
                "removed_criterion"
            ] == "C6"
        )
        &
        (
            removal_df[
                "weight_method"
            ]
            == "BASELINE_WEIGHTS_RENORMALIZED"
        )
    ]
    .iloc[0]
)


c6_remove_recomputed = (
    removal_df[
        (
            removal_df[
                "removed_criterion"
            ] == "C6"
        )
        &
        (
            removal_df[
                "weight_method"
            ]
            == "CRITIC_RECOMPUTED_ON_REMAINING_5"
        )
    ]
    .iloc[0]
)


summary_df = pd.DataFrame(
    [
        {
            "baseline_rank1_grid":
                baseline_rank1_grid,

            "leave_one_out_scenarios":
                len(
                    removal_df
                ),

            "minimum_leave_one_out_spearman":
                float(
                    removal_df[
                        "spearman_rho"
                    ].min()
                ),

            "minimum_leave_one_out_top100_overlap_pct":
                float(
                    removal_df[
                        "top100_overlap_pct"
                    ].min()
                ),

            "maximum_leave_one_out_mean_rank_shift":
                float(
                    removal_df[
                        "mean_abs_rank_shift"
                    ].max()
                ),

            "c6_removed_renormalized_spearman":
                float(
                    c6_remove_fixed[
                        "spearman_rho"
                    ]
                ),

            "c6_removed_renormalized_top100_overlap_pct":
                float(
                    c6_remove_fixed[
                        "top100_overlap_pct"
                    ]
                ),

            "c6_removed_renormalized_mean_abs_rank_shift":
                float(
                    c6_remove_fixed[
                        "mean_abs_rank_shift"
                    ]
                ),

            "c6_removed_recomputed_critic_spearman":
                float(
                    c6_remove_recomputed[
                        "spearman_rho"
                    ]
                ),

            "c6_removed_recomputed_critic_top100_overlap_pct":
                float(
                    c6_remove_recomputed[
                        "top100_overlap_pct"
                    ]
                ),

            "c6_removed_recomputed_critic_mean_abs_rank_shift":
                float(
                    c6_remove_recomputed[
                        "mean_abs_rank_shift"
                    ]
                ),
        }
    ]
)


# ============================================================
# SAVE
# ============================================================

removal_path = (
    OUTPUT_DIR
    / "03_leave_one_criterion_out_metrics.csv"
)

c6_path = (
    OUTPUT_DIR
    / "03_c6_intervention_benefit_sensitivity.csv"
)

c6_grid_path = (
    OUTPUT_DIR
    / "03_c6_removal_grid_rank_changes.csv"
)

summary_path = (
    OUTPUT_DIR
    / "03_criterion_removal_summary.csv"
)


removal_df.to_csv(
    removal_path,
    index=False,
)

c6_df.to_csv(
    c6_path,
    index=False,
)

c6_grid_df.to_csv(
    c6_grid_path,
    index=False,
)

summary_df.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# FIGURE 1 - LEAVE-ONE-OUT SPEARMAN
# ============================================================

plot_df = (
    removal_df[
        removal_df[
            "weight_method"
        ]
        == "BASELINE_WEIGHTS_RENORMALIZED"
    ]
    .copy()
)


fig, ax = plt.subplots(
    figsize=(9, 5)
)

ax.bar(
    plot_df[
        "removed_criterion"
    ],
    plot_df[
        "spearman_rho"
    ],
)

ax.set_ylim(
    0,
    1.0,
)

ax.set_xlabel(
    "Removed Criterion"
)

ax.set_ylabel(
    "Spearman Rank Correlation"
)

ax.set_title(
    "TOPSIS Stability After Removing One Criterion"
)

fig.tight_layout()


figure1_path = (
    FIGURE_DIR
    / "03_leave_one_out_spearman.png"
)

fig.savefig(
    figure1_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 2 - C6 TOP100 OVERLAP
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    100.0
    * c6_df[
        "c6_retained_fraction"
    ],
    c6_df[
        "top100_overlap_pct"
    ],
    marker="o",
)

ax.set_ylim(
    0,
    100,
)

ax.set_xlabel(
    "C6 Weight Retained (%)"
)

ax.set_ylabel(
    "Baseline Top-100 Overlap (%)"
)

ax.set_title(
    "Priority Stability as Intervention-Benefit Weight is Reduced"
)

fig.tight_layout()


figure2_path = (
    FIGURE_DIR
    / "03_c6_top100_overlap_sensitivity.png"
)

fig.savefig(
    figure2_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print(
    "\n============================================================"
)

print(
    "OVERALL CRITERION-REMOVAL SUMMARY"
)

print(
    "============================================================"
)


print(
    summary_df.T.to_string(
        header=False
    )
)


print(
    "\n============================================================"
)

print(
    "TOP-10 GRIDS WITHOUT C6"
)

print(
    "============================================================"
)


print(
    c6_grid_df
    .sort_values(
        "rank_without_c6"
    )
    .head(
        10
    )[
        [
            "grid_id",
            "baseline_rank",
            "rank_without_c6",
            "rank_shift_without_c6",
        ]
    ]
    .to_string(
        index=False
    )
)


print(
    "\nIMPORTANT:"
)

print(
    "Criterion-removal scenarios are sensitivity "
    "tests only. They do not replace the frozen "
    "six-criterion CRITIC-TOPSIS solution."
)

print(
    "For each criterion, two tests are used: "
    "(1) remove it and renormalize the original "
    "weights, and (2) remove it and recompute "
    "CRITIC on the remaining five criteria."
)

print(
    "C6 is additionally reduced to 75%, 50%, "
    "25%, and 0% of its baseline weight because "
    "the scenario-validation stage identified "
    "interpretation uncertainty in intervention "
    "effects."
)

print(
    "No original Step-31/32/33 files, backend, "
    "frontend, or dashboard values are modified."
)


print(
    "\nSaved:"
)

print(removal_path)
print(c6_path)
print(c6_grid_path)
print(summary_path)
print(figure1_path)
print(figure2_path)


print(
    "\nCRITERION REMOVAL + C6 "
    "SENSITIVITY VALIDATION COMPLETE."
)