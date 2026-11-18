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

STEP32_AUDIT_PATH = (
    REPORT_DIR
    / "32_topsis_priority_ranking_audit.csv"
)

BOOTSTRAP_WEIGHTS_PATH = (
    OUTPUT_DIR
    / "01_critic_bootstrap_weights.csv"
)


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_GRIDS = 2194
EXPECTED_CRITERIA = 6
EXPECTED_BOOTSTRAPS = 500

TOL = 1e-12

TOP_K_VALUES = [
    10,
    50,
    100,
]

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


def normalize_weights(values):

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
        values <= 0
    ).any():

        raise RuntimeError(
            "TOPSIS weights must "
            "remain positive."
        )

    total = float(
        values.sum()
    )

    if total <= TOL:

        raise RuntimeError(
            "Weight sum is zero."
        )

    return (
        values
        / total
    )


def topsis(
    matrix,
    grid_ids,
    weights,
):

    """
    Exact Step-32 formulation.

    Step-31D matrix is already:
        min-max normalized
        higher = stronger priority

    Therefore no second normalization.
    """

    weights = normalize_weights(
        weights
    )

    weighted = (
        matrix
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

    distance_best = np.sqrt(
        np.sum(
            (
                weighted
                - ideal_best
            ) ** 2,
            axis=1,
        )
    )

    distance_worst = np.sqrt(
        np.sum(
            (
                weighted
                - ideal_worst
            ) ** 2,
            axis=1,
        )
    )

    denominator = (
        distance_best
        + distance_worst
    )

    if (
        denominator <= TOL
    ).any():

        raise RuntimeError(
            "TOPSIS denominator "
            "zero/near-zero."
        )

    score = (
        distance_worst
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

    # Same deterministic ranking rule as Step 32:
    # score descending, grid_id ascending.
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

    order_df = order_df.sort_values(
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
        dtype=int,
    )

    return score, ranks


def overlap(
    baseline_rank,
    scenario_rank,
    k,
):

    baseline_set = set(
        np.flatnonzero(
            baseline_rank <= k
        ).tolist()
    )

    scenario_set = set(
        np.flatnonzero(
            scenario_rank <= k
        ).tolist()
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

    return {
        "spearman_rho":
            spearman,

        "kendall_tau_b":
            kendall,

        "top10_overlap":
            overlap(
                baseline_rank,
                scenario_rank,
                10,
            ),

        "top50_overlap":
            overlap(
                baseline_rank,
                scenario_rank,
                50,
            ),

        "top100_overlap":
            overlap(
                baseline_rank,
                scenario_rank,
                100,
            ),

        "top10pct_overlap":
            overlap(
                baseline_rank,
                scenario_rank,
                TOP10PCT_K,
            ),

        "top25pct_overlap":
            overlap(
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
    }


# ============================================================
# LOAD
# ============================================================

print(
    "\n============================================================"
)
print(
    "PRIORITY VALIDATION 02"
)
print(
    "TOPSIS RANK STABILITY UNDER CRITIC RESAMPLING"
)
print(
    "============================================================\n"
)


for path, label in [
    (
        MATRIX_PATH,
        "Step 31D normalized matrix",
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
        "Step 32 TOPSIS ranking",
    ),
    (
        STEP32_AUDIT_PATH,
        "Step 32 audit",
    ),
    (
        BOOTSTRAP_WEIGHTS_PATH,
        "Priority validation Step 1 weights",
    ),
]:

    require_file(
        path,
        label,
    )


audit = pd.read_csv(
    STEP32_AUDIT_PATH,
    low_memory=False,
)

require_columns(
    audit,
    ["status"],
    "Step 32 audit",
)

if not (
    audit["status"]
    .astype(str)
    .str.upper()
    .eq("PASS")
    .all()
):

    raise RuntimeError(
        "Step 32 audit is not fully PASS."
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

bootstrap_df = pd.read_csv(
    BOOTSTRAP_WEIGHTS_PATH,
    low_memory=False,
)


# ============================================================
# VALIDATE INPUTS
# ============================================================

require_columns(
    metadata,
    [
        "criterion_order",
        "criterion_code",
        "final_field_name",
    ],
    "Criterion metadata",
)

require_columns(
    weights_df,
    [
        "criterion_order",
        "criterion_code",
        "final_field_name",
        "critic_weight",
    ],
    "CRITIC weights",
)

require_columns(
    baseline_df,
    [
        "grid_id",
        "topsis_priority_score",
        "topsis_rank",
    ],
    "Step 32 ranking",
)

require_columns(
    bootstrap_df,
    [
        "bootstrap_run",
        "criterion_order",
        "criterion_code",
        "final_field_name",
        "critic_weight",
    ],
    "Bootstrap weights",
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


criterion_codes = (
    metadata[
        "criterion_code"
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
    matrix_df["grid_id"]
    .astype(str)
    .str.strip()
)


if len(
    matrix_df
) != EXPECTED_GRIDS:

    raise RuntimeError(
        f"Expected {EXPECTED_GRIDS} grids, "
        f"found {len(matrix_df)}."
    )


if (
    matrix_df[
        "grid_id"
    ].nunique()
    != EXPECTED_GRIDS
):

    raise RuntimeError(
        "Unexpected unique-grid count."
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


# ============================================================
# BASELINE WEIGHTS + RANKING
# ============================================================

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

base_weights = normalize_weights(
    base_weights
)


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


if set(
    matrix_df[
        "grid_id"
    ]
) != set(
    baseline_lookup.index
):

    raise RuntimeError(
        "Step 31D and Step 32 "
        "grid identities do not match."
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


# ============================================================
# 1. REPRODUCE STEP 32 BASELINE
# ============================================================

print(
    "============================================================"
)
print(
    "1. STEP-32 BASELINE REPRODUCTION"
)
print(
    "============================================================"
)


calculated_score, calculated_rank = (
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


max_score_error = float(
    np.max(
        np.abs(
            calculated_score
            - saved_score
        )
    )
)


rank_exact_match = bool(
    np.array_equal(
        calculated_rank,
        saved_rank,
    )
)


print(
    f"Maximum TOPSIS score difference : "
    f"{max_score_error:.12e}"
)

print(
    f"Exact ranking match             : "
    f"{rank_exact_match}"
)


if (
    max_score_error
    > 1e-10
):

    raise RuntimeError(
        "Step 32 TOPSIS score "
        "reproduction failed."
    )


if not rank_exact_match:

    raise RuntimeError(
        "Step 32 ranking "
        "reproduction failed."
    )


print(
    "PASS - Original Step-32 TOPSIS "
    "ranking reproduced exactly."
)


# ============================================================
# 2. VALIDATE BOOTSTRAP WEIGHT SET
# ============================================================

bootstrap_runs = sorted(
    bootstrap_df[
        "bootstrap_run"
    ]
    .astype(int)
    .unique()
    .tolist()
)


if len(
    bootstrap_runs
) != EXPECTED_BOOTSTRAPS:

    raise RuntimeError(
        f"Expected {EXPECTED_BOOTSTRAPS} "
        f"bootstrap runs, found "
        f"{len(bootstrap_runs)}."
    )


print(
    f"\nBootstrap weight sets : "
    f"{len(bootstrap_runs)}"
)

print(
    f"Grids per TOPSIS run  : "
    f"{EXPECTED_GRIDS:,}"
)


# ============================================================
# 3. RECOMPUTE TOPSIS FOR 500 BOOTSTRAP WEIGHT SETS
# ============================================================

print(
    "\n============================================================"
)
print(
    "2. TOPSIS RANKING ACROSS BOOTSTRAP WEIGHTS"
)
print(
    "============================================================"
)


rank_matrix = np.empty(
    (
        EXPECTED_GRIDS,
        EXPECTED_BOOTSTRAPS,
    ),
    dtype=np.int32,
)


score_matrix = np.empty(
    (
        EXPECTED_GRIDS,
        EXPECTED_BOOTSTRAPS,
    ),
    dtype=float,
)


metrics_rows = []


for run_index, run in enumerate(
    bootstrap_runs
):

    current = (
        bootstrap_df[
            bootstrap_df[
                "bootstrap_run"
            ].astype(int)
            == run
        ]
        .copy()
        .sort_values(
            "criterion_order"
        )
    )


    if len(
        current
    ) != EXPECTED_CRITERIA:

        raise RuntimeError(
            f"Bootstrap run {run} does "
            "not contain six weights."
        )


    current_identity = (
        current[
            "final_field_name"
        ]
        .astype(str)
        .tolist()
    )


    if current_identity != fields:

        raise RuntimeError(
            f"Bootstrap run {run} "
            "criterion order mismatch."
        )


    current_weights = (
        current[
            "critic_weight"
        ]
        .to_numpy(
            dtype=float
        )
    )


    current_weights = normalize_weights(
        current_weights
    )


    current_score, current_rank = (
        topsis(
            matrix_array,
            matrix_df[
                "grid_id"
            ].to_numpy(
                dtype=str
            ),
            current_weights,
        )
    )


    rank_matrix[
        :,
        run_index
    ] = current_rank


    score_matrix[
        :,
        run_index
    ] = current_score


    metrics = rank_metrics(
        saved_rank,
        current_rank,
    )


    metrics_rows.append(
        {
            "bootstrap_run":
                run,

            **metrics,
        }
    )


    if (
        run_index + 1
    ) % 100 == 0:

        print(
            f"Processed "
            f"{run_index + 1}/"
            f"{EXPECTED_BOOTSTRAPS}"
        )


metrics_df = pd.DataFrame(
    metrics_rows
)


# ============================================================
# 4. GRID-LEVEL RANK STABILITY
# ============================================================

rank_shift = (
    rank_matrix
    - saved_rank.reshape(
        -1,
        1,
    )
)

abs_rank_shift = np.abs(
    rank_shift
)


grid_stability_df = pd.DataFrame(
    {
        "grid_id":
            matrix_df[
                "grid_id"
            ].astype(str),

        "baseline_rank":
            saved_rank,

        "baseline_topsis_score":
            saved_score,

        "mean_bootstrap_rank":
            np.mean(
                rank_matrix,
                axis=1,
            ),

        "median_bootstrap_rank":
            np.median(
                rank_matrix,
                axis=1,
            ),

        "rank_std":
            np.std(
                rank_matrix,
                axis=1,
                ddof=0,
            ),

        "best_bootstrap_rank":
            np.min(
                rank_matrix,
                axis=1,
            ),

        "worst_bootstrap_rank":
            np.max(
                rank_matrix,
                axis=1,
            ),

        "mean_abs_rank_shift":
            np.mean(
                abs_rank_shift,
                axis=1,
            ),

        "p95_abs_rank_shift":
            np.quantile(
                abs_rank_shift,
                0.95,
                axis=1,
            ),

        "max_abs_rank_shift":
            np.max(
                abs_rank_shift,
                axis=1,
            ),

        "pct_runs_in_top10":
            100.0
            * np.mean(
                rank_matrix
                <= 10,
                axis=1,
            ),

        "pct_runs_in_top50":
            100.0
            * np.mean(
                rank_matrix
                <= 50,
                axis=1,
            ),

        "pct_runs_in_top100":
            100.0
            * np.mean(
                rank_matrix
                <= 100,
                axis=1,
            ),

        "pct_runs_in_top10pct":
            100.0
            * np.mean(
                rank_matrix
                <= TOP10PCT_K,
                axis=1,
            ),

        "pct_runs_in_top25pct":
            100.0
            * np.mean(
                rank_matrix
                <= TOP25PCT_K,
                axis=1,
            ),
    }
)


grid_stability_df = (
    grid_stability_df
    .sort_values(
        "baseline_rank"
    )
    .reset_index(drop=True)
)


# ============================================================
# 5. BASELINE TOP-GRID RETENTION
# ============================================================

baseline_top100 = (
    grid_stability_df[
        grid_stability_df[
            "baseline_rank"
        ] <= 100
    ]
    .copy()
)


top100_mean_retention = float(
    baseline_top100[
        "pct_runs_in_top100"
    ].mean()
)


top100_min_retention = float(
    baseline_top100[
        "pct_runs_in_top100"
    ].min()
)


baseline_top10pct = (
    grid_stability_df[
        grid_stability_df[
            "baseline_rank"
        ] <= TOP10PCT_K
    ]
)


top10pct_mean_retention = float(
    baseline_top10pct[
        "pct_runs_in_top10pct"
    ].mean()
)


top10pct_min_retention = float(
    baseline_top10pct[
        "pct_runs_in_top10pct"
    ].min()
)


# ============================================================
# 6. OVERALL SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    [
        {
            "grid_count":
                EXPECTED_GRIDS,

            "criteria_count":
                EXPECTED_CRITERIA,

            "bootstrap_topsis_runs":
                EXPECTED_BOOTSTRAPS,

            "spearman_min":
                float(
                    metrics_df[
                        "spearman_rho"
                    ].min()
                ),

            "spearman_median":
                float(
                    metrics_df[
                        "spearman_rho"
                    ].median()
                ),

            "spearman_mean":
                float(
                    metrics_df[
                        "spearman_rho"
                    ].mean()
                ),

            "kendall_min":
                float(
                    metrics_df[
                        "kendall_tau_b"
                    ].min()
                ),

            "kendall_median":
                float(
                    metrics_df[
                        "kendall_tau_b"
                    ].median()
                ),

            "kendall_mean":
                float(
                    metrics_df[
                        "kendall_tau_b"
                    ].mean()
                ),

            "top10_overlap_min":
                float(
                    metrics_df[
                        "top10_overlap"
                    ].min()
                ),

            "top10_overlap_median":
                float(
                    metrics_df[
                        "top10_overlap"
                    ].median()
                ),

            "top50_overlap_min":
                float(
                    metrics_df[
                        "top50_overlap"
                    ].min()
                ),

            "top50_overlap_median":
                float(
                    metrics_df[
                        "top50_overlap"
                    ].median()
                ),

            "top100_overlap_min":
                float(
                    metrics_df[
                        "top100_overlap"
                    ].min()
                ),

            "top100_overlap_median":
                float(
                    metrics_df[
                        "top100_overlap"
                    ].median()
                ),

            "top10pct_overlap_min":
                float(
                    metrics_df[
                        "top10pct_overlap"
                    ].min()
                ),

            "top10pct_overlap_median":
                float(
                    metrics_df[
                        "top10pct_overlap"
                    ].median()
                ),

            "top25pct_overlap_min":
                float(
                    metrics_df[
                        "top25pct_overlap"
                    ].min()
                ),

            "top25pct_overlap_median":
                float(
                    metrics_df[
                        "top25pct_overlap"
                    ].median()
                ),

            "mean_abs_rank_shift_mean":
                float(
                    metrics_df[
                        "mean_abs_rank_shift"
                    ].mean()
                ),

            "p95_abs_rank_shift_median":
                float(
                    metrics_df[
                        "p95_abs_rank_shift"
                    ].median()
                ),

            "worst_run_max_abs_rank_shift":
                int(
                    metrics_df[
                        "max_abs_rank_shift"
                    ].max()
                ),

            "baseline_top100_mean_retention_pct":
                top100_mean_retention,

            "baseline_top100_min_retention_pct":
                top100_min_retention,

            "baseline_top10pct_mean_retention_pct":
                top10pct_mean_retention,

            "baseline_top10pct_min_retention_pct":
                top10pct_min_retention,
        }
    ]
)


# ============================================================
# 7. SAVE
# ============================================================

metrics_path = (
    OUTPUT_DIR
    / "02_topsis_bootstrap_run_metrics.csv"
)

grid_stability_path = (
    OUTPUT_DIR
    / "02_topsis_grid_rank_stability.csv"
)

top100_path = (
    OUTPUT_DIR
    / "02_topsis_baseline_top100_stability.csv"
)

summary_path = (
    OUTPUT_DIR
    / "02_topsis_rank_stability_summary.csv"
)


metrics_df.to_csv(
    metrics_path,
    index=False,
)

grid_stability_df.to_csv(
    grid_stability_path,
    index=False,
)

baseline_top100.to_csv(
    top100_path,
    index=False,
)

summary_df.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# 8. FIGURES
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.hist(
    metrics_df[
        "spearman_rho"
    ],
    bins=25,
)

ax.set_xlabel(
    "Spearman Rank Correlation"
)

ax.set_ylabel(
    "Bootstrap Runs"
)

ax.set_title(
    "TOPSIS Rank Stability Under CRITIC Resampling"
)

fig.tight_layout()

figure1_path = (
    FIGURE_DIR
    / "02_topsis_spearman_distribution.png"
)

fig.savefig(
    figure1_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.hist(
    100.0
    * metrics_df[
        "top100_overlap"
    ],
    bins=20,
)

ax.set_xlabel(
    "Top-100 Overlap with Baseline (%)"
)

ax.set_ylabel(
    "Bootstrap Runs"
)

ax.set_title(
    "TOPSIS Top-100 Priority Stability"
)

fig.tight_layout()

figure2_path = (
    FIGURE_DIR
    / "02_topsis_top100_overlap_distribution.png"
)

fig.savefig(
    figure2_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print(
    "\n============================================================"
)
print(
    "TOPSIS RANK-STABILITY SUMMARY"
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
    "BASELINE TOP-10 GRID STABILITY"
)
print(
    "============================================================"
)

print(
    grid_stability_df.head(
        10
    )[
        [
            "baseline_rank",
            "grid_id",
            "mean_bootstrap_rank",
            "worst_bootstrap_rank",
            "max_abs_rank_shift",
            "pct_runs_in_top10",
            "pct_runs_in_top50",
            "pct_runs_in_top100",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nIMPORTANT:"
)

print(
    "This validation uses the 500 CRITIC "
    "weight sets produced by Priority "
    "Validation Step 1."
)

print(
    "The exact Step-32 TOPSIS formulation "
    "is reused; no second normalization "
    "and no criteria are changed."
)

print(
    "The original Step-32 ranking and "
    "dashboard outputs are NOT overwritten."
)

print(
    "No arbitrary PASS threshold is imposed; "
    "rank correlations, overlap and rank-shift "
    "metrics are reported as robustness evidence."
)


print(
    "\nSaved:"
)

print(metrics_path)
print(grid_stability_path)
print(top100_path)
print(summary_path)
print(figure1_path)
print(figure2_path)


print(
    "\nTOPSIS RANK STABILITY "
    "VALIDATION COMPLETE."
)