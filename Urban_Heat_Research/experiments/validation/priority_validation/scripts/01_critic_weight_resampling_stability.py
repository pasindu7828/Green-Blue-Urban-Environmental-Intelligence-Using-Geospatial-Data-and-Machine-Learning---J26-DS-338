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

AUDIT_PATH = (
    REPORT_DIR
    / "31e_critic_weighting_audit.csv"
)


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_GRIDS = 2194
EXPECTED_CRITERIA = 6

BOOTSTRAP_RUNS = 500
RANDOM_SEED = 42

TOL = 1e-10


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
        col
        for col in columns
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{label} missing columns: {missing}"
        )


def critic_weights(X):

    """
    Exact Step-31E CRITIC formulation:

        information_j
            = population_std_j
              * sum_k(1 - Pearson_corr_jk)

        weight_j
            = information_j
              / total_information
    """

    X = pd.DataFrame(X).astype(float)

    corr = X.corr(
        method="pearson"
    )

    if corr.isna().any().any():
        raise RuntimeError(
            "CRITIC correlation contains NaN."
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

    if (
        not np.isfinite(total)
        or total <= 0
    ):
        raise RuntimeError(
            "Invalid CRITIC information total."
        )

    weights = (
        information
        / total
    )

    return (
        weights.to_numpy(dtype=float),
        std_population.to_numpy(dtype=float),
        conflict.to_numpy(dtype=float),
        information.to_numpy(dtype=float),
    )


def rank_weights(weights):

    return (
        pd.Series(weights)
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
        .to_numpy()
    )


# ============================================================
# LOAD
# ============================================================

print("\n============================================================")
print("PRIORITY VALIDATION 01")
print("CRITIC WEIGHT RESAMPLING STABILITY")
print("============================================================\n")


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
        "Step 31E CRITIC weights",
    ),
    (
        AUDIT_PATH,
        "Step 31E audit",
    ),
]:

    require_file(
        path,
        label,
    )


audit = pd.read_csv(
    AUDIT_PATH,
    low_memory=False,
)

require_columns(
    audit,
    ["status"],
    "Step 31E audit",
)

if not (
    audit["status"]
    .astype(str)
    .str.upper()
    .eq("PASS")
    .all()
):
    raise RuntimeError(
        "Step 31E audit is not fully PASS."
    )


matrix = pd.read_csv(
    MATRIX_PATH,
    dtype={"grid_id": str},
    low_memory=False,
)

metadata = pd.read_csv(
    METADATA_PATH,
    low_memory=False,
)

saved_weights = pd.read_csv(
    WEIGHTS_PATH,
    low_memory=False,
)


require_columns(
    matrix,
    ["grid_id"],
    "Normalized matrix",
)

require_columns(
    metadata,
    [
        "criterion_order",
        "criterion_code",
        "decision_dimension",
        "final_indicator_name",
        "final_field_name",
    ],
    "Metadata",
)

require_columns(
    saved_weights,
    [
        "criterion_order",
        "criterion_code",
        "final_field_name",
        "critic_weight",
    ],
    "Saved CRITIC weights",
)


metadata = (
    metadata
    .sort_values("criterion_order")
    .reset_index(drop=True)
)

saved_weights = (
    saved_weights
    .sort_values("criterion_order")
    .reset_index(drop=True)
)


if len(metadata) != EXPECTED_CRITERIA:

    raise RuntimeError(
        f"Expected {EXPECTED_CRITERIA} criteria, "
        f"found {len(metadata)}."
    )


fields = (
    metadata[
        "final_field_name"
    ]
    .astype(str)
    .tolist()
)


require_columns(
    matrix,
    fields,
    "Normalized matrix",
)


if len(matrix) != EXPECTED_GRIDS:

    raise RuntimeError(
        f"Expected {EXPECTED_GRIDS} grids, "
        f"found {len(matrix)}."
    )


if (
    matrix["grid_id"]
    .astype(str)
    .nunique()
    != EXPECTED_GRIDS
):

    raise RuntimeError(
        "Grid count is not 2,194 unique grids."
    )


X = (
    matrix[
        fields
    ]
    .apply(
        pd.to_numeric,
        errors="raise",
    )
)


if not np.isfinite(
    X.to_numpy(dtype=float)
).all():

    raise RuntimeError(
        "Normalized matrix contains "
        "non-finite values."
    )


print(
    f"Grids loaded    : {len(matrix):,}"
)

print(
    f"Criteria loaded : {len(fields)}"
)

print(
    f"Bootstrap runs  : {BOOTSTRAP_RUNS}"
)


# ============================================================
# 1. REPRODUCE BASELINE CRITIC WEIGHTS
# ============================================================

print("\n============================================================")
print("1. BASELINE CRITIC REPRODUCTION")
print("============================================================")


(
    reproduced_weights,
    baseline_std,
    baseline_conflict,
    baseline_information,
) = critic_weights(
    X
)


saved_weight_values = (
    saved_weights[
        "critic_weight"
    ]
    .to_numpy(dtype=float)
)


max_baseline_error = float(
    np.max(
        np.abs(
            reproduced_weights
            - saved_weight_values
        )
    )
)


print(
    f"Maximum difference from saved "
    f"Step-31E weights: "
    f"{max_baseline_error:.12e}"
)


if max_baseline_error > TOL:

    raise RuntimeError(
        "CRITIC reproduction does not match "
        "saved Step-31E weights."
    )


print(
    "PASS - Original Step-31E CRITIC "
    "weights reproduced exactly."
)


baseline_rank = rank_weights(
    reproduced_weights
)


baseline_df = metadata[
    [
        "criterion_order",
        "criterion_code",
        "decision_dimension",
        "final_indicator_name",
        "final_field_name",
    ]
].copy()


baseline_df[
    "baseline_critic_weight"
] = reproduced_weights

baseline_df[
    "baseline_weight_pct"
] = (
    reproduced_weights
    * 100.0
)

baseline_df[
    "baseline_weight_rank"
] = baseline_rank

baseline_df[
    "population_std"
] = baseline_std

baseline_df[
    "conflict_sum"
] = baseline_conflict

baseline_df[
    "critic_information"
] = baseline_information


print(
    baseline_df[
        [
            "criterion_code",
            "decision_dimension",
            "baseline_critic_weight",
            "baseline_weight_rank",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 2. BOOTSTRAP RESAMPLING
# ============================================================

print("\n============================================================")
print("2. BOOTSTRAP CRITIC WEIGHT STABILITY")
print("============================================================")


rng = np.random.default_rng(
    RANDOM_SEED
)

X_array = X.to_numpy(
    dtype=float
)


bootstrap_weight_matrix = np.zeros(
    (
        BOOTSTRAP_RUNS,
        EXPECTED_CRITERIA,
    ),
    dtype=float,
)

bootstrap_rank_matrix = np.zeros(
    (
        BOOTSTRAP_RUNS,
        EXPECTED_CRITERIA,
    ),
    dtype=int,
)


run_rows = []


for run in range(
    BOOTSTRAP_RUNS
):

    sample_index = rng.integers(
        low=0,
        high=EXPECTED_GRIDS,
        size=EXPECTED_GRIDS,
    )

    sample = pd.DataFrame(
        X_array[
            sample_index
        ],
        columns=fields,
    )

    (
        current_weights,
        _,
        _,
        _,
    ) = critic_weights(
        sample
    )

    current_rank = rank_weights(
        current_weights
    )

    bootstrap_weight_matrix[
        run,
        :
    ] = current_weights

    bootstrap_rank_matrix[
        run,
        :
    ] = current_rank


    spearman = float(
        spearmanr(
            baseline_rank,
            current_rank,
        ).statistic
    )

    kendall = float(
        kendalltau(
            baseline_rank,
            current_rank,
        ).statistic
    )


    run_rows.append(
        {
            "bootstrap_run":
                run + 1,

            "spearman_weight_rank_vs_baseline":
                spearman,

            "kendall_weight_rank_vs_baseline":
                kendall,

            "max_absolute_weight_change":
                float(
                    np.max(
                        np.abs(
                            current_weights
                            - reproduced_weights
                        )
                    )
                ),

            "mean_absolute_weight_change":
                float(
                    np.mean(
                        np.abs(
                            current_weights
                            - reproduced_weights
                        )
                    )
                ),

            "baseline_top_weight_retained":
                bool(
                    np.argmin(
                        current_rank
                    )
                    ==
                    np.argmin(
                        baseline_rank
                    )
                ),
        }
    )


run_df = pd.DataFrame(
    run_rows
)


# ============================================================
# 3. CRITERION-LEVEL STABILITY SUMMARY
# ============================================================

summary_rows = []

long_rows = []


for j in range(
    EXPECTED_CRITERIA
):

    values = (
        bootstrap_weight_matrix[
            :,
            j,
        ]
    )

    ranks = (
        bootstrap_rank_matrix[
            :,
            j,
        ]
    )


    for run in range(
        BOOTSTRAP_RUNS
    ):

        long_rows.append(
            {
                "bootstrap_run":
                    run + 1,

                "criterion_order":
                    int(
                        metadata.loc[
                            j,
                            "criterion_order",
                        ]
                    ),

                "criterion_code":
                    metadata.loc[
                        j,
                        "criterion_code",
                    ],

                "decision_dimension":
                    metadata.loc[
                        j,
                        "decision_dimension",
                    ],

                "final_field_name":
                    fields[j],

                "critic_weight":
                    values[run],

                "weight_rank":
                    ranks[run],
            }
        )


    ci_low = float(
        np.quantile(
            values,
            0.025,
        )
    )

    ci_high = float(
        np.quantile(
            values,
            0.975,
        )
    )

    mean_weight = float(
        np.mean(values)
    )

    std_weight = float(
        np.std(
            values,
            ddof=1,
        )
    )

    cv = (
        std_weight
        / mean_weight
        if mean_weight > 0
        else np.nan
    )


    baseline_rank_j = int(
        baseline_rank[j]
    )


    summary_rows.append(
        {
            "criterion_order":
                int(
                    metadata.loc[
                        j,
                        "criterion_order",
                    ]
                ),

            "criterion_code":
                metadata.loc[
                    j,
                    "criterion_code",
                ],

            "decision_dimension":
                metadata.loc[
                    j,
                    "decision_dimension",
                ],

            "final_indicator_name":
                metadata.loc[
                    j,
                    "final_indicator_name",
                ],

            "final_field_name":
                fields[j],

            "baseline_weight":
                reproduced_weights[j],

            "bootstrap_mean_weight":
                mean_weight,

            "bootstrap_std_weight":
                std_weight,

            "coefficient_of_variation":
                cv,

            "ci95_lower":
                ci_low,

            "ci95_upper":
                ci_high,

            "ci95_width":
                ci_high - ci_low,

            "baseline_weight_rank":
                baseline_rank_j,

            "median_bootstrap_rank":
                float(
                    np.median(ranks)
                ),

            "baseline_rank_retained_pct":
                float(
                    np.mean(
                        ranks
                        == baseline_rank_j
                    )
                    * 100.0
                ),

            "rank1_frequency_pct":
                float(
                    np.mean(
                        ranks == 1
                    )
                    * 100.0
                ),

            "min_observed_rank":
                int(
                    np.min(ranks)
                ),

            "max_observed_rank":
                int(
                    np.max(ranks)
                ),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)

long_df = pd.DataFrame(
    long_rows
)


# ============================================================
# 4. GLOBAL STABILITY SUMMARY
# ============================================================

global_summary = pd.DataFrame(
    [
        {
            "bootstrap_runs":
                BOOTSTRAP_RUNS,

            "grid_sample_size_per_run":
                EXPECTED_GRIDS,

            "random_seed":
                RANDOM_SEED,

            "mean_spearman_weight_rank":
                float(
                    run_df[
                        "spearman_weight_rank_vs_baseline"
                    ].mean()
                ),

            "median_spearman_weight_rank":
                float(
                    run_df[
                        "spearman_weight_rank_vs_baseline"
                    ].median()
                ),

            "minimum_spearman_weight_rank":
                float(
                    run_df[
                        "spearman_weight_rank_vs_baseline"
                    ].min()
                ),

            "mean_kendall_weight_rank":
                float(
                    run_df[
                        "kendall_weight_rank_vs_baseline"
                    ].mean()
                ),

            "median_kendall_weight_rank":
                float(
                    run_df[
                        "kendall_weight_rank_vs_baseline"
                    ].median()
                ),

            "top_weight_criterion_retained_pct":
                float(
                    run_df[
                        "baseline_top_weight_retained"
                    ].mean()
                    * 100.0
                ),

            "mean_max_absolute_weight_change":
                float(
                    run_df[
                        "max_absolute_weight_change"
                    ].mean()
                ),

            "p95_max_absolute_weight_change":
                float(
                    run_df[
                        "max_absolute_weight_change"
                    ].quantile(
                        0.95
                    )
                ),
        }
    ]
)


# ============================================================
# 5. SAVE OUTPUTS
# ============================================================

baseline_path = (
    OUTPUT_DIR
    / "01_critic_baseline_reproduction.csv"
)

bootstrap_long_path = (
    OUTPUT_DIR
    / "01_critic_bootstrap_weights.csv"
)

criterion_summary_path = (
    OUTPUT_DIR
    / "01_critic_weight_stability_summary.csv"
)

run_metrics_path = (
    OUTPUT_DIR
    / "01_critic_bootstrap_run_metrics.csv"
)

global_summary_path = (
    OUTPUT_DIR
    / "01_critic_global_stability_summary.csv"
)


baseline_df.to_csv(
    baseline_path,
    index=False,
)

long_df.to_csv(
    bootstrap_long_path,
    index=False,
)

summary_df.to_csv(
    criterion_summary_path,
    index=False,
)

run_df.to_csv(
    run_metrics_path,
    index=False,
)

global_summary.to_csv(
    global_summary_path,
    index=False,
)


# ============================================================
# 6. FIGURE - BASELINE vs BOOTSTRAP MEAN
# ============================================================

x = np.arange(
    EXPECTED_CRITERIA
)

bootstrap_mean = (
    summary_df[
        "bootstrap_mean_weight"
    ].to_numpy()
)

lower_error = (
    bootstrap_mean
    - summary_df[
        "ci95_lower"
    ].to_numpy()
)

upper_error = (
    summary_df[
        "ci95_upper"
    ].to_numpy()
    - bootstrap_mean
)


fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.errorbar(
    x,
    bootstrap_mean,
    yerr=[
        lower_error,
        upper_error,
    ],
    fmt="o",
    capsize=4,
    label="Bootstrap mean + 95% interval",
)

ax.scatter(
    x,
    reproduced_weights,
    marker="x",
    s=70,
    label="Baseline CRITIC weight",
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    metadata[
        "criterion_code"
    ].astype(str),
)

ax.set_xlabel(
    "Priority Criterion"
)

ax.set_ylabel(
    "CRITIC Weight"
)

ax.set_title(
    "CRITIC Weight Stability Under Grid Resampling"
)

ax.legend()

fig.tight_layout()


figure1_path = (
    FIGURE_DIR
    / "01_critic_weight_bootstrap_intervals.png"
)

fig.savefig(
    figure1_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 7. FIGURE - WEIGHT DISTRIBUTIONS
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.boxplot(
    [
        bootstrap_weight_matrix[
            :,
            j,
        ]
        for j in range(
            EXPECTED_CRITERIA
        )
    ],
    tick_labels=metadata[
        "criterion_code"
    ].astype(str).tolist(),
)

ax.set_xlabel(
    "Priority Criterion"
)

ax.set_ylabel(
    "Bootstrap CRITIC Weight"
)

ax.set_title(
    "Distribution of CRITIC Weights Across Bootstrap Runs"
)

fig.tight_layout()


figure2_path = (
    FIGURE_DIR
    / "01_critic_weight_bootstrap_distributions.png"
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

print("\n============================================================")
print("CRITERION-LEVEL WEIGHT STABILITY")
print("============================================================")

print(
    summary_df[
        [
            "criterion_code",
            "decision_dimension",
            "baseline_weight",
            "bootstrap_mean_weight",
            "bootstrap_std_weight",
            "ci95_lower",
            "ci95_upper",
            "baseline_weight_rank",
            "baseline_rank_retained_pct",
            "rank1_frequency_pct",
            "min_observed_rank",
            "max_observed_rank",
        ]
    ].to_string(
        index=False
    )
)


print("\n============================================================")
print("GLOBAL CRITIC STABILITY")
print("============================================================")

print(
    global_summary.to_string(
        index=False
    )
)


print("\nIMPORTANT:")
print(
    "This step evaluates CRITIC weight stability "
    "under repeated resampling of the 2,194-grid "
    "decision matrix."
)

print(
    "It does NOT modify the original Step-31E "
    "weights or Step-32 TOPSIS ranking."
)

print(
    "No arbitrary statistical PASS threshold is "
    "imposed; stability metrics are reported as "
    "research evidence."
)


print("\nSaved:")
print(baseline_path)
print(bootstrap_long_path)
print(criterion_summary_path)
print(run_metrics_path)
print(global_summary_path)
print(figure1_path)
print(figure2_path)

print(
    "\nCRITIC WEIGHT RESAMPLING "
    "VALIDATION COMPLETE."
)