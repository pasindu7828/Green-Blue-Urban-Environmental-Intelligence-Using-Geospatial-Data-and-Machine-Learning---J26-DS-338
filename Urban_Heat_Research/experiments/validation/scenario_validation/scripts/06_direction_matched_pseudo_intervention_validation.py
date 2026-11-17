from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[4]

VALIDATION_DIR = (
    ROOT
    / "experiments"
    / "validation"
    / "scenario_validation"
)

OUTPUT_DIR = VALIDATION_DIR / "outputs"
FIGURE_DIR = VALIDATION_DIR / "figures"

INPUT_PATH = (
    OUTPUT_DIR
    / "05_pseudo_intervention_grid_results.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

EPS = 1e-12

FAMILY_RULES = {
    "GREEN": "increase",
    "BUILT_NDBI": "decrease",
    "BLUE_NDWI": "increase",
}


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

    if (
        a.nunique() < 2
        or b.nunique() < 2
    ):
        return np.nan

    return float(
        a.corr(
            b,
            method=method,
        )
    )


def sign_agreement(observed, modeled):

    observed = np.asarray(
        observed,
        dtype=float,
    )

    modeled = np.asarray(
        modeled,
        dtype=float,
    )

    valid = (
        np.isfinite(observed)
        & np.isfinite(modeled)
        & (np.abs(observed) > EPS)
        & (np.abs(modeled) > EPS)
    )

    if valid.sum() == 0:
        return np.nan, 0

    matched = (
        np.sign(observed[valid])
        == np.sign(modeled[valid])
    )

    return (
        float(
            matched.mean()
            * 100.0
        ),
        int(valid.sum()),
    )


def calculate_metrics(df):

    observed = (
        df[
            "observed_spatial_anomaly_change_c"
        ]
        .to_numpy(dtype=float)
    )

    modeled = (
        df[
            "modeled_pseudo_intervention_response_c"
        ]
        .to_numpy(dtype=float)
    )

    sign_pct, sign_n = sign_agreement(
        observed,
        modeled,
    )

    return {
        "rows": len(df),

        "mean_abs_treatment_change":
            float(
                np.abs(
                    df[
                        "actual_treatment_change"
                    ]
                ).mean()
            ),

        "mean_observed_spatial_change_c":
            float(
                np.mean(observed)
            ),

        "mean_modeled_response_c":
            float(
                np.mean(modeled)
            ),

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
    }


# ============================================================
# LOAD
# ============================================================

print("\n============================================================")
print("SCENARIO VALIDATION 06")
print("DIRECTION-MATCHED HISTORICAL BACKTEST")
print("============================================================\n")

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Step 5 results not found:\n{INPUT_PATH}"
    )

df = pd.read_csv(
    INPUT_PATH,
    dtype={
        "grid_id": str,
    },
    low_memory=False,
)

print(
    f"Step-5 rows loaded : {len(df):,}"
)

print(
    f"Years              : "
    f"{sorted(df['target_year'].unique())}"
)


# ============================================================
# DIRECTION-MATCHED FILTERING
# ============================================================

matched_frames = []
metric_rows = []
strong_metric_rows = []


for family, required_direction in (
    FAMILY_RULES.items()
):

    family_df = (
        df[
            df["scenario_family"]
            == family
        ]
        .copy()
    )

    if required_direction == "increase":

        matched = (
            family_df[
                family_df[
                    "actual_treatment_change"
                ] > EPS
            ]
            .copy()
        )

    elif required_direction == "decrease":

        matched = (
            family_df[
                family_df[
                    "actual_treatment_change"
                ] < -EPS
            ]
            .copy()
        )

    else:

        raise RuntimeError(
            f"Unknown direction: "
            f"{required_direction}"
        )


    matched[
        "scenario_required_direction"
    ] = required_direction

    matched[
        "abs_treatment_change"
    ] = np.abs(
        matched[
            "actual_treatment_change"
        ]
    )

    matched_frames.append(
        matched
    )


    # --------------------------------------------------------
    # Per-year metrics
    # --------------------------------------------------------

    for year in sorted(
        matched["target_year"].unique()
    ):

        sub = (
            matched[
                matched["target_year"]
                == year
            ]
            .copy()
        )

        metrics = calculate_metrics(
            sub
        )

        metric_rows.append(
            {
                "scenario_family":
                    family,

                "required_direction":
                    required_direction,

                "period":
                    str(year),

                **metrics,
            }
        )


    # --------------------------------------------------------
    # Combined 2024-2025 metrics
    # --------------------------------------------------------

    combined_metrics = (
        calculate_metrics(
            matched
        )
    )

    metric_rows.append(
        {
            "scenario_family":
                family,

            "required_direction":
                required_direction,

            "period":
                "2024-2025",

            **combined_metrics,
        }
    )


    # --------------------------------------------------------
    # Strong intervention-direction changes
    # Q75 calculated inside matched direction only
    # --------------------------------------------------------

    q75 = float(
        matched[
            "abs_treatment_change"
        ].quantile(0.75)
    )

    strong = (
        matched[
            matched[
                "abs_treatment_change"
            ] >= q75
        ]
        .copy()
    )

    strong_metrics = (
        calculate_metrics(
            strong
        )
    )

    strong_metric_rows.append(
        {
            "scenario_family":
                family,

            "required_direction":
                required_direction,

            "strong_change_threshold_q75":
                q75,

            **strong_metrics,
        }
    )


# ============================================================
# SAVE MATCHED DATA
# ============================================================

matched_df = pd.concat(
    matched_frames,
    ignore_index=True,
)

metrics_df = pd.DataFrame(
    metric_rows
)

strong_df = pd.DataFrame(
    strong_metric_rows
)


matched_path = (
    OUTPUT_DIR
    / "06_direction_matched_grid_results.csv"
)

metrics_path = (
    OUTPUT_DIR
    / "06_direction_matched_backtest_metrics.csv"
)

strong_path = (
    OUTPUT_DIR
    / "06_strong_direction_matched_metrics.csv"
)


matched_df.to_csv(
    matched_path,
    index=False,
)

metrics_df.to_csv(
    metrics_path,
    index=False,
)

strong_df.to_csv(
    strong_path,
    index=False,
)


# ============================================================
# FIGURE
# ============================================================

combined_plot = (
    metrics_df[
        metrics_df["period"]
        == "2024-2025"
    ]
    .copy()
)

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.bar(
    combined_plot[
        "scenario_family"
    ],
    combined_plot[
        "sign_agreement_pct"
    ],
)

ax.set_ylim(
    0,
    100,
)

ax.set_ylabel(
    "Direction Agreement (%)"
)

ax.set_title(
    "Direction-Matched Historical Scenario Backtest"
)

fig.tight_layout()

figure_path = (
    FIGURE_DIR
    / "06_direction_matched_sign_agreement.png"
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print("\n============================================================")
print("DIRECTION-MATCHED BACKTEST")
print("============================================================")

print(
    metrics_df.to_string(
        index=False
    )
)


print("\n============================================================")
print("STRONG DIRECTION-MATCHED CHANGES")
print("============================================================")

print(
    strong_df.to_string(
        index=False
    )
)


print("\nINTERPRETATION RULE:")

print(
    "GREEN is evaluated only where historical "
    "green cover increased."
)

print(
    "BUILT_NDBI is evaluated only where historical "
    "NDBI decreased."
)

print(
    "BLUE_NDWI is evaluated only where historical "
    "NDWI increased."
)

print(
    "\nThis is more directly aligned with the "
    "actual future scenario intervention directions."
)

print(
    "Results are historical consistency evidence, "
    "not causal intervention proof."
)


print("\nSaved:")
print(matched_path)
print(metrics_path)
print(strong_path)
print(figure_path)

print(
    "\nDIRECTION-MATCHED HISTORICAL "
    "VALIDATION COMPLETE."
)