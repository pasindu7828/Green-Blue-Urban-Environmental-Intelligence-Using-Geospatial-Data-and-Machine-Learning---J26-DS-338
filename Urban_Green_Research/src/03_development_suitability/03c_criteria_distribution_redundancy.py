
"""
J26-DS-338
Section 03 - Stage 3.4
Raw Criteria Distribution and Redundancy Analysis

No AHP, no standardization, no suitability scoring.
Only reviews candidate measurements for final selection.
"""

from pathlib import Path
from itertools import combinations
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "data/processed/development_suitability/"
    "raw_criteria/03b_raw_criteria_100m.csv"
)

REPORT_DIR = (
    ROOT / "outputs/reports/development_suitability"
)

# Strong-correlation REVIEW threshold.
# This is not an automatic criterion-removal rule.
REVIEW_THRESHOLD = 0.85

CRITERIA = {
    "c1_major_road_distance_m": "C1",
    "c2_road_density_250m_km_per_km2": "C2",
    "c2_road_density_500m_km_per_km2": "C2",
    "c2_road_density_1000m_km_per_km2": "C2",
    "c3_building_footprint_pct_cell": "C3",
    "c3_context_building_pct_250m": "C3",
    "c3_context_building_pct_500m": "C3",
    "c4_slope_mean_deg": "C4",
    "c5_elevation_mean_m": "C5",
}


def main():

    print("=" * 65)
    print("STAGE 3.4 - DISTRIBUTION AND REDUNDANCY")
    print("=" * 65)

    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)
    columns = list(CRITERIA.keys())

    required = [
        "grid_id",
        "kmc_overlap_fraction",
        "kmc_overlap_area_m2",
    ] + columns

    missing_columns = [
        c for c in required if c not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    for col in columns + [
        "kmc_overlap_fraction",
        "kmc_overlap_area_m2",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("\n[1] INPUT VALIDATION")

    checks = {
        "9240_grid_cells": len(df) == 9240,
        "unique_grid_ids": (
            df.grid_id.notna().all()
            and df.grid_id.is_unique
        ),
        "all_criteria_finite": bool(
            np.isfinite(
                df[columns].to_numpy(dtype=float)
            ).all()
        ),
        "valid_overlap_fractions": bool(
            df["kmc_overlap_fraction"]
            .between(0, 1).all()
        ),
        "positive_overlap_areas": bool(
            (df["kmc_overlap_area_m2"] > 0).all()
        ),
    }

    for name, result in checks.items():
        print(f"[{'PASS' if result else 'FAIL'}] {name}")

    if not all(checks.values()):
        raise SystemExit(
            "Input QA failed. Review before analysis."
        )

    # Boundary sensitivity comparison:
    # Do not use a minimum-overlap threshold to delete cells.
    # This subset is used only as a diagnostic.
    core = df[
        df["kmc_overlap_fraction"] >= 0.50
    ].copy()

    print("\nTotal cells:", len(df))
    print("Core comparison cells (>=50% KMC):", len(core))
    print("Boundary-small comparison cells:", len(df) - len(core))

    if len(core) < 100:
        raise ValueError("Unexpectedly small core subset")

    # --------------------------------------------------
    # 2. Raw criterion distribution
    # --------------------------------------------------

    print("\n[2] DISTRIBUTION ANALYSIS")

    distribution = []

    for col in columns:

        series = df[col]

        record = {
            "criterion": CRITERIA[col],
            "measurement": col,
            "count": int(series.count()),
            "missing": int(series.isna().sum()),
            "zero_pct": float(
                np.isclose(series, 0).mean() * 100
            ),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "p05": float(series.quantile(0.05)),
            "p25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "p75": float(series.quantile(0.75)),
            "p95": float(series.quantile(0.95)),
            "p99": float(series.quantile(0.99)),
            "max": float(series.max()),
            "unique_values": int(series.nunique()),
        }

        distribution.append(record)

    dist_df = pd.DataFrame(distribution)

    print(
        dist_df[
            [
                "criterion",
                "measurement",
                "zero_pct",
                "median",
                "p95",
                "max",
            ]
        ].round(3).to_string(index=False)
    )

    # --------------------------------------------------
    # 3. Pearson and Spearman correlations
    # --------------------------------------------------

    print("\n[3] CORRELATION ANALYSIS")

    pearson = df[columns].corr(method="pearson")
    spearman = df[columns].corr(method="spearman")

    core_spearman = core[columns].corr(
        method="spearman"
    )

    pairs = []

    for a, b in combinations(columns, 2):

        p = float(pearson.loc[a, b])
        s = float(spearman.loc[a, b])
        sc = float(core_spearman.loc[a, b])

        review_flag = (
            abs(s) >= REVIEW_THRESHOLD
            or abs(sc) >= REVIEW_THRESHOLD
        )

        pairs.append({
            "criterion_a": CRITERIA[a],
            "measurement_a": a,
            "criterion_b": CRITERIA[b],
            "measurement_b": b,
            "same_conceptual_criterion": (
                CRITERIA[a] == CRITERIA[b]
            ),
            "pearson_all": p,
            "spearman_all": s,
            "spearman_core": sc,
            "abs_spearman_all": abs(s),
            "core_difference": abs(s - sc),
            "review_flag": bool(review_flag),
        })

    pairs_df = pd.DataFrame(pairs)

    pairs_df = pairs_df.sort_values(
        "abs_spearman_all",
        ascending=False,
    )

    print("\nTop correlation pairs:")

    print(
        pairs_df[
            [
                "criterion_a",
                "criterion_b",
                "measurement_a",
                "measurement_b",
                "spearman_all",
                "spearman_core",
                "review_flag",
            ]
        ]
        .head(15)
        .round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------
    # 4. Redundancy review summary
    # --------------------------------------------------

    print("\n[4] REDUNDANCY REVIEW")

    flagged = pairs_df[pairs_df["review_flag"]]

    cross_concept = flagged[
        ~flagged["same_conceptual_criterion"]
    ]

    same_concept = flagged[
        flagged["same_conceptual_criterion"]
    ]

    print("Total measurement pairs:", len(pairs_df))
    print("Strong-correlation review pairs:", len(flagged))
    print("Same-concept alternative pairs:", len(same_concept))
    print("Cross-concept review pairs:", len(cross_concept))

    print("\nImportant methodological rules:")
    print("- C2 radii are alternatives, not separate AHP criteria.")
    print("- C3 context radii are alternatives.")
    print("- Current-cell occupation is distinct from surrounding context.")
    print("- C4/C5 redundancy must be reviewed.")
    print("- C5 elevation direction remains undecided.")
    print("- High correlation is a review flag, not automatic removal.")

    # --------------------------------------------------
    # 5. Final QA and compact exports
    # --------------------------------------------------

    print("\n[5] FINAL QA")

    checks["all_measurements_vary"] = bool(
        (dist_df["unique_values"] > 1).all()
    )

    checks["all_correlations_finite"] = bool(
        np.isfinite(
            pairs_df[
                [
                    "pearson_all",
                    "spearman_all",
                    "spearman_core",
                ]
            ].to_numpy(dtype=float)
        ).all()
    )

    checks["expected_pair_count"] = (
        len(pairs_df)
        == len(columns) * (len(columns) - 1) // 2
    )

    status = "PASS" if all(checks.values()) else "REVIEW_REQUIRED"

    for name in [
        "all_measurements_vary",
        "all_correlations_finite",
        "expected_pair_count",
    ]:
        print(
            f"[{'PASS' if checks[name] else 'FAIL'}] {name}"
        )

    summary = {
        "stage": "3.4",
        "status": status,
        "total_grid_cells": len(df),
        "core_comparison_cells": len(core),
        "raw_measurements": len(columns),
        "correlation_pairs": len(pairs_df),
        "review_threshold_abs_spearman": REVIEW_THRESHOLD,
        "strong_correlation_pairs": len(flagged),
        "same_concept_alternative_review_pairs": len(same_concept),
        "cross_concept_review_pairs": len(cross_concept),
        "checks": checks,
        "methodological_controls": {
            "criterion_selection_finalized": False,
            "directions_finalized": False,
            "standardization_done": False,
            "ahp_weights_assigned": False,
            "suitability_score_calculated": False,
        },
        "interpretation": [
            "Correlation is not proof of causality.",
            "Candidate measurement radii are alternatives.",
            "Small boundary cells are retained in primary analysis.",
            "Core-subset correlations are diagnostic only.",
            "Final selection requires planning justification.",
        ],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dist_path = (
        REPORT_DIR / "03c_distribution_summary.csv"
    )

    corr_path = (
        REPORT_DIR / "03c_correlation_review.csv"
    )

    qa_path = (
        REPORT_DIR / "03c_analysis_qa.json"
    )

    dist_df.to_csv(dist_path, index=False)
    pairs_df.to_csv(corr_path, index=False)

    with qa_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nGenerated reports:")
    print(dist_path)
    print(corr_path)
    print(qa_path)

    print("\n" + "=" * 65)
    print("STAGE 3.4 ANALYTICAL QA:", status)
    print("Final criterion selection: PENDING REVIEW")
    print("No standardization or AHP performed")
    print("=" * 65)

    if status != "PASS":
        raise SystemExit(
            "Review analytical QA before continuing."
        )


if __name__ == "__main__":
    main()
