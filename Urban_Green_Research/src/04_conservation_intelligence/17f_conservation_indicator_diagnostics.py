
"""
J26-DS-338 | Section 04 | Stage 4.7
Conservation Indicator QA, Distribution and Redundancy Review.

This diagnostic does NOT select or weight the final indicators.
It does NOT calculate Green Importance or Conservation Priority.

Historical green loss remains a separate THREAT domain.

Excluded from this analysis:
- Urbanization Pressure
- SHAP
- Development Suitability
- AHP
- Provisional UDA zoning

Outputs:
  17f_indicator_distributions.csv
  17f_indicator_correlations.csv
  17f_indicator_diagnostics_qa.json
"""

from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]

DATA = (
    ROOT / "data/processed/conservation_intelligence"
)

REPORT = (
    ROOT / "outputs/reports/conservation_intelligence"
)

SOURCE = {
    "current": (
        DATA / "current_green/"
        "17b_current_green_condition_2025_100m.csv"
    ),
    "persistence": (
        DATA / "persistence/"
        "17c_green_persistence_2011_2024_100m.csv"
    ),
    "connectivity": (
        DATA / "connectivity/"
        "17d_green_connectivity_2025_100m.csv"
    ),
    "historical_threat": (
        DATA / "historical_threat/"
        "17e_historical_green_loss_exposure_100m.csv"
    ),
}

AUDITS = {
    "current": (
        "17b_current_green_condition_qa.json",
        "PASS_CURRENT_GREEN_BASELINE",
    ),
    "persistence": (
        "17c_green_persistence_qa.json",
        "PASS_HISTORICAL_PERSISTENCE",
    ),
    "connectivity": (
        "17d_green_connectivity_qa.json",
        "PASS_CONNECTIVITY_CANDIDATES",
    ),
    "historical_threat": (
        "17e_historical_green_loss_exposure_qa.json",
        "PASS_HISTORICAL_LOSS_EXPOSURE",
    ),
}

GRID_GPKG = (
    DATA / "current_green/"
    "17b_current_green_condition_2025_100m.gpkg"
)

DIST_OUT = REPORT / "17f_indicator_distributions.csv"

CORR_OUT = REPORT / "17f_indicator_correlations.csv"

QA_OUT = REPORT / "17f_indicator_diagnostics_qa.json"


def read_grid_csv(path, name):

    frame = pd.read_csv(path)

    if (
        "grid_id" not in frame.columns
        or len(frame) != 9240
        or frame.grid_id.isna().any()
        or not frame.grid_id.is_unique
    ):
        raise ValueError(
            f"{name}: expected 9,240 unique grid IDs"
        )

    return frame.set_index("grid_id")


def get_numeric(frame, name, source):

    if name not in frame.columns:
        raise ValueError(
            f"{source}: missing column {name}"
        )

    return pd.to_numeric(
        frame[name],
        errors="coerce",
    ).to_numpy(dtype=float)


def quantiles(values):

    values = np.asarray(values, dtype=float)

    values = values[np.isfinite(values)]

    names = (
        "min",
        "p05",
        "p25",
        "median",
        "p75",
        "p95",
        "max",
    )

    if len(values) == 0:
        return {
            name: None
            for name in names
        }

    probabilities = (
        0.0,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        1.0,
    )

    return dict(
        zip(
            names,
            [
                float(np.quantile(values, p))
                for p in probabilities
            ],
        )
    )


def main():

    print("=" * 76)
    print(
        "SECTION 04 | STAGE 4.7 - "
        "INDICATOR QA AND REDUNDANCY DIAGNOSTICS"
    )
    print("=" * 76)

    required = [
        *SOURCE.values(),
        GRID_GPKG,
        *(
            REPORT / item[0]
            for item in AUDITS.values()
        ),
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    checks = {}

    def check(label, ok):

        checks[label] = bool(ok)

        print(
            f"[{'PASS' if ok else 'FAIL'}] {label}"
        )

    # ==================================================
    # 1. VALIDATE COMPLETED SOURCE STAGES
    # ==================================================

    print("\n[1] VERIFY FOUR COMPLETED SOURCE STAGES")

    for name, (filename, expected) in AUDITS.items():

        audit = json.loads(
            (REPORT / filename).read_text(
                encoding="utf-8"
            )
        )

        check(
            f"upstream_{name}_passed",
            audit.get("status") == expected,
        )

    if not all(checks.values()):
        raise SystemExit(
            "Previous stage validation failed."
        )

    # ==================================================
    # 2. ALIGN ALL DATA USING GRID ID
    # ==================================================

    data = {
        name: read_grid_csv(path, name)
        for name, path in SOURCE.items()
    }

    current = data["current"]

    grid_ids = current.index

    for name, frame in data.items():

        check(
            f"grid_ids_{name}_match",
            len(frame) == 9240
            and set(frame.index) == set(grid_ids),
        )

        data[name] = frame.loc[grid_ids]

    geo = gpd.read_file(GRID_GPKG)

    check(
        "gis_9240_unique_ids",
        len(geo) == 9240
        and geo.grid_id.notna().all()
        and geo.grid_id.is_unique,
    )

    check(
        "gis_epsg32644_valid",
        geo.crs is not None
        and geo.crs.to_epsg() == 32644
        and geo.geometry.notna().all()
        and geo.geometry.is_valid.all(),
    )

    check(
        "gis_grid_ids_match",
        set(geo.grid_id) == set(grid_ids),
    )

    if not all(checks.values()):
        raise SystemExit(
            "Grid or GIS verification failed."
        )

    persistence = data["persistence"]
    connectivity = data["connectivity"]
    historical = data["historical_threat"]

    # ==================================================
    # 3. VERIFY CURRENT GREEN AND HISTORICAL DATA
    # ==================================================

    print("\n[2] VERIFY GREEN COVERAGE AND PATCH DOMAINS")

    coverage = get_numeric(
        current,
        "valid_lulc_coverage_fraction",
        "current",
    )

    green = get_numeric(
        current,
        "current_green_fraction",
        "current",
    )

    green_area = get_numeric(
        current,
        "current_green_area_m2",
        "current",
    )

    valid_area = get_numeric(
        current,
        "valid_lulc_area_m2",
        "current",
    )

    kmc_area = get_numeric(
        current,
        "kmc_overlap_area_m2",
        "current",
    )

    valid_years = get_numeric(
        persistence,
        "historical_valid_year_count",
        "persistence",
    )

    valid_current = valid_area > 1e-9

    remaining_green_detected = (
        valid_current
        & (green_area > 1e-7)
    )

    high_coverage_green = (
        remaining_green_detected
        & (coverage >= 0.80)
    )

    check(
        "2025_raw_product",
        current.source_product_type.eq(
            "RAW_RF"
        ).all(),
    )

    check(
        "valid_2025_green_9195_unknown_45",
        int(valid_current.sum()) == 9195
        and int((~valid_current).sum()) == 45
        and np.isnan(
            green[~valid_current]
        ).all(),
    )

    check(
        "2025_green_and_coverage_bounded",
        np.isfinite(
            green[valid_current]
        ).all()
        and (
            (green[valid_current] >= 0)
            & (green[valid_current] <= 1)
        ).all()
        and np.isfinite(coverage).all()
        and (
            (coverage >= 0)
            & (coverage <= 1)
        ).all(),
    )

    check(
        "2025_green_area_valid",
        np.isfinite(kmc_area).all()
        and (kmc_area > 0).all()
        and np.isfinite(valid_area).all()
        and (valid_area >= 0).all()
        and np.isfinite(green_area).all()
        and (green_area >= 0).all()
        and (
            green_area <= valid_area + 1e-6
        ).all(),
    )

    check(
        "historical_valid_year_counts",
        np.isfinite(valid_years).all()
        and (
            (valid_years >= 0)
            & (valid_years <= 14)
        ).all(),
    )

    check(
        "current_green_detected_count",
        int(remaining_green_detected.sum())
        == 8727,
    )

    patch_area = get_numeric(
        connectivity,
        "green_patch_4connected_mapped_area_km2",
        "connectivity",
    )

    patch_flag = (
        connectivity[
            "exploratory_majority_green_patch_candidate"
        ]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    check(
        "patch_flag_valid",
        patch_flag.isin(
            ["true", "false"]
        ).all(),
    )

    is_patch = patch_flag.eq(
        "true"
    ).to_numpy(dtype=bool)

    check(
        "patch_5974_captured",
        int(is_patch.sum()) == 5974,
    )

    check(
        "patch_area_only_for_candidate_cells",
        np.isfinite(
            patch_area[is_patch]
        ).all()
        and (
            patch_area[is_patch] > 0
        ).all()
        and np.isnan(
            patch_area[~is_patch]
        ).all(),
    )

    geo = (
        geo.set_index("grid_id")
        .loc[grid_ids]
    )

    geometry_error = float(
        np.max(
            np.abs(
                geo.geometry.area.to_numpy(dtype=float)
                - kmc_area
            )
        )
    )

    check(
        "gis_areas_match",
        geometry_error <= 1.0,
    )

    check(
        "all_analysis_cells_kept",
        len(grid_ids) == 9240,
    )

    if not all(checks.values()):
        raise SystemExit(
            "Input QA failed. No diagnostics generated."
        )

    # ==================================================
    # 4. REGISTER CANDIDATE INDICATORS
    # ==================================================

    print("\n[3] REGISTER INDICATORS AND DISTRIBUTIONS")

    # Indicator record:
    # name, purpose, domain, values, bounded_0_to_1.
    #
    # Green Importance and Historical Threat remain
    # conceptually separate.
    #
    # For patch area, non-patch cells stay NaN.
    # We must not assume non-members have zero ecological
    # connectivity or zero conservation importance.

    indicators = [
        (
            "current_green_fraction",
            "GREEN_IMPORTANCE",
            "CURRENT",
            green,
            True,
        ),
        (
            "confirmed_green_share_lower_bound",
            "GREEN_IMPORTANCE",
            "CURRENT",
            np.where(
                valid_current,
                get_numeric(
                    current,
                    "mapped_green_share_lower_bound",
                    "current",
                ),
                np.nan,
            ),
            True,
        ),
        (
            "persistence_dominant_year_share",
            "GREEN_IMPORTANCE",
            "PERSISTENCE",
            get_numeric(
                persistence,
                "historical_green_dominant_year_share",
                "persistence",
            ),
            True,
        ),
        (
            "persistence_mean_green_fraction",
            "GREEN_IMPORTANCE",
            "PERSISTENCE",
            get_numeric(
                persistence,
                "historical_mean_green_fraction",
                "persistence",
            ),
            True,
        ),
        (
            "persistence_majority_year_share",
            "GREEN_IMPORTANCE",
            "PERSISTENCE",
            get_numeric(
                persistence,
                "historical_majority_green_year_share",
                "persistence",
            ),
            True,
        ),
        (
            "green_neighbour_context_250m",
            "GREEN_IMPORTANCE",
            "CONNECTIVITY",
            get_numeric(
                connectivity,
                "green_neighbour_context_250m",
                "connectivity",
            ),
            True,
        ),
        (
            "green_neighbour_context_500m",
            "GREEN_IMPORTANCE",
            "CONNECTIVITY",
            get_numeric(
                connectivity,
                "green_neighbour_context_500m",
                "connectivity",
            ),
            True,
        ),
        (
            "green_patch_area_km2",
            "GREEN_IMPORTANCE",
            "CONNECTIVITY",
            patch_area,
            False,
        ),
        (
            "robust_g2b_exposure_share_250m",
            "HISTORICAL_THREAT",
            "LOSS",
            get_numeric(
                historical,
                "historical_g2b_robust_exposure_share_250m",
                "historical_threat",
            ),
            True,
        ),
        (
            "robust_g2b_exposure_share_500m",
            "HISTORICAL_THREAT",
            "LOSS",
            get_numeric(
                historical,
                "historical_g2b_robust_exposure_share_500m",
                "historical_threat",
            ),
            True,
        ),
        (
            "strict_g2b_exposure_share_250m",
            "HISTORICAL_THREAT",
            "LOSS",
            get_numeric(
                historical,
                "historical_g2b_strict_exposure_share_250m",
                "historical_threat",
            ),
            True,
        ),
        (
            "strict_g2b_exposure_share_500m",
            "HISTORICAL_THREAT",
            "LOSS",
            get_numeric(
                historical,
                "historical_g2b_strict_exposure_share_500m",
                "historical_threat",
            ),
            True,
        ),
        (
            "robust_nearest_conversion_pixel_m",
            "HISTORICAL_THREAT",
            "LOSS",
            get_numeric(
                historical,
                "historical_g2b_robust_nearest_pixel_centre_m",
                "historical_threat",
            ),
            False,
        ),
        (
            "robust_conversion_area_500m_km2",
            "HISTORICAL_THREAT",
            "LOSS",
            get_numeric(
                historical,
                "historical_g2b_robust_converted_area_km2_500m",
                "historical_threat",
            ),
            False,
        ),
    ]

    # Three analysis populations:
    #
    # 1. All cells with observed current green condition
    # 2. Cells with green actually detected
    # 3. Detected-green cells with >=80% LULC coverage
    #
    # The 80% rule is ONLY a QA sensitivity cohort.
    # It is not a conservation eligibility threshold.

    cohorts = {
        "ALL_VALID_CURRENT_LULC": (
            valid_current
        ),
        "CURRENT_GREEN_DETECTED": (
            remaining_green_detected
        ),
        "CURRENT_GREEN_COVERAGE_GE80": (
            high_coverage_green
        ),
    }

    distributions = []

    print(
        "Current green detected:",
        int(remaining_green_detected.sum()),
    )

    print(
        "Current green with >=80% coverage:",
        int(high_coverage_green.sum()),
    )

    for name, purpose, domain, values, bounded in indicators:

        for cohort_name, cohort in cohorts.items():

            selected = values[cohort]

            finite = selected[
                np.isfinite(selected)
            ]

            stats = quantiles(finite)

            distributions.append({
                "indicator": name,
                "purpose": purpose,
                "domain": domain,
                "cohort": cohort_name,
                "cohort_size": int(cohort.sum()),
                "observed_count": int(len(finite)),
                "missing_count": int(
                    len(selected) - len(finite)
                ),
                "observed_pct": float(
                    100 * len(finite)
                    / max(len(selected), 1)
                ),
                "distinct_rounded_6dp": int(
                    np.unique(
                        np.round(finite, 6)
                    ).size
                ),
                "at_zero_pct_of_observed": (
                    float(
                        100 * np.isclose(
                            finite, 0
                        ).mean()
                    )
                    if bounded and len(finite)
                    else None
                ),
                "at_one_pct_of_observed": (
                    float(
                        100 * np.isclose(
                            finite, 1
                        ).mean()
                    )
                    if bounded and len(finite)
                    else None
                ),
                **stats,
            })

        shown = next(
            row for row in distributions
            if (
                row["indicator"] == name
                and row["cohort"]
                == "CURRENT_GREEN_DETECTED"
            )
        )

        print(
            f"{name:<40} "
            f"valid={shown['observed_count']:>5} "
            f"median={shown['median']}"
        )

    # ==================================================
    # 5. PEARSON AND SPEARMAN REDUNDANCY
    # ==================================================

    print("\n[4] PEARSON / SPEARMAN REDUNDANCY")

    correlations = []

    for cohort_name, cohort in cohorts.items():

        for i, left in enumerate(indicators):

            for right in indicators[i + 1:]:

                (
                    a_name,
                    a_purpose,
                    a_domain,
                    a_values,
                    _,
                ) = left

                (
                    b_name,
                    b_purpose,
                    b_domain,
                    b_values,
                    _,
                ) = right

                valid_pair = (
                    cohort
                    & np.isfinite(a_values)
                    & np.isfinite(b_values)
                )

                x = a_values[valid_pair]
                y = b_values[valid_pair]

                enough = (
                    len(x) >= 30
                    and np.std(x) > 1e-12
                    and np.std(y) > 1e-12
                )

                pearson = (
                    float(
                        np.corrcoef(x, y)[0, 1]
                    )
                    if enough
                    else None
                )

                spearman = (
                    float(
                        spearmanr(x, y).statistic
                    )
                    if enough
                    else None
                )

                if (
                    a_purpose == "GREEN_IMPORTANCE"
                    and b_purpose == "GREEN_IMPORTANCE"
                ):
                    relation = "WITHIN_GREEN_IMPORTANCE"

                elif (
                    a_purpose == "HISTORICAL_THREAT"
                    and b_purpose == "HISTORICAL_THREAT"
                ):
                    relation = "WITHIN_HISTORICAL_THREAT"

                else:
                    relation = (
                        "CROSS_PURPOSE_DESCRIPTIVE_ONLY"
                    )

                high_correlation = bool(
                    spearman is not None
                    and abs(spearman) >= 0.85
                )

                correlations.append({
                    "cohort": cohort_name,
                    "indicator_a": a_name,
                    "indicator_b": b_name,
                    "domain_a": a_domain,
                    "domain_b": b_domain,
                    "comparison_role": relation,
                    "paired_cells": int(len(x)),
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                    "high_abs_spearman_ge85": (
                        high_correlation
                    ),
                    "interpretation": (
                        "Candidate redundancy - manual review"
                        if (
                            high_correlation
                            and relation !=
                            "CROSS_PURPOSE_DESCRIPTIVE_ONLY"
                        )
                        else (
                            "Descriptive comparison; "
                            "not automatic selection"
                        )
                    ),
                })

    distribution_df = pd.DataFrame(
        distributions
    )

    correlation_df = pd.DataFrame(
        correlations
    )

    selected_review = correlation_df.loc[
        (
            correlation_df["cohort"]
            == "CURRENT_GREEN_DETECTED"
        )
        & (
            correlation_df["high_abs_spearman_ge85"]
        )
        & (
            correlation_df["comparison_role"]
            != "CROSS_PURPOSE_DESCRIPTIVE_ONLY"
        )
    ].copy()

    selected_review["absolute_rho"] = (
        selected_review.spearman_rho.abs()
    )

    selected_review = (
        selected_review.sort_values(
            "absolute_rho",
            ascending=False,
        )
    )

    print(
        "High within-purpose correlations "
        "(|Spearman| >= 0.85):",
        len(selected_review),
    )

    for _, row in selected_review.head(20).iterrows():

        print(
            f"  {row.indicator_a} "
            f"vs {row.indicator_b}: "
            f"rho={row.spearman_rho:.4f}, "
            f"n={row.paired_cells}"
        )

    if len(selected_review) > 20:
        print(
            "Additional flagged pairs saved in CSV."
        )

    # ==================================================
    # 6. FINAL DIAGNOSTIC QA
    # ==================================================

    print("\n[5] FINAL DIAGNOSTIC QA")

    check(
        "indicator_count_14",
        len(indicators) == 14,
    )

    check(
        "all_indicator_lengths_9240",
        all(
            len(item[3]) == 9240
            for item in indicators
        ),
    )

    check(
        "indicator_nonmissing_values_nonnegative",
        all(
            (
                item[3][np.isfinite(item[3])]
                >= -1e-9
            ).all()
            for item in indicators
        ),
    )

    check(
        "fraction_indicator_values_within_0_1",
        all(
            (
                (
                    item[3][np.isfinite(item[3])]
                    >= -1e-9
                )
                &
                (
                    item[3][np.isfinite(item[3])]
                    <= 1 + 1e-9
                )
            ).all()
            for item in indicators
            if item[4]
        ),
    )

    check(
        "expected_distribution_rows",
        len(distribution_df)
        == 3 * len(indicators),
    )

    check(
        "expected_pairwise_rows",
        len(correlation_df)
        == (
            3 * len(indicators)
            * (len(indicators) - 1) // 2
        ),
    )

    check(
        "sufficient_green_cells_for_review",
        int(remaining_green_detected.sum()) > 8000,
    )

    check(
        "historical_loss_kept_separate",
        all(
            item[1] == "HISTORICAL_THREAT"
            for item in indicators
            if item[2] == "LOSS"
        ),
    )

    forbidden = [
        "pressure",
        "suitability",
        "shap",
        "zoning",
        "ahp",
    ]

    check(
        "no_future_pressure_suitability_shap_inputs",
        not any(
            any(
                token in name
                for token in forbidden
            )
            for name, *_ in indicators
        ),
    )

    check(
        "high_correlation_rows_have_sufficient_pairs",
        bool(
            (
                selected_review.paired_cells >= 30
            ).all()
        ),
    )

    if not all(checks.values()):
        raise SystemExit(
            "Stage 4.7 diagnostics failed QA. "
            "No reports exported."
        )

    # ==================================================
    # 7. EXPORT THREE REPORTS
    # ==================================================

    print("\n[6] SAVE THREE EVIDENCE REPORTS")

    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    distribution_df.to_csv(
        DIST_OUT,
        index=False,
    )

    correlation_df.to_csv(
        CORR_OUT,
        index=False,
    )

    qa = {
        "stage": "4.7",
        "status": (
            "PASS_DIAGNOSTICS_FINAL_SELECTION_PENDING"
        ),
        "grid_cells": len(grid_ids),
        "current_green_detected_cells": int(
            remaining_green_detected.sum()
        ),
        "current_green_coverage_ge80_cells": int(
            high_coverage_green.sum()
        ),
        "unknown_current_green_cells": int(
            (~valid_current).sum()
        ),
        "indicator_count": len(indicators),
        "correlation_cohorts": {
            key: int(value.sum())
            for key, value in cohorts.items()
        },
        "correlation_threshold_for_review": 0.85,
        "within_purpose_high_correlation_pairs_current_green": (
            int(len(selected_review))
        ),
        "flagged_pairs_current_green": [
            {
                "a": row.indicator_a,
                "b": row.indicator_b,
                "spearman": float(
                    row.spearman_rho
                ),
                "paired_cells": int(
                    row.paired_cells
                ),
            }
            for row in selected_review.itertuples(
                index=False
            )
        ],
        "checks": checks,
        "selected_final_indicators": None,
        "final_indicator_selection_frozen": False,
        "green_importance_calculated": False,
        "future_pressure_used": False,
        "development_suitability_used": False,
        "notes": [
            "This is candidate-indicator QA and "
            "redundancy review, not automatic selection.",
            "Strong correlation does not prove "
            "different ecological concepts are identical.",
            "Correlation uses observed 2025 green cells "
            "and additional QA sensitivity cohorts.",
            "Patch area is available only for exploratory "
            "majority-green patch members.",
            "Patch size is skewed; consider log scaling "
            "only after indicator selection.",
            "The 80% coverage cohort is diagnostic, "
            "not a conservation exclusion.",
            "Historical conversion exposure belongs "
            "to threat, not Green Importance.",
            "Historical conversion near-everywhere "
            "detection at 500m requires intensity review.",
            "Indicators are not yet standardized or weighted.",
            "No future Urbanization Pressure introduced.",
        ],
    }

    QA_OUT.write_text(
        json.dumps(
            qa,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Distributions:", DIST_OUT)
    print("Correlations:", CORR_OUT)
    print("QA:", QA_OUT)

    print("\n" + "=" * 76)
    print(
        "STAGE 4.7: "
        "PASS_DIAGNOSTICS_FINAL_SELECTION_PENDING"
    )
    print(
        "Final Green Importance indicators: NOT YET FROZEN"
    )
    print(
        "Conservation weighting: NOT CHOSEN"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
