
"""
J26-DS-338 | Section 04, Stage 4.8
Independent Green Importance: proposed criteria and identity standardization.

G1: 2025 RAW_RF observed current green fraction.
G2: 2011-2024 stabilized green-dominant-year share.
G3: 2025 RAW_RF green neighbourhood context, 250 m, focal cell excluded.

No Green Importance weights, composite score, historical loss, future pressure,
SHAP, development suitability or zoning enter this stage.

Corrects the Stage 4.8 numerical-precision QA failure: up to 1e-9 is
accepted for coverage ratios near their physical bounds. Only the exported
neighbourhood-coverage QA field is clipped to [0,1], with the original
rounding deviations documented. Underlying sources are never modified.
"""

from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/conservation_intelligence"
REPORT = ROOT / "outputs/reports/conservation_intelligence"

FILES = {
    "current": DATA / "current_green/17b_current_green_condition_2025_100m.csv",
    "persistence": DATA / "persistence/17c_green_persistence_2011_2024_100m.csv",
    "connectivity": DATA / "connectivity/17d_green_connectivity_2025_100m.csv",
}

GIS_SOURCE = DATA / "current_green/17b_current_green_condition_2025_100m.gpkg"

UPSTREAM = {
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
    "diagnostics": (
        "17f_indicator_diagnostics_qa.json",
        "PASS_DIAGNOSTICS_FINAL_SELECTION_PENDING",
    ),
}

OUT_DIR = DATA / "importance_criteria"

CSV_OUT = (
    OUT_DIR /
    "17g_green_importance_criteria_standardized_100m.csv"
)

GPKG_OUT = (
    OUT_DIR /
    "17g_green_importance_criteria_standardized_100m.gpkg"
)

QA_OUT = (
    REPORT /
    "17g_green_importance_criteria_qa.json"
)

EPS = 1e-9


def load_unique(path, name):

    frame = pd.read_csv(path)

    if (
        "grid_id" not in frame.columns
        or len(frame) != 9240
        or frame["grid_id"].isna().any()
        or not frame["grid_id"].is_unique
    ):
        raise ValueError(
            f"{name}: 9,240 nonmissing unique grid IDs required"
        )

    return frame.set_index("grid_id")


def numeric(frame, column, name):

    if column not in frame.columns:
        raise ValueError(
            f"{name}: missing column '{column}'"
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    ).to_numpy(dtype=float)


def within_unit_interval(array, tolerance=EPS):

    return bool(
        np.isfinite(array).all()
        and (array >= -tolerance).all()
        and (array <= 1.0 + tolerance).all()
    )


def describe(array):

    values = np.asarray(array, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {"n": 0}

    return {
        "n": int(values.size),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def main():

    print("=" * 76)
    print(
        "SECTION 04 | STAGE 4.8 - "
        "GREEN IMPORTANCE CRITERION STANDARDIZATION"
    )
    print("=" * 76)

    required = [
        *FILES.values(),
        GIS_SOURCE,
        *(
            REPORT / item[0]
            for item in UPSTREAM.values()
        ),
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    checks = {}

    def check(name, condition):

        passed = bool(condition)
        checks[name] = passed

        print(
            f"[{'PASS' if passed else 'FAIL'}] "
            f"{name}"
        )

    def require_pass(stage):

        if not all(checks.values()):
            raise SystemExit(
                f"{stage}: QA failed; no outputs exported"
            )

    # ==================================================
    # 1. VERIFY COMPLETED STAGES
    # ==================================================

    print("\n[1] VERIFY COMPLETED STAGES")

    for name, (filename, expected) in UPSTREAM.items():

        audit = json.loads(
            (REPORT / filename).read_text(
                encoding="utf-8"
            )
        )

        check(
            f"prior_{name}_passed",
            audit.get("status") == expected,
        )

    require_pass("Upstream")

    # ==================================================
    # 2. ALIGN DATA BY GRID ID
    # ==================================================

    print("\n[2] ALIGN DATA BY GRID ID")

    frames = {
        name: load_unique(path, name)
        for name, path in FILES.items()
    }

    current = frames["current"]
    ids = current.index

    for name in ("persistence", "connectivity"):

        check(
            f"{name}_ids_match",
            set(frames[name].index) == set(ids),
        )

        frames[name] = frames[name].reindex(ids)

    persistence = frames["persistence"]
    connectivity = frames["connectivity"]

    gis = gpd.read_file(GIS_SOURCE)

    check(
        "gis_9240_unique_matching_ids",
        len(gis) == 9240
        and gis.grid_id.notna().all()
        and gis.grid_id.is_unique
        and set(gis.grid_id) == set(ids),
    )

    check(
        "gis_epsg32644_valid",
        gis.crs is not None
        and gis.crs.to_epsg() == 32644
        and gis.geometry.notna().all()
        and gis.geometry.is_valid.all(),
    )

    require_pass("Alignment")

    gis = (
        gis.set_index("grid_id")
        .loc[ids]
    )

    # ==================================================
    # 3. VERIFY SOURCE PROVENANCE
    # ==================================================

    print("\n[3] VERIFY SOURCE PROVENANCE")

    check(
        "2025_source_raw_rf",
        current.source_product_type.eq("RAW_RF").all(),
    )

    check(
        "history_2011_2024_stabilized",
        persistence.historical_product_type.eq(
            "STABILIZED_ANALYTICAL"
        ).all()
        and np.all(
            numeric(
                persistence,
                "historical_start_year",
                "persistence",
            ) == 2011
        )
        and np.all(
            numeric(
                persistence,
                "historical_end_year",
                "persistence",
            ) == 2024
        ),
    )

    check(
        "connectivity_2025_raw_rf",
        connectivity.connectivity_source_product.eq(
            "RAW_RF"
        ).all(),
    )

    # Three candidate Green Importance criteria.

    g1 = numeric(
        current,
        "current_green_fraction",
        "current",
    )

    g2 = numeric(
        persistence,
        "historical_green_dominant_year_share",
        "persistence",
    )

    g3 = numeric(
        connectivity,
        "green_neighbour_context_250m",
        "connectivity",
    )

    coverage_current = numeric(
        current,
        "valid_lulc_coverage_fraction",
        "current",
    )

    coverage_neighbours_raw = numeric(
        connectivity,
        "green_neighbour_coverage_250m",
        "connectivity",
    )

    historical_years = numeric(
        persistence,
        "historical_valid_year_count",
        "persistence",
    )

    kmc_area = numeric(
        current,
        "kmc_overlap_area_m2",
        "current",
    )

    green_area = numeric(
        current,
        "current_green_area_m2",
        "current",
    )

    current_status = (
        current.green_condition_status
        .astype(str)
        .to_numpy()
    )

    # ==================================================
    # 4. COVERAGE AND CROSS-SOURCE VALIDATION
    # ==================================================

    print(
        "\n[4] CHECK COVERAGE AND "
        "CROSS-SOURCE CONSISTENCY"
    )

    check(
        "current_green_cross_source_matches",
        np.allclose(
            g1,
            numeric(
                persistence,
                "current_green_fraction",
                "persistence",
            ),
            atol=1e-8,
            equal_nan=True,
        )
        and np.allclose(
            g1,
            numeric(
                connectivity,
                "current_green_fraction",
                "connectivity",
            ),
            atol=1e-8,
            equal_nan=True,
        ),
    )

    check(
        "current_coverage_cross_source_matches",
        np.allclose(
            coverage_current,
            numeric(
                persistence,
                "current_2025_lulc_coverage_fraction",
                "persistence",
            ),
            atol=1e-8,
        )
        and np.allclose(
            coverage_current,
            numeric(
                connectivity,
                "valid_lulc_coverage_fraction",
                "connectivity",
            ),
            atol=1e-8,
        ),
    )

    check(
        "gis_area_matches_current_grid",
        np.isfinite(kmc_area).all()
        and (kmc_area > 0).all()
        and np.max(
            np.abs(
                gis.geometry.area.to_numpy(dtype=float)
                - kmc_area
            )
        ) <= 1.0,
    )

    has_green = (
        current_status
        == "GREEN_DETECTED_IN_VALID_AREA"
    )

    no_green = (
        current_status
        == "NO_GREEN_DETECTED_IN_VALID_AREA"
    )

    unknown = (
        current_status
        == "UNKNOWN_NO_VALID_LULC"
    )

    check(
        "green_condition_partition_matches_frozen_counts",
        int(has_green.sum()) == 8727
        and int(no_green.sum()) == 468
        and int(unknown.sum()) == 45
        and np.all(
            has_green | no_green | unknown
        ),
    )

    check(
        "green_area_matches_status",
        (green_area[has_green] > 0).all()
        and np.allclose(
            green_area[no_green],
            0.0,
            atol=1e-7,
        )
        and np.isnan(
            g1[unknown]
        ).all(),
    )

    # ==================================================
    # IMPORTANT CORRECTION
    #
    # Previously: strict coverage <= 1.0
    #
    # Corrected: allow a numerical tolerance of 1e-9.
    #
    # Observed maximum:
    # 1.000000000000012
    #
    # This is floating-point rounding, not genuinely
    # invalid coverage.
    # ==================================================

    check(
        "source_coverage_valid_with_1e_9_tolerance",
        within_unit_interval(
            coverage_current
        )
        and within_unit_interval(
            coverage_neighbours_raw
        ),
    )

    check(
        "historical_year_counts_0_to_14",
        np.isfinite(historical_years).all()
        and (historical_years >= 0).all()
        and (historical_years <= 14).all(),
    )

    check(
        "green_area_nonnegative_within_grid_area",
        np.isfinite(green_area).all()
        and (green_area >= 0).all()
        and (green_area <= kmc_area + 1e-6).all(),
    )

    require_pass("Source")

    above_one = int(
        (coverage_neighbours_raw > 1.0).sum()
    )

    below_zero = int(
        (coverage_neighbours_raw < 0.0).sum()
    )

    max_overshoot = float(
        np.maximum(
            coverage_neighbours_raw - 1.0,
            0.0,
        ).max()
    )

    print(
        "Neighbour coverage values slightly above 1:",
        above_one,
    )

    print(
        "Largest excess above 1:",
        f"{max_overshoot:.16g}",
    )

    # Only the coverage QA diagnostic is bounded to
    # the physically meaningful interval [0,1].
    #
    # The three Green Importance criterion values
    # are NOT modified.

    coverage_neighbours_export = np.clip(
        coverage_neighbours_raw,
        0.0,
        1.0,
    )

    check(
        "rounding_adjustment_within_tolerance",
        np.allclose(
            coverage_neighbours_raw,
            coverage_neighbours_export,
            atol=EPS,
            rtol=0.0,
        ),
    )

    # ==================================================
    # 5. PREPARE THREE PROPOSED CRITERIA
    # ==================================================

    print(
        "\n[5] PREPARE THREE PROPOSED "
        "0-1 BENEFIT CRITERIA"
    )

    candidates = {
        "G1_current_green_0_1": g1,
        "G2_historical_persistence_0_1": g2,
        "G3_local_green_context_0_1": g3,
    }

    for name, values in candidates.items():

        check(
            f"{name}_bounded",
            within_unit_interval(
                values[np.isfinite(values)]
            ),
        )

    # Identity standardization.
    #
    # G1, G2 and G3 already represent 0-1 proportions.
    # No min-max, percentile or z-score transformation.
    #
    # Cells without detected current green remain
    # unscored; no artificial zero values are inserted.

    selected = {
        name: np.where(
            has_green,
            values,
            np.nan,
        )
        for name, values in candidates.items()
    }

    ready = has_green.copy()

    for values in selected.values():
        ready &= np.isfinite(values)

    availability = np.full(
        len(ids),
        "INCOMPLETE_GREEN_IMPORTANCE_CRITERIA",
        dtype=object,
    )

    availability[
        no_green
    ] = "NO_CURRENT_GREEN_DETECTED"

    availability[
        unknown
    ] = "UNKNOWN_2025_GREEN_CONDITION"

    availability[
        ready
    ] = "READY_FOR_WEIGHTING"

    print(
        "Currently green cells:",
        int(has_green.sum()),
    )

    print(
        "Complete green criteria:",
        int(ready.sum()),
    )

    print(
        "No detected current green:",
        int(no_green.sum()),
    )

    print(
        "Unknown current green:",
        int(unknown.sum()),
    )

    # ==================================================
    # 6. BUILD STANDARDIZED CRITERION TABLE
    # ==================================================

    print("\n[6] BUILD STANDARDIZED CRITERION TABLE")

    result = pd.DataFrame({
        "grid_id": list(ids),

        "source_current_year": 2025,

        "source_current_product": "RAW_RF",

        "current_green_condition_status":
            current_status,

        "criterion_availability_status":
            availability,

        "kmc_overlap_area_m2":
            kmc_area,

        "current_mapped_green_area_m2":
            green_area,

        "current_lulc_coverage_fraction":
            coverage_current,

        "historical_valid_year_count":
            historical_years.astype(int),

        "local_green_250m_valid_coverage_fraction":
            coverage_neighbours_export,

        "qa_current_coverage_below_80pct":
            coverage_current < 0.80,

        "qa_neighbour_coverage_below_80pct":
            coverage_neighbours_export < 0.80,

        "qa_history_fewer_than_14_valid_years":
            historical_years < 14,

        **selected,
    })

    # ==================================================
    # 7. STANDARDIZATION AND INDEPENDENCE QA
    # ==================================================

    print(
        "\n[7] VERIFY STANDARDIZATION "
        "AND INDEPENDENCE"
    )

    check(
        "all_9240_cells_retained",
        len(result) == 9240
        and result.grid_id.is_unique,
    )

    check(
        "8727_green_cells_ready",
        int(ready.sum()) == 8727,
    )

    check(
        "unknown_and_no_green_unscored",
        all(
            np.isnan(
                values[~has_green]
            ).all()
            for values in selected.values()
        ),
    )

    check(
        "all_ready_criteria_finite",
        all(
            np.isfinite(
                values[ready]
            ).all()
            for values in selected.values()
        ),
    )

    check(
        "identity_standardization_preserves_source_values",
        all(
            np.allclose(
                selected[name][ready],
                candidates[name][ready],
                atol=1e-12,
            )
            for name in selected
        ),
    )

    check(
        "ready_status_reproduced",
        np.array_equal(
            availability == "READY_FOR_WEIGHTING",
            ready,
        ),
    )

    check(
        "coverage_not_used_as_candidate_exclusion",
        np.array_equal(
            ready,
            has_green,
        ),
    )

    forbidden = (
        "pressure",
        "shap",
        "suitability",
        "ahp",
        "zoning",
        "priority",
        "historical_g2b",
        "historical_loss",
    )

    check(
        "no_threat_or_suitability_input_fields",
        not any(
            any(
                term in col.lower()
                for term in forbidden
            )
            for col in result.columns
        ),
    )

    check(
        "no_weight_or_composite_score_generated",
        not any(
            "score" in col.lower()
            or "weighted" in col.lower()
            for col in result.columns
        ),
    )

    # ==================================================
    # 8. RECHECK CORRELATIONS
    # ==================================================

    print("\n[8] RECHECK REDUNDANCY")

    names = list(selected)
    correlations = {}

    for i, first in enumerate(names):

        for second in names[i + 1:]:

            rho = float(
                spearmanr(
                    selected[first][ready],
                    selected[second][ready],
                ).statistic
            )

            check(
                f"spearman_{i+1}_{names.index(second)+1}_finite",
                np.isfinite(rho),
            )

            correlations[
                f"{first}__VS__{second}"
            ] = rho

            print(
                f"{first} vs {second}: "
                f"rho={rho:.4f}"
            )

    print("\nCandidate criterion distributions:")

    for name, values in selected.items():

        print(
            name,
            describe(values[ready]),
        )

    require_pass("Standardization")

    # ==================================================
    # 9. EXPORT
    # ==================================================

    print("\n[9] EXPORT STANDARDIZED CANDIDATES")

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        CSV_OUT,
        index=False,
    )

    geo = (
        gis[["geometry"]]
        .reset_index()
        .merge(
            result,
            on="grid_id",
            how="inner",
            validate="one_to_one",
        )
    )

    if len(geo) != 9240:
        raise ValueError(
            "GIS merge lost grid cells"
        )

    geo.to_file(
        GPKG_OUT,
        layer="green_importance_candidate_criteria_100m",
        driver="GPKG",
    )

    saved = pd.read_csv(CSV_OUT)

    saved_gis = pyogrio.read_info(
        GPKG_OUT
    )

    check(
        "csv_export_9240_unique",
        len(saved) == 9240
        and saved.grid_id.is_unique,
    )

    check(
        "gpkg_export_9240",
        saved_gis["features"] == 9240,
    )

    check(
        "saved_non_green_criteria_missing",
        all(
            saved.loc[
                ~has_green,
                name,
            ].isna().all()
            for name in selected
        ),
    )

    final_status = (
        "PASS_STANDARDIZED_CANDIDATES_WEIGHTING_PENDING"
        if all(checks.values())
        else "REVIEW_REQUIRED"
    )

    # ==================================================
    # 10. QA AND METHODOLOGY REPORT
    # ==================================================

    qa = {
        "section":
            "04_conservation_intelligence",

        "stage":
            "4.8",

        "status":
            final_status,

        "candidate_selection_provisional":
            True,

        "final_weighting_approved":
            False,

        "green_importance_calculated":
            False,

        "candidate_criteria": [
            {
                "code": "G1",
                "field": "G1_current_green_0_1",
                "direction": "BENEFIT",
                "source": (
                    "2025 RAW_RF observed green share "
                    "of valid LULC area"
                ),
                "standardization": "IDENTITY_0_1",
            },
            {
                "code": "G2",
                "field": "G2_historical_persistence_0_1",
                "direction": "BENEFIT",
                "source": (
                    "2011-2024 stabilized green-dominant "
                    "valid-year share"
                ),
                "standardization": "IDENTITY_0_1",
            },
            {
                "code": "G3",
                "field": "G3_local_green_context_0_1",
                "direction": "BENEFIT",
                "source": (
                    "2025 RAW_RF 250m neighbourhood, "
                    "focal cell excluded"
                ),
                "standardization": "IDENTITY_0_1",
            },
        ],

        "grid_cells":
            len(result),

        "currently_green_detected":
            int(has_green.sum()),

        "criterion_complete_green_cells":
            int(ready.sum()),

        "no_current_green_detected":
            int(no_green.sum()),

        "unknown_current_green_condition":
            int(unknown.sum()),

        "ready_cells_current_coverage_below_80pct":
            int(
                (
                    ready
                    & (coverage_current < 0.80)
                ).sum()
            ),

        "ready_cells_neighbour_coverage_below_80pct":
            int(
                (
                    ready
                    & (coverage_neighbours_export < 0.80)
                ).sum()
            ),

        "ready_cells_partial_history":
            int(
                (
                    ready
                    & (historical_years < 14)
                ).sum()
            ),

        "coverage_precision_diagnostic": {
            "tolerance": EPS,
            "neighbour_values_above_one":
                above_one,
            "neighbour_values_below_zero":
                below_zero,
            "max_neighbour_above_one":
                max_overshoot,
            "exported_neighbour_coverage_clipped_for_rounding_only":
                True,
            "original_source_files_modified":
                False,
            "importance_criteria_modified":
                False,
        },

        "criterion_distributions_ready_cells": {
            name: describe(values[ready])
            for name, values in selected.items()
        },

        "pairwise_spearman":
            correlations,

        "checks":
            checks,

        "alternative_indicators_for_sensitivity": [
            (
                "Confirmed mapped green share instead "
                "of observed green fraction"
            ),
            (
                "Historical mean green fraction "
                "instead of dominant-year share"
            ),
            (
                "500m green neighbourhood "
                "instead of 250m"
            ),
            (
                "Exploratory patch size, noting "
                "largest-component concentration"
            ),
        ],

        "limitations": [
            (
                "Candidate criteria are not approved "
                "final weights or an importance score."
            ),
            (
                "Detected current green is not a validated "
                "ecological habitat designation."
            ),
            (
                "G2 is saturated at one for many cells."
            ),
            (
                "250m neighbourhood context is not "
                "verified ecological connectivity."
            ),
            (
                "80 percent coverage flags are "
                "diagnostics, not exclusion rules."
            ),
            (
                "No-current-green and unknown-current-green "
                "cells are retained unscored."
            ),
        ],

        "historical_loss_used_in_importance":
            False,

        "future_pressure_used_in_importance":
            False,

        "development_suitability_used_in_importance":
            False,
    }

    QA_OUT.write_text(
        json.dumps(qa, indent=2),
        encoding="utf-8",
    )

    print("CSV:", CSV_OUT)
    print("GPKG:", GPKG_OUT)
    print("QA:", QA_OUT)

    print("\n" + "=" * 76)
    print("STAGE 4.8:", final_status)
    print(
        "Standardized ready green cells:",
        int(ready.sum()),
    )
    print("Final weights: NOT ASSIGNED")
    print("Green Importance: NOT CALCULATED")
    print("Future pressure used: NO")
    print("=" * 76)

    if final_status != (
        "PASS_STANDARDIZED_CANDIDATES_WEIGHTING_PENDING"
    ):
        raise SystemExit(
            "Stage 4.8 export verification failed"
        )


if __name__ == "__main__":
    main()
