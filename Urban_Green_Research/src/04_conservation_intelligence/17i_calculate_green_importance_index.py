
"""
J26-DS-338 | Urban Green Research
SECTION 04 - CONSERVATION INTELLIGENCE

STAGE 4.10: CALCULATE INDEPENDENT GREEN IMPORTANCE INDEX

INPUTS:
1. Stage 4.8 standardized G1/G2/G3 criteria
2. Stage 4.9 CRITIC objective weights
3. Stage 4.8 clipped KMC grid geometry

FORMULA:
GI_i = w1*G1_i + w2*G2_i + w3*G3_i

IMPORTANT:
- Read exact, full-precision CRITIC weights from CSV.
- Verify input SHA256 against the Stage 4.9 audit.
- Independently reproduce CRITIC weights before scoring.
- Score only current-green cells with complete criteria.
- Preserve all 9,240 KMC grid IDs.
- Do not treat unknown or non-green cells as GI = 0.
- Do not introduce historical loss, pressure, suitability,
  SHAP, AHP, or provisional zoning.
- Do not create priority categories or freeze the index.
- Report individual weighted contributions for transparency.

OUTPUTS:
  17i_green_importance_index_100m.csv
  17i_green_importance_index_100m.gpkg
  17i_green_importance_index_qa.json
"""

from pathlib import Path
import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from scipy.stats import spearmanr


# ==========================================================
# 0. PROJECT CONFIGURATION
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data/processed/conservation_intelligence"
)

REPORT_DIR = (
    ROOT / "outputs/reports/conservation_intelligence"
)

CRITERIA_CSV = (
    DATA_DIR / "importance_criteria/"
    "17g_green_importance_criteria_standardized_100m.csv"
)

CRITERIA_GPKG = (
    DATA_DIR / "importance_criteria/"
    "17g_green_importance_criteria_standardized_100m.gpkg"
)

CRITERIA_QA = (
    REPORT_DIR / "17g_green_importance_criteria_qa.json"
)

WEIGHTS_CSV = (
    REPORT_DIR / "17h_critic_weights.csv"
)

WEIGHTS_QA = (
    REPORT_DIR / "17h_critic_weighting_qa.json"
)

OUTPUT_DIR = (
    DATA_DIR / "importance_index"
)

CSV_OUT = (
    OUTPUT_DIR / "17i_green_importance_index_100m.csv"
)

GPKG_OUT = (
    OUTPUT_DIR / "17i_green_importance_index_100m.gpkg"
)

QA_OUT = (
    REPORT_DIR / "17i_green_importance_index_qa.json"
)

CRITERIA = {
    "G1": "G1_current_green_0_1",
    "G2": "G2_historical_persistence_0_1",
    "G3": "G3_local_green_context_0_1",
}

CODES = list(CRITERIA)
FIELDS = list(CRITERIA.values())

TOL = 1e-9


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def file_sha256(path):

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def required_columns(frame, columns, source):

    missing = sorted(
        set(columns) - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{source}: missing required columns: {missing}"
        )


def numeric(frame, column):

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    ).to_numpy(dtype=float)


def describe(values):

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {"count": 0}

    return {
        "count": int(len(x)),
        "minimum": float(x.min()),
        "p05": float(np.quantile(x, 0.05)),
        "p25": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "mean": float(x.mean()),
        "p75": float(np.quantile(x, 0.75)),
        "p95": float(np.quantile(x, 0.95)),
        "maximum": float(x.max()),
        "standard_deviation": float(
            x.std(ddof=0)
        ),
    }


def main():

    print("=" * 78)
    print(
        "SECTION 04 | STAGE 4.10 - "
        "INDEPENDENT GREEN IMPORTANCE INDEX"
    )
    print("=" * 78)

    # ======================================================
    # 1. CHECK INPUT FILES AND PREVIOUS STAGES
    # ======================================================

    print("\n[1] VERIFY PREVIOUS STAGES")

    for path in (
        CRITERIA_CSV,
        CRITERIA_GPKG,
        CRITERIA_QA,
        WEIGHTS_CSV,
        WEIGHTS_QA,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    checks = {}

    def check(name, condition):

        passed = bool(condition)
        checks[name] = passed

        print(
            f"[{'PASS' if passed else 'FAIL'}] {name}"
        )

    def require_pass(phase):

        if not all(checks.values()):
            raise SystemExit(
                f"{phase}: validation failed. "
                "No index outputs exported."
            )

    criteria_audit = json.loads(
        CRITERIA_QA.read_text(
            encoding="utf-8"
        )
    )

    weights_audit = json.loads(
        WEIGHTS_QA.read_text(
            encoding="utf-8"
        )
    )

    check(
        "stage_48_passed",
        criteria_audit.get("status")
        == "PASS_STANDARDIZED_CANDIDATES_WEIGHTING_PENDING",
    )

    check(
        "stage_49_passed",
        weights_audit.get("status")
        == "PASS_CRITIC_WEIGHTING_INDEX_PENDING",
    )

    check(
        "weighting_method_is_CRITIC",
        weights_audit.get("method")
        == "CRITIC_OBJECTIVE_WEIGHTING",
    )

    check(
        "stage_48_has_no_importance_index",
        criteria_audit.get(
            "green_importance_calculated"
        ) is False,
    )

    check(
        "stage_49_has_no_importance_index",
        weights_audit.get(
            "green_importance_index_calculated"
        ) is False,
    )

    check(
        "stage_49_not_frozen",
        weights_audit.get(
            "green_importance_frozen"
        ) is False,
    )

    require_pass("Upstream stage QA")

    # ======================================================
    # 2. LOAD AND VERIFY STANDARDIZED CRITERIA
    # ======================================================

    print("\n[2] LOAD STANDARDIZED CRITERIA")

    df = pd.read_csv(
        CRITERIA_CSV
    )

    needed = [
        "grid_id",
        "source_current_year",
        "source_current_product",
        "criterion_availability_status",
        "current_green_condition_status",
        "kmc_overlap_area_m2",
        "current_mapped_green_area_m2",
        "current_lulc_coverage_fraction",
        "historical_valid_year_count",
        "local_green_250m_valid_coverage_fraction",
        *FIELDS,
    ]

    required_columns(
        df,
        needed,
        "Stage 4.8 criteria",
    )

    check(
        "9240_unique_grid_ids",
        len(df) == 9240
        and df.grid_id.notna().all()
        and df.grid_id.is_unique,
    )

    check(
        "source_is_2025_raw_rf",
        df.source_current_product.eq(
            "RAW_RF"
        ).all()
        and pd.to_numeric(
            df.source_current_year,
            errors="coerce",
        ).eq(2025).all(),
    )

    # Protect against changes to the criteria
    # after CRITIC weights were generated.

    current_source_hash = file_sha256(
        CRITERIA_CSV
    )

    expected_source_hash = weights_audit.get(
        "input_csv_sha256"
    )

    check(
        "criteria_sha256_matches_weighting_input",
        isinstance(expected_source_hash, str)
        and current_source_hash == expected_source_hash,
    )

    status = (
        df["criterion_availability_status"]
        .astype(str)
    )

    ready = status.eq(
        "READY_FOR_WEIGHTING"
    ).to_numpy(dtype=bool)

    no_green = status.eq(
        "NO_CURRENT_GREEN_DETECTED"
    ).to_numpy(dtype=bool)

    unknown = status.eq(
        "UNKNOWN_2025_GREEN_CONDITION"
    ).to_numpy(dtype=bool)

    check(
        "expected_grid_population_partition",
        int(ready.sum()) == 8727
        and int(no_green.sum()) == 468
        and int(unknown.sum()) == 45
        and np.all(
            ready | no_green | unknown
        ),
    )

    raw_values = (
        df[FIELDS]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype=float)
    )

    X = raw_values[ready]

    check(
        "8727_complete_three_criterion_records",
        X.shape == (8727, 3)
        and np.isfinite(X).all(),
    )

    check(
        "all_three_criteria_within_0_1",
        np.all(
            (X >= -TOL)
            & (X <= 1.0 + TOL)
        ),
    )

    check(
        "nonready_criteria_remain_missing",
        np.isnan(
            raw_values[~ready]
        ).all(),
    )

    check(
        "green_condition_status_consistent",
        np.all(
            df.loc[
                ready,
                "current_green_condition_status",
            ] == "GREEN_DETECTED_IN_VALID_AREA"
        )
        and np.all(
            df.loc[
                no_green,
                "current_green_condition_status",
            ] == "NO_GREEN_DETECTED_IN_VALID_AREA"
        )
        and np.all(
            df.loc[
                unknown,
                "current_green_condition_status",
            ] == "UNKNOWN_NO_VALID_LULC"
        ),
    )

    require_pass("Criterion integrity")

    print("Total grid cells:", len(df))
    print("Complete green cells:", len(X))
    print("No-current-green cells:", int(no_green.sum()))
    print("Unknown-current-green cells:", int(unknown.sum()))

    # ======================================================
    # 3. LOAD EXACT CRITIC WEIGHTS
    # ======================================================

    print("\n[3] LOAD AND VALIDATE CRITIC WEIGHTS")

    weights_df = pd.read_csv(
        WEIGHTS_CSV
    )

    required_columns(
        weights_df,
        [
            "criterion",
            "source_field",
            "direction",
            "critic_weight",
            "information_content",
        ],
        "Stage 4.9 weights",
    )

    check(
        "three_unique_critic_weight_records",
        len(weights_df) == 3
        and weights_df.criterion.is_unique
        and set(weights_df.criterion) == set(CODES),
    )

    require_pass("Weight record identity")

    weights_df = (
        weights_df
        .set_index("criterion")
        .loc[CODES]
    )

    weights = pd.to_numeric(
        weights_df["critic_weight"],
        errors="coerce",
    ).to_numpy(dtype=float)

    check(
        "weights_correspond_to_correct_criteria",
        all(
            weights_df.loc[
                code,
                "source_field",
            ] == CRITERIA[code]
            for code in CODES
        ),
    )

    check(
        "all_weight_directions_are_benefit",
        weights_df.direction.eq(
            "BENEFIT"
        ).all(),
    )

    check(
        "weights_nonnegative_and_sum_to_one",
        np.isfinite(weights).all()
        and (weights >= 0).all()
        and np.isclose(
            weights.sum(),
            1.0,
            atol=1e-12,
        ),
    )

    # Compare the CSV weights against those
    # recorded in the original CRITIC QA JSON.

    qa_weight_records = {
        item["criterion"]: item
        for item in weights_audit.get(
            "criteria", []
        )
    }

    check(
        "weights_match_original_qa_report",
        set(qa_weight_records) == set(CODES)
        and all(
            qa_weight_records[code]["source_field"]
            == CRITERIA[code]
            for code in CODES
        )
        and all(
            np.isclose(
                weights[i],
                float(
                    qa_weight_records[code][
                        "critic_weight"
                    ]
                ),
                atol=1e-12,
            )
            for i, code in enumerate(CODES)
        ),
    )

    check(
        "weighting_population_matches_current_data",
        weights_audit.get(
            "weighting_observations"
        ) == int(ready.sum()),
    )

    require_pass("Weight integrity")

    for code, weight in zip(CODES, weights):
        print(
            f"{code} CRITIC weight: {weight:.10f}"
        )

    # ======================================================
    # 4. INDEPENDENTLY REPRODUCE CRITIC WEIGHTS
    # ======================================================

    print("\n[4] REPRODUCE CRITIC WEIGHT CALCULATION")

    # Standard CRITIC using:
    # population SD and Pearson correlation.

    sd = X.std(
        axis=0,
        ddof=0,
    )

    pearson = np.corrcoef(
        X,
        rowvar=False,
    )

    check(
        "critic_recalculation_inputs_valid",
        np.isfinite(sd).all()
        and (sd > 1e-12).all()
        and np.isfinite(pearson).all(),
    )

    require_pass("CRITIC reproduction inputs")

    pearson = np.clip(
        pearson,
        -1.0,
        1.0,
    )

    np.fill_diagonal(
        pearson,
        1.0,
    )

    conflict = (
        1.0 - pearson
    ).sum(axis=1)

    information = sd * conflict

    check(
        "critic_information_reproducible",
        np.isfinite(
            information
        ).all()
        and (
            information >= 0
        ).all()
        and information.sum() > 1e-12,
    )

    require_pass("CRITIC information")

    reproduced_weights = (
        information / information.sum()
    )

    check(
        "critic_weights_reproduced_exactly",
        np.allclose(
            reproduced_weights,
            weights,
            atol=1e-10,
            rtol=0.0,
        ),
    )

    print(
        "Maximum CRITIC weight reproduction difference:",
        float(
            np.max(
                np.abs(
                    reproduced_weights - weights
                )
            )
        ),
    )

    require_pass("CRITIC reproduction")

    # ======================================================
    # 5. CALCULATE GREEN IMPORTANCE INDEX
    # ======================================================

    print("\n[5] CALCULATE GREEN IMPORTANCE INDEX")

    # For each currently green cell:
    #
    # GI = G1*w1 + G2*w2 + G3*w3
    #
    # Individual contributions are retained.
    # These contributions are arithmetic terms,
    # not causal explanations or SHAP values.

    contributions = (
        X * weights[np.newaxis, :]
    )

    gi_values = (
        contributions.sum(axis=1)
    )

    n = len(df)

    full_index = np.full(
        n,
        np.nan,
        dtype=float,
    )

    full_contributions = np.full(
        (n, 3),
        np.nan,
        dtype=float,
    )

    full_index[ready] = gi_values

    full_contributions[ready] = (
        contributions
    )

    # Dominant weighted contribution is explanatory
    # arithmetic, not a causal driver.

    dominant_contribution = np.full(
        n,
        None,
        dtype=object,
    )

    dominant_contribution[ready] = (
        np.asarray(CODES, dtype=object)[
            np.argmax(
                contributions,
                axis=1,
            )
        ]
    )

    result_status = np.full(
        n,
        "INCOMPLETE_CRITERIA",
        dtype=object,
    )

    result_status[no_green] = (
        "NO_CURRENT_GREEN_DETECTED"
    )

    result_status[unknown] = (
        "UNKNOWN_CURRENT_GREEN_CONDITION"
    )

    result_status[ready] = (
        "GREEN_IMPORTANCE_CALCULATED"
    )

    check(
        "gi_values_finite_and_bounded",
        np.isfinite(
            gi_values
        ).all()
        and (gi_values >= -TOL).all()
        and (gi_values <= 1.0 + TOL).all(),
    )

    check(
        "all_contributions_nonnegative",
        np.isfinite(
            contributions
        ).all()
        and (
            contributions >= -TOL
        ).all(),
    )

    check(
        "index_reproduced_from_contributions",
        np.allclose(
            gi_values,
            contributions.sum(axis=1),
            atol=1e-12,
        ),
    )

    check(
        "nonready_index_remains_missing",
        np.isnan(
            full_index[~ready]
        ).all(),
    )

    check(
        "nonready_contributions_remain_missing",
        np.isnan(
            full_contributions[~ready]
        ).all(),
    )

    check(
        "exactly_8727_index_values",
        int(
            np.isfinite(
                full_index
            ).sum()
        ) == 8727,
    )

    require_pass("Index computation")

    # ======================================================
    # 6. BUILD GREEN IMPORTANCE DATASET
    # ======================================================

    print("\n[6] BUILD GREEN IMPORTANCE DATASET")

    result = df[
        [
            "grid_id",
            "source_current_year",
            "source_current_product",
            "current_green_condition_status",
            "kmc_overlap_area_m2",
            "current_mapped_green_area_m2",
            "current_lulc_coverage_fraction",
            "historical_valid_year_count",
            "local_green_250m_valid_coverage_fraction",
            "qa_current_coverage_below_80pct",
            "qa_neighbour_coverage_below_80pct",
            "qa_history_fewer_than_14_valid_years",
            *FIELDS,
        ]
    ].copy()

    result["green_importance_status"] = (
        result_status
    )

    result["green_importance_index_0_1"] = (
        full_index
    )

    for j, code in enumerate(CODES):

        result[
            f"{code}_critic_weight"
        ] = float(weights[j])

        result[
            f"{code}_weighted_contribution"
        ] = full_contributions[:, j]

    result[
        "largest_weighted_contribution"
    ] = dominant_contribution

    result["green_importance_method"] = (
        "CRITIC_WEIGHTED_ADDITIVE"
    )

    result["green_importance_frozen"] = False

    # Do not create High/Medium/Low importance
    # classifications or priority labels here.

    # ======================================================
    # 7. FINAL INDEX QA AND DISTRIBUTIONS
    # ======================================================

    print("\n[7] VALIDATE INDEX RESULTS")

    check(
        "all_9240_grid_ids_preserved",
        len(result) == 9240
        and result.grid_id.notna().all()
        and result.grid_id.is_unique,
    )

    check(
        "8727_cells_calculated",
        int(
            (
                result.green_importance_status
                == "GREEN_IMPORTANCE_CALCULATED"
            ).sum()
        ) == 8727,
    )

    check(
        "468_non_green_cells_unscored",
        int(
            (
                result.green_importance_status
                == "NO_CURRENT_GREEN_DETECTED"
            ).sum()
        ) == 468,
    )

    check(
        "45_unknown_cells_unscored",
        int(
            (
                result.green_importance_status
                == "UNKNOWN_CURRENT_GREEN_CONDITION"
            ).sum()
        ) == 45,
    )

    check(
        "index_has_no_artificial_zero_for_unknown",
        np.isnan(
            full_index[unknown]
        ).all(),
    )

    check(
        "dominant_contribution_valid",
        all(
            code in CODES
            for code in dominant_contribution[ready]
        )
        and all(
            value is None
            for value in dominant_contribution[~ready]
        ),
    )

    forbidden = [
        "pressure",
        "shap",
        "suitability",
        "zoning",
        "historical_g2b",
        "historical_loss",
        "priority",
    ]

    check(
        "no_historical_or_future_threat_inputs",
        not any(
            any(
                token in column.lower()
                for token in forbidden
            )
            for column in result.columns
        ),
    )

    check(
        "no_frozen_index_declared",
        not result[
            "green_importance_frozen"
        ].any(),
    )

    require_pass("Final index QA")

    print("\nGreen Importance Index distribution:")
    print(describe(gi_values))

    dominant_counts = {
        code: int(
            (
                dominant_contribution[ready]
                == code
            ).sum()
        )
        for code in CODES
    }

    print("\nLargest weighted contribution by criterion:")

    for code, count in dominant_counts.items():

        print(
            f"{code}: {count:,} cells "
            f"({100 * count / len(gi_values):.2f}%)"
        )

    correlations = {}

    print("\nSpearman association with index:")

    for j, code in enumerate(CODES):

        rho = float(
            spearmanr(
                X[:, j],
                gi_values,
            ).statistic
        )

        check(
            f"index_spearman_{code}_finite",
            np.isfinite(rho),
        )

        correlations[code] = rho

        print(
            f"{code} vs index: rho={rho:.6f}"
        )

    require_pass("Index diagnostics")

    # ======================================================
    # 8. VERIFY GIS JOIN
    # ======================================================

    print("\n[8] VERIFY GIS OUTPUT GEOMETRY")

    geo_source = gpd.read_file(
        CRITERIA_GPKG
    )

    check(
        "source_gis_9240_unique_ids",
        len(geo_source) == 9240
        and geo_source.grid_id.notna().all()
        and geo_source.grid_id.is_unique
        and set(geo_source.grid_id)
        == set(result.grid_id),
    )

    check(
        "gis_projected_crs_and_geometry_valid",
        geo_source.crs is not None
        and geo_source.crs.to_epsg() == 32644
        and geo_source.geometry.notna().all()
        and geo_source.geometry.is_valid.all(),
    )

    require_pass("GIS source QA")

    geo = (
        geo_source[
            ["grid_id", "geometry"]
        ]
        .merge(
            result,
            on="grid_id",
            how="inner",
            validate="one_to_one",
        )
    )

    check(
        "gis_join_9240",
        len(geo) == 9240
        and geo.grid_id.is_unique,
    )

    source_areas = (
        result.set_index("grid_id")
        .loc[geo.grid_id, "kmc_overlap_area_m2"]
        .to_numpy(dtype=float)
    )

    geometry_area_difference = float(
        np.max(
            np.abs(
                geo.geometry.area.to_numpy(dtype=float)
                - source_areas
            )
        )
    )

    check(
        "clipped_gis_areas_match",
        geometry_area_difference <= 1.0,
    )

    require_pass("GIS geometry QA")

    # ======================================================
    # 9. EXPORT INDEX CSV AND GEOPACKAGE
    # ======================================================

    print("\n[9] EXPORT GREEN IMPORTANCE INDEX")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        CSV_OUT,
        index=False,
    )

    geo.to_file(
        GPKG_OUT,
        layer="green_importance_index_100m",
        driver="GPKG",
    )

    saved_csv = pd.read_csv(
        CSV_OUT
    )

    saved_gis = pyogrio.read_info(
        GPKG_OUT
    )

    check(
        "csv_export_9240",
        len(saved_csv) == 9240
        and saved_csv.grid_id.is_unique,
    )

    check(
        "gpkg_export_9240",
        saved_gis["features"] == 9240,
    )

    check(
        "csv_index_values_reproduced",
        np.allclose(
            pd.to_numeric(
                saved_csv[
                    "green_importance_index_0_1"
                ],
                errors="coerce",
            ).to_numpy(dtype=float),
            full_index,
            atol=1e-12,
            equal_nan=True,
        ),
    )

    check(
        "csv_513_unscored_cells_preserved",
        int(
            saved_csv[
                "green_importance_index_0_1"
            ].isna().sum()
        ) == 513,
    )

    require_pass("Export verification")

    # ======================================================
    # 10. WRITE REPRODUCIBLE QA REPORT
    # ======================================================

    print("\n[10] SAVE INDEX QA REPORT")

    final_status = (
        "PASS_GREEN_IMPORTANCE_INDEX_NOT_FROZEN"
    )

    qa = {
        "section":
            "04_conservation_intelligence",

        "stage":
            "4.10",

        "status":
            final_status,

        "index_name":
            "Independent Green Importance Index",

        "weighting_method":
            "CRITIC_OBJECTIVE_WEIGHTING",

        "aggregation_method":
            "WEIGHTED_ADDITIVE_SUM",

        "formula":
            "GI = w1*G1 + w2*G2 + w3*G3",

        "weights": {
            code: float(weights[j])
            for j, code in enumerate(CODES)
        },

        "criteria": CRITERIA,

        "source_criteria_sha256":
            current_source_hash,

        "source_weights_sha256":
            file_sha256(WEIGHTS_CSV),

        "grid_cells":
            len(result),

        "index_calculated_cells":
            int(ready.sum()),

        "no_current_green_cells":
            int(no_green.sum()),

        "unknown_current_green_cells":
            int(unknown.sum()),

        "index_distribution":
            describe(gi_values),

        "largest_weighted_contribution_counts":
            dominant_counts,

        "index_criterion_spearman":
            correlations,

        "maximum_gis_area_difference_m2":
            geometry_area_difference,

        "checks":
            checks,

        "index_interpretation": (
            "A relative composite indicator of "
            "current green condition, historical "
            "green persistence and surrounding "
            "green context among currently green "
            "KMC 100m grid cells. It is not a "
            "probability or an independently "
            "validated ecological-quality measure."
        ),

        "unscored_cell_handling": {
            "no_current_green_detected":
                "INDEX_NAN",

            "unknown_current_green_condition":
                "INDEX_NAN",

            "not_interpreted_as_zero_importance":
                True,
        },

        "explanations": (
            "Weighted component contributions "
            "are arithmetic explanations only, "
            "not SHAP or causal drivers"
        ),

        "green_importance_frozen":
            False,

        "historical_loss_used":
            False,

        "future_pressure_used":
            False,

        "development_suitability_used":
            False,

        "development_suitability_ahp_used":
            False,

        "conservation_priority_calculated":
            False,

        "next_stage":
            "4.11 Green Importance verification and freeze",

        "sensitivity_pending": [
            "Equal-domain weighting baseline",
            "CRITIC weight perturbations",
            "Area-weighted CRITIC comparison",
            "Historical mean green fraction alternative",
            "500m green neighbourhood alternative",
            "Current green coverage uncertainty",
            "Green patch concentration effects",
        ],

        "limitations": [
            (
                "The index reflects three selected "
                "statistical indicators, not a direct "
                "field-based ecological survey."
            ),
            (
                "CRITIC gives substantial weight "
                "to historical persistence, which "
                "has many saturated values."
            ),
            (
                "The index is calculated only for "
                "cells with detected current green."
            ),
            (
                "No importance-class thresholds "
                "or conservation priorities have "
                "been selected."
            ),
            (
                "Historical loss and future "
                "urbanization pressure have not "
                "entered the index."
            ),
            (
                "Sensitivity analysis is required "
                "before final conservation claims."
            ),
        ],
    }

    QA_OUT.write_text(
        json.dumps(
            qa,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("CSV:", CSV_OUT)
    print("GPKG:", GPKG_OUT)
    print("QA:", QA_OUT)

    print("\n" + "=" * 78)
    print("STAGE 4.10:", final_status)
    print("Green Importance calculated:", int(ready.sum()))
    print("Unscored cells:", int((~ready).sum()))
    print("Index frozen: NO")
    print("Historical/future threat used: NO")
    print("Conservation Priority calculated: NO")
    print("=" * 78)


if __name__ == "__main__":
    main()
