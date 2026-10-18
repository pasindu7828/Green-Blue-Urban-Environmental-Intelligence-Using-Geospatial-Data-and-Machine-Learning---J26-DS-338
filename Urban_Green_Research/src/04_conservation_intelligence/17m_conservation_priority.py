
"""
J26-DS-338 | Section 04 | Stage 4.14.

Primary Conservation Priority (CP) = frozen GI * frozen 2026-2028 UP.

UP is a relative model-derived pressure score, not a calibrated probability.
Historical Green-to-Built exposure is supporting evidence ONLY.

All 9,240 cells are retained.
Only the 4,609 common-domain cells are scored.

No new weights, classes, imputations, retraining, or changes to frozen sources.
Sensitivity analyses and ecological validation remain pending.
"""

from pathlib import Path
import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio


ROOT = Path(__file__).resolve().parents[2]

CI = ROOT / "data/processed/conservation_intelligence"

REPORT = ROOT / "outputs/reports/conservation_intelligence"

PRESSURE_DIR = (
    ROOT / "data/processed/urbanization_pressure/inference"
)

GI_CSV = (
    CI / "frozen_importance/"
    "17j_frozen_green_importance_100m.csv"
)

GI_GPKG = (
    CI / "frozen_importance/"
    "17j_frozen_green_importance_100m.gpkg"
)

GI_MANIFEST = (
    REPORT / "17j_green_importance_freeze_manifest.json"
)

INTEGRATED_CSV = (
    CI / "integrated_evidence/"
    "17l_green_importance_threat_integration_100m.csv"
)

INTEGRATED_GPKG = (
    CI / "integrated_evidence/"
    "17l_green_importance_threat_integration_100m.gpkg"
)

INTEGRATED_QA = (
    REPORT / "17l_green_importance_threat_integration_qa.json"
)

HISTORY_SOURCE = (
    CI / "historical_threat/"
    "17e_historical_green_loss_exposure_100m.csv"
)

PRESSURE_SOURCE = (
    PRESSURE_DIR /
    "14c_2026_2028_urbanization_pressure_scores.csv"
)

OUT_DIR = CI / "conservation_priority"

OUT_CSV = (
    OUT_DIR / "17m_primary_conservation_priority_100m.csv"
)

OUT_GPKG = (
    OUT_DIR / "17m_primary_conservation_priority_100m.gpkg"
)

OUT_QA = (
    REPORT / "17m_primary_conservation_priority_qa.json"
)

GI = "green_importance_index_0_1"

UP = "future_urbanization_pressure_score"

CP = "primary_conservation_priority_0_1"

HISTORICAL = [
    "historical_g2b_robust_exposure_share_250m",
    "historical_g2b_robust_exposure_share_500m",
    "historical_g2b_strict_exposure_share_250m",
    "historical_g2b_strict_exposure_share_500m",
    "historical_g2b_robust_nearest_pixel_centre_m",
]


def file_hash(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def read_unique(path, expected):

    frame = pd.read_csv(path)

    if (
        "grid_id" not in frame
        or len(frame) != expected
        or frame.grid_id.isna().any()
        or not frame.grid_id.is_unique
    ):
        raise ValueError(
            f"{path.name}: expected "
            f"{expected:,} unique grid_id values"
        )

    return frame.set_index("grid_id")


def numeric(frame, column):

    if column not in frame.columns:
        raise ValueError(
            f"Required column missing: {column}"
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    ).to_numpy(dtype=float)


def describe(x):

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if not x.size:
        return {"count": 0}

    return {
        "count": int(x.size),
        "min": float(np.min(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p25": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "p75": float(np.quantile(x, 0.75)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
    }


def main():

    print("=" * 78)
    print(
        "SECTION 04 | STAGE 4.14 - "
        "MULTIPLICATIVE CONSERVATION PRIORITY"
    )
    print("=" * 78)

    required = [
        GI_CSV,
        GI_GPKG,
        GI_MANIFEST,
        INTEGRATED_CSV,
        INTEGRATED_GPKG,
        INTEGRATED_QA,
        HISTORY_SOURCE,
        PRESSURE_SOURCE,
    ]

    for path in required:

        if not path.is_file():
            raise FileNotFoundError(path)

    checks = {}

    def check(name, passed):

        checks[name] = bool(passed)

        print(
            f"[{'PASS' if checks[name] else 'FAIL'}] "
            f"{name}"
        )

    def gate(label):

        if not all(checks.values()):
            raise SystemExit(
                f"{label}: QA FAILED; "
                "export not performed"
            )

    # ==================================================
    # 1. VERIFY FROZEN INPUTS
    # ==================================================

    print(
        "\n[1] VERIFY FROZEN INPUTS "
        "AND STAGE 4.13 FINGERPRINTS"
    )

    manifest = json.loads(
        GI_MANIFEST.read_text(encoding="utf-8")
    )

    audit = json.loads(
        INTEGRATED_QA.read_text(encoding="utf-8")
    )

    check(
        "stage_411_frozen",
        manifest.get("status")
        == (
            "PASS_GREEN_IMPORTANCE_FROZEN_BASELINE_"
            "SENSITIVITY_PENDING"
        ),
    )

    check(
        "stage_413_passed",
        audit.get("status")
        == (
            "PASS_DOMAIN_AWARE_INTEGRATION_"
            "PRIORITY_PENDING"
        ),
    )

    for kind, path in (
        ("csv", GI_CSV),
        ("gpkg", GI_GPKG),
    ):

        record = (
            manifest
            .get("frozen_outputs", {})
            .get(kind, {})
        )

        check(
            f"frozen_GI_{kind}_hash_matches",
            file_hash(path) == record.get("sha256"),
        )

    for label, path in (
        ("integrated_csv", INTEGRATED_CSV),
        ("integrated_gpkg", INTEGRATED_GPKG),
    ):

        check(
            f"stage_413_{label}_hash_matches",
            file_hash(path)
            == audit.get(
                "output_sha256", {}
            ).get(label),
        )

    for label, path in (
        ("historical_source_csv", HISTORY_SOURCE),
        ("pressure_source_csv", PRESSURE_SOURCE),
    ):

        check(
            f"{label}_unchanged",
            file_hash(path)
            == audit.get(
                "source_sha256", {}
            ).get(label),
        )

    gate("Input fingerprints")

    # ==================================================
    # 2. VERIFY THE COMMON DOMAIN
    # ==================================================

    print(
        "\n[2] VERIFY CELL IDENTITY "
        "AND 4,609-CELL COMMON DOMAIN"
    )

    gi = read_unique(GI_CSV, 9240)

    integrated = read_unique(
        INTEGRATED_CSV, 9240
    )

    history = read_unique(
        HISTORY_SOURCE, 9240
    )

    pressure = read_unique(
        PRESSURE_SOURCE, 4620
    )

    ids = gi.index

    check(
        "all_integrated_grid_ids_match",
        set(ids) == set(integrated.index),
    )

    check(
        "all_historical_grid_ids_match",
        set(ids) == set(history.index),
    )

    check(
        "pressure_grid_ids_subset",
        set(pressure.index).issubset(set(ids)),
    )

    gate("Grid identity")

    integrated = integrated.loc[ids]

    history = history.loc[ids]

    gi_status = (
        gi["green_importance_status"]
        .astype(str)
        .to_numpy()
    )

    available_gi = (
        gi_status == "GREEN_IMPORTANCE_CALCULATED"
    )

    no_green = (
        gi_status == "NO_CURRENT_GREEN_DETECTED"
    )

    unknown = (
        gi_status == "UNKNOWN_CURRENT_GREEN_CONDITION"
    )

    domain_status = (
        integrated["future_pressure_domain_status"]
        .astype(str)
        .to_numpy()
    )

    in_domain = (
        domain_status == "SCORED_MODEL_DOMAIN"
    )

    out_domain = (
        domain_status == "NOT_IN_DOMAIN"
    )

    both = available_gi & in_domain

    gi_only = available_gi & out_domain

    up_only = (~available_gi) & in_domain

    neither = (~available_gi) & out_domain

    gi_values = numeric(integrated, GI)

    up_values = numeric(integrated, UP)

    check(
        "GI_status_population_8727_468_45",
        (
            int(available_gi.sum()),
            int(no_green.sum()),
            int(unknown.sum()),
        ) == (8727, 468, 45)
        and np.all(
            available_gi | no_green | unknown
        ),
    )

    check(
        "GI_and_pressure_population_4609_4118_11_502",
        (
            int(both.sum()),
            int(gi_only.sum()),
            int(up_only.sum()),
            int(neither.sum()),
        ) == (4609, 4118, 11, 502),
    )

    check(
        "pressure_domain_4620",
        in_domain.sum() == 4620
        and out_domain.sum() == 4620
        and np.all(in_domain | out_domain),
    )

    check(
        "GI_values_present_only_in_GI_domain",
        np.isfinite(
            gi_values[available_gi]
        ).all()
        and np.isnan(
            gi_values[~available_gi]
        ).all()
        and np.all(
            (gi_values[available_gi] >= 0)
            & (gi_values[available_gi] <= 1)
        ),
    )

    check(
        "UP_values_present_only_in_model_domain",
        np.isfinite(
            up_values[in_domain]
        ).all()
        and np.isnan(
            up_values[out_domain]
        ).all()
        and np.all(
            (up_values[in_domain] >= 0)
            & (up_values[in_domain] <= 1)
        ),
    )

    check(
        "frozen_GI_values_preserved_in_integration",
        np.allclose(
            gi_values,
            numeric(gi, GI),
            atol=0,
            rtol=0,
            equal_nan=True,
        ),
    )

    check(
        "source_status_and_model_domain_consistent",
        np.array_equal(
            in_domain,
            ids.isin(pressure.index),
        )
        and np.array_equal(
            integrated[
                "green_importance_status"
            ].astype(str).to_numpy(),
            gi_status,
        ),
    )

    original_up = numeric(
        pressure.loc[ids[in_domain]],
        "urbanization_pressure_score",
    )

    check(
        "frozen_future_pressure_scores_preserved",
        np.allclose(
            up_values[in_domain],
            original_up,
            atol=1e-12,
            rtol=0,
        ),
    )

    check(
        "future_origin_and_interpretation_preserved",
        np.all(
            numeric(
                integrated,
                "future_pressure_origin_year",
            )[in_domain] == 2025
        )
        and integrated.loc[
            ids[in_domain],
            "future_pressure_period",
        ].eq("2026-2028").all()
        and integrated.loc[
            ids[in_domain],
            "future_score_interpretation",
        ].eq(
            "MODEL_DERIVED_PRESSURE_SCORE_"
            "NOT_CALIBRATED_PROBABILITY"
        ).all(),
    )

    check(
        "locked_pressure_threshold_not_modified",
        np.allclose(
            numeric(
                integrated,
                "future_operating_threshold",
            )[in_domain],
            0.715,
            atol=1e-12,
            rtol=0,
        )
        and np.isnan(
            numeric(
                integrated,
                "future_operating_threshold",
            )[out_domain]
        ).all(),
    )

    flags = numeric(
        integrated,
        "future_threshold_flag",
    )

    check(
        "pressure_flags_preserved_255_total_248_overlap",
        np.isin(
            flags[in_domain],
            [0, 1],
        ).all()
        and np.isnan(
            flags[out_domain]
        ).all()
        and int(
            flags[in_domain].sum()
        ) == 255
        and int(
            flags[both].sum()
        ) == 248
        and np.array_equal(
            (
                up_values[in_domain] >= 0.715
            ).astype(int),
            flags[in_domain].astype(int),
        ),
    )

    gate("Input-domain integrity")

    # ==================================================
    # 3. PRESERVE HISTORICAL SUPPORTING EVIDENCE
    # ==================================================

    print(
        "\n[3] VERIFY HISTORICAL EXPOSURE "
        "IS SUPPORTING EVIDENCE ONLY"
    )

    for field in HISTORICAL:

        a = numeric(integrated, field)

        b = numeric(history, field)

        check(
            f"supporting_{field}_unchanged",
            np.isfinite(a).all()
            and (a >= 0).all()
            and np.allclose(
                a,
                b,
                atol=1e-12,
                rtol=0,
            ),
        )

    gate("Historical evidence")

    # ==================================================
    # 4. CALCULATE PRIMARY CONSERVATION PRIORITY
    # ==================================================

    print(
        "\n[4] CALCULATE PRIMARY CP "
        "= FROZEN GI x FROZEN UP"
    )

    cp = np.full(
        len(ids),
        np.nan,
        dtype=float,
    )

    # APPROVED PRIMARY FORMULA:
    #
    # CP = GI * UP
    #
    # This is calculated ONLY within the domain
    # where both frozen inputs exist.
    #
    # Historical loss does not enter this formula.

    cp[both] = (
        gi_values[both]
        * up_values[both]
    )

    assessment_status = np.full(
        len(ids),
        "GI_UNSCORED_AND_PRESSURE_NOT_IN_DOMAIN",
        dtype=object,
    )

    assessment_status[
        both
    ] = "FUTURE_CONSERVATION_PRIORITY_CALCULATED"

    assessment_status[
        gi_only
    ] = "NOT_IN_PRESSURE_DOMAIN"

    assessment_status[
        up_only
    ] = "GI_UNSCORED_PRESSURE_AVAILABLE"

    # Diagnostic rank only within the 4,609 cells.
    #
    # Higher CP -> stronger rank.
    # Equal CP values share the minimum rank.
    #
    # Percentiles are also limited to the
    # 4,609-cell common domain.

    rank = np.full(
        len(ids),
        np.nan,
        dtype=float,
    )

    percentile = np.full(
        len(ids),
        np.nan,
        dtype=float,
    )

    values = pd.Series(
        cp[both]
    )

    rank[both] = values.rank(
        method="min",
        ascending=False,
    ).to_numpy(dtype=float)

    percentile[both] = (
        100
        * values.rank(
            method="average",
            pct=True,
        )
    ).to_numpy(dtype=float)

    check(
        "exactly_4609_finite_primary_scores",
        np.isfinite(
            cp[both]
        ).sum() == 4609
        and np.isnan(
            cp[~both]
        ).all(),
    )

    check(
        "primary_score_reproduces_multiplication",
        np.allclose(
            cp[both],
            gi_values[both] * up_values[both],
            atol=1e-12,
            rtol=0,
        ),
    )

    check(
        "primary_scores_bounded_0_to_1",
        np.all(
            (cp[both] >= 0)
            & (cp[both] <= 1)
        ),
    )

    check(
        "ranks_only_for_4609_common_domain_cells",
        np.isfinite(
            rank[both]
        ).all()
        and np.isnan(
            rank[~both]
        ).all()
        and (rank[both] >= 1).all()
        and (rank[both] <= 4609).all(),
    )

    check(
        "percentiles_only_for_common_domain",
        np.isfinite(
            percentile[both]
        ).all()
        and np.isnan(
            percentile[~both]
        ).all()
        and (percentile[both] > 0).all()
        and (percentile[both] <= 100).all(),
    )

    gate("Primary CP calculation")

    # ==================================================
    # 5. BUILD THE DOMAIN-AWARE OUTPUT
    # ==================================================

    print(
        "\n[5] BUILD DOMAIN-AWARE "
        "HISTORICALLY ANNOTATED OUTPUT"
    )

    keep = [
        "current_green_condition_status",
        "green_importance_status",
        "green_importance_frozen",
        "kmc_overlap_area_m2",
        "current_mapped_green_area_m2",
        "current_lulc_coverage_fraction",
        "G1_current_green_0_1",
        "G2_historical_persistence_0_1",
        "G3_local_green_context_0_1",
        GI,
        "G1_critic_weight",
        "G2_critic_weight",
        "G3_critic_weight",
        "G1_weighted_contribution",
        "G2_weighted_contribution",
        "G3_weighted_contribution",
        *HISTORICAL,
        "future_pressure_domain_status",
        "future_pressure_origin_year",
        "future_pressure_period",
        UP,
        "future_pressure_percentile",
        "future_operating_threshold",
        "future_threshold_flag",
        "future_threshold_status",
        "future_score_interpretation",
        "integration_evidence_status",
    ]

    missing = sorted(
        set(keep) - set(integrated.columns)
    )

    if missing:
        raise ValueError(
            f"Required integrated columns missing: "
            f"{missing}"
        )

    
    result = integrated[keep].copy().reset_index(drop=True)


    result.insert(
        0,
        "grid_id",
        ids.to_numpy(),
    )

    result[
        "primary_priority_status"
    ] = assessment_status

    result[CP] = cp

    result[
        "primary_priority_rank_common_domain"
    ] = rank

    result[
        "primary_priority_percentile_common_domain"
    ] = percentile

    result[
        "primary_priority_method"
    ] = "MULTIPLICATIVE_FROZEN_GI_X_FROZEN_PRESSURE"

    result[
        "historical_exposure_role"
    ] = "SUPPORTING_DIAGNOSTIC_NOT_IN_PRIMARY_CP"

    result[
        "priority_classes_assigned"
    ] = False

    check(
        "9240_unique_output_grid_cells",
        len(result) == 9240
        and result.grid_id.is_unique,
    )

    check(
        "status_counts_4609_4118_11_502",
        (
            sum(
                assessment_status
                == "FUTURE_CONSERVATION_PRIORITY_CALCULATED"
            ),
            sum(
                assessment_status
                == "NOT_IN_PRESSURE_DOMAIN"
            ),
            sum(
                assessment_status
                == "GI_UNSCORED_PRESSURE_AVAILABLE"
            ),
            sum(
                assessment_status
                == "GI_UNSCORED_AND_PRESSURE_NOT_IN_DOMAIN"
            ),
        ) == (4609, 4118, 11, 502),
    )

    check(
        "GI_still_identical_to_frozen_source",
        np.allclose(
            numeric(result, GI),
            numeric(gi, GI),
            rtol=0,
            atol=0,
            equal_nan=True,
        ),
    )

    check(
        "historic_not_in_formula_and_no_priority_classes",
        result[
            "historical_exposure_role"
        ].eq(
            "SUPPORTING_DIAGNOSTIC_NOT_IN_PRIMARY_CP"
        ).all()
        and not result[
            "priority_classes_assigned"
        ].any(),
    )

    gate("Output assembly")

    # ==================================================
    # 6. VALIDATE GIS GEOMETRY
    # ==================================================

    print(
        "\n[6] VERIFY CLIPPED KMC GIS GEOMETRY"
    )

    geo = gpd.read_file(
        INTEGRATED_GPKG
    )

    check(
        "gis_9240_unique_matching_grid_ids",
        len(geo) == 9240
        and geo.grid_id.is_unique
        and set(geo.grid_id) == set(ids),
    )

    check(
        "gis_crs_epsg32644_valid",
        geo.crs is not None
        and geo.crs.to_epsg() == 32644
        and geo.geometry.notna().all()
        and geo.geometry.is_valid.all(),
    )

    gate("GIS source")

    geo_out = geo[
        ["grid_id", "geometry"]
    ].merge(
        result,
        on="grid_id",
        validate="one_to_one",
        how="inner",
    )

    check(
        "gis_geometry_area_preserved",
        len(geo_out) == 9240
        and np.max(
            np.abs(
                geo_out.geometry.area.to_numpy(
                    dtype=float
                )
                - numeric(
                    geo_out,
                    "kmc_overlap_area_m2",
                )
            )
        ) <= 1.0,
    )

    gate("GIS output")

    # ==================================================
    # 7. EXPORT AND VERIFY
    # ==================================================

    print(
        "\n[7] EXPORT PRIMARY CP, GIS AND QA"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUT_CSV,
        index=False,
    )

    geo_out.to_file(
        OUT_GPKG,
        layer="primary_conservation_priority_100m",
        driver="GPKG",
    )

    saved = pd.read_csv(
        OUT_CSV
    )

    gpkg_info = pyogrio.read_info(
        OUT_GPKG
    )

    check(
        "csv_export_9240_unique",
        len(saved) == 9240
        and saved.grid_id.is_unique,
    )

    check(
        "gpkg_export_9240",
        gpkg_info["features"] == 9240,
    )

    check(
        "exported_CP_reproduced",
        np.allclose(
            numeric(saved, CP),
            cp,
            atol=1e-12,
            rtol=0,
            equal_nan=True,
        ),
    )

    check(
        "exported_4631_missing_CP_values",
        int(
            saved[CP].isna().sum()
        ) == 4631,
    )

    check(
        "frozen_GI_files_untouched",
        file_hash(GI_CSV)
        == manifest[
            "frozen_outputs"
        ]["csv"]["sha256"]
        and file_hash(GI_GPKG)
        == manifest[
            "frozen_outputs"
        ]["gpkg"]["sha256"],
    )

    check(
        "integrated_inputs_untouched",
        file_hash(INTEGRATED_CSV)
        == audit[
            "output_sha256"
        ]["integrated_csv"]
        and file_hash(INTEGRATED_GPKG)
        == audit[
            "output_sha256"
        ]["integrated_gpkg"],
    )

    gate("Export QA")

    # ==================================================
    # 8. CREATE METHODOLOGY AND QA REPORT
    # ==================================================

    qa = {
        "section":
            "04_conservation_intelligence",

        "stage":
            "4.14",

        "status": (
            "PASS_MULTIPLICATIVE_CONSERVATION_PRIORITY_"
            "SENSITIVITY_PENDING"
        ),

        "methodology_approved": (
            "Primary CP = frozen GI * "
            "frozen model-derived future pressure"
        ),

        "formula":
            "CP_i = GI_i * UP_i",

        "weighting_between_GI_and_UP":
            "NONE_ADDED",

        "future_pressure_is_calibrated_probability":
            False,

        "pressure_origin_year":
            2025,

        "pressure_period":
            "2026-2028",

        "operating_threshold_preserved":
            0.715,

        "threshold_is_not_priority_cutoff":
            True,

        "green_importance_frozen":
            True,

        "frozen_GI_weights_unchanged":
            True,

        "historical_loss_in_primary_score":
            False,

        "historical_loss_role":
            "SUPPORTING_DIAGNOSTIC_ONLY",

        "priority_categories_assigned":
            False,

        "total_cells":
            9240,

        "primary_priority_scored_cells":
            int(both.sum()),

        "status_counts": {
            "FUTURE_CONSERVATION_PRIORITY_CALCULATED":
                int(both.sum()),

            "NOT_IN_PRESSURE_DOMAIN":
                int(gi_only.sum()),

            "GI_UNSCORED_PRESSURE_AVAILABLE":
                int(up_only.sum()),

            "GI_UNSCORED_AND_PRESSURE_NOT_IN_DOMAIN":
                int(neither.sum()),
        },

        "locked_pressure_flags_in_common_domain":
            int(flags[both].sum()),

        "primary_cp_distribution":
            describe(cp[both]),

        "GI_distribution_common_domain":
            describe(gi_values[both]),

        "UP_distribution_common_domain":
            describe(up_values[both]),

        "rank_scope":
            "ONLY_4609_COMMON_DOMAIN_CELLS",

        "ranking_ties": (
            "MIN_RANK; percentile=average_percentile"
        ),

        "source_sha256": {
            "frozen_gi_csv":
                file_hash(GI_CSV),

            "frozen_gi_gpkg":
                file_hash(GI_GPKG),

            "integrated_csv":
                file_hash(INTEGRATED_CSV),

            "integrated_gpkg":
                file_hash(INTEGRATED_GPKG),

            "original_pressure_csv":
                file_hash(PRESSURE_SOURCE),
        },

        "output_sha256": {
            "csv":
                file_hash(OUT_CSV),

            "gpkg":
                file_hash(OUT_GPKG),
        },

        "checks":
            checks,

        "interpretation": (
            "Relative conservation urgency for "
            "existing green cells where both frozen "
            "Green Importance and frozen model-derived "
            "2026-2028 Urbanization Pressure are available. "
            "Not a probability of green loss, nor a "
            "field-validated ecological risk."
        ),

        "limitations": [
            (
                "The pressure model supports 4620 "
                "original domain cells, of which "
                "4609 have Green Importance."
            ),
            (
                "The 4118 GI-scored out-of-domain "
                "cells are not low priority; "
                "their future pressure is unknown."
            ),
            (
                "Historical Green-to-Built exposure "
                "is supporting evidence, not a "
                "direct GI input or CP term."
            ),
            (
                "A low product can result from low "
                "GI or low pressure; this is a chosen "
                "non-compensatory combination rule."
            ),
            (
                "No HIGH/MEDIUM/LOW priority "
                "thresholds have been selected."
            ),
            (
                "Stage 4.15 must test aggregation "
                "alternatives and spatial/rank sensitivity."
            ),
        ],
    }

    OUT_QA.write_text(
        json.dumps(
            qa,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ==================================================
    # 9. FINAL SUMMARY
    # ==================================================

    print(
        "\nPrimary CP distribution:",
        describe(cp[both]),
    )

    print(
        "Primary CP scored:",
        int(both.sum()),
    )

    print(
        "GI only - future pressure NOT_IN_DOMAIN:",
        int(gi_only.sum()),
    )

    print(
        "GI unscored, pressure available:",
        int(up_only.sum()),
    )

    print(
        "GI unscored, no pressure:",
        int(neither.sum()),
    )

    print("CSV:", OUT_CSV)
    print("GPKG:", OUT_GPKG)
    print("QA:", OUT_QA)

    print("\n" + "=" * 78)

    print(
        "STAGE 4.14: "
        "PASS_MULTIPLICATIVE_CONSERVATION_PRIORITY_"
        "SENSITIVITY_PENDING"
    )

    print(
        "Historical conversion used in CP formula: NO"
    )

    print(
        "Unknown pressure imputed: NO | "
        "Frozen inputs modified: NO"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
