
"""
J26-DS-338 | Section 04 | Stage 4.13

DOMAIN-AWARE GREEN IMPORTANCE AND THREAT EVIDENCE INTEGRATION

Combines validated evidence into ONE joined table, without
creating a composite score or Conservation Priority.

Sources:
  17j Frozen independent Green Importance
  17k Historical and future threat evidence

Rules:
- Preserve exactly 9,240 KMC grid cells.
- Keep frozen GI scores and CRITIC weights unchanged.
- Keep historical ROBUST/STRICT exposure unchanged.
- Preserve frozen 2026-2028 urbanization pressure scores.
- Explicitly distinguish NOT_IN_DOMAIN from low pressure.
- Never impute future pressure for missing-domain cells.
- No new weights, thresholds, risk classes or priorities.
- Correlations are descriptive, not causal or predictive.

Outputs:
  17l_green_importance_threat_integration_100m.csv
  17l_green_importance_threat_integration_100m.gpkg
  17l_green_importance_threat_integration_qa.json
"""

from pathlib import Path
import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data/processed"
CI = DATA / "conservation_intelligence"

REPORT = (
    ROOT / "outputs/reports/conservation_intelligence"
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

THREAT_CSV = (
    CI / "threat_evidence/"
    "17k_historical_future_threat_evidence_100m.csv"
)

THREAT_QA = (
    REPORT / "17k_historical_future_threat_evidence_qa.json"
)

HISTORY_SOURCE = (
    CI / "historical_threat/"
    "17e_historical_green_loss_exposure_100m.csv"
)

PRESSURE_SOURCE = (
    DATA / "urbanization_pressure/inference/"
    "14c_2026_2028_urbanization_pressure_scores.csv"
)

ELIGIBILITY_SOURCE = (
    DATA / "urbanization_pressure/inference/"
    "14b_2025_inference_eligibility.csv"
)

OUT_DIR = CI / "integrated_evidence"

CSV_OUT = (
    OUT_DIR /
    "17l_green_importance_threat_integration_100m.csv"
)

GPKG_OUT = (
    OUT_DIR /
    "17l_green_importance_threat_integration_100m.gpkg"
)

QA_OUT = (
    REPORT /
    "17l_green_importance_threat_integration_qa.json"
)

GI_FIELD = "green_importance_index_0_1"

HISTORY_FIELDS = [
    "historical_g2b_robust_exposure_share_250m",
    "historical_g2b_robust_exposure_share_500m",
    "historical_g2b_strict_exposure_share_250m",
    "historical_g2b_strict_exposure_share_500m",
]

PRESSURE_FIELD = "future_urbanization_pressure_score"

TOL = 1e-10


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def require_columns(frame, columns, name):

    missing = sorted(
        set(columns) - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{name}: missing columns {missing}"
        )


def load_unique(path, name, expected):

    frame = pd.read_csv(path)

    if (
        "grid_id" not in frame.columns
        or len(frame) != expected
        or frame.grid_id.isna().any()
        or not frame.grid_id.is_unique
    ):
        raise ValueError(
            f"{name}: expected {expected:,} "
            "unique grid IDs"
        )

    return frame


def numbers(frame, field):

    return pd.to_numeric(
        frame[field],
        errors="coerce",
    ).to_numpy(dtype=float)


def distribution(values):

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {"count": 0}

    return {
        "count": int(x.size),
        "min": float(x.min()),
        "p25": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "mean": float(x.mean()),
        "p75": float(np.quantile(x, 0.75)),
        "max": float(x.max()),
    }


def main():

    print("=" * 78)
    print(
        "SECTION 04 | STAGE 4.13 - "
        "DOMAIN-AWARE GI AND THREAT INTEGRATION"
    )
    print("=" * 78)

    required = [
        GI_CSV,
        GI_GPKG,
        GI_MANIFEST,
        THREAT_CSV,
        THREAT_QA,
        HISTORY_SOURCE,
        PRESSURE_SOURCE,
        ELIGIBILITY_SOURCE,
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    checks = {}

    def check(name, condition):

        passed = bool(condition)
        checks[name] = passed

        print(
            f"[{'PASS' if passed else 'FAIL'}] {name}"
        )

    def gate(phase):

        if not all(checks.values()):
            raise SystemExit(
                f"{phase}: QA failed. "
                "No integration output exported."
            )

    # ======================================================
    # 1. VERIFY FROZEN GI AND THREAT EVIDENCE
    # ======================================================

    print("\n[1] VERIFY SOURCE INTEGRITY")

    manifest = json.loads(
        GI_MANIFEST.read_text(encoding="utf-8")
    )

    threat_qa = json.loads(
        THREAT_QA.read_text(encoding="utf-8")
    )

    check(
        "stage_411_frozen_gi_passed",
        manifest.get("status")
        == (
            "PASS_GREEN_IMPORTANCE_FROZEN_BASELINE_"
            "SENSITIVITY_PENDING"
        ),
    )

    check(
        "stage_412_threat_evidence_passed",
        threat_qa.get("status")
        == (
            "PASS_SEPARATE_THREAT_EVIDENCE_"
            "INTEGRATION_PENDING"
        ),
    )

    for kind, path in (
        ("csv", GI_CSV),
        ("gpkg", GI_GPKG),
    ):

        expected = (
            manifest.get("frozen_outputs", {})
            .get(kind, {})
            .get("sha256")
        )

        check(
            f"frozen_gi_{kind}_sha256_valid",
            isinstance(expected, str)
            and sha256(path) == expected,
        )

    source_hashes = {
        "historical_source_csv_sha256":
            HISTORY_SOURCE,

        "future_pressure_source_csv_sha256":
            PRESSURE_SOURCE,

        "pressure_eligibility_source_csv_sha256":
            ELIGIBILITY_SOURCE,
    }

    for name, path in source_hashes.items():

        check(
            f"{name}_unchanged",
            sha256(path) == threat_qa.get(name),
        )

    gate("Source fingerprint")

    # ======================================================
    # 2. LOAD AND VALIDATE GRID IDENTITY
    # ======================================================

    print("\n[2] VERIFY GRID ALIGNMENT")

    gi = load_unique(
        GI_CSV,
        "Frozen GI",
        9240,
    )

    threat = load_unique(
        THREAT_CSV,
        "Threat evidence",
        9240,
    )

    history_source = load_unique(
        HISTORY_SOURCE,
        "Original historical exposure",
        9240,
    )

    pressure_source = load_unique(
        PRESSURE_SOURCE,
        "Original pressure scores",
        4620,
    )

    eligibility = load_unique(
        ELIGIBILITY_SOURCE,
        "Pressure eligibility",
        9240,
    )

    check(
        "threat_grid_ids_match_gi",
        set(threat.grid_id) == set(gi.grid_id),
    )

    check(
        "history_grid_ids_match_gi",
        set(history_source.grid_id)
        == set(gi.grid_id),
    )

    check(
        "eligibility_grid_ids_match_gi",
        set(eligibility.grid_id)
        == set(gi.grid_id),
    )

    check(
        "pressure_grid_ids_subset",
        set(pressure_source.grid_id).issubset(
            set(gi.grid_id)
        ),
    )

    gate("Grid identity")

    ids = gi.grid_id

    threat = (
        threat.set_index("grid_id")
        .loc[ids]
        .reset_index()
    )

    history_source = (
        history_source.set_index("grid_id")
        .loc[ids]
        .reset_index()
    )

    eligibility = (
        eligibility.set_index("grid_id")
        .loc[ids]
        .reset_index()
    )

    require_columns(
        gi,
        [
            "green_importance_status",
            "green_importance_frozen",
            GI_FIELD,
            "G1_critic_weight",
            "G2_critic_weight",
            "G3_critic_weight",
        ],
        "Frozen GI",
    )

    require_columns(
        threat,
        [
            "frozen_gi_status",
            "frozen_gi_available",
            "historical_threat_status",
            "future_pressure_domain_status",
            "gi_pressure_overlap_status",
            PRESSURE_FIELD,
            "future_threshold_flag",
            "future_threshold_status",
            "future_operating_threshold",
            "future_score_interpretation",
            *HISTORY_FIELDS,
        ],
        "Threat evidence",
    )

    # ======================================================
    # 3. VERIFY HISTORICAL THREAT WAS PRESERVED
    # ======================================================

    print("\n[3] VERIFY HISTORICAL EVIDENCE")

    for field in HISTORY_FIELDS:

        current = numbers(threat, field)
        original = numbers(history_source, field)

        check(
            f"{field}_unchanged",
            np.isfinite(current).all()
            and np.all(
                (current >= 0)
                & (current <= 1)
            )
            and np.allclose(
                current,
                original,
                atol=TOL,
                rtol=0,
            ),
        )

    check(
        "historical_context_valid_all_cells",
        threat[
            "historical_threat_status"
        ].eq(
            "VALID_RETROSPECTIVE_RASTER_CONTEXT"
        ).all(),
    )

    gate("Historical evidence")

    # ======================================================
    # 4. VERIFY GREEN IMPORTANCE AND PRESSURE DOMAIN
    # ======================================================

    print("\n[4] VERIFY AVAILABILITY AND PRESSURE DOMAIN")

    gi_score = numbers(
        gi, GI_FIELD
    )

    gi_ready = (
        gi.green_importance_status
        .eq("GREEN_IMPORTANCE_CALCULATED")
        .to_numpy()
    )

    no_green = (
        gi.green_importance_status
        .eq("NO_CURRENT_GREEN_DETECTED")
        .to_numpy()
    )

    unknown = (
        gi.green_importance_status
        .eq("UNKNOWN_CURRENT_GREEN_CONDITION")
        .to_numpy()
    )

    model_domain = (
        threat.future_pressure_domain_status
        .eq("SCORED_MODEL_DOMAIN")
        .to_numpy()
    )

    no_domain = (
        threat.future_pressure_domain_status
        .eq("NOT_IN_DOMAIN")
        .to_numpy()
    )

    future_score = numbers(
        threat,
        PRESSURE_FIELD,
    )

    check(
        "gi_8727_468_45_partition",
        int(gi_ready.sum()) == 8727
        and int(no_green.sum()) == 468
        and int(unknown.sum()) == 45
        and np.all(
            gi_ready | no_green | unknown
        ),
    )

    check(
        "gi_score_present_only_when_calculated",
        np.isfinite(
            gi_score[gi_ready]
        ).all()
        and np.isnan(
            gi_score[~gi_ready]
        ).all(),
    )

    check(
        "gi_frozen_flag_true",
        gi.green_importance_frozen
        .astype(str)
        .str.lower()
        .eq("true")
        .all(),
    )

    check(
        "threat_gi_status_matches_frozen",
        np.array_equal(
            gi.green_importance_status
            .astype(str)
            .to_numpy(),
            threat.frozen_gi_status
            .astype(str)
            .to_numpy(),
        ),
    )

    check(
        "threat_gi_availability_matches_frozen",
        np.array_equal(
            threat.frozen_gi_available
            .astype(str)
            .str.lower()
            .eq("true")
            .to_numpy(),
            gi_ready,
        ),
    )

    check(
        "pressure_4620_model_and_4620_outside",
        int(model_domain.sum()) == 4620
        and int(no_domain.sum()) == 4620
        and np.all(
            model_domain | no_domain
        ),
    )

    check(
        "pressure_only_present_in_model_domain",
        np.isfinite(
            future_score[model_domain]
        ).all()
        and np.isnan(
            future_score[no_domain]
        ).all(),
    )

    ready_flags = (
        eligibility.final_inference_ready
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
        .to_numpy()
    )

    check(
        "pressure_domain_matches_original_eligibility",
        np.array_equal(
            model_domain,
            ready_flags,
        ),
    )

    # Cross-check original model predictions.
    original_pressure = (
        pressure_source
        .set_index("grid_id")
        .loc[ids[model_domain]]
    )

    check(
        "future_pressure_scores_unchanged",
        np.allclose(
            future_score[model_domain],
            pd.to_numeric(
                original_pressure[
                    "urbanization_pressure_score"
                ],
                errors="coerce",
            ).to_numpy(dtype=float),
            atol=TOL,
            rtol=0,
        ),
    )

    check(
        "pressure_relative_not_probability",
        threat.loc[
            model_domain,
            "future_score_interpretation",
        ].eq(
            "MODEL_DERIVED_PRESSURE_SCORE_NOT_CALIBRATED_PROBABILITY"
        ).all(),
    )

    flags = numbers(
        threat,
        "future_threshold_flag",
    )

    thresholds = numbers(
        threat,
        "future_operating_threshold",
    )

    check(
        "locked_pressure_threshold_preserved",
        np.allclose(
            thresholds[model_domain],
            0.715,
            atol=1e-12,
            rtol=0,
        )
        and np.isnan(
            thresholds[no_domain]
        ).all(),
    )

    check(
        "frozen_threshold_flags_preserved",
        np.isin(
            flags[model_domain],
            [0, 1],
        ).all()
        and int(
            np.nansum(flags)
        ) == 255
        and np.isnan(
            flags[no_domain]
        ).all(),
    )

    gate("Availability and pressure integrity")

    # ======================================================
    # 5. DEFINE FOUR INTEGRATION POPULATIONS
    # ======================================================

    print("\n[5] CLASSIFY INTEGRATION COVERAGE")

    both = (
        gi_ready & model_domain
    )

    gi_only = (
        gi_ready & no_domain
    )

    pressure_only = (
        ~gi_ready & model_domain
    )

    neither = (
        ~gi_ready & no_domain
    )

    categories = np.select(
        [
            both,
            gi_only,
            pressure_only,
            neither,
        ],
        [
            "GI_AND_FUTURE_PRESSURE_AVAILABLE",
            "GI_HISTORICAL_ONLY_PRESSURE_NOT_IN_DOMAIN",
            "FUTURE_PRESSURE_ONLY_GI_UNSCORED",
            "GI_UNSCORED_PRESSURE_NOT_IN_DOMAIN",
        ],
        default="INVALID_STATUS",
    )

    expected_original = np.select(
        [
            both,
            gi_only,
            pressure_only,
            neither,
        ],
        [
            "GI_AND_PRESSURE_AVAILABLE",
            "GI_ONLY_PRESSURE_NOT_IN_DOMAIN",
            "PRESSURE_ONLY_GI_UNSCORED",
            "NEITHER_GI_NOR_PRESSURE",
        ],
        default="INVALID_OVERLAP",
    )

    check(
        "overlap_4609_both",
        int(both.sum()) == 4609,
    )

    check(
        "overlap_4118_gi_without_pressure",
        int(gi_only.sum()) == 4118,
    )

    check(
        "overlap_11_pressure_without_gi",
        int(pressure_only.sum()) == 11,
    )

    check(
        "overlap_502_neither",
        int(neither.sum()) == 502,
    )

    check(
        "original_overlap_status_reproduced",
        np.array_equal(
            threat.gi_pressure_overlap_status
            .astype(str).to_numpy(),
            expected_original,
        ),
    )

    check(
        "no_invalid_integration_status",
        not np.any(
            categories == "INVALID_STATUS"
        ),
    )

    gate("Integration populations")

    print(
        "GI + future pressure:",
        int(both.sum()),
    )

    print(
        "GI + history; pressure NOT_IN_DOMAIN:",
        int(gi_only.sum()),
    )

    print(
        "Pressure present; GI unscored:",
        int(pressure_only.sum()),
    )

    print(
        "GI unscored; pressure NOT_IN_DOMAIN:",
        int(neither.sum()),
    )

    # ======================================================
    # 6. JOIN FROZEN GI WITH THE SEPARATE THREAT EVIDENCE
    # ======================================================

    print("\n[6] BUILD INTEGRATED EVIDENCE TABLE")

    result = gi.merge(
        threat,
        on="grid_id",
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    result[
        "integration_evidence_status"
    ] = categories

    result[
        "future_informed_assessment_available"
    ] = both

    result[
        "historical_evidence_available"
    ] = True

    # These booleans report data availability,
    # NOT Conservation Priority classes.

    check(
        "all_9240_cells_preserved",
        len(result) == 9240
        and result.grid_id.is_unique
        and result.grid_id.notna().all(),
    )

    check(
        "frozen_gi_scores_unchanged",
        np.allclose(
            numbers(result, GI_FIELD),
            gi_score,
            atol=0,
            rtol=0,
            equal_nan=True,
        ),
    )

    check(
        "future_pressure_scores_unchanged_after_join",
        np.allclose(
            numbers(result, PRESSURE_FIELD),
            future_score,
            atol=0,
            rtol=0,
            equal_nan=True,
        ),
    )

    check(
        "future_assessment_available_only_on_overlap",
        np.array_equal(
            result[
                "future_informed_assessment_available"
            ].to_numpy(dtype=bool),
            both,
        ),
    )

    check(
        "no_new_composite_or_priority_field",
        not any(
            term in column.lower()
            for column in result.columns
            for term in (
                "combined_threat_score",
                "conservation_priority_score",
                "conservation_priority_level",
                "new_risk_class",
            )
        ),
    )

    gate("Integrated table")

    # ======================================================
    # 7. DESCRIPTIVE RELATIONSHIPS IN COMMON DOMAIN
    # ======================================================

    print("\n[7] DESCRIPTIVE COMMON-DOMAIN DIAGNOSTICS")

    robust_250 = numbers(
        result,
        "historical_g2b_robust_exposure_share_250m",
    )

    pairs = {
        "GI_vs_Historical_ROBUST_250m":
            (gi_score, robust_250),

        "GI_vs_Future_Pressure":
            (gi_score, future_score),

        "Historical_ROBUST_250m_vs_Future_Pressure":
            (robust_250, future_score),
    }

    correlations = {}

    for name, (left, right) in pairs.items():

        x = left[both]
        y = right[both]

        pearson = float(
            np.corrcoef(x, y)[0, 1]
        )

        spearman = float(
            spearmanr(x, y).statistic
        )

        check(
            f"{name}_correlations_finite",
            np.isfinite(pearson)
            and np.isfinite(spearman),
        )

        correlations[name] = {
            "paired_cells": int(len(x)),
            "pearson_r": pearson,
            "spearman_rho": spearman,
        }

        print(
            f"{name}: "
            f"Pearson={pearson:.4f}, "
            f"Spearman={spearman:.4f}"
        )

    print("\nGI distributions by pressure availability:")

    print(
        "  With future pressure:",
        distribution(gi_score[both]),
    )

    print(
        "  Pressure NOT_IN_DOMAIN:",
        distribution(gi_score[gi_only]),
    )

    check(
        "descriptive_statistics_do_not_modify_sources",
        len(correlations) == 3,
    )

    gate("Diagnostics")

    # ======================================================
    # 8. PREPARE GIS OUTPUT
    # ======================================================

    print("\n[8] PREPARE GIS INTEGRATION LAYER")

    geo = gpd.read_file(
        GI_GPKG
    )

    check(
        "frozen_gis_9240_unique_ids",
        len(geo) == 9240
        and geo.grid_id.is_unique
        and set(geo.grid_id) == set(ids),
    )

    check(
        "frozen_gis_epsg32644",
        geo.crs is not None
        and geo.crs.to_epsg() == 32644
        and geo.geometry.notna().all()
        and geo.geometry.is_valid.all(),
    )

    gate("GIS source")

    geo_output = (
        geo[["grid_id", "geometry"]]
        .merge(
            result,
            on="grid_id",
            how="inner",
            validate="one_to_one",
        )
    )

    check(
        "integrated_gis_9240",
        len(geo_output) == 9240
        and geo_output.grid_id.is_unique,
    )

    check(
        "integrated_gis_kmc_areas_match",
        np.max(
            np.abs(
                geo_output.geometry.area.to_numpy()
                - numbers(
                    geo_output,
                    "kmc_overlap_area_m2",
                )
            )
        ) <= 1.0,
    )

    gate("GIS join")

    # ======================================================
    # 9. EXPORT
    # ======================================================

    print("\n[9] EXPORT INTEGRATED EVIDENCE")

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

    geo_output.to_file(
        GPKG_OUT,
        layer="gi_historical_future_evidence_100m",
        driver="GPKG",
    )

    saved = pd.read_csv(CSV_OUT)
    info = pyogrio.read_info(
        GPKG_OUT
    )

    check(
        "csv_export_9240",
        len(saved) == 9240
        and saved.grid_id.is_unique,
    )

    check(
        "gpkg_export_9240",
        info["features"] == 9240,
    )

    check(
        "saved_gi_8727_pressure_4620",
        saved[GI_FIELD].notna().sum() == 8727
        and saved[PRESSURE_FIELD].notna().sum() == 4620,
    )

    check(
        "saved_4118_gi_cells_without_future_pressure",
        (
            saved.integration_evidence_status
            == "GI_HISTORICAL_ONLY_PRESSURE_NOT_IN_DOMAIN"
        ).sum() == 4118,
    )

    check(
        "frozen_gi_files_remain_unchanged",
        sha256(GI_CSV)
        == manifest["frozen_outputs"]["csv"]["sha256"]
        and sha256(GI_GPKG)
        == manifest["frozen_outputs"]["gpkg"]["sha256"],
    )

    gate("Export integrity")

    # ======================================================
    # 10. WRITE QA REPORT
    # ======================================================

    print("\n[10] WRITE QA AND METHODOLOGY REPORT")

    qa = {
        "section": "04_conservation_intelligence",
        "stage": "4.13",

        "status": (
            "PASS_DOMAIN_AWARE_INTEGRATION_"
            "PRIORITY_PENDING"
        ),

        "grid_cells": 9240,

        "frozen_gi_available": 8727,

        "future_pressure_available": 4620,

        "historical_context_available": 9240,

        "overlap": {
            "gi_and_future_pressure": int(both.sum()),
            "gi_only_future_not_in_domain": int(gi_only.sum()),
            "future_pressure_only_gi_unscored": int(pressure_only.sum()),
            "neither_gi_nor_future_pressure": int(neither.sum()),
        },

        "full_future_informed_assessment_domain": (
            "4609 cells with both frozen Green Importance "
            "and model-derived future pressure"
        ),

        "partial_evidence_policy": (
            "The 4118 currently green cells outside the "
            "future pressure inference domain retain "
            "Green Importance and historical threat, "
            "but receive no inferred future pressure."
        ),

        "future_pressure_interpretation": (
            "MODEL_DERIVED_PRESSURE_SCORE_NOT_CALIBRATED_PROBABILITY"
        ),

        "historical_primary_indicator": (
            "historical_g2b_robust_exposure_share_250m"
        ),

        "historical_sensitivity_indicators": [
            "historical_g2b_robust_exposure_share_500m",
            "historical_g2b_strict_exposure_share_250m",
            "historical_g2b_strict_exposure_share_500m",
        ],

        "frozen_operating_threshold": 0.715,

        "descriptive_correlations_common_domain": correlations,

        "gi_distribution_future_model_domain": (
            distribution(gi_score[both])
        ),

        "gi_distribution_future_not_in_domain": (
            distribution(gi_score[gi_only])
        ),

        "historical_robust_250m_distribution_both": (
            distribution(robust_250[both])
        ),

        "future_pressure_distribution_both": (
            distribution(future_score[both])
        ),

        "threshold_flagged_in_gi_pressure_overlap": int(
            np.sum(flags[both] == 1)
        ),

        "source_sha256": {
            "frozen_gi_csv": sha256(GI_CSV),
            "frozen_gi_gpkg": sha256(GI_GPKG),
            "stage_412_threat_csv": sha256(THREAT_CSV),
            "historical_source_csv": sha256(HISTORY_SOURCE),
            "pressure_source_csv": sha256(PRESSURE_SOURCE),
        },

        "output_sha256": {
            "integrated_csv": sha256(CSV_OUT),
            "integrated_gpkg": sha256(GPKG_OUT),
        },

        "checks": checks,

        "frozen_gi_modified": False,
        "new_critiс_weights_calculated": False,
        "new_threat_weights_calculated": False,
        "composite_score_created": False,
        "conservation_priority_created": False,

        "next_stage": (
            "4.14 Conservation Priority methodology "
            "and computation, subject to explicit "
            "domain and aggregation decisions"
        ),

        "limitations": [
            "Historical exposure and future pressure are "
            "different constructs and are not averaged.",
            "Future pressure predictions are only available "
            "within the frozen inference domain.",
            "Missing pressure must never imply low risk.",
            "Correlations are descriptive and may reflect "
            "spatial autocorrelation.",
            "No field-validated ecological conservation "
            "priority is claimed.",
            "Sensitivity and robustness analysis remain pending.",
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
    print(
        "STAGE 4.13: "
        "PASS_DOMAIN_AWARE_INTEGRATION_PRIORITY_PENDING"
    )
    print("Frozen Green Importance modified: NO")
    print("Missing future pressure imputed: NO")
    print("Conservation Priority created: NO")
    print("=" * 78)


if __name__ == "__main__":
    main()
