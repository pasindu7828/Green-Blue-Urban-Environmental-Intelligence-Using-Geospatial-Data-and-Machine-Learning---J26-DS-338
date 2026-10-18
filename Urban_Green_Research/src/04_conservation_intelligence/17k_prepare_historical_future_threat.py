
"""
J26-DS-338 | Conservation Intelligence | Stage 4.12

Prepare historical Green-to-Built exposure and frozen future
urbanization pressure as SEPARATE threat evidence.

Historical:
- 2012-2014 to 2021-2023
- ROBUST primary; STRICT sensitivity
- 250m and 500m exposure

Future:
- Frozen 2025-origin predictions for 2026-2028
- 4,620 model-domain cells
- Relative pressure scores, NOT calibrated probabilities
- 0.715 operating threshold remains unchanged

All 9,240 KMC grid cells remain present.
Missing future predictions stay NaN / NOT_IN_DOMAIN.

This stage DOES NOT:
- modify Green Importance
- recalculate Urbanization Pressure
- create a combined threat score
- calculate Conservation Priority
- use Development Suitability AHP or SHAP
"""

from pathlib import Path
import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio


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

HISTORY_CSV = (
    CI / "historical_threat/"
    "17e_historical_green_loss_exposure_100m.csv"
)

HISTORY_QA = (
    REPORT / "17e_historical_green_loss_exposure_qa.json"
)

PRESSURE_CSV = (
    DATA / "urbanization_pressure/inference/"
    "14c_2026_2028_urbanization_pressure_scores.csv"
)

ELIGIBILITY_CSV = (
    DATA / "urbanization_pressure/inference/"
    "14b_2025_inference_eligibility.csv"
)

OUT_DIR = CI / "threat_evidence"

OUT_CSV = (
    OUT_DIR /
    "17k_historical_future_threat_evidence_100m.csv"
)

OUT_GPKG = (
    OUT_DIR /
    "17k_historical_future_threat_evidence_100m.gpkg"
)

OUT_QA = (
    REPORT /
    "17k_historical_future_threat_evidence_qa.json"
)

PRESSURE_INTERPRETATION = (
    "MODEL_DERIVED_PRESSURE_SCORE_NOT_CALIBRATED_PROBABILITY"
)

THRESHOLD = 0.715
TOL = 1e-9


def digest(path):

    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_unique(path, label, expected_count):

    frame = pd.read_csv(path)

    if (
        "grid_id" not in frame
        or len(frame) != expected_count
        or frame.grid_id.isna().any()
        or not frame.grid_id.is_unique
    ):
        raise ValueError(
            f"{label}: expected "
            f"{expected_count:,} unique grid IDs"
        )

    return frame.set_index(
        "grid_id",
        drop=True,
    )


def numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(dtype=float)


def summary(values):

    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[np.isfinite(x)]

    if not len(x):
        return {"count": 0}

    return {
        "count": int(len(x)),
        "min": float(x.min()),
        "p25": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "p75": float(np.quantile(x, 0.75)),
        "max": float(x.max()),
    }


def main():

    print("=" * 76)
    print(
        "SECTION 04 | STAGE 4.12 - "
        "SEPARATE HISTORICAL/FUTURE THREAT EVIDENCE"
    )
    print("=" * 76)

    sources = [
        GI_CSV,
        GI_GPKG,
        GI_MANIFEST,
        HISTORY_CSV,
        HISTORY_QA,
        PRESSURE_CSV,
        ELIGIBILITY_CSV,
    ]

    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)

    checks = {}

    def check(label, condition):

        passed = bool(condition)
        checks[label] = passed

        print(
            f"[{'PASS' if passed else 'FAIL'}] "
            f"{label}"
        )

    def gate(stage):

        if not all(checks.values()):
            raise SystemExit(
                f"{stage} QA failed. "
                "No threat outputs exported."
            )

    # ==================================================
    # 1. VERIFY FROZEN GREEN IMPORTANCE
    # ==================================================

    print("\n[1] VERIFY FROZEN GI INTEGRITY")

    manifest = json.loads(
        GI_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    history_qa = json.loads(
        HISTORY_QA.read_text(
            encoding="utf-8"
        )
    )

    check(
        "frozen_gi_manifest_passed",
        manifest.get("status")
        == (
            "PASS_GREEN_IMPORTANCE_FROZEN_BASELINE_"
            "SENSITIVITY_PENDING"
        ),
    )

    check(
        "historical_source_qa_passed",
        history_qa.get("status")
        == "PASS_HISTORICAL_LOSS_EXPOSURE",
    )

    for kind, path in (
        ("csv", GI_CSV),
        ("gpkg", GI_GPKG),
    ):

        record = (
            manifest.get(
                "frozen_outputs", {}
            ).get(kind, {})
        )

        check(
            f"frozen_gi_{kind}_sha256_valid",
            record.get("sha256")
            == digest(path)
            and record.get(
                "path", ""
            ).replace("\\", "/")
            == str(
                path.relative_to(ROOT)
            ).replace("\\", "/"),
        )

    gate("Frozen sources")

    gi = load_unique(
        GI_CSV,
        "frozen GI",
        9240,
    )

    history = load_unique(
        HISTORY_CSV,
        "historical threat",
        9240,
    )

    eligibility = load_unique(
        ELIGIBILITY_CSV,
        "pressure eligibility",
        9240,
    )

    pressure = load_unique(
        PRESSURE_CSV,
        "future pressure",
        4620,
    )

    ids = gi.index

    check(
        "historical_ids_match_gi",
        set(history.index) == set(ids),
    )

    check(
        "eligibility_ids_match_gi",
        set(eligibility.index) == set(ids),
    )

    check(
        "pressure_ids_subset_of_gi",
        set(pressure.index).issubset(
            set(ids)
        ),
    )

    gate("Grid identity")

    history = history.loc[ids]
    eligibility = eligibility.loc[ids]

    predicted = ids.isin(
        pressure.index
    )

    gi_status = (
        gi["green_importance_status"]
        .astype(str)
    )

    gi_available = gi_status.eq(
        "GREEN_IMPORTANCE_CALCULATED"
    ).to_numpy()

    non_green = gi_status.eq(
        "NO_CURRENT_GREEN_DETECTED"
    ).to_numpy()

    unknown = gi_status.eq(
        "UNKNOWN_CURRENT_GREEN_CONDITION"
    ).to_numpy()

    check(
        "frozen_gi_8727_468_45",
        gi_available.sum() == 8727
        and non_green.sum() == 468
        and unknown.sum() == 45
        and np.all(
            gi_available | non_green | unknown
        ),
    )

    check(
        "frozen_gi_boolean_flag_true",
        gi["green_importance_frozen"]
        .astype(str)
        .str.lower()
        .eq("true")
        .all(),
    )

    check(
        "frozen_gi_numeric_status_consistent",
        pd.to_numeric(
            gi["green_importance_index_0_1"],
            errors="coerce",
        )
        .notna()
        .to_numpy()
        .tolist()
        == gi_available.tolist(),
    )

    gate("GI status")

    # ==================================================
    # 2. VERIFY HISTORICAL GREEN LOSS
    # ==================================================

    print(
        "\n[2] VERIFY HISTORICAL "
        "GREEN-TO-BUILT EXPOSURE"
    )

    check(
        "historical_periods_fixed",
        history[
            "historical_loss_early_period"
        ].eq("2012-2014").all()
        and history[
            "historical_loss_late_period"
        ].eq("2021-2023").all(),
    )

    check(
        "historical_green_status_matches_gi",
        np.array_equal(
            history[
                "green_condition_status"
            ].astype(str).to_numpy(),
            gi[
                "current_green_condition_status"
            ].astype(str).to_numpy(),
        ),
    )

    check(
        "historical_kmc_areas_match_frozen",
        np.allclose(
            numeric(
                history["kmc_overlap_area_m2"]
            ),
            numeric(
                gi["kmc_overlap_area_m2"]
            ),
            atol=1.0,
            rtol=0,
        ),
    )

    history_fields = []

    for version in ("robust", "strict"):

        distance_field = (
            f"historical_g2b_{version}_"
            "nearest_pixel_centre_m"
        )

        history_fields.append(
            distance_field
        )

        for radius in (250, 500):

            prefix = (
                f"historical_g2b_{version}_"
            )

            valid_field = (
                f"{prefix}valid_pixels_{radius}m"
            )

            count_field = (
                f"{prefix}converted_pixels_{radius}m"
            )

            area_field = (
                f"{prefix}converted_area_km2_{radius}m"
            )

            share_field = (
                f"{prefix}exposure_share_{radius}m"
            )

            context_field = (
                f"{prefix}context_status_{radius}m"
            )

            history_fields.extend([
                valid_field,
                count_field,
                area_field,
                share_field,
                context_field,
            ])

            valid = numeric(
                history[valid_field]
            )

            converted = numeric(
                history[count_field]
            )

            area = numeric(
                history[area_field]
            )

            share = numeric(
                history[share_field]
            )

            context = (
                history[context_field]
                .astype(str)
            )

            check(
                f"history_{version}_{radius}m_valid",
                np.isfinite(valid).all()
                and (valid > 0).all()
                and np.isfinite(converted).all()
                and (converted >= 0).all()
                and (converted <= valid).all()
                and np.allclose(
                    converted,
                    np.round(converted),
                    atol=TOL,
                    rtol=0,
                )
                and np.isfinite(share).all()
                and (share >= 0).all()
                and (share <= 1).all()
                and np.allclose(
                    share,
                    converted / valid,
                    atol=TOL,
                    rtol=0,
                )
                and np.allclose(
                    area,
                    converted * 900 / 1e6,
                    atol=TOL,
                    rtol=0,
                )
                and context.eq(
                    "VALID_HISTORICAL_CONTEXT"
                ).all(),
            )

        distance = numeric(
            history[distance_field]
        )

        check(
            f"historical_{version}_distances_valid",
            np.isfinite(distance).all()
            and (distance >= 0).all(),
        )

    gate("Historical threat")

    # ==================================================
    # 3. VERIFY FROZEN FUTURE PRESSURE
    # ==================================================

    print(
        "\n[3] VERIFY FROZEN FUTURE PRESSURE "
        "DOMAIN AND METADATA"
    )

    flags_string = (
        eligibility["final_inference_ready"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    check(
        "eligibility_boolean_encoding_valid",
        flags_string.isin(
            ["true", "false", "1", "0"]
        ).all(),
    )

    ready_ids = set(
        eligibility.index[
            flags_string.isin(["true", "1"])
        ]
    )

    check(
        "pressure_ids_match_eligibility_exactly",
        ready_ids == set(pressure.index),
    )

    check(
        "2025_origin_2026_2028_period_consistent",
        pd.to_numeric(
            pressure["origin_year"],
            errors="coerce",
        ).eq(2025).all()
        and pressure[
            "pressure_period"
        ].eq("2026-2028").all()
        and pd.to_numeric(
            eligibility["origin_year"],
            errors="coerce",
        ).eq(2025).all()
        and eligibility[
            "pressure_period"
        ].eq("2026-2028").all(),
    )

    check(
        "relative_score_not_probability",
        pressure[
            "score_interpretation"
        ].eq(
            PRESSURE_INTERPRETATION
        ).all(),
    )

    raw_score = numeric(
        pressure["urbanization_pressure_score"]
    )

    pct = numeric(
        pressure["pressure_percentile"]
    )

    rank = numeric(
        pressure["pressure_rank"]
    )

    thresholds = numeric(
        pressure["operating_threshold"]
    )

    flags = numeric(
        pressure["threshold_flag"]
    )

    check(
        "pressure_scores_finite_bounded",
        np.isfinite(raw_score).all()
        and (raw_score >= 0).all()
        and (raw_score <= 1).all(),
    )

    check(
        "pressure_percentiles_bounded",
        np.isfinite(pct).all()
        and (pct >= 0).all()
        and (pct <= 100).all(),
    )

    check(
        "pressure_ranks_valid",
        np.isfinite(rank).all()
        and (rank >= 0).all()
        and (rank <= len(pressure)).all()
        and np.allclose(
            rank,
            np.round(rank),
            atol=TOL,
            rtol=0,
        ),
    )

    check(
        "operating_threshold_locked_0715",
        np.isfinite(thresholds).all()
        and np.allclose(
            thresholds,
            THRESHOLD,
            atol=1e-12,
            rtol=0,
        ),
    )

    check(
        "threshold_flags_0_1_valid",
        np.isfinite(flags).all()
        and np.isin(
            flags,
            [0, 1],
        ).all(),
    )

    check(
        "threshold_flag_matches_frozen_scores",
        np.array_equal(
            flags.astype(int),
            (
                raw_score >= THRESHOLD
            ).astype(int),
        ),
    )

    expected_labels = np.where(
        flags == 1,
        "FLAGGED_BY_LOCKED_THRESHOLD",
        "BELOW_LOCKED_THRESHOLD",
    )

    check(
        "locked_threshold_status_matches_flag",
        np.array_equal(
            pressure[
                "threshold_status"
            ].astype(str).to_numpy(),
            expected_labels,
        ),
    )

    check(
        "255_frozen_threshold_flags",
        int(flags.sum()) == 255,
    )

    gate("Future pressure")

    # ==================================================
    # 4. VERIFY SPATIAL DOMAIN OVERLAP
    # ==================================================

    print(
        "\n[4] VERIFY GI/FUTURE-PRESSURE OVERLAP"
    )

    both = (
        gi_available & predicted
    )

    gi_only = (
        gi_available & ~predicted
    )

    pressure_only = (
        ~gi_available & predicted
    )

    neither = (
        ~gi_available & ~predicted
    )

    check(
        "overlap_4609_both",
        int(both.sum()) == 4609,
    )

    check(
        "overlap_4118_gi_only",
        int(gi_only.sum()) == 4118,
    )

    check(
        "overlap_11_pressure_only",
        int(pressure_only.sum()) == 11,
    )

    check(
        "overlap_502_neither",
        int(neither.sum()) == 502,
    )

    check(
        "pressure_domain_4620_not_domain_4620",
        int(predicted.sum()) == 4620
        and int(
            (~predicted).sum()
        ) == 4620,
    )

    gate("Domain overlap")

    for title, value in (
        ("GI + future pressure", both.sum()),
        (
            "GI only; pressure NOT_IN_DOMAIN",
            gi_only.sum(),
        ),
        (
            "Pressure only; GI unscored",
            pressure_only.sum(),
        ),
        (
            "Neither score available",
            neither.sum(),
        ),
    ):
        print(
            f"  {title}: {value:,}"
        )

    # ==================================================
    # 5. BUILD SEPARATE THREAT EVIDENCE
    # ==================================================

    print(
        "\n[5] ASSEMBLE HISTORICAL AND FUTURE "
        "THREAT EVIDENCE"
    )

    # GI is used ONLY to describe coverage/overlap.
    # No Green Importance numeric index is copied
    # into this independent threat evidence table.

    result = pd.DataFrame({
        "grid_id":
            ids.to_numpy(),

        "frozen_gi_status":
            gi_status.to_numpy(),

        "frozen_gi_available":
            gi_available,

        "historical_threat_status":
            "VALID_RETROSPECTIVE_RASTER_CONTEXT",

        "historical_early_period":
            history[
                "historical_loss_early_period"
            ].to_numpy(),

        "historical_late_period":
            history[
                "historical_loss_late_period"
            ].to_numpy(),
    })

    for field in history_fields:

        result[field] = (
            history[field].to_numpy()
        )

    aligned_pressure = (
        pressure.reindex(ids)
    )

    fields_map = {
        "origin_year":
            "future_pressure_origin_year",

        "pressure_period":
            "future_pressure_period",

        "urbanization_pressure_score":
            "future_urbanization_pressure_score",

        "pressure_percentile":
            "future_pressure_percentile",

        "pressure_rank":
            "future_pressure_rank",

        "operating_threshold":
            "future_operating_threshold",

        "threshold_flag":
            "future_threshold_flag",

        "threshold_status":
            "future_threshold_status",

        "score_interpretation":
            "future_score_interpretation",
    }

    for source, output in fields_map.items():

        result[output] = (
            aligned_pressure[source].to_numpy()
        )

    result[
        "future_pressure_domain_status"
    ] = np.where(
        predicted,
        "SCORED_MODEL_DOMAIN",
        "NOT_IN_DOMAIN",
    )

    result[
        "gi_pressure_overlap_status"
    ] = np.select(
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

    pressure_number_fields = [
        "future_urbanization_pressure_score",
        "future_pressure_percentile",
        "future_pressure_rank",
        "future_operating_threshold",
        "future_threshold_flag",
        "future_pressure_origin_year",
    ]

    check(
        "all_9240_ids_preserved",
        len(result) == 9240
        and result.grid_id.is_unique,
    )

    check(
        "pressure_numerics_missing_only_outside_domain",
        all(
            pd.to_numeric(
                result[c],
                errors="coerce",
            ).notna().to_numpy().tolist()
            == predicted.tolist()
            for c in pressure_number_fields
        ),
    )

    check(
        "pressure_status_not_in_domain_explicit",
        (
            result[
                "future_pressure_domain_status"
            ].eq("NOT_IN_DOMAIN").to_numpy()
            == (~predicted)
        ).all(),
    )

    check(
        "pressure_missing_string_fields_outside_domain",
        all(
            result.loc[
                ~predicted, c
            ].isna().all()
            for c in (
                "future_pressure_period",
                "future_threshold_status",
                "future_score_interpretation",
            )
        ),
    )

    check(
        "pressure_score_preserved_exactly",
        np.array_equal(
            numeric(
                result.loc[
                    predicted,
                    "future_urbanization_pressure_score",
                ]
            ),
            numeric(
                aligned_pressure.loc[
                    predicted,
                    "urbanization_pressure_score",
                ]
            ),
        ),
    )

    check(
        "no_gi_index_or_composite_priority_created",
        not any(
            token in col.lower()
            for col in result.columns
            for token in (
                "green_importance_index",
                "critic_weight",
                "combined_threat",
                "conservation_priority",
                "development_suitability",
                "shap",
            )
        ),
    )

    check(
        "only_source_history_and_pressure_measures",
        set(history_fields).issubset(
            set(result.columns)
        )
        and len(history_fields)
        == len(set(history_fields)),
    )

    gate("Evidence assembly")

    # ==================================================
    # 6. VERIFY GIS GEOMETRY
    # ==================================================

    print("\n[6] CHECK GIS GEOMETRY")

    geometry = gpd.read_file(
        GI_GPKG
    )[
        ["grid_id", "geometry"]
    ]

    check(
        "frozen_gis_9240_unique_matching_ids",
        len(geometry) == 9240
        and geometry.grid_id.is_unique
        and set(geometry.grid_id)
        == set(result.grid_id),
    )

    check(
        "frozen_gis_crs_epsg32644",
        geometry.crs is not None
        and geometry.crs.to_epsg() == 32644
        and geometry.geometry.notna().all()
        and geometry.geometry.is_valid.all(),
    )

    gate("GIS reference")

    geo = geometry.merge(
        result,
        on="grid_id",
        how="inner",
        validate="one_to_one",
    )

    check(
        "geo_join_9240",
        len(geo) == 9240
        and geo.grid_id.is_unique,
    )

    gate("GIS join")

    # ==================================================
    # 7. EXPORT THREE REPORTS
    # ==================================================

    print("\n[7] EXPORT THREAT EVIDENCE + QA")

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

    geo.to_file(
        OUT_GPKG,
        layer="historical_future_threat_evidence_100m",
        driver="GPKG",
    )

    saved = pd.read_csv(
        OUT_CSV
    )

    meta = pyogrio.read_info(
        OUT_GPKG
    )

    check(
        "csv_export_9240",
        len(saved) == 9240
        and saved.grid_id.is_unique,
    )

    check(
        "gpkg_export_9240",
        meta["features"] == 9240,
    )

    check(
        "saved_pressure_missingness_preserved",
        int(
            saved[
                "future_urbanization_pressure_score"
            ].notna().sum()
        ) == 4620
        and int(
            saved[
                "future_urbanization_pressure_score"
            ].isna().sum()
        ) == 4620,
    )

    check(
        "frozen_gi_source_files_still_unchanged",
        all(
            digest(path)
            == manifest[
                "frozen_outputs"
            ][kind]["sha256"]
            for kind, path in (
                ("csv", GI_CSV),
                ("gpkg", GI_GPKG),
            )
        ),
    )

    gate("Export")

    # ==================================================
    # 8. WRITE METHODOLOGY AND QA EVIDENCE
    # ==================================================

    qa = {
        "section":
            "04_conservation_intelligence",

        "stage":
            "4.12",

        "status":
            "PASS_SEPARATE_THREAT_EVIDENCE_INTEGRATION_PENDING",

        "grid_cells":
            9240,

        "frozen_gi_status_only_used_for_overlap":
            True,

        "frozen_gi_values_or_weights_modified":
            False,

        "frozen_gi_manifest_path":
            str(
                GI_MANIFEST.relative_to(ROOT)
            ).replace("\\", "/"),

        "frozen_gi_manifest_sha256":
            digest(GI_MANIFEST),

        "historical_source_csv_sha256":
            digest(HISTORY_CSV),

        "historical_source_qa_sha256":
            digest(HISTORY_QA),

        "future_pressure_source_csv_sha256":
            digest(PRESSURE_CSV),

        "pressure_eligibility_source_csv_sha256":
            digest(ELIGIBILITY_CSV),

        "history": {
            "comparison":
                "2012-2014 to 2021-2023",

            "primary":
                "ROBUST_250M_GREEN_TO_BUILT_EXPOSURE_SHARE",

            "sensitivity": [
                "ROBUST_500M",
                "STRICT_250M",
                "STRICT_500M",
            ],

            "scope": (
                "Retrospective raster pixel-centre "
                "neighbourhood, excluding focal grid"
            ),

            "coverage":
                "9240 valid historical neighbourhoods",

            "robust_250m_share_distribution":
                summary(
                    numeric(
                        history[
                            "historical_g2b_robust_exposure_share_250m"
                        ]
                    )
                ),
        },

        "future_pressure": {
            "forecast_origin":
                2025,

            "forecast_period":
                "2026-2028",

            "scored_cells":
                int(predicted.sum()),

            "not_in_domain_cells":
                int((~predicted).sum()),

            "score_interpretation":
                PRESSURE_INTERPRETATION,

            "locked_threshold":
                THRESHOLD,

            "threshold_flagged_cells":
                int(flags.sum()),

            "in_domain_score_distribution":
                summary(raw_score),

            "not_in_domain_handling": (
                "NaN score, explicit NOT_IN_DOMAIN status"
            ),
        },

        "overlap": {
            "frozen_gi_scored":
                int(gi_available.sum()),

            "gi_and_future_pressure":
                int(both.sum()),

            "gi_only_pressure_not_in_domain":
                int(gi_only.sum()),

            "future_pressure_only_gi_unscored":
                int(pressure_only.sum()),

            "neither":
                int(neither.sum()),
        },

        "cross_tab_gi_status_by_pressure_domain": {
            k: {
                "model_domain":
                    int(
                        (
                            gi_status.eq(k).to_numpy()
                            & predicted
                        ).sum()
                    ),

                "not_in_domain":
                    int(
                        (
                            gi_status.eq(k).to_numpy()
                            & ~predicted
                        ).sum()
                    ),
            }
            for k in sorted(gi_status.unique())
        },

        "data_export": {
            "csv":
                str(
                    OUT_CSV.relative_to(ROOT)
                ).replace("\\", "/"),

            "gpkg":
                str(
                    OUT_GPKG.relative_to(ROOT)
                ).replace("\\", "/"),
        },

        "checks":
            checks,

        "historical_future_threat_composite_created":
            False,

        "conservation_priority_calculated":
            False,

        "remaining_decisions": [
            (
                "Decide if the combined GI and future "
                "pressure index is restricted to the "
                "4609 overlapping cells"
            ),
            (
                "Do not impute the 4118 GI cells "
                "outside the future pressure domain"
            ),
            (
                "Specify historical and future "
                "threat scaling transparently"
            ),
            (
                "Conduct sensitivity analysis "
                "before final conservation claims"
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

    print("CSV:", OUT_CSV)
    print("GPKG:", OUT_GPKG)
    print("QA:", OUT_QA)

    print("\n" + "=" * 76)
    print(
        "STAGE 4.12: "
        "PASS_SEPARATE_THREAT_EVIDENCE_INTEGRATION_PENDING"
    )
    print(
        "Frozen GI modified: NO"
    )
    print(
        "Composite threat / priority created: NO"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
