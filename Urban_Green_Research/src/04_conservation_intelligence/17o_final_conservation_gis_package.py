
"""
J26-DS-338 | Urban Green Research | Section 04 | Stage 4.16
Final reproducibility QA and GIS packaging (NOT ecological field validation).

Approved model: primary CP = frozen GI * frozen 2026-2028 UP.
Primary CP only for 4,609 common-domain cells. All 9,240 grid cells retained.
Historical conversion is supporting evidence, not an input to primary CP.
Stage 4.15 sensitivity is documented without changing the approved model.

Creates ONE multi-layer GeoPackage and ONE QA/data-contract JSON:
  final_gis/17o_conservation_intelligence_final.gpkg
  reports/conservation_intelligence/17o_conservation_final_qa.json

No changes to upstream frozen GI, pressure, CP, or sensitivity reports.
No final HIGH/MEDIUM/LOW categories. NOT_IN_DOMAIN is never zero.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio


ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / "data/processed/conservation_intelligence"
REPORT = ROOT / "outputs/reports/conservation_intelligence"

PRESSURE = (
    ROOT / "data/processed/urbanization_pressure/inference/"
    "14c_2026_2028_urbanization_pressure_scores.csv"
)

GI_CSV = CI / "frozen_importance/17j_frozen_green_importance_100m.csv"
GI_GPKG = CI / "frozen_importance/17j_frozen_green_importance_100m.gpkg"
GI_MANIFEST = REPORT / "17j_green_importance_freeze_manifest.json"

HISTORY_CSV = (
    CI / "historical_threat/"
    "17e_historical_green_loss_exposure_100m.csv"
)

THREAT_CSV = (
    CI / "threat_evidence/"
    "17k_historical_future_threat_evidence_100m.csv"
)
THREAT_QA = REPORT / "17k_historical_future_threat_evidence_qa.json"

INTEGRATED_CSV = (
    CI / "integrated_evidence/"
    "17l_green_importance_threat_integration_100m.csv"
)
INTEGRATED_QA = REPORT / "17l_green_importance_threat_integration_qa.json"

CP_CSV = (
    CI / "conservation_priority/"
    "17m_primary_conservation_priority_100m.csv"
)
CP_GPKG = (
    CI / "conservation_priority/"
    "17m_primary_conservation_priority_100m.gpkg"
)
CP_QA = REPORT / "17m_primary_conservation_priority_qa.json"

SENS_SUMMARY = REPORT / "17n_priority_sensitivity_scenarios.csv"
SENS_CELL = REPORT / "17n_priority_cell_rank_instability.csv"
SENS_QA = REPORT / "17n_priority_sensitivity_qa.json"

OUT_DIR = CI / "final_gis"
OUT_GPKG = OUT_DIR / "17o_conservation_intelligence_final.gpkg"
OUT_QA = REPORT / "17o_conservation_final_qa.json"

GI = "green_importance_index_0_1"
UP = "future_urbanization_pressure_score"
CP = "primary_conservation_priority_0_1"

LAYERS = {
    "primary_conservation_priority": 9240,
    "frozen_green_importance": 9240,
    "historical_loss_evidence": 9240,
    "priority_rank_sensitivity": 4609,
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unique_csv(path, expected):
    frame = pd.read_csv(path)
    if (
        "grid_id" not in frame.columns
        or len(frame) != expected
        or frame.grid_id.isna().any()
        or not frame.grid_id.is_unique
    ):
        raise ValueError(
            f"Invalid grid identity in {path.name}: expected {expected}"
        )
    return frame.set_index("grid_id")


def numeric(frame, field):
    if field not in frame.columns:
        raise ValueError(f"Missing field {field}")
    return pd.to_numeric(
        frame[field], errors="coerce"
    ).to_numpy(dtype=float)


def summary(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return {
        "count": int(len(v)),
        "min": float(v.min()),
        "p25": float(np.quantile(v, 0.25)),
        "median": float(np.median(v)),
        "p75": float(np.quantile(v, 0.75)),
        "p95": float(np.quantile(v, 0.95)),
        "max": float(v.max()),
    }


def main():
    print("=" * 77)
    print("SECTION 04 | STAGE 4.16 - FINAL QA AND GIS PACKAGING")
    print("=" * 77)

    required = [
        GI_CSV, GI_GPKG, GI_MANIFEST, HISTORY_CSV,
        THREAT_CSV, THREAT_QA, INTEGRATED_CSV, INTEGRATED_QA,
        CP_CSV, CP_GPKG, CP_QA, SENS_SUMMARY, SENS_CELL,
        SENS_QA, PRESSURE,
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    # A completed release must not be silently replaced.
    for path in (OUT_GPKG, OUT_QA):
        if path.exists():
            raise FileExistsError(
                f"Final output already exists; will not overwrite: {path}"
            )

    checks = {}

    def check(label, valid):
        checks[label] = bool(valid)
        print(f"[{'PASS' if checks[label] else 'FAIL'}] {label}")

    def gate(section):
        if not all(checks.values()):
            raise SystemExit(
                f"{section} QA failed. Final release not created."
            )

    # ======================================================
    # 1. VERIFY SOURCE LINEAGE
    # ======================================================

    print("\n[1] VERIFY SOURCE LINEAGE AND FINGERPRINTS")

    frozen = json.loads(
        GI_MANIFEST.read_text(encoding="utf-8")
    )
    threat_qa = json.loads(
        THREAT_QA.read_text(encoding="utf-8")
    )
    integrated_qa = json.loads(
        INTEGRATED_QA.read_text(encoding="utf-8")
    )
    cp_qa = json.loads(
        CP_QA.read_text(encoding="utf-8")
    )
    sens_qa = json.loads(
        SENS_QA.read_text(encoding="utf-8")
    )

    expected_status = {
        "frozen_GI": (
            frozen,
            "PASS_GREEN_IMPORTANCE_FROZEN_BASELINE_SENSITIVITY_PENDING",
        ),
        "threat": (
            threat_qa,
            "PASS_SEPARATE_THREAT_EVIDENCE_INTEGRATION_PENDING",
        ),
        "integrated": (
            integrated_qa,
            "PASS_DOMAIN_AWARE_INTEGRATION_PRIORITY_PENDING",
        ),
        "primary_CP": (
            cp_qa,
            "PASS_MULTIPLICATIVE_CONSERVATION_PRIORITY_SENSITIVITY_PENDING",
        ),
        "sensitivity": (
            sens_qa,
            "PASS_PRIORITY_SENSITIVITY_ANALYSIS",
        ),
    }

    for label, (record, expected) in expected_status.items():
        check(
            f"upstream_{label}_passed",
            record.get("status") == expected,
        )

    for kind, path in (("csv", GI_CSV), ("gpkg", GI_GPKG)):
        check(
            f"frozen_GI_{kind}_unchanged",
            sha256(path)
            == frozen.get("frozen_outputs", {})
            .get(kind, {}).get("sha256"),
        )

    for kind, path in (("csv", CP_CSV), ("gpkg", CP_GPKG)):
        check(
            f"primary_CP_{kind}_unchanged",
            sha256(path)
            == cp_qa.get("output_sha256", {}).get(kind),
        )

    check(
        "sensitivity_summary_unchanged",
        sha256(SENS_SUMMARY)
        == sens_qa.get("output_sha256", {}).get("scenario_report"),
    )

    check(
        "sensitivity_cell_report_unchanged",
        sha256(SENS_CELL)
        == sens_qa.get("output_sha256", {}).get("cell_rank_report"),
    )

    check(
        "original_historical_evidence_unchanged",
        sha256(HISTORY_CSV)
        == integrated_qa.get("source_sha256", {})
        .get("historical_source_csv"),
    )

    check(
        "original_future_pressure_unchanged",
        sha256(PRESSURE)
        == cp_qa.get("source_sha256", {})
        .get("original_pressure_csv"),
    )

    check(
        "stage_412_threat_evidence_unchanged",
        sha256(THREAT_CSV)
        == integrated_qa.get("source_sha256", {})
        .get("stage_412_threat_csv"),
    )

    check(
        "stage_413_integrated_csv_unchanged",
        sha256(INTEGRATED_CSV)
        == cp_qa.get("source_sha256", {}).get("integrated_csv"),
    )

    gate("Source lineage")

    # ======================================================
    # 2. VERIFY PRIMARY CONSERVATION PRIORITY
    # ======================================================

    print("\n[2] VERIFY PRIMARY CP, MISSINGNESS AND DOMAIN")

    primary = unique_csv(CP_CSV, 9240)
    green = unique_csv(GI_CSV, 9240)
    history = unique_csv(HISTORY_CSV, 9240)
    sensitivity = unique_csv(SENS_CELL, 4609)
    scenario = pd.read_csv(SENS_SUMMARY)

    ids = primary.index

    for label, frame in (
        ("frozen_GI", green),
        ("history", history),
    ):
        check(
            f"{label}_grid_ids_match",
            set(frame.index) == set(ids),
        )

    gate("Grid identity")

    green = green.loc[ids]
    history = history.loc[ids]

    status = primary["primary_priority_status"].astype(str)

    eligible = status.eq(
        "FUTURE_CONSERVATION_PRIORITY_CALCULATED"
    ).to_numpy()

    gi_only = status.eq(
        "NOT_IN_PRESSURE_DOMAIN"
    ).to_numpy()

    pressure_only = status.eq(
        "GI_UNSCORED_PRESSURE_AVAILABLE"
    ).to_numpy()

    neither = status.eq(
        "GI_UNSCORED_AND_PRESSURE_NOT_IN_DOMAIN"
    ).to_numpy()

    gi = numeric(primary, GI)
    up = numeric(primary, UP)
    cp = numeric(primary, CP)

    giready = green.green_importance_status.eq(
        "GREEN_IMPORTANCE_CALCULATED"
    ).to_numpy()

    model_domain = primary.future_pressure_domain_status.eq(
        "SCORED_MODEL_DOMAIN"
    ).to_numpy()

    check(
        "population_partition_4609_4118_11_502",
        (
            int(eligible.sum()),
            int(gi_only.sum()),
            int(pressure_only.sum()),
            int(neither.sum()),
        ) == (4609, 4118, 11, 502)
        and np.all(
            eligible | gi_only | pressure_only | neither
        ),
    )

    check(
        "frozen_GI_8727_scored",
        int(giready.sum()) == 8727,
    )

    check(
        "future_pressure_4620_scored",
        int(model_domain.sum()) == 4620,
    )

    check(
        "common_domain_matches_GI_and_pressure",
        np.array_equal(
            eligible, giready & model_domain
        ),
    )

    check(
        "GI_exactly_equal_to_frozen_source",
        np.allclose(
            gi,
            numeric(green, GI),
            atol=0,
            rtol=0,
            equal_nan=True,
        ),
    )

    check(
        "primary_CP_equals_GI_times_UP",
        np.isfinite(cp[eligible]).all()
        and np.allclose(
            cp[eligible],
            gi[eligible] * up[eligible],
            atol=1e-12,
            rtol=0,
        ),
    )

    check(
        "unscored_CP_missing_not_zero",
        int(np.isnan(cp).sum()) == 4631
        and np.isnan(cp[~eligible]).all(),
    )

    check(
        "pressure_unknown_not_imputed",
        np.isnan(up[~model_domain]).all()
        and np.isfinite(up[model_domain]).all()
        and np.all(
            (up[model_domain] >= 0)
            & (up[model_domain] <= 1)
        ),
    )

    check(
        "no_priority_classes_assigned",
        primary.priority_classes_assigned
        .astype(str).str.lower()
        .isin(("false", "0")).all(),
    )

    check(
        "frozen_GI_flag_preserved",
        primary.green_importance_frozen
        .astype(str).str.lower()
        .isin(("true", "1")).all(),
    )

    check(
        "pressure_is_relative_not_probability",
        primary.loc[
            model_domain,
            "future_score_interpretation",
        ].eq(
            "MODEL_DERIVED_PRESSURE_SCORE_NOT_CALIBRATED_PROBABILITY"
        ).all(),
    )

    flags = numeric(primary, "future_threshold_flag")

    check(
        "locked_pressure_threshold_and_flags_preserved",
        np.allclose(
            numeric(
                primary, "future_operating_threshold"
            )[model_domain],
            0.715,
            atol=1e-12,
            rtol=0,
        )
        and int(np.nansum(flags)) == 255
        and int(np.sum(flags[eligible] == 1)) == 248,
    )

    # Historical loss is present for interpretation
    # but not included in the primary formula.

    hist_cols = [
        c for c in history.columns
        if c.startswith("historical_g2b_")
    ]

    check(
        "historical_support_fields_available",
        len(hist_cols) >= 20,
    )

    for name in (
        "historical_g2b_robust_exposure_share_250m",
        "historical_g2b_robust_exposure_share_500m",
        "historical_g2b_strict_exposure_share_250m",
        "historical_g2b_strict_exposure_share_500m",
    ):
        original = numeric(history, name)
        embedded = numeric(primary, name)

        check(
            f"supporting_{name}_unchanged",
            np.isfinite(original).all()
            and np.allclose(
                original,
                embedded,
                atol=1e-12,
                rtol=0,
            ),
        )

    gate("Primary CP and historical evidence")

    # ======================================================
    # 3. VERIFY SENSITIVITY REPORTS
    # ======================================================

    print("\n[3] VERIFY STAGE 4.15 ROBUSTNESS REPORTS")

    check(
        "13_scenarios_present",
        len(scenario) == 13
        and scenario.scenario.is_unique
        and scenario.scenario.eq(
            "APPROVED_MULTIPLICATIVE_BASELINE"
        ).sum() == 1
        and scenario.scenario.eq(
            "ADDITIVE_EQUAL_TRADEOFF_COUNTERFACTUAL"
        ).sum() == 1,
    )

    check(
        "sensitivity_grid_ids_match_priority_domain",
        set(sensitivity.index) == set(ids[eligible]),
    )

    gate("Sensitivity identity")

    sensitivity = sensitivity.loc[ids[eligible]]

    check(
        "sensitivity_primary_values_match_original",
        np.allclose(
            numeric(sensitivity, "approved_primary_CP"),
            cp[eligible],
            rtol=0,
            atol=1e-12,
        ),
    )

    check(
        "top_decile_is_diagnostic_not_final_class",
        sens_qa.get(
            "top_decile_is_final_priority_class"
        ) is False
        and scenario.top10_cell_count.eq(461).all(),
    )

    check(
        "sensitivity_correlations_finite",
        np.isfinite(
            scenario[
                ["spearman_vs_primary", "kendall_vs_primary"]
            ].to_numpy(dtype=float)
        ).all(),
    )

    baseline = scenario.loc[
        scenario.scenario.eq(
            "APPROVED_MULTIPLICATIVE_BASELINE"
        )
    ].iloc[0]

    check(
        "baseline_sensitivity_identity",
        np.isclose(
            baseline.spearman_vs_primary,
            1.0,
            atol=1e-12,
        )
        and np.isclose(
            baseline.kendall_vs_primary,
            1.0,
            atol=1e-12,
        )
        and int(baseline.top10_overlap_count) == 461,
    )

    gate("Sensitivity")

    others = scenario.loc[
        ~scenario.scenario.eq(
            "APPROVED_MULTIPLICATIVE_BASELINE"
        )
    ]

    print(
        "Scenarios including baseline:",
        len(scenario),
    )
    print(
        "Lowest alternative top-10% retention:",
        f"{others.top10_retention_pct.min():.2f}%",
    )
    print(
        "Lowest alternative Spearman:",
        f"{others.spearman_vs_primary.min():.4f}",
    )

    # ======================================================
    # 4. PREPARE GIS LAYERS
    # ======================================================

    print("\n[4] VALIDATE GEOMETRY AND BUILD FOUR GIS LAYERS")

    priority_geo = gpd.read_file(
        CP_GPKG
    ).set_index("grid_id")

    gi_geo = gpd.read_file(
        GI_GPKG
    ).set_index("grid_id")

    check(
        "source_GIS_id_sets_match",
        len(priority_geo) == 9240
        and priority_geo.index.is_unique
        and len(gi_geo) == 9240
        and gi_geo.index.is_unique
        and set(priority_geo.index) == set(ids)
        and set(gi_geo.index) == set(ids),
    )

    check(
        "both_GIS_sources_EPSG32644",
        priority_geo.crs is not None
        and gi_geo.crs is not None
        and priority_geo.crs.to_epsg() == 32644
        and gi_geo.crs.to_epsg() == 32644,
    )

    gate("Source geometries")

    priority_geo = priority_geo.loc[ids]
    gi_geo = gi_geo.loc[ids]

    check(
        "source_geometries_valid",
        priority_geo.geometry.is_valid.all()
        and gi_geo.geometry.is_valid.all(),
    )

    check(
        "GI_and_priority_polygons_match",
        all(
            a.equals(b)
            for a, b in zip(
                priority_geo.geometry,
                gi_geo.geometry,
            )
        ),
    )

    check(
        "KMC_clipped_area_consistency",
        np.max(
            np.abs(
                priority_geo.geometry.area.to_numpy(
                    dtype=float
                )
                - numeric(
                    primary, "kmc_overlap_area_m2"
                )
            )
        ) <= 1.0,
    )

    check(
        "KMC_area_approx_88_829216_km2",
        np.isclose(
            priority_geo.geometry.area.sum() / 1e6,
            88.829216,
            atol=0.001,
            rtol=0,
        ),
    )

    check(
        "primary_GIS_matches_primary_CSV",
        np.allclose(
            numeric(priority_geo, CP),
            cp,
            equal_nan=True,
            rtol=0,
            atol=1e-12,
        ),
    )

    gate("Geometry and score QA")

    priority_layer = priority_geo.reset_index()
    green_layer = gi_geo.reset_index()

    historical_cols = [
        "historical_loss_early_period",
        "historical_loss_late_period",
        *hist_cols,
    ]

    history_aligned = (
        history[historical_cols]
        .reset_index()
    )

    historical_layer = (
        priority_layer[
            ["grid_id", "geometry"]
        ].merge(
            history_aligned,
            on="grid_id",
            how="inner",
            validate="one_to_one",
        )
    )

    sensitivity_layer = (
        priority_layer[
            ["grid_id", "geometry"]
        ].merge(
            sensitivity.reset_index(),
            on="grid_id",
            how="inner",
            validate="one_to_one",
        )
    )

    check(
        "historical_layer_9240",
        len(historical_layer) == 9240,
    )

    check(
        "sensitivity_layer_4609",
        len(sensitivity_layer) == 4609,
    )

    gate("GIS layers")

    # ======================================================
    # 5. CREATE MULTI-LAYER GEOPACKAGE
    # ======================================================

    print("\n[5] WRITE AND VERIFY FINAL MULTI-LAYER GEOPACKAGE")

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    products = {
        "primary_conservation_priority": priority_layer,
        "frozen_green_importance": green_layer,
        "historical_loss_evidence": historical_layer,
        "priority_rank_sensitivity": sensitivity_layer,
    }

    # Build in a temporary directory.
    # Move to the final path only after validation.

    with TemporaryDirectory(
        prefix="17o_build_",
        dir=OUT_DIR,
    ) as temp_dir:

        temporary_gpkg = (
            Path(temp_dir) / OUT_GPKG.name
        )

        for layer, frame in products.items():

            frame.to_file(
                temporary_gpkg,
                layer=layer,
                driver="GPKG",
                engine="pyogrio",
            )

            print(
                f"Wrote {layer}: "
                f"{len(frame):,} cells"
            )

        existing_layers = set(
            pyogrio.list_layers(
                temporary_gpkg
            )[:, 0]
        )

        check(
            "all_four_GIS_layers_written",
            existing_layers == set(LAYERS),
        )

        for layer, expected in LAYERS.items():

            info = pyogrio.read_info(
                temporary_gpkg,
                layer=layer,
            )

            check(
                f"{layer}_feature_count",
                info["features"] == expected,
            )

            check(
                f"{layer}_EPSG32644",
                info["crs"] == "EPSG:32644",
            )

        packaged_primary = (
            gpd.read_file(
                temporary_gpkg,
                layer="primary_conservation_priority",
            )
            .set_index("grid_id")
            .loc[ids]
        )

        check(
            "packaged_priority_exact_score_reproduction",
            np.allclose(
                numeric(packaged_primary, CP),
                cp,
                equal_nan=True,
                rtol=0,
                atol=1e-12,
            ),
        )

        check(
            "packaged_unscored_priority_values_still_missing",
            np.isnan(
                numeric(packaged_primary, CP)[~eligible]
            ).all(),
        )

        gate("Packaged GIS")

        if OUT_GPKG.exists():
            raise FileExistsError(OUT_GPKG)

        temporary_gpkg.replace(OUT_GPKG)

    check(
        "packaged_GPKG_exists",
        OUT_GPKG.is_file(),
    )

    check(
        "original_frozen_GI_still_unchanged",
        sha256(GI_CSV)
        == frozen["frozen_outputs"]["csv"]["sha256"],
    )

    check(
        "original_primary_CP_still_unchanged",
        sha256(CP_CSV)
        == cp_qa["output_sha256"]["csv"],
    )

    check(
        "original_pressure_still_unchanged",
        sha256(PRESSURE)
        == cp_qa["source_sha256"]["original_pressure_csv"],
    )

    gate("Release integrity")

    # ======================================================
    # 6. CREATE FINAL QA / DATA CONTRACT
    # ======================================================

    print("\n[6] WRITE FINAL QA AND GIS DATA CONTRACT")

    qa = {
        "section": "04_conservation_intelligence",
        "stage": "4.16",
        "status": (
            "PASS_FINAL_GIS_PACKAGE_WITH_DOCUMENTED_LIMITATIONS"
        ),
        "purpose": (
            "Final reproducible GIS packaging, "
            "not ecological field validation"
        ),
        "methodology": {
            "green_importance": (
                "Frozen Stage 4.11 GI: CRITIC-weighted "
                "G1 current green, G2 persistence, "
                "G3 250m green context"
            ),
            "primary_priority_formula": (
                "CP_i = GI_i * UP_i"
            ),
            "future_pressure_period": (
                "2026-2028, forecast origin 2025"
            ),
            "pressure_interpretation": (
                "Model-derived relative score, "
                "NOT a calibrated probability"
            ),
            "historical_loss_role": (
                "Supporting diagnostics only; "
                "not a primary CP or direct frozen GI term"
            ),
            "priority_classification": (
                "No HIGH/MEDIUM/LOW categories selected"
            ),
            "pressure_locked_threshold": 0.715,
            "threshold_flag_is_not_priority_class": True,
        },
        "scope": {
            "all_KMC_grid_cells": 9240,
            "GI_scored_cells": 8727,
            "primary_CP_scored_common_domain": 4609,
            "GI_only_future_pressure_NOT_IN_DOMAIN": 4118,
            "pressure_only_GI_unscored": 11,
            "neither": 502,
            "unscored_primary_CP_nulls": 4631,
            "KMC_area_km2": float(
                priority_geo.geometry.area.sum() / 1e6
            ),
            "CRS": "EPSG:32644",
        },
        "primary_CP_distribution": summary(
            cp[eligible]
        ),
        "sensitivity": {
            "scenarios_including_primary": int(
                len(scenario)
            ),
            "top10_diagnostic_cell_count": 461,
            "lowest_alternative_top10_retention_pct": float(
                others.top10_retention_pct.min()
            ),
            "lowest_alternative_spearman": float(
                others.spearman_vs_primary.min()
            ),
            "largest_alternative_p95_rank_shift": float(
                others.p95_absolute_rank_shift.max()
            ),
            "priority_stability_is_not_field_validation": True,
        },
        "GIS_package": {
            "path": str(
                OUT_GPKG.relative_to(ROOT)
            ).replace("\\", "/"),
            "sha256": sha256(OUT_GPKG),
            "layers": {
                name: {
                    "features": size,
                    "fields": [
                        c for c in products[name].columns
                        if c != "geometry"
                    ],
                }
                for name, size in LAYERS.items()
            },
        },
        "source_sha256": {
            "frozen_GI_csv": sha256(GI_CSV),
            "frozen_GI_gpkg": sha256(GI_GPKG),
            "primary_CP_csv": sha256(CP_CSV),
            "primary_CP_gpkg": sha256(CP_GPKG),
            "historical_source_csv": sha256(HISTORY_CSV),
            "original_pressure_csv": sha256(PRESSURE),
            "sensitivity_scenarios_csv": sha256(SENS_SUMMARY),
            "sensitivity_cells_csv": sha256(SENS_CELL),
        },
        "dashboard_guidance": {
            "primary_score_field": CP,
            "priority_status_field": "primary_priority_status",
            "pressure_domain_field": (
                "future_pressure_domain_status"
            ),
            "GI_field": GI,
            "model_pressure_field": UP,
            "unknown_handling": (
                "Null pressure or CP and explicit status, "
                "never fill with zero"
            ),
            "historical_map": (
                "historical_loss_evidence layer, "
                "independent supporting context"
            ),
            "sensitivity_map": (
                "priority_rank_sensitivity layer; "
                "ranks within 4609 cells only"
            ),
            "GN_division_name": (
                "Not assigned in this packaging step; "
                "requires separately validated spatial join"
            ),
        },
        "limitations": [
            (
                "GI and UP scores are model-derived proxies, "
                "not ecological field validation."
            ),
            (
                "The pressure inference domain covers "
                "only 4620 KMC cells."
            ),
            (
                "4118 currently green cells have NO valid "
                "future pressure prediction."
            ),
            (
                "Rank robustness is conditional on "
                "the 13 tested scenarios and "
                "4609-cell common domain."
            ),
            (
                "Spatial autocorrelation and remote-sensing "
                "classification uncertainty remain."
            ),
            (
                "No conservation-priority classes or policy "
                "action thresholds have been authorized."
            ),
            (
                "GN names, web map tiles, and backend "
                "endpoints are not generated here."
            ),
        ],
        "checks": checks,
        "upstream_sources_modified": False,
        "priority_model_changed": False,
    }

    OUT_QA.write_text(
        json.dumps(qa, indent=2),
        encoding="utf-8",
    )

    print("GeoPackage:", OUT_GPKG)
    print("QA/data-contract JSON:", OUT_QA)

    print("\n" + "=" * 77)
    print(
        "STAGE 4.16: "
        "PASS_FINAL_GIS_PACKAGE_WITH_DOCUMENTED_LIMITATIONS"
    )
    print(
        "Primary CP: 4,609 valid cells | "
        "4,631 unscored cells"
    )
    print(
        "No source models or frozen products modified"
    )
    print("=" * 77)


if __name__ == "__main__":
    main()
