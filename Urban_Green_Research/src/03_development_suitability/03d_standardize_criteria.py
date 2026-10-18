
"""
J26-DS-338 | Section 03 - Stage 3.5

Development Suitability:
Criterion Direction and Standardization

Method:
Empirical P05-P95 clipped linear scaling.

Important:
- Scores are RELATIVE to the KMC study area.
- Percentiles are NOT legal or engineering thresholds.
- No AHP weights or final suitability scores.
- Raw inputs are not modified.
"""

from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    ROOT / "data/processed/development_suitability/raw_criteria"
)

OUTPUT_DIR = (
    ROOT / "data/processed/development_suitability/"
    "standardized_criteria"
)

REPORT_DIR = (
    ROOT / "outputs/reports/development_suitability"
)

RAW_CSV = RAW_DIR / "03b_raw_criteria_100m.csv"
RAW_GPKG = RAW_DIR / "03b_raw_criteria_100m.gpkg"

CRITERIA = {
    "C1": {
        "raw": "c1_major_road_distance_m",
        "score": "c1_road_access_score",
        "direction": "COST",
        "meaning": "Higher score means closer to major roads",
    },
    "C2": {
        "raw": "c2_road_density_500m_km_per_km2",
        "score": "c2_road_context_score",
        "direction": "BENEFIT",
        "meaning": "Higher score means denser local road network",
    },
    "C3": {
        "raw": "c3_context_building_pct_500m",
        "score": "c3_development_context_score",
        "direction": "BENEFIT",
        "meaning": "Higher score means more established building context",
    },
    "C4": {
        "raw": "c4_slope_mean_deg",
        "score": "c4_slope_score",
        "direction": "COST",
        "meaning": "Higher score means gentler terrain",
    },
}


def standardize(values, lower, upper, direction):
    """Clip to empirical anchors and scale to 0-1."""

    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Non-finite percentile anchors")

    if upper <= lower:
        raise ValueError("Invalid standardization range")

    if direction == "BENEFIT":
        result = (values - lower) / (upper - lower)

    elif direction == "COST":
        result = (upper - values) / (upper - lower)

    else:
        raise ValueError(f"Unknown direction: {direction}")

    return np.clip(result, 0.0, 1.0)


def verify_monotonicity(raw_values, scores, direction):
    """Check score direction without relying on correlation alone."""

    order = np.argsort(raw_values, kind="stable")
    sorted_scores = scores[order]
    differences = np.diff(sorted_scores)

    tolerance = 1e-10

    if direction == "BENEFIT":
        return bool(np.all(differences >= -tolerance))

    return bool(np.all(differences <= tolerance))


def main():

    print("=" * 65)
    print("STAGE 3.5 - CRITERION STANDARDIZATION")
    print("=" * 65)

    if not RAW_CSV.is_file() or not RAW_GPKG.is_file():
        raise FileNotFoundError(
            "Stage 3.3 raw CSV or GPKG is missing."
        )

    raw = pd.read_csv(RAW_CSV)

    required = [
        "grid_id",
        "kmc_overlap_area_m2",
        "kmc_overlap_fraction",
        "boundary_cell",
        "coverage_group",
        "terrain_sample_method",
        "c3_building_footprint_pct_cell",
    ]

    required += [
        config["raw"] for config in CRITERIA.values()
    ]

    missing = [c for c in required if c not in raw.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if len(raw) != 9240:
        raise ValueError("Unexpected raw grid count")

    if raw["grid_id"].isna().any() or not raw["grid_id"].is_unique:
        raise ValueError("Missing or duplicate grid IDs")

    numeric = [
        config["raw"] for config in CRITERIA.values()
    ]

    if not np.isfinite(
        raw[numeric].to_numpy(dtype=float)
    ).all():
        raise ValueError("Raw criteria contain NaN or infinity")

    print("\n[1] INPUT VALIDATION")
    print("Raw cells:", len(raw))
    print("Selected weighted candidates:", len(CRITERIA))
    print("Alternative radius measurements remain in raw dataset")

    # Keep the original measurements alongside the new scores.
    # Current-cell occupation is diagnostic only, not weighted.
    result = raw[required].copy()

    parameters = {}
    checks = {}

    # --------------------------------------------------
    # 2. Standardization
    # --------------------------------------------------

    print("\n[2] STANDARDIZATION PARAMETERS")

    for criterion_id, config in CRITERIA.items():

        raw_col = config["raw"]
        score_col = config["score"]
        direction = config["direction"]

        values = raw[raw_col].to_numpy(dtype=float)

        lower = float(np.quantile(values, 0.05))
        upper = float(np.quantile(values, 0.95))

        scores = standardize(
            values, lower, upper, direction
        )

        result[score_col] = scores

        monotonic = verify_monotonicity(
            values, scores, direction
        )

        checks[f"{criterion_id}_direction_correct"] = monotonic

        checks[f"{criterion_id}_finite"] = bool(
            np.isfinite(scores).all()
        )

        checks[f"{criterion_id}_range_0_1"] = bool(
            np.all((scores >= 0) & (scores <= 1))
        )

        parameters[criterion_id] = {
            "raw_measurement": raw_col,
            "standardized_column": score_col,
            "direction": direction,
            "lower_p05": lower,
            "upper_p95": upper,
            "method": "clipped_linear_p05_p95",
            "meaning": config["meaning"],
            "score_min": float(scores.min()),
            "score_median": float(np.median(scores)),
            "score_max": float(scores.max()),
            "score_zero_pct": float(
                100 * np.mean(scores == 0)
            ),
            "score_one_pct": float(
                100 * np.mean(scores == 1)
            ),
        }

        print(f"\n{criterion_id}: {raw_col}")
        print("Direction:", direction)
        print(f"P05: {lower:.6f}")
        print(f"P95: {upper:.6f}")
        print(f"Score median: {np.median(scores):.4f}")
        print(f"Score = 0: {np.mean(scores == 0)*100:.2f}%")
        print(f"Score = 1: {np.mean(scores == 1)*100:.2f}%")
        print("Monotonic direction:", "PASS" if monotonic else "FAIL")

    # --------------------------------------------------
    # 3. GIS geometry validation
    # --------------------------------------------------

    print("\n[3] GIS VALIDATION")

    geometry = gpd.read_file(RAW_GPKG)

    if geometry.crs is None or geometry.crs.to_epsg() != 32644:
        raise ValueError("Incorrect raw GPKG CRS")

    if len(geometry) != 9240:
        raise ValueError("Raw GPKG cell count is incorrect")

    if geometry["grid_id"].isna().any() or not geometry["grid_id"].is_unique:
        raise ValueError("Raw GPKG has missing or duplicate IDs")

    if geometry.geometry.isna().any():
        raise ValueError("Null geometry detected")

    if not geometry.geometry.is_valid.all():
        raise ValueError("Invalid geometry detected")

    geo = geometry[["grid_id", "geometry"]].merge(
        result,
        on="grid_id",
        how="inner",
        validate="one_to_one",
    )

    area_diff = np.abs(
        geo.geometry.area.to_numpy()
        - geo["kmc_overlap_area_m2"].to_numpy()
    )

    checks["all_9240_cells_preserved"] = (
        len(result) == len(geo) == 9240
    )

    checks["unique_output_ids"] = (
        result["grid_id"].is_unique
        and geo["grid_id"].is_unique
    )

    checks["grid_id_sets_equal"] = (
        set(result["grid_id"]) == set(geo["grid_id"])
    )

    checks["geometry_area_reconciliation"] = bool(
        np.isfinite(area_diff).all()
        and area_diff.max() <= 1.0
    )

    checks["correct_gis_crs"] = (
        geo.crs is not None
        and geo.crs.to_epsg() == 32644
    )

    print("CSV rows:", len(result))
    print("GIS features:", len(geo))
    print("CRS:", geo.crs)
    print(f"Maximum geometry-area error: {area_diff.max():.6f} m2")

    # --------------------------------------------------
    # 4. QA and reporting
    # --------------------------------------------------

    print("\n[4] FINAL QA")

    status = "PASS" if all(checks.values()) else "REVIEW_REQUIRED"

    for name, value in checks.items():
        print(f"[{'PASS' if value else 'FAIL'}] {name}")

    if status != "PASS":
        raise SystemExit(
            "Stage 3.5 failed QA. No standardized outputs exported."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = (
        OUTPUT_DIR / "03d_standardized_criteria_100m.csv"
    )

    gpkg_path = (
        OUTPUT_DIR / "03d_standardized_criteria_100m.gpkg"
    )

    qa_path = (
        REPORT_DIR / "03d_standardization_qa.json"
    )

    result.to_csv(csv_path, index=False)

    geo.to_file(
        gpkg_path,
        layer="standardized_criteria_100m",
        driver="GPKG",
    )

    saved_csv = pd.read_csv(
        csv_path, usecols=["grid_id"]
    )

    saved_gpkg = pyogrio.read_info(gpkg_path)

    checks["csv_export_count"] = (
        len(saved_csv) == 9240
        and saved_csv["grid_id"].is_unique
    )

    checks["gpkg_export_count"] = (
        saved_gpkg["features"] == 9240
    )

    status = "PASS" if all(checks.values()) else "REVIEW_REQUIRED"

    report = {
        "stage": "3.5",
        "qa_status": status,
        "methodology_status": (
            "EMPIRICAL_RELATIVE_STANDARDIZATION"
        ),
        "study_area": "Kaduwela Municipal Council",
        "total_cells": len(result),
        "standardization": "P05-P95 clipped linear",
        "weighted_candidate_criteria": list(CRITERIA.keys()),
        "parameters": parameters,
        "max_geometry_area_error_m2": float(area_diff.max()),
        "checks": checks,
        "methodology_controls": {
            "no_pressure_inputs": True,
            "no_shap_inputs": True,
            "no_conservation_inputs": True,
            "no_provisional_zoning_inputs": True,
            "ahp_weights_assigned": False,
            "suitability_score_calculated": False,
        },
        "limitations": [
            "Percentile anchors are empirical and area-specific.",
            "Scores do not establish legal developability.",
            "C2 and C3 positive directions are modelling assumptions.",
            "Current-cell building occupation is not weighted.",
            "Alternative radii remain for sensitivity analysis.",
            "Alternative percentile anchors must be tested in Stage 3.10.",
            "AHP judgments and final suitability remain pending.",
        ],
    }

    qa_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nGenerated outputs:")
    print(csv_path)
    print(gpkg_path)
    print(qa_path)

    print("\n" + "=" * 65)
    print("STAGE 3.5 QA:", status)
    print("Scores generated: C1, C2, C3, C4")
    print("All values expected in [0,1]")
    print("AHP weights assigned: NO")
    print("Final suitability score calculated: NO")
    print("=" * 65)

    if status != "PASS":
        raise SystemExit(
            "Export verification failed. Review outputs."
        )


if __name__ == "__main__":
    main()
