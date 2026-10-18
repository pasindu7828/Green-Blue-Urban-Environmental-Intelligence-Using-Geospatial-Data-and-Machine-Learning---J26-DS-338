
"""
J26-DS-338 | Section 04 - Conservation Intelligence

Stage 4.3: Current Remaining-Green Baseline (2025)

PRIMARY SOURCE:
2025 RAW_RF area-weighted LULC fractions.

PURPOSE:
- Preserve all 9,240 KMC grid cells.
- Use RAW_RF, never STABILIZED_ANALYTICAL, for current green.
- Retain the actual observed green fraction.
- Track valid LULC coverage and unmapped area.
- Keep zero-coverage cells as UNKNOWN, not zero green.
- Calculate defensible green-area lower and upper bounds.
- Preserve actual Grid intersection KMC geometry.
- Perform reproducibility and GIS QA.

EXCLUDED:
- Urbanization Pressure
- SHAP
- Development Suitability
- AHP
- Conservation Priority
- Provisional UDA zoning

No model training or conservation index calculation.
"""

from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio


ROOT = Path(__file__).resolve().parents[2]

GRID_FILE = (
    ROOT
    / "data/processed/urbanization_pressure/grid/"
      "kaduwela_100m_modelling_grid.gpkg"
)

BOUNDARY_FILE = (
    ROOT
    / "data/processed/boundaries/kmc/"
      "kaduwela_kmc_boundary_utm44n.gpkg"
)

FRACTIONS_FILE = (
    ROOT
    / "data/processed/urbanization_pressure/lulc_grid/"
      "annual_fractions/"
      "kaduwela_100m_lulc_fractions_2025.csv"
)

PREVIOUS_AUDIT = (
    ROOT
    / "outputs/reports/conservation_intelligence/"
      "17a_data_readiness_qa.json"
)

OUTPUT_DIR = (
    ROOT
    / "data/processed/conservation_intelligence/"
      "current_green"
)

REPORT_DIR = (
    ROOT
    / "outputs/reports/conservation_intelligence"
)

CSV_OUTPUT = (
    OUTPUT_DIR
    / "17b_current_green_condition_2025_100m.csv"
)

GPKG_OUTPUT = (
    OUTPUT_DIR
    / "17b_current_green_condition_2025_100m.gpkg"
)

QA_OUTPUT = (
    REPORT_DIR
    / "17b_current_green_condition_qa.json"
)


def require_columns(df, names, label):
    missing = sorted(set(names) - set(df.columns))

    if missing:
        raise ValueError(
            f"{label} missing required columns: {missing}"
        )


def is_explicit_false(series):
    """Avoid interpreting missing metadata as False."""

    values = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    return bool(
        values.notna().all()
        and values.isin(
            ["false", "0", "no"]
        ).all()
    )


def main():

    print("=" * 72)
    print("SECTION 04 | STAGE 4.3 - CURRENT GREEN CONDITION")
    print("=" * 72)

    for path in [
        GRID_FILE,
        BOUNDARY_FILE,
        FRACTIONS_FILE,
        PREVIOUS_AUDIT,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    previous = json.loads(
        PREVIOUS_AUDIT.read_text(encoding="utf-8")
    )

    if previous.get("status") != (
        "PASS_SOURCE_READINESS_PENDING_CODEBOOK"
    ):
        raise ValueError(
            "Stages 4.1-4.2 have not passed readiness QA"
        )

    checks = {}

    def check(name, condition):
        passed = bool(condition)
        checks[name] = passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    # ====================================================
    # 1. READ EXISTING GRID AND RAW 2025 PRODUCT
    # ====================================================

    print("\n[1] LOAD SOURCE DATA")

    grid = gpd.read_file(GRID_FILE)
    boundary = gpd.read_file(BOUNDARY_FILE)
    products = pd.read_csv(FRACTIONS_FILE)

    grid_fields = [
        "grid_id",
        "grid_row",
        "grid_col",
        "kmc_overlap_area_m2",
        "kmc_overlap_fraction",
        "boundary_cell",
        "coverage_group",
    ]

    fraction_fields = [
        "grid_id",
        "year",
        "product_type",
        "future_informed",
        "provisional_year",
        "kmc_overlap_area_m2",
        "valid_lulc_area_m2",
        "valid_lulc_coverage_fraction",
        "green_area_m2",
        "built_area_m2",
        "open_bare_area_m2",
        "water_area_m2",
        "green_fraction",
        "built_fraction",
        "open_bare_fraction",
        "water_fraction",
    ]

    require_columns(grid, grid_fields, "Grid")
    require_columns(
        products, fraction_fields, "LULC fractions"
    )

    product_names = (
        products["product_type"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    raw = products.loc[
        product_names.eq("RAW_RF")
    ].copy()

    stabilized = products.loc[
        product_names.eq("STABILIZED_ANALYTICAL")
    ].copy()

    print("All product rows:", len(products))
    print("RAW_RF rows:", len(raw))
    print("STABILIZED_ANALYTICAL rows:", len(stabilized))

    check(
        "raw_product_9240_unique_grid_ids",
        len(raw) == 9240
        and raw.grid_id.notna().all()
        and raw.grid_id.is_unique,
    )

    check(
        "stabilized_reference_9240_unique_grid_ids",
        len(stabilized) == 9240
        and stabilized.grid_id.notna().all()
        and stabilized.grid_id.is_unique,
    )

    check(
        "grid_9240_unique_ids",
        len(grid) == 9240
        and grid.grid_id.notna().all()
        and grid.grid_id.is_unique,
    )

    check(
        "both_products_match_grid",
        set(raw.grid_id) == set(grid.grid_id)
        and set(stabilized.grid_id) == set(grid.grid_id),
    )

    check(
        "raw_origin_year_is_2025",
        pd.to_numeric(
            raw.year, errors="coerce"
        ).eq(2025).all(),
    )

    check(
        "raw_not_future_informed",
        is_explicit_false(raw.future_informed),
    )

    check(
        "raw_not_provisional",
        is_explicit_false(raw.provisional_year),
    )

    check(
        "correct_projected_crs",
        grid.crs is not None
        and boundary.crs is not None
        and grid.crs.to_epsg() == 32644
        and boundary.crs.to_epsg() == 32644,
    )

    if not all(checks.values()):
        raise SystemExit(
            "Source validation failed. No output exported."
        )

    # Align all records using grid_id, never row position
    # from the original two-product file.

    grid = grid.reset_index(drop=True)

    raw = (
        raw.set_index("grid_id")
        .loc[grid["grid_id"]]
        .reset_index()
    )

    # ====================================================
    # 2. CHECK KMC SPATIAL GEOMETRY
    # ====================================================

    print("\n[2] VERIFY GRID INTERSECTION KMC")

    kmc_shape = boundary.geometry.union_all()

    clipped_geometry = (
        grid.geometry.intersection(kmc_shape)
    )

    actual_areas = (
        clipped_geometry.area.to_numpy(dtype=float)
    )

    stored_areas = (
        grid["kmc_overlap_area_m2"]
        .to_numpy(dtype=float)
    )

    kmc_area = float(
        boundary.geometry.area.sum()
    )

    max_geometry_area_difference = float(
        np.max(
            np.abs(actual_areas - stored_areas)
        )
    )

    study_area_difference = abs(
        float(actual_areas.sum()) - kmc_area
    )

    print("KMC area:", round(kmc_area / 1e6, 6), "km2")
    print("Clipped grid cells:", len(clipped_geometry))
    print(
        "Maximum cell area difference:",
        max_geometry_area_difference,
        "m2"
    )

    check(
        "clipped_geometries_valid",
        clipped_geometry.notna().all()
        and clipped_geometry.is_valid.all()
        and (actual_areas > 0).all(),
    )

    check(
        "clipped_grid_area_matches_kmc",
        study_area_difference <= 1.0,
    )

    check(
        "stored_kmc_areas_match_geometries",
        max_geometry_area_difference <= 1.0,
    )

    source_area = (
        raw["kmc_overlap_area_m2"]
        .to_numpy(dtype=float)
    )

    check(
        "source_fraction_areas_match_grid",
        np.isfinite(source_area).all()
        and np.max(
            np.abs(source_area - stored_areas)
        ) <= 1.0,
    )

    if not all(checks.values()):
        raise SystemExit(
            "Geometry validation failed."
        )

    # ====================================================
    # 3. VALIDATE 2025 CLASSIFIED LULC COVERAGE
    # ====================================================

    print("\n[3] ANALYSE VALID LULC COVERAGE")

    valid_area = (
        raw["valid_lulc_area_m2"]
        .to_numpy(dtype=float)
    )

    coverage = (
        raw["valid_lulc_coverage_fraction"]
        .to_numpy(dtype=float)
    )

    class_area_columns = [
        "green_area_m2",
        "built_area_m2",
        "open_bare_area_m2",
        "water_area_m2",
    ]

    class_fraction_columns = [
        "green_fraction",
        "built_fraction",
        "open_bare_fraction",
        "water_fraction",
    ]

    class_areas = raw[
        class_area_columns
    ].to_numpy(dtype=float)

    class_fractions = raw[
        class_fraction_columns
    ].to_numpy(dtype=float)

    valid = valid_area > 1e-9
    unknown = ~valid

    unmapped_area = np.maximum(
        stored_areas - valid_area,
        0.0,
    )

    check(
        "valid_area_and_coverage_finite",
        np.isfinite(valid_area).all()
        and np.isfinite(coverage).all()
        and np.isfinite(stored_areas).all(),
    )

    check(
        "coverage_within_valid_bounds",
        (valid_area >= 0).all()
        and (valid_area <= stored_areas + 1.0).all()
        and (coverage >= -1e-9).all()
        and (coverage <= 1 + 1e-9).all(),
    )

    check(
        "coverage_reproduced",
        np.allclose(
            coverage,
            valid_area / stored_areas,
            atol=1e-7,
        ),
    )

    check(
        "class_areas_match_valid_area",
        np.isfinite(class_areas).all()
        and (class_areas >= -1e-7).all()
        and np.allclose(
            class_areas.sum(axis=1),
            valid_area,
            atol=1.0,
        ),
    )

    check(
        "observed_class_fractions_valid",
        np.isfinite(
            class_fractions[valid]
        ).all()
        and (
            class_fractions[valid] >= -1e-8
        ).all()
        and (
            class_fractions[valid] <= 1 + 1e-8
        ).all()
        and np.allclose(
            class_fractions[valid].sum(axis=1),
            1.0,
            atol=1e-6,
        ),
    )

    check(
        "green_fraction_matches_green_area",
        np.allclose(
            class_fractions[valid, 0],
            class_areas[valid, 0] / valid_area[valid],
            atol=1e-6,
        ),
    )

    print("Valid LULC cells:", int(valid.sum()))
    print("Unknown zero-coverage cells:", int(unknown.sum()))
    print(
        "Cells with coverage below 80%:",
        int((coverage < 0.8).sum()),
    )

    # ====================================================
    # 4. CREATE CURRENT GREEN CONDITION INDICATORS
    # ====================================================

    print("\n[4] GENERATE CURRENT GREEN INDICATORS")

    # Fractions are relative to valid classified LULC area.
    # Unknown cells must remain NaN, not become zero.

    observed_fractions = np.full(
        (len(raw), 4),
        np.nan,
        dtype=float,
    )

    observed_fractions[valid] = (
        class_fractions[valid]
    )

    green_area = class_areas[:, 0]

    # Confirmed green proportion of total KMC-overlap area:
    # A lower bound where LULC coverage is incomplete.
    #
    # If unmapped area were entirely green, the upper bound
    # would be (mapped green area + unmapped area) / KMC area.
    #
    # These are coverage-based bounds, not confidence
    # intervals or Landsat classification uncertainty.

    green_lower = (
        green_area / stored_areas
    )

    green_upper = (
        (green_area + unmapped_area)
        / stored_areas
    )

    green_lower = np.clip(
        green_lower, 0.0, 1.0
    )

    green_upper = np.clip(
        green_upper, 0.0, 1.0
    )

    evidence_status = np.full(
        len(raw),
        "NO_GREEN_DETECTED_IN_VALID_AREA",
        dtype=object,
    )

    evidence_status[
        valid & (green_area > 1e-7)
    ] = "GREEN_DETECTED_IN_VALID_AREA"

    evidence_status[
        unknown
    ] = "UNKNOWN_NO_VALID_LULC"

    coverage_status = np.full(
        len(raw),
        "PARTIAL_VALID_COVERAGE",
        dtype=object,
    )

    coverage_status[
        coverage >= 1 - 1e-7
    ] = "FULL_VALID_COVERAGE"

    coverage_status[
        unknown
    ] = "NO_VALID_COVERAGE"

    result = grid[
        [
            "grid_id",
            "grid_row",
            "grid_col",
            "kmc_overlap_area_m2",
            "kmc_overlap_fraction",
            "boundary_cell",
            "coverage_group",
        ]
    ].copy()

    result["source_year"] = 2025
    result["source_product_type"] = "RAW_RF"

    result["valid_lulc_area_m2"] = valid_area

    result["valid_lulc_coverage_fraction"] = (
        coverage
    )

    result["unmapped_lulc_area_m2"] = (
        unmapped_area
    )

    result["current_green_area_m2"] = (
        class_areas[:, 0]
    )

    result["current_built_area_m2"] = (
        class_areas[:, 1]
    )

    result["current_open_bare_area_m2"] = (
        class_areas[:, 2]
    )

    result["current_water_area_m2"] = (
        class_areas[:, 3]
    )

    result["current_green_fraction"] = (
        observed_fractions[:, 0]
    )

    result["current_built_fraction"] = (
        observed_fractions[:, 1]
    )

    result["current_open_bare_fraction"] = (
        observed_fractions[:, 2]
    )

    result["current_water_fraction"] = (
        observed_fractions[:, 3]
    )

    result["mapped_green_share_lower_bound"] = (
        green_lower
    )

    result["possible_green_share_upper_bound"] = (
        green_upper
    )

    result["green_condition_status"] = (
        evidence_status
    )

    result["lulc_coverage_status"] = (
        coverage_status
    )

    # No conservation candidate mask is chosen yet.
    # No arbitrary >= 0.5 green threshold is used.

    # ====================================================
    # 5. FINAL VALIDATION
    # ====================================================

    print("\n[5] CURRENT GREEN BASELINE QA")

    green_values = result[
        "current_green_fraction"
    ].to_numpy(dtype=float)

    check(
        "9240_cells_preserved",
        len(result) == 9240,
    )

    check(
        "unique_grid_ids_preserved",
        result.grid_id.notna().all()
        and result.grid_id.is_unique,
    )

    check(
        "expected_9195_observed_45_unknown",
        int(valid.sum()) == 9195
        and int(unknown.sum()) == 45,
    )

    check(
        "unknown_green_fraction_is_nan",
        np.isnan(green_values[unknown]).all(),
    )

    check(
        "observed_green_fraction_finite_0_to_1",
        np.isfinite(green_values[valid]).all()
        and (
            (green_values[valid] >= 0)
            & (green_values[valid] <= 1)
        ).all(),
    )

    check(
        "coverage_based_green_bounds_valid",
        np.isfinite(green_lower).all()
        and np.isfinite(green_upper).all()
        and (green_lower >= 0).all()
        and (green_upper <= 1).all()
        and (
            green_lower <= green_upper + 1e-9
        ).all(),
    )

    check(
        "green_lower_bound_matches_observed",
        np.allclose(
            green_lower[valid],
            green_values[valid] * coverage[valid],
            atol=1e-6,
        ),
    )

    check(
        "unknown_cells_labeled_correctly",
        (
            result.loc[
                unknown, "green_condition_status"
            ] == "UNKNOWN_NO_VALID_LULC"
        ).all(),
    )

    forbidden_terms = [
        "pressure",
        "shap",
        "suitability",
        "ahp",
        "zoning",
        "priority",
    ]

    forbidden_columns = [
        col for col in result.columns
        if any(
            term in col.lower()
            for term in forbidden_terms
        )
    ]

    check(
        "no_forbidden_model_inputs",
        len(forbidden_columns) == 0,
    )

    for name, passed in checks.items():
        print(
            f"[{'PASS' if passed else 'FAIL'}] {name}"
        )

    if not all(checks.values()):
        raise SystemExit(
            "Stage 4.3 QA failed. No outputs exported."
        )

    # ====================================================
    # 6. GREEN CONDITION SUMMARY
    # ====================================================

    print("\n[6] CURRENT GREEN SUMMARY")

    observed_green = green_values[valid]

    detected_green_count = int(
        np.count_nonzero(
            valid & (green_area > 1e-7)
        )
    )

    no_green_detected_count = int(
        np.count_nonzero(
            valid & (green_area <= 1e-7)
        )
    )

    # Mapped green area is measured only from valid pixels.
    total_mapped_green_km2 = float(
        green_area.sum() / 1e6
    )

    # Lower bound for area represented by source fractions.
    mapped_green_pct_of_kmc = float(
        100 * green_area.sum()
        / stored_areas.sum()
    )

    quantiles = {
        "p00": float(np.quantile(observed_green, 0.00)),
        "p05": float(np.quantile(observed_green, 0.05)),
        "p25": float(np.quantile(observed_green, 0.25)),
        "p50": float(np.quantile(observed_green, 0.50)),
        "p75": float(np.quantile(observed_green, 0.75)),
        "p95": float(np.quantile(observed_green, 0.95)),
        "p100": float(np.quantile(observed_green, 1.00)),
    }

    print(
        "Cells with detected green:",
        detected_green_count,
    )

    print(
        "Cells without green detected in valid area:",
        no_green_detected_count,
    )

    print(
        "Cells with unknown green condition:",
        int(unknown.sum()),
    )

    print(
        "Mapped green area:",
        round(total_mapped_green_km2, 4),
        "km2"
    )

    print(
        "Confirmed mapped green share of KMC:",
        round(mapped_green_pct_of_kmc, 2),
        "%"
    )

    print(
        "Observed green fraction percentiles:",
        quantiles,
    )

    # ====================================================
    # 7. CSV + GEO-PACKAGE EXPORT
    # ====================================================

    print("\n[7] EXPORT CURRENT GREEN DATA")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        CSV_OUTPUT,
        index=False,
    )

    geo = gpd.GeoDataFrame(
        result.copy(),
        geometry=clipped_geometry,
        crs=grid.crs,
    )

    geo.to_file(
        GPKG_OUTPUT,
        layer="current_green_2025_100m",
        driver="GPKG",
    )

    saved_csv = pd.read_csv(CSV_OUTPUT)

    saved_gis_info = pyogrio.read_info(
        GPKG_OUTPUT
    )

    checks["csv_export_9240"] = bool(
        len(saved_csv) == 9240
        and saved_csv.grid_id.is_unique
    )

    checks["gis_export_9240"] = bool(
        saved_gis_info["features"] == 9240
    )

    checks["csv_unknowns_preserved"] = bool(
        saved_csv[
            "current_green_fraction"
        ].isna().sum() == 45
    )

    check(
        "csv_gpkg_export_verified",
        all([
            checks["csv_export_9240"],
            checks["gis_export_9240"],
            checks["csv_unknowns_preserved"],
        ]),
    )

    status = (
        "PASS_CURRENT_GREEN_BASELINE"
        if all(checks.values())
        else "REVIEW_REQUIRED"
    )

    # ====================================================
    # 8. WRITE REPRODUCIBLE QA REPORT
    # ====================================================

    qa = {
        "section": "04_conservation_intelligence",
        "stage": "4.3",
        "status": status,
        "primary_year": 2025,
        "primary_product": "RAW_RF",
        "lulc_green_class": 1,
        "grid_cells": len(result),
        "kmc_area_km2": kmc_area / 1e6,
        "valid_green_condition_cells": int(
            valid.sum()
        ),
        "unknown_green_condition_cells": int(
            unknown.sum()
        ),
        "detected_green_cells": detected_green_count,
        "no_green_detected_cells": (
            no_green_detected_count
        ),
        "mapped_green_area_km2": (
            total_mapped_green_km2
        ),
        "mapped_green_pct_of_kmc": (
            mapped_green_pct_of_kmc
        ),
        "green_fraction_percentiles_valid_cells": (
            quantiles
        ),
        "max_geometry_area_difference_m2": (
            max_geometry_area_difference
        ),
        "study_area_difference_m2": (
            study_area_difference
        ),
        "checks": checks,
        "current_green_definition": (
            "Green classified area divided by valid "
            "classified LULC area within each grid cell"
        ),
        "green_bounds_definition": (
            "Lower bound assumes unmapped area is not "
            "green; upper bound assumes unmapped area "
            "is entirely green. Not confidence limits."
        ),
        "no_threshold_selected": True,
        "no_green_importance_score_yet": True,
        "no_pressure_used": True,
        "no_suitability_used": True,
        "limitations": [
            "Current green condition is mapped LULC, "
            "not independently verified ecological quality.",
            "Incomplete raster coverage introduces "
            "unknown land-cover area.",
            "Zero-valid-coverage cells retain NaN "
            "current green fraction.",
            "No arbitrary conservation candidate "
            "threshold has been applied.",
            "Green Importance and connectivity "
            "are not yet computed.",
            "Historical Green-to-Built transition "
            "codebook will be verified separately "
            "before historical threat processing.",
        ],
    }

    QA_OUTPUT.write_text(
        json.dumps(qa, indent=2),
        encoding="utf-8",
    )

    print("CSV:", CSV_OUTPUT)
    print("GPKG:", GPKG_OUTPUT)
    print("QA:", QA_OUTPUT)

    print("\n" + "=" * 72)
    print("STAGE 4.3:", status)
    print("Grid cells:", len(result))
    print("Current green source: 2025 RAW_RF")
    print("Unknown cells:", int(unknown.sum()))
    print("Conservation candidate threshold: NOT SELECTED")
    print("Green Importance: NOT YET CALCULATED")
    print("Future pressure used: NO")
    print("=" * 72)

    if status != "PASS_CURRENT_GREEN_BASELINE":
        raise SystemExit(
            "Stage 4.3 export verification failed."
        )


if __name__ == "__main__":
    main()
