
from pathlib import Path
from datetime import datetime, timezone
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


# ============================================================
# PART 4 - SCRIPT 19
# FINAL BACKEND INTEGRITY AND SCIENTIFIC CLAIMS AUDIT
#
# Read-only validation of completed analysis outputs.
# Writes only a new audit report if all checks pass.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/part4_waterlogging"
METRICS = ROOT / "outputs/metrics/part4_waterlogging"
REPORTS = ROOT / "outputs/reports/part4_waterlogging"

RASTERS = {
    "study": "06_Kaduwela_Study_Mask_30m.tif",
    "screen": "12_Landcover_Screening_30m.tif",
    "score": "13_Terrain_Urban_Susceptibility_Score_30m.tif",
    "class": "13_Terrain_Urban_Susceptibility_Class_30m.tif",
    "waterway_class": "16_Waterway_Exploratory_Class_30m.tif",
    "score_range": "17_Scenario_Score_Range_30m.tif",
    "minimum": "17_Scenario_Minimum_Class_30m.tif",
    "maximum": "17_Scenario_Maximum_Class_30m.tif",
    "span": "17_Scenario_Class_Span_30m.tif",
}

EXPECTED_SCREEN_COUNTS = {
    1: 51707,
    2: 25386,
    3: 12181,
    4: 4509,
    5: 648,
}

EXPECTED_HIGH_CELLS = 28610


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(filename):
    return json.loads(
        (REPORTS / filename).read_text(encoding="utf-8")
    )


def main():
    print("=" * 70)
    print("PART 4 - FINAL BACKEND INTEGRITY AUDIT")
    print("SCRIPT 19")
    print("=" * 70)

    arrays = {}
    expected_grid = None
    profile = None

    for key, filename in RASTERS.items():
        with rasterio.open(DATA / filename) as src:
            grid = (src.crs, src.transform, src.shape)

            if expected_grid is None:
                expected_grid = grid
                profile = src.profile.copy()
            else:
                require(
                    grid == expected_grid,
                    f"Grid mismatch: {key}",
                )

            arrays[key] = src.read(1)

    study = arrays["study"] == 1
    general = study & (arrays["screen"] == 1)

    require(int(study.sum()) == 94431, "Study count mismatch")
    require(int(general.sum()) == 51707, "General-land count mismatch")
    require(
        str(profile["crs"]) == "EPSG:32644",
        "Unexpected analysis CRS",
    )

    for code, count in EXPECTED_SCREEN_COUNTS.items():
        observed = int(
            np.count_nonzero(
                study & (arrays["screen"] == code)
            )
        )
        require(
            observed == count,
            f"Screening count mismatch: code {code}",
        )

    print("\n1. COMMON GRID AND LAND-COVER SCREENING: PASS")

    # --------------------------------------------------------
    # Raster values and consistency
    # --------------------------------------------------------

    for name in ("class", "waterway_class", "minimum", "maximum"):
        require(
            np.isin(arrays[name][study], [1, 2, 3, 4]).all(),
            f"Invalid classes: {name}",
        )

    for name in ("score", "score_range"):
        values = arrays[name][study]
        require(
            np.isfinite(values).all()
            and (values >= 0).all()
            and (values <= 1).all(),
            f"Invalid scores: {name}",
        )

    minimum = arrays["minimum"][study].astype(int)
    maximum = arrays["maximum"][study].astype(int)
    span = arrays["span"][study].astype(int)
    reference = arrays["class"][study].astype(int)

    require(
        np.array_equal(maximum - minimum, span),
        "Scenario class-span mismatch",
    )

    require(
        ((reference >= minimum) & (reference <= maximum)).all(),
        "Reference class outside scenario range",
    )

    high_cells = int(
        np.count_nonzero(
            general & (arrays["class"] >= 3)
        )
    )

    require(
        high_cells == EXPECTED_HIGH_CELLS,
        "General-land High/Very High mismatch",
    )

    print("2. SUSCEPTIBILITY AND UNCERTAINTY RASTERS: PASS")

    # --------------------------------------------------------
    # GN summary and detailed class reporting
    # --------------------------------------------------------

    summary = pd.read_csv(
        METRICS / "script18_gn_susceptibility_summary.csv",
        dtype={"gn_number": str, "admin_code": str},
    )

    breakdown = pd.read_csv(
        METRICS / "script18_gn_class_breakdown.csv",
        dtype={"gn_number": str, "admin_code": str},
    )

    require(len(summary) == 56, "Incorrect GN summary rows")
    require(len(breakdown) == 224, "Incorrect GN breakdown rows")

    for col in ("gn_name", "gn_number", "admin_code"):
        require(
            summary[col].notna().all()
            and summary[col].nunique() == 56,
            f"GN identifiers invalid: {col}",
        )

    require(
        not summary.isna().any().any(),
        "GN summary contains missing values",
    )

    require(
        int(summary["total_cells"].sum()) == 94431,
        "GN study-area total mismatch",
    )

    require(
        int(summary["general_cells"].sum()) == 51707,
        "GN general-land total mismatch",
    )

    require(
        int(summary["high_very_high_cells"].sum())
        == EXPECTED_HIGH_CELLS,
        "GN High/Very High total mismatch",
    )

    class_names = ["low", "moderate", "high", "very_high"]

    class_sum = summary[
        [f"{name}_cells" for name in class_names]
    ].sum(axis=1)

    require(
        (class_sum == summary["general_cells"]).all(),
        "GN classification total mismatch",
    )

    grouped = (
        breakdown.groupby("admin_code")[
            "general_land_class_cells"
        ].sum()
    )

    for record in summary.itertuples():
        require(
            int(grouped.loc[record.admin_code])
            == int(record.general_cells),
            f"Breakdown mismatch: {record.gn_name}",
        )

    print("3. ALL 56 GN CSV REPORTS: PASS")

    # --------------------------------------------------------
    # GeoPackage consistency
    # --------------------------------------------------------

    gpkg = gpd.read_file(
        DATA / "18_GN_Waterlogging_Susceptibility.gpkg",
        layer="gn_susceptibility",
    )

    require(len(gpkg) == 56, "GeoPackage GN count mismatch")
    require(gpkg.crs == profile["crs"], "GeoPackage CRS mismatch")
    require(
        gpkg.geometry.notna().all()
        and (~gpkg.geometry.is_empty).all()
        and gpkg.is_valid.all(),
        "Invalid GN GeoPackage geometry",
    )

    require(
        gpkg["admin_code"].astype(str).nunique() == 56,
        "GeoPackage identifiers not unique",
    )

    merged = summary.merge(
        gpkg.drop(columns="geometry"),
        on="admin_code",
        suffixes=("_csv", "_gpkg"),
        validate="one_to_one",
    )

    require(len(merged) == 56, "GN join mismatch")

    for column in (
        "total_cells",
        "general_cells",
        "high_very_high_cells",
    ):
        require(
            np.array_equal(
                merged[f"{column}_csv"].to_numpy(),
                merged[f"{column}_gpkg"].to_numpy(),
            ),
            f"GeoPackage attribute mismatch: {column}",
        )

    print("4. GN GEOPACKAGE ATTRIBUTES: PASS")

    # --------------------------------------------------------
    # Existing scientific evidence reports
    # --------------------------------------------------------

    report14 = read_json(
        "script14_weight_dem_sensitivity.json"
    )
    report15 = read_json(
        "script15_historical_evidence_evaluation.json"
    )
    report16 = read_json(
        "script16_waterway_proximity_sensitivity.json"
    )
    report17 = read_json(
        "script17_scenario_uncertainty.json"
    )
    report18 = read_json(
        "script18_gn_susceptibility_report.json"
    )

    require(
        len(report14["scenario_results"]) == 8,
        "Script 14 scenario count mismatch",
    )

    require(
        len(report15["event_dates"]) == 2,
        "Script 15 event count mismatch",
    )

    require(
        len(report16["scenarios"]) == 4,
        "Script 16 scenario count mismatch",
    )

    require(
        report17["scenario_count"] == 9,
        "Script 17 scenario count mismatch",
    )

    require(
        report18["gn_count"] == 56,
        "Script 18 GN count mismatch",
    )

    print("5. METHODOLOGY AND EVIDENCE REPORTS: PASS")

    # --------------------------------------------------------
    # Verify general-land uncertainty totals
    # --------------------------------------------------------

    general_minimum = arrays["minimum"][general]
    general_maximum = arrays["maximum"][general]
    general_span = arrays["span"][general]

    stable_pct = float(
        100 * np.mean(general_span == 0)
    )

    crossing_pct = float(
        100 * np.mean(
            (general_minimum <= 2)
            & (general_maximum >= 3)
        )
    )

    require(
        abs(stable_pct - 18.9123) < 0.001,
        "General-land stability mismatch",
    )

    require(
        abs(crossing_pct - 48.8831) < 0.001,
        "Moderate/High boundary mismatch",
    )

    print("6. SCENARIO UNCERTAINTY RECONCILIATION: PASS")

    # --------------------------------------------------------
    # Save audit record
    # --------------------------------------------------------

    audit = {
        "component": "Part 4 - Seasonal Waterlogging",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "status": "INTERNAL_INTEGRITY_CHECKS_PASSED",
        "checks": {
            "raster_grid": "PASS",
            "landcover_screening": "PASS",
            "susceptibility_rasters": "PASS",
            "uncertainty_rasters": "PASS",
            "gn_csvs": "PASS",
            "gn_geopackage": "PASS",
            "methodology_reports": "PASS",
            "uncertainty_reconciliation": "PASS",
        },
        "study_cells": int(study.sum()),
        "general_land_cells": int(general.sum()),
        "gn_divisions": len(summary),
        "reference_model_high_veryhigh_cells": high_cells,
        "general_land_stable_class_percent":
            round(stable_pct, 4),
        "general_land_boundary_cross_percent":
            round(crossing_pct, 4),
        "reference_model":
            "GEDTM30 Terrain_Urban, provisional",
        "scientific_status": {
            "supervised_ml_trained": False,
            "independent_predictive_validation": False,
            "season_specific_prediction": False,
            "historical_event_comparison":
                "Exploratory supporting evidence only",
            "uncertainty_interpretation":
                "Scenario sensitivity, not confidence",
            "planning_interpretation":
                "Relative susceptibility, not flood probability",
        },
        "known_limitations": [
            "Exploratory GIS weighted-overlay model",
            "Provisional factor weights and class thresholds",
            "Strong sensitivity to DEM and factor assumptions",
            "Only two manually interpreted 2018 event dates",
            "Substantial spatial overlap of event footprints",
            "2020 land-cover and 2026 OSM temporal mismatch",
            "No independent predictive accuracy established",
            "No seasonal timing or flood depth prediction",
        ],
        "manual_checks_pending": [
            "Verify GN labels against authoritative source",
            "Review map appearance and GIS symbology in QGIS",
            "Finalize reference scenario and claims",
            "Review research-methodology documentation",
        ],
        "source_data_written": False,
    }

    REPORTS.mkdir(parents=True, exist_ok=True)

    report_path = (
        REPORTS / "script19_backend_integrity_audit.json"
    )

    report_path.write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("SCRIPT 19 - SUMMARY")
    print("=" * 70)
    print("Internal integrity checks: PASSED")
    print("GN divisions verified:", len(summary))
    print("Reference High/Very High cells:", high_cells)
    print("Stable general-land classes:", round(stable_pct, 4), "%")
    print("Moderate/High boundary crossing:",
          round(crossing_pct, 4), "%")
    print("Saved:", report_path)

    print("\nINTERNAL BACKEND INTEGRITY: PASS")
    print("INDEPENDENT VALIDATION: NOT ESTABLISHED")
    print("FINAL SCIENTIFIC APPROVAL: PENDING")
    print("SOURCE DATA MODIFIED: NO")


if __name__ == "__main__":
    main()
