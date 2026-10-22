
from pathlib import Path
from datetime import datetime, timezone
import json
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize


# ============================================================
# PART 4 - SCRIPT 18
# GN-WISE SUSCEPTIBILITY AND UNCERTAINTY REPORTING
#
# Reference: Script 13 Terrain + Urban
# Sensitivity: Script 17 nine-scenario assessment
#
# All percentages use GENERAL LAND as denominator
# unless explicitly identified otherwise.
#
# Area values are approximate raster-cell areas.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data/processed/part4_waterlogging"
METRICS = ROOT / "outputs/metrics/part4_waterlogging"
REPORTS = ROOT / "outputs/reports/part4_waterlogging"

GN_FILE = ROOT / "data/raw/boundaries/Kaduwela TO Area.kml"
GN_LAYER = "GN Boundary Colombo District (#2)"

RASTERS = {
    "mask": "06_Kaduwela_Study_Mask_30m.tif",
    "screen": "12_Landcover_Screening_30m.tif",
    "score": (
        "13_Terrain_Urban_Susceptibility_Score_30m.tif"
    ),
    "class": (
        "13_Terrain_Urban_Susceptibility_Class_30m.tif"
    ),
    "score_range": (
        "17_Scenario_Score_Range_30m.tif"
    ),
    "minimum_class": (
        "17_Scenario_Minimum_Class_30m.tif"
    ),
    "maximum_class": (
        "17_Scenario_Maximum_Class_30m.tif"
    ),
    "class_span": (
        "17_Scenario_Class_Span_30m.tif"
    ),
}

SCREEN_NAMES = {
    1: "general",
    2: "paddy",
    3: "wetland",
    4: "open_water",
    5: "unknown",
}

CLASS_NAMES = {
    1: "low",
    2: "moderate",
    3: "high",
    4: "very_high",
}


def extract_code(description, field):

    pattern = (
        rf"<td[^>]*>\s*{re.escape(field)}\s*</td>"
        rf"\s*<td[^>]*>\s*([^<]+)\s*</td>"
    )

    match = re.search(
        pattern,
        str(description),
        flags=re.IGNORECASE,
    )

    return match.group(1).strip() if match else None


def percentage(numerator, denominator):

    if denominator == 0:
        return None

    return round(
        100.0 * numerator / denominator,
        4,
    )


def main():

    print("=" * 70)
    print("PART 4 - GN-WISE SUSCEPTIBILITY REPORTING")
    print("SCRIPT 18")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Read aligned rasters
    # --------------------------------------------------------

    arrays = {}
    expected = None
    raster_profile = None

    for name, filename in RASTERS.items():

        with rasterio.open(DATA / filename) as src:

            alignment = (
                src.crs,
                src.transform,
                src.shape,
            )

            if expected is None:
                expected = alignment
                raster_profile = src.profile.copy()

            elif expected != alignment:
                raise ValueError(
                    f"Raster alignment mismatch: {name}"
                )

            arrays[name] = src.read(1)

    study = arrays["mask"] == 1
    screen = arrays["screen"]

    if int(study.sum()) != 94431:
        raise ValueError("Unexpected study cell count")

    if not np.isin(
        screen[study], list(SCREEN_NAMES)
    ).all():
        raise ValueError("Invalid screening categories")

    if not np.isin(
        arrays["class"][study], [1, 2, 3, 4]
    ).all():
        raise ValueError("Invalid susceptibility classes")

    if not np.isin(
        arrays["minimum_class"][study], [1, 2, 3, 4]
    ).all():
        raise ValueError("Invalid minimum classes")

    if not np.isin(
        arrays["maximum_class"][study], [1, 2, 3, 4]
    ).all():
        raise ValueError("Invalid maximum classes")

    for key in ("score", "score_range"):

        values = arrays[key][study]

        if (
            not np.isfinite(values).all()
            or np.any(values < 0)
            or np.any(values > 1)
        ):
            raise ValueError(
                f"Invalid study-area values: {key}"
            )

    minimum = arrays["minimum_class"][study].astype(
        np.int16
    )

    maximum = arrays["maximum_class"][study].astype(
        np.int16
    )

    span = arrays["class_span"][study].astype(
        np.int16
    )

    if not np.array_equal(maximum - minimum, span):
        raise ValueError(
            "Script 17 class-span inconsistency"
        )

    # --------------------------------------------------------
    # 2. Read GN boundaries and identifiers
    # --------------------------------------------------------

    gn = gpd.read_file(
        GN_FILE,
        layer=GN_LAYER,
    )

    if len(gn) != 56:
        raise ValueError("Expected exactly 56 GNs")

    gn = gn.to_crs(raster_profile["crs"]).reset_index(
        drop=True
    )

    if (
        gn.geometry.is_empty.any()
        or gn.geometry.isna().any()
        or (~gn.is_valid).any()
    ):
        raise ValueError("Invalid GN geometry found")

    gn["gn_name"] = gn["description"].apply(
        lambda value: extract_code(value, "GND_N")
    )

    gn["gn_number"] = gn["description"].apply(
        lambda value: extract_code(value, "GND_NO")
    )

    gn["admin_code"] = gn["description"].apply(
        lambda value: extract_code(value, "Admin_Code")
    )

    for field in ("gn_name", "gn_number", "admin_code"):

        if gn[field].isna().any():
            raise ValueError(
                f"Missing GN identifier: {field}"
            )

        if gn[field].nunique() != 56:
            raise ValueError(
                f"Nonunique GN identifier: {field}"
            )

    print("\nGN identifier validation: PASS")
    print("GN divisions:", len(gn))

    # --------------------------------------------------------
    # 3. Build GN partition and validate cell coverage
    # --------------------------------------------------------

    grid_shape = study.shape
    transform = raster_profile["transform"]

    gn_arrays = []
    coverage = np.zeros(
        grid_shape,
        dtype=np.uint16,
    )

    for geometry in gn.geometry:

        gn_mask = rasterize(
            [(geometry, 1)],
            out_shape=grid_shape,
            transform=transform,
            fill=0,
            all_touched=False,
            dtype="uint8",
        ).astype(bool)

        coverage += gn_mask.astype(np.uint16)

        gn_arrays.append(gn_mask & study)

    missing = int(
        np.count_nonzero(study & (coverage == 0))
    )

    duplicate = int(
        np.count_nonzero(study & (coverage > 1))
    )

    outside = int(
        np.count_nonzero(~study & (coverage > 0))
    )

    if missing or duplicate or outside:
        raise ValueError(
            "GN coverage validation failed: "
            f"missing={missing}, "
            f"duplicate={duplicate}, "
            f"outside={outside}"
        )

    print("GN raster partition: PASS")
    print("Missing cells:", missing)
    print("Duplicate cells:", duplicate)

    # Grid is projected in metres.
    cell_area_ha = abs(
        transform.a * transform.e
    ) / 10000.0

    # --------------------------------------------------------
    # 4. Calculate GN-level statistics
    # --------------------------------------------------------

    summary_rows = []
    breakdown_rows = []

    for index, record in gn.iterrows():

        region = gn_arrays[index]

        total_cells = int(region.sum())

        if total_cells == 0:
            raise ValueError(
                f"GN has zero raster cells: {record['Name']}"
            )

        row = {
            "gn_name": str(record["gn_name"]),
            "gn_number": str(record["gn_number"]),
            "admin_code": str(record["admin_code"]),
            "total_cells": total_cells,
            "total_area_ha": round(
                total_cells * cell_area_ha, 4
            ),
        }

        # Land-cover screening contexts
        for code, name in SCREEN_NAMES.items():

            count = int(
                np.count_nonzero(
                    region & (screen == code)
                )
            )

            row[f"{name}_cells"] = count
            row[f"{name}_area_ha"] = round(
                count * cell_area_ha, 4
            )

        if sum(
            row[f"{name}_cells"]
            for name in SCREEN_NAMES.values()
        ) != total_cells:
            raise ValueError(
                "Screening category total mismatch"
            )

        general_region = region & (screen == 1)
        general_count = int(general_region.sum())

        # Calculate susceptibility on general land only.
        if general_count > 0:

            scores = arrays["score"][general_region]

            classes = arrays["class"][general_region]

            min_classes = arrays[
                "minimum_class"
            ][general_region]

            max_classes = arrays[
                "maximum_class"
            ][general_region]

            spans = arrays[
                "class_span"
            ][general_region]

            score_ranges = arrays[
                "score_range"
            ][general_region]

            row["mean_reference_score"] = round(
                float(np.mean(scores)), 6
            )

            row["mean_scenario_score_range"] = round(
                float(np.mean(score_ranges)), 6
            )

            same_class_count = int(
                np.count_nonzero(spans == 0)
            )

            large_span_count = int(
                np.count_nonzero(spans >= 2)
            )

            crossing_count = int(
                np.count_nonzero(
                    (min_classes <= 2)
                    & (max_classes >= 3)
                )
            )

            row["same_class_all_scenarios_percent"] = (
                percentage(
                    same_class_count, general_count
                )
            )

            row["class_span_ge_2_percent"] = (
                percentage(
                    large_span_count, general_count
                )
            )

            row["moderate_high_boundary_cross_percent"] = (
                percentage(
                    crossing_count, general_count
                )
            )

            for class_id, class_name in CLASS_NAMES.items():

                class_count = int(
                    np.count_nonzero(
                        classes == class_id
                    )
                )

                row[f"{class_name}_cells"] = class_count

                row[f"{class_name}_area_ha"] = round(
                    class_count * cell_area_ha, 4
                )

                row[f"{class_name}_percent"] = percentage(
                    class_count,
                    general_count,
                )

                breakdown_rows.append({
                    "gn_name": row["gn_name"],
                    "gn_number": row["gn_number"],
                    "admin_code": row["admin_code"],
                    "susceptibility_class": class_name,
                    "general_land_class_cells":
                        class_count,
                    "area_ha": round(
                        class_count * cell_area_ha, 4
                    ),
                    "percent_of_gn_general_land": (
                        percentage(
                            class_count, general_count
                        )
                    ),
                })

            high_count = (
                row["high_cells"]
                + row["very_high_cells"]
            )

            row["high_very_high_cells"] = high_count

            row["high_very_high_area_ha"] = round(
                high_count * cell_area_ha, 4
            )

            row["high_very_high_percent"] = percentage(
                high_count, general_count
            )

        else:

            # No general-land cells:
            # percentages/means cannot be calculated.
            for key in (
                "mean_reference_score",
                "mean_scenario_score_range",
                "same_class_all_scenarios_percent",
                "class_span_ge_2_percent",
                "moderate_high_boundary_cross_percent",
                "high_very_high_percent",
            ):
                row[key] = None

            row["high_very_high_cells"] = 0
            row["high_very_high_area_ha"] = 0.0

            for class_id, class_name in CLASS_NAMES.items():

                row[f"{class_name}_cells"] = 0
                row[f"{class_name}_area_ha"] = 0.0
                row[f"{class_name}_percent"] = None

                breakdown_rows.append({
                    "gn_name": row["gn_name"],
                    "gn_number": row["gn_number"],
                    "admin_code": row["admin_code"],
                    "susceptibility_class": class_name,
                    "general_land_class_cells": 0,
                    "area_ha": 0.0,
                    "percent_of_gn_general_land": None,
                })

        if sum(
            row[f"{name}_cells"]
            for name in CLASS_NAMES.values()
        ) != general_count:
            raise ValueError(
                f"General-land class mismatch: "
                f"{record['Name']}"
            )

        summary_rows.append(row)

    # --------------------------------------------------------
    # 5. Verify city-wide totals
    # --------------------------------------------------------

    summary = pd.DataFrame(summary_rows)

    if len(summary) != 56:
        raise ValueError("GN output count mismatch")

    if int(summary["total_cells"].sum()) != 94431:
        raise ValueError(
            "Study-area reconciliation failed"
        )

    if int(summary["general_cells"].sum()) != 51707:
        raise ValueError(
            "General-land reconciliation failed"
        )

    for code, name in SCREEN_NAMES.items():

        expected_count = int(
            np.count_nonzero(study & (screen == code))
        )

        actual_count = int(
            summary[f"{name}_cells"].sum()
        )

        if actual_count != expected_count:
            raise ValueError(
                f"Land-cover total mismatch: {name}"
            )

    expected_high = int(
        np.count_nonzero(
            study
            & (screen == 1)
            & (arrays["class"] >= 3)
        )
    )

    actual_high = int(
        summary["high_very_high_cells"].sum()
    )

    if actual_high != expected_high:
        raise ValueError(
            "High/Very High total mismatch"
        )

    print("\nCITY-WIDE RECONCILIATION: PASS")
    print("Total GNs:", len(summary))
    print("Total study cells:", int(
        summary["total_cells"].sum()
    ))
    print("General-land cells:", int(
        summary["general_cells"].sum()
    ))
    print("General-land High/Very High:", actual_high)

    # --------------------------------------------------------
    # 6. Save CSV outputs
    # --------------------------------------------------------

    METRICS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    summary_path = (
        METRICS
        / "script18_gn_susceptibility_summary.csv"
    )

    breakdown_path = (
        METRICS
        / "script18_gn_class_breakdown.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    pd.DataFrame(breakdown_rows).to_csv(
        breakdown_path,
        index=False,
    )

    # --------------------------------------------------------
    # 7. Save GN GeoPackage for QGIS
    # --------------------------------------------------------

    gn_output = gn[
        ["gn_name", "gn_number", "admin_code", "geometry"]
    ].copy()

    for column in summary.columns:

        if column in (
            "gn_name", "gn_number", "admin_code"
        ):
            continue

        gn_output[column] = summary[column].values

    gpkg_path = (
        DATA
        / "18_GN_Waterlogging_Susceptibility.gpkg"
    )

    gn_output.to_file(
        gpkg_path,
        layer="gn_susceptibility",
        driver="GPKG",
        index=False,
    )

    # --------------------------------------------------------
    # 8. Save methodology report
    # --------------------------------------------------------

    report_path = (
        REPORTS
        / "script18_gn_susceptibility_report.json"
    )

    report = {
        "component":
            "Part 4 - Seasonal Waterlogging",
        "stage":
            "GN-wise exploratory susceptibility reporting",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "gn_count": 56,
        "reference_model":
            "Script 13 Terrain_Urban (provisional)",
        "scenario_uncertainty_source":
            "Script 17 nine-scenario comparison",
        "study_cells": 94431,
        "general_land_cells": 51707,
        "raster_cell_area_ha": cell_area_ha,
        "area_method":
            "Count 30m grid-cell centres allocated "
            "to exactly one GN and multiply by "
            "pixel area; values are approximate.",
        "coverage_validation": {
            "missing_cells": missing,
            "multiply_assigned_cells": duplicate,
            "outside_study_cells": outside,
        },
        "general_land_high_veryhigh_cells":
            actual_high,
        "output_files": {
            "summary_csv":
                str(summary_path.relative_to(ROOT)),
            "breakdown_csv":
                str(breakdown_path.relative_to(ROOT)),
            "gn_geopackage":
                str(gpkg_path.relative_to(ROOT)),
        },
        "interpretation": (
            "GN statistics describe relative "
            "susceptibility indices and scenario "
            "sensitivity, not observed flood frequency "
            "or waterlogging probabilities."
        ),
        "limitations": [
            "Reference scenario is provisional.",
            "Land-cover screening is based on 2020 "
            "classification, not current ground truth.",
            "Building density is derived from "
            "2026 OSM mapping.",
            "Model-sensitivity measures are not "
            "statistical confidence intervals.",
            "Raster-cell areas approximate "
            "GN polygon areas.",
            "Paddy, wetland and open-water areas "
            "are reported separately from general "
            "land rather than treated as zero risk.",
            "No independent predictive accuracy "
            "is claimed.",
            "Seasonal timing and flood depth "
            "are not predicted.",
        ],
        "raw_data_modified": False,
        "status":
            "GN_SUSCEPTIBILITY_REPORTING_COMPLETE",
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 9. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SCRIPT 18 - SUMMARY")
    print("=" * 70)

    print("GN divisions reported:", len(summary))
    print("GN polygon coverage: PASS")
    print("General-land class reconciliation: PASS")

    print("\nSaved:", summary_path)
    print("Saved:", breakdown_path)
    print("Saved:", gpkg_path)
    print("Saved:", report_path)

    print("\nGN REPORTING: COMPLETE")
    print("FINAL SCIENTIFIC REVIEW: PENDING")
    print("RAW DATA MODIFIED: NO")


if __name__ == "__main__":
    main()
