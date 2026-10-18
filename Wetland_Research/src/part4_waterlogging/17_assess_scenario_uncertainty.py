
from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# PART 4 - SCRIPT 17
# SCENARIO CONSISTENCY AND UNCERTAINTY ASSESSMENT
#
# Reconstruct 8 Script 14 scenarios and the
# exploratory 10% waterway scenario from Script 16.
#
# Outputs:
# - Scenario score range
# - Minimum predicted susceptibility class
# - Maximum predicted susceptibility class
# - Class span across scenarios
# - Screening-wise stability statistics
#
# IMPORTANT:
# This measures model-assumption sensitivity.
# It is NOT prediction accuracy or probability.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data/processed/part4_waterlogging"
METRICS = ROOT / "outputs/metrics/part4_waterlogging"
REPORTS = ROOT / "outputs/reports/part4_waterlogging"

FILES = {
    "mask": "06_Kaduwela_Study_Mask_30m.tif",
    "screen": "12_Landcover_Screening_30m.tif",
    "density": "09_OSM_Building_Density_300m_30m.tif",
    "ged_slope": "07_GEDTM30_Slope_Degrees_30m.tif",
    "ged_tpi": "07_GEDTM30_TPI_300m_30m.tif",
    "nasa_slope": "07_NASADEM_Slope_Degrees_30m.tif",
    "nasa_tpi": "07_NASADEM_TPI_300m_30m.tif",
    "distance":
        "08_OSM_River_Stream_Canal_Distance_30m.tif",
    "reference_score":
        "13_Terrain_Urban_Susceptibility_Score_30m.tif",
    "reference_class":
        "13_Terrain_Urban_Susceptibility_Class_30m.tif",
    "waterway_score":
        "16_Waterway_Exploratory_Score_30m.tif",
    "waterway_class":
        "16_Waterway_Exploratory_Class_30m.tif",
}

SCENARIOS = {
    "Terrain_Only": ("GEDTM30", 0.50, 0.50, 0.00),
    "Terrain_Urban": ("GEDTM30", 0.40, 0.40, 0.20),
    "Urban_10pct": ("GEDTM30", 0.45, 0.45, 0.10),
    "Urban_30pct": ("GEDTM30", 0.35, 0.35, 0.30),
    "Slope_Emphasis": ("GEDTM30", 0.60, 0.20, 0.20),
    "TPI_Emphasis": ("GEDTM30", 0.20, 0.60, 0.20),
    "NASADEM_Terrain_Urban":
        ("NASADEM", 0.40, 0.40, 0.20),
    "NASADEM_Terrain_Only":
        ("NASADEM", 0.50, 0.50, 0.00),
}

SCREEN_NAMES = {
    1: "General_Land",
    2: "Paddy_Agriculture",
    3: "Wetland_Marsh",
    4: "Open_Water",
    5: "Unknown_Landcover",
}

NODATA_FLOAT = -9999.0


def normalize(values, low, high, inverse=False):
    if high <= low:
        raise ValueError("Invalid normalization limits")

    score = np.clip(
        (values - low) / (high - low),
        0.0,
        1.0,
    )

    return 1.0 - score if inverse else score


def classify(score):
    return (
        np.digitize(
            score,
            bins=[0.25, 0.50, 0.75],
            right=False,
        ).astype(np.uint8)
        + 1
    )


def write_raster(path, values, profile, nodata):
    updated = profile.copy()
    updated.update(
        driver="GTiff",
        count=1,
        dtype=str(values.dtype),
        nodata=nodata,
        compress="deflate",
    )

    with rasterio.open(path, "w", **updated) as dst:
        dst.write(values, 1)


def main():

    print("=" * 70)
    print("PART 4 - SCENARIO CONSISTENCY / UNCERTAINTY")
    print("SCRIPT 17")
    print("=" * 70)

    for folder in (DATA, METRICS, REPORTS):
        folder.mkdir(parents=True, exist_ok=True)

    arrays = {}
    expected = None
    profile = None

    # --------------------------------------------------------
    # 1. Read and validate all existing raster inputs
    # --------------------------------------------------------

    for name, filename in FILES.items():

        with rasterio.open(DATA / filename) as src:

            alignment = (
                src.crs,
                src.transform,
                src.shape,
            )

            if expected is None:
                expected = alignment
                profile = src.profile.copy()

            elif alignment != expected:
                raise ValueError(
                    f"Grid alignment mismatch: {name}"
                )

            arrays[name] = src.read(1)

            if name not in ("mask", "screen"):
                selected = arrays[name][
                    arrays["mask"] == 1
                ]

                if (
                    not np.isfinite(selected).all()
                    or (
                        src.nodata is not None
                        and np.any(selected == src.nodata)
                    )
                ):
                    raise ValueError(
                        f"Invalid raster values: {name}"
                    )

    study = arrays["mask"] == 1
    screen = arrays["screen"]

    if int(study.sum()) != 94431:
        raise ValueError("Unexpected study-area count")

    if not np.isin(
        screen[study], list(SCREEN_NAMES)
    ).all():
        raise ValueError("Invalid screening categories")

    if not np.isin(
        arrays["reference_class"][study], [1, 2, 3, 4]
    ).all():
        raise ValueError("Invalid baseline classes")

    if not np.isin(
        arrays["waterway_class"][study], [1, 2, 3, 4]
    ).all():
        raise ValueError("Invalid waterway classes")

    # Work only with valid study-area cells.
    values = {
        name: array[study].astype(np.float64)
        for name, array in arrays.items()
        if name not in ("mask", "screen")
    }

    if np.any(values["distance"] < 0):
        raise ValueError("Negative waterway distances")

    # --------------------------------------------------------
    # 2. Recover exact Script 13 normalization
    # --------------------------------------------------------

    report13_path = (
        REPORTS
        / "script13_initial_susceptibility_model.json"
    )

    report13 = json.loads(
        report13_path.read_text(encoding="utf-8")
    )

    limits = report13["normalization"]

    def factor_score(key, reference_name):

        settings = limits[reference_name]

        inverse = (
            settings["direction"]
            == "lower_higher_susceptibility"
        )

        return normalize(
            values[key],
            settings["reference_p05"],
            settings["reference_p95"],
            inverse=inverse,
        )

    factors = {}

    for source, prefix in [
        ("GEDTM30", "ged"),
        ("NASADEM", "nasa"),
    ]:

        factors[source] = {
            "slope": factor_score(
                prefix + "_slope", "slope"
            ),
            "tpi": factor_score(
                prefix + "_tpi", "tpi"
            ),
            "density": factor_score(
                "density", "density"
            ),
        }

    # --------------------------------------------------------
    # 3. Reconstruct the tested scenarios
    # --------------------------------------------------------

    scores = {}
    scenario_details = []

    for name, (source, ws, wt, wu) in SCENARIOS.items():

        if not np.isclose(ws + wt + wu, 1.0):
            raise ValueError(f"Invalid weights: {name}")

        f = factors[source]

        score = (
            ws * f["slope"]
            + wt * f["tpi"]
            + wu * f["density"]
        )

        scores[name] = score

        scenario_details.append({
            "scenario": name,
            "dem": source,
            "slope_weight": ws,
            "tpi_weight": wt,
            "building_weight": wu,
            "waterway_weight": 0.0,
        })

    reference = scores["Terrain_Urban"]

    # Check against existing Script 13 results.
    error13 = float(np.max(np.abs(
        reference - values["reference_score"]
    )))

    if (
        error13 > 1e-5
        or not np.array_equal(
            classify(reference),
            values["reference_class"].astype(np.uint8)
        )
    ):
        raise ValueError(
            "Script 13 reference reproduction failed"
        )

    print("\nSCRIPT 13 REPRODUCTION: PASS")
    print("Maximum score error:", round(error13, 8))

    # Reconstruct the saved Script 16 waterway scenario.
    proximity = np.exp(
        -values["distance"] / 300.0
    )

    waterway_score = (
        0.90 * reference + 0.10 * proximity
    )

    error16 = float(np.max(np.abs(
        waterway_score - values["waterway_score"]
    )))

    if (
        error16 > 1e-5
        or not np.array_equal(
            classify(waterway_score),
            values["waterway_class"].astype(np.uint8)
        )
    ):
        raise ValueError(
            "Script 16 waterway reproduction failed"
        )

    print("SCRIPT 16 REPRODUCTION: PASS")
    print("Maximum score error:", round(error16, 8))

    scores["Waterway_10pct_300m"] = waterway_score

    scenario_details.append({
        "scenario": "Waterway_10pct_300m",
        "dem": "GEDTM30",
        "slope_weight": 0.36,
        "tpi_weight": 0.36,
        "building_weight": 0.18,
        "waterway_weight": 0.10,
        "waterway_decay_scale_m": 300,
    })

    # --------------------------------------------------------
    # 4. Measure scenario disagreement
    # --------------------------------------------------------

    scenario_names = list(scores)

    score_stack = np.stack(
        [scores[name] for name in scenario_names],
        axis=0,
    )

    if (
        not np.isfinite(score_stack).all()
        or np.any(score_stack < 0)
        or np.any(score_stack > 1)
    ):
        raise ValueError("Invalid scenario scores")

    class_stack = classify(score_stack)

    score_range = (
        score_stack.max(axis=0)
        - score_stack.min(axis=0)
    )

    minimum_class = class_stack.min(axis=0)
    maximum_class = class_stack.max(axis=0)

    class_span = (
        maximum_class.astype(np.int16)
        - minimum_class.astype(np.int16)
    ).astype(np.uint8)

    print("\nSCENARIO COMPARISON")
    print("Scenarios:", len(scenario_names))
    print("Study-area cells:", int(study.sum()))

    # --------------------------------------------------------
    # 5. Summarize disagreement by screening category
    # --------------------------------------------------------

    study_screen = screen[study]
    summary_rows = []

    groups = {
        "All_Study_Area":
            np.ones(len(study_screen), dtype=bool),
    }

    for code, name in SCREEN_NAMES.items():
        groups[name] = study_screen == code

    for name, selected in groups.items():

        count = int(selected.sum())

        if count == 0:
            raise ValueError(
                f"Empty screening category: {name}"
            )

        spans = class_span[selected]
        ranges = score_range[selected]

        row = {
            "category": name,
            "cells": count,
            "stable_same_class_percent":
                round(100 * np.mean(spans == 0), 4),
            "adjacent_class_span_percent":
                round(100 * np.mean(spans == 1), 4),
            "two_or_more_class_span_percent":
                round(100 * np.mean(spans >= 2), 4),
            "crosses_moderate_high_boundary_percent":
                round(
                    100 * np.mean(
                        (minimum_class[selected] <= 2)
                        & (maximum_class[selected] >= 3)
                    ),
                    4,
                ),
            "mean_score_range":
                round(float(ranges.mean()), 6),
            "p95_score_range":
                round(
                    float(np.percentile(ranges, 95)),
                    6,
                ),
            "max_score_range":
                round(float(ranges.max()), 6),
        }

        summary_rows.append(row)

        print("\nCATEGORY:", name)
        print("Cells:", count)
        print(
            "Same class in every scenario:",
            row["stable_same_class_percent"], "%"
        )
        print(
            "Class span >= 2:",
            row["two_or_more_class_span_percent"], "%"
        )
        print(
            "Mean score range:",
            row["mean_score_range"]
        )

    # --------------------------------------------------------
    # 6. Save uncertainty and class-range rasters
    # --------------------------------------------------------

    raster_outputs = {}

    definitions = {
        "Score_Range": (
            score_range, np.float32, NODATA_FLOAT
        ),
        "Minimum_Class": (
            minimum_class, np.uint8, 0
        ),
        "Maximum_Class": (
            maximum_class, np.uint8, 0
        ),
        "Class_Span": (
            class_span, np.uint8, 255
        ),
    }

    for name, (values_1d, dtype, nodata) in definitions.items():

        output = np.full(
            arrays["mask"].shape,
            nodata,
            dtype=dtype,
        )

        output[study] = values_1d.astype(dtype)

        path = (
            DATA / f"17_Scenario_{name}_30m.tif"
        )

        write_raster(
            path, output, profile, nodata
        )

        raster_outputs[name] = str(
            path.relative_to(ROOT)
        )

    # --------------------------------------------------------
    # 7. Save permanent methodology and statistics
    # --------------------------------------------------------

    metrics_path = (
        METRICS
        / "script17_scenario_uncertainty_summary.csv"
    )

    pd.DataFrame(summary_rows).to_csv(
        metrics_path,
        index=False,
    )

    roster_path = (
        METRICS
        / "script17_scenario_roster.csv"
    )

    pd.DataFrame(scenario_details).to_csv(
        roster_path,
        index=False,
    )

    report_path = (
        REPORTS
        / "script17_scenario_uncertainty.json"
    )

    report = {
        "component": "Part 4 - Seasonal Waterlogging",
        "stage": "Scenario consistency assessment",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "scenario_count": len(scenario_names),
        "scenario_names": scenario_names,
        "reference_model": "Terrain_Urban",
        "script13_reproduction_max_error": error13,
        "script16_reproduction_max_error": error16,
        "study_cells": int(study.sum()),
        "method": {
            "score_range":
                "Maximum minus minimum susceptibility "
                "score across the tested scenarios.",
            "minimum_class":
                "Lowest categorical prediction "
                "across the tested scenarios.",
            "maximum_class":
                "Highest categorical prediction "
                "across the tested scenarios.",
            "class_span":
                "Maximum class minus minimum class. "
                "Zero means identical categorical "
                "predictions across all scenarios.",
            "class_codes": {
                "1": "Low",
                "2": "Moderate",
                "3": "High",
                "4": "Very_High",
            },
        },
        "screening_summary": summary_rows,
        "scenario_settings": scenario_details,
        "output_rasters": raster_outputs,
        "metrics_csv":
            str(metrics_path.relative_to(ROOT)),
        "roster_csv":
            str(roster_path.relative_to(ROOT)),
        "limitations": [
            "Scenario disagreement measures "
            "sensitivity to tested assumptions.",
            "This is not a calibrated confidence "
            "interval or flooding probability.",
            "Scenarios are related variants of "
            "the same weighted-overlay framework, "
            "not independent trained models.",
            "The scenario set is intentionally "
            "exploratory and not an exhaustive "
            "sample of possible model structures.",
            "Stable classification does not "
            "establish prediction correctness.",
            "DEM quality, wetland screening, "
            "historical evidence interpretation "
            "and unmapped drainage remain "
            "important sources of uncertainty.",
            "The results are not independent "
            "predictive validation.",
            "No final model is selected.",
        ],
        "raw_data_modified": False,
        "status": "SCENARIO_CONSISTENCY_ASSESSED",
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("SCRIPT 17 - SUMMARY")
    print("=" * 70)

    print("Scenarios assessed:", len(scenario_names))
    print("Rasters generated:", len(raster_outputs))

    for filename in raster_outputs.values():
        print("Saved:", ROOT / filename)

    print("Saved:", metrics_path)
    print("Saved:", roster_path)
    print("Saved:", report_path)

    print("\nSCENARIO UNCERTAINTY: ASSESSED")
    print("FINAL MODEL SELECTION: PENDING")
    print("56-GN SUMMARIES: PENDING")
    print("RAW SOURCE DATA MODIFIED: NO")


if __name__ == "__main__":
    main()
