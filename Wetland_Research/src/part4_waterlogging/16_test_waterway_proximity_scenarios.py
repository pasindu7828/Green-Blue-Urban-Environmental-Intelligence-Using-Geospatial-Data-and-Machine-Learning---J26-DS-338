
from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# PART 4 - SCRIPT 16
# WATERWAY PROXIMITY SCENARIO SENSITIVITY
#
# Explore possible waterway-related exposure.
# This is NOT evidence of drainage failure,
# fluvial flooding, or confirmed waterlogging.
#
# No historical-event labels used for weight selection.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data/processed/part4_waterlogging"
METRICS = ROOT / "outputs/metrics/part4_waterlogging"
REPORTS = ROOT / "outputs/reports/part4_waterlogging"

FILES = {
    "mask": "06_Kaduwela_Study_Mask_30m.tif",
    "screen": "12_Landcover_Screening_30m.tif",
    "slope": "07_GEDTM30_Slope_Degrees_30m.tif",
    "tpi": "07_GEDTM30_TPI_300m_30m.tif",
    "density": "09_OSM_Building_Density_300m_30m.tif",
    "distance":
        "08_OSM_River_Stream_Canal_Distance_30m.tif",
    "baseline_score":
        "13_Terrain_Urban_Susceptibility_Score_30m.tif",
    "baseline_class":
        "13_Terrain_Urban_Susceptibility_Class_30m.tif",
}

BASELINE_REPORT = (
    REPORTS / "script13_initial_susceptibility_model.json"
)

SCENARIOS = {
    "Waterway_10pct_300m": (0.10, 300.0),
    "Waterway_20pct_300m": (0.20, 300.0),
    "Waterway_10pct_150m": (0.10, 150.0),
    "Waterway_10pct_600m": (0.10, 600.0),
}

NODATA = -9999.0


def normalize(values, low, high, inverse=False):
    if high <= low:
        raise ValueError("Invalid normalization limits")

    result = np.clip(
        (values - low) / (high - low),
        0.0,
        1.0,
    )

    return 1.0 - result if inverse else result


def classify(values):
    return (
        np.digitize(
            values,
            [0.25, 0.50, 0.75],
            right=False,
        ).astype(np.uint8)
        + 1
    )


def save_raster(path, values, profile, categorical=False):
    new_profile = profile.copy()

    new_profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8" if categorical else "float32",
        nodata=0 if categorical else NODATA,
        compress="deflate",
    )

    with rasterio.open(path, "w", **new_profile) as dst:
        dst.write(values, 1)


def main():

    print("=" * 70)
    print("PART 4 - WATERWAY PROXIMITY SENSITIVITY")
    print("SCRIPT 16")
    print("=" * 70)

    METRICS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    arrays = {}
    expected = None
    profile = None

    for key, filename in FILES.items():

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
                    f"Grid mismatch: {key}"
                )

            arrays[key] = src.read(1)

    study = arrays["mask"] == 1
    general = study & (arrays["screen"] == 1)

    if (
        int(study.sum()) != 94431
        or int(general.sum()) != 51707
    ):
        raise ValueError("Unexpected study-area counts")

    for key in [
        "slope",
        "tpi",
        "density",
        "distance",
        "baseline_score",
    ]:
        values = arrays[key][study]

        if (
            not np.isfinite(values).all()
            or np.any(values == NODATA)
        ):
            raise ValueError(
                f"Invalid study-area raster: {key}"
            )

    distance = arrays["distance"].astype(np.float64)

    if np.any(distance[study] < 0):
        raise ValueError("Negative waterway distances")

    # --------------------------------------------------------
    # 1. Reuse the original model normalization
    # --------------------------------------------------------

    baseline_report = json.loads(
        BASELINE_REPORT.read_text(encoding="utf-8")
    )

    normalization = baseline_report["normalization"]

    factors = {}

    for name in ["slope", "tpi", "density"]:

        settings = normalization[name]

        inverse = (
            settings["direction"]
            == "lower_higher_susceptibility"
        )

        factors[name] = normalize(
            arrays[name].astype(np.float64),
            settings["reference_p05"],
            settings["reference_p95"],
            inverse=inverse,
        )

    reference_score = (
        0.4 * factors["slope"]
        + 0.4 * factors["tpi"]
        + 0.2 * factors["density"]
    )

    reference_class = classify(
        reference_score[general]
    )

    score_error = float(
        np.max(
            np.abs(
                reference_score[general]
                - arrays["baseline_score"][general]
            )
        )
    )

    class_matches = np.array_equal(
        reference_class,
        arrays["baseline_class"][general],
    )

    if score_error > 1e-5 or not class_matches:
        raise ValueError(
            "Script 13 reference model not reproduced"
        )

    print("\nBASELINE REPRODUCTION: PASS")
    print("Maximum score error:", round(score_error, 8))

    # --------------------------------------------------------
    # 2. Test alternative waterway assumptions
    # --------------------------------------------------------

    results = []
    saved_outputs = {}

    for scenario, (weight, scale_m) in SCENARIOS.items():

        # Near-waterway exposure assumption.
        # Distance 0m -> 1.0, declines with distance.
        # Scale is exploratory, not calibrated.
        proximity = np.exp(
            -distance / scale_m
        )

        score = (
            (1.0 - weight) * reference_score
            + weight * proximity
        )

        selected_score = score[general]
        selected_class = classify(selected_score)

        if (
            not np.isfinite(selected_score).all()
            or np.any(selected_score < 0)
            or np.any(selected_score > 1)
        ):
            raise ValueError(
                f"Invalid score: {scenario}"
            )

        changed = selected_class != reference_class

        row = {
            "scenario": scenario,
            "waterway_weight": weight,
            "distance_decay_scale_m": scale_m,
            "slope_weight": 0.4 * (1 - weight),
            "tpi_weight": 0.4 * (1 - weight),
            "building_weight": 0.2 * (1 - weight),
            "general_land_cells": int(general.sum()),
            "mean_score": round(
                float(selected_score.mean()), 6
            ),
            "mean_absolute_score_change": round(
                float(np.mean(np.abs(
                    selected_score
                    - reference_score[general]
                ))),
                6,
            ),
            "changed_class_percent": round(
                100 * float(changed.mean()), 4
            ),
            "moved_higher_percent": round(
                100 * float(np.mean(
                    selected_class > reference_class
                )),
                4,
            ),
            "moved_lower_percent": round(
                100 * float(np.mean(
                    selected_class < reference_class
                )),
                4,
            ),
            "high_veryhigh_percent": round(
                100 * float(np.mean(
                    selected_class >= 3
                )),
                4,
            ),
        }

        results.append(row)

        print("\nSCENARIO:", scenario)
        print(
            "Class changed:",
            row["changed_class_percent"], "%"
        )
        print(
            "Moved higher:",
            row["moved_higher_percent"], "%"
        )
        print(
            "Moved lower:",
            row["moved_lower_percent"], "%"
        )
        print(
            "High / Very High:",
            row["high_veryhigh_percent"], "%"
        )

        # Save only the reference exploratory waterway
        # scenario as rasters. Alternatives remain
        # reproducible through the settings and metrics.
        if scenario == "Waterway_10pct_300m":

            score_output = np.full(
                study.shape,
                NODATA,
                dtype=np.float32,
            )

            class_output = np.zeros(
                study.shape,
                dtype=np.uint8,
            )

            score_output[study] = score[
                study
            ].astype(np.float32)

            class_output[study] = classify(
                score[study]
            )

            score_path = (
                DATA
                / "16_Waterway_Exploratory_Score_30m.tif"
            )

            class_path = (
                DATA
                / "16_Waterway_Exploratory_Class_30m.tif"
            )

            save_raster(
                score_path,
                score_output,
                profile,
            )

            save_raster(
                class_path,
                class_output,
                profile,
                categorical=True,
            )

            saved_outputs = {
                "score": str(
                    score_path.relative_to(ROOT)
                ),
                "class": str(
                    class_path.relative_to(ROOT)
                ),
            }

    # --------------------------------------------------------
    # 3. Save permanent sensitivity evidence
    # --------------------------------------------------------

    csv_path = (
        METRICS
        / "script16_waterway_proximity_sensitivity.csv"
    )

    pd.DataFrame(results).to_csv(
        csv_path,
        index=False,
    )

    report_path = (
        REPORTS
        / "script16_waterway_proximity_sensitivity.json"
    )

    report = {
        "component": "Part 4 - Seasonal Waterlogging",
        "stage": "Exploratory waterway proximity sensitivity",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "reference_model":
            "Script 13 GEDTM30 Terrain_Urban",
        "baseline_reproduced": True,
        "baseline_max_score_error": score_error,
        "general_land_cells": int(general.sum()),
        "proximity_method":
            "exp(-Euclidean distance / decay scale)",
        "waterway_types": [
            "river", "stream", "canal"
        ],
        "source_network_year": 2026,
        "scenarios": results,
        "saved_exploratory_rasters": saved_outputs,
        "limitations": [
            "This tests a hypothetical positive "
            "waterway-exposure relationship.",
            "Waterway proximity does not establish "
            "flooding, overflow or poor drainage.",
            "Rivers, streams and canals can have "
            "different hydrological functions.",
            "Distance scales and weights are "
            "sensitivity assumptions, not "
            "empirically calibrated parameters.",
            "The 2026 OSM network is not verified "
            "as the exact network present in 2018.",
            "No historical event observations were "
            "used to optimize the tested parameters.",
            "Changes in model classes do not "
            "establish predictive improvement.",
            "Final model selection remains pending.",
        ],
        "status":
            "EXPLORATORY_WATERWAY_SENSITIVITY_COMPLETE",
        "raw_data_modified": False,
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("SCRIPT 16 - SUMMARY")
    print("=" * 70)

    print("Scenarios tested:", len(results))
    print("General-land cells:", int(general.sum()))
    print("Saved:", csv_path)
    print("Saved:", report_path)

    for filename in saved_outputs.values():
        print("Saved:", ROOT / filename)

    print("\nWATERWAY SENSITIVITY: COMPLETE")
    print("FINAL MODEL SELECTION: PENDING")
    print("RAW DATA MODIFIED: NO")


if __name__ == "__main__":
    main()
