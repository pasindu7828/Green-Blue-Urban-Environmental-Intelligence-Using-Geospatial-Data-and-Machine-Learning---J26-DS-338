
from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# PART 4 - SCRIPT 13
# INITIAL SEASONAL WATERLOGGING SUSCEPTIBILITY MODEL
#
# Exploratory GIS-based weighted overlay:
# A. Terrain-only baseline
# B. Terrain + urban development scenario
#
# This is NOT a trained ML classifier.
# Scores are NOT calibrated flood probabilities.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data/processed/part4_waterlogging"
METRICS = ROOT / "outputs/metrics/part4_waterlogging"
REPORTS = ROOT / "outputs/reports/part4_waterlogging"

INPUTS = {
    "mask": "06_Kaduwela_Study_Mask_30m.tif",
    "slope": "07_GEDTM30_Slope_Degrees_30m.tif",
    "tpi": "07_GEDTM30_TPI_300m_30m.tif",
    "density": "09_OSM_Building_Density_300m_30m.tif",
    "screen": "12_Landcover_Screening_30m.tif",
}

SCENARIOS = {
    "Terrain_Only": {
        "slope": 0.50,
        "tpi": 0.50,
        "density": 0.00,
    },
    "Terrain_Urban": {
        "slope": 0.40,
        "tpi": 0.40,
        "density": 0.20,
    },
}

SCREEN_NAMES = {
    1: "General_Land",
    2: "Paddy_Agriculture",
    3: "Wetland_Marsh",
    4: "Open_Water",
    5: "Unknown_Landcover",
}

CLASS_NAMES = {
    1: "Low",
    2: "Moderate",
    3: "High",
    4: "Very_High",
}

NODATA = -9999.0


def normalize(values, reference, direction):
    """
    Percentile normalization:
    5th percentile -> 0
    95th percentile -> 1

    Values outside the reference range are clipped.
    """
    low = float(np.percentile(reference, 5))
    high = float(np.percentile(reference, 95))

    if not high > low:
        raise ValueError(
            "Insufficient variation for normalization"
        )

    normalized = np.clip(
        (values - low) / (high - low),
        0.0,
        1.0,
    )

    if direction == "lower_higher_susceptibility":
        normalized = 1.0 - normalized
    elif direction != "higher_higher_susceptibility":
        raise ValueError("Unknown factor direction")

    return normalized, {
        "reference_p05": low,
        "reference_p95": high,
        "direction": direction,
    }


def save_raster(path, values, profile, categorical=False):
    metadata = profile.copy()

    metadata.update(
        driver="GTiff",
        count=1,
        dtype="uint8" if categorical else "float32",
        nodata=0 if categorical else NODATA,
        compress="deflate",
    )

    with rasterio.open(path, "w", **metadata) as dst:
        dst.write(values, 1)


def main():

    print("=" * 70)
    print("PART 4 - INITIAL SUSCEPTIBILITY MODEL")
    print("SCRIPT 13")
    print("=" * 70)

    for folder in (DATA, METRICS, REPORTS):
        folder.mkdir(parents=True, exist_ok=True)

    arrays = {}
    reference = None
    profile = None

    # --------------------------------------------------------
    # 1. Load and validate aligned factor rasters
    # --------------------------------------------------------

    for name, filename in INPUTS.items():

        with rasterio.open(DATA / filename) as src:

            if reference is None:
                reference = (
                    src.crs, src.transform, src.shape
                )
                profile = src.profile.copy()

            elif (
                src.crs,
                src.transform,
                src.shape,
            ) != reference:
                raise ValueError(
                    f"Grid alignment mismatch: {name}"
                )

            arrays[name] = src.read(1)

    study = arrays["mask"] == 1
    screening = arrays["screen"]

    if int(study.sum()) != 94431:
        raise ValueError("Unexpected study-area size")

    if not np.isin(
        screening[study], list(SCREEN_NAMES)
    ).all():
        raise ValueError("Invalid screening categories")

    general = study & (screening == 1)

    if not np.any(general):
        raise ValueError("No general-land reference cells")

    for name in ["slope", "tpi", "density"]:

        values = arrays[name][study]

        if (
            not np.isfinite(values).all()
            or np.any(values == NODATA)
        ):
            raise ValueError(
                f"Invalid study-area feature: {name}"
            )

    if (
        np.any(arrays["slope"][study] < 0)
        or np.any(arrays["slope"][study] >= 90)
        or np.any(arrays["density"][study] < 0)
        or np.any(arrays["density"][study] > 100)
    ):
        raise ValueError("Unexpected factor range")

    # --------------------------------------------------------
    # 2. Normalize based on general-land reference cells
    # --------------------------------------------------------

    factor_settings = {
        "slope": "lower_higher_susceptibility",
        "tpi": "lower_higher_susceptibility",
        "density": "higher_higher_susceptibility",
    }

    factor_scores = {}
    normalization_report = {}

    for name, direction in factor_settings.items():

        scores, details = normalize(
            arrays[name].astype(np.float64),
            arrays[name][general].astype(np.float64),
            direction,
        )

        factor_scores[name] = scores
        normalization_report[name] = details

        print(
            f"\n{name}: "
            f"reference P05={details['reference_p05']:.4f}, "
            f"P95={details['reference_p95']:.4f}"
        )

    # --------------------------------------------------------
    # 3. Calculate scenario scores
    # --------------------------------------------------------

    rows = []
    output_records = {}
    scenario_classes = {}

    for scenario, weights in SCENARIOS.items():

        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError("Scenario weights must total 1")

        score = sum(
            factor_scores[name] * weight
            for name, weight in weights.items()
        )

        if (
            not np.isfinite(score[study]).all()
            or np.any(score[study] < 0)
            or np.any(score[study] > 1)
        ):
            raise ValueError(
                f"Invalid susceptibility score: {scenario}"
            )

        # Exploratory equal-interval classification.
        # Thresholds are not calibrated hazard boundaries.
        classes = np.zeros(
            study.shape,
            dtype=np.uint8,
        )

        classes[study] = np.digitize(
            score[study],
            bins=[0.25, 0.50, 0.75],
            right=False,
        ).astype(np.uint8) + 1

        score_output = np.full(
            study.shape,
            NODATA,
            dtype=np.float32,
        )

        score_output[study] = (
            score[study].astype(np.float32)
        )

        score_path = (
            DATA
            / f"13_{scenario}_Susceptibility_Score_30m.tif"
        )

        class_path = (
            DATA
            / f"13_{scenario}_Susceptibility_Class_30m.tif"
        )

        save_raster(score_path, score_output, profile)
        save_raster(
            class_path,
            classes,
            profile,
            categorical=True,
        )

        scenario_classes[scenario] = classes

        for screen_code, screen_name in SCREEN_NAMES.items():

            subset = (
                study & (screening == screen_code)
            )

            total = int(subset.sum())

            for class_code, class_name in CLASS_NAMES.items():

                count = int(np.count_nonzero(
                    subset & (classes == class_code)
                ))

                rows.append({
                    "scenario": scenario,
                    "screening_category": screen_name,
                    "susceptibility_class": class_name,
                    "cells": count,
                    "share_within_screen_percent":
                        round(
                            100 * count / total
                            if total else 0,
                            4,
                        ),
                })

        general_scores = score[general]

        output_records[scenario] = {
            "weights": weights,
            "score_file": str(
                score_path.relative_to(ROOT)
            ),
            "class_file": str(
                class_path.relative_to(ROOT)
            ),
            "general_land_mean_score":
                round(float(general_scores.mean()), 6),
            "general_land_p95_score":
                round(float(np.percentile(
                    general_scores, 95
                )), 6),
        }

        print("\nSCENARIO:", scenario)
        print("Weights:", weights)
        print(
            "General-land mean score:",
            output_records[scenario][
                "general_land_mean_score"
            ]
        )

    # --------------------------------------------------------
    # 4. Compare terrain and terrain-urban scenarios
    # --------------------------------------------------------

    terrain_classes = scenario_classes["Terrain_Only"]
    urban_classes = scenario_classes["Terrain_Urban"]

    changed = int(np.count_nonzero(
        terrain_classes[general]
        != urban_classes[general]
    ))

    changed_pct = (
        100 * changed / int(general.sum())
    )

    print("\nMODEL COMPARISON")
    print("General-land cells:", int(general.sum()))
    print(
        "Cells changing class between scenarios:",
        changed
    )
    print(
        "Percent changing class:",
        round(changed_pct, 4), "%"
    )

    # --------------------------------------------------------
    # 5. Save permanent evidence
    # --------------------------------------------------------

    metrics_path = (
        METRICS
        / "script13_initial_susceptibility_distribution.csv"
    )

    pd.DataFrame(rows).to_csv(
        metrics_path,
        index=False,
    )

    report_path = (
        REPORTS
        / "script13_initial_susceptibility_model.json"
    )

    report = {
        "component": "Part 4 - Seasonal Waterlogging",
        "stage": "Exploratory susceptibility modelling",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "method": "GIS weighted linear combination",
        "model_type": "Untrained exploratory spatial index",
        "study_cells": int(study.sum()),
        "primary_general_land_cells":
            int(general.sum()),
        "normalization":
            normalization_report,
        "scenarios": output_records,
        "class_thresholds": {
            "Low": "0 <= score < 0.25",
            "Moderate": "0.25 <= score < 0.50",
            "High": "0.50 <= score < 0.75",
            "Very_High": "0.75 <= score <= 1.00",
        },
        "general_land_class_changed_cells": changed,
        "general_land_class_changed_percent":
            round(changed_pct, 4),
        "metrics_csv":
            str(metrics_path.relative_to(ROOT)),
        "limitations": [
            "Scores are relative indices, not "
            "calibrated flooding probabilities.",
            "Weights and class thresholds are "
            "provisional, not empirically optimized.",
            "Factor percentile normalization uses "
            "2020 general-land screening cells.",
            "Low slope and low TPI are assumed "
            "to increase terrain susceptibility.",
            "Building density is an urban-development "
            "proxy, not measured imperviousness.",
            "The urban factor represents 2026 OSM "
            "mapping and is not 2018 land cover.",
            "No precipitation or drainage capacity "
            "is modelled at individual grid cells.",
            "Flat elevated areas can receive high "
            "terrain scores; hydrological interpretation "
            "requires additional evaluation.",
            "Scores over wetland, paddy, open water "
            "and unknown land cover require separate "
            "screening interpretation.",
            "Two manually digitized 2018 events do not "
            "provide independent supervised ML labels.",
            "DEM, factor, normalization and weighting "
            "sensitivity tests remain required.",
            "No predictive accuracy is claimed.",
        ],
        "raw_data_modified": False,
        "status": "INITIAL_EXPLORATORY_MODEL_COMPLETE",
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("SCRIPT 13 - SUMMARY")
    print("=" * 70)

    for scenario, record in output_records.items():
        print("\nScenario:", scenario)
        print("Saved:", ROOT / record["score_file"])
        print("Saved:", ROOT / record["class_file"])

    print("\nSaved:", metrics_path)
    print("Saved:", report_path)

    print("\nINITIAL SUSCEPTIBILITY MODEL: COMPLETE")
    print("SENSITIVITY / EVENT EVALUATION: PENDING")
    print("FINAL PREDICTION MAP: NOT APPROVED YET")
    print("RAW DATA MODIFIED: NO")


if __name__ == "__main__":
    main()
