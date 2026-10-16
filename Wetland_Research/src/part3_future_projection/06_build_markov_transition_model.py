
from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import rasterio

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PART 3 - FUTURE WETLAND BOUNDARY PROJECTION
# SCRIPT 06 - HISTORICAL MARKOV TRANSITION MODEL
#
# Fit transitions using: 2015 -> 2020
# Hindcast starting map: 2020
# Comparison map: 2025 (classified estimate)
#
# No CA spatial allocation occurs in this script.
# No Markov parameters are fitted from 2025.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MAP_DIR = ROOT / "data/processed/part3_future_projection"
METRIC_DIR = ROOT / "outputs/metrics/part3_future_projection"
REPORT_DIR = ROOT / "outputs/reports/part3_future_projection"

YEARS = [2015, 2020, 2025]
CLASSES = np.arange(1, 7, dtype=np.uint8)

CLASS_NAMES = {
    1: "Wetland_Marsh",
    2: "Open_Water",
    3: "Built_Up",
    4: "Paddy_Agriculture",
    5: "Other_Vegetation",
    6: "Bare_Fill_Disturbed",
}

EXPECTED_CRS = "EPSG:32644"
EXPECTED_PIXEL_COUNT = 94258


def load_maps():
    maps = {}
    reference_grid = None

    for year in YEARS:
        path = MAP_DIR / f"LULC_{year}_6Class_CommonMask.tif"

        if not path.exists():
            raise FileNotFoundError(path)

        with rasterio.open(path) as src:
            if str(src.crs) != EXPECTED_CRS:
                raise ValueError(f"{year}: unexpected CRS")

            grid = (
                src.width,
                src.height,
                src.transform,
                src.crs,
            )

            if reference_grid is None:
                reference_grid = grid
                pixel_area_ha = abs(
                    src.transform.a * src.transform.e
                    - src.transform.b * src.transform.d
                ) / 10000.0
            elif grid != reference_grid:
                raise ValueError("Historical raster grids differ")

            if src.count != 1 or src.nodata != 0:
                raise ValueError(
                    f"{year}: unexpected map encoding"
                )

            maps[year] = src.read(1)

    valid = maps[2015] != 0

    for year in YEARS:
        if not np.array_equal(valid, maps[year] != 0):
            raise ValueError("Common masks are inconsistent")

        if not np.isin(maps[year][valid], CLASSES).all():
            raise ValueError(
                f"{year}: invalid land-cover class detected"
            )

    if int(valid.sum()) != EXPECTED_PIXEL_COUNT:
        raise ValueError("Unexpected number of common pixels")

    return maps, valid, pixel_area_ha


def transition_counts(earlier, later):
    """
    Rows: class at earlier date.
    Columns: class at later date.
    """

    matrix = np.zeros((6, 6), dtype=np.int64)

    np.add.at(
        matrix,
        (earlier.astype(int) - 1, later.astype(int) - 1),
        1,
    )

    return matrix


def transition_probabilities(counts):
    row_totals = counts.sum(axis=1)

    if np.any(row_totals == 0):
        raise ValueError(
            "At least one historical class has no observations"
        )

    probabilities = (
        counts / row_totals[:, None]
    )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Transition probability rows do not sum to one"
        )

    return probabilities


def save_matrix(matrix, path):
    labels = [
        f"{class_id}_{CLASS_NAMES[class_id]}"
        for class_id in CLASSES
    ]

    pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    ).rename_axis(
        "earlier_class"
    ).to_csv(path)


def main():

    print("=" * 70)
    print("PART 3 - FUTURE WETLAND BOUNDARY PROJECTION")
    print("SCRIPT 06 - HISTORICAL MARKOV TRANSITION MODEL")
    print("=" * 70)

    maps, valid, pixel_area_ha = load_maps()

    y2015 = maps[2015][valid]
    y2020 = maps[2020][valid]
    y2025 = maps[2025][valid]

    n_pixels = len(y2020)

    print("\nINPUT MAP CHECKS: PASS")
    print("Common valid pixels:", n_pixels)
    print(
        "Common area:",
        f"{n_pixels * pixel_area_ha:.2f} ha"
    )

    # --------------------------------------------------------
    # 1. FIT 2015 -> 2020 TRANSITION MATRIX
    # --------------------------------------------------------

    counts = transition_counts(y2015, y2020)
    probabilities = transition_probabilities(counts)

    if int(counts.sum()) != n_pixels:
        raise ValueError(
            "Transition count accounting failed"
        )

    print("\n2015 -> 2020 TRANSITION COUNTS")
    print(counts)

    print("\n2015 -> 2020 MARKOV PROBABILITIES")
    print(
        np.array2string(
            probabilities,
            precision=4,
            suppress_small=True,
        )
    )

    print("\nCLASS PERSISTENCE PROBABILITIES")

    for i, class_id in enumerate(CLASSES):
        print(
            f"{CLASS_NAMES[int(class_id)]:22s}: "
            f"{probabilities[i, i]:.4f}"
        )

    # --------------------------------------------------------
    # 2. ESTIMATE 2025 CLASS TOTALS USING MARKOV
    # --------------------------------------------------------

    # Treat 2020 as the starting date.
    # Each 2020 class contributes its pixel total according
    # to the corresponding 2015->2020 transition probabilities.
    #
    # This estimates class quantities, NOT pixel locations.

    counts_2020 = np.bincount(
        y2020.astype(int),
        minlength=7,
    )[1:7]

    estimated_2025 = (
        counts_2020 @ probabilities
    )

    counts_2025 = np.bincount(
        y2025.astype(int),
        minlength=7,
    )[1:7]

    if not np.isclose(
        estimated_2025.sum(),
        n_pixels,
        atol=1e-6,
    ):
        raise ValueError(
            "Markov expected pixel totals do not sum correctly"
        )

    area_rows = []

    print("\n2025 CLASS-AREA ESTIMATES")
    print("These are non-spatial Markov expectations.")

    for index, class_id in enumerate(CLASSES):
        expected_area = (
            estimated_2025[index] * pixel_area_ha
        )

        comparison_area = (
            counts_2025[index] * pixel_area_ha
        )

        persistence_area = (
            counts_2020[index] * pixel_area_ha
        )

        area_rows.append({
            "class_id": int(class_id),
            "class_name": CLASS_NAMES[int(class_id)],
            "markov_expected_2025_ha": expected_area,
            "persistence_2020_area_ha": persistence_area,
            "classified_2025_comparison_ha": comparison_area,
            "markov_area_difference_ha":
                expected_area - comparison_area,
        })

        print(
            f"{CLASS_NAMES[int(class_id)]:22s} | "
            f"Markov: {expected_area:9.2f} ha | "
            f"2025 classification: {comparison_area:9.2f} ha"
        )

    # --------------------------------------------------------
    # 3. SIMPLE SPATIAL PERSISTENCE BASELINE
    # --------------------------------------------------------

    # Baseline: predict that each pixel's 2020 land-cover
    # class remains unchanged in 2025.

    persistence_predictions = y2020.copy()

    baseline_accuracy = float(
        accuracy_score(y2025, persistence_predictions)
    )

    baseline_balanced_accuracy = float(
        balanced_accuracy_score(
            y2025,
            persistence_predictions,
        )
    )

    baseline_macro_f1 = float(
        f1_score(
            y2025,
            persistence_predictions,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        )
    )

    actual_wetland = y2025 == 1
    predicted_wetland = persistence_predictions == 1

    wetland_intersection = int(
        (actual_wetland & predicted_wetland).sum()
    )

    wetland_union = int(
        (actual_wetland | predicted_wetland).sum()
    )

    wetland_iou = (
        wetland_intersection / wetland_union
        if wetland_union > 0
        else None
    )

    wetland_f1 = float(
        f1_score(
            actual_wetland,
            predicted_wetland,
            zero_division=0,
        )
    )

    baseline = {
        "overall_accuracy": baseline_accuracy,
        "balanced_accuracy": baseline_balanced_accuracy,
        "macro_f1": baseline_macro_f1,
        "wetland_iou": wetland_iou,
        "wetland_f1": wetland_f1,
        "wetland_intersection_pixels": wetland_intersection,
        "wetland_union_pixels": wetland_union,
        "compared_pixels": n_pixels,
    }

    print("\n" + "=" * 70)
    print("2020 -> 2025 PERSISTENCE BASELINE")
    print("=" * 70)

    for key, value in baseline.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # This confusion matrix is for the persistence
    # benchmark, NOT for the Markov class-area estimate.
    baseline_confusion = confusion_matrix(
        y2025,
        persistence_predictions,
        labels=CLASSES,
    )

    # --------------------------------------------------------
    # 4. SAVE RESEARCH EVIDENCE
    # --------------------------------------------------------

    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    count_file = (
        METRIC_DIR
        / "script06_markov_2015_2020_counts.csv"
    )

    probability_file = (
        METRIC_DIR
        / "script06_markov_2015_2020_probabilities.csv"
    )

    expected_area_file = (
        METRIC_DIR
        / "script06_expected_2025_class_areas.csv"
    )

    baseline_file = (
        METRIC_DIR
        / "script06_persistence_baseline.json"
    )

    baseline_confusion_file = (
        METRIC_DIR
        / "script06_persistence_confusion_matrix.csv"
    )

    metadata_file = (
        REPORT_DIR
        / "script06_markov_methodology.json"
    )

    save_matrix(counts, count_file)
    save_matrix(probabilities, probability_file)
    save_matrix(baseline_confusion, baseline_confusion_file)

    pd.DataFrame(area_rows).to_csv(
        expected_area_file,
        index=False,
    )

    baseline_file.write_text(
        json.dumps(baseline, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "component": "Part 3 - Future Wetland Boundary Projection",
        "stage": "Historical Markov transitions and hindcast baseline",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "model_type": "Six-class discrete-time Markov chain",
        "transition_training_period": "2015-2020",
        "transition_step_years": 5,
        "hindcast_start_year": 2020,
        "hindcast_target_year": 2025,
        "target_used_for_markov_fitting": False,
        "common_valid_pixel_count": n_pixels,
        "pixel_area_ha": pixel_area_ha,
        "class_ids": {
            str(key): value
            for key, value in CLASS_NAMES.items()
        },
        "row_definition": "Earlier land-cover class",
        "column_definition": "Later land-cover class",
        "markov_expected_2025_pixels":
            estimated_2025.tolist(),
        "persistence_baseline": baseline,
        "interpretation": (
            "Transition probabilities and expected class "
            "areas are derived from model-classified historical "
            "maps. Spatial CA allocation is not included."
        ),
        "limitations": [
            "Historical maps have classification errors.",
            "Large class reversals may represent apparent "
            "rather than genuine land-cover change.",
            "The 2025 comparison is a classified map, "
            "not independent ground truth.",
            "The historical classifier was developed using "
            "reference samples from 2015, 2020 and 2025.",
            "A Markov model assumes historical transition "
            "behaviour is informative for the next interval.",
            "The persistence benchmark predicts no changes.",
            "No cellular automata allocation was performed.",
        ],
    }

    metadata_file.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nSAVED OUTPUTS")
    print(count_file)
    print(probability_file)
    print(expected_area_file)
    print(baseline_file)
    print(baseline_confusion_file)
    print(metadata_file)

    print("\nSCRIPT 06: COMPLETED")
    print("MARKOV TRANSITION MODEL: BUILT")
    print("PERSISTENCE BASELINE: EVALUATED")
    print("CA SPATIAL HINDCAST: NOT YET IMPLEMENTED")


if __name__ == "__main__":
    main()
