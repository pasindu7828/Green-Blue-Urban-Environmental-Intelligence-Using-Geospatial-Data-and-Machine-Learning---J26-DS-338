
from pathlib import Path
from datetime import datetime, timezone
import heapq
import json

import numpy as np
import pandas as pd
import rasterio

from scipy.ndimage import convolve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# PART 3 - FUTURE WETLAND BOUNDARY PROJECTION
# SCRIPT 07 - CA-MARKOV SPATIAL HINDCAST
#
# Historical transitions: 2015 -> 2020
# Starting map:          2020
# Predicted map:         2025
#
# Comparison models:
# 1. Persistence (no change)
# 2. Markov + random spatial allocation
# 3. Markov + 3x3-neighbourhood CA allocation
#
# The classified 2025 map is used ONLY for evaluation.
#
# This is an exploratory one-step CA-Markov hindcast,
# NOT a verified future wetland forecast.
# ============================================================


ROOT = Path(__file__).resolve().parents[2]

MAP_DIR = (
    ROOT / "data/processed/part3_future_projection"
)

METRIC_DIR = (
    ROOT / "outputs/metrics/part3_future_projection"
)

REPORT_DIR = (
    ROOT / "outputs/reports/part3_future_projection"
)

PROBABILITY_FILE = (
    METRIC_DIR
    / "script06_markov_2015_2020_probabilities.csv"
)

CLASSES = np.arange(1, 7, dtype=int)

CLASS_NAMES = {
    1: "Wetland_Marsh",
    2: "Open_Water",
    3: "Built_Up",
    4: "Paddy_Agriculture",
    5: "Other_Vegetation",
    6: "Bare_Fill_Disturbed",
}

SEED = 42
EXPECTED_PIXELS = 94258

CLASS_COLORS = {
    0: (0, 0, 0, 0),
    1: (48, 125, 93, 255),
    2: (45, 125, 211, 255),
    3: (195, 80, 75, 255),
    4: (225, 190, 80, 255),
    5: (115, 174, 78, 255),
    6: (171, 141, 106, 255),
}


def load_inputs():

    maps = {}
    reference_grid = None
    profile = None

    for year in [2015, 2020, 2025]:

        path = (
            MAP_DIR
            / f"LULC_{year}_6Class_CommonMask.tif"
        )

        with rasterio.open(path) as src:

            grid = (
                src.crs,
                src.transform,
                src.width,
                src.height,
            )

            if reference_grid is None:
                reference_grid = grid
                profile = src.profile.copy()
            elif grid != reference_grid:
                raise ValueError(
                    "Historical raster grids do not match."
                )

            if (
                str(src.crs) != "EPSG:32644"
                or src.count != 1
                or src.nodata != 0
            ):
                raise ValueError(
                    f"Unexpected raster metadata: {year}"
                )

            maps[year] = src.read(1)

    valid = maps[2020] != 0

    if int(valid.sum()) != EXPECTED_PIXELS:
        raise ValueError(
            "Unexpected number of common valid pixels."
        )

    for year in [2015, 2020, 2025]:

        if not np.array_equal(
            valid,
            maps[year] != 0
        ):
            raise ValueError(
                "The historical masks differ."
            )

        if not np.isin(
            maps[year][valid],
            CLASSES
        ).all():
            raise ValueError(
                f"Invalid class ID in {year}."
            )

    if not PROBABILITY_FILE.exists():
        raise FileNotFoundError(
            PROBABILITY_FILE
        )

    probabilities = pd.read_csv(
        PROBABILITY_FILE,
        index_col=0
    ).to_numpy(dtype=float)

    if probabilities.shape != (6, 6):
        raise ValueError(
            "Invalid Markov matrix shape."
        )

    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(
            probabilities.sum(axis=1),
            1.0,
            atol=1e-8
        )
    ):
        raise ValueError(
            "Invalid Markov probabilities."
        )

    # Independently verify that the saved matrix really
    # represents transitions from 2015 to 2020.
    earlier = maps[2015][valid].astype(int) - 1
    later = maps[2020][valid].astype(int) - 1

    counts = np.zeros(
        (6, 6),
        dtype=np.int64
    )

    np.add.at(
        counts,
        (earlier, later),
        1
    )

    calculated = (
        counts / counts.sum(axis=1, keepdims=True)
    )

    if not np.allclose(
        calculated,
        probabilities,
        atol=1e-10
    ):
        # Saved CSV may contain normal floating-point
        # serialization differences.
        if not np.allclose(
            calculated,
            probabilities,
            atol=1e-6
        ):
            raise ValueError(
                "Saved Markov matrix does not match "
                "historical classifications."
            )

    return maps, valid, profile, probabilities


def calculate_transition_quotas(
    starting_map,
    valid,
    probabilities,
):
    """
    Convert expected Markov transitions into exact
    integer transition quotas for each source class.

    Largest-remainder rounding preserves the number
    of source-class pixels.
    """

    source = starting_map[valid]

    quotas = np.zeros(
        (6, 6),
        dtype=np.int64
    )

    for i in range(6):

        n = int(
            (source == i + 1).sum()
        )

        expected = (
            n * probabilities[i]
        )

        allocated = np.floor(
            expected
        ).astype(np.int64)

        remaining = (
            n - int(allocated.sum())
        )

        fractional = (
            expected - allocated
        )

        order = np.argsort(
            -fractional,
            kind="stable"
        )

        allocated[
            order[:remaining]
        ] += 1

        if int(allocated.sum()) != n:
            raise ValueError(
                "Markov quota accounting failed."
            )

        quotas[i] = allocated

    if int(quotas.sum()) != int(valid.sum()):
        raise ValueError(
            "Total transition quotas do not match "
            "the study-area pixel count."
        )

    return quotas


def random_spatial_allocation(
    starting_map,
    valid,
    quotas,
):
    """
    Spatial control model.

    Preserves exactly the same Markov quotas but assigns
    destinations randomly within each source class.
    """

    rng = np.random.default_rng(SEED)

    result = np.zeros(
        starting_map.shape,
        dtype=np.uint8
    )

    start_flat = starting_map.ravel()
    result_flat = result.ravel()

    for i in range(6):

        indices = np.flatnonzero(
            (start_flat == i + 1)
        )

        rng.shuffle(indices)

        position = 0

        for j in range(6):

            count = int(
                quotas[i, j]
            )

            selected = indices[
                position:position + count
            ]

            result_flat[selected] = j + 1

            position += count

        if position != len(indices):
            raise ValueError(
                "Random allocation count mismatch."
            )

    if not np.array_equal(
        result != 0,
        valid
    ):
        raise ValueError(
            "Random allocation changed the valid mask."
        )

    return result


def neighbourhood_support(
    starting_map,
    valid,
):
    """
    Calculate the fraction of valid neighbouring cells
    in each land-cover class.

    Uses the eight surrounding cells in a 3x3 window.
    The centre cell is deliberately excluded.

    Neighbours outside the common mask do not contribute.
    """

    kernel = np.ones(
        (3, 3),
        dtype=np.float32
    )

    kernel[1, 1] = 0

    neighbour_total = convolve(
        valid.astype(np.float32),
        kernel,
        mode="constant",
        cval=0
    )

    denominator = np.maximum(
        neighbour_total,
        1
    )

    support = []

    for class_id in CLASSES:

        class_neighbours = convolve(
            (starting_map == class_id).astype(
                np.float32
            ),
            kernel,
            mode="constant",
            cval=0
        )

        fraction = (
            class_neighbours / denominator
        )

        support.append(
            fraction.ravel()
        )

    return support


def ca_spatial_allocation(
    starting_map,
    valid,
    quotas,
):
    """
    One-step, quota-constrained neighbourhood CA.

    For each source class:
    1. Allocate required persistent cells to pixels
       with stronger same-class neighbourhood support.
    2. Allocate the remaining source pixels to target
       classes using neighbourhood target-class support.
    3. Enforce exact Markov source-to-target quotas.

    Ties are resolved reproducibly using a fixed seed.

    No information from 2025 is used in allocation.
    """

    rng = np.random.default_rng(SEED)

    support = neighbourhood_support(
        starting_map,
        valid
    )

    result = np.zeros(
        starting_map.shape,
        dtype=np.uint8
    )

    result_flat = result.ravel()
    source_flat = starting_map.ravel()

    for i in range(6):

        class_id = i + 1

        source_indices = np.flatnonzero(
            source_flat == class_id
        )

        n_persist = int(
            quotas[i, i]
        )

        # Prioritize same-class neighbourhood support
        # for cells assigned to remain unchanged.
        persistence_score = (
            support[i][source_indices]
            + rng.random(
                len(source_indices)
            ) * 1e-8
        )

        ranked_positions = np.argsort(
            -persistence_score,
            kind="stable"
        )

        retained_indices = source_indices[
            ranked_positions[:n_persist]
        ]

        changed_indices = source_indices[
            ranked_positions[n_persist:]
        ]

        result_flat[
            retained_indices
        ] = class_id

        remaining_quotas = quotas[i].copy()

        remaining_quotas[i] = 0

        # Candidate target assignments ranked globally
        # within this source class.
        #
        # Each remaining pixel competes for the allowed
        # destination classes. Higher local support wins.
        candidates = []

        for j in range(6):

            if j == i or remaining_quotas[j] == 0:
                continue

            scores = support[j][
                changed_indices
            ]

            ties = rng.random(
                len(changed_indices)
            )

            candidates.extend(
                (
                    -float(score),
                    float(tie),
                    int(index),
                    int(j),
                )
                for score, tie, index in zip(
                    scores,
                    ties,
                    changed_indices
                )
            )

        heapq.heapify(candidates)

        unassigned = len(changed_indices)

        while unassigned > 0 and candidates:

            (
                negative_score,
                tie,
                index,
                destination_index
            ) = heapq.heappop(candidates)

            if result_flat[index] != 0:
                continue

            if remaining_quotas[
                destination_index
            ] <= 0:
                continue

            result_flat[index] = (
                destination_index + 1
            )

            remaining_quotas[
                destination_index
            ] -= 1

            unassigned -= 1

        if unassigned != 0:
            raise ValueError(
                f"CA could not allocate all pixels "
                f"for source class {class_id}."
            )

        if np.any(remaining_quotas != 0):
            raise ValueError(
                f"CA transition quotas not satisfied "
                f"for source class {class_id}."
            )

    if not np.array_equal(
        result != 0,
        valid
    ):
        raise ValueError(
            "CA allocation changed the common mask."
        )

    return result


def observed_transition_counts(
    starting_map,
    prediction,
    valid,
):
    earlier = starting_map[valid].astype(int) - 1
    later = prediction[valid].astype(int) - 1

    counts = np.zeros(
        (6, 6),
        dtype=np.int64
    )

    np.add.at(
        counts,
        (earlier, later),
        1
    )

    return counts


def evaluate(
    reference,
    prediction,
    valid,
):
    """
    Compare hindcast with the classified 2025 map.

    This is map-to-map agreement, not independent
    verification against observed field conditions.
    """

    actual = reference[valid]
    predicted = prediction[valid]

    actual_wetland = actual == 1
    predicted_wetland = predicted == 1

    intersection = int(
        np.logical_and(
            actual_wetland,
            predicted_wetland
        ).sum()
    )

    union = int(
        np.logical_or(
            actual_wetland,
            predicted_wetland
        ).sum()
    )

    return {
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                actual,
                predicted
            )
        ),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                labels=CLASSES,
                average="macro",
                zero_division=0
            )
        ),
        "wetland_precision": float(
            precision_score(
                actual_wetland,
                predicted_wetland,
                zero_division=0
            )
        ),
        "wetland_recall": float(
            recall_score(
                actual_wetland,
                predicted_wetland,
                zero_division=0
            )
        ),
        "wetland_f1": float(
            f1_score(
                actual_wetland,
                predicted_wetland,
                zero_division=0
            )
        ),
        "wetland_iou": float(
            intersection / union
        ) if union else None,
        "predicted_wetland_pixels": int(
            predicted_wetland.sum()
        ),
        "reference_wetland_pixels": int(
            actual_wetland.sum()
        ),
        "wetland_area_difference_ha": float(
            (
                int(predicted_wetland.sum())
                - int(actual_wetland.sum())
            ) * 0.09
        ),
        "changed_pixels_from_2020": None,
    }


def write_prediction(
    path,
    array,
    profile,
    method,
):
    output_profile = profile.copy()

    output_profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
    )

    with rasterio.open(
        path,
        "w",
        **output_profile
    ) as dst:

        dst.write(array, 1)

        dst.write_colormap(
            1,
            CLASS_COLORS
        )

        dst.set_band_description(
            1,
            f"2025 {method} hindcast"
        )

        dst.update_tags(
            predicted_year="2025",
            initial_year="2020",
            transition_period="2015_to_2020",
            method=method,
            interpretation=(
                "Exploratory model-based hindcast; "
                "not verified ground truth"
            ),
        )


def save_matrix(
    matrix,
    filename,
):
    labels = [
        f"{c}_{CLASS_NAMES[c]}"
        for c in CLASSES
    ]

    pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    ).rename_axis(
        "source_class"
    ).to_csv(filename)


def main():

    print("=" * 70)
    print("PART 3 - FUTURE WETLAND BOUNDARY PROJECTION")
    print("SCRIPT 07 - CA-MARKOV SPATIAL HINDCAST")
    print("=" * 70)

    maps, valid, profile, probabilities = (
        load_inputs()
    )

    print("\nINPUT VALIDATION: PASS")
    print("Common pixels:", int(valid.sum()))
    print("Historical Markov matrix: verified")
    print("Start year: 2020")
    print("Hindcast target: 2025")

    # --------------------------------------------------------
    # 1. MARKOV TRANSITION QUOTAS
    # --------------------------------------------------------

    quotas = calculate_transition_quotas(
        maps[2020],
        valid,
        probabilities
    )

    print("\nMARKOV 2020 -> 2025 TRANSITION QUOTAS")
    print("Rows: source class; columns: target class")
    print(quotas)

    print(
        "\nExpected wetland pixels:",
        int(quotas[:, 0].sum())
    )

    # --------------------------------------------------------
    # 2. CREATE HINDCASTS WITHOUT USING 2025 TARGET
    # --------------------------------------------------------

    print("\nGenerating random Markov allocation...")

    random_map = random_spatial_allocation(
        maps[2020],
        valid,
        quotas
    )

    print("Random allocation: completed")

    print("\nGenerating 3x3-neighbourhood CA allocation...")

    ca_map = ca_spatial_allocation(
        maps[2020],
        valid,
        quotas
    )

    print("CA allocation: completed")

    for method, prediction in [
        ("Random", random_map),
        ("CA", ca_map),
    ]:

        actual_quotas = observed_transition_counts(
            maps[2020],
            prediction,
            valid
        )

        if not np.array_equal(
            actual_quotas,
            quotas
        ):
            raise ValueError(
                f"{method} allocation failed "
                f"the transition-quota check."
            )

    print("\nBoth allocations satisfy exact Markov quotas.")

    # --------------------------------------------------------
    # 3. EVALUATE AGAINST CLASSIFIED 2025 MAP
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("2025 HINDCAST COMPARISON")
    print("=" * 70)

    predictions = {
        "Persistence": maps[2020],
        "Markov_RandomAllocation": random_map,
        "Markov_NeighbourhoodCA": ca_map,
    }

    results = []

    for method, prediction in predictions.items():

        metrics = evaluate(
            maps[2025],
            prediction,
            valid
        )

        metrics["changed_pixels_from_2020"] = int(
            (
                prediction[valid]
                != maps[2020][valid]
            ).sum()
        )

        results.append({
            "method": method,
            **metrics
        })

    comparison = pd.DataFrame(
        results
    )

    columns = [
        "method",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "wetland_precision",
        "wetland_recall",
        "wetland_f1",
        "wetland_iou",
        "wetland_area_difference_ha",
        "changed_pixels_from_2020",
    ]

    print(
        comparison[columns]
        .round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 4. SAVE IMPORTANT OUTPUTS
    # --------------------------------------------------------

    MAP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    METRIC_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    random_path = (
        MAP_DIR
        / "LULC_2025_Hindcast_Markov_Random.tif"
    )

    ca_path = (
        MAP_DIR
        / "LULC_2025_Hindcast_CA_Markov.tif"
    )

    metrics_path = (
        METRIC_DIR
        / "script07_hindcast_model_comparison.csv"
    )

    quotas_path = (
        METRIC_DIR
        / "script07_markov_transition_quotas.csv"
    )

    confusion_path = (
        METRIC_DIR
        / "script07_ca_hindcast_confusion_matrix.csv"
    )

    metadata_path = (
        REPORT_DIR
        / "script07_ca_markov_hindcast_metadata.json"
    )

    write_prediction(
        random_path,
        random_map,
        profile,
        "Markov random allocation"
    )

    write_prediction(
        ca_path,
        ca_map,
        profile,
        "Neighbourhood CA-Markov"
    )

    comparison.to_csv(
        metrics_path,
        index=False
    )

    save_matrix(
        quotas,
        quotas_path
    )

    ca_confusion = confusion_matrix(
        maps[2025][valid],
        ca_map[valid],
        labels=CLASSES
    )

    save_matrix(
        ca_confusion,
        confusion_path
    )

    metadata = {
        "component": "Part 3 - Future Wetland Boundary Projection",
        "stage": "CA-Markov historical hindcast",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "historical_transition_period": "2015-2020",
        "initial_map_year": 2020,
        "hindcast_target_year": 2025,
        "markov_matrix_source": str(
            PROBABILITY_FILE
        ),
        "neighbourhood": "3x3 excluding centre cell",
        "allocation_method": (
            "One-step, neighbourhood-scored allocation "
            "with exact source-to-target Markov quotas"
        ),
        "random_seed": SEED,
        "random_allocation_control": True,
        "target_map_used_during_allocation": False,
        "study_area_pixels": int(valid.sum()),
        "model_comparison": results,
        "model_acceptance": "PENDING_RESEARCH_REVIEW",
        "limitations": [
            "Historical LULC maps are model-classified estimates.",
            "The 2025 map is not independently surveyed truth.",
            "Historical classifier used samples from 2025.",
            "The Markov model assumes transition stationarity.",
            "The CA stage allocates fixed Markov quantities; "
            "it cannot correct errors in expected class totals.",
            "3x3 neighbourhood is a fixed experimental rule, "
            "not an empirically optimized spatial scale.",
            "This experiment is a single five-year CA step.",
            "Map agreement does not establish forecast validity.",
        ],
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\nSAVED OUTPUTS")
    print(random_path)
    print(ca_path)
    print(metrics_path)
    print(quotas_path)
    print(confusion_path)
    print(metadata_path)

    print("\nSCRIPT 07: COMPLETED")
    print("CA-MARKOV HINDCAST: GENERATED")
    print("MODEL ACCEPTANCE: PENDING RESULTS REVIEW")
    print("2030 SCENARIO: NOT YET GENERATED")


if __name__ == "__main__":
    main()
