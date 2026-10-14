
from pathlib import Path
import json

import numpy as np
import pandas as pd
import rasterio

from scipy.spatial import cKDTree
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data" / "interim" / "part3_future_projection"
)

INPUT_FILE = DATA_DIR / "lulc_training_samples.csv"

RASTER_FILE = (
    ROOT / "data" / "raw" / "landsat"
    / "Landsat_2025_KMC_9Band.tif"
)

YEARS = [2015, 2020, 2025]
CLASSES = [1, 2, 3, 4, 5, 6]

SEED = 161
BLOCK_SIZE_M = 1000
BUFFER_M = 150.0

TEST_BLOCK_FRACTION = 0.20
VALIDATION_BLOCK_FRACTION = 0.25

MIN_TRAIN_CLASS_YEAR = 10
MIN_VALIDATION_CLASS_YEAR = 5
MIN_TEST_CLASS_YEAR = 5

COORDS = ["pixel_center_x", "pixel_center_y"]

PIXEL_KEYS = [
    "year",
    "raster_row",
    "raster_col",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def class_year_counts(frame):
    expected = pd.MultiIndex.from_product(
        [YEARS, CLASSES],
        names=["year", "class_id"]
    )

    actual = frame.groupby(
        ["year", "class_id"]
    ).size()

    return actual.reindex(
        expected,
        fill_value=0
    )


def minimum_class_year_count(frame):
    return int(class_year_counts(frame).min())


def remove_nearby(source, protected, distance_m):
    """
    Keep source observations whose pixel centres are at
    least distance_m from every protected observation.
    """
    if source.empty:
        return source.copy(), source.copy()

    if protected.empty:
        return source.copy(), source.iloc[0:0].copy()

    tree = cKDTree(
        protected[COORDS].to_numpy(dtype=float)
    )

    distances, _ = tree.query(
        source[COORDS].to_numpy(dtype=float)
    )

    keep = distances >= distance_m

    retained = source.iloc[
        np.flatnonzero(keep)
    ].copy()

    excluded = source.iloc[
        np.flatnonzero(~keep)
    ].copy()

    return retained, excluded


def minimum_distance(first, second):
    if first.empty or second.empty:
        raise ValueError(
            "Cannot calculate distance for an empty partition."
        )

    tree = cKDTree(
        second[COORDS].to_numpy(dtype=float)
    )

    distances, _ = tree.query(
        first[COORDS].to_numpy(dtype=float)
    )

    return float(distances.min())


def partition_summary(frame):
    return {
        "observations": int(len(frame)),
        "spatial_blocks": int(
            frame["spatial_group"].nunique()
        ),
        "years": {
            str(year): int(
                (frame["year"] == year).sum()
            )
            for year in YEARS
        },
        "class_year_counts": {
            str(year): {
                str(class_id): int(
                    (
                        (frame["year"] == year)
                        & (frame["class_id"] == class_id)
                    ).sum()
                )
                for class_id in CLASSES
            }
            for year in YEARS
        },
        "minimum_class_year_count":
            minimum_class_year_count(frame),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PART 3 - FUTURE WETLAND BOUNDARY PROJECTION")
    print("SCRIPT 03 - BUFFERED SPATIAL SPLIT")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. LOAD AND VALIDATE INPUT
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    required = (
        PIXEL_KEYS
        + [
            "sample_id",
            "class_id",
            "x_utm",
            "y_utm",
        ]
    )

    if not set(required).issubset(df.columns):
        raise ValueError(
            "Missing required training-data columns."
        )

    if df[required].isna().any().any():
        raise ValueError(
            "Missing required values in training data."
        )

    if len(df) != 1118:
        raise ValueError(
            f"Expected 1118 records, found {len(df)}."
        )

    if set(df["year"]) != set(YEARS):
        raise ValueError("Unexpected reference years.")

    if set(df["class_id"]) != set(CLASSES):
        raise ValueError("Unexpected LULC classes.")

    df = df.sort_values(
        [
            "year",
            "raster_row",
            "raster_col",
            "sample_id",
        ]
    ).reset_index(drop=True)

    df["audit_row_id"] = np.arange(
        1,
        len(df) + 1
    )

    df["observation_uid"] = (
        df["year"].astype(str)
        + "_r"
        + df["raster_row"].astype(str)
        + "_c"
        + df["raster_col"].astype(str)
    )

    df["split"] = "UNASSIGNED"
    df["exclusion_reason"] = ""

    print("\nOriginal usable records:", len(df))

    # --------------------------------------------------------
    # 2. VERIFY DUPLICATE PIXEL LABELS
    # --------------------------------------------------------

    conflicts = (
        df.groupby(PIXEL_KEYS)["class_id"]
        .nunique()
    )

    if (conflicts > 1).any():
        raise ValueError(
            "Conflicting labels exist within a pixel-year."
        )

    duplicates = df.duplicated(
        subset=PIXEL_KEYS,
        keep="first"
    )

    duplicate_rows = df.loc[
        duplicates
    ].copy()

    unique_df = df.loc[
        ~duplicates
    ].copy()

    print(
        "Duplicate pixel records excluded:",
        len(duplicate_rows)
    )

    print(
        "Unique pixel-year observations:",
        len(unique_df)
    )

    if len(unique_df) != 1117:
        raise ValueError(
            "Unexpected unique pixel-year count."
        )

    df.loc[
        duplicate_rows.index,
        "split"
    ] = "EXCLUDED"

    df.loc[
        duplicate_rows.index,
        "exclusion_reason"
    ] = "DUPLICATE_PIXEL_YEAR"

    # --------------------------------------------------------
    # 3. RECORD SAMPLE NAME / YEAR DISCREPANCIES
    # --------------------------------------------------------

    id_year = pd.to_numeric(
        df["sample_id"].astype(str).str.extract(
            r"^(\d{4})_",
            expand=False
        ),
        errors="coerce"
    )

    df["sample_name_year_mismatch"] = (
        id_year.isna()
        | (id_year != df["year"])
    )

    mismatch_count = int(
        df["sample_name_year_mismatch"].sum()
    )

    print(
        "Sample name/year discrepancies:",
        mismatch_count
    )

    if mismatch_count != 4:
        raise ValueError(
            "Unexpected number of temporal-name discrepancies."
        )

    # These records remain flagged for manual temporal QA.
    # A name mismatch alone does not establish an invalid label.

    # --------------------------------------------------------
    # 4. CALCULATE LANDSAT PIXEL CENTRES
    # --------------------------------------------------------

    with rasterio.open(RASTER_FILE) as src:

        if str(src.crs) != "EPSG:32644":
            raise ValueError(
                "Unexpected Landsat coordinate system."
            )

        if (src.width, src.height) != (531, 331):
            raise ValueError(
                "Unexpected Landsat dimensions."
            )

        if (
            (unique_df["raster_row"] < 0).any()
            or (unique_df["raster_row"] >= src.height).any()
            or (unique_df["raster_col"] < 0).any()
            or (unique_df["raster_col"] >= src.width).any()
        ):
            raise ValueError(
                "Training observations outside raster grid."
            )

        xs, ys = rasterio.transform.xy(
            src.transform,
            unique_df["raster_row"].to_numpy(),
            unique_df["raster_col"].to_numpy(),
            offset="center"
        )

    unique_df["pixel_center_x"] = np.asarray(xs)
    unique_df["pixel_center_y"] = np.asarray(ys)

    # --------------------------------------------------------
    # 5. DEFINE GEOGRAPHIC BLOCKS
    # --------------------------------------------------------

    # Reproduce the original geographical block definition
    # used during our feasibility investigations.

    unique_df["block_x"] = np.floor(
        unique_df["x_utm"] / BLOCK_SIZE_M
    ).astype(int)

    unique_df["block_y"] = np.floor(
        unique_df["y_utm"] / BLOCK_SIZE_M
    ).astype(int)

    unique_df["spatial_group"] = (
        unique_df["block_x"].astype(str)
        + "_"
        + unique_df["block_y"].astype(str)
    )

    print(
        "Spatial groups:",
        unique_df["spatial_group"].nunique()
    )

    # --------------------------------------------------------
    # 6. CREATE REPRODUCIBLE INITIAL PARTITIONS
    # --------------------------------------------------------

    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_BLOCK_FRACTION,
        random_state=SEED
    )

    trainval_idx, test_idx = next(
        outer.split(
            unique_df,
            groups=unique_df["spatial_group"]
        )
    )

    trainval = unique_df.iloc[
        trainval_idx
    ].copy()

    test = unique_df.iloc[
        test_idx
    ].copy()

    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_BLOCK_FRACTION,
        random_state=SEED + 1000
    )

    train_idx, validation_idx = next(
        inner.split(
            trainval,
            groups=trainval["spatial_group"]
        )
    )

    train_initial = trainval.iloc[
        train_idx
    ].copy()

    validation_initial = trainval.iloc[
        validation_idx
    ].copy()

    # --------------------------------------------------------
    # 7. APPLY 150 METRE PIXEL-CENTRE BUFFER
    # --------------------------------------------------------

    # Protect the test partition first.
    validation, excluded_val = remove_nearby(
        validation_initial,
        test,
        BUFFER_M
    )

    # Protect test and retained validation observations.
    protected = pd.concat(
        [test, validation],
        ignore_index=True
    )

    train, excluded_train = remove_nearby(
        train_initial,
        protected,
        BUFFER_M
    )

    print("\nBUFFER RESULTS")
    print("Train:", len(train))
    print("Validation:", len(validation))
    print("Test:", len(test))

    print(
        "Validation exclusions:",
        len(excluded_val)
    )

    print(
        "Training exclusions:",
        len(excluded_train)
    )

    # --------------------------------------------------------
    # 8. INDEPENDENT GEOMETRY AND COUNT CHECKS
    # --------------------------------------------------------

    partitions = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    for name, part in partitions.items():
        if part.empty:
            raise ValueError(
                f"Empty partition: {name}"
            )

    minimum_requirements = {
        "train": MIN_TRAIN_CLASS_YEAR,
        "validation": MIN_VALIDATION_CLASS_YEAR,
        "test": MIN_TEST_CLASS_YEAR,
    }

    for name, part in partitions.items():

        minimum_count = minimum_class_year_count(part)

        if minimum_count < minimum_requirements[name]:
            raise ValueError(
                f"Insufficient class/year coverage in {name}: "
                f"{minimum_count}"
            )

    # No geographical group may appear in two splits.
    group_sets = {
        name: set(part["spatial_group"])
        for name, part in partitions.items()
    }

    for first, second in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:

        if group_sets[first] & group_sets[second]:
            raise ValueError(
                f"Spatial block leakage: {first}/{second}"
            )

    distances = {}

    for first, second in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:

        distance = minimum_distance(
            partitions[first],
            partitions[second]
        )

        key = f"{first}_vs_{second}"
        distances[key] = distance

        if distance < BUFFER_M - 1e-7:
            raise ValueError(
                f"Insufficient separation: {key}"
            )

    # Check observation identity across all retained splits.
    retained = pd.concat(
        [
            train.assign(split="train"),
            validation.assign(split="validation"),
            test.assign(split="test"),
        ]
    )

    if retained["observation_uid"].duplicated().any():
        raise ValueError(
            "Duplicate pixel-year observations in retained splits."
        )

    expected_exclusions = (
        len(unique_df) - len(retained)
    )

    actual_exclusions = (
        len(excluded_val) + len(excluded_train)
    )

    if expected_exclusions != actual_exclusions:
        raise ValueError(
            "Buffer-exclusion accounting mismatch."
        )

    if (
        len(retained)
        + len(excluded_val)
        + len(excluded_train)
        + len(duplicate_rows)
        != len(df)
    ):
        raise ValueError(
            "Record accounting failed."
        )

    # --------------------------------------------------------
    # 9. UPDATE FULL AUDIT TABLE
    # --------------------------------------------------------

    df.loc[train.index, "split"] = "train"

    df.loc[
        validation.index,
        "split"
    ] = "validation"

    df.loc[test.index, "split"] = "test"

    df.loc[
        excluded_val.index,
        "split"
    ] = "EXCLUDED"

    df.loc[
        excluded_val.index,
        "exclusion_reason"
    ] = "WITHIN_150M_OF_TEST"

    df.loc[
        excluded_train.index,
        "split"
    ] = "EXCLUDED"

    df.loc[
        excluded_train.index,
        "exclusion_reason"
    ] = "WITHIN_150M_OF_EVALUATION"

    if (df["split"] == "UNASSIGNED").any():
        raise ValueError(
            "Unassigned observations remain in audit."
        )

    # --------------------------------------------------------
    # 10. GENERATE VALIDATION REPORT
    # --------------------------------------------------------

    report = {
        "component": "Part 3 - Future Wetland Boundary Projection",
        "stage": "Historical LULC spatial splitting",
        "status": "GEOMETRIC_CHECKS_PASS_TEMPORAL_QA_PENDING",
        "method": "Buffered spatial GroupShuffleSplit",
        "selected_seed": SEED,
        "seed_selection_basis": (
            "Pre-model geographical separation and "
            "per-class/year reference coverage."
        ),
        "spatial_block_size_m": BLOCK_SIZE_M,
        "pixel_centre_buffer_m": BUFFER_M,
        "raster_resolution_m": 30,
        "original_usable_records": int(len(df)),
        "duplicate_pixel_records": int(
            len(duplicate_rows)
        ),
        "unique_pixel_year_records": int(
            len(unique_df)
        ),
        "spatial_buffer_exclusions": int(
            actual_exclusions
        ),
        "retained_records": int(
            len(retained)
        ),
        "temporal_name_discrepancies": mismatch_count,
        "partitions": {
            name: partition_summary(part)
            for name, part in partitions.items()
        },
        "minimum_pixel_centre_distances_m": {
            key: round(value, 4)
            for key, value in distances.items()
        },
        "checks": {
            "six_classes_present_per_year": True,
            "minimum_class_year_counts_met": True,
            "no_spatial_group_overlap": True,
            "minimum_150m_pixel_centre_separation": True,
            "unique_retained_pixel_year_observations": True,
            "record_accounting_consistent": True,
        },
        "limitations": [
            "150 m separation is a methodological choice, "
            "not proof of complete spatial independence.",
            "Some reference point names have historical "
            "year prefixes inconsistent with their KMZ year.",
            "Historical reference labels require temporal "
            "quality review before final model evaluation.",
            "The final held-out test set must not be used "
            "for model selection or hyperparameter tuning.",
        ],
    }

    # --------------------------------------------------------
    # 11. SAVE REPRODUCIBLE OUTPUTS
    # --------------------------------------------------------

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    split_path = (
        DATA_DIR / "lulc_spatial_split.csv"
    )

    audit_path = (
        DATA_DIR / "lulc_spatial_split_audit.csv"
    )

    duplicate_path = (
        DATA_DIR / "lulc_duplicate_exclusions.csv"
    )

    report_path = (
        DATA_DIR / "lulc_spatial_split_validation.json"
    )

    text_path = (
        DATA_DIR / "lulc_spatial_split_validation.txt"
    )

    retained = retained.sort_values(
        ["year", "observation_uid"]
    ).reset_index(drop=True)

    retained.to_csv(
        split_path,
        index=False
    )

    df.to_csv(
        audit_path,
        index=False
    )

    duplicate_rows.to_csv(
        duplicate_path,
        index=False
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=2
        )

    lines = [
        "PART 3 - SCRIPT 03 VALIDATION",
        "=" * 55,
        f"Selected seed: {SEED}",
        f"Spatial block size: {BLOCK_SIZE_M} m",
        f"Pixel-centre buffer: {BUFFER_M} m",
        "",
        f"Training records: {len(train)}",
        f"Validation records: {len(validation)}",
        f"Test records: {len(test)}",
        f"Buffer exclusions: {actual_exclusions}",
        f"Duplicate-pixel exclusions: {len(duplicate_rows)}",
        "",
        "PIXEL-CENTRE SEPARATION",
    ]

    for key, value in distances.items():
        lines.append(
            f"{key}: {value:.2f} m - PASS"
        )

    lines.extend([
        "",
        "SPATIAL BLOCK OVERLAP: NONE",
        "CLASS/YEAR COUNT REQUIREMENTS: PASS",
        "RECORD ACCOUNTING: PASS",
        "TEMPORAL LABEL QA: PENDING",
        "",
        "Note: A 150 m buffer does not by itself prove",
        "statistical independence of observations.",
        "",
        "SCRIPT 03 GEOMETRIC VALIDATION: PASS",
    ])

    text_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # 12. TERMINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL SPATIAL VALIDATION")
    print("=" * 65)

    for key, value in distances.items():
        print(
            f"{key}: {value:.2f} m - PASS"
        )

    print("\nRetained observations:", len(retained))

    print(
        "Buffer exclusions:",
        actual_exclusions
    )

    print(
        "Duplicate-pixel exclusions:",
        len(duplicate_rows)
    )

    print("\nVALIDATION FILES")
    print(split_path)
    print(audit_path)
    print(duplicate_path)
    print(report_path)
    print(text_path)

    print("\nSCRIPT 03 GEOMETRIC VALIDATION: PASS")
    print("TEMPORAL REFERENCE QA: PENDING")


if __name__ == "__main__":
    main()
