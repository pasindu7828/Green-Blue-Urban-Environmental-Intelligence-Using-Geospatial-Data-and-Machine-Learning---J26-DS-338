
from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import rasterio

from scipy.spatial import cKDTree


# ============================================================
# PART 3 - INDEPENDENT SPATIAL SPLIT VALIDATION
#
# This script does NOT construct, select, or change a split.
# It independently checks the saved outputs of Script 03.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data" / "interim" / "part3_future_projection"
)

REPORT_DIR = (
    ROOT / "outputs" / "reports" / "part3_future_projection"
)

SOURCE_FILE = DATA_DIR / "lulc_training_samples.csv"
SPLIT_FILE = DATA_DIR / "lulc_spatial_split.csv"
AUDIT_FILE = DATA_DIR / "lulc_spatial_split_audit.csv"
DUPLICATE_FILE = DATA_DIR / "lulc_duplicate_exclusions.csv"

RASTER_FILE = (
    ROOT / "data" / "raw" / "landsat"
    / "Landsat_2025_KMC_9Band.tif"
)

BUFFER_M = 150.0
BLOCK_SIZE_M = 1000

YEARS = [2015, 2020, 2025]
CLASSES = [1, 2, 3, 4, 5, 6]

PARTITIONS = ["train", "validation", "test"]

MIN_COUNTS = {
    "train": 10,
    "validation": 5,
    "test": 5,
}

PIXEL_KEYS = ["year", "raster_row", "raster_col"]

checks = []
details = {}


def check(name, passed, detail=""):
    passed = bool(passed)

    checks.append({
        "name": name,
        "passed": passed,
        "detail": str(detail),
    })

    print(
        f"{'PASS' if passed else 'FAIL'} | "
        f"{name}"
    )

    if detail:
        print(f"       {detail}")


def minimum_distance(first, second):
    a = first[
        ["pixel_center_x", "pixel_center_y"]
    ].to_numpy(dtype=float)

    b = second[
        ["pixel_center_x", "pixel_center_y"]
    ].to_numpy(dtype=float)

    if len(a) == 0 or len(b) == 0:
        return float("nan")

    distances, _ = cKDTree(b).query(a)

    return float(distances.min())


def add_pixel_centres(frame, transform):
    frame = frame.copy()

    xs, ys = rasterio.transform.xy(
        transform,
        frame["raster_row"].to_numpy(),
        frame["raster_col"].to_numpy(),
        offset="center"
    )

    frame["pixel_center_x"] = np.asarray(xs)
    frame["pixel_center_y"] = np.asarray(ys)

    return frame


def class_year_count(frame):
    counts = frame.groupby(
        ["year", "class_id"]
    ).size()

    expected = pd.MultiIndex.from_product(
        [YEARS, CLASSES],
        names=["year", "class_id"]
    )

    return counts.reindex(expected, fill_value=0)


def main():
    print("=" * 65)
    print("PART 3 - INDEPENDENT SPATIAL VALIDATION")
    print("SCRIPT 03B")
    print("=" * 65)

    source = pd.read_csv(SOURCE_FILE)
    split = pd.read_csv(SPLIT_FILE)
    audit = pd.read_csv(AUDIT_FILE)
    duplicates = pd.read_csv(DUPLICATE_FILE)

    print("\n========== RECORD ACCOUNTING ==========")

    check(
        "Source record count",
        len(source) == 1118,
        f"Found: {len(source)}"
    )

    check(
        "Audit record count",
        len(audit) == len(source),
        f"Found: {len(audit)}"
    )

    check(
        "Retained split count",
        len(split) == 1066,
        f"Found: {len(split)}"
    )

    check(
        "Duplicate exclusion count",
        len(duplicates) == 1,
        f"Found: {len(duplicates)}"
    )

    counts = audit["split"].value_counts()

    expected_counts = {
        "train": 485,
        "validation": 347,
        "test": 234,
        "EXCLUDED": 52,
    }

    for category, expected in expected_counts.items():
        actual = int(counts.get(category, 0))

        check(
            f"{category} record count",
            actual == expected,
            f"Expected {expected}, found {actual}"
        )

    check(
        "No unexpected audit split categories",
        set(counts.index).issubset(
            set(expected_counts)
        ),
        sorted(counts.index.tolist())
    )

    check(
        "Unique audit row identifiers",
        audit["audit_row_id"].is_unique
        and split["audit_row_id"].is_unique
    )

    audit_retained = audit[
        audit["split"].isin(PARTITIONS)
    ]

    check(
        "Saved split matches retained audit records",
        set(split["audit_row_id"])
        == set(audit_retained["audit_row_id"])
    )

    # Compare assignments, not only row IDs.
    a = audit_retained.set_index(
        "audit_row_id"
    )["split"]

    b = split.set_index(
        "audit_row_id"
    )["split"]

    check(
        "Retained partition assignments agree",
        a.sort_index().equals(b.sort_index())
    )

    # Independently inspect duplicate pixels.
    duplicate_groups = source.groupby(
        PIXEL_KEYS
    )["class_id"].nunique()

    check(
        "No conflicting labels within pixel-year",
        int((duplicate_groups > 1).sum()) == 0
    )

    expected_unique_pixels = source.drop_duplicates(
        subset=PIXEL_KEYS
    )

    check(
        "Unique pixel-year record accounting",
        len(expected_unique_pixels) == 1117
        and len(duplicates) == 1
        and len(split) + int(counts.get("EXCLUDED", 0))
        == len(source)
    )

    duplicate_audit = audit[
        audit["exclusion_reason"]
        == "DUPLICATE_PIXEL_YEAR"
    ]

    check(
        "Duplicate exclusion audit is consistent",
        set(duplicate_audit["audit_row_id"])
        == set(duplicates["audit_row_id"])
    )

    # Recalculate naming mismatches independently.
    id_year = pd.to_numeric(
        audit["sample_id"].astype(str).str.extract(
            r"^(\d{4})_",
            expand=False
        ),
        errors="coerce"
    )

    mismatch = (
        id_year.isna()
        | (id_year != audit["year"])
    )

    mismatch_ids = sorted(
        audit.loc[mismatch, "sample_id"].tolist()
    )

    check(
        "Year-prefix discrepancies documented",
        int(mismatch.sum()) == 4,
        mismatch_ids
    )

    print("\n========== SPATIAL VALIDATION ==========")

    with rasterio.open(RASTER_FILE) as raster:
        check(
            "Raster CRS",
            str(raster.crs) == "EPSG:32644",
            str(raster.crs)
        )

        check(
            "Raster resolution",
            np.allclose(raster.res, (30, 30)),
            raster.res
        )

        check(
            "Raster grid dimensions",
            (raster.width, raster.height) == (531, 331)
        )

        for name, frame in [
            ("split", split),
            ("audit", audit),
        ]:
            valid_pixels = (
                frame["raster_row"].between(
                    0, raster.height - 1
                ).all()
                and frame["raster_col"].between(
                    0, raster.width - 1
                ).all()
            )

            check(
                f"{name} pixel indices inside raster",
                valid_pixels
            )

        # Independently derive actual satellite pixel
        # centres from the Landsat grid.
        stored_x = split["pixel_center_x"].to_numpy()
        stored_y = split["pixel_center_y"].to_numpy()

        split = add_pixel_centres(
            split, raster.transform
        )

        audit = add_pixel_centres(
            audit, raster.transform
        )

        check(
            "Saved pixel centres match raster geometry",
            np.allclose(
                stored_x,
                split["pixel_center_x"],
                rtol=0,
                atol=1e-5
            )
            and np.allclose(
                stored_y,
                split["pixel_center_y"],
                rtol=0,
                atol=1e-5
            )
        )

    check(
        "Retained pixel-year records are unique",
        not split.duplicated(
            subset=PIXEL_KEYS
        ).any()
    )

    # Verify spatial blocks independently, using
    # the original projected reference coordinates.
    split["calculated_block"] = (
        np.floor(
            split["x_utm"] / BLOCK_SIZE_M
        ).astype(int).astype(str)
        + "_"
        + np.floor(
            split["y_utm"] / BLOCK_SIZE_M
        ).astype(int).astype(str)
    )

    check(
        "Recorded spatial groups are correct",
        (
            split["calculated_block"]
            == split["spatial_group"]
        ).all()
    )

    groups = {
        name: set(
            split.loc[
                split["split"] == name,
                "calculated_block"
            ]
        )
        for name in PARTITIONS
    }

    for first, second in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        overlap = groups[first] & groups[second]

        check(
            f"No shared spatial blocks: {first}/{second}",
            len(overlap) == 0,
            f"Overlapping blocks: {len(overlap)}"
        )

    partitions = {
        name: split[
            split["split"] == name
        ].copy()
        for name in PARTITIONS
    }

    for first, second in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        distance = minimum_distance(
            partitions[first],
            partitions[second]
        )

        details[f"{first}_vs_{second}_distance_m"] = (
            round(distance, 4)
        )

        check(
            f"150 m pixel-centre separation: "
            f"{first}/{second}",
            np.isfinite(distance)
            and distance >= BUFFER_M - 1e-7,
            f"Minimum distance: {distance:.2f} m"
        )

    print("\n========== CLASS/YEAR COVERAGE ==========")

    for name, frame in partitions.items():
        counts_by_class_year = class_year_count(frame)

        minimum = int(counts_by_class_year.min())

        details[f"{name}_minimum_class_year"] = minimum

        check(
            f"{name}: minimum class/year coverage",
            minimum >= MIN_COUNTS[name],
            f"Minimum: {minimum}"
        )

        check(
            f"{name}: all historical years represented",
            set(frame["year"]) == set(YEARS)
        )

        check(
            f"{name}: all six classes represented",
            set(frame["class_id"]) == set(CLASSES)
        )

    print("\n========== BUFFER EXCLUSION AUDIT ==========")

    excluded_validation = audit[
        audit["exclusion_reason"]
        == "WITHIN_150M_OF_TEST"
    ]

    excluded_training = audit[
        audit["exclusion_reason"]
        == "WITHIN_150M_OF_EVALUATION"
    ]

    check(
        "Validation buffer exclusion count",
        len(excluded_validation) == 12
    )

    check(
        "Training buffer exclusion count",
        len(excluded_training) == 39
    )

    if not excluded_validation.empty:
        nearest = cKDTree(
            partitions["test"][
                ["pixel_center_x", "pixel_center_y"]
            ].to_numpy()
        ).query(
            excluded_validation[
                ["pixel_center_x", "pixel_center_y"]
            ].to_numpy()
        )[0]

        check(
            "Validation exclusions fall inside test buffer",
            np.all(nearest < BUFFER_M)
        )

    protected = pd.concat(
        [
            partitions["validation"],
            partitions["test"],
        ],
        ignore_index=True
    )

    if not excluded_training.empty:
        nearest = cKDTree(
            protected[
                ["pixel_center_x", "pixel_center_y"]
            ].to_numpy()
        ).query(
            excluded_training[
                ["pixel_center_x", "pixel_center_y"]
            ].to_numpy()
        )[0]

        check(
            "Training exclusions fall inside evaluation buffer",
            np.all(nearest < BUFFER_M)
        )

    check(
        "All 52 exclusions have documented reasons",
        (
            audit.loc[
                audit["split"] == "EXCLUDED",
                "exclusion_reason"
            ].isin([
                "DUPLICATE_PIXEL_YEAR",
                "WITHIN_150M_OF_TEST",
                "WITHIN_150M_OF_EVALUATION",
            ])
        ).all()
    )

    print("\n========== FINAL RESULT ==========")

    failures = [
        item for item in checks
        if not item["passed"]
    ]

    passed_count = len(checks) - len(failures)

    status = (
        "PASS"
        if not failures
        else "FAIL"
    )

    print(f"Checks passed: {passed_count}/{len(checks)}")
    print(f"Checks failed: {len(failures)}")

    print(
        f"INDEPENDENT SPATIAL VALIDATION: {status}"
    )

    print(
        "TEMPORAL LABEL QUALITY ASSURANCE: PENDING"
    )

    report = {
        "component": "Part 3",
        "validation": "Independent spatial split validation",
        "validation_timestamp_utc":
            datetime.now(timezone.utc).isoformat(),
        "status": status,
        "buffer_distance_m": BUFFER_M,
        "spatial_block_size_m": BLOCK_SIZE_M,
        "records_checked": len(audit),
        "retained_records": len(split),
        "passed_checks": passed_count,
        "failed_checks": len(failures),
        "checks": checks,
        "details": details,
        "limitations": [
            "Spatial buffering reduces geographical proximity "
            "but does not prove statistical independence.",
            "Four source-name/year mismatches require "
            "historical temporal-label review.",
            "A class-year may contain relatively few "
            "held-out observations.",
            "The final test partition must not be used "
            "for classifier tuning or split reselection.",
        ],
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    json_path = (
        REPORT_DIR
        / "script03b_independent_spatial_validation.json"
    )

    text_path = (
        REPORT_DIR
        / "script03b_independent_spatial_validation.txt"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=2
        )

    lines = [
        "PART 3 - INDEPENDENT SPATIAL VALIDATION",
        "=" * 60,
        f"Status: {status}",
        f"Checks passed: {passed_count}/{len(checks)}",
        "",
    ]

    for item in checks:
        lines.append(
            f"{'PASS' if item['passed'] else 'FAIL'} | "
            f"{item['name']}"
        )

        if item["detail"]:
            lines.append(
                f"    {item['detail']}"
            )

    lines.extend([
        "",
        "Temporal-label quality assurance: PENDING",
        "150 m buffering does not prove spatial independence.",
        "Final test partition is reserved for final evaluation.",
    ])

    text_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    print("\nSAVED VALIDATION EVIDENCE:")
    print(json_path)
    print(text_path)

    if failures:
        raise SystemExit(
            "Independent validation failed. "
            "Review the saved report before proceeding."
        )


if __name__ == "__main__":
    main()
