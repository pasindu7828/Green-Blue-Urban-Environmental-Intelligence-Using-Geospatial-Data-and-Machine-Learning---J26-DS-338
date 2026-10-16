
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import Counter
import csv
import json
import math
import os


# ============================================================
# PART 4 - SEASONAL WATERLOGGING SUSCEPTIBILITY
# SCRIPT 03 - IMERG RAINFALL TIME-SERIES VALIDATION
#
# Validates:
# - CSV metadata and event dates
# - Expected timestamps and sampling intervals
# - Duplicate and missing observations
# - Fill values, invalid numbers, negative values
# - Consistency of rainfall spatial metadata
#
# IMPORTANT:
# Rainfall units are NOT assumed or converted here.
# Original source files are never modified.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAIN_DIR = ROOT / "data/raw/rainfall/imerg"

REPORT_DIR = ROOT / "outputs/reports/part4_waterlogging"
METRIC_DIR = ROOT / "outputs/metrics/part4_waterlogging"

DATASETS = {
    "E001_daily": {
        "folder": "E001",
        "start": "2018-05-22",
        "end": "2018-05-29",
        "event_date": "2018-05-29",
        "minutes": 1440,
    },
    "E001_half_hourly": {
        "folder": "E001 half-hourly rainfall",
        "start": "2018-05-22",
        "end": "2018-05-29",
        "event_date": "2018-05-29",
        "minutes": 30,
    },
    "E002_daily": {
        "folder": "E002",
        "start": "2018-11-18",
        "end": "2018-11-25",
        "event_date": "2018-11-25",
        "minutes": 1440,
    },
    "E002_half_hourly": {
        "folder": "E002 half-hourly rainfall",
        "start": "2018-11-18",
        "end": "2018-11-25",
        "event_date": "2018-11-25",
        "minutes": 30,
    },
}


def long_path(path):
    """Support existing long Windows filenames."""
    absolute = os.path.abspath(str(path))

    if os.name == "nt" and len(absolute) >= 260:
        return "\\\\?\\" + absolute

    return absolute


def find_csv(folder):
    """Find CSV files without Windows pathlib long-path problems."""
    with os.scandir(folder) as entries:
        files = sorted(
            [
                entry.path
                for entry in entries
                if entry.is_file()
                and entry.name.lower().endswith(".csv")
            ]
        )

    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one CSV in {folder}; "
            f"found {len(files)}"
        )

    return files[0]


def load_giovanni_csv(path):
    """Separate Giovanni metadata from rainfall records."""
    with open(
        long_path(path),
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.reader(handle))

    metadata = {}
    header_index = None

    for index, row in enumerate(rows):
        if not row:
            continue

        first = row[0].strip()

        if first.lower() == "time":
            header_index = index
            break

        if first.endswith(":") and len(row) >= 2:
            metadata[first.rstrip(":")] = row[1].strip()

    if header_index is None:
        raise ValueError("Time-series header not found")

    header = rows[header_index]

    if len(header) < 2:
        raise ValueError("Rainfall value column not found")

    data = []

    for row_number, row in enumerate(
        rows[header_index + 1:],
        start=header_index + 2,
    ):
        if not row or all(not cell.strip() for cell in row):
            continue

        if len(row) < 2:
            raise ValueError(
                f"Malformed data at CSV row {row_number}"
            )

        data.append({
            "timestamp": row[0].strip(),
            "raw_value": row[1].strip(),
            "source_row": row_number,
        })

    return metadata, header[1].strip(), data


def expected_timestamps(info):
    start = datetime.fromisoformat(info["start"])
    final_day = datetime.fromisoformat(info["end"])

    # Include the entire final day.
    end_exclusive = final_day + timedelta(days=1)

    interval = timedelta(minutes=info["minutes"])

    timestamps = []
    current = start

    while current < end_exclusive:
        timestamps.append(current)
        current += interval

    return timestamps


def main():
    print("=" * 70)
    print("PART 4 - IMERG RAINFALL VALIDATION")
    print("SCRIPT 03")
    print("=" * 70)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    observation_rows = []
    global_issues = []
    spatial_boxes = set()

    for name, info in DATASETS.items():
        print("\n" + "-" * 70)
        print("DATASET:", name)

        folder = RAIN_DIR / info["folder"]
        path = find_csv(folder)

        metadata, value_column, data = (
            load_giovanni_csv(path)
        )

        expected = expected_timestamps(info)
        expected_set = set(expected)

        parsed_times = []
        numeric_values = []
        dataset_issues = []

        missing_value_count = 0
        invalid_value_count = 0
        negative_value_count = 0

        fill_string = next(
            (
                value
                for key, value in metadata.items()
                if key.startswith("Fill Value")
            ),
            "-9999.9",
        )

        fill_value = float(fill_string)

        for record in data:
            raw_timestamp = record["timestamp"]
            raw_value = record["raw_value"]

            try:
                timestamp = datetime.fromisoformat(
                    raw_timestamp.replace("Z", "+00:00")
                )

                if timestamp.tzinfo is not None:
                    timestamp = timestamp.astimezone(
                        timezone.utc
                    ).replace(tzinfo=None)

                parsed_times.append(timestamp)

            except ValueError:
                dataset_issues.append(
                    f"Invalid timestamp: {raw_timestamp}"
                )
                timestamp = None

            value = None
            status = "valid"

            try:
                numeric = float(raw_value)

                if (
                    not math.isfinite(numeric)
                    or numeric == fill_value
                ):
                    status = "missing"
                    missing_value_count += 1

                elif numeric < 0:
                    status = "negative"
                    negative_value_count += 1

                else:
                    value = numeric
                    numeric_values.append(numeric)

            except ValueError:
                status = "invalid"
                invalid_value_count += 1

            observation_rows.append({
                "dataset": name,
                "event_date": info["event_date"],
                "timestamp": raw_timestamp,
                "raw_value": raw_value,
                "validated_numeric_value": value,
                "value_status": status,
                "source_row": record["source_row"],
            })

        frequency = Counter(parsed_times)

        duplicate_count = sum(
            count - 1
            for count in frequency.values()
            if count > 1
        )

        observed_set = set(parsed_times)

        missing_times = sorted(
            expected_set - observed_set
        )
        extra_times = sorted(
            observed_set - expected_set
        )

        if duplicate_count:
            dataset_issues.append(
                f"{duplicate_count} duplicate timestamps"
            )

        if missing_times:
            dataset_issues.append(
                f"{len(missing_times)} missing timestamps"
            )

        if extra_times:
            dataset_issues.append(
                f"{len(extra_times)} unexpected timestamps"
            )

        if missing_value_count:
            dataset_issues.append(
                f"{missing_value_count} missing/fill measurements"
            )

        if invalid_value_count:
            dataset_issues.append(
                f"{invalid_value_count} invalid measurements"
            )

        if negative_value_count:
            dataset_issues.append(
                f"{negative_value_count} negative measurements"
            )

        expected_start = (
            info["start"] + "T00:00:00Z"
        )
        expected_end = (
            info["end"] + "T23:59:59Z"
        )

        if metadata.get("User Start Date") != expected_start:
            dataset_issues.append(
                "Unexpected requested start date"
            )

        if metadata.get("User End Date") != expected_end:
            dataset_issues.append(
                "Unexpected requested end date"
            )

        bbox = metadata.get("Data Bounding Box", "")
        spatial_boxes.add(bbox)

        summary = {
            "dataset": name,
            "event_date": info["event_date"],
            "source_file": os.path.relpath(
                path, ROOT
            ),
            "product_description": metadata.get(
                "Title", ""
            ),
            "value_column": value_column,
            "requested_start": metadata.get(
                "User Start Date", ""
            ),
            "requested_end": metadata.get(
                "User End Date", ""
            ),
            "requested_bbox": metadata.get(
                "User Bounding Box", ""
            ),
            "actual_data_bbox": bbox,
            "sampling_minutes": info["minutes"],
            "expected_observations": len(expected),
            "actual_rows": len(data),
            "unique_timestamps": len(observed_set),
            "duplicate_timestamps": duplicate_count,
            "missing_timestamps": len(missing_times),
            "unexpected_timestamps": len(extra_times),
            "fill_value": fill_value,
            "missing_measurements": missing_value_count,
            "invalid_measurements": invalid_value_count,
            "negative_measurements": negative_value_count,
            "valid_measurements": len(numeric_values),
            "min_raw_value": (
                min(numeric_values)
                if numeric_values else None
            ),
            "max_raw_value": (
                max(numeric_values)
                if numeric_values else None
            ),
            "rainfall_units": "NOT_YET_VERIFIED",
            "rainfall_accumulation": None,
            "issues": dataset_issues,
        }

        summaries.append(summary)

        global_issues.extend(
            f"{name}: {issue}"
            for issue in dataset_issues
        )

        print(
            "Expected timestamps:",
            len(expected)
        )
        print("Actual rows:", len(data))
        print("Duplicate timestamps:", duplicate_count)
        print("Missing timestamps:", len(missing_times))
        print("Unexpected timestamps:", len(extra_times))
        print("Fill/missing values:", missing_value_count)
        print("Invalid values:", invalid_value_count)
        print("Negative values:", negative_value_count)
        print(
            "Raw value range:",
            summary["min_raw_value"],
            "to",
            summary["max_raw_value"]
        )
        print("Data bounding box:", bbox)
        print("Issues:", len(dataset_issues))

    if len(spatial_boxes) != 1:
        global_issues.append(
            "Rainfall datasets have inconsistent data bounding boxes"
        )

    # --------------------------------------------------------
    # Save permanent validation evidence
    # --------------------------------------------------------

    summary_csv = (
        METRIC_DIR / "script03_imerg_dataset_summary.csv"
    )

    observations_csv = (
        METRIC_DIR / "script03_imerg_observation_audit.csv"
    )

    report_json = (
        REPORT_DIR / "script03_imerg_validation.json"
    )

    with summary_csv.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            key for key in summaries[0]
            if key != "issues"
        ]
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames
        )
        writer.writeheader()

        for item in summaries:
            writer.writerow({
                key: item[key] for key in fieldnames
            })

    with observations_csv.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "event_date",
                "timestamp",
                "raw_value",
                "validated_numeric_value",
                "value_status",
                "source_row",
            ],
        )
        writer.writeheader()
        writer.writerows(observation_rows)

    report = {
        "component": "Part 4 - Seasonal Waterlogging",
        "stage": "IMERG CSV temporal and value validation",
        "generated_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "datasets": summaries,
        "issues": global_issues,
        "data_bounding_boxes": sorted(spatial_boxes),
        "validation_status": (
            "REVIEW_REQUIRED"
            if global_issues
            else "TIME_SERIES_STRUCTURE_PASS_"
                 "UNITS_PENDING"
        ),
        "scientific_limitations": [
            "Rainfall measurement units are not yet verified.",
            "Rainfall accumulation has not been calculated.",
            "NetCDF variable attributes are not yet audited.",
            "Area-average data must not be presented as "
            "GN-level rainfall observations.",
            "Spatial coverage relative to all 56 GN areas "
            "is not yet validated.",
            "Rainfall association with mapped inundation "
            "does not establish causation.",
        ],
        "raw_data_modified": False,
    }

    report_json.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("SCRIPT 03 - VALIDATION SUMMARY")
    print("=" * 70)

    print("Datasets examined:", len(summaries))
    print(
        "Total observations examined:",
        len(observation_rows)
    )
    print("Issues identified:", len(global_issues))

    for issue in global_issues:
        print("REVIEW:", issue)

    print("Saved:", summary_csv)
    print("Saved:", observations_csv)
    print("Saved:", report_json)

    print(
        "\nVALIDATION STATUS:",
        report["validation_status"]
    )
    print("RAINFALL UNITS: PENDING")
    print("RAW DATA MODIFIED: NO")


if __name__ == "__main__":
    main()
