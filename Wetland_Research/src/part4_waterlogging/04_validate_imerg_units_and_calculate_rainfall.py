
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import csv
import json
import math
import os

import netCDF4
import numpy as np


# ============================================================
# PART 4 - SEASONAL WATERLOGGING SUSCEPTIBILITY
# SCRIPT 04 - IMERG UNITS AND RAINFALL DEPTH VALIDATION
#
# - Reads original CSV and NetCDF files
# - Confirms rainfall units from NetCDF attributes
# - Uses NetCDF time bounds for interval duration
# - Checks CSV vs NetCDF measurement consistency
# - Calculates event and antecedent rainfall depths
# - Compares daily and half-hourly products
#
# Raw source files are not modified.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
RAIN_DIR = ROOT / "data/raw/rainfall/imerg"

REPORT_DIR = ROOT / "outputs/reports/part4_waterlogging"
METRIC_DIR = ROOT / "outputs/metrics/part4_waterlogging"

EVENTS = {
    "E001": {
        "date": "2018-05-29",
        "start": "2018-05-22",
    },
    "E002": {
        "date": "2018-11-25",
        "start": "2018-11-18",
    },
}

PRODUCTS = {
    "daily": {
        "folder_suffix": "",
        "variable": "GPM_3IMERGDF_07_precipitation",
        "expected_unit": "mm/day",
        "expected_interval_seconds": 86400,
    },
    "half_hourly": {
        "folder_suffix": " half-hourly rainfall",
        "variable": "GPM_3IMERGHH_07_precipitation",
        "expected_unit": "mm/hr",
        "expected_interval_seconds": 1800,
    },
}


def windows_path(path):
    absolute = os.path.abspath(str(path))
    if os.name == "nt" and len(absolute) >= 260:
        return "\\\\?\\" + absolute
    return absolute


def find_one_file(folder, extension):
    with os.scandir(folder) as entries:
        matches = [
            entry.path
            for entry in entries
            if entry.is_file()
            and entry.name.lower().endswith(extension)
        ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one {extension} file in {folder}; "
            f"found {len(matches)}"
        )

    return matches[0]


def read_csv_values(path):
    with open(
        windows_path(path),
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.reader(handle))

    header_index = next(
        (
            i for i, row in enumerate(rows)
            if row and row[0].strip().lower() == "time"
        ),
        None,
    )

    if header_index is None:
        raise ValueError("CSV time column not found")

    results = []

    for row in rows[header_index + 1:]:
        if not row or not any(cell.strip() for cell in row):
            continue

        if len(row) < 2:
            raise ValueError("Incomplete CSV data row")

        timestamp = datetime.fromisoformat(
            row[0].strip().replace("Z", "+00:00")
        )

        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone(
                timezone.utc
            ).replace(tzinfo=None)

        value = float(row[1].strip())

        if not math.isfinite(value) or value < 0:
            raise ValueError(
                "Missing, invalid or negative CSV rainfall"
            )

        results.append((timestamp, value))

    return results


def as_datetime(value, units, calendar):
    converted = netCDF4.num2date(
        value,
        units=units,
        calendar=calendar,
    )

    return datetime(
        converted.year,
        converted.month,
        converted.day,
        converted.hour,
        converted.minute,
        converted.second,
        converted.microsecond,
    )


def read_and_validate_product(event_id, product_name):
    product = PRODUCTS[product_name]
    event = EVENTS[event_id]

    folder_name = (
        event_id + product["folder_suffix"]
    )

    folder = RAIN_DIR / folder_name

    csv_path = find_one_file(folder, ".csv")
    nc_path = find_one_file(folder, ".nc")

    csv_values = read_csv_values(csv_path)

    with netCDF4.Dataset(
        windows_path(nc_path), "r"
    ) as ds:

        variable = ds.variables[product["variable"]]
        unit = getattr(variable, "units", None)

        if unit != product["expected_unit"]:
            raise ValueError(
                f"{event_id} {product_name}: "
                f"Unexpected NetCDF unit {unit!r}"
            )

        values = np.asarray(
            np.ma.filled(variable[:], np.nan),
            dtype=float,
        )

        bounds_var = ds.variables["time_bnds"]

        time_var = ds.variables["time"]

        time_units = getattr(
            bounds_var,
            "units",
            getattr(time_var, "units", None),
        )

        calendar = getattr(
            bounds_var,
            "calendar",
            getattr(time_var, "calendar", "standard"),
        )

        bounds = np.asarray(
            bounds_var[:],
            dtype=float,
        )

        if (
            len(values) != len(csv_values)
            or bounds.shape != (len(values), 2)
        ):
            raise ValueError(
                f"{event_id} {product_name}: "
                "CSV/NetCDF lengths or time bounds differ"
            )

        records = []

        previous_end = None
        normalized_intervals = 0

        for index, ((csv_time, csv_value), nc_value) in (
            enumerate(zip(csv_values, values))
        ):
            if not math.isfinite(nc_value):
                raise ValueError(
                    "Invalid NetCDF precipitation value"
                )

            difference = abs(csv_value - nc_value)

            tolerance = max(
                0.001,
                abs(nc_value) * 0.0001,
            )

            if difference > tolerance:
                raise ValueError(
                    f"{event_id} {product_name}, "
                    f"row {index + 1}: CSV/NetCDF mismatch "
                    f"({csv_value} vs {nc_value})"
                )

            start = as_datetime(
                bounds[index, 0],
                time_units,
                calendar,
            )

            end = as_datetime(
                bounds[index, 1],
                time_units,
                calendar,
            )

            # Some Giovanni NetCDF intervals use an inclusive
            # final second, e.g. 23:59:59 rather than 00:00:00
            # of the following day.
            #
            # Normalize only when the raw interval is exactly
            # one second shorter than the expected interval.

            raw_duration_seconds = (end - start).total_seconds()

            if raw_duration_seconds == (
                product["expected_interval_seconds"] - 1
            ):
                end += timedelta(seconds=1)
                normalized_intervals += 1

            duration_seconds = (
                end - start
            ).total_seconds()

            if abs(
                duration_seconds
                - product["expected_interval_seconds"]
            ) > 1:
                raise ValueError(
                    f"Unexpected time interval: "
                    f"{duration_seconds} seconds"
                )

            if abs(
                (start - csv_time).total_seconds()
            ) > 1:
                raise ValueError(
                    f"CSV timestamp and NetCDF time "
                    f"bounds differ at row {index + 1}"
                )

            if (
                previous_end is not None
                and abs(
                    (start - previous_end).total_seconds()
                ) > 1
            ):
                raise ValueError(
                    "Non-contiguous NetCDF time intervals"
                )

            previous_end = end

            if unit == "mm/day":
                depth_mm = (
                    nc_value * duration_seconds / 86400
                )

            elif unit == "mm/hr":
                depth_mm = (
                    nc_value * duration_seconds / 3600
                )

            else:
                raise ValueError(
                    f"Unsupported rainfall unit: {unit}"
                )

            records.append({
                "event_id": event_id,
                "product": product_name,
                "start": start,
                "end": end,
                "raw_rate": float(nc_value),
                "unit": unit,
                "depth_mm": float(depth_mm),
            })

    expected_start = datetime.fromisoformat(
        event["start"]
    )

    expected_end = (
        datetime.fromisoformat(event["date"])
        + timedelta(days=1)
    )

    if (
        not records
        or records[0]["start"] != expected_start
        or records[-1]["end"] != expected_end
    ):
        raise ValueError(
            f"{event_id} {product_name}: "
            "Unexpected temporal coverage"
        )

    return records, {
        "event_id": event_id,
        "product": product_name,
        "unit": unit,
        "observations": len(records),
        "interval_seconds":
            product["expected_interval_seconds"],
        "csv_netcdf_agreement": "PASS",
        "time_bounds_validation": "PASS",
        "inclusive_end_intervals_normalized": normalized_intervals,
        "time_bounds_normalization_rule": (
            "Added one second to the interpreted interval end "
            "only when the original NetCDF bound represented "
            "an interval exactly one second shorter than "
            "the expected duration."
        ),
        "source_csv": os.path.relpath(
            csv_path, ROOT
        ),
        "source_netcdf": os.path.relpath(
            nc_path, ROOT
        ),
    }


def sum_depth(records, start_date, end_date):
    return sum(
        r["depth_mm"]
        for r in records
        if start_date <= r["start"].date() <= end_date
    )


def round_result(value):
    return round(float(value), 4)


def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    print("=" * 70)
    print("PART 4 - IMERG RAINFALL UNITS AND DEPTH")
    print("SCRIPT 04")
    print("=" * 70)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)

    source_audit = []
    event_summaries = []
    comparison_rows = []

    for event_id, info in EVENTS.items():
        print("\nEVENT:", event_id)

        daily, daily_audit = (
            read_and_validate_product(event_id, "daily")
        )

        half_hourly, hh_audit = (
            read_and_validate_product(
                event_id, "half_hourly"
            )
        )

        source_audit.extend([
            daily_audit,
            hh_audit,
        ])

        event_date = datetime.fromisoformat(
            info["date"]
        ).date()

        prior_3_start = event_date - timedelta(days=3)
        prior_7_start = event_date - timedelta(days=7)
        previous_day = event_date - timedelta(days=1)

        daily_by_date = defaultdict(float)
        hh_by_date = defaultdict(float)

        for record in daily:
            daily_by_date[
                record["start"].date()
            ] += record["depth_mm"]

        for record in half_hourly:
            hh_by_date[
                record["start"].date()
            ] += record["depth_mm"]

        for date in sorted(daily_by_date):
            daily_mm = daily_by_date[date]
            hh_mm = hh_by_date[date]

            difference_mm = hh_mm - daily_mm

            comparison_rows.append({
                "event_id": event_id,
                "date": date.isoformat(),
                "daily_product_mm":
                    round_result(daily_mm),
                "half_hourly_product_mm":
                    round_result(hh_mm),
                "difference_hh_minus_daily_mm":
                    round_result(difference_mm),
                "absolute_difference_mm":
                    round_result(abs(difference_mm)),
            })

        # Each pair covers two consecutive 30-minute
        # intervals, giving a rolling 1-hour depth.
        rolling_1h = [
            half_hourly[i]["depth_mm"]
            + half_hourly[i + 1]["depth_mm"]
            for i in range(len(half_hourly) - 1)
        ]

        summary = {
            "event_id": event_id,
            "event_date": info["date"],
            "daily_event_day_mm": round_result(
                sum_depth(
                    daily, event_date, event_date
                )
            ),
            "daily_prior_3_days_mm": round_result(
                sum_depth(
                    daily, prior_3_start, previous_day
                )
            ),
            "daily_prior_7_days_mm": round_result(
                sum_depth(
                    daily, prior_7_start, previous_day
                )
            ),
            "daily_full_8_days_mm": round_result(
                sum(r["depth_mm"] for r in daily)
            ),
            "hh_event_day_mm": round_result(
                sum_depth(
                    half_hourly, event_date, event_date
                )
            ),
            "hh_prior_3_days_mm": round_result(
                sum_depth(
                    half_hourly,
                    prior_3_start,
                    previous_day,
                )
            ),
            "hh_prior_7_days_mm": round_result(
                sum_depth(
                    half_hourly,
                    prior_7_start,
                    previous_day,
                )
            ),
            "hh_full_8_days_mm": round_result(
                sum(r["depth_mm"] for r in half_hourly)
            ),
            "maximum_30min_depth_mm": round_result(
                max(
                    r["depth_mm"]
                    for r in half_hourly
                )
            ),
            "maximum_1h_depth_mm": round_result(
                max(rolling_1h)
            ),
            "maximum_30min_rate_mm_per_hr":
                round_result(
                    max(
                        r["raw_rate"]
                        for r in half_hourly
                    )
                ),
        }

        event_summaries.append(summary)

        print("Daily units: mm/day")
        print("Half-hourly units: mm/hr")
        print("CSV/NetCDF values: PASS")
        print("NetCDF time intervals: PASS")
        print(
            "Event-day rainfall (daily product):",
            summary["daily_event_day_mm"],
            "mm"
        )
        print(
            "Prior 3 days (daily product):",
            summary["daily_prior_3_days_mm"],
            "mm"
        )
        print(
            "Prior 7 days (daily product):",
            summary["daily_prior_7_days_mm"],
            "mm"
        )
        print(
            "Maximum 1-hour depth:",
            summary["maximum_1h_depth_mm"],
            "mm"
        )

    summary_path = (
        METRIC_DIR
        / "script04_event_rainfall_summary.csv"
    )

    comparison_path = (
        METRIC_DIR
        / "script04_daily_vs_half_hourly.csv"
    )

    report_path = (
        REPORT_DIR
        / "script04_imerg_unit_validation.json"
    )

    write_csv(summary_path, event_summaries)
    write_csv(comparison_path, comparison_rows)

    report = {
        "component": "Part 4 - Seasonal Waterlogging",
        "stage": "IMERG rainfall units and accumulation",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "unit_evidence": source_audit,
        "event_rainfall_summaries": event_summaries,
        "daily_product_comparison":
            comparison_rows,
        "conversion_rules": {
            "daily":
                "mm/day multiplied by interval days",
            "half_hourly":
                "mm/hr multiplied by interval hours",
        },
        "validation_status":
            "UNIT_AND_TIME_INTEGRATION_PASS",
        "scientific_limitations": [
            "Rainfall observations are spatial area "
            "averages, not GN-specific rainfall.",
            "Daily and half-hourly IMERG products "
            "may give different rainfall totals.",
            "Different product estimates are compared "
            "but are not required to be identical.",
            "Rainfall alone does not confirm that "
            "mapped polygons represent waterlogging.",
            "Two event dates are insufficient to "
            "establish a seasonal rainfall climatology.",
        ],
        "raw_data_modified": False,
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("SCRIPT 04 - VALIDATION SUMMARY")
    print("=" * 70)

    print("NetCDF files validated:", len(source_audit))
    print("Event summaries:", len(event_summaries))
    print("Daily product comparisons:",
          len(comparison_rows))

    print("Saved:", summary_path)
    print("Saved:", comparison_path)
    print("Saved:", report_path)

    print("\nUNIT AND TIME-INTEGRATION CHECK: PASS")
    print("EVENT EVIDENCE RELIABILITY: STILL PENDING")
    print("RAW DATA MODIFIED: NO")


if __name__ == "__main__":
    main()
