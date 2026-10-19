
"""
Urban Green Research - Historical Green-Cover Service
Project: J26-DS-338
Section 01: LULC and Green-Cover Transition

Reads finalized Step 08, 09, 10A, 10B and 10C reports.

No retraining, raster modification, model inference,
transition recalculation or output rewriting.
"""

import csv
import math
import re

from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.config import URBAN_GREEN_ROOT


# ============================================================
# FINAL RESEARCH REPORT PATHS
# ============================================================

HISTORICAL_REPORTS_DIR = (
    URBAN_GREEN_ROOT / "outputs" / "reports"
)

REPORT_FILES = {
    "annual_class_area":
        "08_annual_lulc_class_area_summary.csv",

    "annual_qa":
        "09_annual_lulc_qa_summary.csv",

    "annual_trajectory":
        "10a_green_cover_annual_trajectory.csv",

    "key_summary":
        "10a_green_cover_key_summary.csv",

    "period_status":
        "10a_lulc_period_status_summary.csv",

    "transition_analysis":
        "10b_consensus_transition_analysis_summary.csv",

    "transition_summary":
        "10b_green_transition_summary.csv",

    "transition_breakdown":
        "10b_green_transition_class_breakdown.csv",

    "reporting_decision":
        "10c_transition_reporting_method_decision.csv",

    "confidence_summary":
        "10c_green_transition_confidence_summary.csv",
}


# ============================================================
# DATA TYPE CONVERSION
# ============================================================

_INTEGER_PATTERN = re.compile(
    r"^[+-]?(?:0|[1-9][0-9]*)$"
)


def _convert_value(value: str | None) -> Any:
    """
    Convert CSV values to JSON-compatible Python values.

    Blank values -> None
    True/False -> bool
    Numeric values -> int/float
    Other values -> str

    Numeric-looking identifiers with leading zeros remain
    text rather than losing their original representation.
    """
    if value is None:
        return None

    value = value.strip()

    if not value:
        return None

    lower = value.lower()

    if lower in {"null", "none", "nan"}:
        return None

    if lower == "true":
        return True

    if lower == "false":
        return False

    if _INTEGER_PATTERN.fullmatch(value):
        return int(value)

    try:
        number = float(value)
    except ValueError:
        return value

    if not math.isfinite(number):
        return None

    return number


# ============================================================
# FROZEN RESEARCH REPORT LOADER
# ============================================================

@lru_cache(maxsize=len(REPORT_FILES))
def _load_report(
    report_key: str,
) -> tuple[dict[str, Any], ...]:
    """
    Read an approved research report once per backend process.

    No arbitrary file paths are accepted.
    """
    if report_key not in REPORT_FILES:
        raise ValueError(
            f"Unknown historical report: {report_key}"
        )

    path: Path = (
        HISTORICAL_REPORTS_DIR
        / REPORT_FILES[report_key]
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Historical report missing: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(
                f"CSV has no header: {path.name}"
            )

        records = []

        for row in reader:
            record = {
                column: _convert_value(value)
                for column, value in row.items()
            }

            records.append(record)

    return tuple(records)


def _single_record(
    report_key: str,
) -> dict[str, Any]:
    records = _load_report(report_key)

    if len(records) != 1:
        raise ValueError(
            f"Expected one row in {report_key}, "
            f"found {len(records)}"
        )

    return dict(records[0])


# ============================================================
# 1. ANNUAL GREEN-COVER TRAJECTORY
# ============================================================

def get_annual_trajectory(
    core_only: bool = False,
) -> list[dict[str, Any]]:
    """
    Complete annual record: 2010-2026.

    The validated primary trend contains only years
    explicitly marked included_in_core_trend=True.
    """
    records = _load_report("annual_trajectory")

    if core_only:
        records = tuple(
            record for record in records
            if record["included_in_core_trend"] is True
        )

    return [dict(record) for record in records]


# ============================================================
# 2. CORE GREEN-COVER AND BUILT-UP SUMMARY
# ============================================================

def get_green_key_summary() -> dict[str, Any]:
    return _single_record("key_summary")


# ============================================================
# 3. ANNUAL LULC CLASS AREAS
# ============================================================

def get_annual_class_areas() -> list[dict[str, Any]]:
    """
    Return original raw/stabilized annual LULC
    area-statistic records without recomputation.
    """
    return [
        dict(record)
        for record in _load_report("annual_class_area")
    ]


# ============================================================
# 4. ANNUAL RESEARCH QUALITY ASSURANCE
# ============================================================

def get_annual_lulc_qa() -> dict[str, Any]:
    return _single_record("annual_qa")


# ============================================================
# 5. ANNUAL PERIOD STATUS
# ============================================================

def get_period_status() -> list[dict[str, Any]]:
    return [
        dict(record)
        for record in _load_report("period_status")
    ]


# ============================================================
# 6. TRANSITION METHODOLOGY
# ============================================================

def get_transition_reporting_decision() -> dict[str, Any]:
    return _single_record("reporting_decision")


# ============================================================
# 7. HISTORICAL TRANSITION COMPARISON
# ============================================================

def get_transition_comparison() -> list[dict[str, Any]]:
    """
    Return both existing consensus methods.

    ROBUST_2_OF_3 is the locked primary result.
    STRICT_3_OF_3 is a conservative confirmation subset.
    """
    return [
        dict(record)
        for record in _load_report("transition_summary")
    ]


def get_transition_analysis() -> dict[str, Any]:
    return _single_record("transition_analysis")


def get_transition_breakdown() -> list[dict[str, Any]]:
    return [
        dict(record)
        for record in _load_report("transition_breakdown")
    ]


def get_transition_confidence() -> list[dict[str, Any]]:
    return [
        dict(record)
        for record in _load_report("confidence_summary")
    ]


# ============================================================
# 8. HISTORICAL DATA HEALTH
# ============================================================

def validate_historical_reports() -> dict[str, Any]:
    """
    Verify research report availability and methodological
    consistency without modifying original files.
    """
    for report_key in REPORT_FILES:
        _load_report(report_key)

    qa = get_annual_lulc_qa()
    decision = get_transition_reporting_decision()
    summary = get_green_key_summary()

    all_years = get_annual_trajectory()
    core_years = get_annual_trajectory(core_only=True)

    transitions = get_transition_comparison()

    if qa["overall_step09_status"] != "PASS":
        raise ValueError("Annual LULC QA is not PASS")

    if decision["transition_method_locked"] is not True:
        raise ValueError("Transition method is not locked")

    if (
        decision["historical_transition_primary_rule"]
        != "ROBUST_2_OF_3"
    ):
        raise ValueError("Unexpected primary transition method")

    if (
        decision["historical_transition_sensitivity_rule"]
        != "STRICT_3_OF_3"
    ):
        raise ValueError("Unexpected sensitivity method")

    if summary["core_period"] != "2011-2024":
        raise ValueError("Unexpected core trend period")

    if len(all_years) != 17:
        raise ValueError("Unexpected annual record count")

    if len(core_years) != 14:
        raise ValueError("Unexpected core trend year count")

    if len(transitions) != 2:
        raise ValueError("Expected two transition methods")

    return {
        "status": "ok",
        "component": "Urban Green Historical Research",
        "reports_available": len(REPORT_FILES),
        "annual_records": len(all_years),
        "core_trend_records": len(core_years),
        "core_period": summary["core_period"],
        "primary_transition_method": (
            decision["historical_transition_primary_rule"]
        ),
        "sensitivity_transition_method": (
            decision["historical_transition_sensitivity_rule"]
        ),
        "annual_lulc_qa": qa["overall_step09_status"],
        "mode": "read-only",
    }
