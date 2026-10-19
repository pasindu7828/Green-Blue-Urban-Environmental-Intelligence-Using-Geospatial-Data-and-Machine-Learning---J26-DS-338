
"""
Urban Green Research - Data Service
Project: J26-DS-338

Read-only access to validated Section 05 dashboard exports.

No model retraining, score recalculation, or modification
of research outputs is performed here.
"""

import csv
import hashlib
import json
import math

from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.config import (
    URBAN_GREEN_MANIFEST_FILE,
    URBAN_GREEN_METADATA_FILE,
    URBAN_GREEN_GRID_RECORDS_FILE,
    URBAN_GREEN_GRID_MAP_FILE,
    URBAN_GREEN_GN_SUMMARY_FILE,
    URBAN_GREEN_GN_MAP_FILE,
    URBAN_GREEN_HOTSPOT_GROUPS_FILE,
    URBAN_GREEN_HOTSPOT_MAP_FILE,
)


# ============================================================
# COLUMN TYPE HANDLING
# ============================================================

# Identifiers and research classifications must stay as text.
TEXT_COLUMNS = {
    "grid_id",
    "gn_id",
    "gn_name",
    "gn_assignment_status",
    "pressure_domain_status",
    "review_rule_id",
    "review_stream",
    "review_action",
    "explanation_of_rule_trigger",
    "historical_context_type",
    "dci_screen_status",
    "dci_cluster_id",
    "cp_screen_status",
    "cp_cluster_id",
    "gn_summary_method",
    "index_type",
    "cluster_id",
    "screen_status",
    "dominant_intersected_gn_id",
    "dominant_intersected_gn_name",
    "intersected_gn_names",
}

BOOLEAN_COLUMNS = {
    "pressure_threshold_flag_frozen",
    "dci_candidate_hotspot",
    "cp_candidate_hotspot",
    "is_official_planning_classification",
    "development_approval_inferred",
    "is_candidate_hotspot",
    "is_statistically_significant_hotspot",
}


def _convert_csv_value(column: str, value: str | None) -> Any:
    """
    Preserve unavailable research values as None (JSON null).

    Convert known boolean columns to bool.
    Convert numeric measurements to JSON numbers.
    Keep identifiers and classifications as text.
    """
    if value is None:
        return None

    value = value.strip()

    if not value or value.lower() in {"nan", "null", "none"}:
        return None

    if column in TEXT_COLUMNS:
        return value

    
    if column in BOOLEAN_COLUMNS:
        normalized = value.lower()

        if normalized in {"true", "1", "1.0"}:
            return True

        if normalized in {"false", "0", "0.0"}:
            return False

        raise ValueError(
            f"Unexpected boolean value in {column}: {value!r}"
        )


    # Preserve integer-valued measurements as integers.
    if value.lstrip("+-").isdigit():
        return int(value)

    # Convert other finite numeric values to floats.
    try:
        number = float(value)
    except ValueError:
        return value

    if not math.isfinite(number):
        return None

    return number


# ============================================================
# READ-ONLY FILE LOADERS
# ============================================================

@lru_cache(maxsize=4)
def _load_csv(path: Path) -> tuple[dict[str, Any], ...]:
    """
    Read a frozen dashboard CSV once and cache its records.

    Cached data is refreshed when the backend process restarts.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Urban Green data file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")

        records = []

        for row in reader:
            record = {
                column: _convert_csv_value(column, value)
                for column, value in row.items()
            }

            records.append(record)

    return tuple(records)


@lru_cache(maxsize=2)
def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Urban Green JSON file not found: {path}"
        )

    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


# ============================================================
# MANIFEST AND RESEARCH METADATA
# ============================================================

def get_manifest() -> dict[str, Any]:
    return _load_json(URBAN_GREEN_MANIFEST_FILE)


def get_metadata() -> dict[str, Any]:
    return _load_json(URBAN_GREEN_METADATA_FILE)


# ============================================================
# GRID DATA
# ============================================================

def get_grid_records() -> tuple[dict[str, Any], ...]:
    return _load_csv(URBAN_GREEN_GRID_RECORDS_FILE)


@lru_cache(maxsize=1)
def _grid_index() -> dict[str, dict[str, Any]]:
    return {
        record["grid_id"]: record
        for record in get_grid_records()
    }


def get_grid_record(grid_id: str) -> dict[str, Any] | None:
    """
    Look up an existing grid without calculating new scores.
    """
    return _grid_index().get(grid_id)


# ============================================================
# GN DIVISION DATA
# ============================================================

def get_gn_summaries() -> tuple[dict[str, Any], ...]:
    return _load_csv(URBAN_GREEN_GN_SUMMARY_FILE)


@lru_cache(maxsize=1)
def _gn_index() -> dict[str, dict[str, Any]]:
    return {
        record["gn_id"]: record
        for record in get_gn_summaries()
    }


def get_gn_summary(gn_id: str) -> dict[str, Any] | None:
    return _gn_index().get(gn_id)


# ============================================================
# EXPLORATORY HOTSPOT GROUPS
# ============================================================

def get_hotspot_groups() -> tuple[dict[str, Any], ...]:
    """
    All screened spatial groups, including non-candidates.

    These are exploratory groups, not statistically
    significant hotspots.
    """
    return _load_csv(URBAN_GREEN_HOTSPOT_GROUPS_FILE)


def get_candidate_hotspot_groups() -> list[dict[str, Any]]:
    return [
        record
        for record in get_hotspot_groups()
        if record["is_candidate_hotspot"] is True
    ]


# ============================================================
# MAP FILE PATHS
# ============================================================

MAP_FILES = {
    "grid": URBAN_GREEN_GRID_MAP_FILE,
    "gn": URBAN_GREEN_GN_MAP_FILE,
    "hotspot": URBAN_GREEN_HOTSPOT_MAP_FILE,
}


def get_map_path(map_type: str) -> Path:
    """
    Return an approved existing GeoJSON file.

    The API router can serve the file directly without
    changing geometries or coordinate systems.
    """
    if map_type not in MAP_FILES:
        raise ValueError(
            f"Unsupported Urban Green map type: {map_type}"
        )

    path = MAP_FILES[map_type]

    if not path.is_file():
        raise FileNotFoundError(
            f"Urban Green map not found: {path}"
        )

    return path


# ============================================================
# MANIFEST INTEGRITY VALIDATION
# ============================================================

def validate_dashboard_exports() -> dict[str, Any]:
    """
    Validate every file listed in the frozen manifest.

    Verify both file size and SHA-256 hash.
    Does not write or modify files.
    """
    manifest = get_manifest()
    base = URBAN_GREEN_MANIFEST_FILE.parent

    file_results = []

    for filename, entry in manifest["files"].items():
        path = base / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Required dashboard export missing: {path}"
            )

        actual_size = path.stat().st_size
        expected_size = entry["bytes"]

        if actual_size != expected_size:
            raise ValueError(
                f"File size mismatch: {filename}"
            )

        digest = hashlib.sha256()

        with path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)

        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch: {filename}"
            )

        file_results.append({
            "file": filename,
            "status": "verified",
        })

    return {
        "status": "ok",
        "component": "Urban Green Research",
        "integrity_check": "SHA-256 and file size",
        "files_verified": len(file_results),
        "files": file_results,
    }
