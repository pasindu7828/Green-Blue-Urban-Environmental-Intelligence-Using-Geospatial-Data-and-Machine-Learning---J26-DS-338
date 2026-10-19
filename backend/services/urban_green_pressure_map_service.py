
"""
J26-DS-338 - Urban Green Research

STEP 5D.6 - Pressure GIS Map Service

Read-only access to the validated Step 5D.5
Urbanization Pressure GeoJSON and metadata.

No GIS conversion, model inference,
score recalculation or file modification.
"""

import json
import math

from collections import Counter
from copy import deepcopy
from functools import lru_cache

from backend.config import URBAN_GREEN_ROOT


MAP_DIRECTORY = (
    URBAN_GREEN_ROOT
    / "outputs"
    / "dashboard"
    / "urbanization_pressure"
)

MAP_FILE = MAP_DIRECTORY / "pressure_map.geojson"
METADATA_FILE = MAP_DIRECTORY / "metadata.json"

MAP_API_URL = "/api/urban-green/pressure/map"

EXPECTED_CATEGORIES = {
    "NOT_IN_INFERENCE_DOMAIN": 4620,
    "LOW_RELATIVE": 2310,
    "MODERATE_RELATIVE": 1155,
    "ELEVATED_RELATIVE": 693,
    "HIGH_RELATIVE": 207,
    "THRESHOLD_FLAGGED": 255,
}


@lru_cache(maxsize=1)
def _load_verified_contract():
    """
    Validate exported map and metadata once,
    then retain only the metadata in memory.

    The GeoJSON itself is delivered as a file,
    rather than being serialized on each request.
    """

    for path in (MAP_FILE, METADATA_FILE):
        if not path.is_file():
            raise FileNotFoundError(path)

    with METADATA_FILE.open(
        "r", encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    if not isinstance(metadata, dict):
        raise ValueError("Invalid pressure map metadata")

    required = {
        "permanent_grid_count": 9240,
        "eligible_scored_count": 4620,
        "not_in_inference_domain_count": 4620,
        "threshold_flagged_count": 255,
        "origin_year": 2025,
        "pressure_period": "2026-2028",
        "prediction_window_type": "SINGLE_THREE_YEAR_WINDOW",
        "coordinate_reference_system": "EPSG:4326",
        "geojson_file": "pressure_map.geojson",
        "model": "XGBoost",
        "score_is_calibrated_probability": False,
        "categories_are_probability_bands": False,
    }

    for field, expected in required.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"Pressure metadata mismatch: {field}"
            )

    if not math.isclose(
        metadata.get("operating_threshold", -1),
        0.715,
        abs_tol=1e-10,
        rel_tol=0,
    ):
        raise ValueError("Pressure threshold mismatch")

    if metadata.get("category_counts") != EXPECTED_CATEGORIES:
        raise ValueError("Category metadata mismatch")

    legend = metadata.get("display_legend", [])

    if {
        item["category"] for item in legend
    } != set(EXPECTED_CATEGORIES):
        raise ValueError("Incomplete pressure map legend")

    bounds = metadata.get(
        "bounds_west_south_east_north"
    )

    if not (
        isinstance(bounds, list)
        and len(bounds) == 4
        and all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            for value in bounds
        )
        and -180 <= bounds[0] < bounds[2] <= 180
        and -90 <= bounds[1] < bounds[3] <= 90
    ):
        raise ValueError("Invalid map geographic bounds")

    # Read once for integrity verification.
    with MAP_FILE.open(
        "r", encoding="utf-8"
    ) as file:
        geojson = json.load(file)

    if geojson.get("type") != "FeatureCollection":
        raise ValueError("Invalid GeoJSON collection")

    features = geojson.get("features")

    if not isinstance(features, list) or len(features) != 9240:
        raise ValueError("Incorrect GeoJSON feature count")

    grid_ids = set()
    category_counts = Counter()
    flagged_count = 0

    for feature in features:
        if feature.get("type") != "Feature":
            raise ValueError("Invalid GeoJSON feature")

        props = feature.get("properties")
        geometry = feature.get("geometry")

        if not isinstance(props, dict):
            raise ValueError("Invalid feature properties")

        if (
            not isinstance(geometry, dict)
            or geometry.get("type") != "Polygon"
        ):
            raise ValueError("Invalid feature geometry type")

        grid_id = props.get("grid_id")

        if not isinstance(grid_id, str) or grid_id in grid_ids:
            raise ValueError("Missing or duplicate grid ID")

        grid_ids.add(grid_id)

        if (
            props.get("origin_year") != 2025
            or props.get("pressure_period") != "2026-2028"
        ):
            raise ValueError("Pressure period mismatch")

        if props.get("score_is_calibrated_probability") is not False:
            raise ValueError("Incorrect score interpretation")

        category = props.get("pressure_category")

        if category not in EXPECTED_CATEGORIES:
            raise ValueError("Unknown pressure category")

        category_counts[category] += 1

        ready = props.get("final_inference_ready")
        score = props.get("urbanization_pressure_score")
        flagged = props.get("threshold_flag")

        if ready is True:
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                raise ValueError("Invalid pressure score")

            expected_flag = score >= 0.715

            if flagged is not expected_flag:
                raise ValueError("Pressure flag mismatch")

            if expected_flag:
                flagged_count += 1

                if category != "THRESHOLD_FLAGGED":
                    raise ValueError("Flag category mismatch")

            elif category == "THRESHOLD_FLAGGED":
                raise ValueError("Unexpected flagged category")

        elif ready is False:
            if (
                score is not None
                or flagged is not None
                or category != "NOT_IN_INFERENCE_DOMAIN"
            ):
                raise ValueError(
                    "Outside-domain values are invalid"
                )

        else:
            raise ValueError("Invalid inference domain flag")

    if dict(category_counts) != EXPECTED_CATEGORIES:
        raise ValueError("GeoJSON category count mismatch")

    if flagged_count != 255:
        raise ValueError("Flagged grid count mismatch")

    # Store file identity to detect modifications
    # after the verified map was loaded.
    stat = MAP_FILE.stat()

    return {
        "metadata": metadata,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def get_pressure_map_metadata():
    """
    Frontend-ready map metadata, legend,
    geographic bounds and map URL.
    """

    contract = _load_verified_contract()

    result = deepcopy(contract["metadata"])

    result["status"] = "ok"
    result["map_url"] = MAP_API_URL
    result["join_field"] = "grid_id"
    result["media_type"] = "application/geo+json"
    result["file_size_bytes"] = contract["size_bytes"]

    return result


def get_pressure_map_file_path():
    """
    Return only the approved GeoJSON path.

    No client-controlled filesystem paths.
    """

    contract = _load_verified_contract()

    stat = MAP_FILE.stat()

    if (
        stat.st_size != contract["size_bytes"]
        or stat.st_mtime_ns != contract["mtime_ns"]
    ):
        raise ValueError(
            "Pressure map changed after verification"
        )

    return MAP_FILE


def get_pressure_map_health():
    """
    Confirm the validated map is available.
    """

    path = get_pressure_map_file_path()

    return {
        "status": "ok",
        "component": "Urban Green Pressure GIS",
        "file": path.name,
        "feature_count": 9240,
        "eligible_scored_cells": 4620,
        "outside_inference_domain_cells": 4620,
        "threshold_flagged_cells": 255,
        "coordinate_reference_system": "EPSG:4326",
        "mode": "read-only",
    }
