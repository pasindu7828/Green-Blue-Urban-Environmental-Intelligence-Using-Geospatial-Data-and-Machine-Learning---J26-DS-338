
"""
J26-DS-338 - Urban Green Research
Historical GIS Map Service

Serves only the 38 approved historical web-map
overlays created in Step 4F.2 and validated
in Step 4F.3.

READ ONLY. No raster processing, training,
recalculation or file modification.
"""

import json
import re

from collections import Counter
from copy import deepcopy
from functools import lru_cache
from pathlib import Path, PurePosixPath

from backend.config import URBAN_GREEN_ROOT


MAP_ROOT = (
    URBAN_GREEN_ROOT
    / "outputs"
    / "dashboard"
    / "historical_maps"
)

MANIFEST_PATH = MAP_ROOT / "manifest.json"

API_PREFIX = "/api/urban-green/historical/maps"

VALID_ID = re.compile(r"^[A-Z0-9_]+$")


# ============================================================
# SAFE MANIFEST-ONLY IMAGE ACCESS
# ============================================================

def _safe_image_path(image_relative):
    """
    Resolve only PNG paths inside historical_maps.
    Reject absolute paths and directory traversal.
    """

    if not isinstance(image_relative, str):
        raise ValueError("Invalid map image path")

    relative = PurePosixPath(image_relative)

    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in image_relative
        or relative.suffix.lower() != ".png"
    ):
        raise ValueError(
            "Unsafe historical map image path"
        )

    root = MAP_ROOT.resolve()

    image_path = (
        MAP_ROOT / image_relative
    ).resolve()

    if not image_path.is_relative_to(root):
        raise ValueError(
            "Map image escapes approved directory"
        )

    return image_path


# ============================================================
# VALIDATE AND LOAD THE FROZEN MAP CATALOG
# ============================================================

@lru_cache(maxsize=1)
def _load_catalog():

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Historical map manifest missing: "
            f"{MANIFEST_PATH}"
        )

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    maps = manifest.get("maps")

    if not isinstance(maps, list):
        raise ValueError(
            "Historical map list is invalid"
        )

    if (
        manifest.get("overlay_count") != 38
        or len(maps) != 38
    ):
        raise ValueError(
            "Expected exactly 38 historical maps"
        )

    if (
        manifest.get("primary_transition")
        != "ROBUST_2_OF_3"
    ):
        raise ValueError(
            "Primary transition method mismatch"
        )

    if (
        manifest.get("conservative_transition")
        != "STRICT_3_OF_3"
    ):
        raise ValueError(
            "Conservative transition method mismatch"
        )

    if (
        manifest.get(
            "statistical_confidence_interval"
        ) is not False
    ):
        raise ValueError(
            "Incorrect confidence interpretation"
        )

    index = {}
    kinds = Counter()

    for record in maps:

        if not isinstance(record, dict):
            raise ValueError(
                "Invalid map metadata record"
            )

        map_id = record.get("id")

        if (
            not isinstance(map_id, str)
            or not VALID_ID.fullmatch(map_id)
        ):
            raise ValueError(
                f"Invalid map identifier: {map_id}"
            )

        if map_id in index:
            raise ValueError(
                f"Duplicate map identifier: {map_id}"
            )

        image_path = _safe_image_path(
            record.get("image")
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Map image missing: {image_path}"
            )

        with image_path.open("rb") as image:
            signature = image.read(8)

        if signature != b"\x89PNG\r\n\x1a\n":
            raise ValueError(
                f"Invalid PNG signature: {map_id}"
            )

        index[map_id] = record
        kinds[record.get("kind")] += 1

    expected_kinds = {
        "annual_lulc": 34,
        "green_transition": 2,
        "transition_confidence": 2,
    }

    if dict(kinds) != expected_kinds:
        raise ValueError(
            f"Unexpected map categories: {dict(kinds)}"
        )

    return manifest, index


# ============================================================
# PUBLIC SERVICE FUNCTIONS
# ============================================================

def get_historical_map_catalog():
    """
    Return all 38 available historical layers
    with frontend-friendly image URLs.
    """

    manifest, index = _load_catalog()

    maps = [
        get_historical_map_metadata(map_id)
        for map_id in index
    ]

    return {
        "status": "ok",
        "total": len(maps),
        "primary_core_trend": (
            manifest["primary_core_trend"]
        ),
        "primary_transition": (
            manifest["primary_transition"]
        ),
        "conservative_transition": (
            manifest["conservative_transition"]
        ),
        "statistical_confidence_interval": False,
        "maps": maps,
    }


def get_historical_map_metadata(map_id):
    """
    Return metadata for one registered map ID.

    Raises KeyError if the map does not exist.
    """

    _, index = _load_catalog()

    if map_id not in index:
        raise KeyError(
            f"Unknown historical map: {map_id}"
        )

    metadata = deepcopy(index[map_id])

    metadata["image_url"] = (
        f"{API_PREFIX}/{map_id}/image"
    )

    return metadata


def get_historical_map_image_path(map_id):
    """
    Return a verified image file path for FileResponse.

    Arbitrary client-supplied filesystem paths
    are never accepted.
    """

    _, index = _load_catalog()

    if map_id not in index:
        raise KeyError(
            f"Unknown historical map: {map_id}"
        )

    image_path = _safe_image_path(
        index[map_id]["image"]
    )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Historical image missing: {map_id}"
        )

    return image_path


def get_historical_map_health():
    """
    Validate the historical map catalog.
    """

    manifest, index = _load_catalog()

    return {
        "status": "ok",
        "component": "Urban Green Historical GIS",
        "overlay_count": len(index),
        "annual_overlay_count": (
            manifest["annual_overlay_count"]
        ),
        "transition_overlay_count": (
            manifest["transition_overlay_count"]
        ),
        "confidence_overlay_count": (
            manifest["confidence_overlay_count"]
        ),
        "mode": "read-only",
    }
