
"""
J26-DS-338 | Urban Green Research

Final Development Suitability and Conservation API service.

Reads previously validated GeoJSON dashboard exports.
No model fitting, score recalculation or rewriting.

Suitability: 16G expert-AHP continuous GIS-MCDA.
Conservation: 17O frozen GI multiplied by frozen UP.
"""

from collections import Counter
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from shapely.geometry import shape
import json
import math

from backend.config import URBAN_GREEN_ROOT


OUTPUT = URBAN_GREEN_ROOT / "outputs" / "dashboard"

SPECS = {
    "suitability": {
        "folder": "development_suitability",
        "map": "suitability_map.geojson",
        "score": "development_suitability_score",
        "status": "suitability_class",
        "stage": "16G",
    },
    "conservation": {
        "folder": "conservation_intelligence",
        "map": "conservation_map.geojson",
        "score": "primary_conservation_priority_0_1",
        "status": "primary_priority_status",
        "stage": "17O",
    },
}

CONSERVATION_STATUSES = {
    "FUTURE_CONSERVATION_PRIORITY_CALCULATED": 4609,
    "NOT_IN_PRESSURE_DOMAIN": 4118,
    "GI_UNSCORED_PRESSURE_AVAILABLE": 11,
    "GI_UNSCORED_AND_PRESSURE_NOT_IN_DOMAIN": 502,
}


def _numeric(value):
    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("Invalid numeric GIS value")

    return float(value)


def _close(a, b):
    return math.isclose(
        a, b, rel_tol=0, abs_tol=1e-8
    )


@lru_cache(maxsize=2)
def _load(kind):
    if kind not in SPECS:
        raise ValueError("Unknown research component")

    spec = SPECS[kind]
    folder = OUTPUT / spec["folder"]
    map_file = folder / spec["map"]
    metadata_file = folder / "metadata.json"

    for path in (map_file, metadata_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    with metadata_file.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    with map_file.open("r", encoding="utf-8") as f:
        collection = json.load(f)

    if metadata.get("status") != "ok":
        raise ValueError("Invalid research metadata status")

    if metadata.get("grid_count") != 9240:
        raise ValueError("Incorrect metadata grid count")

    if metadata.get("source_stage") != spec["stage"]:
        raise ValueError("Incorrect research release stage")

    if metadata.get("crs") != "EPSG:4326":
        raise ValueError("Unexpected map CRS")

    if metadata.get("geojson_file") != spec["map"]:
        raise ValueError("Unexpected map filename")

    if metadata.get("score_field") != spec["score"]:
        raise ValueError("Unexpected score field")

    if collection.get("type") != "FeatureCollection":
        raise ValueError("Invalid GeoJSON")

    features = collection.get("features")

    if not isinstance(features, list) or len(features) != 9240:
        raise ValueError("Expected 9240 GIS features")

    if kind == "suitability":
        weights = metadata["criteria"]

        if not all(
            _close(float(weights[c]["weight"]), w)
            for c, w in (
                ("C1", 0.2),
                ("C2", 0.6),
                ("C4", 0.2),
            )
        ):
            raise ValueError("AHP weights do not match release")

        if (
            metadata.get("final_policy_classes_assigned")
            is not False
        ):
            raise ValueError("Suitability classes not approved")

        if not _close(
            float(
                metadata["expert_consultation"][
                    "consistency_ratio"
                ]
            ),
            0.0,
        ):
            raise ValueError("AHP consistency mismatch")

    else:
        if metadata.get("primary_priority_scored_count") != 4609:
            raise ValueError("Conservation count mismatch")

        if metadata.get("primary_priority_unscored_count") != 4631:
            raise ValueError("Conservation null count mismatch")

        if metadata.get("final_priority_classes_assigned") is not False:
            raise ValueError("Conservation classes not approved")

    lookup = {}
    statuses = Counter()
    scored_count = 0
    gi_count = 0

    for feature in features:
        if feature.get("type") != "Feature":
            raise ValueError("Invalid feature")

        geometry = feature.get("geometry")
        props = feature.get("properties")

        
        if not isinstance(geometry, dict) or not isinstance(props, dict):
            raise ValueError("Invalid GIS geometry or properties")

        geometry_type = geometry.get("type")

        if geometry_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(
                f"Unsupported GIS geometry: {geometry_type}"
            )

        polygon = shape(geometry)

        if (
            polygon.is_empty
            or not polygon.is_valid
            or polygon.geom_type not in ("Polygon", "MultiPolygon")
        ):
            raise ValueError(
                f"Invalid polygon geometry: {props.get('grid_id')}"
            )


        grid_id = props.get("grid_id")

        if (
            not isinstance(grid_id, str)
            or not grid_id
            or grid_id in lookup
        ):
            raise ValueError("Duplicate or missing grid ID")

        score = _numeric(props.get(spec["score"]))
        status = props.get(spec["status"])

        if not isinstance(status, str):
            raise ValueError("Missing grid status")

        statuses[status] += 1

        if kind == "suitability":
            if score is None or not 0 <= score <= 1:
                raise ValueError("Missing suitability score")

            expected = (
                0.2 * _numeric(props["c1_road_access_score"])
                + 0.6 * _numeric(props["c2_road_context_score"])
                + 0.2 * _numeric(props["c4_slope_score"])
            )

            if not _close(score, expected):
                raise ValueError(
                    f"Suitability formula mismatch: {grid_id}"
                )

            if status != "NOT_YET_CLASSIFIED":
                raise ValueError("Unsupported suitability class")

            scored_count += 1

        else:
            gi = _numeric(
                props.get("green_importance_index_0_1")
            )
            up = _numeric(
                props.get("future_urbanization_pressure_score")
            )

            if gi is not None:
                if not 0 <= gi <= 1:
                    raise ValueError("Invalid Green Importance")
                gi_count += 1

            if score is not None:
                if (
                    gi is None or up is None
                    or not 0 <= up <= 1
                    or not 0 <= score <= 1
                    or not _close(score, gi * up)
                    or status != (
                        "FUTURE_CONSERVATION_PRIORITY_CALCULATED"
                    )
                ):
                    raise ValueError(
                        f"Conservation formula mismatch: {grid_id}"
                    )

                scored_count += 1

            else:
                if (
                    props.get(
                        "primary_priority_rank_common_domain"
                    ) is not None
                    or props.get(
                        "primary_priority_percentile_common_domain"
                    ) is not None
                    or status == (
                        "FUTURE_CONSERVATION_PRIORITY_CALCULATED"
                    )
                ):
                    raise ValueError(
                        f"Unscored conservation cell invalid: {grid_id}"
                    )

            if props.get("priority_classes_assigned") not in (
                False, 0
            ):
                raise ValueError("Unsupported priority class")

        lookup[grid_id] = props

    if len(lookup) != 9240:
        raise ValueError("Incomplete grid lookup")

    if kind == "suitability":
        if scored_count != 9240:
            raise ValueError("Suitability coverage mismatch")
    else:
        if (
            scored_count != 4609
            or gi_count != 8727
            or dict(statuses) != CONSERVATION_STATUSES
        ):
            raise ValueError(
                "Conservation coverage/status mismatch"
            )

    file_states = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in (map_file, metadata_file)
    }

    return {
        "metadata": metadata,
        "lookup": lookup,
        "ordered_ids": [f["properties"]["grid_id"] for f in features],
        "scored_count": scored_count,
        "status_counts": dict(statuses),
        "map_file": map_file,
        "file_states": file_states,
    }


def _verified(kind):
    data = _load(kind)

    for path, expected in data["file_states"].items():
        stat = path.stat()

        if (stat.st_size, stat.st_mtime_ns) != expected:
            raise ValueError(
                "GIS export changed after initial verification"
            )

    return data


def get_decision_overview(kind):
    data = _verified(kind)
    result = deepcopy(data["metadata"])

    result["component"] = kind
    result["scored_grid_count"] = data["scored_count"]
    result["status_counts"] = deepcopy(data["status_counts"])
    result["map_url"] = f"/api/urban-green/{kind}/map"
    result["read_only"] = True

    return result


def get_decision_grid(kind, grid_id):
    data = _verified(kind)
    props = data["lookup"].get(grid_id)

    if props is None:
        raise KeyError(grid_id)

    score_name = SPECS[kind]["score"]

    return {
        "component": kind,
        "grid_id": grid_id,
        "score": props.get(score_name),
        "score_available": props.get(score_name) is not None,
        "properties": deepcopy(props),
        "interpretation": (
            "Continuous criterion-based suitability; "
            "not legal permission."
            if kind == "suitability"
            else
            "Relative conservation priority only where "
            "Green Importance and future pressure are available; "
            "not a calibrated probability."
        ),
    }


def get_decision_grids(kind, offset=0, limit=100):
    if (
        isinstance(offset, bool)
        or isinstance(limit, bool)
        or not isinstance(offset, int)
        or not isinstance(limit, int)
        or offset < 0
        or not 1 <= limit <= 200
    ):
        raise ValueError("Invalid pagination")

    data = _verified(kind)
    selected = data["ordered_ids"][offset:offset + limit]
    score_name = SPECS[kind]["score"]

    grids = []

    for grid_id in selected:
        props = data["lookup"][grid_id]

        grids.append({
            "grid_id": grid_id,
            "score": props.get(score_name),
            "status": props.get(SPECS[kind]["status"]),
            "properties": deepcopy(props),
        })

    return {
        "component": kind,
        "total": 9240,
        "offset": offset,
        "limit": limit,
        "returned": len(grids),
        "grids": grids,
    }


def get_decision_map_metadata(kind):
    data = _verified(kind)
    result = deepcopy(data["metadata"])

    result["map_url"] = f"/api/urban-green/{kind}/map"
    result["media_type"] = "application/geo+json"
    result["file_size_bytes"] = data["map_file"].stat().st_size

    return result


def get_decision_map_file(kind):
    return _verified(kind)["map_file"]
