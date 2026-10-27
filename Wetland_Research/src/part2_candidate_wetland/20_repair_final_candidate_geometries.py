from pathlib import Path
import json

import geopandas as gpd
import numpy as np
from shapely import make_valid
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAP_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "maps"
    / "part2_candidate_wetland"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "part2_candidate_wetland"
)


# ============================================================
# INPUT
#
# Repair the complete UTM layer once.
# All other final layers will be regenerated from this repaired
# parent layer so geometry stays consistent.
# ============================================================

INPUT_UTM = (
    MAP_DIR
    / "18_final_candidate_priority_utm.geojson"
)


# ============================================================
# OUTPUTS
# ============================================================

REPAIRED_FULL_UTM = (
    MAP_DIR
    / "20_final_candidate_priority_utm_repaired.geojson"
)

REPAIRED_FULL_WGS84 = (
    MAP_DIR
    / "20_final_candidate_priority_wgs84_repaired.geojson"
)

REPAIRED_PRIMARY_UTM = (
    MAP_DIR
    / "20_primary_candidate_layer_utm_repaired.geojson"
)

REPAIRED_PRIMARY_WGS84 = (
    MAP_DIR
    / "20_primary_candidate_layer_wgs84_repaired.geojson"
)

REPAIRED_REVIEW_UTM = (
    MAP_DIR
    / "20_review_candidate_layer_utm_repaired.geojson"
)

REPAIRED_FALSE_POSITIVE_UTM = (
    MAP_DIR
    / "20_manual_false_positive_flags_utm_repaired.geojson"
)

REPAIR_TABLE = (
    REPORT_DIR
    / "20_geometry_repair_diagnostics.csv"
)

REPAIR_AUDIT = (
    REPORT_DIR
    / "20_geometry_repair_audit.json"
)


# ============================================================
# HELPERS
# ============================================================

def extract_polygonal_geometry(geometry):

    if geometry is None:
        return geometry

    if geometry.is_empty:
        return geometry

    if isinstance(
        geometry,
        (Polygon, MultiPolygon),
    ):
        return geometry

    if isinstance(
        geometry,
        GeometryCollection,
    ):

        polygon_parts = []

        for part in geometry.geoms:

            if isinstance(
                part,
                Polygon,
            ):
                polygon_parts.append(
                    part
                )

            elif isinstance(
                part,
                MultiPolygon,
            ):
                polygon_parts.extend(
                    list(
                        part.geoms
                    )
                )

        if not polygon_parts:
            return Polygon()

        return unary_union(
            polygon_parts
        )

    return geometry


def repair_geometry(geometry):

    if geometry is None:
        return geometry

    if geometry.is_empty:
        return geometry

    repaired = geometry

    if not repaired.is_valid:

        repaired = make_valid(
            repaired
        )

    repaired = extract_polygonal_geometry(
        repaired
    )

    if (
        repaired is not None
        and
        not repaired.is_empty
        and
        not repaired.is_valid
    ):

        repaired = repaired.buffer(
            0
        )

        repaired = extract_polygonal_geometry(
            repaired
        )

    return repaired


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "PART 2 FINAL GEOMETRY REPAIR"
    )

    print(
        "========================================"
    )

    print(
        "\nIMPORTANT:"
        "\nGeometry topology only will be repaired."
        "\nCandidate IDs and attributes will not change."
        "\nNo model, threshold, priority or manual QA "
        "decision will be changed."
    )

    # ========================================================
    # 1. LOAD
    # ========================================================

    candidates = gpd.read_file(
        INPUT_UTM
    )

    print(
        f"\nInput candidate polygons: "
        f"{len(candidates):,}"
    )

    print(
        f"Input CRS: "
        f"{candidates.crs}"
    )

    if (
        candidates.crs.to_epsg()
        !=
        32644
    ):

        raise RuntimeError(
            "Expected EPSG:32644 input."
        )

    if (
        candidates[
            "candidate_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate candidate IDs detected."
        )

    # ========================================================
    # 2. BEFORE-REPAIR DIAGNOSTICS
    # ========================================================

    candidates[
        "_valid_before"
    ] = candidates.geometry.is_valid

    candidates[
        "_empty_before"
    ] = candidates.geometry.is_empty

    candidates[
        "_area_before_ha"
    ] = (
        candidates.geometry.area
        /
        10000.0
    )

    invalid_before = int(
        (
            ~candidates[
                "_valid_before"
            ]
        ).sum()
    )

    empty_before = int(
        candidates[
            "_empty_before"
        ].sum()
    )

    print(
        f"\nInvalid geometries before repair: "
        f"{invalid_before:,}"
    )

    print(
        f"Empty geometries before repair: "
        f"{empty_before:,}"
    )

    # ========================================================
    # 3. REPAIR
    # ========================================================

    candidates[
        "geometry"
    ] = candidates.geometry.apply(
        repair_geometry
    )

    # ========================================================
    # 4. AFTER-REPAIR DIAGNOSTICS
    # ========================================================

    candidates[
        "_valid_after"
    ] = candidates.geometry.is_valid

    candidates[
        "_empty_after"
    ] = candidates.geometry.is_empty

    candidates[
        "_area_after_ha"
    ] = (
        candidates.geometry.area
        /
        10000.0
    )

    candidates[
        "_area_change_ha"
    ] = (
        candidates[
            "_area_after_ha"
        ]
        -
        candidates[
            "_area_before_ha"
        ]
    )

    candidates[
        "_abs_area_change_ha"
    ] = candidates[
        "_area_change_ha"
    ].abs()

    invalid_after = int(
        (
            ~candidates[
                "_valid_after"
            ]
        ).sum()
    )

    empty_after = int(
        candidates[
            "_empty_after"
        ].sum()
    )

    print(
        f"Invalid geometries after repair: "
        f"{invalid_after:,}"
    )

    print(
        f"Empty geometries after repair: "
        f"{empty_after:,}"
    )

    if invalid_after != 0:

        raise RuntimeError(
            "Some geometries remain invalid "
            "after repair."
        )

    if empty_after != 0:

        raise RuntimeError(
            "Geometry repair produced "
            "empty polygons."
        )

    # ========================================================
    # 5. AREA CHANGE QA
    # ========================================================

    maximum_area_change = float(
        candidates[
            "_abs_area_change_ha"
        ].max()
    )

    total_area_before = float(
        candidates[
            "_area_before_ha"
        ].sum()
    )

    total_area_after = float(
        candidates[
            "_area_after_ha"
        ].sum()
    )

    total_area_change = (
        total_area_after
        -
        total_area_before
    )

    print(
        "\nArea QA:"
    )

    print(
        f"Total before: "
        f"{total_area_before:.8f} ha"
    )

    print(
        f"Total after: "
        f"{total_area_after:.8f} ha"
    )

    print(
        f"Total change: "
        f"{total_area_change:.8f} ha"
    )

    print(
        f"Maximum per-candidate absolute change: "
        f"{maximum_area_change:.8f} ha"
    )

    # ========================================================
    # 6. SAVE REPAIR DIAGNOSTICS
    # ========================================================

    diagnostic_columns = [
        "candidate_id",
        "_valid_before",
        "_valid_after",
        "_area_before_ha",
        "_area_after_ha",
        "_area_change_ha",
        "_abs_area_change_ha",
    ]

    candidates[
        diagnostic_columns
    ].to_csv(
        REPAIR_TABLE,
        index=False,
    )

    # ========================================================
    # 7. REMOVE TEMPORARY QA FIELDS FROM SPATIAL OUTPUT
    # ========================================================

    temporary_fields = [
        "_valid_before",
        "_empty_before",
        "_area_before_ha",
        "_valid_after",
        "_empty_after",
        "_area_after_ha",
        "_area_change_ha",
        "_abs_area_change_ha",
    ]

    repaired_utm = (
        candidates.drop(
            columns=temporary_fields
        )
        .copy()
    )

    # ========================================================
    # 8. RECREATE WGS84 FROM REPAIRED UTM
    # ========================================================

    repaired_wgs84 = (
        repaired_utm.to_crs(
            epsg=4326
        )
    )

    # ========================================================
    # 9. RECREATE DERIVED FINAL LAYERS
    # ========================================================

    primary_utm = (
        repaired_utm[
            repaired_utm[
                "include_primary_candidate_layer"
            ]
            ==
            True
        ]
        .copy()
    )

    primary_wgs84 = (
        repaired_wgs84[
            repaired_wgs84[
                "include_primary_candidate_layer"
            ]
            ==
            True
        ]
        .copy()
    )

    review_utm = (
        repaired_utm[
            repaired_utm[
                "candidate_reporting_status"
            ]
            ==
            "review_priority_unreviewed_candidate"
        ]
        .copy()
    )

    false_positive_utm = (
        repaired_utm[
            repaired_utm[
                "candidate_reporting_status"
            ]
            ==
            "manual_qa_false_positive_flag"
        ]
        .copy()
    )

    # ========================================================
    # 10. COUNT QA
    # ========================================================

    if len(repaired_utm) != 1097:

        raise RuntimeError(
            "Full repaired count changed."
        )

    if len(primary_utm) != 48:

        raise RuntimeError(
            "Primary candidate count changed."
        )

    if len(review_utm) != 1005:

        raise RuntimeError(
            "Review candidate count changed."
        )

    if len(false_positive_utm) != 44:

        raise RuntimeError(
            "False-positive count changed."
        )

    # ========================================================
    # 11. WRITE OUTPUTS
    # ========================================================

    repaired_utm.to_file(
        REPAIRED_FULL_UTM,
        driver="GeoJSON",
    )

    repaired_wgs84.to_file(
        REPAIRED_FULL_WGS84,
        driver="GeoJSON",
    )

    primary_utm.to_file(
        REPAIRED_PRIMARY_UTM,
        driver="GeoJSON",
    )

    primary_wgs84.to_file(
        REPAIRED_PRIMARY_WGS84,
        driver="GeoJSON",
    )

    review_utm.to_file(
        REPAIRED_REVIEW_UTM,
        driver="GeoJSON",
    )

    false_positive_utm.to_file(
        REPAIRED_FALSE_POSITIVE_UTM,
        driver="GeoJSON",
    )

    # ========================================================
    # 12. FINAL DISK-READ VALIDATION
    #
    # Read outputs back from disk to ensure serialization
    # did not reintroduce geometry problems.
    # ========================================================

    disk_layers = {
        "full_utm":
            gpd.read_file(
                REPAIRED_FULL_UTM
            ),

        "full_wgs84":
            gpd.read_file(
                REPAIRED_FULL_WGS84
            ),

        "primary_utm":
            gpd.read_file(
                REPAIRED_PRIMARY_UTM
            ),

        "primary_wgs84":
            gpd.read_file(
                REPAIRED_PRIMARY_WGS84
            ),

        "review_utm":
            gpd.read_file(
                REPAIRED_REVIEW_UTM
            ),

        "false_positive_utm":
            gpd.read_file(
                REPAIRED_FALSE_POSITIVE_UTM
            ),
    }

    disk_validation = {}

    for (
        layer_name,
        layer,
    ) in disk_layers.items():

        disk_validation[
            f"{layer_name}_valid"
        ] = bool(
            layer.geometry.is_valid.all()
        )

        disk_validation[
            f"{layer_name}_nonempty"
        ] = bool(
            (
                ~layer.geometry.is_empty
            ).all()
        )

    failed_disk_checks = [
        name
        for name, passed
        in disk_validation.items()
        if not passed
    ]

    if failed_disk_checks:

        raise RuntimeError(
            "Disk-read geometry validation "
            f"failed: {failed_disk_checks}"
        )

    # ========================================================
    # 13. AUDIT
    # ========================================================

    audit = {
        "input_candidate_count":
            int(
                len(
                    candidates
                )
            ),

        "invalid_before_repair":
            invalid_before,

        "invalid_after_repair":
            invalid_after,

        "empty_before_repair":
            empty_before,

        "empty_after_repair":
            empty_after,

        "total_area_before_ha":
            total_area_before,

        "total_area_after_ha":
            total_area_after,

        "total_area_change_ha":
            total_area_change,

        "maximum_candidate_area_change_ha":
            maximum_area_change,

        "final_counts":
            {
                "full":
                    int(
                        len(
                            repaired_utm
                        )
                    ),

                "primary":
                    int(
                        len(
                            primary_utm
                        )
                    ),

                "review":
                    int(
                        len(
                            review_utm
                        )
                    ),

                "manual_false_positive":
                    int(
                        len(
                            false_positive_utm
                        )
                    ),
            },

        "disk_validation":
            disk_validation,

        "candidate_ids_changed":
            False,

        "attributes_changed":
            False,

        "model_changed":
            False,

        "threshold_changed":
            False,

        "priority_rules_changed":
            False,

        "manual_qa_decisions_changed":
            False,
    }

    with open(
        REPAIR_AUDIT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            audit,
            file,
            indent=4,
        )

    # ========================================================
    # TERMINAL REPORT
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "REPAIRED FINAL LAYER COUNTS"
    )

    print(
        "========================================"
    )

    print(
        f"Full: "
        f"{len(repaired_utm):,}"
    )

    print(
        f"Primary: "
        f"{len(primary_utm):,}"
    )

    print(
        f"Review: "
        f"{len(review_utm):,}"
    )

    print(
        f"Manual false-positive: "
        f"{len(false_positive_utm):,}"
    )

    print(
        "\nDisk-read validation:"
    )

    for (
        name,
        passed,
    ) in disk_validation.items():

        print(
            f"{name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print(
        "\nOutputs:"
    )

    print(
        REPAIRED_FULL_UTM
    )

    print(
        REPAIRED_FULL_WGS84
    )

    print(
        REPAIRED_PRIMARY_UTM
    )

    print(
        REPAIRED_PRIMARY_WGS84
    )

    print(
        REPAIRED_REVIEW_UTM
    )

    print(
        REPAIRED_FALSE_POSITIVE_UTM
    )

    print(
        REPAIR_TABLE
    )

    print(
        REPAIR_AUDIT
    )

    print(
        "\n========================================"
    )

    print(
        "FINAL GEOMETRY REPAIR: COMPLETE"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()