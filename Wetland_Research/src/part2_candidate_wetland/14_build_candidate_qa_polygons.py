from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from rasterio.features import shapes
from shapely.geometry import shape


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "part2_candidate_wetland"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "part2_candidate_wetland"
)

MAP_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "maps"
    / "part2_candidate_wetland"
)

MAP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUTS
# ============================================================

COMPONENT_ID_INPUT = (
    PREDICTION_DIR
    / "13_preliminary_candidate_component_ids_10m.tif"
)

PROBABILITY_INPUT = (
    PREDICTION_DIR
    / "12_candidate_wetland_probability_10m.tif"
)

COMPONENT_STATS_INPUT = (
    REPORT_DIR
    / "13_candidate_component_statistics.csv"
)


# ============================================================
# OUTPUTS
# ============================================================

CANDIDATE_UTM_OUTPUT = (
    MAP_DIR
    / "14_candidate_qa_polygons_mmu_0p10ha_utm.geojson"
)

CANDIDATE_WGS84_OUTPUT = (
    MAP_DIR
    / "14_candidate_qa_polygons_mmu_0p10ha_wgs84.geojson"
)

CANDIDATE_TABLE_OUTPUT = (
    REPORT_DIR
    / "14_candidate_qa_polygon_table.csv"
)

SUMMARY_OUTPUT = (
    REPORT_DIR
    / "14_candidate_qa_summary.csv"
)

AUDIT_OUTPUT = (
    REPORT_DIR
    / "14_candidate_qa_polygon_audit.json"
)


# ============================================================
# SETTINGS
# ============================================================

WORKING_MMU_HA = 0.10

PIXEL_AREA_HA = 0.01

FROZEN_MODEL_THRESHOLD = 0.50

HIGH_SUPPORT_THRESHOLD = 0.70

VERY_HIGH_SUPPORT_THRESHOLD = 0.80

EXTREME_SUPPORT_THRESHOLD = 0.90


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "PART 2 CANDIDATE QA POLYGONS"
    )

    print(
        "========================================"
    )

    print(
        f"Working MMU: "
        f"{WORKING_MMU_HA:.2f} ha"
    )

    print(
        f"Frozen model threshold: "
        f"{FROZEN_MODEL_THRESHOLD:.2f}"
    )

    print(
        "\nIMPORTANT:"
        "\nThe 0.10 ha MMU is a mapping/generalization "
        "threshold for QA."
        "\nIt is not an ecological minimum wetland size."
        "\nNo automatic open-water removal is applied."
    )

    # ========================================================
    # 1. LOAD COMPONENT STATISTICS
    # ========================================================

    component_stats = pd.read_csv(
        COMPONENT_STATS_INPUT
    )

    print(
        f"\nAll preliminary components: "
        f"{len(component_stats):,}"
    )

    selected_stats = (
        component_stats[
            component_stats[
                "area_ha"
            ]
            >=
            WORKING_MMU_HA
        ]
        .copy()
    )

    selected_ids = (
        selected_stats[
            "component_id"
        ]
        .astype(int)
        .tolist()
    )

    selected_id_set = set(
        selected_ids
    )

    print(
        f"Components retained by MMU: "
        f"{len(selected_ids):,}"
    )

    if (
        len(selected_ids)
        == 0
    ):

        raise RuntimeError(
            "No components passed the working MMU."
        )

    # ========================================================
    # 2. LOAD COMPONENT + PROBABILITY RASTERS
    # ========================================================

    with (
        rasterio.open(
            COMPONENT_ID_INPUT
        ) as component_src,

        rasterio.open(
            PROBABILITY_INPUT
        ) as probability_src
    ):

        if (
            component_src.crs
            !=
            probability_src.crs
            or
            component_src.width
            !=
            probability_src.width
            or
            component_src.height
            !=
            probability_src.height
            or
            component_src.transform
            !=
            probability_src.transform
        ):

            raise RuntimeError(
                "Component and probability rasters "
                "are not aligned."
            )

        component_array = (
            component_src.read(1)
            .astype(np.int32)
        )

        probability_array = (
            probability_src.read(1)
            .astype(np.float32)
        )

        raster_crs = (
            component_src.crs
        )

        transform = (
            component_src.transform
        )

        # ====================================================
        # 3. HIGH-PROBABILITY SUPPORT WITHIN COMPONENTS
        # ====================================================

        flat_components = (
            component_array.ravel()
        )

        flat_probability = (
            probability_array.ravel()
        )

        max_component_id = int(
            flat_components.max()
        )

        all_component_pixels = (
            np.bincount(
                flat_components,
                minlength=max_component_id + 1,
            )
        )

        valid_component_mask = (
            flat_components > 0
        )

        p70_mask = (
            valid_component_mask
            &
            (
                flat_probability
                >=
                HIGH_SUPPORT_THRESHOLD
            )
        )

        p80_mask = (
            valid_component_mask
            &
            (
                flat_probability
                >=
                VERY_HIGH_SUPPORT_THRESHOLD
            )
        )

        p90_mask = (
            valid_component_mask
            &
            (
                flat_probability
                >=
                EXTREME_SUPPORT_THRESHOLD
            )
        )

        p70_counts = np.bincount(
            flat_components[
                p70_mask
            ],
            minlength=max_component_id + 1,
        )

        p80_counts = np.bincount(
            flat_components[
                p80_mask
            ],
            minlength=max_component_id + 1,
        )

        p90_counts = np.bincount(
            flat_components[
                p90_mask
            ],
            minlength=max_component_id + 1,
        )

        selected_stats[
            "pixels_ge_0p70"
        ] = (
            selected_stats[
                "component_id"
            ]
            .astype(int)
            .map(
                lambda component_id:
                    int(
                        p70_counts[
                            component_id
                        ]
                    )
            )
        )

        selected_stats[
            "pixels_ge_0p80"
        ] = (
            selected_stats[
                "component_id"
            ]
            .astype(int)
            .map(
                lambda component_id:
                    int(
                        p80_counts[
                            component_id
                        ]
                    )
            )
        )

        selected_stats[
            "pixels_ge_0p90"
        ] = (
            selected_stats[
                "component_id"
            ]
            .astype(int)
            .map(
                lambda component_id:
                    int(
                        p90_counts[
                            component_id
                        ]
                    )
            )
        )

        selected_stats[
            "fraction_ge_0p70"
        ] = (
            selected_stats[
                "pixels_ge_0p70"
            ]
            /
            selected_stats[
                "pixel_count"
            ]
        )

        selected_stats[
            "fraction_ge_0p80"
        ] = (
            selected_stats[
                "pixels_ge_0p80"
            ]
            /
            selected_stats[
                "pixel_count"
            ]
        )

        selected_stats[
            "fraction_ge_0p90"
        ] = (
            selected_stats[
                "pixels_ge_0p90"
            ]
            /
            selected_stats[
                "pixel_count"
            ]
        )

        selected_stats[
            "area_ge_0p70_ha"
        ] = (
            selected_stats[
                "pixels_ge_0p70"
            ]
            *
            PIXEL_AREA_HA
        )

        selected_stats[
            "area_ge_0p80_ha"
        ] = (
            selected_stats[
                "pixels_ge_0p80"
            ]
            *
            PIXEL_AREA_HA
        )

        selected_stats[
            "area_ge_0p90_ha"
        ] = (
            selected_stats[
                "pixels_ge_0p90"
            ]
            *
            PIXEL_AREA_HA
        )

        # ====================================================
        # 4. BUILD MMU-FILTERED COMPONENT RASTER IN MEMORY
        # ====================================================

        selected_mask = np.isin(
            component_array,
            np.asarray(
                selected_ids,
                dtype=np.int32,
            ),
        )

        selected_component_array = np.where(
            selected_mask,
            component_array,
            0,
        ).astype(
            np.int32
        )

        # ====================================================
        # 5. VECTORIZE
        #
        # connectivity=8 matches the component labelling used
        # in Script 13.
        # ====================================================

        vector_records = []

        for (
            geometry_mapping,
            component_value,
        ) in shapes(
            selected_component_array,
            mask=selected_mask,
            transform=transform,
            connectivity=8,
        ):

            component_id = int(
                component_value
            )

            if (
                component_id
                not in
                selected_id_set
            ):

                continue

            vector_records.append(
                {
                    "component_id":
                        component_id,

                    "geometry":
                        shape(
                            geometry_mapping
                        ),
                }
            )

    if (
        len(vector_records)
        == 0
    ):

        raise RuntimeError(
            "Vectorization produced no polygons."
        )

    polygons = gpd.GeoDataFrame(
        vector_records,
        geometry="geometry",
        crs=raster_crs,
    )

    # ========================================================
    # 6. ENSURE ONE FEATURE PER CONNECTED COMPONENT
    # ========================================================

    polygons = (
        polygons
        .dissolve(
            by="component_id",
            as_index=False,
        )
    )

    print(
        f"Vector polygons after dissolve: "
        f"{len(polygons):,}"
    )

    # ========================================================
    # 7. JOIN COMPONENT ATTRIBUTES
    # ========================================================

    candidates = (
        polygons.merge(
            selected_stats,
            on="component_id",
            how="left",
            validate="one_to_one",
        )
    )

    if (
        candidates[
            "area_ha"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "Component statistics failed to join "
            "to one or more polygons."
        )

    # ========================================================
    # 8. VECTOR AREA QA
    # ========================================================

    candidates[
        "vector_area_ha"
    ] = (
        candidates.geometry.area
        /
        10000.0
    )

    candidates[
        "area_difference_ha"
    ] = (
        candidates[
            "vector_area_ha"
        ]
        -
        candidates[
            "area_ha"
        ]
    )

    candidates[
        "absolute_area_difference_ha"
    ] = (
        candidates[
            "area_difference_ha"
        ]
        .abs()
    )

    maximum_area_difference = float(
        candidates[
            "absolute_area_difference_ha"
        ]
        .max()
    )

    # ========================================================
    # 9. STABLE CANDIDATE IDS
    #
    # Candidate IDs are tied to raster component IDs.
    # ========================================================

    candidates[
        "candidate_id"
    ] = (
        candidates[
            "component_id"
        ]
        .astype(int)
        .map(
            lambda component_id:
                f"CW_{component_id:05d}"
        )
    )

    # ========================================================
    # 10. QA RANKING ATTRIBUTES
    #
    # These are not new classifications.
    # They only help us choose which polygons to inspect first.
    # ========================================================

    candidates[
        "rank_by_area"
    ] = (
        candidates[
            "area_ha"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    candidates[
        "rank_by_mean_probability"
    ] = (
        candidates[
            "mean_probability"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    candidates[
        "rank_by_p80_area"
    ] = (
        candidates[
            "area_ge_0p80_ha"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    # ========================================================
    # 11. SORT FOR QA
    #
    # Large components with substantial >=0.80 probability
    # support appear first.
    # ========================================================

    candidates = (
        candidates
        .sort_values(
            [
                "area_ge_0p80_ha",
                "mean_probability",
                "area_ha",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    candidates[
        "qa_order"
    ] = np.arange(
        1,
        len(candidates) + 1,
    )

    # ========================================================
    # 12. SAVE UTM OUTPUT
    # ========================================================

    candidates.to_file(
        CANDIDATE_UTM_OUTPUT,
        driver="GeoJSON",
    )

    # ========================================================
    # 13. SAVE WGS84 OUTPUT
    #
    # Better for web mapping / Leaflet later.
    # ========================================================

    candidates_wgs84 = (
        candidates.to_crs(
            epsg=4326
        )
    )

    candidates_wgs84.to_file(
        CANDIDATE_WGS84_OUTPUT,
        driver="GeoJSON",
    )

    # ========================================================
    # 14. SAVE NON-SPATIAL TABLE
    # ========================================================

    table_columns = [
        column
        for column
        in candidates.columns
        if column
        != "geometry"
    ]

    candidates[
        table_columns
    ].to_csv(
        CANDIDATE_TABLE_OUTPUT,
        index=False,
    )

    # ========================================================
    # 15. SUMMARY
    # ========================================================

    summary_rows = [
        {
            "metric":
                "retained_candidate_components",

            "value":
                len(candidates),
        },

        {
            "metric":
                "total_candidate_area_ha",

            "value":
                candidates[
                    "area_ha"
                ].sum(),
        },

        {
            "metric":
                "median_candidate_area_ha",

            "value":
                candidates[
                    "area_ha"
                ].median(),
        },

        {
            "metric":
                "largest_candidate_area_ha",

            "value":
                candidates[
                    "area_ha"
                ].max(),
        },

        {
            "metric":
                "mean_candidate_probability",

            "value":
                candidates[
                    "mean_probability"
                ].mean(),
        },

        {
            "metric":
                "candidates_near_known_20m",

            "value":
                int(
                    candidates[
                        "near_known_wetland_20m"
                    ].sum()
                ),
        },

        {
            "metric":
                "total_area_ge_0p80_ha",

            "value":
                candidates[
                    "area_ge_0p80_ha"
                ].sum(),
        },
    ]

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    # ========================================================
    # 16. AUDIT
    # ========================================================

    audit = {
        "working_mmu_ha":
            WORKING_MMU_HA,

        "working_mmu_pixels":
            int(
                WORKING_MMU_HA
                /
                PIXEL_AREA_HA
            ),

        "working_mmu_interpretation":
            (
                "Raster mapping/generalization threshold "
                "for candidate spatial QA; not an "
                "ecological minimum wetland size."
            ),

        "frozen_model_threshold":
            FROZEN_MODEL_THRESHOLD,

        "component_connectivity":
            8,

        "input_component_count":
            int(
                len(
                    component_stats
                )
            ),

        "retained_component_count":
            int(
                len(
                    candidates
                )
            ),

        "retained_area_ha":
            float(
                candidates[
                    "area_ha"
                ].sum()
            ),

        "maximum_vector_vs_raster_area_difference_ha":
            maximum_area_difference,

        "open_water_automatic_exclusion":
            False,

        "known_wetland_pixels_already_removed":
            True,

        "near_known_20m_polygons_removed":
            False,

        "important_note":
            (
                "These polygons are candidate-screening "
                "objects for spatial QA. They must not "
                "be presented as officially confirmed "
                "wetland boundaries."
            ),
    }

    with open(
        AUDIT_OUTPUT,
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
        "CANDIDATE QA POLYGON SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        f"Retained candidate polygons: "
        f"{len(candidates):,}"
    )

    print(
        f"Total retained area: "
        f"{candidates['area_ha'].sum():.2f} ha"
    )

    print(
        f"Median polygon area: "
        f"{candidates['area_ha'].median():.2f} ha"
    )

    print(
        f"Largest polygon area: "
        f"{candidates['area_ha'].max():.2f} ha"
    )

    print(
        f"Candidates within 20 m of known wetlands: "
        f"{int(candidates['near_known_wetland_20m'].sum()):,}"
    )

    print(
        f"Total >=0.80 probability support area: "
        f"{candidates['area_ge_0p80_ha'].sum():.2f} ha"
    )

    print(
        f"Maximum raster/vector area difference: "
        f"{maximum_area_difference:.8f} ha"
    )

    print(
        "\nTop 25 QA-priority polygons:"
    )

    print(
        candidates[
            [
                "qa_order",
                "candidate_id",
                "area_ha",
                "mean_probability",
                "max_probability",
                "fraction_ge_0p70",
                "fraction_ge_0p80",
                "fraction_ge_0p90",
                "area_ge_0p80_ha",
                "near_known_wetland_20m",
                "mean_dry_ndvi",
                "mean_wet_ndvi",
                "mean_dry_mndwi",
                "mean_wet_mndwi",
                "mean_elevation_m",
                "mean_distance_to_drainage_m",
            ]
        ]
        .head(25)
        .to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )

    print(
        CANDIDATE_UTM_OUTPUT
    )

    print(
        CANDIDATE_WGS84_OUTPUT
    )

    print(
        CANDIDATE_TABLE_OUTPUT
    )

    print(
        SUMMARY_OUTPUT
    )

    print(
        AUDIT_OUTPUT
    )

    print(
        "\n========================================"
    )

    print(
        "CANDIDATE QA POLYGONS: COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        "\nIMPORTANT:"
        "\nThese are screening polygons for QA."
        "\nDo not call all 1,097 polygons confirmed wetlands."
        "\nNo automatic open-water removal was performed."
        "\nThe next step is spatial inspection in QGIS / "
        "Google Earth before final candidate reporting."
    )


if __name__ == "__main__":
    main()