from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from rasterio.features import rasterize
from scipy import ndimage


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PART1_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "part1_historical_change"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "part2_candidate_wetland"
)

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

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUTS
# ============================================================

PROBABILITY_INPUT = (
    PREDICTION_DIR
    / "12_candidate_wetland_probability_10m.tif"
)

MODEL_POSITIVE_INPUT = (
    PREDICTION_DIR
    / "12_model_positive_mask_10m.tif"
)

FEATURE_STACK_INPUT = (
    PROCESSED_DIR
    / "feature_stack_10m.tif"
)

FEATURE_METADATA_INPUT = (
    PROCESSED_DIR
    / "feature_stack_metadata.csv"
)

KNOWN_WETLANDS_INPUT = (
    PART1_DIR
    / "wetland_latest_status_gn_wgs84.geojson"
)


# ============================================================
# OUTPUTS
# ============================================================

PRELIMINARY_CANDIDATE_MASK_OUTPUT = (
    PREDICTION_DIR
    / "13_preliminary_candidate_mask_outside_known_10m.tif"
)

COMPONENT_ID_OUTPUT = (
    PREDICTION_DIR
    / "13_preliminary_candidate_component_ids_10m.tif"
)

COMPONENT_STATS_OUTPUT = (
    REPORT_DIR
    / "13_candidate_component_statistics.csv"
)

MMU_SCENARIOS_OUTPUT = (
    REPORT_DIR
    / "13_candidate_mmu_scenarios.csv"
)

AUDIT_OUTPUT = (
    REPORT_DIR
    / "13_candidate_component_diagnostics_audit.json"
)


# ============================================================
# SETTINGS
# ============================================================

FROZEN_THRESHOLD = 0.50

# Used only to identify whether a preliminary component
# lies very close to an existing known wetland.
#
# It does NOT remove the component.
KNOWN_WETLAND_NEAR_DISTANCE_M = 20.0

# 8-neighbour connectivity:
# diagonal pixels are treated as connected.
CONNECTIVITY_STRUCTURE = np.ones(
    (
        3,
        3,
    ),
    dtype=np.uint8,
)

# One 10 m x 10 m pixel = 100 m² = 0.01 ha.
PIXEL_AREA_HA = 0.01

# Diagnostic MMU scenarios only.
# Nothing is permanently filtered in this script.
MMU_SCENARIOS_HA = [
    0.01,
    0.02,
    0.05,
    0.10,
    0.25,
    0.50,
    1.00,
    2.00,
    5.00,
]

BINARY_NODATA = 255

COMPONENT_NODATA = 0


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "PART 2 CANDIDATE COMPONENT DIAGNOSTICS"
    )

    print(
        "========================================"
    )

    print(
        f"Frozen model threshold: "
        f"{FROZEN_THRESHOLD:.2f}"
    )

    print(
        "\nIMPORTANT:"
        "\nNo new model tuning is performed."
        "\nNo MMU is selected yet."
        "\nNo open-water spectral rule is applied yet."
    )

    # ========================================================
    # 1. LOAD FEATURE METADATA
    # ========================================================

    feature_metadata = pd.read_csv(
        FEATURE_METADATA_INPUT
    )

    feature_names = (
        feature_metadata[
            "feature"
        ]
        .tolist()
    )

    print(
        f"\nFeature count: "
        f"{len(feature_names)}"
    )

    required_diagnostic_features = [
        "dry_ndvi",
        "wet_ndvi",
        "dry_mndwi",
        "wet_mndwi",
        "elevation_m",
        "distance_to_drainage_m",
    ]

    missing_diagnostic_features = [
        feature
        for feature
        in required_diagnostic_features
        if feature
        not in feature_names
    ]

    if (
        missing_diagnostic_features
    ):

        raise RuntimeError(
            "Required diagnostic features are missing: "
            f"{missing_diagnostic_features}"
        )

    # ========================================================
    # 2. OPEN RASTERS
    # ========================================================

    with (
        rasterio.open(
            PROBABILITY_INPUT
        ) as probability_src,

        rasterio.open(
            MODEL_POSITIVE_INPUT
        ) as positive_src,

        rasterio.open(
            FEATURE_STACK_INPUT
        ) as feature_src
    ):

        print(
            "\nRaster grid:"
        )

        print(
            f"CRS: "
            f"{probability_src.crs}"
        )

        print(
            f"Size: "
            f"{probability_src.width} x "
            f"{probability_src.height}"
        )

        print(
            f"Resolution: "
            f"{probability_src.res}"
        )

        # ----------------------------------------------------
        # ALIGNMENT CHECK
        # ----------------------------------------------------

        for raster_source in [
            positive_src,
            feature_src,
        ]:

            if (
                raster_source.crs
                != probability_src.crs
                or
                raster_source.width
                != probability_src.width
                or
                raster_source.height
                != probability_src.height
                or
                raster_source.transform
                != probability_src.transform
            ):

                raise RuntimeError(
                    "Input rasters are not "
                    "spatially aligned."
                )

        if (
            feature_src.count
            != len(
                feature_names
            )
        ):

            raise RuntimeError(
                "Feature-stack band count "
                "does not match metadata."
            )

        probability = (
            probability_src.read(1)
            .astype(
                np.float32
            )
        )

        model_positive = (
            positive_src.read(1)
        )

        transform = (
            probability_src.transform
        )

        raster_crs = (
            probability_src.crs
        )

        height = (
            probability_src.height
        )

        width = (
            probability_src.width
        )

        # ----------------------------------------------------
        # VALID STUDY-AREA PIXELS
        # ----------------------------------------------------

        valid_probability = (
            np.isfinite(
                probability
            )
        )

        if (
            probability_src.nodata
            is not None
        ):

            valid_probability &= (
                probability
                !=
                probability_src.nodata
            )

        valid_pixel_count = int(
            valid_probability.sum()
        )

        # ----------------------------------------------------
        # MODEL-POSITIVE PIXELS
        # ----------------------------------------------------

        raw_positive = (
            valid_probability
            &
            (
                model_positive
                == 1
            )
        )

        raw_positive_count = int(
            raw_positive.sum()
        )

        # ====================================================
        # 3. LOAD KNOWN WETLAND POLYGONS
        # ====================================================

        known_wetlands = gpd.read_file(
            KNOWN_WETLANDS_INPUT
        )

        print(
            f"\nKnown wetland features loaded: "
            f"{len(known_wetlands)}"
        )

        if (
            known_wetlands.crs
            is None
        ):

            raise RuntimeError(
                "Known wetland layer has no CRS."
            )

        known_wetlands = (
            known_wetlands[
                known_wetlands.geometry.notna()
                &
                ~known_wetlands.geometry.is_empty
            ]
            .copy()
        )

        invalid_count = int(
            (
                ~known_wetlands.geometry.is_valid
            ).sum()
        )

        if (
            invalid_count > 0
        ):

            known_wetlands[
                "geometry"
            ] = (
                known_wetlands.geometry
                .buffer(
                    0
                )
            )

        known_wetlands = (
            known_wetlands.to_crs(
                raster_crs
            )
        )

        print(
            f"Known wetlands after geometry QA: "
            f"{len(known_wetlands)}"
        )

        print(
            f"Invalid geometries repaired: "
            f"{invalid_count}"
        )

        # ====================================================
        # 4. RASTERIZE KNOWN WETLANDS
        # ====================================================

        known_shapes = [
            (
                geometry,
                1,
            )
            for geometry
            in known_wetlands.geometry
            if (
                geometry
                is not None
                and
                not geometry.is_empty
            )
        ]

        known_mask = rasterize(
            known_shapes,
            out_shape=(
                height,
                width,
            ),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        ).astype(
            bool
        )

        # ----------------------------------------------------
        # 20 m proximity zone
        #
        # Diagnostic only. It is NOT removed.
        # ----------------------------------------------------

        buffered_shapes = [
            (
                geometry.buffer(
                    KNOWN_WETLAND_NEAR_DISTANCE_M
                ),
                1,
            )
            for geometry
            in known_wetlands.geometry
            if (
                geometry
                is not None
                and
                not geometry.is_empty
            )
        ]

        known_buffer_mask = rasterize(
            buffered_shapes,
            out_shape=(
                height,
                width,
            ),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        ).astype(
            bool
        )

        near_known_zone = (
            known_buffer_mask
            &
            ~known_mask
        )

        # ====================================================
        # 5. REMOVE ONLY EXISTING KNOWN-WETLAND PIXELS
        #
        # We are searching for ADDITIONAL candidate areas.
        #
        # The 20 m proximity zone is retained and only flagged.
        # ====================================================

        positive_inside_known = (
            raw_positive
            &
            known_mask
        )

        preliminary_candidate = (
            raw_positive
            &
            ~known_mask
        )

        positive_inside_known_count = int(
            positive_inside_known.sum()
        )

        preliminary_candidate_count = int(
            preliminary_candidate.sum()
        )

        candidate_near_known_count = int(
            (
                preliminary_candidate
                &
                near_known_zone
            ).sum()
        )

        # ====================================================
        # 6. CONNECTED-COMPONENT LABELLING
        # ====================================================

        component_labels, component_count = (
            ndimage.label(
                preliminary_candidate,
                structure=CONNECTIVITY_STRUCTURE,
            )
        )

        component_labels = (
            component_labels
            .astype(
                np.int32
            )
        )

        print(
            f"\nConnected preliminary components: "
            f"{component_count:,}"
        )

        if (
            component_count == 0
        ):

            raise RuntimeError(
                "No preliminary candidate components "
                "were detected."
            )

        component_ids = np.arange(
            1,
            component_count + 1,
            dtype=np.int32,
        )

        # ====================================================
        # 7. BASIC COMPONENT STATISTICS
        # ====================================================

        pixel_counts = (
            np.bincount(
                component_labels.ravel()
            )[
                1:
                component_count + 1
            ]
        )

        areas_ha = (
            pixel_counts
            *
            PIXEL_AREA_HA
        )

        probability_for_stats = (
            np.where(
                valid_probability,
                probability,
                0.0,
            )
        )

        mean_probability = (
            ndimage.mean(
                probability_for_stats,
                labels=component_labels,
                index=component_ids,
            )
        )

        min_probability = (
            ndimage.minimum(
                probability_for_stats,
                labels=component_labels,
                index=component_ids,
            )
        )

        max_probability = (
            ndimage.maximum(
                probability_for_stats,
                labels=component_labels,
                index=component_ids,
            )
        )

        # ----------------------------------------------------
        # Whether each component intersects the 20 m zone
        # surrounding an existing known wetland.
        # ----------------------------------------------------

        near_known_component = (
            ndimage.maximum(
                near_known_zone.astype(
                    np.uint8
                ),
                labels=component_labels,
                index=component_ids,
            )
            > 0
        )

        # ====================================================
        # 8. COMPONENT CENTROIDS
        # ====================================================

        centroids = (
            ndimage.center_of_mass(
                preliminary_candidate.astype(
                    np.uint8
                ),
                labels=component_labels,
                index=component_ids,
            )
        )

        centroid_rows = np.array(
            [
                centroid[0]
                for centroid
                in centroids
            ],
            dtype=float,
        )

        centroid_cols = np.array(
            [
                centroid[1]
                for centroid
                in centroids
            ],
            dtype=float,
        )

        centroid_x, centroid_y = (
            rasterio.transform.xy(
                transform,
                centroid_rows,
                centroid_cols,
                offset="center",
            )
        )

        centroid_x = np.asarray(
            centroid_x,
            dtype=float,
        )

        centroid_y = np.asarray(
            centroid_y,
            dtype=float,
        )

        # ====================================================
        # 9. SELECTED FEATURE DIAGNOSTICS
        #
        # These values are descriptive only.
        # They are NOT being used to create a new classifier.
        # ====================================================

        component_feature_means = {}

        for feature_name in (
            required_diagnostic_features
        ):

            band_number = (
                feature_names.index(
                    feature_name
                )
                + 1
            )

            feature_array = (
                feature_src.read(
                    band_number
                )
                .astype(
                    np.float32
                )
            )

            means = (
                ndimage.mean(
                    feature_array,
                    labels=component_labels,
                    index=component_ids,
                )
            )

            component_feature_means[
                feature_name
            ] = (
                np.asarray(
                    means,
                    dtype=float,
                )
            )

        # ====================================================
        # 10. BUILD COMPONENT TABLE
        # ====================================================

        component_stats = pd.DataFrame(
            {
                "component_id":
                    component_ids,

                "pixel_count":
                    pixel_counts,

                "area_ha":
                    areas_ha,

                "mean_probability":
                    mean_probability,

                "min_probability":
                    min_probability,

                "max_probability":
                    max_probability,

                "centroid_x_utm":
                    centroid_x,

                "centroid_y_utm":
                    centroid_y,

                "near_known_wetland_20m":
                    near_known_component,
            }
        )

        for (
            feature_name,
            values,
        ) in component_feature_means.items():

            component_stats[
                f"mean_{feature_name}"
            ] = values

        component_stats = (
            component_stats
            .sort_values(
                [
                    "area_ha",
                    "mean_probability",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )

        component_stats.to_csv(
            COMPONENT_STATS_OUTPUT,
            index=False,
        )

        # ====================================================
        # 11. MMU SCENARIO ANALYSIS
        #
        # Diagnostic only.
        #
        # We calculate how many components and how much area
        # would remain under several possible minimum mapping
        # units.
        # ====================================================

        mmu_rows = []

        total_preliminary_area_ha = (
            preliminary_candidate_count
            *
            PIXEL_AREA_HA
        )

        for minimum_area_ha in (
            MMU_SCENARIOS_HA
        ):

            retained = (
                component_stats[
                    "area_ha"
                ]
                >=
                minimum_area_ha
            )

            retained_components = int(
                retained.sum()
            )

            retained_pixels = int(
                component_stats.loc[
                    retained,
                    "pixel_count",
                ]
                .sum()
            )

            retained_area_ha = (
                retained_pixels
                *
                PIXEL_AREA_HA
            )

            retained_near_known = int(
                component_stats.loc[
                    retained,
                    "near_known_wetland_20m",
                ]
                .sum()
            )

            mmu_rows.append(
                {
                    "minimum_area_ha":
                        minimum_area_ha,

                    "minimum_pixels":
                        int(
                            np.ceil(
                                minimum_area_ha
                                /
                                PIXEL_AREA_HA
                            )
                        ),

                    "retained_components":
                        retained_components,

                    "retained_pixels":
                        retained_pixels,

                    "retained_area_ha":
                        retained_area_ha,

                    "percentage_of_preliminary_area":
                        (
                            100.0
                            *
                            retained_area_ha
                            /
                            total_preliminary_area_ha
                            if
                            total_preliminary_area_ha > 0
                            else
                            0.0
                        ),

                    "retained_components_near_known_20m":
                        retained_near_known,
                }
            )

        mmu_table = pd.DataFrame(
            mmu_rows
        )

        mmu_table.to_csv(
            MMU_SCENARIOS_OUTPUT,
            index=False,
        )

        # ====================================================
        # 12. SAVE PRELIMINARY CANDIDATE MASK
        # ====================================================

        candidate_mask = np.full(
            (
                height,
                width,
            ),
            BINARY_NODATA,
            dtype=np.uint8,
        )

        candidate_mask[
            valid_probability
        ] = 0

        candidate_mask[
            preliminary_candidate
        ] = 1

        candidate_profile = (
            probability_src.profile.copy()
        )

        candidate_profile.update(
            {
                "count": 1,
                "dtype": "uint8",
                "nodata": BINARY_NODATA,
                "compress": "deflate",
            }
        )

        with rasterio.open(
            PRELIMINARY_CANDIDATE_MASK_OUTPUT,
            "w",
            **candidate_profile,
        ) as dst:

            dst.write(
                candidate_mask,
                1,
            )

        # ====================================================
        # 13. SAVE COMPONENT-ID RASTER
        # ====================================================

        component_profile = (
            probability_src.profile.copy()
        )

        component_profile.update(
            {
                "count": 1,
                "dtype": "int32",
                "nodata": COMPONENT_NODATA,
                "compress": "deflate",
            }
        )

        with rasterio.open(
            COMPONENT_ID_OUTPUT,
            "w",
            **component_profile,
        ) as dst:

            dst.write(
                component_labels,
                1,
            )

    # ========================================================
    # 14. SUMMARY / AUDIT
    # ========================================================

    largest_component_ha = float(
        component_stats[
            "area_ha"
        ].max()
    )

    median_component_ha = float(
        component_stats[
            "area_ha"
        ].median()
    )

    single_pixel_components = int(
        (
            component_stats[
                "pixel_count"
            ]
            == 1
        ).sum()
    )

    near_known_components = int(
        component_stats[
            "near_known_wetland_20m"
        ].sum()
    )

    audit = {
        "frozen_probability_threshold":
            FROZEN_THRESHOLD,

        "connectivity":
            8,

        "pixel_area_ha":
            PIXEL_AREA_HA,

        "valid_study_pixels":
            valid_pixel_count,

        "raw_model_positive_pixels":
            raw_positive_count,

        "raw_model_positive_area_ha":
            raw_positive_count
            *
            PIXEL_AREA_HA,

        "model_positive_inside_known_wetlands_pixels":
            positive_inside_known_count,

        "model_positive_inside_known_wetlands_area_ha":
            positive_inside_known_count
            *
            PIXEL_AREA_HA,

        "preliminary_candidate_pixels_outside_known":
            preliminary_candidate_count,

        "preliminary_candidate_area_ha_outside_known":
            preliminary_candidate_count
            *
            PIXEL_AREA_HA,

        "preliminary_pixels_within_20m_of_known":
            candidate_near_known_count,

        "connected_component_count":
            int(
                component_count
            ),

        "components_near_known_wetland_20m":
            near_known_components,

        "single_pixel_component_count":
            single_pixel_components,

        "largest_component_area_ha":
            largest_component_ha,

        "median_component_area_ha":
            median_component_ha,

        "important_note":
            (
                "This script does not define the final "
                "candidate wetland layer. Existing known "
                "wetland pixels are excluded, while MMU "
                "selection and open-water ambiguity remain "
                "for later evidence-based post-processing."
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
        "PRELIMINARY CANDIDATE SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        f"Valid study pixels: "
        f"{valid_pixel_count:,}"
    )

    print(
        f"\nRaw model-positive pixels: "
        f"{raw_positive_count:,}"
    )

    print(
        f"Raw model-positive area: "
        f"{raw_positive_count * PIXEL_AREA_HA:.2f} ha"
    )

    print(
        f"\nModel-positive pixels inside "
        f"known wetlands: "
        f"{positive_inside_known_count:,}"
    )

    print(
        f"Area inside known wetlands: "
        f"{positive_inside_known_count * PIXEL_AREA_HA:.2f} ha"
    )

    print(
        f"\nPreliminary positive pixels "
        f"outside known wetlands: "
        f"{preliminary_candidate_count:,}"
    )

    print(
        f"Preliminary outside-known area: "
        f"{preliminary_candidate_count * PIXEL_AREA_HA:.2f} ha"
    )

    print(
        f"\nConnected components: "
        f"{component_count:,}"
    )

    print(
        f"Single-pixel components: "
        f"{single_pixel_components:,}"
    )

    print(
        f"Median component area: "
        f"{median_component_ha:.4f} ha"
    )

    print(
        f"Largest component area: "
        f"{largest_component_ha:.2f} ha"
    )

    print(
        f"Components within 20 m of "
        f"known wetlands: "
        f"{near_known_components:,}"
    )

    print(
        "\nMMU scenario comparison:"
    )

    print(
        mmu_table.to_string(
            index=False
        )
    )

    print(
        "\nTop 20 largest preliminary components:"
    )

    print(
        component_stats[
            [
                "component_id",
                "pixel_count",
                "area_ha",
                "mean_probability",
                "max_probability",
                "near_known_wetland_20m",
                "mean_dry_ndvi",
                "mean_wet_ndvi",
                "mean_dry_mndwi",
                "mean_wet_mndwi",
                "mean_elevation_m",
                "mean_distance_to_drainage_m",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )

    print(
        PRELIMINARY_CANDIDATE_MASK_OUTPUT
    )

    print(
        COMPONENT_ID_OUTPUT
    )

    print(
        COMPONENT_STATS_OUTPUT
    )

    print(
        MMU_SCENARIOS_OUTPUT
    )

    print(
        AUDIT_OUTPUT
    )

    print(
        "\n========================================"
    )

    print(
        "CANDIDATE COMPONENT DIAGNOSTICS: COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        "\nIMPORTANT:"
        "\nThis is still a preliminary candidate layer."
        "\nNo MMU has been permanently selected."
        "\nNo open-water pixels have been automatically "
        "removed using an unvalidated spectral rule."
        "\nThe next decision will use the component-size "
        "distribution and spatial QA results."
    )


if __name__ == "__main__":
    main()