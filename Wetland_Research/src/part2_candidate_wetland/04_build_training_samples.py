from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from rasterio.features import rasterize
from rasterio.transform import rowcol


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

LULC_DIR = (
    RAW_DIR
    / "lulc_reference"
)

PART1_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "part1_historical_change"
)

PART2_PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "part2_candidate_wetland"
)

PART2_INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "part2_candidate_wetland"
)

PART2_PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PART2_INTERIM_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUTS
# ============================================================

FEATURE_STACK = (
    PART2_PROCESSED_DIR
    / "feature_stack_10m.tif"
)

STUDY_MASK = (
    PART2_PROCESSED_DIR
    / "study_mask_10m.tif"
)

LULC_2025 = (
    LULC_DIR
    / "LULC_Reference_Samples_2025.kmz"
)

KNOWN_WETLANDS = (
    PART1_DIR
    / "wetland_latest_status_gn_wgs84.geojson"
)


# ============================================================
# OUTPUTS
# ============================================================

TRAINING_OUTPUT = (
    PART2_PROCESSED_DIR
    / "training_samples_raw.csv"
)

TRAINING_GEOJSON_OUTPUT = (
    PART2_PROCESSED_DIR
    / "training_samples_raw_wgs84.geojson"
)

LABEL_AUDIT_OUTPUT = (
    PART2_INTERIM_DIR
    / "training_label_audit.csv"
)

WETLAND_SAMPLING_AUDIT_OUTPUT = (
    PART2_INTERIM_DIR
    / "known_wetland_sampling_audit.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

# Cap prevents large wetlands from dominating.
POSITIVE_PIXELS_PER_WETLAND = 20

# Remove boundary-edge uncertainty from polygon sampling.
POSITIVE_INNER_BUFFER_M = 10.0

# Two Sentinel pixels around known wetlands are treated
# as ambiguous for NON-WETLAND reference points.
NEGATIVE_EXCLUSION_BUFFER_M = 20.0

# Used later for spatially independent splitting.
SPATIAL_BLOCK_SIZE_M = 1000.0


# ============================================================
# LULC CLASS DEFINITIONS
# ============================================================

CLASS_TARGETS = {
    "01_Wetland_Marsh": 1,
    "02_Open_Water": 0,
    "03_Built_Up": 0,
    "04_Paddy_Agriculture": 0,
    "05_Other_Vegetation": 0,
    "06_Bare_Fill_Disturbed": 0,
}

CLASS_LABELS = {
    "01_Wetland_Marsh": "wetland_marsh",
    "02_Open_Water": "open_water",
    "03_Built_Up": "built_up",
    "04_Paddy_Agriculture": "paddy_agriculture",
    "05_Other_Vegetation": "other_vegetation",
    "06_Bare_Fill_Disturbed": "bare_fill_disturbed",
}


# ============================================================
# HELPERS
# ============================================================

def parse_lulc_kmz(
    kmz_path,
):

    rows = []

    with TemporaryDirectory() as temp_dir:

        with zipfile.ZipFile(
            kmz_path,
            "r",
        ) as archive:

            archive.extractall(
                temp_dir
            )

        kml_files = list(
            Path(temp_dir).rglob(
                "*.kml"
            )
        )

        if not kml_files:

            raise RuntimeError(
                "No KML file found inside "
                "the 2025 LULC KMZ."
            )

        tree = ET.parse(
            kml_files[0]
        )

        root = tree.getroot()

        ns = {
            "kml":
                "http://www.opengis.net/kml/2.2"
        }

        for folder in root.findall(
            ".//kml:Folder",
            ns,
        ):

            folder_name_node = (
                folder.find(
                    "kml:name",
                    ns,
                )
            )

            if folder_name_node is None:
                continue

            class_name = (
                folder_name_node.text
                or ""
            )

            if (
                class_name
                not in CLASS_TARGETS
            ):
                continue

            for placemark in folder.findall(
                "kml:Placemark",
                ns,
            ):

                point = placemark.find(
                    ".//kml:Point",
                    ns,
                )

                if point is None:
                    continue

                coordinates = point.find(
                    "kml:coordinates",
                    ns,
                )

                if (
                    coordinates is None
                    or
                    not coordinates.text
                ):
                    continue

                parts = (
                    coordinates.text
                    .strip()
                    .split(",")
                )

                longitude = float(
                    parts[0]
                )

                latitude = float(
                    parts[1]
                )

                name_node = (
                    placemark.find(
                        "kml:name",
                        ns,
                    )
                )

                sample_name = (
                    name_node.text
                    if (
                        name_node is not None
                        and
                        name_node.text
                    )
                    else ""
                )

                rows.append(
                    {
                        "reference_class":
                            class_name,

                        "reference_label":
                            CLASS_LABELS[
                                class_name
                            ],

                        "target":
                            CLASS_TARGETS[
                                class_name
                            ],

                        "sample_name":
                            sample_name,

                        "longitude":
                            longitude,

                        "latitude":
                            latitude,
                    }
                )

    return pd.DataFrame(
        rows
    )


def spatial_block_id(
    x,
    y,
):

    block_x = int(
        np.floor(
            x
            / SPATIAL_BLOCK_SIZE_M
        )
    )

    block_y = int(
        np.floor(
            y
            / SPATIAL_BLOCK_SIZE_M
        )
    )

    return (
        f"B_{block_x}_{block_y}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("PART 2 TRAINING SAMPLE CONSTRUCTION")
    print("========================================")

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # ========================================================
    # 1. LOAD FEATURE STACK
    # ========================================================

    with rasterio.open(
        FEATURE_STACK
    ) as src:

        features = (
            src.read()
            .astype(
                np.float32
            )
        )

        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        width = src.width
        height = src.height

        feature_names = list(
            src.descriptions
        )

    print(
        f"Feature bands loaded: "
        f"{len(feature_names)}"
    )

    print(
        f"Grid: "
        f"{width} x {height}"
    )

    # ========================================================
    # 2. LOAD VALID ANALYSIS MASK
    # ========================================================

    with rasterio.open(
        STUDY_MASK
    ) as src:

        valid_mask = (
            src.read(
                1
            )
            == 1
        )

    print(
        f"Valid analysis cells: "
        f"{int(valid_mask.sum()):,}"
    )

    # ========================================================
    # 3. LOAD KNOWN WETLAND POLYGONS
    # ========================================================

    wetlands = gpd.read_file(
        KNOWN_WETLANDS
    )

    wetlands = wetlands.to_crs(
        crs
    )

    wetlands[
        "geometry"
    ] = (
        wetlands.geometry
        .make_valid()
    )

    known_wetland_union = (
        wetlands.geometry
        .union_all()
    )

    known_wetland_exclusion = (
        known_wetland_union
        .buffer(
            NEGATIVE_EXCLUSION_BUFFER_M
        )
    )

    print(
        f"Known wetlands loaded: "
        f"{wetlands['wetland_id'].nunique()}"
    )

    # ========================================================
    # 4. PARSE 2025 LULC POINTS
    # ========================================================

    lulc = parse_lulc_kmz(
        LULC_2025
    )

    print(
        f"2025 LULC points parsed: "
        f"{len(lulc)}"
    )

    lulc_gdf = gpd.GeoDataFrame(
        lulc.copy(),
        geometry=gpd.points_from_xy(
            lulc[
                "longitude"
            ],
            lulc[
                "latitude"
            ],
        ),
        crs="EPSG:4326",
    )

    lulc_gdf = lulc_gdf.to_crs(
        crs
    )

    # ========================================================
    # 5. EXTRACT LULC REFERENCE FEATURES
    # ========================================================

    lulc_rows = []

    lulc_outside_grid = 0
    lulc_invalid_feature = 0
    ambiguous_negative_count = 0

    for _, row in lulc_gdf.iterrows():

        x = row.geometry.x
        y = row.geometry.y

        raster_row, raster_col = rowcol(
            transform,
            x,
            y,
        )

        if (
            raster_row < 0
            or
            raster_row >= height
            or
            raster_col < 0
            or
            raster_col >= width
        ):

            lulc_outside_grid += 1

            continue

        if not valid_mask[
            raster_row,
            raster_col
        ]:

            lulc_invalid_feature += 1

            continue

        # ----------------------------------------------------
        # REMOVE AMBIGUOUS NEGATIVE LABELS
        #
        # A non-wetland point within 20 m of a mapped known
        # wetland is not used for model training.
        # ----------------------------------------------------

        if (
            row["target"] == 0
            and
            known_wetland_exclusion.intersects(
                row.geometry
            )
        ):

            ambiguous_negative_count += 1

            continue

        feature_vector = (
            features[
                :,
                raster_row,
                raster_col,
            ]
        )

        if (
            nodata is not None
            and
            np.any(
                feature_vector
                == nodata
            )
        ):

            lulc_invalid_feature += 1

            continue

        sample = {
            "sample_source":
                "lulc_reference_2025",

            "source_priority":
                0,

            "source_site_id":
                row[
                    "sample_name"
                ],

            "reference_class":
                row[
                    "reference_class"
                ],

            "reference_label":
                row[
                    "reference_label"
                ],

            "wetland_id":
                "",

            "target":
                int(
                    row[
                        "target"
                    ]
                ),

            "x_utm":
                x,

            "y_utm":
                y,

            "raster_row":
                raster_row,

            "raster_col":
                raster_col,

            "spatial_block_id":
                spatial_block_id(
                    x,
                    y,
                ),
        }

        for (
            feature_name,
            feature_value,
        ) in zip(
            feature_names,
            feature_vector,
        ):

            sample[
                feature_name
            ] = float(
                feature_value
            )

        lulc_rows.append(
            sample
        )

    lulc_samples = pd.DataFrame(
        lulc_rows
    )

    print(
        f"\nUsable LULC samples: "
        f"{len(lulc_samples)}"
    )

    print(
        f"LULC outside grid: "
        f"{lulc_outside_grid}"
    )

    print(
        f"LULC invalid/no-data: "
        f"{lulc_invalid_feature}"
    )

    print(
        f"Ambiguous negative points excluded: "
        f"{ambiguous_negative_count}"
    )

    # ========================================================
    # 6. SUPPLEMENTARY POSITIVE PIXELS FROM KNOWN WETLANDS
    #
    # Sampling hierarchy:
    #
    # 1. Prefer pixels whose centres fall inside a
    #    10 m inward-buffered wetland.
    #
    # 2. If that produces zero pixels, use the original
    #    wetland geometry with pixel-centre sampling.
    #
    # 3. If that still produces zero pixels, use the
    #    original geometry with all_touched=True.
    #
    # This prevents small/narrow wetlands from silently
    # disappearing from the training evidence.
    # ========================================================

    print(
        "\nSampling known wetland interiors..."
    )

    polygon_positive_rows = []
    sampling_audit_rows = []

    for _, wetland in wetlands.iterrows():

        wetland_id = (
            wetland[
                "wetland_id"
            ]
        )

        observation_date = (
            wetland[
                "observation_date"
            ]
        )

        original_geometry = (
            wetland.geometry
        )

        inner_geometry = (
            original_geometry
            .buffer(
                -POSITIVE_INNER_BUFFER_M
            )
        )

        # ----------------------------------------------------
        # PREPARE SAMPLING ATTEMPTS
        # ----------------------------------------------------

        attempts = []

        if (
            inner_geometry is not None
            and
            not inner_geometry.is_empty
            and
            inner_geometry.area > 0
        ):

            attempts.append(
                (
                    "inner_10m_pixel_center",
                    inner_geometry,
                    False,
                )
            )

        attempts.append(
            (
                "original_pixel_center",
                original_geometry,
                False,
            )
        )

        attempts.append(
            (
                "original_all_touched",
                original_geometry,
                True,
            )
        )

        selected_method = None
        candidate_rows = np.array(
            [],
            dtype=int,
        )

        candidate_cols = np.array(
            [],
            dtype=int,
        )

        # ----------------------------------------------------
        # TRY EACH METHOD UNTIL USABLE CELLS EXIST
        # ----------------------------------------------------

        for (
            method_name,
            sampling_geometry,
            all_touched_value,
        ) in attempts:

            wetland_mask = rasterize(
                [
                    (
                        sampling_geometry,
                        1,
                    )
                ],
                out_shape=(
                    height,
                    width,
                ),
                transform=transform,
                fill=0,
                all_touched=all_touched_value,
                dtype="uint8",
            )

            rows_found, cols_found = (
                np.where(
                    (
                        wetland_mask
                        == 1
                    )
                    &
                    valid_mask
                )
            )

            if len(
                rows_found
            ) > 0:

                selected_method = (
                    method_name
                )

                candidate_rows = (
                    rows_found
                )

                candidate_cols = (
                    cols_found
                )

                break

        available_pixels = len(
            candidate_rows
        )

        # ----------------------------------------------------
        # HANDLE TRUE ZERO-PIXEL CASE
        # ----------------------------------------------------

        if available_pixels == 0:

            sampling_audit_rows.append(
                {
                    "wetland_id":
                        wetland_id,

                    "observation_date":
                        observation_date,

                    "available_pixels":
                        0,

                    "sampled_pixels":
                        0,

                    "sampling_method":
                        "none",

                    "inner_buffer_fallback":
                        True,
                }
            )

            continue

        # ----------------------------------------------------
        # SAMPLE UP TO THE SITE CAP
        # ----------------------------------------------------

        sample_count = min(
            POSITIVE_PIXELS_PER_WETLAND,
            available_pixels,
        )

        selected_indices = (
            rng.choice(
                available_pixels,
                size=sample_count,
                replace=False,
            )
        )

        selected_rows = (
            candidate_rows[
                selected_indices
            ]
        )

        selected_cols = (
            candidate_cols[
                selected_indices
            ]
        )

        xs, ys = (
            rasterio.transform.xy(
                transform,
                selected_rows,
                selected_cols,
                offset="center",
            )
        )

        for (
            raster_row,
            raster_col,
            x,
            y,
        ) in zip(
            selected_rows,
            selected_cols,
            xs,
            ys,
        ):

            feature_vector = (
                features[
                    :,
                    raster_row,
                    raster_col,
                ]
            )

            sample = {
                "sample_source":
                    "known_wetland_polygon",

                "source_priority":
                    1,

                "source_site_id":
                    wetland_id,

                "reference_class":
                    "Known_Wetland",

                "reference_label":
                    "known_wetland",

                "wetland_id":
                    wetland_id,

                "target":
                    1,

                "x_utm":
                    float(
                        x
                    ),

                "y_utm":
                    float(
                        y
                    ),

                "raster_row":
                    int(
                        raster_row
                    ),

                "raster_col":
                    int(
                        raster_col
                    ),

                "spatial_block_id":
                    spatial_block_id(
                        x,
                        y,
                    ),
            }

            for (
                feature_name,
                feature_value,
            ) in zip(
                feature_names,
                feature_vector,
            ):

                sample[
                    feature_name
                ] = float(
                    feature_value
                )

            polygon_positive_rows.append(
                sample
            )

        sampling_audit_rows.append(
            {
                "wetland_id":
                    wetland_id,

                "observation_date":
                    observation_date,

                "available_pixels":
                    available_pixels,

                "sampled_pixels":
                    sample_count,

                "sampling_method":
                    selected_method,

                "inner_buffer_fallback":
                    (
                        selected_method
                        !=
                        "inner_10m_pixel_center"
                    ),
            }
        )

    polygon_samples = pd.DataFrame(
        polygon_positive_rows
    )

    wetland_sampling_audit = (
        pd.DataFrame(
            sampling_audit_rows
        )
    )

    wetland_sampling_audit.to_csv(
        WETLAND_SAMPLING_AUDIT_OUTPUT,
        index=False,
    )

    print(
        f"Supplementary wetland positive pixels: "
        f"{len(polygon_samples)}"
    )

    print(
        f"Wetlands requiring fallback sampling: "
        f"{int(wetland_sampling_audit['inner_buffer_fallback'].sum())}"
    )

    print(
        "\nSampling methods:"
    )

    print(
        wetland_sampling_audit[
            "sampling_method"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    # ========================================================
    # 7. COMBINE LABEL SOURCES
    # ========================================================

    combined = pd.concat(
        [
            lulc_samples,
            polygon_samples,
        ],
        ignore_index=True,
    )

    initial_combined_count = len(
        combined
    )

    # ========================================================
    # 8. IDENTIFY CONFLICTING CELLS
    #
    # If one source calls the same 10 m cell wetland and
    # another calls it non-wetland, remove that cell rather
    # than forcing one source to win.
    # ========================================================

    target_counts = (
        combined
        .groupby(
            [
                "raster_row",
                "raster_col",
            ]
        )[
            "target"
        ]
        .nunique()
    )

    conflict_cells = set(
        target_counts[
            target_counts > 1
        ].index
    )

    conflict_mask = combined.apply(
        lambda row:
            (
                row[
                    "raster_row"
                ],
                row[
                    "raster_col"
                ],
            )
            in conflict_cells,
        axis=1,
    )

    conflict_rows_removed = int(
        conflict_mask.sum()
    )

    combined = combined[
        ~conflict_mask
    ].copy()

    # ========================================================
    # 9. REMOVE DUPLICATE SAME-LABEL CELLS
    #
    # Prefer the directly interpreted 2025 LULC point over
    # the supplementary polygon sample when both occupy the
    # same raster cell.
    # ========================================================

    combined = (
        combined
        .sort_values(
            [
                "source_priority",
                "raster_row",
                "raster_col",
            ]
        )
        .drop_duplicates(
            subset=[
                "raster_row",
                "raster_col",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    duplicate_rows_removed = (
        initial_combined_count
        -
        conflict_rows_removed
        -
        len(
            combined
        )
    )

    # ========================================================
    # 10. FINAL SPATIAL GROUP IDS
    # ========================================================

    combined[
        "sample_id"
    ] = [
        f"S{i:04d}"
        for i in range(
            1,
            len(combined) + 1,
        )
    ]

    # ========================================================
    # 11. SAVE CSV
    # ========================================================

    output_columns = [
        "sample_id",
        "sample_source",
        "source_site_id",
        "reference_class",
        "reference_label",
        "wetland_id",
        "target",
        "x_utm",
        "y_utm",
        "raster_row",
        "raster_col",
        "spatial_block_id",
    ] + feature_names

    combined[
        output_columns
    ].to_csv(
        TRAINING_OUTPUT,
        index=False,
    )

    # ========================================================
    # 12. SAVE SPATIAL SAMPLE LAYER
    # ========================================================

    training_gdf = gpd.GeoDataFrame(
        combined.copy(),
        geometry=gpd.points_from_xy(
            combined[
                "x_utm"
            ],
            combined[
                "y_utm"
            ],
        ),
        crs=crs,
    )

    training_gdf = (
        training_gdf
        .to_crs(
            "EPSG:4326"
        )
    )

    training_gdf.to_file(
        TRAINING_GEOJSON_OUTPUT,
        driver="GeoJSON",
    )

    # ========================================================
    # 13. LABEL AUDIT
    # ========================================================

    final_positive = int(
        (
            combined[
                "target"
            ]
            == 1
        ).sum()
    )

    final_negative = int(
        (
            combined[
                "target"
            ]
            == 0
        ).sum()
    )

    audit_rows = [
        {
            "metric":
                "original_lulc_points",

            "value":
                len(
                    lulc
                ),
        },
        {
            "metric":
                "usable_lulc_samples",

            "value":
                len(
                    lulc_samples
                ),
        },
        {
            "metric":
                "supplementary_polygon_positive_samples",

            "value":
                len(
                    polygon_samples
                ),
        },
        {
            "metric":
                "ambiguous_negative_points_excluded",

            "value":
                ambiguous_negative_count,
        },
        {
            "metric":
                "conflicting_rows_removed",

            "value":
                conflict_rows_removed,
        },
        {
            "metric":
                "duplicate_same_cell_rows_removed",

            "value":
                duplicate_rows_removed,
        },
        {
            "metric":
                "final_training_samples",

            "value":
                len(
                    combined
                ),
        },
        {
            "metric":
                "final_positive_samples",

            "value":
                final_positive,
        },
        {
            "metric":
                "final_negative_samples",

            "value":
                final_negative,
        },
        {
            "metric":
                "unique_spatial_blocks",

            "value":
                combined[
                    "spatial_block_id"
                ]
                .nunique(),
        },
    ]

    audit = pd.DataFrame(
        audit_rows
    )

    audit.to_csv(
        LABEL_AUDIT_OUTPUT,
        index=False,
    )

    # ========================================================
    # TERMINAL REPORT
    # ========================================================

    print("\n========================================")
    print("TRAINING LABEL SUMMARY")
    print("========================================")

    print(
        f"Combined samples before conflict/"
        f"duplicate removal: "
        f"{initial_combined_count}"
    )

    print(
        f"Conflicting rows removed: "
        f"{conflict_rows_removed}"
    )

    print(
        f"Duplicate same-cell rows removed: "
        f"{duplicate_rows_removed}"
    )

    print(
        f"Final samples: "
        f"{len(combined)}"
    )

    print(
        f"Positive samples: "
        f"{final_positive}"
    )

    print(
        f"Negative samples: "
        f"{final_negative}"
    )

    print(
        f"Spatial blocks: "
        f"{combined['spatial_block_id'].nunique()}"
    )

    print(
        "\nFinal reference labels:"
    )

    print(
        combined[
            "reference_label"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nSample sources:"
    )

    print(
        combined[
            "sample_source"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nOutputs:"
    )

    print(
        TRAINING_OUTPUT
    )

    print(
        TRAINING_GEOJSON_OUTPUT
    )

    print(
        LABEL_AUDIT_OUTPUT
    )

    print(
        WETLAND_SAMPLING_AUDIT_OUTPUT
    )

    print(
        "\nIMPORTANT:"
        "\nKnown-wetland polygon pixels are "
        "supplementary positives, not independent "
        "2025 point observations."
        "\nNon-wetland points within 20 m of known "
        "wetland polygons are excluded as ambiguous."
        "\nThe 1 km spatial block IDs are preparation "
        "for spatially separated model evaluation."
    )


if __name__ == "__main__":
    main()