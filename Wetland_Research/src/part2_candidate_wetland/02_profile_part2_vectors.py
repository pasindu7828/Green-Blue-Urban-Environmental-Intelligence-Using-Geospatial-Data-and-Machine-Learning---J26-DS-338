from pathlib import Path
from tempfile import TemporaryDirectory
from collections import Counter
import zipfile
import xml.etree.ElementTree as ET

import geopandas as gpd
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

OSM_DIR = RAW_DIR / "osm"
LULC_DIR = RAW_DIR / "lulc_reference"

PART1_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "part1_historical_change"
)

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "part2_candidate_wetland"
)

INTERIM_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUT FILES
# ============================================================

OSM_ROADS_BUILDINGS = (
    OSM_DIR
    / "Kaduwela_OSM_Roads_Buildings_2026.gpkg"
)

OSM_WATER_DRAINAGE = (
    OSM_DIR
    / "Kaduwela_OSM_Water_Drainage_2026.gpkg"
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

OSM_PROFILE_OUTPUT = (
    INTERIM_DIR
    / "osm_attribute_profile.csv"
)

LULC_PROFILE_OUTPUT = (
    INTERIM_DIR
    / "lulc_2025_reference_profile.csv"
)


# ============================================================
# HELPERS
# ============================================================

def profile_vector(
    path,
    label,
):

    print("\n========================================")
    print(label)
    print("========================================")

    data = gpd.read_file(
        path
    )

    print(
        f"Features: {len(data)}"
    )

    print(
        f"CRS: {data.crs}"
    )

    print("\nGeometry types:")

    print(
        data.geometry
        .geom_type
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print("\nColumns:")

    for column in data.columns:

        print(
            f" - {column}"
        )

    # --------------------------------------------------------
    # PROFILE ATTRIBUTE COLUMNS
    # --------------------------------------------------------

    rows = []

    for column in data.columns:

        if column == "geometry":
            continue

        non_null = data[
            column
        ].dropna()

        unique_count = (
            non_null
            .astype(str)
            .nunique()
        )

        example_values = (
            non_null
            .astype(str)
            .value_counts()
            .head(10)
        )

        example_text = " | ".join(
            [
                f"{index}:{value}"
                for index, value
                in example_values.items()
            ]
        )

        rows.append(
            {
                "dataset":
                    label,

                "column":
                    column,

                "non_null_count":
                    len(non_null),

                "unique_count":
                    unique_count,

                "top_values":
                    example_text,
            }
        )

        # Print only useful non-empty columns.
        if len(non_null) > 0:

            print(
                f"\n{column}"
            )

            print(
                f"  Non-null: "
                f"{len(non_null)}"
            )

            print(
                f"  Unique: "
                f"{unique_count}"
            )

            print(
                "  Top values:"
            )

            print(
                example_values.to_string()
            )

    return data, rows


def profile_lulc_kmz(
    kmz_path,
):

    print("\n========================================")
    print("2025 LULC REFERENCE")
    print("========================================")

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
                "LULC 2025 KMZ."
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
                or "Unnamed"
            )

            placemarks = folder.findall(
                "kml:Placemark",
                ns,
            )

            for placemark in placemarks:

                point = placemark.find(
                    ".//kml:Point",
                    ns,
                )

                if point is None:
                    continue

                coord_node = point.find(
                    "kml:coordinates",
                    ns,
                )

                if (
                    coord_node is None
                    or
                    not coord_node.text
                ):
                    continue

                coordinate_text = (
                    coord_node.text
                    .strip()
                )

                parts = (
                    coordinate_text
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
                        "class_name":
                            class_name,

                        "sample_name":
                            sample_name,

                        "longitude":
                            longitude,

                        "latitude":
                            latitude,
                    }
                )

    lulc = pd.DataFrame(
        rows
    )

    print(
        f"Samples parsed: "
        f"{len(lulc)}"
    )

    print(
        "\nClass counts:"
    )

    print(
        lulc[
            "class_name"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return lulc


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("PART 2 VECTOR / REFERENCE PROFILE")
    print("========================================")

    all_profile_rows = []

    # --------------------------------------------------------
    # OSM ROADS / BUILDINGS
    # --------------------------------------------------------

    roads_buildings, profile_rows = (
        profile_vector(
            OSM_ROADS_BUILDINGS,
            "OSM Roads / Buildings",
        )
    )

    all_profile_rows.extend(
        profile_rows
    )

    # --------------------------------------------------------
    # OSM WATER / DRAINAGE
    # --------------------------------------------------------

    water_drainage, profile_rows = (
        profile_vector(
            OSM_WATER_DRAINAGE,
            "OSM Water / Drainage",
        )
    )

    all_profile_rows.extend(
        profile_rows
    )

    # --------------------------------------------------------
    # LULC REFERENCE
    # --------------------------------------------------------

    lulc = profile_lulc_kmz(
        LULC_2025
    )

    lulc.to_csv(
        LULC_PROFILE_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # KNOWN WETLANDS
    # --------------------------------------------------------

    print("\n========================================")
    print("KNOWN WETLAND POSITIVE AREAS")
    print("========================================")

    wetlands = gpd.read_file(
        KNOWN_WETLANDS
    )

    print(
        f"Wetlands: "
        f"{wetlands['wetland_id'].nunique()}"
    )

    print(
        "\nLatest observation date distribution:"
    )

    print(
        wetlands[
            "observation_date"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # SAVE ATTRIBUTE PROFILE
    # --------------------------------------------------------

    profile = pd.DataFrame(
        all_profile_rows
    )

    profile.to_csv(
        OSM_PROFILE_OUTPUT,
        index=False,
    )

    print("\n========================================")
    print("OUTPUTS")
    print("========================================")

    print(
        OSM_PROFILE_OUTPUT
    )

    print(
        LULC_PROFILE_OUTPUT
    )

    print(
        "\nNEXT:"
        "\nUse the confirmed OSM fields to build "
        "distance-to-drainage, distance-to-road "
        "and building/urban context features."
    )


if __name__ == "__main__":
    main()