from pathlib import Path

import pyogrio
import geopandas as gpd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GN_KML = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "boundaries"
    / "Kaduwela TO Area.kml"
)


# ============================================================
# MAIN
# ============================================================

def main():

    if not GN_KML.exists():
        raise FileNotFoundError(
            f"GN boundary KML not found:\n{GN_KML}"
        )

    print("\n========================================")
    print("GN BOUNDARY DATA INSPECTION")
    print("========================================")

    # --------------------------------------------------------
    # LIST AVAILABLE KML LAYERS
    # --------------------------------------------------------

    layers = pyogrio.list_layers(
        GN_KML
    )

    print("\nAvailable layers:")

    for layer in layers:
        print(layer)

    # --------------------------------------------------------
    # TRY EXPECTED GN LAYER
    # --------------------------------------------------------

    target_layer = (
        "GN Boundary Colombo District (#2)"
    )

    available_names = [
        row[0]
        for row in layers
    ]

    if target_layer not in available_names:

        print(
            "\nExpected GN layer was not found "
            "with the exact name."
        )

        print(
            "We will inspect the available names "
            "before continuing."
        )

        return

    # --------------------------------------------------------
    # LOAD GN POLYGONS
    # --------------------------------------------------------

    gn = gpd.read_file(
        GN_KML,
        layer=target_layer,
    )

    print("\nGN layer loaded successfully.")

    print(
        f"Feature count: {len(gn)}"
    )

    print(
        f"CRS: {gn.crs}"
    )

    print(
        f"Geometry types:"
    )

    print(
        gn.geometry
        .geom_type
        .value_counts()
        .to_string()
    )

    print("\nAvailable columns:")

    for column in gn.columns:
        print(
            f" - {column}"
        )

    print("\nFirst five records:")

    non_geometry_columns = [
        column
        for column in gn.columns
        if column != "geometry"
    ]

    print(
        gn[
            non_geometry_columns
        ]
        .head()
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()