"""
STEP 05E HELPER - EXPORT EXACT LANDSAT PIXEL FOOTPRINTS TO KML

Purpose:
- Read the editable 40-point high-confidence quality-audit layer.
- Read the processed 2021 Landsat raster.
- Find the exact 30 m Landsat pixel associated with each audit point.
- Create the exact pixel polygon.
- Export the 40 pixel footprints to KML for Google Earth Pro.

IMPORTANT:
- This script does not modify the audit GeoPackage.
- This script does not perform validation.
- The KML is only a visual/manual-review navigation aid.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import geopandas as gpd
import pandas as pd
import rasterio

from rasterio.windows import Window
from shapely.geometry import box


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lulc"
    / "reference_samples"
)

LANDSAT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landsat"
    / "annual_composites_kmc"
)


# =============================================================================
# 2. INPUTS
# =============================================================================

INPUT_GPKG = (
    REFERENCE_DIR
    / "lulc_high_confidence_quality_audit_review_40_2021.gpkg"
)

LANDSAT_2021 = (
    LANDSAT_DIR
    / "Kaduwela_Landsat_2021.tif"
)


# =============================================================================
# 3. OUTPUT
# =============================================================================

OUTPUT_KML = (
    REFERENCE_DIR
    / "lulc_high_confidence_quality_audit_pixel_footprints_40_2021.kml"
)


# =============================================================================
# 4. CONSTANTS
# =============================================================================

ANALYSIS_CRS = "EPSG:32644"

OUTPUT_CRS = "EPSG:4326"

EXPECTED_TOTAL = 40


CLASS_DEFINITIONS = {
    1: "Green / Vegetation",
    2: "Built-up",
    3: "Open / Bare",
    4: "Water",
}


# KML colours use AABBGGRR format.
CLASS_LINE_COLOURS = {
    1: "ff00aa00",   # green
    2: "ff0000ff",   # red
    3: "ff00a5ff",   # orange
    4: "ffff9900",   # blue-ish
}


# =============================================================================
# 5. HELPERS
# =============================================================================

def relative_path(path: Path) -> str:
    """
    Return a project-relative path when possible.
    """

    try:
        return path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:
        return str(path)


def safe_value(value):
    """
    Convert missing values to an empty string.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass

    return str(value)


# =============================================================================
# 6. START
# =============================================================================

print("=" * 100)
print("STEP 05E HELPER - EXPORT EXACT LANDSAT PIXEL FOOTPRINTS")
print("=" * 100)


for required_path in [
    INPUT_GPKG,
    LANDSAT_2021,
]:

    if not required_path.exists():

        raise FileNotFoundError(
            f"Required input not found:\n"
            f"{required_path}"
        )


# =============================================================================
# 7. LOAD AUDIT DATA
# =============================================================================

audit_gdf = gpd.read_file(
    INPUT_GPKG
)


print(
    f"\nAudit layer    : "
    f"{relative_path(INPUT_GPKG)}"
)

print(
    f"Audit features : "
    f"{len(audit_gdf)}"
)

print(
    f"Audit CRS      : "
    f"{audit_gdf.crs}"
)


if str(
    audit_gdf.crs
) != ANALYSIS_CRS:

    raise ValueError(
        f"Unexpected audit CRS: "
        f"{audit_gdf.crs}. "
        f"Expected {ANALYSIS_CRS}."
    )


if len(
    audit_gdf
) != EXPECTED_TOTAL:

    raise ValueError(
        f"Expected {EXPECTED_TOTAL} audit points, "
        f"but found {len(audit_gdf)}."
    )


required_columns = {
    "sample_id",
    "class_id",
    "class_name",
}


missing_columns = (
    required_columns
    -
    set(
        audit_gdf.columns
    )
)


if missing_columns:

    raise ValueError(
        f"Missing required columns: "
        f"{sorted(missing_columns)}"
    )


if audit_gdf[
    "sample_id"
].duplicated().any():

    raise ValueError(
        "Duplicate sample IDs found."
    )


if not audit_gdf.geometry.geom_type.eq(
    "Point"
).all():

    raise ValueError(
        "All audit geometries must be points."
    )


audit_gdf[
    "class_id"
] = pd.to_numeric(
    audit_gdf[
        "class_id"
    ],
    errors="raise",
).astype(int)


# =============================================================================
# 8. READ LANDSAT GRID INFORMATION
# =============================================================================

with rasterio.open(
    LANDSAT_2021
) as src:

    landsat_crs = src.crs
    landsat_transform = src.transform
    landsat_width = src.width
    landsat_height = src.height


print(
    f"\nLandsat raster : "
    f"{relative_path(LANDSAT_2021)}"
)

print(
    f"Landsat CRS    : "
    f"{landsat_crs}"
)

print(
    f"Raster size    : "
    f"{landsat_width} x {landsat_height}"
)

print(
    f"Pixel size     : "
    f"{abs(landsat_transform.a):.3f} m x "
    f"{abs(landsat_transform.e):.3f} m"
)


if str(
    landsat_crs
) != ANALYSIS_CRS:

    raise ValueError(
        f"Unexpected Landsat CRS: "
        f"{landsat_crs}"
    )


# =============================================================================
# 9. CREATE EXACT LANDSAT PIXEL POLYGONS
# =============================================================================

print("\n" + "=" * 100)
print("CREATING EXACT LANDSAT PIXEL FOOTPRINTS")
print("=" * 100)


pixel_records = []


for _, row in audit_gdf.iterrows():

    sample_id = str(
        row[
            "sample_id"
        ]
    )

    class_id = int(
        row[
            "class_id"
        ]
    )

    class_name = str(
        row[
            "class_name"
        ]
    )


    x = float(
        row.geometry.x
    )

    y = float(
        row.geometry.y
    )


    # -------------------------------------------------------------------------
    # Prefer existing Landsat row/column fields when available.
    # Otherwise calculate them directly from the audit point.
    # -------------------------------------------------------------------------

    if (
        "landsat_row"
        in audit_gdf.columns
        and
        "landsat_col"
        in audit_gdf.columns
        and
        pd.notna(
            row[
                "landsat_row"
            ]
        )
        and
        pd.notna(
            row[
                "landsat_col"
            ]
        )
    ):

        raster_row = int(
            row[
                "landsat_row"
            ]
        )

        raster_col = int(
            row[
                "landsat_col"
            ]
        )

    else:

        raster_row, raster_col = (
            rasterio.transform.rowcol(
                landsat_transform,
                x,
                y,
            )
        )

        raster_row = int(
            raster_row
        )

        raster_col = int(
            raster_col
        )


    if not (
        0
        <= raster_row
        < landsat_height
        and
        0
        <= raster_col
        < landsat_width
    ):

        raise ValueError(
            f"{sample_id} falls outside "
            f"the Landsat raster."
        )


    # -------------------------------------------------------------------------
    # Exact raster pixel bounds
    # -------------------------------------------------------------------------

    pixel_window = Window(
        raster_col,
        raster_row,
        1,
        1,
    )


    pixel_bounds = (
        rasterio.windows.bounds(
            pixel_window,
            landsat_transform,
        )
    )


    left, bottom, right, top = (
        pixel_bounds
    )


    pixel_width_m = (
        right
        -
        left
    )

    pixel_height_m = (
        top
        -
        bottom
    )


    pixel_polygon = box(
        left,
        bottom,
        right,
        top,
    )


    # -------------------------------------------------------------------------
    # Confirm the audit point falls inside its calculated pixel.
    # -------------------------------------------------------------------------

    point_inside_pixel = bool(
        pixel_polygon.covers(
            row.geometry
        )
    )


    if not point_inside_pixel:

        raise ValueError(
            f"Point {sample_id} is not inside "
            f"its calculated Landsat pixel."
        )


    pixel_records.append(
        {
            "sample_id": sample_id,
            "class_id": class_id,
            "class_name": class_name,
            "landsat_row": raster_row,
            "landsat_col": raster_col,
            "pixel_width_m": pixel_width_m,
            "pixel_height_m": pixel_height_m,
            "point_inside_pixel": (
                point_inside_pixel
            ),
            "screen_confidence": (
                safe_value(
                    row.get(
                        "screen_confidence",
                        ""
                    )
                )
            ),
            "wc_pixel_purity": (
                safe_value(
                    row.get(
                        "wc_pixel_purity",
                        ""
                    )
                )
            ),
            "review_status": (
                safe_value(
                    row.get(
                        "review_status",
                        ""
                    )
                )
            ),
            "geometry": pixel_polygon,
        }
    )


    print(
        f"{sample_id:<8} "
        f"class={class_id}  "
        f"row={raster_row:<4} "
        f"col={raster_col:<4} "
        f"pixel={pixel_width_m:.1f} x "
        f"{pixel_height_m:.1f} m"
    )


# =============================================================================
# 10. CREATE POLYGON GEODATAFRAME
# =============================================================================

pixel_gdf = gpd.GeoDataFrame(
    pixel_records,
    geometry="geometry",
    crs=ANALYSIS_CRS,
)


if pixel_gdf[
    "sample_id"
].duplicated().any():

    raise ValueError(
        "Duplicate sample IDs exist "
        "in pixel footprint output."
    )


if not pixel_gdf[
    "point_inside_pixel"
].all():

    raise ValueError(
        "At least one audit point is not "
        "inside its Landsat pixel."
    )


# =============================================================================
# 11. CONVERT TO WGS84 FOR GOOGLE EARTH
# =============================================================================

pixel_wgs84 = pixel_gdf.to_crs(
    OUTPUT_CRS
)


# =============================================================================
# 12. CREATE KML
# =============================================================================

KML_NAMESPACE = (
    "http://www.opengis.net/kml/2.2"
)

ET.register_namespace(
    "",
    KML_NAMESPACE
)


def kml_tag(tag_name):
    return (
        f"{{{KML_NAMESPACE}}}"
        f"{tag_name}"
    )


kml = ET.Element(
    kml_tag(
        "kml"
    )
)


document = ET.SubElement(
    kml,
    kml_tag(
        "Document"
    )
)


document_name = ET.SubElement(
    document,
    kml_tag(
        "name"
    )
)

document_name.text = (
    "2021 LULC Audit - Exact Landsat 30 m Pixel Footprints"
)


document_description = ET.SubElement(
    document,
    kml_tag(
        "description"
    )
)

document_description.text = (
    "Exact processed Landsat 2021 pixel footprints for the "
    "40-point high-confidence LULC reference-quality audit."
)


# =============================================================================
# 13. CREATE CLASS STYLES
# =============================================================================

for class_id in CLASS_DEFINITIONS:

    style = ET.SubElement(
        document,
        kml_tag(
            "Style"
        ),
        id=f"class_{class_id}_pixel",
    )


    line_style = ET.SubElement(
        style,
        kml_tag(
            "LineStyle"
        )
    )


    line_colour = ET.SubElement(
        line_style,
        kml_tag(
            "color"
        )
    )

    line_colour.text = (
        CLASS_LINE_COLOURS[
            class_id
        ]
    )


    line_width = ET.SubElement(
        line_style,
        kml_tag(
            "width"
        )
    )

    line_width.text = "3"


    poly_style = ET.SubElement(
        style,
        kml_tag(
            "PolyStyle"
        )
    )


    fill = ET.SubElement(
        poly_style,
        kml_tag(
            "fill"
        )
    )

    fill.text = "0"


    outline = ET.SubElement(
        poly_style,
        kml_tag(
            "outline"
        )
    )

    outline.text = "1"


# =============================================================================
# 14. CREATE CLASS FOLDERS + PIXEL POLYGONS
# =============================================================================

for class_id, class_name in (
    CLASS_DEFINITIONS.items()
):

    class_pixels = (
        pixel_wgs84[
            pixel_wgs84[
                "class_id"
            ]
            == class_id
        ]
        .copy()
        .sort_values(
            "sample_id"
        )
    )


    folder = ET.SubElement(
        document,
        kml_tag(
            "Folder"
        )
    )


    folder_name = ET.SubElement(
        folder,
        kml_tag(
            "name"
        )
    )

    folder_name.text = (
        f"{class_id} - "
        f"{class_name} Pixel Footprints"
    )


    for _, row in class_pixels.iterrows():

        sample_id = str(
            row[
                "sample_id"
            ]
        )


        placemark = ET.SubElement(
            folder,
            kml_tag(
                "Placemark"
            )
        )


        placemark_name = ET.SubElement(
            placemark,
            kml_tag(
                "name"
            )
        )

        placemark_name.text = (
            f"{sample_id} - 30m pixel"
        )


        style_url = ET.SubElement(
            placemark,
            kml_tag(
                "styleUrl"
            )
        )

        style_url.text = (
            f"#class_{class_id}_pixel"
        )


        description = ET.SubElement(
            placemark,
            kml_tag(
                "description"
            )
        )

        description.text = (
            f"Sample ID: {sample_id}\n"
            f"Reference class: {row['class_name']}\n"
            f"Landsat row: {row['landsat_row']}\n"
            f"Landsat column: {row['landsat_col']}\n"
            f"Pixel size: "
            f"{float(row['pixel_width_m']):.1f} m x "
            f"{float(row['pixel_height_m']):.1f} m\n"
            f"Screen confidence: {row['screen_confidence']}\n"
            f"WorldCover pixel purity: {row['wc_pixel_purity']}\n"
            f"Review status: {row['review_status']}"
        )


        polygon_element = ET.SubElement(
            placemark,
            kml_tag(
                "Polygon"
            )
        )


        altitude_mode = ET.SubElement(
            polygon_element,
            kml_tag(
                "altitudeMode"
            )
        )

        altitude_mode.text = (
            "clampToGround"
        )


        outer_boundary = ET.SubElement(
            polygon_element,
            kml_tag(
                "outerBoundaryIs"
            )
        )


        linear_ring = ET.SubElement(
            outer_boundary,
            kml_tag(
                "LinearRing"
            )
        )


        coordinates_element = ET.SubElement(
            linear_ring,
            kml_tag(
                "coordinates"
            )
        )


        polygon_coordinates = list(
            row.geometry.exterior.coords
        )


        coordinate_text = " ".join(
            [
                f"{longitude:.10f},"
                f"{latitude:.10f},0"

                for longitude, latitude
                in polygon_coordinates
            ]
        )


        coordinates_element.text = (
            coordinate_text
        )


# =============================================================================
# 15. SAVE KML
# =============================================================================

tree = ET.ElementTree(
    kml
)


ET.indent(
    tree,
    space="  "
)


tree.write(
    OUTPUT_KML,
    encoding="UTF-8",
    xml_declaration=True,
)


# =============================================================================
# 16. FINAL VALIDATION
# =============================================================================

output_exists = (
    OUTPUT_KML.exists()
)


correct_total = (
    len(
        pixel_gdf
    )
    == EXPECTED_TOTAL
)


all_points_inside = bool(
    pixel_gdf[
        "point_inside_pixel"
    ].all()
)


approximately_30m = bool(
    (
        pixel_gdf[
            "pixel_width_m"
        ]
        .between(
            29.9,
            30.1
        )
        .all()
    )
    and
    (
        pixel_gdf[
            "pixel_height_m"
        ]
        .between(
            29.9,
            30.1
        )
        .all()
    )
)


overall_pass = all(
    [
        output_exists,
        correct_total,
        all_points_inside,
        approximately_30m,
    ]
)


# =============================================================================
# 17. TERMINAL SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("PIXEL FOOTPRINT KML SUMMARY")
print("=" * 100)


print(
    f"Audit points              : "
    f"{len(audit_gdf)}"
)

print(
    f"Pixel footprints created  : "
    f"{len(pixel_gdf)}"
)

print(
    f"All points inside pixels  : "
    f"{all_points_inside}"
)

print(
    f"All pixels approximately "
    f"30 m x 30 m              : "
    f"{approximately_30m}"
)

print(
    f"Output exists             : "
    f"{output_exists}"
)

print(
    f"\nOutput:"
)

print(
    f" - {relative_path(OUTPUT_KML)}"
)

print(
    f"\nOverall status            : "
    f"{'PASS' if overall_pass else 'REVIEW REQUIRED'}"
)


print("\nIMPORTANT:")

print(
    "1. These polygons represent the exact processed Landsat "
    "30 m pixels used for the audit candidates."
)

print(
    "2. Use these pixel boxes together with the named point KML "
    "in Google Earth Pro."
)

print(
    "3. Judge the full pixel footprint, not only the centre pin."
)

print(
    "4. Prefer imagery close to the 2021 reference year."
)

print(
    "5. This KML is a navigation aid only; record audit decisions "
    "in the editable GeoPackage."
)

print("=" * 100)