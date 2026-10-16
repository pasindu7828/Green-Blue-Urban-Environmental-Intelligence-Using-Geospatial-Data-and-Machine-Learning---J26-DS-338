"""
STEP 05E HELPER - EXPORT NAMED QUALITY-AUDIT KML

Purpose:
- Read the editable 40-point HIGH-confidence LULC quality-audit GeoPackage.
- Convert coordinates to EPSG:4326 for Google Earth Pro.
- Export a KML where each placemark name is the sample_id.
- Group placemarks by LULC class.

Example placemark names:
C1_002
C2_016
C3_002
C4_008

IMPORTANT:
- This script does not modify the GeoPackage.
- This script does not perform validation.
- The generated KML is only a navigation aid for Google Earth Pro.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import geopandas as gpd
import pandas as pd


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


# =============================================================================
# 2. INPUT / OUTPUT
# =============================================================================

INPUT_GPKG = (
    REFERENCE_DIR
    / "lulc_high_confidence_quality_audit_review_40_2021.gpkg"
)

OUTPUT_KML = (
    REFERENCE_DIR
    / "lulc_high_confidence_quality_audit_named_40_2021.kml"
)


# =============================================================================
# 3. CONSTANTS
# =============================================================================

EXPECTED_CRS = "EPSG:32644"

OUTPUT_CRS = "EPSG:4326"

EXPECTED_TOTAL = 40


CLASS_DEFINITIONS = {
    1: "Green / Vegetation",
    2: "Built-up",
    3: "Open / Bare",
    4: "Water",
}


# =============================================================================
# 4. HELPERS
# =============================================================================

def relative_path(path: Path) -> str:
    """
    Return project-relative path when possible.
    """

    try:
        return path.relative_to(PROJECT_ROOT).as_posix()

    except ValueError:
        return str(path)


def safe_value(value):
    """
    Convert missing values into a readable empty string.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass

    return str(value)


def add_extended_data(parent, row):
    """
    Add useful audit attributes to one KML placemark.
    """

    fields_to_include = [
        "sample_id",
        "class_id",
        "class_name",
        "reference_year",
        "screen_confidence",
        "wc_pixel_purity",
        "max_spectral_robust_z",
        "spectral_margin",
        "landsat_pixel_id",
        "review_status",
        "manual_class_id",
        "manual_class_name",
        "imagery_source",
        "imagery_date",
        "temporal_match",
        "review_notes",
    ]

    extended_data = ET.SubElement(
        parent,
        "ExtendedData"
    )

    for field_name in fields_to_include:

        if field_name not in row.index:
            continue

        data_element = ET.SubElement(
            extended_data,
            "Data",
            name=field_name,
        )

        value_element = ET.SubElement(
            data_element,
            "value"
        )

        value_element.text = safe_value(
            row[field_name]
        )


# =============================================================================
# 5. START
# =============================================================================

print("=" * 100)
print("STEP 05E HELPER - EXPORT NAMED QUALITY-AUDIT KML")
print("=" * 100)


if not INPUT_GPKG.exists():

    raise FileNotFoundError(
        f"Editable audit GeoPackage not found:\n"
        f"{INPUT_GPKG}"
    )


audit_gdf = gpd.read_file(
    INPUT_GPKG
)


print(
    f"\nInput file     : "
    f"{relative_path(INPUT_GPKG)}"
)

print(
    f"Input features : "
    f"{len(audit_gdf)}"
)

print(
    f"Input CRS      : "
    f"{audit_gdf.crs}"
)


# =============================================================================
# 6. VALIDATE INPUT
# =============================================================================

required_columns = {
    "sample_id",
    "class_id",
    "class_name",
}


missing_columns = (
    required_columns
    -
    set(audit_gdf.columns)
)


if missing_columns:

    raise ValueError(
        f"Missing required columns: "
        f"{sorted(missing_columns)}"
    )


if str(audit_gdf.crs) != EXPECTED_CRS:

    raise ValueError(
        f"Unexpected CRS: {audit_gdf.crs}. "
        f"Expected {EXPECTED_CRS}."
    )


if len(audit_gdf) != EXPECTED_TOTAL:

    raise ValueError(
        f"Expected {EXPECTED_TOTAL} audit points, "
        f"but found {len(audit_gdf)}."
    )


if audit_gdf["sample_id"].duplicated().any():

    duplicates = (
        audit_gdf.loc[
            audit_gdf["sample_id"].duplicated(
                keep=False
            ),
            "sample_id",
        ]
        .astype(str)
        .tolist()
    )

    raise ValueError(
        f"Duplicate sample IDs found: "
        f"{duplicates}"
    )


if not audit_gdf.geometry.geom_type.eq(
    "Point"
).all():

    raise ValueError(
        "All audit geometries must be Point features."
    )


audit_gdf["class_id"] = pd.to_numeric(
    audit_gdf["class_id"],
    errors="raise",
).astype(int)


# =============================================================================
# 7. CONVERT TO GOOGLE EARTH COORDINATES
# =============================================================================

audit_wgs84 = audit_gdf.to_crs(
    OUTPUT_CRS
)


print(
    f"Output CRS     : "
    f"{OUTPUT_CRS}"
)


# =============================================================================
# 8. CREATE KML DOCUMENT
# =============================================================================

KML_NAMESPACE = "http://www.opengis.net/kml/2.2"

ET.register_namespace(
    "",
    KML_NAMESPACE
)


kml = ET.Element(
    f"{{{KML_NAMESPACE}}}kml"
)


document = ET.SubElement(
    kml,
    "Document"
)


document_name = ET.SubElement(
    document,
    "name"
)

document_name.text = (
    "2021 High-Confidence LULC Quality Audit - 40 Points"
)


document_description = ET.SubElement(
    document,
    "description"
)

document_description.text = (
    "Manual reference-quality audit points for the 2021 "
    "high-confidence LULC candidate pool. "
    "This is not the final model accuracy-validation dataset."
)


# =============================================================================
# 9. CREATE CLASS FOLDERS AND PLACEMARKS
# =============================================================================

print("\n" + "=" * 100)
print("EXPORTING PLACEMARKS")
print("=" * 100)


exported_count = 0


for class_id, class_name in CLASS_DEFINITIONS.items():

    class_gdf = (
        audit_wgs84[
            audit_wgs84["class_id"]
            == class_id
        ]
        .copy()
        .sort_values(
            "sample_id"
        )
    )


    folder = ET.SubElement(
        document,
        "Folder"
    )


    folder_name = ET.SubElement(
        folder,
        "name"
    )

    folder_name.text = (
        f"{class_id} - {class_name}"
    )


    print(
        f"\n{class_id} - "
        f"{class_name}: "
        f"{len(class_gdf)} points"
    )


    for _, row in class_gdf.iterrows():

        sample_id = str(
            row["sample_id"]
        )


        geometry = row.geometry


        longitude = float(
            geometry.x
        )

        latitude = float(
            geometry.y
        )


        placemark = ET.SubElement(
            folder,
            "Placemark"
        )


        placemark_name = ET.SubElement(
            placemark,
            "name"
        )

        placemark_name.text = sample_id


        placemark_description = ET.SubElement(
            placemark,
            "description"
        )

        placemark_description.text = (
            f"Sample ID: {sample_id}\n"
            f"Reference class: {safe_value(row['class_name'])}\n"
            f"Class ID: {safe_value(row['class_id'])}\n"
            f"Reference year: {safe_value(row.get('reference_year', '2021'))}\n"
            f"Screen confidence: {safe_value(row.get('screen_confidence', ''))}\n"
            f"Review status: {safe_value(row.get('review_status', 'PENDING'))}"
        )


        add_extended_data(
            placemark,
            row,
        )


        point = ET.SubElement(
            placemark,
            "Point"
        )


        altitude_mode = ET.SubElement(
            point,
            "altitudeMode"
        )

        altitude_mode.text = (
            "clampToGround"
        )


        coordinates = ET.SubElement(
            point,
            "coordinates"
        )

        coordinates.text = (
            f"{longitude:.10f},"
            f"{latitude:.10f},"
            f"0"
        )


        exported_count += 1

        print(
            f"  {sample_id}"
        )


# =============================================================================
# 10. WRITE KML
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
# 11. FINAL CHECK
# =============================================================================

output_exists = OUTPUT_KML.exists()

correct_feature_count = (
    exported_count
    == EXPECTED_TOTAL
)


overall_pass = (
    output_exists
    and
    correct_feature_count
)


# =============================================================================
# 12. TERMINAL SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("NAMED KML EXPORT SUMMARY")
print("=" * 100)


print(
    f"Input audit points       : "
    f"{len(audit_gdf)}"
)

print(
    f"Exported placemarks      : "
    f"{exported_count}"
)

print(
    f"Expected placemarks      : "
    f"{EXPECTED_TOTAL}"
)

print(
    f"Output file exists       : "
    f"{output_exists}"
)

print(
    f"Sample IDs used as names : "
    f"True"
)

print(
    f"Class folders created    : "
    f"{len(CLASS_DEFINITIONS)}"
)

print(
    f"\nOutput:"
)

print(
    f" - {relative_path(OUTPUT_KML)}"
)

print(
    f"\nOverall status           : "
    f"{'PASS' if overall_pass else 'REVIEW REQUIRED'}"
)


print("\nIMPORTANT:")

print(
    "1. This KML is only a Google Earth Pro navigation aid."
)

print(
    "2. Each Google Earth placemark name is the sample_id."
)

print(
    "3. The original editable GeoPackage remains the authoritative "
    "manual-review layer."
)

print(
    "4. Record manual review results back in the GeoPackage, "
    "not in this KML."
)

print("=" * 100)