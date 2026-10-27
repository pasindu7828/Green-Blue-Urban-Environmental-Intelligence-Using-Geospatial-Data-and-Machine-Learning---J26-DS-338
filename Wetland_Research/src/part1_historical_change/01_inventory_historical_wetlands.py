from pathlib import Path
import re
import xml.etree.ElementTree as ET
from datetime import date

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "historical_wetlands"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "part1_historical_change"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# KML SETTINGS
# ============================================================

KML_NS = {
    "k": "http://www.opengis.net/kml/2.2"
}

DATE_RE = re.compile(
    r"(?P<year>\d{4})_(?P<month>\d{1,2})_(?P<day>\d{1,2})"
)

ID_RE = re.compile(
    r"W\d{3}",
    re.IGNORECASE
)


# ============================================================
# SEASON CLASSIFICATION
# ============================================================

def season_from_month(month: int) -> str:
    """
    Assign the observation to a broad Sri Lankan
    climatological season.

    Rainfall-based comparability will be added later.
    """

    if month in (12, 1, 2):
        return "Northeast monsoon"

    if month in (3, 4):
        return "First inter-monsoon"

    if month in (5, 6, 7, 8, 9):
        return "Southwest monsoon"

    return "Second inter-monsoon"


# ============================================================
# DATE PARSING
# ============================================================

def parse_date_from_name(name: str):
    """
    Extract observation date and optional patch label
    from a placemark name.

    Examples:
        W001_2025_12_25
        W001_2010_11_16_A
        W024-2025_12_25_B
    """

    match = DATE_RE.search(name)

    if not match:
        return None, None

    try:
        observation_date = date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )

    except ValueError:
        return None, None

    remainder = name[match.end():].strip("_- ")

    patch_label = remainder if remainder else None

    return observation_date, patch_label


# ============================================================
# GEOMETRY INFORMATION
# ============================================================

def geometry_info(placemark):
    polygon_count = len(
        placemark.findall(".//k:Polygon", KML_NS)
    )

    point_count = len(
        placemark.findall(".//k:Point", KML_NS)
    )

    line_count = len(
        placemark.findall(".//k:LineString", KML_NS)
    )

    geometry_types = []

    if polygon_count:
        geometry_types.append("Polygon")

    if point_count:
        geometry_types.append("Point")

    if line_count:
        geometry_types.append("LineString")

    geometry_type = (
        "+".join(geometry_types)
        if geometry_types
        else "Unknown"
    )

    return (
        polygon_count,
        point_count,
        line_count,
        geometry_type,
    )


# ============================================================
# MAIN INVENTORY PROCESS
# ============================================================

def main():

    kml_files = sorted(
        RAW_DIR.glob("W*_Historical_Wetland.kml")
    )

    if not kml_files:
        raise FileNotFoundError(
            f"No historical wetland KML files found in:\n{RAW_DIR}"
        )

    rows = []

    for kml_path in kml_files:

        expected_match = ID_RE.search(kml_path.stem)

        expected_id = (
            expected_match.group(0).upper()
            if expected_match
            else None
        )

        try:
            root = ET.parse(kml_path).getroot()

        except ET.ParseError as error:

            rows.append(
                {
                    "source_file": kml_path.name,
                    "expected_wetland_id": expected_id,
                    "placemark_name": None,
                    "parse_status":
                        f"KML_XML_ERROR: {error}",
                }
            )

            continue

        placemarks = root.findall(
            ".//k:Placemark",
            KML_NS
        )

        for placemark in placemarks:

            placemark_name = placemark.findtext(
                "k:name",
                default="",
                namespaces=KML_NS,
            ).strip()

            id_match = ID_RE.search(
                placemark_name
            )

            placemark_id = (
                id_match.group(0).upper()
                if id_match
                else None
            )

            observation_date, patch_label = (
                parse_date_from_name(
                    placemark_name
                )
            )

            (
                polygon_count,
                point_count,
                line_count,
                geometry_type,
            ) = geometry_info(placemark)

            issues = []

            if not placemark_name:
                issues.append(
                    "MISSING_PLACEMARK_NAME"
                )

            if placemark_id is None:
                issues.append(
                    "MISSING_WETLAND_ID_IN_NAME"
                )

            elif (
                expected_id
                and placemark_id != expected_id
            ):
                issues.append(
                    "WETLAND_ID_MISMATCH"
                )

            if observation_date is None:
                issues.append(
                    "DATE_PARSE_FAILED"
                )

            if polygon_count == 0:
                issues.append(
                    "NO_POLYGON_GEOMETRY"
                )

            rows.append(
                {
                    "source_file":
                        kml_path.name,

                    "expected_wetland_id":
                        expected_id,

                    "placemark_wetland_id":
                        placemark_id,

                    "placemark_name":
                        placemark_name,

                    "observation_date":
                        (
                            observation_date.isoformat()
                            if observation_date
                            else None
                        ),

                    "year":
                        (
                            observation_date.year
                            if observation_date
                            else None
                        ),

                    "month":
                        (
                            observation_date.month
                            if observation_date
                            else None
                        ),

                    "day":
                        (
                            observation_date.day
                            if observation_date
                            else None
                        ),

                    "season":
                        (
                            season_from_month(
                                observation_date.month
                            )
                            if observation_date
                            else None
                        ),

                    "patch_label":
                        patch_label,

                    "geometry_type":
                        geometry_type,

                    "polygon_count":
                        polygon_count,

                    "point_count":
                        point_count,

                    "line_count":
                        line_count,

                    "parse_status":
                        (
                            "OK"
                            if not issues
                            else ";".join(issues)
                        ),
                }
            )

    # ========================================================
    # CREATE INVENTORY TABLE
    # ========================================================

    inventory = pd.DataFrame(rows)

    inventory_path = (
        OUTPUT_DIR
        / "historical_wetland_inventory.csv"
    )

    inventory.to_csv(
        inventory_path,
        index=False
    )


    # ========================================================
    # CREATE WETLAND SUMMARY
    # ========================================================

    valid = inventory[
        inventory["observation_date"].notna()
    ].copy()

    if not valid.empty:

        per_date = (
            valid
            .groupby(
                [
                    "expected_wetland_id",
                    "observation_date",
                ],
                dropna=False,
            )
            .size()
            .rename("patches_on_date")
            .reset_index()
        )

        multi_patch_lookup = (
            per_date
            .assign(
                is_multi=
                    per_date["patches_on_date"] > 1
            )
            .groupby(
                "expected_wetland_id"
            )["is_multi"]
            .sum()
        )

        summary = (
            inventory
            .groupby(
                "expected_wetland_id",
                dropna=False,
            )
            .agg(
                placemark_count=(
                    "placemark_name",
                    "count",
                ),

                valid_date_rows=(
                    "observation_date",
                    lambda values:
                        values.notna().sum(),
                ),

                unique_observation_dates=(
                    "observation_date",
                    lambda values:
                        values.dropna().nunique(),
                ),

                earliest_date=(
                    "observation_date",
                    lambda values:
                        (
                            values.dropna().min()
                            if values.notna().any()
                            else None
                        ),
                ),

                latest_date=(
                    "observation_date",
                    lambda values:
                        (
                            values.dropna().max()
                            if values.notna().any()
                            else None
                        ),
                ),

                issue_rows=(
                    "parse_status",
                    lambda values:
                        (values != "OK").sum(),
                ),
            )
            .reset_index()
        )

        summary["multi_patch_dates"] = (
            summary["expected_wetland_id"]
            .map(multi_patch_lookup)
            .fillna(0)
            .astype(int)
        )

    else:
        summary = pd.DataFrame()

    summary_path = (
        OUTPUT_DIR
        / "historical_wetland_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )


    # ========================================================
    # SAVE ISSUES SEPARATELY
    # ========================================================

    issues_df = inventory[
        inventory["parse_status"] != "OK"
    ].copy()

    issues_path = (
        OUTPUT_DIR
        / "historical_wetland_issues.csv"
    )

    issues_df.to_csv(
        issues_path,
        index=False
    )


    # ========================================================
    # TERMINAL REPORT
    # ========================================================

    print("\n========================================")
    print("HISTORICAL WETLAND DATA INVENTORY")
    print("========================================")

    print(
        f"KML files found: {len(kml_files)}"
    )

    print(
        f"Placemark rows: {len(inventory)}"
    )

    print(
        "Unique wetland IDs: "
        f"{inventory['expected_wetland_id'].nunique()}"
    )

    print(
        "Unique observation dates: "
        f"{valid['observation_date'].nunique()}"
    )

    print(
        f"Rows with issues: {len(issues_df)}"
    )

    print("\nOutputs:")

    print(
        f"Inventory: {inventory_path}"
    )

    print(
        f"Summary:   {summary_path}"
    )

    print(
        f"Issues:    {issues_path}"
    )

    if not issues_df.empty:

        print("\nIssues requiring review:")

        print(
            issues_df[
                [
                    "source_file",
                    "placemark_name",
                    "parse_status",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()