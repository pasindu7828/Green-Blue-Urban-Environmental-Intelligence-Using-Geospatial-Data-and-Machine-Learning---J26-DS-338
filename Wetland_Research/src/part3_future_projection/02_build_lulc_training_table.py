
from pathlib import Path
from collections import Counter
import json
import runpy

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from shapely.geometry import Point
from rasterio.transform import rowcol
from rasterio.warp import transform as transform_coords


ROOT = Path(__file__).resolve().parents[2]

SCRIPT_01 = Path(__file__).with_name(
    "01_validate_part3_inputs.py"
)

previous = runpy.run_path(str(SCRIPT_01))

LANDSAT = previous["LANDSAT"]
REFERENCE = previous["REFERENCE"]
parse_reference_kmz = previous["parse_reference_kmz"]

OUTPUT_DIR = (
    ROOT / "data" / "interim" / "part3_future_projection"
)

YEARS = (2015, 2020, 2025)

CLASS_NAMES = {
    1: "Wetland_Marsh",
    2: "Open_Water",
    3: "Built_Up",
    4: "Paddy_Agriculture",
    5: "Other_Vegetation",
    6: "Bare_Fill_Disturbed",
}

FEATURE_NAMES = [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "ndvi",
    "mndwi",
    "ndbi",
]

# These sample IDs were independently checked in Script 01.
KNOWN_NODATA_IDS = {
    2015: {
        "2015_BAR_045",
        "2015_BAR_046",
        "2015_BAR_047",
    },
    2020: {
        "2020_BAR_046",
        "2020_BAR_047",
    },
    2025: {
        "2025_BAR_046",
        "2025_BAR_047",
    },
}


def get_class_label(folder):
    final_folder = folder.split("/")[-1].strip()

    for class_id, class_name in CLASS_NAMES.items():
        expected = f"{class_id:02d}_{class_name}"

        if final_folder == expected:
            return class_id, class_name

    raise ValueError(
        f"Unexpected LULC reference class: {folder}"
    )


def main():
    print("=" * 60)
    print("PART 3 - FUTURE WETLAND BOUNDARY PROJECTION")
    print("SCRIPT 02 - BUILD LULC TRAINING TABLES")
    print("=" * 60)

    boundary_path = (
        ROOT / "data" / "raw" / "boundaries"
        / "Kaduwela TO Area.kml"
    )

    gn = gpd.read_file(
        boundary_path,
        layer="GN Boundary Colombo District (#2)",
        engine="pyogrio"
    )

    if len(gn) != 56 or not gn.is_valid.all():
        raise ValueError(
            "GN boundary validation failed."
        )

    gn = gn.to_crs("EPSG:4326")
    gn_union = gn.geometry.union_all()

    print("\nGN divisions:", len(gn))

    all_rows = []
    yearly_summary = {}

    for year in YEARS:
        print(f"\n========== YEAR {year} ==========")

        points, non_points = parse_reference_kmz(
            REFERENCE[year]
        )

        if len(points) != 375 or non_points != 0:
            raise ValueError(
                f"Unexpected reference count for {year}"
            )

        year_rows = []

        with rasterio.open(LANDSAT[year]) as src:
            for point in points:
                class_id, class_name = get_class_label(
                    point["folder"]
                )

                lon = point["longitude"]
                lat = point["latitude"]

                if (
                    lon is None or lat is None
                    or not np.isfinite(lon)
                    or not np.isfinite(lat)
                ):
                    raise ValueError(
                        f"Invalid coordinates: {point['name']}"
                    )

                inside_gn = bool(
                    gn_union.covers(Point(lon, lat))
                )

                xs, ys = transform_coords(
                    "EPSG:4326",
                    src.crs,
                    [lon],
                    [lat]
                )

                x, y = xs[0], ys[0]
                r, c = rowcol(src.transform, x, y)

                inside_raster = bool(
                    0 <= r < src.height
                    and 0 <= c < src.width
                )

                row = {
                    "sample_id": point["name"],
                    "year": year,
                    "class_id": class_id,
                    "class_name": class_name,
                    "longitude": lon,
                    "latitude": lat,
                    "x_utm": x,
                    "y_utm": y,
                    "raster_row": r,
                    "raster_col": c,
                    "inside_gn": inside_gn,
                    "inside_raster": inside_raster,
                    "status": "EXCLUDED",
                    "exclusion_reason": "",
                }

                for feature in FEATURE_NAMES:
                    row[feature] = np.nan

                if not inside_raster:
                    row["exclusion_reason"] = (
                        "OUTSIDE_LANDSAT_RASTER"
                    )

                else:
                    sample = next(
                        src.sample(
                            [(x, y)],
                            masked=True
                        )
                    )

                    values = np.asarray(
                        sample.data,
                        dtype=float
                    )

                    masked = np.ma.getmaskarray(sample)

                    valid = bool(
                        not np.any(masked)
                        and np.all(np.isfinite(values))
                        and len(values) == len(FEATURE_NAMES)
                    )

                    if valid:
                        row["status"] = "USABLE"
                        row["exclusion_reason"] = ""

                        for feature, value in zip(
                            FEATURE_NAMES,
                            values
                        ):
                            row[feature] = float(value)

                    else:
                        row["exclusion_reason"] = (
                            "LANDSAT_NODATA_OR_INVALID"
                        )

                year_rows.append(row)

        df_year = pd.DataFrame(year_rows)

        excluded = df_year[
            df_year["status"] == "EXCLUDED"
        ]

        actual_excluded_ids = set(
            excluded["sample_id"].tolist()
        )

        if actual_excluded_ids != KNOWN_NODATA_IDS[year]:
            raise ValueError(
                f"Unexpected excluded samples in {year}: "
                f"{sorted(actual_excluded_ids)}"
            )

        if not df_year["class_id"].nunique() == 6:
            raise ValueError(
                f"Not all six classes found in {year}"
            )

        usable = df_year[
            df_year["status"] == "USABLE"
        ]

        summary = {
            "reference_count": len(df_year),
            "usable_count": len(usable),
            "excluded_count": len(excluded),
            "inside_gn_count": int(
                df_year["inside_gn"].sum()
            ),
            "outside_gn_count": int(
                (~df_year["inside_gn"]).sum()
            ),
            "usable_outside_gn_count": int(
                (~usable["inside_gn"]).sum()
            ),
            "class_distribution_usable": {
                str(k): int(v)
                for k, v in sorted(
                    Counter(
                        usable["class_id"]
                    ).items()
                )
            },
            "excluded_sample_ids": sorted(
                actual_excluded_ids
            ),
        }

        yearly_summary[str(year)] = summary
        all_rows.extend(year_rows)

        print("Original reference records:", len(df_year))
        print("Usable reference records:", len(usable))
        print("Excluded records:", len(excluded))
        print(
            "Usable outside GN coverage:",
            summary["usable_outside_gn_count"]
        )

        print("Usable class distribution:")
        for class_id, count in (
            summary["class_distribution_usable"].items()
        ):
            print(
                f"  Class {class_id}: {count}"
            )

        print("Excluded IDs:")
        for sample_id in sorted(actual_excluded_ids):
            print(" ", sample_id)

    audit_df = pd.DataFrame(all_rows)

    training_df = audit_df[
        audit_df["status"] == "USABLE"
    ].copy()

    if len(audit_df) != 1125:
        raise ValueError(
            "Expected 1125 original reference records."
        )

    if len(training_df) != 1118:
        raise ValueError(
            "Expected 1118 usable reference records."
        )

    if training_df[FEATURE_NAMES].isna().any().any():
        raise ValueError(
            "Training data contains missing features."
        )

    # Check for multiple labelled samples from one
    # raster pixel within the same historical year.
    pixel_groups = training_df.groupby(
        ["year", "raster_row", "raster_col"]
    )

    duplicate_groups = int(
        (pixel_groups.size() > 1).sum()
    )

    conflicting_groups = int(
        (
            pixel_groups["class_id"].nunique() > 1
        ).sum()
    )

    print("\nDuplicate pixel groups:", duplicate_groups)
    print("Conflicting-label pixel groups:", conflicting_groups)

    if conflicting_groups:
        print(
            "WARNING: Conflicting labels require review "
            "before classifier training."
        )

    summary = {
        "part": 3,
        "stage": "LULC training table preparation",
        "original_reference_count": len(audit_df),
        "usable_reference_count": len(training_df),
        "excluded_reference_count": int(
            len(audit_df) - len(training_df)
        ),
        "duplicate_pixel_groups": duplicate_groups,
        "conflicting_label_pixel_groups": conflicting_groups,
        "years": yearly_summary,
        "policy": (
            "Original reference data preserved. "
            "NoData samples excluded from training. "
            "Valid outside-GN reference points retained "
            "and explicitly flagged for later spatial "
            "validation decisions."
        ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    audit_path = (
        OUTPUT_DIR / "lulc_reference_audit.csv"
    )

    training_path = (
        OUTPUT_DIR / "lulc_training_samples.csv"
    )

    summary_path = (
        OUTPUT_DIR / "lulc_training_summary.json"
    )

    audit_df.to_csv(
        audit_path,
        index=False
    )

    training_df.to_csv(
        training_path,
        index=False
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=2
        )

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print("Original samples:", len(audit_df))
    print("Usable samples:", len(training_df))
    print(
        "Excluded samples:",
        len(audit_df) - len(training_df)
    )

    print("\nGenerated files:")
    print(audit_path)
    print(training_path)
    print(summary_path)

    print("\nSCRIPT 02 DATA PREPARATION: PASS")
    print(
        "Next: examine duplicate samples and "
        "prepare the spatial validation design."
    )


if __name__ == "__main__":
    main()
