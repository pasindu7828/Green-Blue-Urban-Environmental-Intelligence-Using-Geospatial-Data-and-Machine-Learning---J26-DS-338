
"""
J26-DS-338 - Urban Green Research
Step 4F.2: Historical GIS Web Map Export

Converts existing classified GeoTIFFs into
transparent EPSG:4326 PNG map overlays.

The underlying classification codes are preserved
through nearest-neighbour resampling.

Only visualization assets are created.
No scientific statistics are recalculated.
"""

import csv
import json
import tempfile

from pathlib import Path

import numpy as np
import rasterio

from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import (
    calculate_default_transform,
    reproject,
)


ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    ROOT / "data" / "processed" / "lulc"
)

REPORTS = ROOT / "outputs" / "reports"

DESTINATION = (
    ROOT / "outputs" / "dashboard"
    / "historical_maps"
)


# --------------------------------------------------
# DISPLAY COLORS: RED, GREEN, BLUE, ALPHA
# --------------------------------------------------

ANNUAL_COLORS = {
    1: (35, 139, 69, 255),
    2: (214, 91, 82, 255),
    3: (217, 163, 93, 255),
    4: (52, 127, 194, 255),
}

TRANSITION_COLORS = {
    1: (35, 139, 69, 255),
    2: (214, 49, 49, 255),
    3: (245, 152, 58, 255),
    4: (56, 166, 180, 255),
    5: (135, 200, 93, 255),
    6: (185, 196, 80, 255),
    7: (79, 182, 155, 255),
    8: (145, 150, 155, 110),
}

CONFIDENCE_COLORS = {
    1: (237, 155, 54, 255),
    2: (43, 130, 191, 255),
}


def read_csv(name):
    with (REPORTS / name).open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        return list(csv.DictReader(file))


# --------------------------------------------------
# SOURCE RESEARCH VALIDATION
# --------------------------------------------------

qa = read_csv("09_annual_lulc_qa_summary.csv")
decision = read_csv(
    "10c_transition_reporting_method_decision.csv"
)
trajectory = read_csv(
    "10a_green_cover_annual_trajectory.csv"
)

assert len(qa) == 1
assert qa[0]["overall_step09_status"] == "PASS"

assert len(decision) == 1
assert (
    decision[0]["historical_transition_primary_rule"]
    == "ROBUST_2_OF_3"
)
assert (
    decision[0]["historical_transition_sensitivity_rule"]
    == "STRICT_3_OF_3"
)
assert (
    decision[0]["transition_method_locked"].lower()
    == "true"
)

assert len(trajectory) == 17

year_metadata = {
    int(row["year"]): {
        "period_status": row["period_status"],
        "included_in_core_trend": (
            row["included_in_core_trend"].lower()
            == "true"
        ),
        "provisional_year": (
            row["provisional_year"].lower()
            == "true"
        ),
    }
    for row in trajectory
}


annual_rows = read_csv(
    "08_annual_lulc_class_area_summary.csv"
)

annual_labels = {
    int(row["class_id"]): row["class_name"]
    for row in annual_rows
    if row["year"] == "2011"
    and row["product_type"] == "RAW_RF"
}

transition_rows = read_csv(
    "10b_green_transition_class_breakdown.csv"
)

transition_labels = {
    int(row["green_transition_code"]):
        row["green_transition_name"]
    for row in transition_rows
    if row["consensus_method"] == "ROBUST_2_OF_3"
}

assert set(annual_labels) == {1, 2, 3, 4}
assert set(transition_labels) == set(range(1, 9))


# --------------------------------------------------
# PREPARE 38 MAP SPECIFICATIONS
# --------------------------------------------------

specifications = []

annual_products = [
    (
        "RAW_RF",
        "raw",
        "Kaduwela_LULC_RAW_",
    ),
    (
        "STABILIZED_ANALYTICAL",
        "stabilized_mode_3yr",
        "Kaduwela_LULC_STABILIZED_",
    ),
]

for product, folder, prefix in annual_products:
    for year in range(2010, 2027):

        specifications.append({
            "id": f"{product}_{year}",
            "source": (
                SOURCE / "annual_maps" / folder
                / f"{prefix}{year}.tif"
            ),
            "image": (
                f"annual/{product.lower()}/{year}.png"
            ),
            "kind": "annual_lulc",
            "product_type": product,
            "year": year,
            "colors": ANNUAL_COLORS,
            "labels": annual_labels,
        })


transition_root = SOURCE / "transitions"

for method in ("ROBUST", "STRICT"):

    specifications.append({
        "id": f"GREEN_TRANSITION_{method}",
        "source": (
            transition_root
            / (
                "Kaduwela_Green_Transition_"
                "2012_2014_to_2021_2023_"
                f"{method}.tif"
            )
        ),
        "image": f"transitions/{method.lower()}.png",
        "kind": "green_transition",
        "method": (
            "ROBUST_2_OF_3"
            if method == "ROBUST"
            else "STRICT_3_OF_3"
        ),
        "colors": TRANSITION_COLORS,
        "labels": transition_labels,
    })


confidence_specs = [
    (
        "GREEN_TO_BUILT_CONFIDENCE",
        "Kaduwela_GreenToBuilt_Confidence_"
        "2012_2014_to_2021_2023.tif",
        "confidence/green_to_built.png",
        {
            1: "ROBUST-only green-to-built",
            2: "STRICT-confirmed green-to-built",
        },
    ),
    (
        "TRANSITION_CONFIDENCE",
        "Kaduwela_Transition_Confidence_"
        "2012_2014_to_2021_2023.tif",
        "confidence/all_transitions.png",
        {
            1: "ROBUST-only resolved transition",
            2: "STRICT-confirmed transition",
        },
    ),
]

for identifier, filename, image_path, labels in (
    confidence_specs
):

    specifications.append({
        "id": identifier,
        "source": transition_root / filename,
        "image": image_path,
        "kind": "transition_confidence",
        "colors": CONFIDENCE_COLORS,
        "labels": labels,
    })


assert len(specifications) == 38


# --------------------------------------------------
# CONVERT ONE CLASSIFIED RASTER TO PNG
# --------------------------------------------------

def export_overlay(spec, temporary_root):

    source_path = spec["source"]

    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    with rasterio.open(source_path) as src:

        if src.crs is None:
            raise ValueError(
                f"Source CRS missing: {source_path.name}"
            )

        if src.crs.to_epsg() != 32644:
            raise ValueError(
                f"Unexpected CRS: {source_path.name}"
            )

        if src.nodata != 0:
            raise ValueError(
                f"Unexpected NoData: {source_path.name}"
            )

        if src.count != 1 or src.dtypes[0] != "uint8":
            raise ValueError(
                f"Unexpected raster format: {source_path.name}"
            )

        if spec["kind"] == "green_transition":
            actual_method = src.tags().get(
                "consensus_method"
            )

            if actual_method != spec["method"]:
                raise ValueError(
                    f"Consensus tag mismatch: "
                    f"{source_path.name}"
                )

        transform, width, height = (
            calculate_default_transform(
                src.crs,
                "EPSG:4326",
                src.width,
                src.height,
                *src.bounds,
            )
        )

        destination_codes = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        reproject(
            source=rasterio.band(src, 1),
            destination=destination_codes,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=0,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

    allowed = set(spec["colors"]) | {0}
    found = set(
        int(value)
        for value in np.unique(destination_codes)
    )

    unexpected = found - allowed

    if unexpected:
        raise ValueError(
            f"Unexpected classification codes: {unexpected}"
        )

    # RGBA code 0 remains transparent.
    rgba = np.zeros(
        (height, width, 4),
        dtype=np.uint8,
    )

    for code, color in spec["colors"].items():
        rgba[destination_codes == code] = color

    target = temporary_root / spec["image"]
    target.parent.mkdir(
        parents=True, exist_ok=True
    )

    Image.fromarray(
        rgba,
        mode="RGBA",
    ).save(
        target,
        format="PNG",
        optimize=True,
    )

    west, south, east, north = array_bounds(
        height,
        width,
        transform,
    )

    metadata = {
        "id": spec["id"],
        "kind": spec["kind"],
        "source_file": str(
            source_path.relative_to(ROOT)
        ).replace("\\", "/"),
        "image": spec["image"],
        "crs": "EPSG:4326",
        "bounds": [
            [south, west],
            [north, east],
        ],
        "width": width,
        "height": height,
        "resampling": "nearest",
        "nodata_code": 0,
        "nodata_transparent": True,
        "legend": [
            {
                "code": int(code),
                "label": spec["labels"][code],
                "rgba": list(color),
            }
            for code, color in spec["colors"].items()
        ],
    }

    if "year" in spec:
        metadata["year"] = spec["year"]
        metadata["product_type"] = spec["product_type"]
        metadata.update(
            year_metadata[spec["year"]]
        )

    if "method" in spec:
        metadata["consensus_method"] = spec["method"]

    return metadata


# --------------------------------------------------
# BUILD IN TEMPORARY FOLDER, THEN PUBLISH
# --------------------------------------------------

def main():

    if DESTINATION.exists():
        raise FileExistsError(
            "Historical map output folder already exists. "
            "No existing assets will be overwritten: "
            f"{DESTINATION}"
        )

    # Validate source existence before writing.
    for spec in specifications:
        if not spec["source"].is_file():
            raise FileNotFoundError(spec["source"])

    DESTINATION.parent.mkdir(
        parents=True, exist_ok=True
    )

    with tempfile.TemporaryDirectory(
        prefix=".urban_green_maps_",
        dir=DESTINATION.parent,
    ) as temporary_directory:

        temporary_root = (
            Path(temporary_directory)
            / "historical_maps"
        )

        temporary_root.mkdir()

        records = []

        for index, spec in enumerate(
            specifications,
            start=1,
        ):

            record = export_overlay(
                spec,
                temporary_root,
            )

            records.append(record)

            print(
                f"[{index:02d}/38] "
                f"{record['id']} - OK"
            )

        manifest = {
            "project": "J26-DS-338",
            "component": "Urban Green Research",
            "dataset": "Historical LULC and Green Transition",
            "overlay_count": len(records),
            "annual_overlay_count": 34,
            "transition_overlay_count": 2,
            "confidence_overlay_count": 2,
            "primary_core_trend": "2011-2024",
            "primary_transition": "ROBUST_2_OF_3",
            "conservative_transition": "STRICT_3_OF_3",
            "transition_early_window": "2012-2014",
            "transition_late_window": "2021-2023",
            "statistical_confidence_interval": False,
            "note": (
                "Visualization overlays only. "
                "Use the original validated CSV reports "
                "for area statistics and research conclusions."
            ),
            "maps": records,
        }

        manifest_path = (
            temporary_root / "manifest.json"
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        # New folder only: existing dashboard files
        # and the original rasters are untouched.
        temporary_root.rename(DESTINATION)

    print("\n[PASS] 38 historical map overlays created")
    print("[PASS] New map manifest created")
    print("[PASS] Original raster files unchanged")
    print("[PASS] Existing dashboard exports unchanged")
    print("\nOutput:", DESTINATION)


if __name__ == "__main__":
    main()
