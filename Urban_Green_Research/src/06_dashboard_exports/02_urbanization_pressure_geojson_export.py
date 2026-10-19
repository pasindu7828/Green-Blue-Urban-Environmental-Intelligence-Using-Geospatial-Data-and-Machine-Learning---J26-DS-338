
"""
J26-DS-338 - Urban Green Research

STEP 5D.4
Export final Urbanization Pressure GIS to GeoJSON.

Research source:
Step 14D finalized pressure-category GeoPackage.

Rules:
- Preserve original 9,240 grid IDs
- Preserve scored and unscored domains
- Preserve model scores and categories
- Use WGS84 for frontend GIS
- No model retraining or recalculation
- No existing file overwriting
"""

from pathlib import Path
from collections import Counter
import csv
import json
import math
import tempfile

import geopandas as gpd


ROOT = Path(__file__).resolve().parents[2]

SOURCE_ROOT = (
    ROOT / "data" / "processed"
    / "urbanization_pressure" / "inference"
)

SOURCE_GIS = (
    SOURCE_ROOT
    / "14d_2026_2028_pressure_categories.gpkg"
)

SOURCE_CSV = (
    SOURCE_ROOT
    / "14d_2026_2028_pressure_categories.csv"
)

OUTPUT_ROOT = (
    ROOT / "outputs" / "dashboard"
    / "urbanization_pressure"
)


COLORS = {
    "NOT_IN_INFERENCE_DOMAIN": "#A6ADB5",
    "LOW_RELATIVE": "#51A477",
    "MODERATE_RELATIVE": "#C8BE61",
    "ELEVATED_RELATIVE": "#EAA461",
    "HIGH_RELATIVE": "#CE774E",
    "THRESHOLD_FLAGGED": "#B83F47",
}

EXPECTED_COUNTS = {
    "NOT_IN_INFERENCE_DOMAIN": 4620,
    "LOW_RELATIVE": 2310,
    "MODERATE_RELATIVE": 1155,
    "ELEVATED_RELATIVE": 693,
    "HIGH_RELATIVE": 207,
    "THRESHOLD_FLAGGED": 255,
}

KEEP_COLUMNS = [
    "grid_id",
    "grid_row",
    "grid_col",
    "kmc_overlap_fraction",
    "kmc_overlap_area_m2",
    "spatial_overlap_eligible",
    "coverage_eligible",
    "current_nonbuilt_eligible",
    "past_stability_eligible",
    "final_inference_ready",
    "urbanization_pressure_score",
    "pressure_percentile",
    "pressure_rank",
    "operating_threshold",
    "threshold_flag",
    "origin_year",
    "pressure_period",
    "map_status",
    "pressure_category",
    "pressure_category_order",
    "pressure_category_basis",
    "score_is_calibrated_probability",
    "category_is_probability_class",
]


def load_csv_lookup():
    with SOURCE_CSV.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    if len(rows) != 4620:
        raise ValueError(
            "Expected 4620 final scored CSV records"
        )

    lookup = {}

    for row in rows:
        grid_id = row["grid_id"]

        if grid_id in lookup:
            raise ValueError(
                f"Duplicate CSV grid: {grid_id}"
            )

        lookup[grid_id] = row

    return lookup


def main():

    print("\n========== STEP 5D.4: GIS EXPORT ==========")

    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            "Output folder already exists; refusing "
            f"to overwrite: {OUTPUT_ROOT}"
        )

    if not SOURCE_GIS.is_file():
        raise FileNotFoundError(SOURCE_GIS)

    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(SOURCE_CSV)

    print("\n1. Loading final research GIS...")

    gdf = gpd.read_file(
        SOURCE_GIS,
        layer="pressure_categories_2026_2028",
    )

    assert len(gdf) == 9240
    assert gdf.crs.to_epsg() == 32644
    assert gdf["grid_id"].nunique() == 9240
    assert gdf.geometry.notna().all()
    assert (~gdf.geometry.is_empty).all()
    assert gdf.geometry.is_valid.all()

    missing_columns = (
        set(KEEP_COLUMNS) - set(gdf.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing source columns: {missing_columns}"
        )

    print("[PASS] 9240 original polygons loaded")


    print("\n2. Verifying research classifications...")

    counts = Counter(
        gdf["pressure_category"].astype(str)
    )

    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(
            f"Category count mismatch: {counts}"
        )

    score_lookup = load_csv_lookup()

    scored = gdf[
        gdf["final_inference_ready"] == 1
    ]

    unscored = gdf[
        gdf["final_inference_ready"] == 0
    ]

    assert len(scored) == 4620
    assert len(unscored) == 4620

    assert (
        set(scored["grid_id"])
        == set(score_lookup)
    )

    assert unscored[
        "urbanization_pressure_score"
    ].isna().all()

    assert unscored[
        "pressure_category"
    ].eq("NOT_IN_INFERENCE_DOMAIN").all()

    assert unscored["threshold_flag"].isna().all()

    assert scored[
        "urbanization_pressure_score"
    ].notna().all()

    for row in scored.itertuples():

        original = score_lookup[row.grid_id]

        if original["pressure_category"] != row.pressure_category:
            raise ValueError(
                f"Category mismatch: {row.grid_id}"
            )

        if not math.isclose(
            float(original["urbanization_pressure_score"]),
            float(row.urbanization_pressure_score),
            abs_tol=1e-8,
            rel_tol=0,
        ):
            raise ValueError(
                f"Score mismatch: {row.grid_id}"
            )

        if not math.isclose(
            float(row.operating_threshold),
            0.715,
            abs_tol=1e-10,
            rel_tol=0,
        ):
            raise ValueError(
                f"Threshold mismatch: {row.grid_id}"
            )

    print("[PASS] Final CSV and GIS scores agree")
    print("[PASS] Outside-domain values remain NULL")
    print("[PASS] Locked threshold unchanged")


    print("\n3. Converting polygons to WGS84...")

    selected = gdf[
        KEEP_COLUMNS + ["geometry"]
    ].copy()

    wgs84 = selected.to_crs(epsg=4326)

    assert wgs84.crs.to_epsg() == 4326
    assert wgs84.geometry.is_valid.all()

    print("[PASS] Reprojected to EPSG:4326")


    print("\n4. Building frontend GIS features...")

    geojson = json.loads(
        wgs84.to_json(
            na="null",
            drop_id=True,
        )
    )

    features = geojson["features"]

    assert len(features) == 9240

    for feature in features:

        props = feature["properties"]

        for field in (
            "final_inference_ready",
            "spatial_overlap_eligible",
            "coverage_eligible",
            "current_nonbuilt_eligible",
            "past_stability_eligible",
            "score_is_calibrated_probability",
            "category_is_probability_class",
        ):
            value = props[field]

            if value not in (0, 1, False, True):
                raise ValueError(
                    f"Invalid boolean field: {field}"
                )

            props[field] = bool(value)

        if props["threshold_flag"] is not None:
            props["threshold_flag"] = bool(
                props["threshold_flag"]
            )

        ready = props["final_inference_ready"]

        if not ready:

            assert (
                props["urbanization_pressure_score"]
                is None
            )

            assert props["threshold_flag"] is None

            assert (
                props["pressure_category"]
                == "NOT_IN_INFERENCE_DOMAIN"
            )

        else:

            score = props[
                "urbanization_pressure_score"
            ]

            assert score is not None
            assert 0 <= score <= 1

            assert props["threshold_flag"] is (
                score >= 0.715
            )

        assert (
            props["score_is_calibrated_probability"]
            is False
        )

        assert (
            props["category_is_probability_class"]
            is False
        )

        assert props["origin_year"] == 2025
        assert props["pressure_period"] == "2026-2028"

    print("[PASS] All features and types verified")


    print("\n5. Preparing output metadata...")

    metadata = {
        "research_component": "Urban Green Research",
        "source_stage": "14D",
        "origin_year": 2025,
        "pressure_period": "2026-2028",
        "prediction_window_type": "SINGLE_THREE_YEAR_WINDOW",
        "model": "XGBoost",
        "operating_threshold": 0.715,
        "score_is_calibrated_probability": False,
        "categories_are_probability_bands": False,
        "permanent_grid_count": 9240,
        "eligible_scored_count": 4620,
        "not_in_inference_domain_count": 4620,
        "threshold_flagged_count": 255,
        "coordinate_reference_system": "EPSG:4326",
        "source_coordinate_reference_system": "EPSG:32644",
        "source_gis": str(
            SOURCE_GIS.relative_to(ROOT)
        ).replace("\\", "/"),
        "geojson_file": "pressure_map.geojson",
        "bounds_west_south_east_north": (
            wgs84.total_bounds.tolist()
        ),
        "category_counts": EXPECTED_COUNTS,
        "display_legend": [
            {
                "category": category,
                "color": color,
                "grid_count": EXPECTED_COUNTS[category],
            }
            for category, color in COLORS.items()
        ],
        "notes": [
            "Single 2026-2028 prediction window",
            "Pressure scores are not calibrated probabilities",
            "Categories are relative visualization bands",
            "Unscored cells are outside the inference domain",
            "No research model or scores recalculated",
        ],
    }


    print("\n6. Writing new dashboard GIS files...")

    OUTPUT_ROOT.parent.mkdir(
        parents=True, exist_ok=True
    )

    with tempfile.TemporaryDirectory(
        prefix=".pressure_export_",
        dir=OUTPUT_ROOT.parent,
    ) as temporary_dir:

        temporary_output = (
            Path(temporary_dir)
            / "urbanization_pressure"
        )

        temporary_output.mkdir()

        map_file = (
            temporary_output
            / "pressure_map.geojson"
        )

        map_file.write_text(
            json.dumps(
                geojson,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        metadata_file = (
            temporary_output
            / "metadata.json"
        )

        metadata_file.write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        # Validate saved output before publishing.
        saved_map = json.loads(
            map_file.read_text(
                encoding="utf-8"
            )
        )

        assert len(saved_map["features"]) == 9240

        temporary_output.rename(OUTPUT_ROOT)


    print("\n========== FINAL RESULT ==========")

    print("Total GIS features:", len(features))
    print("Scored features:", len(scored))
    print("Unscored features:", len(unscored))
    print("CRS: EPSG:4326")
    print("Output:", OUTPUT_ROOT)

    print("\n[PASS] STEP 5D.4 GIS EXPORT COMPLETE")
    print("[PASS] Original GIS and research data unchanged")


if __name__ == "__main__":
    main()
