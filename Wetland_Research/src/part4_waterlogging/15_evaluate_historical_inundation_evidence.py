
from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import json

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import mapping


# ============================================================
# PART 4 - SCRIPT 15
# HISTORICAL INUNDATION EVIDENCE EVALUATION
#
# Compare exploratory susceptibility scenarios with
# visually interpreted 2018 inundation footprints.
#
# IMPORTANT:
# - Event footprints are NOT verified ground truth.
# - No negative waterlogging labels are created.
# - No classification accuracy / ROC-AUC is claimed.
# - Overlapping event areas are identified separately.
# - Existing data and model outputs are not modified.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data/processed/part4_waterlogging"
METRICS = ROOT / "outputs/metrics/part4_waterlogging"
REPORTS = ROOT / "outputs/reports/part4_waterlogging"

MASK_FILE = DATA / "06_Kaduwela_Study_Mask_30m.tif"
SCREEN_FILE = DATA / "12_Landcover_Screening_30m.tif"

SCENARIOS = {
    "Terrain_Only": {
        "score": (
            "13_Terrain_Only_Susceptibility_Score_30m.tif"
        ),
        "class": (
            "13_Terrain_Only_Susceptibility_Class_30m.tif"
        ),
    },
    "Terrain_Urban": {
        "score": (
            "13_Terrain_Urban_Susceptibility_Score_30m.tif"
        ),
        "class": (
            "13_Terrain_Urban_Susceptibility_Class_30m.tif"
        ),
    },
}

SCREEN_CATEGORIES = {
    1: "General_Land",
    2: "Paddy_Agriculture",
    3: "Wetland_Marsh",
    4: "Open_Water",
    5: "Unknown_Landcover",
}

CLASS_NAMES = {
    1: "Low",
    2: "Moderate",
    3: "High",
    4: "Very_High",
}


def load_event_reader():
    """
    Reuse Script 05's documented geometry-repair
    procedure without changing original event KMLs.
    """
    path = (
        ROOT
        / "src/part4_waterlogging"
        / "05_assess_inundation_wetland_context.py"
    )

    spec = importlib.util.spec_from_file_location(
        "part4_event_reader", path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.read_event_polygon


def main():

    print("=" * 70)
    print("PART 4 - HISTORICAL INUNDATION EVIDENCE")
    print("SCRIPT 15")
    print("=" * 70)

    METRICS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Read common study grid and screening categories
    # --------------------------------------------------------

    with rasterio.open(MASK_FILE) as src:
        study = src.read(1) == 1
        expected = (src.crs, src.transform, src.shape)
        raster_transform = src.transform
        raster_shape = src.shape
        pixel_area_ha = abs(
            src.transform.a * src.transform.e
        ) / 10000

    if int(study.sum()) != 94431:
        raise ValueError("Unexpected study-area size")

    with rasterio.open(SCREEN_FILE) as src:
        if (
            src.crs, src.transform, src.shape
        ) != expected:
            raise ValueError("Screening grid mismatch")

        screening = src.read(1)

    if not np.isin(
        screening[study],
        list(SCREEN_CATEGORIES),
    ).all():
        raise ValueError("Unexpected screening categories")

    general = study & (screening == 1)

    # --------------------------------------------------------
    # 2. Load susceptibility model outputs
    # --------------------------------------------------------

    models = {}

    for scenario, filenames in SCENARIOS.items():

        models[scenario] = {}

        for kind, filename in filenames.items():

            with rasterio.open(DATA / filename) as src:

                if (
                    src.crs, src.transform, src.shape
                ) != expected:
                    raise ValueError(
                        f"Grid mismatch: {scenario} {kind}"
                    )

                values = src.read(1)

                if kind == "class":
                    if not np.isin(
                        values[study], [1, 2, 3, 4]
                    ).all():
                        raise ValueError(
                            "Invalid susceptibility classes"
                        )
                else:
                    if (
                        not np.isfinite(values[study]).all()
                        or np.any(values[study] < 0)
                        or np.any(values[study] > 1)
                    ):
                        raise ValueError(
                            "Invalid susceptibility scores"
                        )

                models[scenario][kind] = values

    # --------------------------------------------------------
    # 3. Rasterize the two repaired historical footprints
    # --------------------------------------------------------

    read_event_polygon = load_event_reader()

    event_masks = {}
    repair_summaries = {}

    for event_id in ("E001", "E002"):

        geometry, audit = read_event_polygon(event_id)

        if geometry.is_empty or not geometry.is_valid:
            raise ValueError(
                f"Unusable geometry for {event_id}"
            )

        event_mask = geometry_mask(
            [mapping(geometry)],
            out_shape=raster_shape,
            transform=raster_transform,
            invert=True,
            all_touched=False,
        ) & study

        if not event_mask.any():
            raise ValueError(
                f"No sampled pixels for {event_id}"
            )

        event_masks[event_id] = event_mask

        repair_summaries[event_id] = {
            "polygon_count": len(audit),
            "repaired_polygons": sum(
                item["repair_applied"]
                for item in audit
            ),
        }

    e001 = event_masks["E001"]
    e002 = event_masks["E002"]

    overlap = e001 & e002

    # These subdivisions prevent duplicate counting
    # when inspecting the two event footprints.
    regions = {
        "E001": e001,
        "E002": e002,
        "E001_Only": e001 & ~e002,
        "E002_Only": e002 & ~e001,
        "Both_Events": overlap,
    }

    print("\nEVENT FOOTPRINTS")
    print("E001 sampled cells:", int(e001.sum()))
    print("E002 sampled cells:", int(e002.sum()))
    print("Shared sampled cells:", int(overlap.sum()))

    print(
        "E002 sampled cells also in E001:",
        round(
            100 * overlap.sum() / e002.sum(), 2
        ),
        "%"
    )

    # --------------------------------------------------------
    # 4. Evaluate event land-cover composition
    # --------------------------------------------------------

    context_rows = []

    for region_name, region in regions.items():

        total = int(region.sum())

        for code, category in SCREEN_CATEGORIES.items():

            count = int(
                np.count_nonzero(
                    region & (screening == code)
                )
            )

            context_rows.append({
                "region": region_name,
                "screening_category": category,
                "sampled_cells": count,
                "approximate_area_ha":
                    round(count * pixel_area_ha, 4),
                "percent_of_region":
                    round(
                        100 * count / total
                        if total else 0,
                        4,
                    ),
            })

    # --------------------------------------------------------
    # 5. Compare general-land susceptibility distributions
    # --------------------------------------------------------

    comparison_rows = []
    class_rows = []

    for scenario, model in models.items():

        scores = model["score"]
        classes = model["class"]

        background_high = float(
            np.mean(classes[general] >= 3)
        )

        for region_name, region in regions.items():

            selected = region & general
            count = int(selected.sum())

            if count:

                event_scores = scores[selected]
                event_classes = classes[selected]

                mean_score = float(
                    event_scores.mean()
                )

                high_share = float(
                    np.mean(event_classes >= 3)
                )

                difference_pp = (
                    100 * (high_share - background_high)
                )

            else:
                mean_score = None
                high_share = None
                difference_pp = None

            comparison_rows.append({
                "scenario": scenario,
                "region": region_name,
                "general_land_event_cells": count,
                "mean_event_score":
                    round(mean_score, 6)
                    if mean_score is not None else None,
                "event_high_veryhigh_percent":
                    round(100 * high_share, 4)
                    if high_share is not None else None,
                "background_high_veryhigh_percent":
                    round(100 * background_high, 4),
                "difference_percentage_points":
                    round(difference_pp, 4)
                    if difference_pp is not None else None,
            })

            for class_id, class_name in CLASS_NAMES.items():

                class_count = int(
                    np.count_nonzero(
                        selected & (classes == class_id)
                    )
                )

                class_rows.append({
                    "scenario": scenario,
                    "region": region_name,
                    "susceptibility_class": class_name,
                    "general_land_cells": class_count,
                    "percent_of_region_general_land":
                        round(
                            100 * class_count / count
                            if count else 0,
                            4,
                        ),
                })

            if region_name in ("E001", "E002"):
                print(
                    "\n", scenario, region_name,
                    "| General-land cells:", count,
                    "| High/Very High:",
                    (
                        f"{100 * high_share:.2f}%"
                        if high_share is not None
                        else "NOT AVAILABLE"
                    )
                )

    # --------------------------------------------------------
    # 6. Save permanent supporting evidence
    # --------------------------------------------------------

    context_path = (
        METRICS
        / "script15_event_landcover_context.csv"
    )

    comparison_path = (
        METRICS
        / "script15_event_susceptibility_comparison.csv"
    )

    classes_path = (
        METRICS
        / "script15_event_class_distributions.csv"
    )

    pd.DataFrame(context_rows).to_csv(
        context_path, index=False
    )

    pd.DataFrame(comparison_rows).to_csv(
        comparison_path, index=False
    )

    pd.DataFrame(class_rows).to_csv(
        classes_path, index=False
    )

    report_path = (
        REPORTS
        / "script15_historical_evidence_evaluation.json"
    )

    report = {
        "component": "Part 4 - Seasonal Waterlogging",
        "stage": "Historical inundation evidence evaluation",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "evaluation_type":
            "Exploratory spatial consistency assessment",
        "event_dates": {
            "E001": "2018-05-29",
            "E002": "2018-11-25",
        },
        "raster_sampling":
            "30m grid-cell centre inclusion",
        "sampled_event_cells": {
            "E001": int(e001.sum()),
            "E002": int(e002.sum()),
            "shared": int(overlap.sum()),
        },
        "e002_shared_with_e001_percent":
            round(
                100 * overlap.sum() / e002.sum(), 4
            ),
        "geometry_repairs": repair_summaries,
        "screening_source_year": 2020,
        "urban_osm_mapping_year": 2026,
        "context_rows": context_rows,
        "comparisons": comparison_rows,
        "output_csvs": [
            str(context_path.relative_to(ROOT)),
            str(comparison_path.relative_to(ROOT)),
            str(classes_path.relative_to(ROOT)),
        ],
        "limitations": [
            "Both event footprints were manually "
            "interpreted from historical imagery.",
            "Visible inundation does not prove "
            "abnormal seasonal waterlogging.",
            "The two footprints overlap substantially "
            "and must not be counted as independent "
            "waterlogging confirmations.",
            "2020 land cover may not represent "
            "land cover on either 2018 event date.",
            "2026 OSM mapping does not describe "
            "verified drainage or building conditions "
            "during the 2018 events.",
            "Permanent water, natural wetlands "
            "and paddy inundation are potential "
            "confounders.",
            "Event coverage is not exhaustive; "
            "unmapped areas are not negative examples.",
            "General-land comparison is conditioned "
            "on the 2020 screening classification.",
            "High-class enrichment relative to "
            "background is descriptive, not "
            "classification accuracy or validation.",
            "Cell observations are spatially "
            "autocorrelated.",
            "No model is approved as final here.",
        ],
        "raw_data_modified": False,
        "status":
            "EXPLORATORY_EVENT_CONCORDANCE_COMPLETE",
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("SCRIPT 15 - SUMMARY")
    print("=" * 70)

    print("Scenarios evaluated:", len(SCENARIOS))
    print("Historical event dates:", 2)
    print("Shared event cells:", int(overlap.sum()))

    print("\nSaved:", context_path)
    print("Saved:", comparison_path)
    print("Saved:", classes_path)
    print("Saved:", report_path)

    print("\nHISTORICAL EVIDENCE COMPARISON: COMPLETE")
    print("INDEPENDENT MODEL VALIDATION: NOT CLAIMED")
    print("FINAL MODEL SELECTION: PENDING")
    print("RAW DATA MODIFIED: NO")


if __name__ == "__main__":
    main()
