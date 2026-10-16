
from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# PART 4 - SCRIPT 14
# INITIAL SUSCEPTIBILITY MODEL SENSITIVITY ANALYSIS
#
# Tests:
# - Alternative slope / TPI / building weights
# - Alternative DEM source (NASADEM)
# - Class stability, direction of change, score changes
#
# Uses the exact Script 13 normalization reference
# for all scenarios to keep scores comparable.
#
# No new susceptibility labels or model training.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data/processed/part4_waterlogging"
METRICS = ROOT / "outputs/metrics/part4_waterlogging"
REPORTS = ROOT / "outputs/reports/part4_waterlogging"

FILES = {
    "mask": "06_Kaduwela_Study_Mask_30m.tif",
    "screen": "12_Landcover_Screening_30m.tif",
    "density": "09_OSM_Building_Density_300m_30m.tif",
    "ged_slope": "07_GEDTM30_Slope_Degrees_30m.tif",
    "ged_tpi": "07_GEDTM30_TPI_300m_30m.tif",
    "nasa_slope": "07_NASADEM_Slope_Degrees_30m.tif",
    "nasa_tpi": "07_NASADEM_TPI_300m_30m.tif",
    "script13_score":
        "13_Terrain_Urban_Susceptibility_Score_30m.tif",
    "script13_class":
        "13_Terrain_Urban_Susceptibility_Class_30m.tif",
}

SCENARIOS = {
    "Terrain_Only": ("GEDTM30", 0.50, 0.50, 0.00),
    "Terrain_Urban": ("GEDTM30", 0.40, 0.40, 0.20),
    "Urban_10pct": ("GEDTM30", 0.45, 0.45, 0.10),
    "Urban_30pct": ("GEDTM30", 0.35, 0.35, 0.30),
    "Slope_Emphasis": ("GEDTM30", 0.60, 0.20, 0.20),
    "TPI_Emphasis": ("GEDTM30", 0.20, 0.60, 0.20),
    "NASADEM_Terrain_Urban":
        ("NASADEM", 0.40, 0.40, 0.20),
    "NASADEM_Terrain_Only":
        ("NASADEM", 0.50, 0.50, 0.00),
}

CLASS_NAMES = ["Low", "Moderate", "High", "Very_High"]


def normalize(values, low, high, inverse=False):
    result = np.clip(
        (values - low) / (high - low),
        0.0,
        1.0,
    )

    return 1.0 - result if inverse else result


def classify(values):
    return (
        np.digitize(
            values,
            [0.25, 0.50, 0.75],
            right=False,
        ).astype(np.uint8)
        + 1
    )


def main():

    print("=" * 70)
    print("PART 4 - MODEL SENSITIVITY ANALYSIS")
    print("SCRIPT 14")
    print("=" * 70)

    METRICS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    arrays = {}
    expected = None

    for key, filename in FILES.items():

        with rasterio.open(DATA / filename) as src:

            alignment = (
                src.crs,
                src.transform,
                src.shape,
            )

            if expected is None:
                expected = alignment
            elif alignment != expected:
                raise ValueError(
                    f"Grid alignment mismatch: {key}"
                )

            arrays[key] = src.read(1)

            if key not in ("mask", "screen"):
                valid_source = arrays[key]
                arrays[key] = valid_source.astype(
                    np.float64
                )

    study = arrays["mask"] == 1
    general = study & (arrays["screen"] == 1)

    if (
        int(study.sum()) != 94431
        or int(general.sum()) != 51707
    ):
        raise ValueError(
            "Study-area or general-land count mismatch"
        )

    for key in [
        "density",
        "ged_slope",
        "ged_tpi",
        "nasa_slope",
        "nasa_tpi",
        "script13_score",
    ]:
        values = arrays[key][study]

        if (
            not np.isfinite(values).all()
            or np.any(values == -9999.0)
        ):
            raise ValueError(
                f"Invalid study-area values: {key}"
            )

    # --------------------------------------------------------
    # 1. Fixed GEDTM30 normalization reference
    # --------------------------------------------------------

    limits = {}

    for factor, key in [
        ("slope", "ged_slope"),
        ("tpi", "ged_tpi"),
        ("density", "density"),
    ]:
        reference = arrays[key][general]

        low, high = np.percentile(
            reference, [5, 95]
        )

        if high <= low:
            raise ValueError(
                f"Invalid normalization range: {factor}"
            )

        limits[factor] = (float(low), float(high))

    # --------------------------------------------------------
    # 2. Prepare factor scores
    # --------------------------------------------------------

    factors = {}

    for source in ("GEDTM30", "NASADEM"):

        prefix = (
            "ged" if source == "GEDTM30" else "nasa"
        )

        slope = arrays[f"{prefix}_slope"][general]
        tpi = arrays[f"{prefix}_tpi"][general]
        density = arrays["density"][general]

        factors[source] = {
            "slope": normalize(
                slope,
                *limits["slope"],
                inverse=True,
            ),
            "tpi": normalize(
                tpi,
                *limits["tpi"],
                inverse=True,
            ),
            "density": normalize(
                density,
                *limits["density"],
                inverse=False,
            ),
        }

    # --------------------------------------------------------
    # 3. Reproduce Script 13 baseline exactly
    # --------------------------------------------------------

    baseline = (
        0.40 * factors["GEDTM30"]["slope"]
        + 0.40 * factors["GEDTM30"]["tpi"]
        + 0.20 * factors["GEDTM30"]["density"]
    )

    baseline_classes = classify(baseline)

    original_score = arrays["script13_score"][general]
    original_classes = arrays["script13_class"][general]

    max_score_error = float(
        np.max(np.abs(
            baseline - original_score
        ))
    )

    if (
        max_score_error > 1e-5
        or not np.array_equal(
            baseline_classes,
            original_classes,
        )
    ):
        raise ValueError(
            "Cannot reproduce Script 13 baseline. "
            "Do not continue sensitivity analysis."
        )

    print("\nSCRIPT 13 REPRODUCTION: PASS")
    print(
        "Maximum score difference:",
        round(max_score_error, 8)
    )

    # --------------------------------------------------------
    # 4. Evaluate alternative scenarios
    # --------------------------------------------------------

    rows = []
    transition_records = {}

    for name, definition in SCENARIOS.items():

        source, w_slope, w_tpi, w_density = definition

        weights = [w_slope, w_tpi, w_density]

        if not np.isclose(sum(weights), 1.0):
            raise ValueError(
                f"Invalid scenario weights: {name}"
            )

        selected = factors[source]

        score = (
            w_slope * selected["slope"]
            + w_tpi * selected["tpi"]
            + w_density * selected["density"]
        )

        classes = classify(score)

        if not np.isfinite(score).all():
            raise ValueError(
                f"Nonfinite score in {name}"
            )

        if np.any(score < 0) or np.any(score > 1):
            raise ValueError(
                f"Out-of-range score in {name}"
            )

        changed = classes != baseline_classes

        up = int(
            np.count_nonzero(
                classes > baseline_classes
            )
        )

        down = int(
            np.count_nonzero(
                classes < baseline_classes
            )
        )

        class_shares = {
            CLASS_NAMES[c - 1]: round(
                100 * float(np.mean(classes == c)),
                4,
            )
            for c in range(1, 5)
        }

        rows.append({
            "scenario": name,
            "dem": source,
            "slope_weight": w_slope,
            "tpi_weight": w_tpi,
            "building_weight": w_density,
            "mean_score": round(
                float(score.mean()), 6
            ),
            "mean_absolute_score_change": round(
                float(np.mean(
                    np.abs(score - baseline)
                )),
                6,
            ),
            "changed_class_percent": round(
                100 * float(np.mean(changed)),
                4,
            ),
            "moved_higher_percent": round(
                100 * up / len(classes), 4
            ),
            "moved_lower_percent": round(
                100 * down / len(classes), 4
            ),
            **{
                f"{key}_percent": value
                for key, value in class_shares.items()
            },
        })

        transition_records[name] = {
            "changed_cells": int(changed.sum()),
            "moved_higher_cells": up,
            "moved_lower_cells": down,
            "unchanged_cells": int(
                np.count_nonzero(~changed)
            ),
        }

        print("\nSCENARIO:", name)
        print("DEM:", source)
        print(
            "Class changed:",
            rows[-1]["changed_class_percent"], "%"
        )
        print(
            "Moved higher:",
            rows[-1]["moved_higher_percent"], "%"
        )
        print(
            "Moved lower:",
            rows[-1]["moved_lower_percent"], "%"
        )

    # --------------------------------------------------------
    # 5. Save permanent evidence
    # --------------------------------------------------------

    metrics_path = (
        METRICS
        / "script14_weight_dem_sensitivity.csv"
    )

    pd.DataFrame(rows).to_csv(
        metrics_path,
        index=False,
    )

    report_path = (
        REPORTS
        / "script14_weight_dem_sensitivity.json"
    )

    report = {
        "component":
            "Part 4 - Seasonal Waterlogging",
        "stage":
            "Initial model sensitivity assessment",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "reference_scenario":
            "Terrain_Urban using GEDTM30",
        "reference_model_reproduced": True,
        "reference_score_max_difference":
            max_score_error,
        "general_land_cells": int(general.sum()),
        "normalization_source":
            "GEDTM30 general-land P05/P95 reference "
            "for slope and TPI, and general-land "
            "P05/P95 for 2026 building density. "
            "Fixed across all scenarios.",
        "normalization_limits": {
            key: {
                "p05": values[0],
                "p95": values[1],
            }
            for key, values in limits.items()
        },
        "scenario_results": rows,
        "transition_counts": transition_records,
        "class_boundaries": [
            0.25, 0.50, 0.75
        ],
        "limitations": [
            "Scenario weights are exploratory.",
            "Changes are measured relative to the "
            "GEDTM30 Terrain_Urban reference model.",
            "Cell-level counts are spatially "
            "autocorrelated, not independent samples.",
            "Fixed GEDTM30 normalization thresholds "
            "allow DEM substitution comparisons, "
            "but do not remove vertical datum "
            "or absolute accuracy concerns.",
            "Equal-interval class boundaries are "
            "not calibrated hazard thresholds.",
            "Class stability is not proof of "
            "predictive accuracy.",
            "Historical inundation comparison "
            "and other modelling uncertainties "
            "still require evaluation.",
            "No final model is selected here.",
        ],
        "raw_data_modified": False,
        "status":
            "INITIAL_SENSITIVITY_ASSESSMENT_COMPLETE",
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("SCRIPT 14 - SUMMARY")
    print("=" * 70)

    print("Scenarios tested:", len(rows))
    print("General-land cells:", int(general.sum()))
    print("Saved:", metrics_path)
    print("Saved:", report_path)

    print("\nSENSITIVITY ANALYSIS: COMPLETE")
    print("HISTORICAL EVENT EVALUATION: PENDING")
    print("FINAL MODEL SELECTION: PENDING")
    print("RAW DATA MODIFIED: NO")


if __name__ == "__main__":
    main()
