
from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import json

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# PART 3 - FUTURE WETLAND BOUNDARY PROJECTION
# SCRIPT 08 - EXPLORATORY 2030 CA-MARKOV SCENARIO
#
# Estimate transitions from: 2020 -> 2025
# Initial land-cover map:    2025
# Scenario year:             2030
#
# Uses the SAME neighbourhood allocation method
# already implemented in Script 07.
#
# IMPORTANT:
# This is an exploratory scenario, NOT a validated
# prediction of 2030 land cover or wetland boundaries.
# ============================================================


ROOT = Path(__file__).resolve().parents[2]

SOURCE_SCRIPT = (
    ROOT / "src/part3_future_projection"
    / "07_ca_markov_hindcast.py"
)

MAP_DIR = (
    ROOT / "data/processed/part3_future_projection"
)

METRIC_DIR = (
    ROOT / "outputs/metrics/part3_future_projection"
)

REPORT_DIR = (
    ROOT / "outputs/reports/part3_future_projection"
)

YEARS = [2020, 2025, 2030]
CLASSES = np.arange(1, 7)

CLASS_NAMES = {
    1: "Wetland_Marsh",
    2: "Open_Water",
    3: "Built_Up",
    4: "Paddy_Agriculture",
    5: "Other_Vegetation",
    6: "Bare_Fill_Disturbed",
}

EXPECTED_PIXELS = 94258
SEED = 42

LULC_OUTPUT = (
    MAP_DIR / "LULC_2030_Exploratory_CA_Markov.tif"
)

WETLAND_OUTPUT = (
    MAP_DIR / "Wetland_2030_Exploratory_CA_Markov.tif"
)


def load_existing_ca_method():
    """
    Reuse Script 07's frozen CA allocation implementation.

    Importing the module does not execute its main()
    function or regenerate the 2025 hindcast.
    """

    if not SOURCE_SCRIPT.is_file():
        raise FileNotFoundError(SOURCE_SCRIPT)

    spec = importlib.util.spec_from_file_location(
        "part3_ca_hindcast_methods",
        SOURCE_SCRIPT
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            "Unable to import Script 07."
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if module.SEED != SEED:
        raise ValueError(
            "CA random seed differs from Script 07."
        )

    return module


def transition_probabilities(counts):
    """
    Rows = earlier class; columns = later class.
    """

    counts = np.asarray(
        counts,
        dtype=np.int64
    )

    if counts.shape != (6, 6):
        raise ValueError(
            "Expected a 6x6 transition matrix."
        )

    if (counts < 0).any():
        raise ValueError(
            "Negative transition counts."
        )

    totals = counts.sum(axis=1)

    if (totals == 0).any():
        raise ValueError(
            "At least one class has zero source pixels."
        )

    probabilities = (
        counts / totals[:, None]
    )

    if not np.allclose(
        probabilities.sum(axis=1),
        1,
        atol=1e-10
    ):
        raise ValueError(
            "Transition probabilities do not sum to 1."
        )

    return probabilities


def write_lulc(
    path,
    array,
    profile,
    ca_module,
):
    updated = profile.copy()

    updated.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
    )

    with rasterio.open(
        path,
        "w",
        **updated
    ) as dst:

        dst.write(array.astype("uint8"), 1)

        dst.write_colormap(
            1,
            ca_module.CLASS_COLORS
        )

        dst.set_band_description(
            1,
            "2030 exploratory CA-Markov LULC scenario"
        )

        dst.update_tags(
            scenario_year="2030",
            initial_year="2025",
            transition_training="2020_to_2025",
            model="One-step neighbourhood CA-Markov",
            validation_status="UNVALIDATED_EXPLORATORY_SCENARIO",
            interpretation=(
                "Scenario only; not a confirmed "
                "future land-cover prediction."
            ),
        )


def write_wetland_screening(
    path,
    lulc,
    valid,
    profile,
):
    """
    Values:
      1   = scenario wetland
      0   = scenario non-wetland
      255 = outside common study-area mask

    A predicted wetland pixel is not a surveyed boundary.
    """

    wetland = np.full(
        lulc.shape,
        255,
        dtype=np.uint8
    )

    wetland[valid] = (
        lulc[valid] == 1
    ).astype(np.uint8)

    updated = profile.copy()

    updated.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=255,
        compress="deflate",
    )

    with rasterio.open(
        path,
        "w",
        **updated
    ) as dst:

        dst.write(wetland, 1)

        dst.write_colormap(
            1,
            {
                0: (220, 220, 220, 255),
                1: (48, 125, 93, 255),
                255: (0, 0, 0, 0),
            }
        )

        dst.set_band_description(
            1,
            "2030 exploratory wetland screening"
        )

        dst.update_tags(
            scenario_year="2030",
            wetland_class="1",
            non_wetland_class="0",
            nodata_class="255",
            validation_status="UNVALIDATED",
            interpretation=(
                "Scenario-derived wetland screening only; "
                "not a confirmed wetland boundary."
            ),
        )

    return wetland


def main():

    print("=" * 70)
    print("PART 3 - FUTURE WETLAND BOUNDARY PROJECTION")
    print("SCRIPT 08 - EXPLORATORY 2030 CA-MARKOV SCENARIO")
    print("=" * 70)

    ca = load_existing_ca_method()

    # Script 07 checks raster geometry and historical
    # Markov inputs without regenerating its outputs.
    maps, valid, profile, earlier_probs = (
        ca.load_inputs()
    )

    if int(valid.sum()) != EXPECTED_PIXELS:
        raise ValueError(
            "Unexpected common study-area size."
        )

    initial = maps[2025]

    print("\nINPUT DATA CHECKS: PASS")
    print("Common pixels:", int(valid.sum()))
    print("Scenario starting year: 2025")
    print("Scenario target year: 2030")
    print("CA method: reused from Script 07")

    # --------------------------------------------------------
    # 1. ESTIMATE 2020 -> 2025 MARKOV TRANSITIONS
    # --------------------------------------------------------

    counts = ca.observed_transition_counts(
        maps[2020],
        maps[2025],
        valid
    )

    probabilities = transition_probabilities(
        counts
    )

    print("\n2020 -> 2025 TRANSITION PROBABILITIES")
    print(
        np.array2string(
            probabilities,
            precision=4,
            suppress_small=True
        )
    )

    # --------------------------------------------------------
    # 2. DOCUMENT TRANSITION INSTABILITY
    # --------------------------------------------------------

    # Total variation between corresponding transition
    # probability rows. This is descriptive evidence,
    # not a forecast-validity test.
    row_variation = 0.5 * np.abs(
        earlier_probs - probabilities
    ).sum(axis=1)

    print("\nHISTORICAL TRANSITION STABILITY CHECK")
    print("2015-2020 versus 2020-2025")
    print("Row total variation: 0=identical, 1=maximally different")

    for i, class_id in enumerate(CLASSES):
        print(
            f"{CLASS_NAMES[int(class_id)]:22s}: "
            f"{row_variation[i]:.4f}"
        )

    # --------------------------------------------------------
    # 3. ESTIMATE 2025 -> 2030 TRANSITION QUOTAS
    # --------------------------------------------------------

    quotas = ca.calculate_transition_quotas(
        initial,
        valid,
        probabilities
    )

    if int(quotas.sum()) != EXPECTED_PIXELS:
        raise ValueError(
            "2030 transition quota accounting failed."
        )

    print("\n2025 -> 2030 MARKOV QUOTAS")
    print("Rows = 2025 classes; columns = 2030 classes")
    print(quotas)

    # --------------------------------------------------------
    # 4. ALLOCATE 2030 SCENARIO USING FROZEN CA METHOD
    # --------------------------------------------------------

    print("\nGenerating neighbourhood-based 2030 scenario...")

    scenario = ca.ca_spatial_allocation(
        initial,
        valid,
        quotas
    )

    if not np.array_equal(
        scenario != 0,
        valid
    ):
        raise ValueError(
            "2030 scenario common mask mismatch."
        )

    if not np.isin(
        scenario[valid],
        CLASSES
    ).all():
        raise ValueError(
            "2030 scenario contains invalid classes."
        )

    actual_quotas = ca.observed_transition_counts(
        initial,
        scenario,
        valid
    )

    if not np.array_equal(
        actual_quotas,
        quotas
    ):
        raise ValueError(
            "2030 scenario does not satisfy Markov quotas."
        )

    print("2030 scenario allocation: COMPLETED")
    print("Transition quota verification: PASS")

    # --------------------------------------------------------
    # 5. SUMMARIZE SCENARIO CHANGES
    # --------------------------------------------------------

    pixel_area_ha = abs(
        profile["transform"].a
        * profile["transform"].e
        - profile["transform"].b
        * profile["transform"].d
    ) / 10000.0

    area_rows = []

    print("\n2025 CLASSIFICATION VS 2030 SCENARIO")

    for class_id in CLASSES:

        current_pixels = int(
            (initial[valid] == class_id).sum()
        )

        future_pixels = int(
            (scenario[valid] == class_id).sum()
        )

        current_ha = (
            current_pixels * pixel_area_ha
        )

        future_ha = (
            future_pixels * pixel_area_ha
        )

        change_ha = future_ha - current_ha

        area_rows.append({
            "class_id": int(class_id),
            "class_name": CLASS_NAMES[int(class_id)],
            "classified_2025_pixels": current_pixels,
            "scenario_2030_pixels": future_pixels,
            "classified_2025_ha": round(
                current_ha, 4
            ),
            "scenario_2030_ha": round(
                future_ha, 4
            ),
            "scenario_change_ha": round(
                change_ha, 4
            ),
        })

        print(
            f"{CLASS_NAMES[int(class_id)]:22s} | "
            f"2025: {current_ha:8.2f} ha | "
            f"2030: {future_ha:8.2f} ha | "
            f"Difference: {change_ha:+8.2f} ha"
        )

    changed = int(
        (
            initial[valid]
            != scenario[valid]
        ).sum()
    )

    wetland_loss = int(
        (
            (initial[valid] == 1)
            & (scenario[valid] != 1)
        ).sum()
    )

    wetland_gain = int(
        (
            (initial[valid] != 1)
            & (scenario[valid] == 1)
        ).sum()
    )

    builtup_to_other = int(
        (
            (initial[valid] == 3)
            & (scenario[valid] != 3)
        ).sum()
    )

    print("\nSCENARIO DIAGNOSTICS")
    print("Predicted changing pixels:", changed)

    print(
        "Scenario wetland loss:",
        f"{wetland_loss * pixel_area_ha:.2f} ha"
    )

    print(
        "Scenario wetland gain:",
        f"{wetland_gain * pixel_area_ha:.2f} ha"
    )

    print(
        "Scenario Built-up -> Other transitions:",
        builtup_to_other,
        "pixels"
    )

    # --------------------------------------------------------
    # 6. SAVE SCENARIO AND IMPORTANT RESEARCH OUTPUTS
    # --------------------------------------------------------

    MAP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    METRIC_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    probability_path = (
        METRIC_DIR
        / "script08_markov_2020_2025_probabilities.csv"
    )

    counts_path = (
        METRIC_DIR
        / "script08_markov_2020_2025_counts.csv"
    )

    quotas_path = (
        METRIC_DIR
        / "script08_scenario_2025_2030_quotas.csv"
    )

    area_path = (
        METRIC_DIR
        / "script08_2030_scenario_area_summary.csv"
    )

    report_path = (
        REPORT_DIR
        / "script08_2030_scenario_metadata.json"
    )

    ca.save_matrix(
        counts,
        counts_path
    )

    ca.save_matrix(
        probabilities,
        probability_path
    )

    ca.save_matrix(
        quotas,
        quotas_path
    )

    pd.DataFrame(
        area_rows
    ).to_csv(
        area_path,
        index=False
    )

    write_lulc(
        LULC_OUTPUT,
        scenario,
        profile,
        ca
    )

    wetland_raster = write_wetland_screening(
        WETLAND_OUTPUT,
        scenario,
        valid,
        profile
    )

    if not np.array_equal(
        wetland_raster[valid] == 1,
        scenario[valid] == 1
    ):
        raise ValueError(
            "Wetland screening raster does not match scenario."
        )

    metadata = {
        "component": (
            "Part 3 - Future Wetland Boundary Projection"
        ),
        "stage": "Exploratory 2030 scenario",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "status": "EXPLORATORY_UNVALIDATED",
        "transition_training_period": "2020-2025",
        "starting_land_cover_year": 2025,
        "scenario_year": 2030,
        "transition_step_years": 5,
        "model": (
            "Six-class Markov with one-step "
            "3x3-neighbourhood CA allocation"
        ),
        "ca_implementation_source": str(
            SOURCE_SCRIPT
        ),
        "ca_seed": SEED,
        "common_valid_pixels": int(valid.sum()),
        "pixel_area_ha": pixel_area_ha,
        "transition_row_total_variation": {
            CLASS_NAMES[int(class_id)]: float(
                row_variation[i]
            )
            for i, class_id in enumerate(CLASSES)
        },
        "scenario_changed_pixels": changed,
        "scenario_wetland_loss_pixels": wetland_loss,
        "scenario_wetland_gain_pixels": wetland_gain,
        "scenario_builtup_to_other_pixels":
            builtup_to_other,
        "markov_quotas_verified": True,
        "common_mask_verified": True,
        "outputs": {
            "lulc_scenario": str(LULC_OUTPUT),
            "wetland_screening": str(
                WETLAND_OUTPUT
            ),
            "area_summary": str(area_path),
        },
        "historical_hindcast_limitations": {
            "ca_accuracy": 0.4583,
            "persistence_accuracy": 0.5248,
            "ca_wetland_f1": 0.2483,
            "persistence_wetland_f1": 0.3118,
            "interpretation": (
                "Previous CA hindcast did not outperform "
                "persistence on the classified 2025 map."
            ),
        },
        "limitations": [
            "This is an exploratory scenario, not a validated forecast.",
            "The historical LULC maps contain classification error.",
            "The 2025 hindcast underperformed persistence.",
            "The 2020-2025 transitions may include false changes.",
            "The Markov stationarity assumption is unverified.",
            "A fixed 3x3 neighbourhood and single five-year "
            "CA allocation step were used.",
            "The scenario does not establish legal or protected "
            "wetland boundaries.",
            "2030 class probabilities and predictive uncertainty "
            "have not been independently estimated.",
        ],
    }

    report_path.write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\nSAVED OUTPUTS")
    print(LULC_OUTPUT)
    print(WETLAND_OUTPUT)
    print(counts_path)
    print(probability_path)
    print(quotas_path)
    print(area_path)
    print(report_path)

    print("\nSCRIPT 08: COMPLETED")
    print("2030 EXPLORATORY SCENARIO: GENERATED")
    print("FORECAST VALIDATION: NOT ESTABLISHED")
    print("WETLAND POLYGON EXTRACTION: NOT YET STARTED")


if __name__ == "__main__":
    main()
