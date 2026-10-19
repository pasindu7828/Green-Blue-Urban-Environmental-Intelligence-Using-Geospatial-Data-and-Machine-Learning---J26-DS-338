
"""
J26-DS-338 - River Corridor Research

STEP 17: EXPERIMENTAL PROJECTION EXPORT

Creates:
- 2026 persistence baseline projection
- 2026 exploratory Random Forest candidate
- GIS data for the future PP1 dashboard
- Forecast-availability QA

No validated erosion or bank-failure claims.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

ANNUAL_FILE = (
    ROOT / "data/processed/morphology"
    / "10_annual_pilot_widths.csv"
)

TRANSECT_FILE = (
    ROOT / "data/processed/transects"
    / "01_transects_50m.gpkg"
)

MODEL_FILE = (
    ROOT / "models"
    / "11_rf_pilot_change.joblib"
)

METRICS_FILE = (
    ROOT / "outputs/reports"
    / "11_ml_benchmark_metrics.csv"
)

METADATA_FILE = (
    ROOT / "outputs/reports"
    / "11_ml_model_metadata.json"
)

DASHBOARD_DIR = ROOT / "outputs/dashboard"
REPORT_DIR = ROOT / "outputs/reports"
FIGURE_DIR = ROOT / "outputs/figures"

for folder in (
    DASHBOARD_DIR,
    REPORT_DIR,
    FIGURE_DIR
):
    folder.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD EXPERIMENTAL INPUTS
# ==========================================

print("=" * 75)
print("STEP 17 - 2026 EXPERIMENTAL PROJECTION")
print("=" * 75)

annual = pd.read_csv(ANNUAL_FILE)

metrics = pd.read_csv(METRICS_FILE)

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:
    metadata = json.load(file)

# Only load locally generated trusted model files.
artifact = joblib.load(MODEL_FILE)

selected_method = metadata[
    "selected_by_validation"
]

if selected_method != "Persistence":
    raise ValueError(
        "This export expects Persistence "
        "to be the validation-selected method."
    )

if metadata["production_approved"] is not False:
    raise ValueError(
        "Unexpected production approval."
    )

if artifact["selected_method"] != selected_method:
    raise ValueError(
        "Model artifact and metadata disagree."
    )


# ==========================================
# 3. SELECT 2025 OBSERVATIONS
# ==========================================

latest = annual[
    annual["year"] == 2025
].copy()

if len(latest) != 271:
    raise ValueError(
        "Expected 271 observations for 2025."
    )

if latest["transect_id"].duplicated().any():
    raise ValueError(
        "Duplicate 2025 transects."
    )

latest["screened"] = (
    (latest["qa_status"] == "PILOT_UNVALIDATED")
    & latest["pilot_width_m"].notna()
    & (latest["pilot_width_m"] >= 60)
    & (latest["pilot_width_m"] <= 300)
)

screened = latest[
    latest["screened"]
].copy()

print(f"Total transects: {len(latest)}")
print(f"Screened 2025 observations: {len(screened)}")

if screened.empty:
    raise ValueError(
        "No screened 2025 observations."
    )


# ==========================================
# 4. PREPARE MODEL FEATURES
# ==========================================

screened["current_width_m"] = (
    screened["pilot_width_m"]
)

screened["reach_number"] = pd.to_numeric(
    screened["reach_id"]
    .astype(str)
    .str.replace(
        "R", "",
        regex=False
    ),
    errors="raise"
)

screened["base_year"] = 2025

features = artifact["feature_columns"]

X = screened[features].copy()

if not np.isfinite(
    X.to_numpy(dtype=float)
).all():
    raise ValueError(
        "Missing model features."
    )


# ==========================================
# 5. GENERATE TWO CANDIDATES
# ==========================================

# Validation-selected persistence baseline.
screened["persistence_2026_m"] = (
    screened["current_width_m"]
)

# Random Forest is NOT the selected method.
rf_model = artifact["model"]

rf_changes = rf_model.predict(X)

screened["rf_candidate_2026_m"] = np.clip(
    screened["current_width_m"].to_numpy()
    + rf_changes,
    0,
    300
)

screened["rf_candidate_change_m"] = (
    screened["rf_candidate_2026_m"]
    - screened["current_width_m"]
)

screened["projection_status"] = (
    "EXPERIMENTAL_BASELINE_ONLY"
)

screened["production_approved"] = False


# ==========================================
# 6. BUILD COMPLETE 271-TRANSECT TABLE
# ==========================================

projection_fields = [
    "transect_id",
    "persistence_2026_m",
    "rf_candidate_2026_m",
    "rf_candidate_change_m",
    "projection_status"
]

table = latest.merge(
    screened[projection_fields],
    on="transect_id",
    how="left",
    validate="one_to_one"
)

table["selected_method"] = np.where(
    table["screened"],
    "Persistence",
    "NO_PROJECTION"
)

table["target_year"] = 2026

table["physical_erosion_prediction"] = False
table["production_approved"] = False

table["projection_status"] = (
    table["projection_status"].fillna(
        "NO_PROJECTION_INPUT_FAILED_QA"
    )
)


# ==========================================
# 7. EXPORT CSV
# ==========================================

csv_file = (
    DASHBOARD_DIR / "12_projection.csv"
)

table.to_csv(
    csv_file,
    index=False
)


# ==========================================
# 8. EXPORT GIS GEOJSON
# ==========================================

transects = gpd.read_file(
    TRANSECT_FILE,
    layer="river_transects"
)

if len(transects) != 271:
    raise ValueError(
        "Transect GIS record mismatch."
    )

attributes = table.drop(
    columns=[
        "reach_id",
        "chainage_m"
    ]
)

gis = transects.merge(
    attributes,
    on="transect_id",
    how="left",
    validate="one_to_one"
)

if len(gis) != len(table):
    raise ValueError(
        "GIS join row count mismatch."
    )

if gis["target_year"].isna().any():
    raise ValueError(
        "Missing joined projection rows."
    )

# Frontend-friendly geographic coordinates.
gis = gis.to_crs("EPSG:4326")

geojson_file = (
    DASHBOARD_DIR / "12_projection.geojson"
)

gis.to_file(
    geojson_file,
    driver="GeoJSON"
)


# ==========================================
# 9. EVALUATION COVERAGE AUDIT
# ==========================================

pairs = pd.read_csv(
    ROOT / "data/processed/morphology"
    / "10_next_year_pilot_pairs.csv"
)

coverage_records = []

for target_year in [2024, 2025]:

    current_year = target_year - 1

    current_count = int(
        annual.loc[
            (annual["year"] == current_year)
            & (
                annual["qa_status"]
                == "PILOT_UNVALIDATED"
            ),
            "transect_id"
        ].nunique()
    )

    paired_count = int(
        pairs.loc[
            pairs["target_year"] == target_year,
            "transect_id"
        ].nunique()
    )

    coverage_pct = (
        100 * paired_count / current_count
        if current_count else 0
    )

    coverage_records.append({
        "target_year": target_year,
        "current_year": current_year,
        "current_screened_transects": current_count,
        "evaluated_pairs": paired_count,
        "evaluation_coverage_pct": round(
            coverage_pct, 2
        ),
        "note": (
            "EVALUATION_REQUIRES_NEXT_YEAR_QA"
        )
    })

coverage_records.append({
    "target_year": 2026,
    "current_year": 2025,
    "current_screened_transects": len(screened),
    "evaluated_pairs": 0,
    "evaluation_coverage_pct": np.nan,
    "note": "FUTURE_OUTCOME_NOT_OBSERVED"
})

qa = pd.DataFrame(coverage_records)

qa_file = (
    REPORT_DIR / "12_projection_qa.csv"
)

qa.to_csv(
    qa_file,
    index=False
)


# ==========================================
# 10. DASHBOARD SUMMARY JSON
# ==========================================

test_metrics = metrics[
    (metrics["split"] == "test_2024_2025")
    & (metrics["method"] == selected_method)
]

if len(test_metrics) != 1:
    raise ValueError(
        "Missing selected test metric."
    )

summary = {
    "component": "River Corridor",
    "observation_year": 2025,
    "projection_year": 2026,
    "total_transects": int(len(table)),
    "projected_transects": int(len(screened)),
    "unavailable_transects": int(
        len(table) - len(screened)
    ),
    "selected_method": selected_method,
    "selected_by": "2023 validation MAE",
    "test_period": "2024-2025",
    "selected_test_mae_m": float(
        test_metrics.iloc[0]["mae_m"]
    ),
    "test_metric_scope": (
        "Pairs with acceptable observations "
        "in both years only"
    ),
    "prediction_type": (
        "EXPERIMENTAL_WATER_WIDTH_PROXY"
    ),
    "independent_validation": False,
    "physical_erosion_prediction": False,
    "production_approved": False,
    "limitations": [
        "Landsat 30 m spatial resolution",
        "No surveyed riverbank ground truth",
        "Year-dependent observation availability",
        "Spatially correlated transects",
        "Next-year QA conditions historical evaluation",
        "2026 baseline assumes no detected width change"
    ]
}

summary_file = (
    DASHBOARD_DIR / "12_projection_summary.json"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=2)


# ==========================================
# 11. PREPARE RESEARCH FIGURE
# ==========================================

ordered = screened.sort_values(
    "chainage_m"
)

fig, ax = plt.subplots(
    figsize=(12, 5)
)

ax.plot(
    ordered["chainage_m"] / 1000,
    ordered["current_width_m"],
    label="2025 pilot observed proxy",
    linewidth=1.5
)

ax.plot(
    ordered["chainage_m"] / 1000,
    ordered["rf_candidate_2026_m"],
    label="2026 RF experimental candidate",
    linestyle="--",
    linewidth=1
)

ax.set_title(
    "2026 Experimental Water-Width Projection"
)

ax.set_xlabel(
    "River chainage (km)"
)

ax.set_ylabel(
    "Classified water-width proxy (m)"
)

ax.legend()
ax.grid(alpha=0.2)

fig.tight_layout()

figure_file = (
    FIGURE_DIR / "12_projection.png"
)

fig.savefig(
    figure_file,
    dpi=150
)

plt.close(fig)


# ==========================================
# 12. FINAL REPORT
# ==========================================

print("\n" + "=" * 75)
print("STEP 17 - FINAL SUMMARY")
print("=" * 75)

print(f"Selected method: {selected_method}")
print(f"2025 input transects: {len(table)}")
print(f"2026 projected transects: {len(screened)}")
print(
    "No projection due to input QA: "
    f"{len(table) - len(screened)}"
)

print("\nHistorical evaluation coverage:")
print(qa.to_string(index=False))

print("\nSaved:")
print(csv_file)
print(geojson_file)
print(summary_file)
print(qa_file)
print(figure_file)

print("\nIMPORTANT:")
print("2026 results are experimental projections.")
print("Persistence is the selected baseline.")
print("RF is an unselected experimental candidate.")
print("No actual 2026 water widths were measured.")
print("No validated erosion or risk prediction.")

print("\nSTEP 17 FINISHED")
