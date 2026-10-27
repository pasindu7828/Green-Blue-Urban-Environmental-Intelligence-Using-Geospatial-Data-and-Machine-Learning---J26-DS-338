from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT_PATH.parents[1]
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
REPORT_DIR = OUTPUT_ROOT / "reports"
PREDICTION_DIR = OUTPUT_ROOT / "predictions"
TRAJECTORY_DIR = OUTPUT_ROOT / "trajectory"
MAP_DIR = OUTPUT_ROOT / "maps"
MODEL_DIR = EXPERIMENT_ROOT / "models"

EXPECTED_GRIDS = 2194
FORECAST_HORIZON = 5
EXPECTED_CLASSES = {
    "Persistent Cool",
    "Diminishing",
    "Emerging",
    "Intensifying",
    "Persistent Hot",
}
CENTER_TOL = 1e-10
MEAN_IDENTITY_TOL = 1e-10


def section(title: str) -> None:
    print("\n" + "=" * 125)
    print(title)
    print("=" * 125)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")


def all_pass(path: Path) -> tuple[bool, str]:
    df = pd.read_csv(path)
    if "status" not in df.columns:
        return False, "missing status column"
    accepted = {"PASS", "NOT_APPLICABLE"}
    bad = df[~df["status"].astype(str).str.upper().isin(accepted)]
    return (len(bad) == 0, f"rows={len(df)}, fail_rows={len(bad)}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_check(rows: list[dict], name: str, ok: bool, details: str) -> None:
    status = "PASS" if ok else "FAIL"
    rows.append({"check": name, "status": status, "details": details})
    print(f"{status} | {name}: {details}")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    section("DYNAMIC 5-YEAR FORECAST - STEP 06 FINAL SERVICE AUDIT")

    validated_history = OUTPUT_ROOT / "01_validated_history.csv"
    trajectory_csv = TRAJECTORY_DIR / "02_dynamic_trajectory_classes.csv"
    trajectory_geojson = TRAJECTORY_DIR / "02_dynamic_trajectory_wgs84.geojson"

    previous_audits = {
        "step02_trajectory": REPORT_DIR / "02_dynamic_trajectory_audit.csv",
        "step03_model_update": REPORT_DIR / "03_dynamic_model_update_audit.csv",
        "step04_future_inputs": REPORT_DIR / "04_dynamic_future_input_projection_audit.csv",
        "step05_forecast_generation": REPORT_DIR / "05_dynamic_forecast_audit.csv",
    }

    require(validated_history)
    require(trajectory_csv)
    require(trajectory_geojson)
    for path in previous_audits.values():
        require(path)

    history = pd.read_csv(
        validated_history,
        dtype={"grid_id": str, "area_name": str},
        low_memory=False,
    )
    history["year"] = pd.to_numeric(history["year"], errors="raise").astype(int)

    observed_years = sorted(history["year"].unique().tolist())
    latest_year = int(observed_years[-1])
    future_years = list(range(latest_year + 1, latest_year + FORECAST_HORIZON + 1))
    start_future, end_future = future_years[0], future_years[-1]

    dynamic_model = MODEL_DIR / f"03_dynamic_lightgbm_anomaly_trained_2018_{latest_year}.joblib"
    future_base = PREDICTION_DIR / f"04_future_base_predictors_{start_future}_{end_future}.csv"
    annual_levels = PREDICTION_DIR / f"04_future_annual_thermal_level_{start_future}_{end_future}.csv"
    future_features = PREDICTION_DIR / f"05_future_model_features_{start_future}_{end_future}.csv"
    future_predictions = PREDICTION_DIR / f"05_dynamic_grid_lst_predictions_{start_future}_{end_future}.csv"
    future_summary = PREDICTION_DIR / f"05_dynamic_future_year_summary_{start_future}_{end_future}.csv"
    combined_map = MAP_DIR / f"05_dynamic_future_heat_{start_future}_{end_future}_wgs84.geojson"
    yearly_maps = [MAP_DIR / f"05_dynamic_future_heat_{year}_wgs84.geojson" for year in future_years]

    runtime_files = [
        dynamic_model,
        future_base,
        annual_levels,
        future_features,
        future_predictions,
        future_summary,
        combined_map,
        *yearly_maps,
    ]
    for path in runtime_files:
        require(path)

    print(f"Observed period       : {observed_years[0]}-{latest_year}")
    print(f"Dynamic forecast      : {start_future}-{end_future}")
    print(f"Expected grids/year   : {EXPECTED_GRIDS:,}")

    audit_rows: list[dict] = []

    section("1. VERIFY PREVIOUS STEP AUDITS")
    add_check(
        audit_rows,
        "step01_validated_history_available",
        True,
        str(validated_history),
    )
    for label, path in previous_audits.items():
        ok, details = all_pass(path)
        add_check(audit_rows, label, ok, details)

    section("2. VALIDATE OBSERVED HISTORY + TRAJECTORY")
    add_check(
        audit_rows,
        "observed_years_contiguous",
        observed_years == list(range(observed_years[0], latest_year + 1)),
        f"years={observed_years}",
    )
    add_check(
        audit_rows,
        "observed_unique_grid_count",
        history["grid_id"].nunique() == EXPECTED_GRIDS,
        f"actual={history['grid_id'].nunique()}, expected={EXPECTED_GRIDS}",
    )
    hist_per_year = history.groupby("year")["grid_id"].nunique()
    add_check(
        audit_rows,
        "observed_balanced_grid_panel",
        hist_per_year.min() == EXPECTED_GRIDS and hist_per_year.max() == EXPECTED_GRIDS,
        f"min={int(hist_per_year.min())}, max={int(hist_per_year.max())}",
    )

    trajectory = pd.read_csv(trajectory_csv, dtype={"grid_id": str}, low_memory=False)
    actual_classes = set(trajectory["trajectory_class"].dropna().astype(str).unique().tolist())
    add_check(
        audit_rows,
        "trajectory_grid_count",
        len(trajectory) == EXPECTED_GRIDS,
        f"actual={len(trajectory)}, expected={EXPECTED_GRIDS}",
    )
    add_check(
        audit_rows,
        "trajectory_class_contract",
        actual_classes.issubset(EXPECTED_CLASSES),
        f"classes={sorted(actual_classes)}",
    )

    section("3. VALIDATE FIVE-YEAR FORECAST CONTRACT")
    pred = pd.read_csv(
        future_predictions,
        dtype={"grid_id": str, "area_name": str},
        low_memory=False,
    )
    pred["year"] = pd.to_numeric(pred["year"], errors="raise").astype(int)

    actual_future_years = sorted(pred["year"].unique().tolist())
    expected_rows = EXPECTED_GRIDS * FORECAST_HORIZON

    add_check(
        audit_rows,
        "forecast_year_window",
        actual_future_years == future_years,
        f"actual={actual_future_years}, expected={future_years}",
    )
    add_check(
        audit_rows,
        "forecast_row_count",
        len(pred) == expected_rows,
        f"actual={len(pred)}, expected={expected_rows}",
    )
    pred_per_year = pred.groupby("year")["grid_id"].nunique()
    add_check(
        audit_rows,
        "forecast_balanced_grid_panel",
        pred_per_year.min() == EXPECTED_GRIDS and pred_per_year.max() == EXPECTED_GRIDS,
        f"min={int(pred_per_year.min())}, max={int(pred_per_year.max())}",
    )
    duplicate_count = int(pred.duplicated(subset=["grid_id", "year"]).sum())
    add_check(
        audit_rows,
        "forecast_unique_grid_year",
        duplicate_count == 0,
        f"duplicate_rows={duplicate_count}",
    )

    required_cols = [
        "grid_id",
        "year",
        "area_name",
        "forecast_origin_year",
        "horizon_years",
        "raw_spatial_anomaly_prediction_c",
        "centered_spatial_anomaly_c",
        "forecast_annual_mean_lst_c",
        "predicted_lst_c",
    ]
    missing_cols = [c for c in required_cols if c not in pred.columns]
    add_check(
        audit_rows,
        "forecast_schema",
        len(missing_cols) == 0,
        f"missing={missing_cols}",
    )

    numeric_cols = [
        "raw_spatial_anomaly_prediction_c",
        "centered_spatial_anomaly_c",
        "forecast_annual_mean_lst_c",
        "predicted_lst_c",
    ]
    finite_ok = len(missing_cols) == 0 and np.isfinite(pred[numeric_cols].to_numpy(dtype=float)).all()
    add_check(
        audit_rows,
        "forecast_values_finite",
        bool(finite_ok),
        f"rows={len(pred)}",
    )

    max_centered_mean = float(
        pred.groupby("year")["centered_spatial_anomaly_c"].mean().abs().max()
    )
    add_check(
        audit_rows,
        "centered_anomaly_zero_mean_each_year",
        max_centered_mean <= CENTER_TOL,
        f"max_abs_year_mean={max_centered_mean:.12e}",
    )

    annual_identity = pred.groupby("year").agg(
        annual_level=("forecast_annual_mean_lst_c", "first"),
        mean_prediction=("predicted_lst_c", "mean"),
    )
    max_identity_error = float(
        (annual_identity["annual_level"] - annual_identity["mean_prediction"]).abs().max()
    )
    add_check(
        audit_rows,
        "annual_level_plus_centered_anomaly_identity",
        max_identity_error <= MEAN_IDENTITY_TOL,
        f"max_abs_error={max_identity_error:.12e}",
    )

    section("4. VALIDATE MAP + MODEL RUNTIME ARTIFACTS")
    add_check(
        audit_rows,
        "versioned_model_available",
        dynamic_model.exists(),
        str(dynamic_model),
    )
    add_check(
        audit_rows,
        "five_yearly_geojson_layers",
        all(path.exists() for path in yearly_maps),
        f"saved={sum(path.exists() for path in yearly_maps)}, expected=5",
    )
    add_check(
        audit_rows,
        "combined_forecast_geojson",
        combined_map.exists(),
        str(combined_map),
    )

    section("5. FINAL READINESS DECISION")
    final_audit = REPORT_DIR / "06_final_dynamic_service_audit.csv"
    final_manifest = REPORT_DIR / "06_dynamic_service_manifest.json"
    final_hashes = REPORT_DIR / "06_dynamic_service_hashes.csv"

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(final_audit, index=False)
    failed = audit_df[audit_df["status"] == "FAIL"]
    module_status = "READY_FOR_API_INTEGRATION" if failed.empty else "NOT_READY"

    hash_targets = [
        validated_history,
        trajectory_csv,
        trajectory_geojson,
        dynamic_model,
        future_base,
        annual_levels,
        future_features,
        future_predictions,
        future_summary,
        combined_map,
        *yearly_maps,
    ]
    hash_rows = [
        {
            "file": str(path.relative_to(EXPERIMENT_ROOT)),
            "sha256": sha256(path),
        }
        for path in hash_targets
    ]
    pd.DataFrame(hash_rows).to_csv(final_hashes, index=False)

    manifest = {
        "module": "Rolling Five-Year Urban Heat Forecasting Service",
        "status": module_status,
        "observed_period": f"{observed_years[0]}-{latest_year}",
        "latest_observed_year": latest_year,
        "forecast_years": future_years,
        "forecast_horizon_years": FORECAST_HORIZON,
        "expected_grids_per_year": EXPECTED_GRIDS,
        "trajectory_classes": sorted(actual_classes),
        "model_artifact": str(dynamic_model),
        "validated_history": str(validated_history),
        "trajectory_csv": str(trajectory_csv),
        "trajectory_geojson": str(trajectory_geojson),
        "future_base_predictors": str(future_base),
        "future_annual_levels": str(annual_levels),
        "future_model_features": str(future_features),
        "future_predictions": str(future_predictions),
        "future_year_summary": str(future_summary),
        "combined_forecast_geojson": str(combined_map),
        "yearly_forecast_geojson": [str(path) for path in yearly_maps],
        "forecast_equation": "forecast_annual_mean_lst_c + centered_spatial_anomaly_c",
        "trajectory_scope": "Observed-history relative spatial heat trajectory",
        "future_pseudo_observations_used_for_training": False,
        "failed_checks": int(len(failed)),
    }
    with final_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(audit_df.to_string(index=False))
    print(f"\nFinal audit    : {final_audit}")
    print(f"Final manifest : {final_manifest}")
    print(f"Final hashes   : {final_hashes}")
    print(f"\nMODULE STATUS  : {module_status}")

    if not failed.empty:
        raise RuntimeError(f"Final audit failed: {len(failed)} check(s).")

    section("STEP 06 COMPLETED SUCCESSFULLY")
    print("Dynamic rolling five-year forecast module is internally consistent.")
    print("NEXT: connect upload, update, forecast and map outputs to FastAPI.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "!" * 125)
        print("STEP 06 FAILED")
        print("!" * 125)
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)
