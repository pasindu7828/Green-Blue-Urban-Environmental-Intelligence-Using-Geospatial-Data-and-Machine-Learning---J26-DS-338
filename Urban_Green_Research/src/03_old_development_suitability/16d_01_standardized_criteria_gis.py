
"""
STEP 16D-A — PROVISIONAL STANDARDIZED CRITERIA GIS

Prepare a reproducible, spatially joined scoring-ready dataset for
independent general urban development suitability (9,240 Kaduwela cells).

C1, C2, C4: proposed core indicators, percentile-capped (P05-P95).
C3: both opposing interpretations retained as alternatives.
C5: raw elevation retained only; NOT converted to suitability score.
Percentile rank versions: sensitivity alternatives.

No expert judgments, AHP weights, combined suitability, planning permission
or conservation exclusions are created by this script.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "outputs" / "reports"
RAW = ROOT / "data/processed/development_suitability/raw_criteria/16b_raw_suitability_criteria_100m.csv"
GRID = ROOT / "data/processed/development_suitability/raw_criteria/16b_raw_suitability_criteria_100m.gpkg"
F_SCORES = REPORTS / "16c_f_candidate_scoring_scenarios_100m.csv"
F_SUMMARY = REPORTS / "16c_f_scoring_method_summary.json"
G_SUMMARY = REPORTS / "16c_g_scoring_sensitivity_summary.json"
J_SUMMARY = REPORTS / "16c_j_zoning_reconciliation_summary.json"
K_SUMMARY = REPORTS / "16c_k_criterion_robustness_summary.json"
L_SUMMARY = REPORTS / "16c_l_expert_review_summary.json"
M_SUMMARY = REPORTS / "16c_m_grid_zoning_screening_summary.json"

OUT_CSV = REPORTS / "16d_a_provisional_standardized_criteria_100m.csv"
OUT_GPKG = REPORTS / "16d_a_provisional_standardized_criteria_100m.gpkg"
OUT_PARAMS = REPORTS / "16d_a_normalization_parameter_registry.csv"
OUT_QA = REPORTS / "16d_a_standardized_criteria_qa.csv"
OUT_SUMMARY = REPORTS / "16d_a_standardized_criteria_summary.json"

C = {
    "C1": "c1_major_road_distance_m",
    "C2": "c2_road_density_500m_km_per_km2",
    "C3": "c3_context_building_pct_500m",
    "C4": "c4_slope_mean_deg",
    "C5": "c5_elevation_mean_m",
}

# (new output name, criterion, direction, Step 16C-F reference name)
SCORE_CONFIG = [
    ("s_c1_road_pc", "C1", "DECREASING", "c1_pctclip_decreasing"),
    ("s_c2_road_density_pc", "C2", "INCREASING", "c2_pctclip_increasing"),
    ("alt_c3_high_building_pc", "C3", "INCREASING", "c3_pctclip_increasing"),
    ("alt_c3_low_building_pc", "C3", "DECREASING", "c3_pctclip_decreasing"),
    ("s_c4_slope_pc", "C4", "DECREASING", "c4_pctclip_decreasing"),
    ("sens_c1_rank", "C1", "DECREASING", "c1_rank_decreasing"),
    ("sens_c2_rank", "C2", "INCREASING", "c2_rank_increasing"),
    ("sens_c3_high_rank", "C3", "INCREASING", "c3_rank_increasing"),
    ("sens_c3_low_rank", "C3", "DECREASING", "c3_rank_decreasing"),
    ("sens_c4_rank", "C4", "DECREASING", "c4_rank_decreasing"),
]


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_load(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def calculate_score(values, direction, method):
    if method == "pc":
        lo, hi = float(values.quantile(0.05)), float(values.quantile(0.95))
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            raise ValueError("Invalid P05/P95 limits")
        score = (values.clip(lower=lo, upper=hi) - lo) / (hi - lo)
    else:
        score = (values.rank(method="average") - 1) / (len(values) - 1)
    if direction == "DECREASING":
        score = 1 - score
    return score.clip(0, 1)


def main():
    print("=" * 88)
    print("STEP 16D-A - PROVISIONAL STANDARDIZED CRITERIA GIS")
    print("=" * 88)
    required = [RAW, GRID, F_SCORES, F_SUMMARY, G_SUMMARY, J_SUMMARY,
                K_SUMMARY, L_SUMMARY, M_SUMMARY]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    REPORTS.mkdir(parents=True, exist_ok=True)

    summaries = {name: json_load(path) for name, path in [
        ("F", F_SUMMARY), ("G", G_SUMMARY), ("J", J_SUMMARY),
        ("K", K_SUMMARY), ("L", L_SUMMARY), ("M", M_SUMMARY),
    ]}
    original_hashes = {str(p): checksum(p) for p in required}
    raw_hash = original_hashes[str(RAW)]
    f_hash = original_hashes[str(F_SCORES)]

    qa = []
    def check(name, passed, detail=""):
        qa.append({"check": name, "passed": bool(passed), "detail": str(detail)})
    def checkpoint():
        if not all(r["passed"] for r in qa):
            pd.DataFrame(qa).to_csv(OUT_QA, index=False)
            for r in qa:
                if not r["passed"]:
                    print("REVIEW:", r["check"], r["detail"])
            raise RuntimeError("Step 16D-A validation failed; inspect QA CSV")

    check("All prerequisite step QA statuses PASS",
          all(x.get("qa_status") == "PASS" for x in summaries.values()))
    check("Previous experiments are not final model scores",
          summaries["F"].get("methodology_status") == "EXPERIMENTAL"
          and summaries["G"].get("methodology_status") == "EXPERIMENTAL"
          and summaries["K"].get("methodology_status") == "DIAGNOSTIC_ONLY_FINAL_CRITERIA_PENDING")
    check("Expert judgments remain uncollected by preparation script",
          summaries["L"].get("expert_judgments_collected_by_script") is False
          and summaries["L"].get("weights_assigned") is False)
    check("Frozen raw dataset matches previous audits",
          all(v == raw_hash for v in [
              summaries["F"].get("input_sha256"),
              summaries["G"].get("input_sha256"),
              summaries["J"].get("frozen_input_sha256"),
              summaries["K"].get("frozen_raw_sha256"),
              summaries["L"].get("frozen_input_sha256"),
              summaries["M"].get("raw_csv_sha256"),
          ]))
    check("Previously assessed candidate-score CSV unchanged",
          summaries["G"].get("scoring_input_sha256") == f_hash
          and summaries["K"].get("candidate_scoring_sha256") == f_hash)
    checkpoint()

    raw = pd.read_csv(RAW, dtype={"grid_id": "string"})
    past = pd.read_csv(F_SCORES, dtype={"grid_id": "string"})
    grid = gpd.read_file(GRID, layer="raw_suitability_criteria_100m")
    grid["grid_id"] = grid["grid_id"].astype("string").str.strip()
    raw["grid_id"] = raw["grid_id"].str.strip()
    past["grid_id"] = past["grid_id"].str.strip()

    check("All three inputs contain 9240 unique grid IDs",
          all(len(df) == 9240 and df["grid_id"].is_unique and
              df["grid_id"].notna().all() for df in [raw, past, grid]))
    check("All grid ID sets match",
          set(raw["grid_id"]) == set(past["grid_id"]) == set(grid["grid_id"]))
    check("Five frozen raw criteria available", set(C.values()).issubset(raw.columns))
    check("Previous ten scored columns available",
          all(old in past.columns for _, _, _, old in SCORE_CONFIG))
    check("Research geometry valid, projected EPSG:32644",
          grid.crs is not None and grid.crs.to_epsg() == 32644
          and grid.geometry.notna().all() and
          (~grid.geometry.is_empty).all() and grid.geometry.is_valid.all())
    checkpoint()

    raw_values = raw[list(C.values())].apply(pd.to_numeric, errors="coerce")
    check("All original five criteria are finite",
          np.isfinite(raw_values.to_numpy(dtype=float)).all())
    check("All four standardized criteria are nonconstant",
          (raw_values[[C[x] for x in ["C1", "C2", "C3", "C4"]]].nunique() > 1).all())
    checkpoint()

    past_aligned = past.set_index("grid_id").reindex(raw["grid_id"])
    out = raw[["grid_id"]].copy()
    for field in C.values():
        out[field] = raw_values[field].to_numpy(dtype=float)

    parameters = []
    for code, field in C.items():
        v = raw_values[field]
        params = {
            "criterion": code, "source_variable": field,
            "raw_min": float(v.min()), "raw_max": float(v.max()),
            "raw_p05": float(v.quantile(0.05)),
            "raw_p95": float(v.quantile(0.95)),
            "use_status": "RAW_CONTEXT_ONLY" if code == "C5"
                          else ("DIRECTION_UNDECIDED_TWO_ALTERNATIVES" if code == "C3"
                                else "PROVISIONAL_CORE"),
            "primary_candidate_method": "NONE" if code == "C5" else "PERCENTILE_CAPPED_P05_P95",
            "sensitivity_method": "NONE" if code == "C5" else "PERCENTILE_RANK",
            "final_approved": False,
            "note": "Percentiles are study-specific statistical anchors, not UDA legal thresholds",
        }
        parameters.append(params)

    comparisons = []
    for new_name, code, direction, reference_name in SCORE_CONFIG:
        method = "rank" if "_rank" in new_name else "pc"
        scores = calculate_score(raw_values[C[code]], direction, method)
        out[new_name] = scores.to_numpy(dtype=float)
        prior = pd.to_numeric(past_aligned[reference_name], errors="coerce").to_numpy(dtype=float)
        diff = np.abs(scores.to_numpy(dtype=float) - prior)
        comparisons.append((new_name, float(np.nanmax(diff))))

    scored_names = [name for name, _, _, _ in SCORE_CONFIG]
    check("Ten provisional candidate score columns produced", len(scored_names) == 10
          and all(col in out.columns for col in scored_names))
    check("All provisional score values finite and within 0-1",
          np.isfinite(out[scored_names].to_numpy(dtype=float)).all()
          and out[scored_names].ge(0).all().all()
          and out[scored_names].le(1).all().all())
    check("Scores reproduce Step 16C-F results", all(np.isfinite(x) and x < 1e-10
          for _, x in comparisons), f"Largest deviation: {max(x for _, x in comparisons):.3g}")
    check("Opposite C3 interpretations remain complementary",
          np.allclose(out["alt_c3_high_building_pc"] + out["alt_c3_low_building_pc"], 1)
          and np.allclose(out["sens_c3_high_rank"] + out["sens_c3_low_rank"], 1))
    check("C5 elevation is not standardized", not any(name.startswith("s_c5") or
          name.startswith("alt_c5") or name.startswith("sens_c5") for name in scored_names))
    checkpoint()

    # Join polygon by ID rather than positional order; preserve boundary-clipped geometry.
    geom = grid.set_index("grid_id").geometry.reindex(raw["grid_id"])
    check("Aligned grid has no missing polygon", geom.notna().all())
    checkpoint()
    spatial = gpd.GeoDataFrame(out, geometry=gpd.GeoSeries(geom.to_numpy(), crs=grid.crs), crs=grid.crs)
    spatial["cell_area_m2"] = spatial.geometry.area.to_numpy(dtype=float)
    spatial["fraction_of_100m_square"] = spatial["cell_area_m2"] / 10000.0
    spatial["is_partial_boundary_cell"] = spatial["cell_area_m2"] < 9999.9

    total_area_ha = float(spatial["cell_area_m2"].sum() / 10000)
    check("Original grid area matches reconciled KMC boundary",
          abs(total_area_ha - float(summaries["J"]["kmc_area_ha"])) <= 0.05,
          f"Total area {total_area_ha:.4f} ha")
    check("All grid cells have positive polygon area", (spatial["cell_area_m2"] > 0).all())
    check("Source files remain byte-for-byte unchanged",
          all(checksum(p) == original_hashes[str(p)] for p in required))
    checkpoint()

    # Test a staging GeoPackage before publishing it.
    temp_path = OUT_GPKG.with_name(OUT_GPKG.stem + "_staging.gpkg")
    if temp_path.exists():
        temp_path.unlink()
    spatial.to_file(temp_path, layer="provisional_standardized_criteria", driver="GPKG", index=False)
    saved = gpd.read_file(temp_path, layer="provisional_standardized_criteria")
    check("GIS round-trip preserves all grid IDs and geometry",
          len(saved) == 9240 and saved["grid_id"].is_unique
          and set(saved["grid_id"].astype(str)) == set(raw["grid_id"])
          and saved.geometry.notna().all() and saved.geometry.is_valid.all())
    check("GIS round-trip preserves the core scores",
          np.allclose(saved["s_c1_road_pc"].to_numpy(dtype=float), spatial["s_c1_road_pc"].to_numpy(dtype=float), atol=1e-9)
          and np.allclose(saved["s_c2_road_density_pc"].to_numpy(dtype=float), spatial["s_c2_road_density_pc"].to_numpy(dtype=float), atol=1e-9)
          and np.allclose(saved["s_c4_slope_pc"].to_numpy(dtype=float), spatial["s_c4_slope_pc"].to_numpy(dtype=float), atol=1e-9))
    checkpoint()
    del saved

    os.replace(temp_path, OUT_GPKG)
    spatial.drop(columns="geometry").to_csv(OUT_CSV, index=False)
    pd.DataFrame(parameters).to_csv(OUT_PARAMS, index=False)
    pd.DataFrame(qa).to_csv(OUT_QA, index=False)

    summary = {
        "step": "16D-A", "qa_status": "PASS", "status": "PROVISIONAL_STANDARDIZATION_ONLY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "grid_cells": len(spatial), "crs": str(spatial.crs),
        "grid_area_ha": total_area_ha, "raw_sha256": raw_hash,
        "input_gpkg_sha256": original_hashes[str(GRID)],
        "c1_c2_c4": "PROPOSED_CORE_NOT_YET_FINAL",
        "c3": "TWO_UNSELECTED_ALTERNATIVE_DIRECTIONS",
        "c5": "RAW_ONLY_NO_CITYWIDE_SUITABILITY_DIRECTION_ESTABLISHED",
        "primary_candidate": "P05_P95_CAPPED_RELATIVE_TO_KADUWELA_GRID_DISTRIBUTION",
        "secondary_candidate": "RANK_BASED_SENSITIVITY",
        "score_columns": scored_names,
        "highest_difference_from_step_16c_f": max(x for _, x in comparisons),
        "partial_boundary_cells": int(spatial["is_partial_boundary_cell"].sum()),
        "cells_below_5000_m2": int((spatial["cell_area_m2"] < 5000).sum()),
        "min_cell_area_m2": float(spatial["cell_area_m2"].min()),
        "qa_checks": len(qa), "qa_passed": int(pd.DataFrame(qa)["passed"].sum()),
        "interpretation": [
            "No final criterion weights or combined suitability scores computed",
            "Percentile 5/95 bounds are study-specific and not legal planning restrictions",
            "C3 direction and inclusion require genuine justification; both alternatives retained",
            "C5 elevation retained as raw context, never assigned a universal suitability score",
            "Very small boundary fragments are retained and explicitly identified, not declared buildable",
            "UDA provisional zoning remains independent of criterion scoring",
            "Expert questionnaires have been prepared but expert judgments are not claimed",
        ],
        "outputs": [p.name for p in [OUT_CSV, OUT_GPKG, OUT_PARAMS, OUT_QA, OUT_SUMMARY]],
    }
    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== SCORING-READY SPATIAL DATA ===")
    print("Grids:", len(spatial), "| CRS:", spatial.crs)
    print("Clipped KMC area (ha):", round(total_area_ha, 4))
    print("Partial boundary cells:", summary["partial_boundary_cells"])
    print("Cells smaller than 5000 m2:", summary["cells_below_5000_m2"])
    print("Minimum area (m2):", round(summary["min_cell_area_m2"], 4))
    print("C1/C2/C4: provisional core standardized indicators")
    print("C3: two unselected alternative directions | C5: raw only")
    print("Max difference vs Step 16C-F:", f"{summary['highest_difference_from_step_16c_f']:.3g}")
    print("\n=== QA RESULTS ===")
    for record in qa:
        print("PASS:" if record["passed"] else "REVIEW:", record["check"])
    print(f"\nOverall QA: PASS ({len(qa)}/{len(qa)})")
    print("\n=== SAVED REPORTS ===")
    for path in [OUT_CSV, OUT_GPKG, OUT_PARAMS, OUT_QA, OUT_SUMMARY]:
        print(path.relative_to(ROOT))
    print("\nNo AHP weights, composite score or zoning exclusions calculated.")
    print("=" * 88)


if __name__ == "__main__":
    main()
