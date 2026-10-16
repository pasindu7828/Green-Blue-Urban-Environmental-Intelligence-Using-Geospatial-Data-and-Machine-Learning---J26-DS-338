from pathlib import Path
import hashlib, json
import pandas as pd

# STEP 31A - PRIORITY CANDIDATE INDICATOR INVENTORY
# Inventory only: no final selection, no CRITIC, no TOPSIS, no retraining.

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DATA = ROOT / "data" / "processed"
MET = OUT / "metrics"
FOR = OUT / "forecasting"
SCE = OUT / "scenarios"
REP = OUT / "reports"
PRI = OUT / "priority"
PRI.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)

def h(path):
    x = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            x.update(b)
    return x.hexdigest()

def req(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path

def audit_pass(path):
    df = pd.read_csv(path)
    c = next((c for c in df.columns if c.lower()=="status"), None)
    if c is None:
        raise RuntimeError(f"No status column in {path.name}")
    ok = df[c].astype(str).str.upper().eq("PASS").all()
    if not ok:
        raise RuntimeError(f"Audit not fully PASS: {path.name}")

print("="*120)
print("STEP 31A - PRIORITY CANDIDATE INDICATOR INVENTORY")
print("="*120)

# ------------------------------------------------------------------
# 1. Prerequisite audits
# ------------------------------------------------------------------
audits = [
    REP/"23_mann_kendall_sens_slope_audit.csv",
    REP/"23b_heat_trajectory_audit.csv",
    REP/"28_future_prediction_audit.csv",
    REP/"29a_v2_persistent_green_scenario_audit.csv",
    REP/"29b_built_up_intensity_scenario_audit.csv",
    REP/"29c_combined_scenario_audit.csv",
    REP/"29d_blue_space_scenario_audit.csv",
    REP/"30_uncertainty_audit.csv",
]
for p in audits:
    req(p, "Prerequisite audit")
    audit_pass(p)
print("PASS: all prerequisite audits are fully PASS.")

# ------------------------------------------------------------------
# 2. Frozen/completed source catalog
# ------------------------------------------------------------------
sources = {
    "historical_base": DATA/"kaduwela_preprocessed_base.csv",
    "step23_trend": MET/"23_grid_mann_kendall_sens_slope.csv",
    "step23b_trajectory": MET/"23b_grid_heat_trajectory_classes.csv",
    "step28_future_base": FOR/"28_future_base_predictors_2026_2030.csv",
    "step28_future_lst": FOR/"28_future_grid_lst_predictions_2026_2030.csv",
    "step29a_green": SCE/"29a_v2_persistent_green_scenarios_2026_2030.csv",
    "step29b_built": SCE/"29b_built_up_intensity_scenarios_2026_2030.csv",
    "step29c_combined": SCE/"29c_combined_green_built_scenarios_2026_2030.csv",
    "step29d_blue": SCE/"29d_blue_space_ndwi_scenarios_2026_2030.csv",
}
source_rows = []
for key,p in sources.items():
    req(p,key)
    df = pd.read_csv(p, low_memory=False)
    grid = "grid_id" if "grid_id" in df.columns else None
    year = "year" if "year" in df.columns else None
    source_rows.append({
        "source_key":key,
        "path":str(p),
        "rows":len(df),
        "columns":len(df.columns),
        "unique_grids":df[grid].astype(str).nunique() if grid else "",
        "year_min":int(df[year].min()) if year else "",
        "year_max":int(df[year].max()) if year else "",
        "sha256":h(p),
        "columns_list":";".join(df.columns.astype(str)),
    })
source_df = pd.DataFrame(source_rows)
source_df.to_csv(PRI/"31a_priority_source_catalog.csv", index=False)

# ------------------------------------------------------------------
# 3. Candidate inventory
# ------------------------------------------------------------------
rows = []
def add(cid, dim, name, source, representation, direction, note=""):
    rows.append({
        "candidate_id":cid,
        "decision_dimension":dim,
        "indicator_name":name,
        "source_key":source,
        "candidate_representation":representation,
        "raw_priority_direction":direction,
        "notes":note,
        "31a_status":"CANDIDATE_NOT_YET_SELECTED"
    })

# Future heat severity
add("HEAT_2030","Future Heat Severity","Predicted LST","step28_future_lst",
    "2030 value","HIGHER=HIGHER_PRIORITY")
add("HEAT_MEAN","Future Heat Severity","Predicted LST","step28_future_lst",
    "2026-2030 mean","HIGHER=HIGHER_PRIORITY")
add("HEAT_MAX","Future Heat Severity","Predicted LST","step28_future_lst",
    "2026-2030 maximum","HIGHER=HIGHER_PRIORITY")

# Trend urgency
add("TREND_ABS_SEN","Trend Urgency","Absolute LST Sen slope","step23_trend",
    "2015-2025 robust slope","MORE_POSITIVE=HIGHER_PRIORITY",
    "Step23 significance remains separate.")
add("TREND_REL_SEN","Trend Urgency","Relative heat-percentile Sen slope","step23b_trajectory",
    "2015-2025 relative slope","MORE_POSITIVE=HIGHER_PRIORITY")
add("TREND_CLASS","Trend Urgency","Relative heat trajectory class","step23b_trajectory",
    "five descriptive classes","ENCODING_NOT_YET_FROZEN",
    "Do not treat class as statistical significance.")

# Vegetation / green deficit
for v in ["NDVI","EVI","green_mask"]:
    add(f"VEG_{v}_2030","Vegetation / Green Deficit",v,"step28_future_base",
        "2030 projected value","LOWER=HIGHER_PRIORITY")
    add(f"VEG_{v}_MEAN","Vegetation / Green Deficit",v,"step28_future_base",
        "2026-2030 mean","LOWER=HIGHER_PRIORITY")
add("VEG_DEFICIT_2030","Vegetation / Green Deficit","1 - green_mask","step28_future_base",
    "2030 derived deficit","HIGHER=HIGHER_PRIORITY")
add("VEG_DEFICIT_MEAN","Vegetation / Green Deficit","1 - green_mask","step28_future_base",
    "2026-2030 mean derived deficit","HIGHER=HIGHER_PRIORITY")

# Built-up / urban pressure
for v in ["NDBI","NDBI_ADJ","NightLights"]:
    add(f"BUILT_{v}_2030","Built-up / Urban Pressure",v,"step28_future_base",
        "2030 projected value","HIGHER=HIGHER_PRIORITY")
    add(f"BUILT_{v}_MEAN","Built-up / Urban Pressure",v,"step28_future_base",
        "2026-2030 mean","HIGHER=HIGHER_PRIORITY")
for v in ["building_mask","road_mask","dist_road","dist_main_road","LCZ"]:
    add(f"BUILT_CONTEXT_{v}","Built-up / Urban Pressure",v,"historical_base",
        "structural/static context","DIRECTION_TO_BE_SCREENED",
        "May be removed later if static, redundant or weakly intervention-relevant.")

# Blue-space / water deficit
add("BLUE_NDWI_2030","Blue-space / Water Deficit","NDWI","step28_future_base",
    "2030 projected value","LOWER=HIGHER_PRIORITY",
    "NDWI is water/wetness intensity, not literal water-cover percentage.")
add("BLUE_NDWI_MEAN","Blue-space / Water Deficit","NDWI","step28_future_base",
    "2026-2030 mean","LOWER=HIGHER_PRIORITY")
add("BLUE_DIST_WATER","Blue-space / Water Deficit","dist_water","historical_base",
    "static structural context","HIGHER=HIGHER_PRIORITY")

# Scenario benefit candidates
for pp in [5,10,20]:
    add(f"SCEN_GREEN_{pp}PP","Modeled Intervention Benefit",
        f"Green scenario cooling benefit +{pp}pp","step29a_green",
        "temporal summary not yet frozen","HIGHER_BENEFIT=HIGHER_PRIORITY",
        "Model-based sensitivity; not causal.")
for level in ["LOW","MODERATE","HIGH"]:
    add(f"SCEN_BUILT_{level}","Modeled Intervention Benefit",
        f"Built-up reduction cooling benefit {level}","step29b_built",
        "temporal summary not yet frozen","HIGHER_BENEFIT=HIGHER_PRIORITY")
    add(f"SCEN_BLUE_{level}","Modeled Intervention Benefit",
        f"Blue-space cooling benefit {level}","step29d_blue",
        "temporal summary not yet frozen","HIGHER_BENEFIT=HIGHER_PRIORITY")
for pp in [5,10,20]:
    for level in ["LOW","MODERATE","HIGH"]:
        add(f"SCEN_COMBINED_G{pp}_{level}","Modeled Intervention Benefit",
            f"Combined green + built benefit (+{pp}pp,{level})","step29c_combined",
            "temporal summary not yet frozen","HIGHER_BENEFIT=HIGHER_PRIORITY",
            "True joint Step29C output; never arithmetic sum.")

# Support-only evidence; explicitly not direct ranking criteria
rows += [
    {
        "candidate_id":"SUPPORT_SHAP",
        "decision_dimension":"Supporting Evidence",
        "indicator_name":"Future/historical SHAP evidence",
        "source_key":"Step24/28B",
        "candidate_representation":"support only",
        "raw_priority_direction":"NOT_A_DIRECT_CRITERION",
        "notes":"SHAP supports interpretation; it does not become CRITIC/TOPSIS criterion automatically.",
        "31a_status":"SUPPORT_ONLY"
    },
    {
        "candidate_id":"SUPPORT_UNCERTAINTY",
        "decision_dimension":"Supporting Evidence",
        "indicator_name":"Step30 uncertainty/confidence",
        "source_key":"Step30",
        "candidate_representation":"support only",
        "raw_priority_direction":"NOT_A_DIRECT_CRITERION",
        "notes":"Show alongside priority rank; do not directly reduce priority because uncertainty is high.",
        "31a_status":"SUPPORT_ONLY"
    }
]

inv = pd.DataFrame(rows)
inv.to_csv(PRI/"31a_candidate_priority_indicator_inventory.csv", index=False)

# ------------------------------------------------------------------
# 4. Summary + strict audit
# ------------------------------------------------------------------
direct = inv[inv["31a_status"]=="CANDIDATE_NOT_YET_SELECTED"]
required_dims = {
    "Future Heat Severity",
    "Trend Urgency",
    "Vegetation / Green Deficit",
    "Built-up / Urban Pressure",
    "Blue-space / Water Deficit",
    "Modeled Intervention Benefit",
}
present = set(direct["decision_dimension"].unique())

audit = pd.DataFrame([
    ["prerequisite_audits","PASS","Step23/23B/28/29A-D/30 all fully PASS"],
    ["required_dimensions_present","PASS" if required_dims.issubset(present) else "FAIL",
     str(sorted(present))],
    ["candidate_ids_unique","PASS" if inv["candidate_id"].is_unique else "FAIL",
     f"rows={len(inv)}, unique={inv['candidate_id'].nunique()}"],
    ["future_heat_from_step28_only","PASS",
     "Future heat candidates use Step28 forecasts, not future actual LST"],
    ["shap_direct_criterion","PASS","No. SHAP is support-only"],
    ["uncertainty_direct_criterion","PASS","No. Step30 uncertainty is support-only"],
    ["final_indicator_selection","PASS","Not performed in Step31A"],
    ["critic_weighting","PASS","Not performed in Step31A"],
    ["topsis_ranking","PASS","Not performed in Step31A"],
    ["model_retraining","PASS","None"],
    ["upstream_outputs_modified","PASS","No; only new Step31A files written"],
], columns=["check","status","details"])
audit.to_csv(REP/"31a_priority_candidate_inventory_audit.csv", index=False)

if not audit["status"].eq("PASS").all():
    print(audit.to_string(index=False))
    raise RuntimeError("Step31A audit failed.")

summary = direct.groupby("decision_dimension").size().reset_index(name="candidate_count")

method = {
    "step":"31A",
    "purpose":"Inventory all planning-relevant candidate priority indicators before screening.",
    "direct_candidate_count":int(len(direct)),
    "support_only_count":int((inv["31a_status"]=="SUPPORT_ONLY").sum()),
    "rules":[
        "Do not use all 93 ML features as priority criteria.",
        "Do not use future actual LST.",
        "SHAP is supporting evidence only.",
        "Step30 uncertainty is a separate confidence layer.",
        "Step29 scenario outputs are model-based sensitivity, not causal effects.",
        "No indicator is selected or removed in Step31A."
    ],
    "next":"31B eligibility + leakage screening"
}
with open(REP/"31a_priority_candidate_inventory_methodology.json","w",encoding="utf-8") as f:
    json.dump(method,f,indent=2)

report = [
    "STEP 31A - PRIORITY CANDIDATE INDICATOR INVENTORY",
    "="*65,
    f"Direct candidates: {len(direct)}",
    f"Support-only rows: {(inv['31a_status']=='SUPPORT_ONLY').sum()}",
    "",
    summary.to_string(index=False),
    "",
    "No final indicator selection occurred.",
    "No CRITIC weighting occurred.",
    "No TOPSIS ranking occurred.",
    "NEXT: Step31B eligibility + leakage screening."
]
(REP/"31a_priority_candidate_inventory_report.txt").write_text(
    "\n".join(report), encoding="utf-8"
)

hash_files = [
    PRI/"31a_priority_source_catalog.csv",
    PRI/"31a_candidate_priority_indicator_inventory.csv",
    REP/"31a_priority_candidate_inventory_audit.csv",
    REP/"31a_priority_candidate_inventory_methodology.json",
    REP/"31a_priority_candidate_inventory_report.txt",
]
pd.DataFrame(
    [{"file":p.name,"sha256":h(p)} for p in hash_files]
).to_csv(REP/"31a_priority_candidate_inventory_hashes.csv", index=False)

print("\n"+"="*120)
print("CANDIDATE COUNTS")
print("="*120)
print(summary.to_string(index=False))

print("\n"+"="*120)
print("STRICT STEP 31A AUDIT")
print("="*120)
print(audit.to_string(index=False))

print("\n"+"="*120)
print("STEP 31A COMPLETED SUCCESSFULLY")
print("="*120)
print("Inventory created only.")
print("No candidate selected/removed yet.")
print("No CRITIC/TOPSIS calculation.")
print("No model retraining.")
print("NEXT: review output before Step31B.")
