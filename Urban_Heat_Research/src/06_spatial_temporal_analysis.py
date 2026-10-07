"""
STEP 06 - SPATIAL & TEMPORAL ANALYSIS
Urban Heat Research - Kaduwela, Sri Lanka

Purpose
-------
Perform formal spatial autocorrelation diagnostics and temporal-persistence
analysis using the development period only (2015-2024).

LOCKBOX RULE
------------
2025 is the final lockbox year and is NOT used in Step 06 inferential analysis.

Main analyses
-------------
1. Queen-contiguity spatial weights (primary)
2. KNN k=8 weights (sensitivity analysis)
3. Global Moran's I for LST_C for every year 2015-2024
4. Spatial-weight sensitivity: Queen vs KNN8
5. Bivariate Moran's I:
      NDBI      vs spatial lag of LST_C
      NDVI      vs spatial lag of LST_C
      Albedo    vs spatial lag of LST_C
      green_mask vs spatial lag of LST_C
6. Local Moran's I (LISA) for 2024 LST_C
7. Benjamini-Hochberg FDR correction for local permutation p-values
8. Neighbor-LST diagnostics
9. Year-to-year temporal persistence of LST across the same grids
10. Reproducible reports, figures and maps

IMPORTANT
---------
- No raw or processed dataset is modified.
- No model is trained.
- No feature is selected.
- Spatial lag of LST is DIAGNOSTIC ONLY and must not be used as a same-year
  predictor in the later ML model because that would leak target information.
- Mann-Kendall and Sen's slope are NOT performed here; they remain part of the
  later formal trend-analysis stage.
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr

from libpysal.weights import Queen, KNN, lag_spatial
from esda.moran import Moran, Moran_BV, Moran_Local


# =============================================================================
# 1. SETTINGS
# =============================================================================

SEED = 42
PERMUTATIONS = 999
LOCAL_ALPHA = 0.05
KNN_K = 8

DEVELOPMENT_START_YEAR = 2015
DEVELOPMENT_END_YEAR = 2024
LOCKBOX_YEAR = 2025

TARGET = "LST_C"

BIVARIATE_FEATURES = [
    "NDBI",
    "NDVI",
    "Albedo",
    "green_mask",
]


# =============================================================================
# 2. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "kaduwela_preprocessed_base.csv"
)

GRID_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "kaduwela_grid_geometry_utm44n.geojson"
)

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
MAP_DIR = PROJECT_ROOT / "outputs" / "maps"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
MAP_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_REPORT = REPORT_DIR / "06_spatial_weights_summary.csv"
GLOBAL_MORAN_PATH = REPORT_DIR / "06_global_moran_lst_by_year.csv"
WEIGHTS_SENSITIVITY_PATH = (
    REPORT_DIR / "06_global_moran_weights_sensitivity.csv"
)
BIVARIATE_PATH = REPORT_DIR / "06_bivariate_moran_by_year.csv"
NEIGHBOR_PATH = REPORT_DIR / "06_neighbor_lst_diagnostics.csv"
LOCAL_2024_PATH = REPORT_DIR / "06_local_moran_2024.csv"
LOCAL_SUMMARY_PATH = REPORT_DIR / "06_local_moran_2024_cluster_summary.csv"
TEMPORAL_PATH = REPORT_DIR / "06_temporal_persistence.csv"
MAIN_REPORT = REPORT_DIR / "06_spatial_temporal_report.txt"

GLOBAL_MORAN_FIG = FIGURE_DIR / "06_global_moran_lst_by_year.png"
BIVARIATE_FIG = FIGURE_DIR / "06_bivariate_moran_by_year.png"
TEMPORAL_FIG = FIGURE_DIR / "06_temporal_persistence_by_transition.png"
MORAN_SCATTER_FIG = FIGURE_DIR / "06_moran_scatter_2024.png"
NEIGHBOR_SCATTER_FIG = FIGURE_DIR / "06_neighbor_lst_vs_lst_2024.png"

LISA_MAP = MAP_DIR / "06_lisa_clusters_2024_fdr.png"
LISA_RAW_MAP = MAP_DIR / "06_lisa_clusters_2024_raw_p05.png"


# =============================================================================
# 3. HELPERS
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 112)
    print(title)
    print("=" * 112)


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def benjamini_hochberg(p_values, alpha=0.05):
    """
    Benjamini-Hochberg false-discovery-rate correction.

    Returns
    -------
    reject : boolean array
        True where the null hypothesis is rejected at the requested FDR.
    adjusted_p : array
        BH-adjusted p-values.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)

    order = np.argsort(p)
    ranked_p = p[order]

    ranks = np.arange(1, n + 1)

    # BH rejection threshold.
    threshold = alpha * ranks / n
    below = ranked_p <= threshold

    reject_sorted = np.zeros(n, dtype=bool)

    if below.any():
        max_index = np.where(below)[0].max()
        reject_sorted[: max_index + 1] = True

    reject = np.zeros(n, dtype=bool)
    reject[order] = reject_sorted

    # Adjusted p-values.
    adjusted_sorted = ranked_p * n / ranks
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)

    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_sorted

    return reject, adjusted


def build_aligned_year(year: int, id_order: list[str]) -> pd.DataFrame:
    yearly = (
        development_df.loc[
            development_df["year"] == year,
            ["grid_id", TARGET] + BIVARIATE_FEATURES,
        ]
        .copy()
    )

    yearly["grid_id"] = yearly["grid_id"].astype(str)
    yearly = yearly.set_index("grid_id").reindex(id_order)

    if yearly.index.has_duplicates:
        raise ValueError(f"Duplicate grid IDs found for {year}.")

    if yearly.isna().any().any():
        missing_counts = yearly.isna().sum()
        raise ValueError(
            f"Missing aligned values found for {year}: "
            f"{missing_counts[missing_counts > 0].to_dict()}"
        )

    return yearly


def run_global_moran(y, w, seed_offset=0):
    # Moran uses NumPy's random state for permutation inference.
    np.random.seed(SEED + seed_offset)

    return Moran(
        y,
        w,
        transformation="r",
        permutations=PERMUTATIONS,
        two_tailed=True,
    )


def run_bivariate_moran(x, y, w, seed_offset=0):
    np.random.seed(SEED + seed_offset)

    return Moran_BV(
        x,
        y,
        w,
        transformation="r",
        permutations=PERMUTATIONS,
    )


def interpret_global_moran(i_value, expected_i, p_sim, alpha=0.05):
    if p_sim > alpha:
        return "not statistically significant"

    if i_value > expected_i:
        return "significant positive spatial autocorrelation"

    if i_value < expected_i:
        return "significant negative spatial autocorrelation"

    return "approximately spatially random"


# =============================================================================
# 4. LOAD DATA
# =============================================================================

section("STEP 06 - SPATIAL & TEMPORAL ANALYSIS")

print(f"Project root : {PROJECT_ROOT}")
print(f"Data input   : {DATA_PATH}")
print(f"Grid input   : {GRID_PATH}")

if not DATA_PATH.exists():
    print("\nERROR: Processed base dataset not found. Run Step 04 first.")
    sys.exit(1)

if not GRID_PATH.exists():
    print("\nERROR: Corrected UTM grid geometry not found. Run Step 04 first.")
    sys.exit(1)

df = pd.read_csv(
    DATA_PATH,
    low_memory=False,
    dtype={"grid_id": "string", "area_name": "string"},
)

grid = gpd.read_file(GRID_PATH)

if df.empty or grid.empty:
    print("\nERROR: Input dataset or geometry is empty.")
    sys.exit(1)

if TARGET not in df.columns:
    print(f"\nERROR: Target column {TARGET} not found.")
    sys.exit(1)

missing_bivariate = sorted(
    set(BIVARIATE_FEATURES) - set(df.columns)
)

if missing_bivariate:
    print(
        f"\nERROR: Missing bivariate features: {missing_bivariate}"
    )
    sys.exit(1)

if "grid_id" not in grid.columns:
    print("\nERROR: Grid geometry does not contain grid_id.")
    sys.exit(1)

df = df.copy()
grid = grid.copy()

df["grid_id"] = df["grid_id"].astype(str).str.strip()
grid["grid_id"] = grid["grid_id"].astype(str).str.strip()

print(f"Dataset rows      : {len(df):,}")
print(f"Unique grids      : {df['grid_id'].nunique():,}")
print(f"Geometry rows     : {len(grid):,}")
print(f"Geometry CRS      : {grid.crs}")


# =============================================================================
# 5. LOCKBOX PROTECTION
# =============================================================================

section("1. 2025 LOCKBOX PROTECTION")

development_df = df[
    df["year"].between(
        DEVELOPMENT_START_YEAR,
        DEVELOPMENT_END_YEAR,
    )
].copy()

lockbox_df = df[df["year"] == LOCKBOX_YEAR].copy()

print(
    f"Development analysis period : "
    f"{DEVELOPMENT_START_YEAR}-{DEVELOPMENT_END_YEAR}"
)
print(f"Development rows            : {len(development_df):,}")
print(f"2025 lockbox rows           : {len(lockbox_df):,}")
print("2025 is NOT used in Moran's I, LISA, bivariate Moran, or temporal diagnostics.")

expected_development_rows = (
    df["grid_id"].nunique()
    * (
        DEVELOPMENT_END_YEAR
        - DEVELOPMENT_START_YEAR
        + 1
    )
)

if len(development_df) != expected_development_rows:
    raise ValueError(
        f"Expected {expected_development_rows:,} development rows "
        f"but found {len(development_df):,}."
    )


# =============================================================================
# 6. GEOMETRY / ID ALIGNMENT
# =============================================================================

section("2. GEOMETRY AND GRID-ID ALIGNMENT")

csv_grid_ids = set(development_df["grid_id"].unique())
geometry_grid_ids = set(grid["grid_id"].unique())

missing_geometry = sorted(csv_grid_ids - geometry_grid_ids)
missing_data = sorted(geometry_grid_ids - csv_grid_ids)

print(f"CSV grids missing geometry : {len(missing_geometry):,}")
print(f"Geometry grids missing data: {len(missing_data):,}")
print(f"Invalid geometries         : {(~grid.geometry.is_valid).sum():,}")
print(f"Empty geometries           : {grid.geometry.is_empty.sum():,}")

if missing_geometry or missing_data:
    raise ValueError(
        "CSV and geometry grid IDs do not align exactly."
    )

if grid["grid_id"].duplicated().any():
    raise ValueError("Duplicate grid_id values found in geometry.")

if (~grid.geometry.is_valid).any():
    raise ValueError("Invalid geometry detected.")

if grid.geometry.is_empty.any():
    raise ValueError("Empty geometry detected.")

if grid.crs is None or not grid.crs.is_projected:
    raise ValueError(
        "Step 06 requires the corrected projected UTM grid geometry."
    )

# Keep deterministic grid order.
grid = (
    grid.sort_values("grid_id")
    .set_index("grid_id", drop=False)
)

grid_id_order = grid.index.astype(str).tolist()


# =============================================================================
# 7. SPATIAL WEIGHTS
# =============================================================================

section("3. SPATIAL WEIGHTS CONSTRUCTION")

# Primary weights: Queen contiguity.
w_queen = Queen.from_dataframe(
    grid,
    use_index=True,
)

# Sensitivity weights: KNN(k=8) based on projected grid centroids.
centroids = grid.copy()
centroids.geometry = grid.geometry.centroid

w_knn8 = KNN.from_dataframe(
    centroids,
    k=KNN_K,
    use_index=True,
)

w_queen.transform = "R"
w_knn8.transform = "R"

queen_islands = list(w_queen.islands)
knn_islands = list(w_knn8.islands)

queen_neighbor_counts = np.array(
    [len(w_queen.neighbors[i]) for i in w_queen.id_order],
    dtype=float,
)

knn_neighbor_counts = np.array(
    [len(w_knn8.neighbors[i]) for i in w_knn8.id_order],
    dtype=float,
)

weights_summary = pd.DataFrame(
    [
        {
            "weights": "Queen_contiguity_primary",
            "n": w_queen.n,
            "n_components": w_queen.n_components,
            "island_count": len(queen_islands),
            "mean_neighbors": queen_neighbor_counts.mean(),
            "median_neighbors": np.median(queen_neighbor_counts),
            "min_neighbors": queen_neighbor_counts.min(),
            "max_neighbors": queen_neighbor_counts.max(),
            "pct_nonzero": w_queen.pct_nonzero,
            "transformation": w_queen.transform,
        },
        {
            "weights": f"KNN_k{KNN_K}_sensitivity",
            "n": w_knn8.n,
            "n_components": w_knn8.n_components,
            "island_count": len(knn_islands),
            "mean_neighbors": knn_neighbor_counts.mean(),
            "median_neighbors": np.median(knn_neighbor_counts),
            "min_neighbors": knn_neighbor_counts.min(),
            "max_neighbors": knn_neighbor_counts.max(),
            "pct_nonzero": w_knn8.pct_nonzero,
            "transformation": w_knn8.transform,
        },
    ]
)

weights_summary.to_csv(WEIGHTS_REPORT, index=False)

print(weights_summary.to_string(index=False))

if queen_islands:
    warnings.warn(
        f"Queen weights contain {len(queen_islands)} islands. "
        "The KNN sensitivity analysis becomes especially important."
    )


# =============================================================================
# 8. GLOBAL MORAN'S I - LST BY YEAR
# =============================================================================

section("4. GLOBAL MORAN'S I FOR LST_C - 2015 TO 2024")

global_rows = []

years = list(
    range(
        DEVELOPMENT_START_YEAR,
        DEVELOPMENT_END_YEAR + 1,
    )
)

for year in years:
    yearly = build_aligned_year(year, w_queen.id_order)
    y = yearly[TARGET].to_numpy(dtype=float)

    moran = run_global_moran(
        y,
        w_queen,
        seed_offset=year,
    )

    global_rows.append(
        {
            "year": year,
            "n": len(y),
            "weights": "Queen",
            "moran_i": moran.I,
            "expected_i": moran.EI,
            "z_norm": moran.z_norm,
            "p_norm_two_tailed": moran.p_norm,
            "permutation_p_sim": moran.p_sim,
            "permutations": PERMUTATIONS,
            "interpretation": interpret_global_moran(
                moran.I,
                moran.EI,
                moran.p_sim,
            ),
        }
    )

global_moran = pd.DataFrame(global_rows)
global_moran.to_csv(GLOBAL_MORAN_PATH, index=False)

print(global_moran.to_string(index=False))


# =============================================================================
# 9. SPATIAL-WEIGHTS SENSITIVITY - QUEEN VS KNN8
# =============================================================================

section("5. SPATIAL-WEIGHTS SENSITIVITY")

sensitivity_rows = []

for year in years:
    # Queen alignment
    q_year = build_aligned_year(year, w_queen.id_order)
    q_y = q_year[TARGET].to_numpy(dtype=float)

    q_moran = run_global_moran(
        q_y,
        w_queen,
        seed_offset=10000 + year,
    )

    sensitivity_rows.append(
        {
            "year": year,
            "weights": "Queen",
            "moran_i": q_moran.I,
            "expected_i": q_moran.EI,
            "permutation_p_sim": q_moran.p_sim,
        }
    )

    # KNN alignment
    k_year = build_aligned_year(year, w_knn8.id_order)
    k_y = k_year[TARGET].to_numpy(dtype=float)

    k_moran = run_global_moran(
        k_y,
        w_knn8,
        seed_offset=20000 + year,
    )

    sensitivity_rows.append(
        {
            "year": year,
            "weights": f"KNN{KNN_K}",
            "moran_i": k_moran.I,
            "expected_i": k_moran.EI,
            "permutation_p_sim": k_moran.p_sim,
        }
    )

weights_sensitivity = pd.DataFrame(sensitivity_rows)
weights_sensitivity.to_csv(
    WEIGHTS_SENSITIVITY_PATH,
    index=False,
)

print(weights_sensitivity.to_string(index=False))


# =============================================================================
# 10. BIVARIATE MORAN'S I
# =============================================================================

section("6. BIVARIATE MORAN'S I")

print(
    "Definition used here:\n"
    "feature at each grid (x) vs spatial lag of LST_C in neighboring grids (Wy).\n"
    "This is a spatial association diagnostic, not a causal effect estimate."
)

bivariate_rows = []

for year in years:
    yearly = build_aligned_year(year, w_queen.id_order)
    y_lst = yearly[TARGET].to_numpy(dtype=float)

    for feature_index, feature in enumerate(BIVARIATE_FEATURES):
        x = yearly[feature].to_numpy(dtype=float)

        mbi = run_bivariate_moran(
            x,
            y_lst,
            w_queen,
            seed_offset=30000 + year * 10 + feature_index,
        )

        bivariate_rows.append(
            {
                "year": year,
                "x_feature": feature,
                "y_spatial_lag_variable": TARGET,
                "weights": "Queen",
                "bivariate_moran_i": mbi.I,
                "permutation_p_sim": mbi.p_sim,
                "permutations": PERMUTATIONS,
                "significant_p05": mbi.p_sim <= 0.05,
            }
        )

bivariate_moran = pd.DataFrame(bivariate_rows)
bivariate_moran.to_csv(BIVARIATE_PATH, index=False)

print(bivariate_moran.to_string(index=False))


# =============================================================================
# 11. NEIGHBOR-LST DIAGNOSTICS BY YEAR
# =============================================================================

section("7. NEIGHBOR-LST DIAGNOSTICS")

neighbor_rows = []

for year in years:
    yearly = build_aligned_year(year, w_queen.id_order)
    y = yearly[TARGET].to_numpy(dtype=float)

    neighbor_mean_lst = lag_spatial(
        w_queen,
        y,
    )

    pearson_r, pearson_p = pearsonr(
        y,
        neighbor_mean_lst,
    )

    spearman_r, spearman_p = spearmanr(
        y,
        neighbor_mean_lst,
    )

    neighbor_rows.append(
        {
            "year": year,
            "mean_lst": y.mean(),
            "mean_neighbor_lst": neighbor_mean_lst.mean(),
            "mean_absolute_own_neighbor_difference": (
                np.abs(y - neighbor_mean_lst).mean()
            ),
            "pearson_own_vs_neighbor_lst": pearson_r,
            "pearson_p_value": pearson_p,
            "spearman_own_vs_neighbor_lst": spearman_r,
            "spearman_p_value": spearman_p,
        }
    )

neighbor_summary = pd.DataFrame(neighbor_rows)
neighbor_summary.to_csv(NEIGHBOR_PATH, index=False)

print(neighbor_summary.to_string(index=False))

print(
    "\nIMPORTANT: neighbor mean LST is TARGET-DERIVED and is diagnostic only. "
    "Do not use it as a same-year ML predictor."
)


# =============================================================================
# 12. LOCAL MORAN'S I - 2024
# =============================================================================

section("8. LOCAL MORAN'S I (LISA) - 2024")

year_2024 = build_aligned_year(
    DEVELOPMENT_END_YEAR,
    w_queen.id_order,
)

y_2024 = year_2024[TARGET].to_numpy(dtype=float)

local_moran = Moran_Local(
    y_2024,
    w_queen,
    transformation="r",
    permutations=PERMUTATIONS,
    seed=SEED,
    n_jobs=1,
    keep_simulations=False,
)

raw_significant = local_moran.p_sim <= LOCAL_ALPHA

fdr_reject, fdr_adjusted_p = benjamini_hochberg(
    local_moran.p_sim,
    alpha=LOCAL_ALPHA,
)

quadrant_label = {
    1: "High-High",
    2: "Low-High",
    3: "Low-Low",
    4: "High-Low",
}

local_results = pd.DataFrame(
    {
        "grid_id": w_queen.id_order,
        "year": DEVELOPMENT_END_YEAR,
        TARGET: y_2024,
        "local_moran_i": local_moran.Is,
        "quadrant_code": local_moran.q,
        "quadrant_label": [
            quadrant_label.get(int(q), "Unknown")
            for q in local_moran.q
        ],
        "permutation_p_sim": local_moran.p_sim,
        "significant_raw_p05": raw_significant,
        "bh_fdr_adjusted_p": fdr_adjusted_p,
        "significant_fdr_05": fdr_reject,
    }
)

local_results["lisa_cluster_raw_p05"] = np.where(
    local_results["significant_raw_p05"],
    local_results["quadrant_label"],
    "Not Significant",
)

local_results["lisa_cluster_fdr"] = np.where(
    local_results["significant_fdr_05"],
    local_results["quadrant_label"],
    "Not Significant",
)

local_results.to_csv(
    LOCAL_2024_PATH,
    index=False,
)

raw_cluster_summary = (
    local_results["lisa_cluster_raw_p05"]
    .value_counts()
    .rename_axis("cluster")
    .reset_index(name="count")
)

raw_cluster_summary["significance_rule"] = "raw_p<=0.05"

fdr_cluster_summary = (
    local_results["lisa_cluster_fdr"]
    .value_counts()
    .rename_axis("cluster")
    .reset_index(name="count")
)

fdr_cluster_summary["significance_rule"] = "BH_FDR_0.05"

local_cluster_summary = pd.concat(
    [raw_cluster_summary, fdr_cluster_summary],
    ignore_index=True,
)

local_cluster_summary["percent"] = (
    100
    * local_cluster_summary["count"]
    / len(local_results)
)

local_cluster_summary.to_csv(
    LOCAL_SUMMARY_PATH,
    index=False,
)

print("Raw p<=0.05 cluster counts:")
print(raw_cluster_summary.to_string(index=False))

print("\nBH-FDR 0.05 cluster counts:")
print(fdr_cluster_summary.to_string(index=False))

print(
    "\nBH-FDR correction is included because 2,194 local significance tests "
    "are performed simultaneously."
)


# =============================================================================
# 13. LOCAL MORAN MAPS
# =============================================================================

section("9. LISA CLUSTER MAPS")

map_gdf = grid.copy()

local_map_values = local_results.set_index("grid_id")

map_gdf = map_gdf.join(
    local_map_values[
        [
            "lisa_cluster_raw_p05",
            "lisa_cluster_fdr",
            "permutation_p_sim",
            "bh_fdr_adjusted_p",
        ]
    ],
    how="left",
)

if map_gdf["lisa_cluster_fdr"].isna().any():
    raise ValueError("Some geometries are missing LISA results.")

# FDR map
fig, ax = plt.subplots(figsize=(10, 10))
map_gdf.plot(
    column="lisa_cluster_fdr",
    categorical=True,
    legend=True,
    ax=ax,
)
ax.set_axis_off()
ax.set_title("2024 LST Local Moran Clusters - BH FDR 0.05")
save_figure(LISA_MAP)

# Raw p<=0.05 map
fig, ax = plt.subplots(figsize=(10, 10))
map_gdf.plot(
    column="lisa_cluster_raw_p05",
    categorical=True,
    legend=True,
    ax=ax,
)
ax.set_axis_off()
ax.set_title("2024 LST Local Moran Clusters - Raw p <= 0.05")
save_figure(LISA_RAW_MAP)

print(f"Saved: {LISA_MAP}")
print(f"Saved: {LISA_RAW_MAP}")


# =============================================================================
# 14. TEMPORAL PERSISTENCE - SAME GRID ACROSS YEARS
# =============================================================================

section("10. TEMPORAL PERSISTENCE OF LST_C")

temporal_rows = []

for previous_year, current_year in zip(
    years[:-1],
    years[1:],
):
    previous = (
        development_df.loc[
            development_df["year"] == previous_year,
            ["grid_id", TARGET],
        ]
        .rename(columns={TARGET: "lst_previous"})
    )

    current = (
        development_df.loc[
            development_df["year"] == current_year,
            ["grid_id", TARGET],
        ]
        .rename(columns={TARGET: "lst_current"})
    )

    paired = previous.merge(
        current,
        on="grid_id",
        how="inner",
        validate="one_to_one",
    )

    delta = paired["lst_current"] - paired["lst_previous"]

    pearson_r, pearson_p = pearsonr(
        paired["lst_previous"],
        paired["lst_current"],
    )

    spearman_r, spearman_p = spearmanr(
        paired["lst_previous"],
        paired["lst_current"],
    )

    temporal_rows.append(
        {
            "from_year": previous_year,
            "to_year": current_year,
            "grid_count": len(paired),
            "pearson_grid_persistence": pearson_r,
            "pearson_p_value": pearson_p,
            "spearman_grid_persistence": spearman_r,
            "spearman_p_value": spearman_p,
            "mean_change_c": delta.mean(),
            "median_change_c": delta.median(),
            "mean_absolute_change_c": delta.abs().mean(),
            "rmse_change_c": np.sqrt(np.mean(delta ** 2)),
            "q95_absolute_change_c": delta.abs().quantile(0.95),
            "max_absolute_change_c": delta.abs().max(),
        }
    )

temporal_persistence = pd.DataFrame(temporal_rows)
temporal_persistence.to_csv(TEMPORAL_PATH, index=False)

print(temporal_persistence.to_string(index=False))


# =============================================================================
# 15. FIGURES - GLOBAL MORAN
# =============================================================================

section("11. SPATIAL / TEMPORAL FIGURES")

plt.figure(figsize=(10, 6))
plt.plot(
    global_moran["year"],
    global_moran["moran_i"],
    marker="o",
    label="Queen Moran's I",
)
plt.axhline(
    global_moran["expected_i"].mean(),
    linestyle="--",
    label="Approx. expected I",
)
plt.xlabel("Year")
plt.ylabel("Global Moran's I")
plt.title("Global Spatial Autocorrelation of LST_C - 2015-2024")
plt.legend()
plt.grid(alpha=0.2)
save_figure(GLOBAL_MORAN_FIG)

print(f"Saved: {GLOBAL_MORAN_FIG}")


# =============================================================================
# 16. FIGURE - BIVARIATE MORAN
# =============================================================================

plt.figure(figsize=(11, 7))

for feature, group in bivariate_moran.groupby("x_feature"):
    group = group.sort_values("year")
    plt.plot(
        group["year"],
        group["bivariate_moran_i"],
        marker="o",
        label=feature,
    )

plt.axhline(0, linewidth=1)
plt.xlabel("Year")
plt.ylabel("Bivariate Moran's I")
plt.title(
    "Feature vs Spatial Lag of LST_C - Queen Weights, 2015-2024"
)
plt.legend()
plt.grid(alpha=0.2)
save_figure(BIVARIATE_FIG)

print(f"Saved: {BIVARIATE_FIG}")


# =============================================================================
# 17. FIGURE - TEMPORAL PERSISTENCE
# =============================================================================

transition_labels = [
    f"{int(a)}-{int(b)}"
    for a, b in zip(
        temporal_persistence["from_year"],
        temporal_persistence["to_year"],
    )
]

plt.figure(figsize=(11, 6))
plt.plot(
    transition_labels,
    temporal_persistence["pearson_grid_persistence"],
    marker="o",
    label="Pearson",
)
plt.plot(
    transition_labels,
    temporal_persistence["spearman_grid_persistence"],
    marker="o",
    label="Spearman",
)
plt.xlabel("Year transition")
plt.ylabel("Same-grid temporal correlation")
plt.title("Grid-level LST Temporal Persistence")
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.2)
save_figure(TEMPORAL_FIG)

print(f"Saved: {TEMPORAL_FIG}")


# =============================================================================
# 18. FIGURE - MORAN SCATTER 2024
# =============================================================================

z_2024 = (
    y_2024 - y_2024.mean()
) / y_2024.std(ddof=0)

wz_2024 = lag_spatial(
    w_queen,
    z_2024,
)

plt.figure(figsize=(8, 7))
plt.scatter(
    z_2024,
    wz_2024,
    alpha=0.45,
    s=12,
)

# Regression slope in a row-standardized Moran scatterplot is related to I.
coef = np.polyfit(z_2024, wz_2024, 1)
x_line = np.linspace(
    z_2024.min(),
    z_2024.max(),
    100,
)
plt.plot(
    x_line,
    coef[0] * x_line + coef[1],
)

plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)
plt.xlabel("Standardized LST_C")
plt.ylabel("Spatial Lag of Standardized LST_C")
plt.title("Moran Scatterplot - LST_C, 2024")
plt.grid(alpha=0.2)
save_figure(MORAN_SCATTER_FIG)

print(f"Saved: {MORAN_SCATTER_FIG}")


# =============================================================================
# 19. FIGURE - OWN LST VS NEIGHBOR LST 2024
# =============================================================================

neighbor_lst_2024 = lag_spatial(
    w_queen,
    y_2024,
)

plt.figure(figsize=(8, 7))
hb = plt.hexbin(
    y_2024,
    neighbor_lst_2024,
    gridsize=45,
    mincnt=1,
)
plt.colorbar(hb, label="Grid count")
plt.xlabel("Grid LST_C (°C)")
plt.ylabel("Mean Neighbor LST_C (°C)")
plt.title("Grid LST_C vs Neighbor Mean LST_C - 2024")
plt.grid(alpha=0.15)
save_figure(NEIGHBOR_SCATTER_FIG)

print(f"Saved: {NEIGHBOR_SCATTER_FIG}")


# =============================================================================
# 20. MAIN TEXT REPORT
# =============================================================================

section("12. SAVE SPATIAL-TEMPORAL REPORT")

strongest_spatial_year = global_moran.loc[
    global_moran["moran_i"].idxmax()
]

weakest_spatial_year = global_moran.loc[
    global_moran["moran_i"].idxmin()
]

fdr_significant_count = int(
    local_results["significant_fdr_05"].sum()
)

raw_significant_count = int(
    local_results["significant_raw_p05"].sum()
)

report_lines = [
    "STEP 06 - SPATIAL & TEMPORAL ANALYSIS REPORT",
    "=" * 95,
    "",
    f"Input dataset: {DATA_PATH}",
    f"Input grid: {GRID_PATH}",
    "",
    "LOCKBOX POLICY",
    "-" * 95,
    f"Analysis period: {DEVELOPMENT_START_YEAR}-{DEVELOPMENT_END_YEAR}",
    f"2025 lockbox rows excluded: {len(lockbox_df):,}",
    "",
    "SPATIAL WEIGHTS",
    "-" * 95,
    "Primary weights: Queen contiguity, row standardized.",
    f"Queen islands: {len(queen_islands)}",
    f"Queen connected components: {w_queen.n_components}",
    f"Queen mean neighbors: {queen_neighbor_counts.mean():.4f}",
    f"Sensitivity weights: KNN k={KNN_K}, row standardized.",
    f"KNN islands: {len(knn_islands)}",
    "",
    "GLOBAL MORAN'S I - LST_C",
    "-" * 95,
    f"Highest Moran's I year: {int(strongest_spatial_year['year'])}",
    f"Highest Moran's I: {strongest_spatial_year['moran_i']:.6f}",
    f"Lowest Moran's I year: {int(weakest_spatial_year['year'])}",
    f"Lowest Moran's I: {weakest_spatial_year['moran_i']:.6f}",
    "Full yearly results are stored in 06_global_moran_lst_by_year.csv.",
    "",
    "WEIGHTS SENSITIVITY",
    "-" * 95,
    "Queen and KNN8 Moran's I values were both calculated for every development year.",
    "This checks whether the spatial conclusion is sensitive to one neighbor definition.",
    "",
    "BIVARIATE MORAN'S I",
    "-" * 95,
    "Bivariate tests assess each feature value at a grid against spatially lagged",
    "LST_C in neighboring grids. They are spatial association diagnostics only.",
    "",
    "LOCAL MORAN / LISA - 2024",
    "-" * 95,
    f"Raw p<=0.05 significant grids: {raw_significant_count:,}",
    f"BH-FDR 0.05 significant grids: {fdr_significant_count:,}",
    "FDR correction was applied because many local tests were performed simultaneously.",
    "",
    "TEMPORAL PERSISTENCE",
    "-" * 95,
    "Same-grid year-to-year Pearson and Spearman persistence was calculated for",
    "every transition from 2015->2016 through 2023->2024.",
    "",
    "LEAKAGE SAFEGUARD",
    "-" * 95,
    "Spatial lag of LST_C and neighbor mean LST_C are diagnostic outputs only.",
    "They must not be used as same-year predictors of LST_C in the ML pipeline.",
    "",
    "NOT PERFORMED HERE",
    "-" * 95,
    "No model training.",
    "No feature selection.",
    "No preprocessing fit.",
    "No 2025 analysis.",
    "No Mann-Kendall or Sen's slope.",
]

MAIN_REPORT.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(f"Saved: {MAIN_REPORT}")
print(f"Saved: {WEIGHTS_REPORT}")
print(f"Saved: {GLOBAL_MORAN_PATH}")
print(f"Saved: {WEIGHTS_SENSITIVITY_PATH}")
print(f"Saved: {BIVARIATE_PATH}")
print(f"Saved: {NEIGHBOR_PATH}")
print(f"Saved: {LOCAL_2024_PATH}")
print(f"Saved: {LOCAL_SUMMARY_PATH}")
print(f"Saved: {TEMPORAL_PATH}")


# =============================================================================
# 21. FINAL STATUS
# =============================================================================

section("STEP 06 COMPLETED SUCCESSFULLY")

print(
    "Spatial and temporal analysis completed using 2015-2024 only.\n"
    "2025 remained untouched as the final lockbox.\n"
    "Review all Step 06 outputs before moving to Step 07 Feature Engineering."
)
