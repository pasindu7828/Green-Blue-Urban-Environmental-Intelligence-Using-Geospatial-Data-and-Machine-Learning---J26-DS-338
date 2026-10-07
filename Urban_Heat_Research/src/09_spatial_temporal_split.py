"""
STEP 09 - STRICT TEMPORAL + SPATIAL SPLIT
Urban Heat Research - Kaduwela, Sri Lanka

Purpose
-------
Create leakage-safe, reproducible model-development splits from the finalized
Step 08 v3 model-specific feature datasets.

FINAL STEP 08 INPUTS
--------------------
Tree/nonlinear dataset:
    data/processed/kaduwela_tree_model_features.csv

Linear baseline dataset:
    data/processed/kaduwela_linear_model_features.csv

Corrected projected grid geometry:
    data/processed/kaduwela_grid_geometry_utm44n.geojson

TEMPORAL PROTOCOL
-----------------
2015-2017 : HISTORY CONTEXT ONLY
            These years helped create lag/history features but are not used as
            model samples because 3-year engineered history is incomplete.

2018-2023 : DEVELOPMENT / TRAINING PERIOD
            Step 08 feature evidence already used data through 2023, therefore
            2023 must NOT now be presented as an untouched validation year.

2024      : RESERVED TEMPORAL VALIDATION
            No Step 08 supervised feature selection used the 2024 target.

2025      : FINAL LOCKBOX TEST
            Saved as a locked split, but this script does NOT calculate,
            print, compare, rank, tune, or otherwise inspect 2025 target
            performance.

SPATIAL PROTOCOL
----------------
Create 5 geometry-only spatial folds from UTM grid centroids using KMeans.
No LST target or predictor values are used to create the spatial folds.

These folds are for later blocked spatial cross-validation inside the
2018-2023 development period.

IMPORTANT
---------
This step performs NO:
- imputation
- scaling
- feature fitting
- model fitting
- hyperparameter tuning
- 2024 evaluation
- 2025 evaluation

Train-only preprocessing happens in Step 10.
"""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


# =============================================================================
# 1. SETTINGS
# =============================================================================

SEED = 42
TARGET = "LST_C"

HISTORY_YEARS = [2015, 2016, 2017]
TRAIN_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
VALIDATION_YEAR = 2024
TEST_YEAR = 2025

N_SPATIAL_FOLDS = 5

ID_COLUMNS = [
    "grid_id",
    "year",
    TARGET,
    "area_name",
]


# =============================================================================
# 2. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TREE_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "kaduwela_tree_model_features.csv"
)

LINEAR_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "kaduwela_linear_model_features.csv"
)

GRID_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "kaduwela_grid_geometry_utm44n.geojson"
)

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
TREE_SPLIT_DIR = SPLIT_DIR / "tree"
LINEAR_SPLIT_DIR = SPLIT_DIR / "linear"
SPATIAL_DIR = SPLIT_DIR / "spatial"

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

for directory in [
    SPLIT_DIR,
    TREE_SPLIT_DIR,
    LINEAR_SPLIT_DIR,
    SPATIAL_DIR,
    REPORT_DIR,
    FIGURE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

TREE_TRAIN_PATH = TREE_SPLIT_DIR / "tree_train_2018_2023.csv"
TREE_VAL_PATH = TREE_SPLIT_DIR / "tree_validation_2024.csv"
TREE_TEST_PATH = TREE_SPLIT_DIR / "tree_test_2025_LOCKED.csv"

LINEAR_TRAIN_PATH = LINEAR_SPLIT_DIR / "linear_train_2018_2023.csv"
LINEAR_VAL_PATH = LINEAR_SPLIT_DIR / "linear_validation_2024.csv"
LINEAR_TEST_PATH = LINEAR_SPLIT_DIR / "linear_test_2025_LOCKED.csv"

TREE_DEV_SPATIAL_PATH = (
    SPATIAL_DIR
    / "tree_development_2018_2023_with_spatial_folds.csv"
)

LINEAR_DEV_SPATIAL_PATH = (
    SPATIAL_DIR
    / "linear_development_2018_2023_with_spatial_folds.csv"
)

SPATIAL_ASSIGNMENT_PATH = (
    SPATIAL_DIR
    / "spatial_fold_assignment.csv"
)

SPATIAL_GRID_PATH = (
    SPATIAL_DIR
    / "spatial_fold_grid_utm44n.geojson"
)

HISTORY_KEYS_PATH = (
    SPLIT_DIR
    / "history_context_2015_2017_keys.csv"
)

MANIFEST_PATH = REPORT_DIR / "09_split_manifest.csv"
SPATIAL_SUMMARY_PATH = REPORT_DIR / "09_spatial_fold_summary.csv"
TEMPORAL_AUDIT_PATH = REPORT_DIR / "09_temporal_split_audit.csv"
SPATIAL_AUDIT_PATH = REPORT_DIR / "09_spatial_split_audit.csv"
FILE_HASH_PATH = REPORT_DIR / "09_split_file_hashes.csv"
MAIN_REPORT_PATH = REPORT_DIR / "09_spatial_temporal_split_report.txt"

SPATIAL_MAP_PATH = FIGURE_DIR / "09_spatial_folds_map.png"


# =============================================================================
# 3. HELPERS
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 122)
    print(title)
    print("=" * 122)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )


def validate_model_dataset(
    df: pd.DataFrame,
    name: str,
) -> None:
    required = set(ID_COLUMNS)
    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"{name}: missing required columns: {missing}"
        )

    if df.empty:
        raise ValueError(
            f"{name}: dataset is empty."
        )

    if df["grid_id"].isna().any():
        raise ValueError(
            f"{name}: grid_id contains missing values."
        )

    if df["year"].isna().any():
        raise ValueError(
            f"{name}: year contains missing values."
        )

    duplicate_count = int(
        df.duplicated(
            ["grid_id", "year"],
            keep=False,
        ).sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"{name}: duplicate grid_id-year rows found: "
            f"{duplicate_count}"
        )

    expected_years = list(range(2015, 2026))
    actual_years = sorted(
        df["year"].astype(int).unique().tolist()
    )

    if actual_years != expected_years:
        raise ValueError(
            f"{name}: unexpected year coverage: {actual_years}"
        )


def split_temporally(
    df: pd.DataFrame,
):
    train = df[
        df["year"].isin(TRAIN_YEARS)
    ].copy()

    validation = df[
        df["year"] == VALIDATION_YEAR
    ].copy()

    test = df[
        df["year"] == TEST_YEAR
    ].copy()

    history = df[
        df["year"].isin(HISTORY_YEARS)
    ].copy()

    return history, train, validation, test


def check_nonoverlap(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    train_keys = set(
        map(
            tuple,
            train[["grid_id", "year"]].to_numpy(),
        )
    )

    val_keys = set(
        map(
            tuple,
            validation[["grid_id", "year"]].to_numpy(),
        )
    )

    test_keys = set(
        map(
            tuple,
            test[["grid_id", "year"]].to_numpy(),
        )
    )

    return {
        "train_validation_overlap": len(
            train_keys & val_keys
        ),
        "train_test_overlap": len(
            train_keys & test_keys
        ),
        "validation_test_overlap": len(
            val_keys & test_keys
        ),
    }


# =============================================================================
# 4. LOAD FINAL STEP 08 INPUTS
# =============================================================================

section("STEP 09 - STRICT TEMPORAL + SPATIAL SPLIT")

print(f"Project root : {PROJECT_ROOT}")
print(f"Tree input   : {TREE_INPUT}")
print(f"Linear input : {LINEAR_INPUT}")
print(f"Grid geometry: {GRID_PATH}")

for path in [
    TREE_INPUT,
    LINEAR_INPUT,
    GRID_PATH,
]:
    if not path.exists():
        print(f"\nERROR: required input not found: {path}")
        sys.exit(1)

tree_df = pd.read_csv(
    TREE_INPUT,
    low_memory=False,
    dtype={
        "grid_id": "string",
        "area_name": "string",
    },
)

linear_df = pd.read_csv(
    LINEAR_INPUT,
    low_memory=False,
    dtype={
        "grid_id": "string",
        "area_name": "string",
    },
)

grid = gpd.read_file(
    GRID_PATH
)

grid["grid_id"] = (
    grid["grid_id"]
    .astype("string")
)

validate_model_dataset(
    tree_df,
    "TREE",
)

validate_model_dataset(
    linear_df,
    "LINEAR",
)

print(f"Tree rows / columns   : {len(tree_df):,} / {len(tree_df.columns):,}")
print(f"Linear rows / columns : {len(linear_df):,} / {len(linear_df.columns):,}")
print(f"Unique grids          : {tree_df['grid_id'].nunique():,}")
print(f"Years                 : {sorted(tree_df['year'].unique().tolist())}")


# =============================================================================
# 5. CROSS-DATASET INTEGRITY
# =============================================================================

section("1. CROSS-DATASET INTEGRITY")

tree_keys = (
    tree_df[
        ["grid_id", "year"]
    ]
    .sort_values(
        ["grid_id", "year"]
    )
    .reset_index(drop=True)
)

linear_keys = (
    linear_df[
        ["grid_id", "year"]
    ]
    .sort_values(
        ["grid_id", "year"]
    )
    .reset_index(drop=True)
)

same_keys = tree_keys.equals(
    linear_keys
)

tree_target = (
    tree_df[
        ["grid_id", "year", TARGET]
    ]
    .sort_values(
        ["grid_id", "year"]
    )
    .reset_index(drop=True)
)

linear_target = (
    linear_df[
        ["grid_id", "year", TARGET]
    ]
    .sort_values(
        ["grid_id", "year"]
    )
    .reset_index(drop=True)
)

same_target = np.allclose(
    tree_target[TARGET],
    linear_target[TARGET],
    equal_nan=True,
)

same_area = (
    tree_df[
        ["grid_id", "year", "area_name"]
    ]
    .sort_values(
        ["grid_id", "year"]
    )
    .reset_index(drop=True)
    .equals(
        linear_df[
            ["grid_id", "year", "area_name"]
        ]
        .sort_values(
            ["grid_id", "year"]
        )
        .reset_index(drop=True)
    )
)

print(f"Same grid-year keys : {same_keys}")
print(f"Same target values  : {same_target}")
print(f"Same area labels    : {same_area}")

if not all(
    [
        same_keys,
        same_target,
        same_area,
    ]
):
    print(
        "\nERROR: Tree and linear datasets are not aligned."
    )
    sys.exit(1)


# =============================================================================
# 6. TEMPORAL SPLITS
# =============================================================================

section("2. TEMPORAL SPLITS")

(
    tree_history,
    tree_train,
    tree_val,
    tree_test,
) = split_temporally(tree_df)

(
    linear_history,
    linear_train,
    linear_val,
    linear_test,
) = split_temporally(linear_df)

print(
    f"History context 2015-2017 : "
    f"{len(tree_history):,} rows"
)
print(
    f"Train 2018-2023           : "
    f"{len(tree_train):,} rows"
)
print(
    f"Validation 2024           : "
    f"{len(tree_val):,} rows"
)
print(
    f"Final test 2025           : "
    f"{len(tree_test):,} rows"
)

expected_per_year = tree_df["grid_id"].nunique()

expected_counts = {
    "history": expected_per_year * len(HISTORY_YEARS),
    "train": expected_per_year * len(TRAIN_YEARS),
    "validation": expected_per_year,
    "test": expected_per_year,
}

actual_counts = {
    "history": len(tree_history),
    "train": len(tree_train),
    "validation": len(tree_val),
    "test": len(tree_test),
}

print("\nExpected row counts:")
print(expected_counts)

print("\nActual row counts:")
print(actual_counts)

if actual_counts != expected_counts:
    print(
        "\nERROR: temporal split row counts do not match expectations."
    )
    sys.exit(1)


# =============================================================================
# 7. STRUCTURAL MISSINGNESS CHECK IN MODELING YEARS
# =============================================================================

section("3. MODELING-PERIOD MISSINGNESS CHECK")

tree_model_feature_cols = [
    c
    for c in tree_df.columns
    if c not in [
        "grid_id",
        TARGET,
        "area_name",
    ]
]

linear_model_feature_cols = [
    c
    for c in linear_df.columns
    if c not in [
        "grid_id",
        TARGET,
        "area_name",
    ]
]

tree_2018_2025 = tree_df[
    tree_df["year"].between(2018, 2025)
]

linear_2018_2025 = linear_df[
    linear_df["year"].between(2018, 2025)
]

tree_missing = int(
    tree_2018_2025[
        tree_model_feature_cols
    ]
    .isna()
    .sum()
    .sum()
)

linear_missing = int(
    linear_2018_2025[
        linear_model_feature_cols
    ]
    .isna()
    .sum()
    .sum()
)

print(
    f"Tree predictor missing cells 2018-2025   : "
    f"{tree_missing:,}"
)
print(
    f"Linear predictor missing cells 2018-2025 : "
    f"{linear_missing:,}"
)

if tree_missing != 0 or linear_missing != 0:
    print(
        "\nERROR: unexpected predictor missingness exists in modeling years."
    )
    sys.exit(1)

print(
    "PASS: structural lag/history missingness is confined to the earlier "
    "history-context years."
)


# =============================================================================
# 8. SPLIT NON-OVERLAP
# =============================================================================

section("4. TEMPORAL NON-OVERLAP AUDIT")

tree_overlap = check_nonoverlap(
    tree_train,
    tree_val,
    tree_test,
)

linear_overlap = check_nonoverlap(
    linear_train,
    linear_val,
    linear_test,
)

print("Tree split overlaps:")
print(tree_overlap)

print("\nLinear split overlaps:")
print(linear_overlap)

if any(tree_overlap.values()) or any(linear_overlap.values()):
    print(
        "\nERROR: temporal split overlap detected."
    )
    sys.exit(1)


# =============================================================================
# 9. SAVE TEMPORAL SPLITS
# =============================================================================

section("5. SAVE TEMPORAL SPLITS")

save_csv(
    tree_train,
    TREE_TRAIN_PATH,
)

save_csv(
    tree_val,
    TREE_VAL_PATH,
)

save_csv(
    tree_test,
    TREE_TEST_PATH,
)

save_csv(
    linear_train,
    LINEAR_TRAIN_PATH,
)

save_csv(
    linear_val,
    LINEAR_VAL_PATH,
)

save_csv(
    linear_test,
    LINEAR_TEST_PATH,
)

history_keys = (
    tree_history[
        ["grid_id", "year"]
    ]
    .copy()
)

save_csv(
    history_keys,
    HISTORY_KEYS_PATH,
)

print(f"Saved: {TREE_TRAIN_PATH}")
print(f"Saved: {TREE_VAL_PATH}")
print(f"Saved: {TREE_TEST_PATH}")
print(f"Saved: {LINEAR_TRAIN_PATH}")
print(f"Saved: {LINEAR_VAL_PATH}")
print(f"Saved: {LINEAR_TEST_PATH}")
print(f"Saved: {HISTORY_KEYS_PATH}")

print(
    "\nIMPORTANT: 2025 was only separated and saved. "
    "No 2025 target performance was calculated."
)


# =============================================================================
# 10. GEOMETRY INTEGRITY
# =============================================================================

section("6. GEOMETRY INTEGRITY")

if grid.crs is None:
    print(
        "\nERROR: grid geometry has no CRS."
    )
    sys.exit(1)

print(f"Grid CRS                : {grid.crs}")
print(f"Geometry rows           : {len(grid):,}")
print(f"Unique geometry grid_id : {grid['grid_id'].nunique():,}")

geometry_duplicate_ids = int(
    grid["grid_id"].duplicated(
        keep=False
    ).sum()
)

invalid_geometries = int(
    (~grid.geometry.is_valid).sum()
)

empty_geometries = int(
    grid.geometry.is_empty.sum()
)

dataset_grid_ids = set(
    tree_df["grid_id"].unique().tolist()
)

geometry_grid_ids = set(
    grid["grid_id"].unique().tolist()
)

missing_geometry_ids = sorted(
    dataset_grid_ids
    - geometry_grid_ids
)

extra_geometry_ids = sorted(
    geometry_grid_ids
    - dataset_grid_ids
)

print(
    f"Duplicate geometry grid IDs : "
    f"{geometry_duplicate_ids}"
)
print(
    f"Invalid geometries           : "
    f"{invalid_geometries}"
)
print(
    f"Empty geometries             : "
    f"{empty_geometries}"
)
print(
    f"Dataset grids missing geometry: "
    f"{len(missing_geometry_ids)}"
)
print(
    f"Extra geometry IDs            : "
    f"{len(extra_geometry_ids)}"
)

if any(
    [
        geometry_duplicate_ids != 0,
        invalid_geometries != 0,
        empty_geometries != 0,
        len(missing_geometry_ids) != 0,
        len(extra_geometry_ids) != 0,
    ]
):
    print(
        "\nERROR: geometry integrity check failed."
    )
    sys.exit(1)


# =============================================================================
# 11. CREATE GEOMETRY-ONLY SPATIAL FOLDS
# =============================================================================

section("7. GEOMETRY-ONLY SPATIAL FOLDS")

spatial_grid = grid[
    ["grid_id", "geometry"]
].copy()

centroids = spatial_grid.geometry.centroid

spatial_grid["centroid_x"] = centroids.x
spatial_grid["centroid_y"] = centroids.y

coords = spatial_grid[
    ["centroid_x", "centroid_y"]
].to_numpy()

kmeans = KMeans(
    n_clusters=N_SPATIAL_FOLDS,
    random_state=SEED,
    n_init=50,
)

raw_cluster = kmeans.fit_predict(
    coords
)

spatial_grid["raw_cluster"] = raw_cluster

# Relabel cluster IDs deterministically from west to east, then south to north.
cluster_order = (
    spatial_grid
    .groupby("raw_cluster")[
        ["centroid_x", "centroid_y"]
    ]
    .mean()
    .reset_index()
    .sort_values(
        ["centroid_x", "centroid_y"]
    )
    .reset_index(drop=True)
)

cluster_map = {
    int(raw_id): int(new_fold)
    for new_fold, raw_id in enumerate(
        cluster_order["raw_cluster"],
        start=1,
    )
}

spatial_grid["spatial_fold"] = (
    spatial_grid["raw_cluster"]
    .map(cluster_map)
    .astype(int)
)

spatial_assignment = spatial_grid[
    [
        "grid_id",
        "centroid_x",
        "centroid_y",
        "spatial_fold",
    ]
].copy()

fold_summary = (
    spatial_assignment
    .groupby("spatial_fold")
    .agg(
        grid_count=("grid_id", "nunique"),
        centroid_x_mean=("centroid_x", "mean"),
        centroid_y_mean=("centroid_y", "mean"),
        centroid_x_min=("centroid_x", "min"),
        centroid_x_max=("centroid_x", "max"),
        centroid_y_min=("centroid_y", "min"),
        centroid_y_max=("centroid_y", "max"),
    )
    .reset_index()
)

fold_summary["grid_fraction"] = (
    fold_summary["grid_count"]
    / spatial_assignment["grid_id"].nunique()
)

print("Spatial fold summary:")
print(fold_summary.to_string(index=False))

min_fraction = float(
    fold_summary["grid_fraction"].min()
)

max_fraction = float(
    fold_summary["grid_fraction"].max()
)

print(
    f"\nSmallest fold fraction: "
    f"{min_fraction:.3f}"
)
print(
    f"Largest fold fraction : "
    f"{max_fraction:.3f}"
)

# Wide enough to tolerate irregular Kaduwela geometry while catching
# pathological cluster imbalance.
if min_fraction < 0.10 or max_fraction > 0.35:
    print(
        "\nERROR: spatial folds are too imbalanced. "
        "Review spatial fold design before continuing."
    )
    sys.exit(1)


# =============================================================================
# 12. ATTACH SPATIAL FOLDS TO DEVELOPMENT DATA
# =============================================================================

section("8. ATTACH SPATIAL FOLDS TO DEVELOPMENT PERIOD")

tree_dev_spatial = tree_train.merge(
    spatial_assignment[
        ["grid_id", "spatial_fold"]
    ],
    on="grid_id",
    how="left",
    validate="many_to_one",
)

linear_dev_spatial = linear_train.merge(
    spatial_assignment[
        ["grid_id", "spatial_fold"]
    ],
    on="grid_id",
    how="left",
    validate="many_to_one",
)

tree_missing_fold = int(
    tree_dev_spatial[
        "spatial_fold"
    ].isna().sum()
)

linear_missing_fold = int(
    linear_dev_spatial[
        "spatial_fold"
    ].isna().sum()
)

print(
    f"Tree development rows missing spatial fold   : "
    f"{tree_missing_fold}"
)
print(
    f"Linear development rows missing spatial fold : "
    f"{linear_missing_fold}"
)

if tree_missing_fold != 0 or linear_missing_fold != 0:
    print(
        "\nERROR: spatial fold attachment failed."
    )
    sys.exit(1)

tree_fold_year_counts = (
    tree_dev_spatial
    .groupby(
        ["spatial_fold", "year"]
    )
    .size()
    .rename("row_count")
    .reset_index()
)

print("\nTree development fold-year counts:")
print(
    tree_fold_year_counts.to_string(
        index=False
    )
)


# =============================================================================
# 13. SAVE SPATIAL OUTPUTS
# =============================================================================

section("9. SAVE SPATIAL OUTPUTS")

save_csv(
    spatial_assignment,
    SPATIAL_ASSIGNMENT_PATH,
)

save_csv(
    tree_dev_spatial,
    TREE_DEV_SPATIAL_PATH,
)

save_csv(
    linear_dev_spatial,
    LINEAR_DEV_SPATIAL_PATH,
)

spatial_grid_output = spatial_grid.drop(
    columns=["raw_cluster"]
).copy()

spatial_grid_output.to_file(
    SPATIAL_GRID_PATH,
    driver="GeoJSON",
)

fold_summary.to_csv(
    SPATIAL_SUMMARY_PATH,
    index=False,
)

print(f"Saved: {SPATIAL_ASSIGNMENT_PATH}")
print(f"Saved: {TREE_DEV_SPATIAL_PATH}")
print(f"Saved: {LINEAR_DEV_SPATIAL_PATH}")
print(f"Saved: {SPATIAL_GRID_PATH}")
print(f"Saved: {SPATIAL_SUMMARY_PATH}")


# =============================================================================
# 14. SPATIAL FOLD MAP
# =============================================================================

section("10. SPATIAL FOLD MAP")

fig, ax = plt.subplots(
    figsize=(10, 8)
)

spatial_grid_output.plot(
    column="spatial_fold",
    categorical=True,
    legend=True,
    ax=ax,
)

ax.set_title(
    "Geometry-Only Spatial Cross-Validation Folds"
)
ax.set_xlabel("UTM Easting")
ax.set_ylabel("UTM Northing")

plt.tight_layout()
plt.savefig(
    SPATIAL_MAP_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

print(f"Saved: {SPATIAL_MAP_PATH}")


# =============================================================================
# 15. SPLIT MANIFEST
# =============================================================================

section("11. SPLIT MANIFEST")

manifest_rows = [
    {
        "dataset_family": "tree",
        "split": "history_context",
        "years": "2015-2017",
        "rows": len(tree_history),
        "unique_grids": tree_history["grid_id"].nunique(),
        "purpose": "lag/history context only; not model samples",
        "target_use": "not evaluated",
        "path": str(HISTORY_KEYS_PATH),
    },
    {
        "dataset_family": "tree",
        "split": "train_development",
        "years": "2018-2023",
        "rows": len(tree_train),
        "unique_grids": tree_train["grid_id"].nunique(),
        "purpose": "model development and train-only fitting",
        "target_use": "allowed for training/tuning protocol",
        "path": str(TREE_TRAIN_PATH),
    },
    {
        "dataset_family": "tree",
        "split": "validation",
        "years": "2024",
        "rows": len(tree_val),
        "unique_grids": tree_val["grid_id"].nunique(),
        "purpose": "reserved temporal validation",
        "target_use": "later validation only",
        "path": str(TREE_VAL_PATH),
    },
    {
        "dataset_family": "tree",
        "split": "final_test_lockbox",
        "years": "2025",
        "rows": len(tree_test),
        "unique_grids": tree_test["grid_id"].nunique(),
        "purpose": "final one-time test after strategy freeze",
        "target_use": "LOCKED - do not inspect during development",
        "path": str(TREE_TEST_PATH),
    },
    {
        "dataset_family": "linear",
        "split": "train_development",
        "years": "2018-2023",
        "rows": len(linear_train),
        "unique_grids": linear_train["grid_id"].nunique(),
        "purpose": "linear baseline development",
        "target_use": "allowed for training/tuning protocol",
        "path": str(LINEAR_TRAIN_PATH),
    },
    {
        "dataset_family": "linear",
        "split": "validation",
        "years": "2024",
        "rows": len(linear_val),
        "unique_grids": linear_val["grid_id"].nunique(),
        "purpose": "reserved temporal validation",
        "target_use": "later validation only",
        "path": str(LINEAR_VAL_PATH),
    },
    {
        "dataset_family": "linear",
        "split": "final_test_lockbox",
        "years": "2025",
        "rows": len(linear_test),
        "unique_grids": linear_test["grid_id"].nunique(),
        "purpose": "final one-time test after strategy freeze",
        "target_use": "LOCKED - do not inspect during development",
        "path": str(LINEAR_TEST_PATH),
    },
]

manifest_df = pd.DataFrame(
    manifest_rows
)

manifest_df.to_csv(
    MANIFEST_PATH,
    index=False,
)

print(
    manifest_df.to_string(
        index=False
    )
)


# =============================================================================
# 16. AUDITS
# =============================================================================

section("12. TEMPORAL + SPATIAL AUDITS")

temporal_audit = pd.DataFrame(
    [
        {
            "check": "2015_2017_used_as_model_samples",
            "status": "PASS",
            "details": "No. They are retained only as history context.",
        },
        {
            "check": "training_years",
            "status": "PASS",
            "details": "2018-2023.",
        },
        {
            "check": "2024_reserved_validation",
            "status": "PASS",
            "details": "2024 separated and not evaluated in Step 09.",
        },
        {
            "check": "2025_final_lockbox",
            "status": "PASS",
            "details": "2025 separated and not evaluated in Step 09.",
        },
        {
            "check": "temporal_split_overlap",
            "status": "PASS",
            "details": (
                f"Tree={tree_overlap}; Linear={linear_overlap}"
            ),
        },
        {
            "check": "predictor_missingness_2018_2025",
            "status": (
                "PASS"
                if tree_missing == 0
                and linear_missing == 0
                else "FAIL"
            ),
            "details": (
                f"tree_missing={tree_missing}; "
                f"linear_missing={linear_missing}"
            ),
        },
        {
            "check": "preprocessing_fitted_in_step09",
            "status": "PASS",
            "details": "No imputation/scaling/model fitting performed.",
        },
    ]
)

spatial_audit = pd.DataFrame(
    [
        {
            "check": "spatial_fold_target_used",
            "status": "PASS",
            "details": "No. Folds created from UTM centroid coordinates only.",
        },
        {
            "check": "spatial_fold_predictors_used",
            "status": "PASS",
            "details": "No environmental/model predictors used.",
        },
        {
            "check": "spatial_fold_count",
            "status": "PASS",
            "details": f"{N_SPATIAL_FOLDS} spatial folds.",
        },
        {
            "check": "spatial_fold_missing_assignments",
            "status": "PASS",
            "details": (
                f"tree={tree_missing_fold}; "
                f"linear={linear_missing_fold}"
            ),
        },
        {
            "check": "spatial_fold_balance",
            "status": "PASS",
            "details": (
                f"min_fraction={min_fraction:.4f}; "
                f"max_fraction={max_fraction:.4f}"
            ),
        },
        {
            "check": "geometry_integrity",
            "status": "PASS",
            "details": (
                "Unique valid non-empty geometry for every dataset grid."
            ),
        },
    ]
)

temporal_audit.to_csv(
    TEMPORAL_AUDIT_PATH,
    index=False,
)

spatial_audit.to_csv(
    SPATIAL_AUDIT_PATH,
    index=False,
)

print("Temporal audit:")
print(
    temporal_audit.to_string(
        index=False
    )
)

print("\nSpatial audit:")
print(
    spatial_audit.to_string(
        index=False
    )
)


# =============================================================================
# 17. FILE HASHES
# =============================================================================

section("13. REPRODUCIBILITY HASHES")

hash_paths = [
    TREE_INPUT,
    LINEAR_INPUT,
    TREE_TRAIN_PATH,
    TREE_VAL_PATH,
    TREE_TEST_PATH,
    LINEAR_TRAIN_PATH,
    LINEAR_VAL_PATH,
    LINEAR_TEST_PATH,
    SPATIAL_ASSIGNMENT_PATH,
    TREE_DEV_SPATIAL_PATH,
    LINEAR_DEV_SPATIAL_PATH,
]

hash_rows = []

for path in hash_paths:
    digest = sha256_file(path)

    hash_rows.append(
        {
            "file": str(path),
            "sha256": digest,
        }
    )

    print(
        f"{path.name}: {digest}"
    )

hash_df = pd.DataFrame(
    hash_rows
)

hash_df.to_csv(
    FILE_HASH_PATH,
    index=False,
)


# =============================================================================
# 18. MAIN REPORT
# =============================================================================

section("14. SAVE STEP 09 REPORT")

report_lines = [
    "STEP 09 - STRICT TEMPORAL + SPATIAL SPLIT REPORT",
    "=" * 104,
    "",
    "TEMPORAL PROTOCOL",
    "-" * 104,
    "2015-2017: history context only",
    "2018-2023: development/training period",
    "2024: reserved temporal validation",
    "2025: final lockbox test",
    "",
    f"History-context rows: {len(tree_history):,}",
    f"Training rows: {len(tree_train):,}",
    f"2024 validation rows: {len(tree_val):,}",
    f"2025 lockbox rows: {len(tree_test):,}",
    "",
    "WHY 2023 IS IN TRAINING",
    "-" * 104,
    "Step 08 feature evidence already used 2023 target information through",
    "rolling permutation importance and cross-year stability. Therefore 2023",
    "is not treated as an untouched validation year. The first untouched",
    "temporal validation year is 2024.",
    "",
    "SPATIAL PROTOCOL",
    "-" * 104,
    f"Spatial folds: {N_SPATIAL_FOLDS}",
    "Method: KMeans on projected UTM grid-centroid coordinates only",
    "Target used for fold assignment: No",
    "Predictors used for fold assignment: No",
    f"Smallest fold fraction: {min_fraction:.4f}",
    f"Largest fold fraction: {max_fraction:.4f}",
    "",
    "MODEL-SPECIFIC INPUTS",
    "-" * 104,
    f"Tree features input columns: {len(tree_df.columns):,}",
    f"Linear features input columns: {len(linear_df.columns):,}",
    "",
    "MISSINGNESS",
    "-" * 104,
    f"Tree predictor missing cells 2018-2025: {tree_missing}",
    f"Linear predictor missing cells 2018-2025: {linear_missing}",
    "",
    "LEAKAGE CONTROL",
    "-" * 104,
    "No scaling, imputation, model fitting, tuning, or evaluation occurred.",
    "2024 was split but not evaluated.",
    "2025 was split and saved as LOCKED but not evaluated.",
    "",
    "NEXT",
    "-" * 104,
    "Step 10 should fit any required preprocessing on the 2018-2023 training",
    "split only, then apply the fitted transformations unchanged to 2024 and",
    "2025. Spatial fold assignments should be preserved for later spatial CV.",
]

MAIN_REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(f"Saved: {MAIN_REPORT_PATH}")
print(f"Saved: {MANIFEST_PATH}")
print(f"Saved: {TEMPORAL_AUDIT_PATH}")
print(f"Saved: {SPATIAL_AUDIT_PATH}")
print(f"Saved: {FILE_HASH_PATH}")


# =============================================================================
# 19. FINAL STATUS
# =============================================================================

section("STEP 09 COMPLETED SUCCESSFULLY")

print(
    "Strict temporal and geometry-only spatial splitting completed.\n"
    "2018-2023 is the development/training period.\n"
    "2024 remains reserved for temporal validation.\n"
    "2025 remains the final lockbox test and was not evaluated.\n"
    "Five spatial folds were created from geometry only.\n"
    "Review all Step 09 outputs before Step 10 train-only preprocessing."
)
