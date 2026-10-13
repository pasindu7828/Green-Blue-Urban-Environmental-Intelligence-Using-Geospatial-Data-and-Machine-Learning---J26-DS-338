"""
STEP 24 - SHAP EXPLAINABILITY
Urban Heat Research - Kaduwela, Sri Lanka

UPDATED VERSION
---------------
This version explains the final Step-22B production LightGBM spatial-anomaly
model and also links SHAP outputs with the finalized Step-23B v2 trajectory
classes.

WHAT STEP 24 EXPLAINS
---------------------
The final production model target is:

    within-year spatial LST anomaly
    = grid LST - same-year Kaduwela annual mean LST

Therefore SHAP answers:

    "Why does the model place this grid hotter or cooler relative to
     Kaduwela's annual thermal level?"

It does NOT explain the future annual thermal-level forecast.
It does NOT create 2026-2030 future SHAP yet.
Future SHAP will be generated after future predictors and future LST forecasts
exist.

CENTERED SHAP
-------------
The frozen final strategy centers predicted spatial anomalies across all
2,194 grids for each target year.

Raw model:
    f_i = base + sum_j(phi_ij)

Centered spatial anomaly:
    g_i = f_i - mean_i(f_i)

Because the SHAP base value is common across rows:
    g_i = sum_j(phi_ij - mean_i(phi_ij))

So this script saves:
1) raw TreeSHAP contributions;
2) within-year centered SHAP contributions that reconstruct the actual
   centered spatial-anomaly prediction.

TRAJECTORY-CONDITIONED SHAP
---------------------------
Step 23B v2 provides one RELATIVE SPATIAL HEAT TRAJECTORY class per grid:
- Persistent Hot
- Intensifying
- Emerging
- Diminishing
- Persistent Cool

Step 24 joins those classes with SHAP outputs and saves:
- SHAP summaries by trajectory class across 2018-2025;
- 2025 SHAP summaries by trajectory class;
- top model-based heating associations by trajectory class;
- top model-based cooling associations by trajectory class;
- one report-quality SHAP figure per trajectory class.

IMPORTANT:
These outputs describe MODEL-BASED ASSOCIATIONS.
They do NOT prove that a feature caused a trajectory class.

SPATIAL OUTPUTS
---------------
Spatial SHAP maps use:
- actual 2,194 Kaduwela grid polygons;
- the real Kaduwela boundary source under data/raw/boundary;
- EPSG:4326 for web/frontend GeoJSON;
- 300-DPI report-quality PNGs.

No future prediction, future SHAP, new tuning, or new test evaluation occurs.
"""

from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import joblib

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "\nSHAP is not installed in this virtual environment.\n"
        "Install it once with:\n"
        "    pip install shap\n"
        "Then rerun:\n"
        "    python src/24_shap_explainability.py\n"
    ) from exc


# =============================================================================
# SETTINGS
# =============================================================================

GRID_COL = "grid_id"
YEAR_COL = "year"
AREA_COL = "area_name"

EXPECTED_GRIDS = 2194
EXPECTED_PRODUCTION_ROWS = 17552
EXPECTED_YEARS = list(range(2018, 2026))
EXPECTED_FEATURES = 93

LOCAL_YEAR = 2025
LOCAL_TOP_K = 10
GLOBAL_TOP_N = 20
TRAJECTORY_TOP_K = 10
SPATIAL_TOP_N = 3

BEESWARM_MAX_ROWS = 5000
RANDOM_SEED = 42

TRAJECTORY_CLASSES = [
    "Persistent Cool",
    "Diminishing",
    "Emerging",
    "Intensifying",
    "Persistent Hot",
]

EXPECTED_HASHES = {
    "tree_train_2018_2023_ready.csv":
        "37139f64919d50ef09ccdb8453fa48a6e522894742c4517fb1b3e47fead6fa9c",
    "tree_validation_2024_ready.csv":
        "9547a58ecabe6946df6929191a851e5761c884e8d83c12ff059b3a90aafb752b",
    "tree_test_2025_LOCKED_ready.csv":
        "37d6403434e9341077c8f5d75d10ea5a4904915b924fc2dbba4581dbe5ad9464",
    "21_final_feature_list.csv":
        "6769a195b5027850ebd458a4bef7c9da0ecc43c4969bcc81f746e46bce3be9c0",
    "22b_production_lightgbm_anomaly_trained_2018_2025.joblib":
        "a3548c58ba0a708c81b4f6471e9183aa70ce9572c716ee55bf2eddafe183e24f",
    "kaduwela_grid_geometry_wgs84.geojson":
        "6b758ddaae19b54219404b86740472038cfd21391cf349ea766124678050f608",
    "23b_grid_heat_trajectory_classes.csv":
        "9f6e039329934bd6a4d4d020c01223f4be7ef932c0ec69dea47e4902893546a8",
    "23b_heat_trajectory_audit.csv":
        "359a3d86808a44ff07367421542c4cf1da1c4d1492579a377d24179f18d31f12",
}


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "models"
    / "final"
    / "22b_production_lightgbm_anomaly_trained_2018_2025.joblib"
)

FEATURE_LIST_PATH = (
    ROOT
    / "outputs"
    / "reports"
    / "21_final_feature_list.csv"
)

TRAJECTORY_PATH = (
    ROOT
    / "outputs"
    / "metrics"
    / "23b_grid_heat_trajectory_classes.csv"
)

TRAJECTORY_AUDIT_PATH = (
    ROOT
    / "outputs"
    / "reports"
    / "23b_heat_trajectory_audit.csv"
)

GRID_WGS84_PATH = (
    ROOT
    / "data"
    / "processed"
    / "kaduwela_grid_geometry_wgs84.geojson"
)

BOUNDARY_DIR = (
    ROOT
    / "data"
    / "raw"
    / "boundary"
)

EXPLAIN_DIR = ROOT / "outputs" / "explainability"
FIGURES_DIR = ROOT / "outputs" / "figures"
MAPS_DIR = ROOT / "outputs" / "maps"
REPORTS_DIR = ROOT / "outputs" / "reports"

for folder in [
    EXPLAIN_DIR,
    FIGURES_DIR,
    MAPS_DIR,
    REPORTS_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


# Global SHAP
GLOBAL_IMPORTANCE_PATH = (
    EXPLAIN_DIR
    / "24_global_shap_importance.csv"
)

RAW_SHAP_ALL_PATH = (
    EXPLAIN_DIR
    / "24_raw_shap_values_2018_2025.csv"
)

CENTERED_SHAP_ALL_PATH = (
    EXPLAIN_DIR
    / "24_centered_shap_values_2018_2025.csv"
)

YEAR_RECON_PATH = (
    EXPLAIN_DIR
    / "24_shap_reconstruction_audit_by_year.csv"
)


# Per-grid current / 2025 SHAP
LOCAL_TOP10_PATH = (
    EXPLAIN_DIR
    / "24_2025_grid_local_shap_top10.csv"
)

LOCAL_SUMMARY_PATH = (
    EXPLAIN_DIR
    / "24_2025_grid_shap_summary.csv"
)

LOCAL_WITH_TRAJECTORY_PATH = (
    EXPLAIN_DIR
    / "24_2025_grid_shap_with_trajectory.csv"
)


# Trajectory-conditioned SHAP
TRAJECTORY_ALL_YEARS_PATH = (
    EXPLAIN_DIR
    / "24_shap_by_trajectory_class_2018_2025.csv"
)

TRAJECTORY_2025_PATH = (
    EXPLAIN_DIR
    / "24_shap_by_trajectory_class_2025.csv"
)

TRAJECTORY_HEATING_PATH = (
    EXPLAIN_DIR
    / "24_trajectory_top_heat_drivers_2025.csv"
)

TRAJECTORY_COOLING_PATH = (
    EXPLAIN_DIR
    / "24_trajectory_top_cooling_drivers_2025.csv"
)


# Figures
GLOBAL_BAR_PATH = (
    FIGURES_DIR
    / "24_global_shap_importance_bar.png"
)

GLOBAL_BEESWARM_PATH = (
    FIGURES_DIR
    / "24_global_shap_beeswarm.png"
)

ANOMALY_MAP_PATH = (
    FIGURES_DIR
    / "24_2025_centered_spatial_anomaly_map.png"
)


# Map output
MAP_GEOJSON_PATH = (
    MAPS_DIR
    / "24_2025_shap_explanation_wgs84.geojson"
)


# Reports
METHOD_PATH = (
    REPORTS_DIR
    / "24_shap_methodology.json"
)

AUDIT_PATH = (
    REPORTS_DIR
    / "24_shap_audit.csv"
)

REPORT_PATH = (
    REPORTS_DIR
    / "24_shap_report.txt"
)

HASH_PATH = (
    REPORTS_DIR
    / "24_shap_hashes.csv"
)


# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print(
        "\n"
        + "=" * 128
    )
    print(
        title
    )
    print(
        "=" * 128
    )


def sha256_file(path):
    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:
        for chunk in iter(
            lambda: f.read(
                1024
                * 1024
            ),
            b"",
        ):
            h.update(
                chunk
            )

    return h.hexdigest()


def verify_hash(
    path,
    expected,
    label,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    actual = sha256_file(
        path
    )

    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual  : {actual}"
        )

    print(
        f"PASS hash: {label} = {actual}"
    )

    return actual


def require_audit_all_pass(
    path,
    label,
):
    audit = pd.read_csv(
        path
    )

    if "status" not in audit.columns:
        raise RuntimeError(
            f"{label} audit has no status column"
        )

    bad = audit[
        audit[
            "status"
        ]
        .astype(
            str
        )
        .str.upper()
        != "PASS"
    ]

    if not bad.empty:
        print(
            bad.to_string(
                index=False
            )
        )

        raise RuntimeError(
            f"{label} audit is not fully PASS"
        )


def find_unique_filename(
    filename,
):
    matches = [
        p
        for p in ROOT.rglob(
            filename
        )
        if p.is_file()
    ]

    if len(
        matches
    ) == 0:
        raise FileNotFoundError(
            "Could not find required file anywhere under project root: "
            f"{filename}"
        )

    if len(
        matches
    ) > 1:
        print(
            f"WARNING: multiple matches found for {filename}:"
        )

        for match in matches:
            print(
                f"  - {match}"
            )

        split_matches = [
            p
            for p in matches
            if "splits"
            in [
                part.lower()
                for part in p.parts
            ]
        ]

        if len(
            split_matches
        ) == 1:
            return split_matches[
                0
            ]

        raise RuntimeError(
            f"Multiple ambiguous copies of {filename} found."
        )

    return matches[
        0
    ]


def detect_feature_column(
    feature_df,
    available_columns,
):
    candidates = []

    for column in feature_df.columns:
        values = (
            feature_df[
                column
            ]
            .dropna()
            .astype(
                str
            )
            .tolist()
        )

        if (
            len(
                values
            )
            == EXPECTED_FEATURES
            and len(
                set(
                    values
                )
            )
            == EXPECTED_FEATURES
            and all(
                value
                in available_columns
                for value in values
            )
        ):
            candidates.append(
                column
            )

    if len(
        candidates
    ) != 1:
        raise RuntimeError(
            "Could not uniquely identify the ordered 93-feature column "
            f"in {FEATURE_LIST_PATH}. Candidates={candidates}"
        )

    return candidates[
        0
    ]


def scalar_expected_value(
    value,
):
    arr = np.asarray(
        value
    )

    if arr.size != 1:
        raise RuntimeError(
            "Expected one regression SHAP base value, "
            f"got shape {arr.shape}"
        )

    return float(
        arr.reshape(
            -1
        )[
            0
        ]
    )


def normalize_shap_output(
    shap_output,
    n_rows,
    n_features,
):
    if isinstance(
        shap_output,
        list,
    ):
        if len(
            shap_output
        ) != 1:
            raise RuntimeError(
                "Unexpected SHAP list length for regression: "
                f"{len(shap_output)}"
            )

        shap_output = shap_output[
            0
        ]

    arr = np.asarray(
        shap_output,
        dtype=float,
    )

    if arr.shape == (
        n_features,
        n_rows,
    ):
        arr = arr.T

    if arr.shape != (
        n_rows,
        n_features,
    ):
        raise RuntimeError(
            f"Unexpected SHAP array shape {arr.shape}; "
            f"expected {(n_rows, n_features)}"
        )

    return arr


def safe_name(
    text,
):
    value = "".join(
        char
        if (
            char.isalnum()
            or char
            in "_-"
        )
        else "_"
        for char
        in str(
            text
        )
    )

    while "__" in value:
        value = value.replace(
            "__",
            "_",
        )

    return value.strip(
        "_"
    )


def choose_real_boundary(
    grid_wgs84,
):
    """
    Select the readable vector boundary file that best overlaps the actual
    Kaduwela grid coverage. This avoids trusting only the filename.
    """
    if not BOUNDARY_DIR.exists():
        raise FileNotFoundError(
            f"Boundary directory does not exist: {BOUNDARY_DIR}"
        )

    vector_extensions = {
        ".kml",
        ".geojson",
        ".json",
        ".gpkg",
        ".shp",
    }

    candidates = [
        path
        for path
        in BOUNDARY_DIR.rglob(
            "*"
        )
        if (
            path.is_file()
            and path.suffix.lower()
            in vector_extensions
        )
    ]

    if not candidates:
        raise FileNotFoundError(
            "No vector boundary file found under "
            f"{BOUNDARY_DIR}"
        )

    grid_metric = grid_wgs84.to_crs(
        "EPSG:32644"
    )

    grid_union = (
        grid_metric.geometry
        .union_all()
    )

    grid_area = float(
        grid_union.area
    )

    scored = []

    for path in candidates:
        try:
            boundary = gpd.read_file(
                path
            )

            if boundary.empty:
                continue

            boundary = boundary[
                boundary.geometry.notna()
                & ~boundary.geometry.is_empty
            ].copy()

            if boundary.empty:
                continue

            if boundary.crs is None:
                warnings.warn(
                    "Boundary file has no CRS metadata and was skipped: "
                    f"{path}"
                )
                continue

            boundary = boundary.to_crs(
                "EPSG:4326"
            )

            boundary_metric = boundary.to_crs(
                "EPSG:32644"
            )

            boundary_union = (
                boundary_metric.geometry
                .union_all()
            )

            intersection_area = float(
                boundary_union.intersection(
                    grid_union
                ).area
            )

            overlap_fraction = (
                intersection_area
                / grid_area
                if grid_area
                > 0
                else 0.0
            )

            scored.append(
                (
                    overlap_fraction,
                    path,
                    boundary,
                )
            )

        except Exception as exc:
            print(
                "Boundary candidate skipped: "
                f"{path} -> {exc}"
            )

    if not scored:
        raise RuntimeError(
            "Boundary files were found, but none could be safely read."
        )

    scored.sort(
        key=lambda item: item[
            0
        ],
        reverse=True,
    )

    overlap_fraction, path, boundary = scored[
        0
    ]

    if overlap_fraction < 0.80:
        raise RuntimeError(
            "Best boundary does not sufficiently overlap the Kaduwela grid. "
            f"Best overlap={overlap_fraction:.4f}, file={path}"
        )

    print(
        f"Selected real boundary : {path}"
    )

    print(
        f"Grid-overlap fraction  : {overlap_fraction:.4f}"
    )

    return (
        boundary,
        path,
        overlap_fraction,
    )


def plot_spatial_continuous(
    gdf,
    column,
    boundary,
    title,
    output_path,
    legend_label,
):
    fig, ax = plt.subplots(
        figsize=(
            9,
            9,
        )
    )

    gdf.plot(
        column=column,
        legend=True,
        ax=ax,
    )

    boundary.boundary.plot(
        ax=ax,
        linewidth=1.4,
    )

    ax.set_title(
        title
    )

    ax.set_axis_off()

    if len(
        fig.axes
    ) > 1:
        try:
            fig.axes[
                -1
            ].set_ylabel(
                legend_label
            )
        except Exception:
            pass

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def build_trajectory_shap_summary(
    meta_df,
    shap_matrix,
    features,
    trajectory_col="trajectory_class",
):
    """
    One row per trajectory class + feature.

    IMPORTANT:
    Do NOT concatenate the SHAP matrix with metadata columns using raw feature
    names because 'year' is both a valid model feature and a metadata column.
    A duplicate 'year' column would make class_rows['year'] return both the
    calendar year and the SHAP contribution, corrupting trajectory summaries.
    We therefore apply class masks directly to the numeric SHAP matrix.
    """
    meta = (
        meta_df[
            [
                GRID_COL,
                YEAR_COL,
                trajectory_col,
            ]
        ]
        .reset_index(
            drop=True
        )
        .copy()
    )

    matrix = np.asarray(
        shap_matrix,
        dtype=float,
    )

    if matrix.shape != (
        len(meta),
        len(features),
    ):
        raise RuntimeError(
            "Trajectory SHAP matrix shape mismatch: "
            f"actual={matrix.shape}, expected={(len(meta), len(features))}"
        )

    rows = []

    for class_name in TRAJECTORY_CLASSES:
        class_mask = (
            meta[
                trajectory_col
            ]
            .astype(str)
            .eq(
                class_name
            )
            .to_numpy()
        )

        class_matrix = matrix[
            class_mask
        ]

        if class_matrix.shape[
            0
        ] == 0:
            continue

        n_rows = int(
            class_matrix.shape[
                0
            ]
        )

        for feature_idx, feature in enumerate(
            features
        ):
            values = class_matrix[
                :,
                feature_idx,
            ]

            rows.append(
                {
                    "trajectory_class": class_name,
                    "feature": feature,
                    "row_count": n_rows,
                    "mean_centered_shap_c": float(
                        np.mean(
                            values
                        )
                    ),
                    "median_centered_shap_c": float(
                        np.median(
                            values
                        )
                    ),
                    "mean_abs_centered_shap_c": float(
                        np.mean(
                            np.abs(
                                values
                            )
                        )
                    ),
                    "pct_positive_contribution": float(
                        np.mean(
                            values
                            > 0
                        )
                        * 100.0
                    ),
                    "pct_negative_contribution": float(
                        np.mean(
                            values
                            < 0
                        )
                        * 100.0
                    ),
                }
            )

    summary = pd.DataFrame(
        rows
    )

    if summary.empty:
        raise RuntimeError(
            "Trajectory-conditioned SHAP summary is empty."
        )

    summary[
        "rank_by_mean_absolute_contribution_within_class"
    ] = (
        summary.groupby(
            "trajectory_class"
        )[
            "mean_abs_centered_shap_c"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(
            int
        )
    )

    return summary


def audit_trajectory_shap_summary(
    summary,
    reference_shap_matrix,
    features,
    label,
):
    """
    Check that the trajectory-class summaries exactly aggregate back to the
    reference SHAP matrix. This catches metadata/feature column collisions and
    class-partition alignment errors.
    """
    matrix = np.asarray(
        reference_shap_matrix,
        dtype=float,
    )

    if matrix.shape[
        1
    ] != len(
        features
    ):
        raise RuntimeError(
            f"{label}: reference SHAP feature count mismatch."
        )

    total_rows = int(
        matrix.shape[
            0
        ]
    )

    max_mean_error = 0.0
    max_mean_abs_error = 0.0

    for feature_idx, feature in enumerate(
        features
    ):
        part = summary[
            summary[
                "feature"
            ]
            == feature
        ]

        if part.empty:
            raise RuntimeError(
                f"{label}: missing trajectory summary for feature {feature}"
            )

        summarized_rows = int(
            part[
                "row_count"
            ].sum()
        )

        if summarized_rows != total_rows:
            raise RuntimeError(
                f"{label}: feature {feature} summarizes {summarized_rows} rows, "
                f"expected {total_rows}."
            )

        weights = part[
            "row_count"
        ].to_numpy(
            dtype=float
        )

        reconstructed_mean = float(
            np.average(
                part[
                    "mean_centered_shap_c"
                ].to_numpy(
                    dtype=float
                ),
                weights=weights,
            )
        )

        reconstructed_mean_abs = float(
            np.average(
                part[
                    "mean_abs_centered_shap_c"
                ].to_numpy(
                    dtype=float
                ),
                weights=weights,
            )
        )

        reference_values = matrix[
            :,
            feature_idx,
        ]

        reference_mean = float(
            np.mean(
                reference_values
            )
        )

        reference_mean_abs = float(
            np.mean(
                np.abs(
                    reference_values
                )
            )
        )

        max_mean_error = max(
            max_mean_error,
            abs(
                reconstructed_mean
                - reference_mean
            ),
        )

        max_mean_abs_error = max(
            max_mean_abs_error,
            abs(
                reconstructed_mean_abs
                - reference_mean_abs
            ),
        )

    return (
        max_mean_error,
        max_mean_abs_error,
    )


def top_signed_drivers(
    summary,
    direction,
    top_k,
):
    rows = []

    for class_name in TRAJECTORY_CLASSES:
        part = summary[
            summary[
                "trajectory_class"
            ]
            == class_name
        ].copy()

        if direction == "heating":
            part = (
                part[
                    part[
                        "mean_centered_shap_c"
                    ]
                    > 0
                ]
                .sort_values(
                    "mean_centered_shap_c",
                    ascending=False,
                )
            )

        elif direction == "cooling":
            part = (
                part[
                    part[
                        "mean_centered_shap_c"
                    ]
                    < 0
                ]
                .sort_values(
                    "mean_centered_shap_c",
                    ascending=True,
                )
            )

        else:
            raise ValueError(
                f"Unknown direction: {direction}"
            )

        part = part.head(
            top_k
        ).copy()

        part.insert(
            1,
            "rank",
            np.arange(
                1,
                len(
                    part
                )
                + 1,
            ),
        )

        rows.append(
            part
        )

    if not rows:
        return pd.DataFrame()

    return pd.concat(
        rows,
        ignore_index=True,
    )


def save_trajectory_driver_figure(
    class_name,
    class_summary_2025,
    output_path,
):
    part = (
        class_summary_2025[
            class_summary_2025[
                "trajectory_class"
            ]
            == class_name
        ]
        .copy()
    )

    if part.empty:
        return

    part[
        "abs_mean"
    ] = (
        part[
            "mean_centered_shap_c"
        ]
        .abs()
    )

    part = (
        part.sort_values(
            "abs_mean",
            ascending=False,
        )
        .head(
            12
        )
        .sort_values(
            "mean_centered_shap_c"
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            9,
            6,
        )
    )

    ax.barh(
        part[
            "feature"
        ],
        part[
            "mean_centered_shap_c"
        ],
    )

    ax.axvline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xlabel(
        "Mean centered SHAP contribution (°C)"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        f"2025 Model-Based Heat Associations — {class_name} Grids"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# 0. VERIFY FROZEN INPUTS
# =============================================================================

section(
    "STEP 24 - SHAP EXPLAINABILITY + TRAJECTORY-CONDITIONED SHAP"
)

train_path = find_unique_filename(
    "tree_train_2018_2023_ready.csv"
)

validation_path = find_unique_filename(
    "tree_validation_2024_ready.csv"
)

test_path = find_unique_filename(
    "tree_test_2025_LOCKED_ready.csv"
)

verify_hash(
    train_path,
    EXPECTED_HASHES[
        "tree_train_2018_2023_ready.csv"
    ],
    "2018-2023 model-ready data",
)

verify_hash(
    validation_path,
    EXPECTED_HASHES[
        "tree_validation_2024_ready.csv"
    ],
    "2024 model-ready data",
)

verify_hash(
    test_path,
    EXPECTED_HASHES[
        "tree_test_2025_LOCKED_ready.csv"
    ],
    "2025 historical model-ready data",
)

verify_hash(
    FEATURE_LIST_PATH,
    EXPECTED_HASHES[
        "21_final_feature_list.csv"
    ],
    "Frozen Step-21 feature list",
)

verify_hash(
    MODEL_PATH,
    EXPECTED_HASHES[
        "22b_production_lightgbm_anomaly_trained_2018_2025.joblib"
    ],
    "Step-22B production LightGBM",
)

verify_hash(
    GRID_WGS84_PATH,
    EXPECTED_HASHES[
        "kaduwela_grid_geometry_wgs84.geojson"
    ],
    "WGS84 grid geometry",
)

verify_hash(
    TRAJECTORY_PATH,
    EXPECTED_HASHES[
        "23b_grid_heat_trajectory_classes.csv"
    ],
    "Step-23B v2 trajectory classes",
)

verify_hash(
    TRAJECTORY_AUDIT_PATH,
    EXPECTED_HASHES[
        "23b_heat_trajectory_audit.csv"
    ],
    "Step-23B v2 audit",
)

require_audit_all_pass(
    TRAJECTORY_AUDIT_PATH,
    "Step 23B",
)

print(
    "\nPASS: Step-23B v2 audit is fully PASS."
)

print(
    "\nExplainability scope:"
)

print(
    "- final production LightGBM spatial-anomaly model"
)

print(
    "- global + per-grid + spatial + trajectory-conditioned SHAP"
)

print(
    "- no future 2026-2030 SHAP yet"
)

print(
    "- no new test score"
)


# =============================================================================
# 1. LOAD + VALIDATE PRODUCTION EXPLANATION DATA
# =============================================================================

section(
    "1. LOAD + VALIDATE 2018-2025 PRODUCTION EXPLANATION DATA"
)

blocks = [
    pd.read_csv(
        train_path,
        dtype={
            GRID_COL: str,
        },
    ),
    pd.read_csv(
        validation_path,
        dtype={
            GRID_COL: str,
        },
    ),
    pd.read_csv(
        test_path,
        dtype={
            GRID_COL: str,
        },
    ),
]

data = pd.concat(
    blocks,
    ignore_index=True,
)

if len(
    data
) != EXPECTED_PRODUCTION_ROWS:
    raise RuntimeError(
        f"Rows={len(data)}, expected={EXPECTED_PRODUCTION_ROWS}"
    )

if (
    GRID_COL
    not in data.columns
    or YEAR_COL
    not in data.columns
):
    raise RuntimeError(
        "grid_id/year missing from production data"
    )

data[
    YEAR_COL
] = pd.to_numeric(
    data[
        YEAR_COL
    ],
    errors="raise",
).astype(
    int
)

years = sorted(
    data[
        YEAR_COL
    ].unique().tolist()
)

if years != EXPECTED_YEARS:
    raise RuntimeError(
        f"Years={years}, expected={EXPECTED_YEARS}"
    )

if data[
    GRID_COL
].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        "Production grid count mismatch"
    )

if data.duplicated(
    [
        GRID_COL,
        YEAR_COL,
    ]
).any():
    raise RuntimeError(
        "Duplicate grid-year rows found"
    )

rows_per_year = (
    data.groupby(
        YEAR_COL
    )[
        GRID_COL
    ]
    .nunique()
)

if not (
    rows_per_year
    == EXPECTED_GRIDS
).all():
    raise RuntimeError(
        "Every production year must have all 2,194 grids"
    )

feature_df = pd.read_csv(
    FEATURE_LIST_PATH
)

feature_column = detect_feature_column(
    feature_df,
    set(
        data.columns
    ),
)

features = (
    feature_df[
        feature_column
    ]
    .astype(
        str
    )
    .tolist()
)

if len(
    features
) != EXPECTED_FEATURES:
    raise RuntimeError(
        f"Feature count={len(features)}, expected={EXPECTED_FEATURES}"
    )

X = data[
    features
].copy()

if X.isna().any().any():
    missing_count = int(
        X.isna().sum().sum()
    )

    raise RuntimeError(
        "Frozen model-ready feature matrix contains "
        f"{missing_count} missing values"
    )

if not np.isfinite(
    X.to_numpy(
        dtype=float
    )
).all():
    raise RuntimeError(
        "Feature matrix contains non-finite values"
    )

trajectory = pd.read_csv(
    TRAJECTORY_PATH,
    dtype={
        GRID_COL: str,
    },
)

required_trajectory_columns = {
    GRID_COL,
    AREA_COL,
    "trajectory_class",
    "relative_sen_slope_percentile_per_year",
    "sen_slope_c_per_year",
    "mk_p_value",
    "mk_fdr_q_value",
}

missing_trajectory = (
    required_trajectory_columns
    - set(
        trajectory.columns
    )
)

if missing_trajectory:
    raise RuntimeError(
        "Step-23B trajectory file missing columns: "
        f"{sorted(missing_trajectory)}"
    )

if len(
    trajectory
) != EXPECTED_GRIDS:
    raise RuntimeError(
        f"Trajectory rows={len(trajectory)}, expected={EXPECTED_GRIDS}"
    )

if trajectory[
    GRID_COL
].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        "Trajectory grid IDs are not unique"
    )

unexpected_classes = sorted(
    set(
        trajectory[
            "trajectory_class"
        ].unique()
    )
    - set(
        TRAJECTORY_CLASSES
    )
)

if unexpected_classes:
    raise RuntimeError(
        f"Unexpected trajectory classes: {unexpected_classes}"
    )

# -------------------------------------------------------------------------
# Join Step-23B trajectory information without creating area_name_x/y.
# The model-ready data already contains area_name in this project.
# If area_name exists in both sources, verify that the grid-to-area mapping
# agrees and merge only the non-duplicate trajectory fields.
# -------------------------------------------------------------------------

trajectory_join_columns = [
    GRID_COL,
    "trajectory_class",
    "relative_sen_slope_percentile_per_year",
    "sen_slope_c_per_year",
    "mk_p_value",
    "mk_fdr_q_value",
]

if AREA_COL in data.columns:
    production_area_map = (
        data[
            [
                GRID_COL,
                AREA_COL,
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    if production_area_map[
        GRID_COL
    ].duplicated().any():
        raise RuntimeError(
            "A grid_id maps to more than one area_name in the model-ready data."
        )

    trajectory_area_map = (
        trajectory[
            [
                GRID_COL,
                AREA_COL,
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    area_check = production_area_map.merge(
        trajectory_area_map,
        on=GRID_COL,
        how="outer",
        suffixes=(
            "_production",
            "_trajectory",
        ),
        validate="one_to_one",
        indicator=True,
    )

    if not (
        area_check[
            "_merge"
        ]
        == "both"
    ).all():
        raise RuntimeError(
            "Production data and Step-23B trajectory data have different grid_id sets."
        )

    production_area_col = (
        f"{AREA_COL}_production"
    )
    trajectory_area_col = (
        f"{AREA_COL}_trajectory"
    )

    area_values_match = (
        area_check[
            production_area_col
        ]
        .fillna(
            ""
        )
        .astype(
            str
        )
        == area_check[
            trajectory_area_col
        ]
        .fillna(
            ""
        )
        .astype(
            str
        )
    )

    if not area_values_match.all():
        mismatch_examples = area_check.loc[
            ~area_values_match,
            [
                GRID_COL,
                production_area_col,
                trajectory_area_col,
            ],
        ].head(
            10
        )

        print(
            mismatch_examples.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "area_name mapping differs between model-ready data and Step-23B trajectory data."
        )

    print(
        "PASS: existing model-ready area_name matches Step-23B grid-to-area mapping."
    )

else:
    trajectory_join_columns.insert(
        1,
        AREA_COL,
    )

data = data.merge(
    trajectory[
        trajectory_join_columns
    ],
    on=GRID_COL,
    how="left",
    validate="many_to_one",
)

if AREA_COL not in data.columns:
    raise RuntimeError(
        "area_name is unavailable after trajectory join."
    )

if data[
    "trajectory_class"
].isna().any():
    raise RuntimeError(
        "Some production rows did not receive a trajectory class"
    )

print(
    f"Rows             : {len(data):,}"
)

print(
    f"Unique grids     : {data[GRID_COL].nunique():,}"
)

print(
    f"Years            : {years}"
)

print(
    f"Rows/year        : {rows_per_year.min():,} to {rows_per_year.max():,}"
)

print(
    f"Frozen predictors: {len(features)}"
)

print(
    f"Feature column   : {feature_column}"
)

print(
    "Trajectory classes:"
)

print(
    trajectory[
        "trajectory_class"
    ]
    .value_counts()
    .reindex(
        TRAJECTORY_CLASSES
    )
    .to_string()
)


# =============================================================================
# 2. LOAD FINAL PRODUCTION MODEL
# =============================================================================

section(
    "2. LOAD FINAL PRODUCTION LIGHTGBM"
)

model = joblib.load(
    MODEL_PATH
)

model_feature_names = getattr(
    model,
    "feature_name_",
    None,
)

if model_feature_names is not None:
    model_feature_names = list(
        model_feature_names
    )

    if model_feature_names != features:
        raise RuntimeError(
            "Production model feature order does not match "
            "the frozen feature list"
        )

print(
    f"Model class  : {model.__class__.__name__}"
)

print(
    "Model target : within-year spatial LST anomaly"
)

print(
    "PASS: production model and ordered feature list aligned."
)


# =============================================================================
# 3. COMPUTE TREE SHAP
# =============================================================================

section(
    "3. COMPUTE TREE SHAP FOR ALL 2018-2025 ROWS"
)

explainer = shap.TreeExplainer(
    model
)

raw_shap = normalize_shap_output(
    explainer.shap_values(
        X
    ),
    n_rows=len(
        X
    ),
    n_features=len(
        features
    ),
)

base_value = scalar_expected_value(
    explainer.expected_value
)

raw_prediction = np.asarray(
    model.predict(
        X
    ),
    dtype=float,
).reshape(
    -1
)

raw_reconstruction = (
    base_value
    + raw_shap.sum(
        axis=1
    )
)

max_raw_reconstruction_error = float(
    np.max(
        np.abs(
            raw_prediction
            - raw_reconstruction
        )
    )
)

print(
    f"SHAP base value                    : {base_value:.12f} °C"
)

print(
    "Max raw SHAP reconstruction error : "
    f"{max_raw_reconstruction_error:.12e} °C"
)

if max_raw_reconstruction_error > 1e-6:
    raise RuntimeError(
        "Raw SHAP reconstruction failed tolerance."
    )


# =============================================================================
# 4. CENTER SHAP BY YEAR
# =============================================================================

section(
    "4. YEAR-CENTER SHAP TO MATCH FINAL SPATIAL-ANOMALY INFERENCE"
)

centered_shap = np.empty_like(
    raw_shap
)

centered_prediction = np.empty_like(
    raw_prediction
)

year_values = data[
    YEAR_COL
].to_numpy()

year_audit_rows = []

for year in EXPECTED_YEARS:
    idx = np.where(
        year_values
        == year
    )[
        0
    ]

    if len(
        idx
    ) != EXPECTED_GRIDS:
        raise RuntimeError(
            f"Year {year} rows={len(idx)}, expected={EXPECTED_GRIDS}"
        )

    raw_year_prediction = raw_prediction[
        idx
    ]

    centered_prediction[
        idx
    ] = (
        raw_year_prediction
        - raw_year_prediction.mean()
    )

    year_feature_means = (
        raw_shap[
            idx
        ]
        .mean(
            axis=0
        )
    )

    centered_shap[
        idx
    ] = (
        raw_shap[
            idx
        ]
        - year_feature_means
    )

    reconstructed = (
        centered_shap[
            idx
        ]
        .sum(
            axis=1
        )
    )

    max_error = float(
        np.max(
            np.abs(
                centered_prediction[
                    idx
                ]
                - reconstructed
            )
        )
    )

    year_audit_rows.append(
        {
            "year": year,
            "rows": len(
                idx
            ),
            "raw_prediction_mean_c": float(
                raw_year_prediction.mean()
            ),
            "centered_prediction_mean_c": float(
                centered_prediction[
                    idx
                ].mean()
            ),
            "centered_shap_sum_mean_c": float(
                reconstructed.mean()
            ),
            "max_centered_reconstruction_error_c": max_error,
        }
    )

year_reconstruction = pd.DataFrame(
    year_audit_rows
)

year_reconstruction.to_csv(
    YEAR_RECON_PATH,
    index=False,
)

max_centered_reconstruction_error = float(
    year_reconstruction[
        "max_centered_reconstruction_error_c"
    ].max()
)

print(
    year_reconstruction.to_string(
        index=False
    )
)

print(
    "\nMax centered SHAP reconstruction error: "
    f"{max_centered_reconstruction_error:.12e} °C"
)

if max_centered_reconstruction_error > 1e-6:
    raise RuntimeError(
        "Centered SHAP reconstruction failed tolerance."
    )


# =============================================================================
# 5. GLOBAL SHAP IMPORTANCE
# =============================================================================

section(
    "5. GLOBAL SHAP IMPORTANCE"
)

global_importance = pd.DataFrame(
    {
        "feature": features,
        "mean_abs_centered_shap_c": np.mean(
            np.abs(
                centered_shap
            ),
            axis=0,
        ),
        "mean_centered_shap_c": np.mean(
            centered_shap,
            axis=0,
        ),
        "mean_abs_raw_shap_c": np.mean(
            np.abs(
                raw_shap
            ),
            axis=0,
        ),
    }
).sort_values(
    "mean_abs_centered_shap_c",
    ascending=False,
).reset_index(
    drop=True
)

global_importance.insert(
    0,
    "rank",
    np.arange(
        1,
        len(
            global_importance
        )
        + 1,
    ),
)

global_importance.to_csv(
    GLOBAL_IMPORTANCE_PATH,
    index=False,
)

print(
    global_importance.head(
        GLOBAL_TOP_N
    ).to_string(
        index=False
    )
)

print(
    f"\nSaved: {GLOBAL_IMPORTANCE_PATH}"
)


# =============================================================================
# 6. SAVE FULL RAW + CENTERED SHAP MATRICES
# =============================================================================

section(
    "6. SAVE FULL RAW + CENTERED SHAP MATRICES"
)

meta_columns = [
    GRID_COL,
    YEAR_COL,
    AREA_COL,
    "trajectory_class",
]

id_meta = data[
    meta_columns
].reset_index(
    drop=True
)

raw_shap_df = pd.DataFrame(
    raw_shap,
    columns=[
        f"raw_shap__{feature}"
        for feature
        in features
    ],
)

raw_out = pd.concat(
    [
        id_meta,
        pd.Series(
            raw_prediction,
            name="raw_model_prediction_anomaly_c",
        ),
        raw_shap_df,
    ],
    axis=1,
)

raw_out.to_csv(
    RAW_SHAP_ALL_PATH,
    index=False,
)

centered_shap_df = pd.DataFrame(
    centered_shap,
    columns=[
        f"centered_shap__{feature}"
        for feature
        in features
    ],
)

centered_out = pd.concat(
    [
        id_meta,
        pd.Series(
            centered_prediction,
            name="centered_predicted_spatial_anomaly_c",
        ),
        centered_shap_df,
    ],
    axis=1,
)

centered_out.to_csv(
    CENTERED_SHAP_ALL_PATH,
    index=False,
)

print(
    f"Saved: {RAW_SHAP_ALL_PATH}"
)

print(
    f"Saved: {CENTERED_SHAP_ALL_PATH}"
)


# =============================================================================
# 7. 2025 PER-GRID LOCAL SHAP
# =============================================================================

section(
    "7. BUILD 2025 PER-GRID LOCAL SHAP EXPLANATIONS"
)

local_indices = np.where(
    data[
        YEAR_COL
    ].to_numpy()
    == LOCAL_YEAR
)[
    0
]

if len(
    local_indices
) != EXPECTED_GRIDS:
    raise RuntimeError(
        f"{LOCAL_YEAR} rows={len(local_indices)}, expected={EXPECTED_GRIDS}"
    )

local_data = (
    data.iloc[
        local_indices
    ]
    .reset_index(
        drop=True
    )
)

local_X = (
    X.iloc[
        local_indices
    ]
    .reset_index(
        drop=True
    )
)

local_centered_shap = centered_shap[
    local_indices
]

local_centered_prediction = centered_prediction[
    local_indices
]

top_rows = []
summary_rows = []

for i in range(
    len(
        local_data
    )
):
    values = local_centered_shap[
        i
    ]

    order = np.argsort(
        np.abs(
            values
        )
    )[
        ::-1
    ]

    grid_id = str(
        local_data.iloc[
            i
        ][
            GRID_COL
        ]
    )

    top_indices = order[
        :LOCAL_TOP_K
    ]

    for rank, feature_idx in enumerate(
        top_indices,
        start=1,
    ):
        contribution = float(
            values[
                feature_idx
            ]
        )

        top_rows.append(
            {
                GRID_COL: grid_id,
                YEAR_COL: LOCAL_YEAR,
                AREA_COL: local_data.iloc[
                    i
                ][
                    AREA_COL
                ],
                "trajectory_class": local_data.iloc[
                    i
                ][
                    "trajectory_class"
                ],
                "rank_by_absolute_contribution": rank,
                "feature": features[
                    feature_idx
                ],
                "feature_value": float(
                    local_X.iloc[
                        i,
                        feature_idx,
                    ]
                ),
                "centered_shap_contribution_c": contribution,
                "effect_direction": (
                    "Hotter relative to annual level"
                    if contribution
                    > 0
                    else "Cooler relative to annual level"
                    if contribution
                    < 0
                    else "Neutral"
                ),
            }
        )

    positive_indices = np.where(
        values
        > 0
    )[
        0
    ]

    negative_indices = np.where(
        values
        < 0
    )[
        0
    ]

    if len(
        positive_indices
    ) > 0:
        heating_idx = positive_indices[
            np.argmax(
                values[
                    positive_indices
                ]
            )
        ]

        top_heating_feature = features[
            heating_idx
        ]

        top_heating_feature_value = float(
            local_X.iloc[
                i,
                heating_idx,
            ]
        )

        top_heating_contribution = float(
            values[
                heating_idx
            ]
        )

    else:
        top_heating_feature = None
        top_heating_feature_value = np.nan
        top_heating_contribution = np.nan

    if len(
        negative_indices
    ) > 0:
        cooling_idx = negative_indices[
            np.argmin(
                values[
                    negative_indices
                ]
            )
        ]

        top_cooling_feature = features[
            cooling_idx
        ]

        top_cooling_feature_value = float(
            local_X.iloc[
                i,
                cooling_idx,
            ]
        )

        top_cooling_contribution = float(
            values[
                cooling_idx
            ]
        )

    else:
        top_cooling_feature = None
        top_cooling_feature_value = np.nan
        top_cooling_contribution = np.nan

    dominant_idx = order[
        0
    ]

    summary_rows.append(
        {
            GRID_COL: grid_id,
            YEAR_COL: LOCAL_YEAR,
            AREA_COL: local_data.iloc[
                i
            ][
                AREA_COL
            ],
            "trajectory_class": local_data.iloc[
                i
            ][
                "trajectory_class"
            ],
            "relative_sen_slope_percentile_per_year": float(
                local_data.iloc[
                    i
                ][
                    "relative_sen_slope_percentile_per_year"
                ]
            ),
            "absolute_sen_slope_c_per_year": float(
                local_data.iloc[
                    i
                ][
                    "sen_slope_c_per_year"
                ]
            ),
            "mk_p_value": float(
                local_data.iloc[
                    i
                ][
                    "mk_p_value"
                ]
            ),
            "mk_fdr_q_value": float(
                local_data.iloc[
                    i
                ][
                    "mk_fdr_q_value"
                ]
            ),
            "centered_predicted_spatial_anomaly_c": float(
                local_centered_prediction[
                    i
                ]
            ),
            "dominant_feature": features[
                dominant_idx
            ],
            "dominant_feature_value": float(
                local_X.iloc[
                    i,
                    dominant_idx,
                ]
            ),
            "dominant_centered_shap_c": float(
                values[
                    dominant_idx
                ]
            ),
            "top_heating_feature": top_heating_feature,
            "top_heating_feature_value": top_heating_feature_value,
            "top_heating_contribution_c": top_heating_contribution,
            "top_cooling_feature": top_cooling_feature,
            "top_cooling_feature_value": top_cooling_feature_value,
            "top_cooling_contribution_c": top_cooling_contribution,
            "local_shap_sum_c": float(
                values.sum()
            ),
            "local_reconstruction_error_c": float(
                local_centered_prediction[
                    i
                ]
                - values.sum()
            ),
        }
    )

local_top10 = pd.DataFrame(
    top_rows
)

local_summary = pd.DataFrame(
    summary_rows
)

local_top10.to_csv(
    LOCAL_TOP10_PATH,
    index=False,
)

local_summary.to_csv(
    LOCAL_SUMMARY_PATH,
    index=False,
)

local_summary.to_csv(
    LOCAL_WITH_TRAJECTORY_PATH,
    index=False,
)

print(
    f"2025 grids explained : {len(local_summary):,}"
)

print(
    f"Top-{LOCAL_TOP_K} local rows : {len(local_top10):,}"
)

print(
    f"Saved: {LOCAL_TOP10_PATH}"
)

print(
    f"Saved: {LOCAL_SUMMARY_PATH}"
)

print(
    f"Saved: {LOCAL_WITH_TRAJECTORY_PATH}"
)


# =============================================================================
# 8. TRAJECTORY-CONDITIONED SHAP
# =============================================================================

section(
    "8. TRAJECTORY-CONDITIONED SHAP SUMMARIES"
)

trajectory_summary_all = build_trajectory_shap_summary(
    meta_df=data[
        [
            GRID_COL,
            YEAR_COL,
            "trajectory_class",
        ]
    ],
    shap_matrix=centered_shap,
    features=features,
)

trajectory_summary_all.to_csv(
    TRAJECTORY_ALL_YEARS_PATH,
    index=False,
)

local_meta = local_data[
    [
        GRID_COL,
        YEAR_COL,
        "trajectory_class",
    ]
].copy()

trajectory_summary_2025 = build_trajectory_shap_summary(
    meta_df=local_meta,
    shap_matrix=local_centered_shap,
    features=features,
)

trajectory_summary_2025.to_csv(
    TRAJECTORY_2025_PATH,
    index=False,
)

trajectory_all_mean_error, trajectory_all_mean_abs_error = (
    audit_trajectory_shap_summary(
        summary=trajectory_summary_all,
        reference_shap_matrix=centered_shap,
        features=features,
        label="2018-2025 trajectory SHAP",
    )
)

trajectory_2025_mean_error, trajectory_2025_mean_abs_error = (
    audit_trajectory_shap_summary(
        summary=trajectory_summary_2025,
        reference_shap_matrix=local_centered_shap,
        features=features,
        label="2025 trajectory SHAP",
    )
)

print(
    "\nTrajectory-SHAP aggregation audit:"
)

print(
    "2018-2025 max weighted-mean reconstruction error     : "
    f"{trajectory_all_mean_error:.12e} °C"
)

print(
    "2018-2025 max weighted-mean-absolute reconstruction : "
    f"{trajectory_all_mean_abs_error:.12e} °C"
)

print(
    "2025 max weighted-mean reconstruction error          : "
    f"{trajectory_2025_mean_error:.12e} °C"
)

print(
    "2025 max weighted-mean-absolute reconstruction      : "
    f"{trajectory_2025_mean_abs_error:.12e} °C"
)

if max(
    trajectory_all_mean_error,
    trajectory_all_mean_abs_error,
    trajectory_2025_mean_error,
    trajectory_2025_mean_abs_error,
) > 1e-10:
    raise RuntimeError(
        "Trajectory-conditioned SHAP aggregation audit failed."
    )

top_heating = top_signed_drivers(
    summary=trajectory_summary_2025,
    direction="heating",
    top_k=TRAJECTORY_TOP_K,
)

top_cooling = top_signed_drivers(
    summary=trajectory_summary_2025,
    direction="cooling",
    top_k=TRAJECTORY_TOP_K,
)

top_heating.to_csv(
    TRAJECTORY_HEATING_PATH,
    index=False,
)

top_cooling.to_csv(
    TRAJECTORY_COOLING_PATH,
    index=False,
)

print(
    f"Saved: {TRAJECTORY_ALL_YEARS_PATH}"
)

print(
    f"Saved: {TRAJECTORY_2025_PATH}"
)

print(
    f"Saved: {TRAJECTORY_HEATING_PATH}"
)

print(
    f"Saved: {TRAJECTORY_COOLING_PATH}"
)

print(
    "\nTop 5 model-based heating associations by trajectory class:"
)

for class_name in TRAJECTORY_CLASSES:
    print(
        f"\n[{class_name}]"
    )

    class_rows = top_heating[
        top_heating[
            "trajectory_class"
        ]
        == class_name
    ].head(
        5
    )

    if class_rows.empty:
        print(
            "No positive mean SHAP association."
        )
    else:
        print(
            class_rows[
                [
                    "rank",
                    "feature",
                    "mean_centered_shap_c",
                    "mean_abs_centered_shap_c",
                    "pct_positive_contribution",
                ]
            ].to_string(
                index=False
            )
        )


# =============================================================================
# 9. GLOBAL SHAP FIGURES
# =============================================================================

section(
    "9. SAVE REPORT-QUALITY GLOBAL SHAP FIGURES"
)

top_global = (
    global_importance.head(
        GLOBAL_TOP_N
    )
    .iloc[
        ::-1
    ]
)

fig, ax = plt.subplots(
    figsize=(
        9,
        7,
    )
)

ax.barh(
    top_global[
        "feature"
    ],
    top_global[
        "mean_abs_centered_shap_c"
    ],
)

ax.set_xlabel(
    "Mean |centered SHAP contribution| (°C)"
)

ax.set_ylabel(
    "Feature"
)

ax.set_title(
    "Global Drivers of Spatial LST Anomaly in Kaduwela (2018–2025)"
)

ax.grid(
    axis="x",
    alpha=0.2,
)

plt.tight_layout()

plt.savefig(
    GLOBAL_BAR_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)

rng = np.random.default_rng(
    RANDOM_SEED
)

if len(
    X
) > BEESWARM_MAX_ROWS:
    sample_idx = np.sort(
        rng.choice(
            len(
                X
            ),
            size=BEESWARM_MAX_ROWS,
            replace=False,
        )
    )
else:
    sample_idx = np.arange(
        len(
            X
        )
    )

plt.figure(
    figsize=(
        10,
        8,
    )
)

shap.summary_plot(
    centered_shap[
        sample_idx
    ],
    X.iloc[
        sample_idx
    ],
    feature_names=features,
    max_display=GLOBAL_TOP_N,
    show=False,
)

plt.title(
    "SHAP Direction and Magnitude for Kaduwela Spatial Heat Anomaly"
)

plt.tight_layout()

plt.savefig(
    GLOBAL_BEESWARM_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    f"Saved: {GLOBAL_BAR_PATH}"
)

print(
    f"Saved: {GLOBAL_BEESWARM_PATH}"
)


# =============================================================================
# 10. TRAJECTORY SHAP FIGURES
# =============================================================================

section(
    "10. SAVE TRAJECTORY-CONDITIONED SHAP FIGURES"
)

trajectory_figure_paths = []

for class_name in TRAJECTORY_CLASSES:
    output_path = (
        FIGURES_DIR
        / (
            "24_shap_trajectory_"
            + safe_name(
                class_name
            ).lower()
            + ".png"
        )
    )

    save_trajectory_driver_figure(
        class_name=class_name,
        class_summary_2025=trajectory_summary_2025,
        output_path=output_path,
    )

    if output_path.exists():
        trajectory_figure_paths.append(
            output_path
        )

        print(
            f"Saved: {output_path}"
        )


# =============================================================================
# 11. REAL-KADUWELA-BOUNDARY SPATIAL SHAP MAPS
# =============================================================================

section(
    "11. SAVE REAL-KADUWELA-BOUNDARY SPATIAL SHAP MAPS"
)

grid = gpd.read_file(
    GRID_WGS84_PATH
)

if GRID_COL not in grid.columns:
    raise RuntimeError(
        "WGS84 grid geometry missing grid_id"
    )

grid[
    GRID_COL
] = grid[
    GRID_COL
].astype(
    str
)

if len(
    grid
) != EXPECTED_GRIDS:
    raise RuntimeError(
        f"Geometry rows={len(grid)}, expected={EXPECTED_GRIDS}"
    )

grid = grid.to_crs(
    "EPSG:4326"
)

boundary, boundary_source, boundary_overlap = choose_real_boundary(
    grid
)

map_gdf = grid[
    [
        GRID_COL,
        "geometry",
    ]
].merge(
    local_summary,
    on=GRID_COL,
    how="left",
    validate="one_to_one",
)

if map_gdf[
    "centered_predicted_spatial_anomaly_c"
].isna().any():
    raise RuntimeError(
        "2025 SHAP summary failed to align with all geometries"
    )

map_gdf = gpd.GeoDataFrame(
    map_gdf,
    geometry="geometry",
    crs=grid.crs,
)

top_spatial_features = (
    global_importance.head(
        SPATIAL_TOP_N
    )[
        "feature"
    ]
    .tolist()
)

local_shap_feature_df = pd.DataFrame(
    local_centered_shap,
    columns=features,
)

local_shap_feature_df.insert(
    0,
    GRID_COL,
    local_data[
        GRID_COL
    ]
    .astype(
        str
    )
    .tolist(),
)

local_shap_feature_df = local_shap_feature_df.set_index(
    GRID_COL
)

for feature in top_spatial_features:
    map_gdf[
        f"shap__{feature}"
    ] = map_gdf[
        GRID_COL
    ].map(
        local_shap_feature_df[
            feature
        ]
    )

map_gdf.to_file(
    MAP_GEOJSON_PATH,
    driver="GeoJSON",
)

plot_spatial_continuous(
    gdf=map_gdf,
    column="centered_predicted_spatial_anomaly_c",
    boundary=boundary,
    title=(
        "Kaduwela 2025 Production-Model Spatial LST Anomaly\n"
        "Model explanation layer — not a new held-out test"
    ),
    output_path=ANOMALY_MAP_PATH,
    legend_label="Centered predicted spatial anomaly (°C)",
)

spatial_map_paths = []

for rank, feature in enumerate(
    top_spatial_features,
    start=1,
):
    output_path = (
        FIGURES_DIR
        / (
            f"24_2025_shap_map_rank{rank}_"
            f"{safe_name(feature)}.png"
        )
    )

    plot_spatial_continuous(
        gdf=map_gdf,
        column=f"shap__{feature}",
        boundary=boundary,
        title=(
            f"Kaduwela 2025 SHAP Contribution — {feature}\n"
            "Contribution to centered spatial LST anomaly"
        ),
        output_path=output_path,
        legend_label="Centered SHAP contribution (°C)",
    )

    spatial_map_paths.append(
        output_path
    )

print(
    f"Saved: {MAP_GEOJSON_PATH}"
)

print(
    f"Saved: {ANOMALY_MAP_PATH}"
)

for path in spatial_map_paths:
    print(
        f"Saved: {path}"
    )


# =============================================================================
# 12. METHODOLOGY MANIFEST
# =============================================================================

section(
    "12. SAVE STEP-24 METHODOLOGY"
)

methodology = {
    "step": 24,
    "version": "production_centered_shap_plus_trajectory_v2",
    "analysis": "SHAP explainability for final production spatial-anomaly model",
    "explained_model": str(
        MODEL_PATH
    ),
    "model_type": model.__class__.__name__,
    "model_training_years": EXPECTED_YEARS,
    "model_target": "within-year spatial LST anomaly",
    "feature_count": len(
        features
    ),
    "shap_method": "TreeSHAP",
    "raw_shap_definition": (
        "Feature contributions to raw LightGBM spatial-anomaly output"
    ),
    "centered_shap_definition": (
        "Within-year raw SHAP contribution minus the same feature's "
        "mean SHAP contribution across all 2,194 grids in that year"
    ),
    "centered_shap_identity": (
        "sum(centered feature SHAP values) = centered model-predicted "
        "spatial anomaly"
    ),
    "global_importance_definition": (
        "Mean absolute centered SHAP contribution across all 17,552 "
        "2018-2025 production-training rows"
    ),
    "local_explanation_year": LOCAL_YEAR,
    "local_top_k": LOCAL_TOP_K,
    "trajectory_source": str(
        TRAJECTORY_PATH
    ),
    "trajectory_version": "23B_relative_sen_v2",
    "trajectory_conditioned_shap": {
        "all_years_summary": str(
            TRAJECTORY_ALL_YEARS_PATH
        ),
        "2025_summary": str(
            TRAJECTORY_2025_PATH
        ),
        "top_heating_associations": str(
            TRAJECTORY_HEATING_PATH
        ),
        "top_cooling_associations": str(
            TRAJECTORY_COOLING_PATH
        ),
        "interpretation": (
            "Model-based SHAP patterns associated with trajectory classes; "
            "not causal explanations of trajectory formation"
        ),
    },
    "spatial_map_top_feature_count": SPATIAL_TOP_N,
    "map_boundary_source": str(
        boundary_source
    ),
    "map_boundary_grid_overlap_fraction": boundary_overlap,
    "map_crs": "EPSG:4326",
    "future_predictions_explained": False,
    "future_shap_deferred": (
        "2026-2030 future SHAP is deferred until future predictors, "
        "annual thermal-level forecasts, and future LST predictions exist"
    ),
    "interpretation": (
        "Positive centered SHAP pushes a grid hotter relative to "
        "Kaduwela's annual thermal level; negative centered SHAP pushes "
        "it cooler relative to that level"
    ),
    "causal_warning": (
        "SHAP explains the fitted model. It does not prove causal effects."
    ),
    "evaluation_warning": (
        "2018-2025 rows are used for production-model explanation only. "
        "Step 24 does not create a new held-out accuracy result."
    ),
}

METHOD_PATH.write_text(
    json.dumps(
        methodology,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    json.dumps(
        methodology,
        indent=2,
    )
)


# =============================================================================
# 13. STRICT AUDIT
# =============================================================================

section(
    "13. STRICT STEP-24 AUDIT"
)

local_max_reconstruction_error = float(
    local_summary[
        "local_reconstruction_error_c"
    ]
    .abs()
    .max()
)

local_grid_count_ok = bool(
    len(
        local_summary
    )
    == EXPECTED_GRIDS
    and local_summary[
        GRID_COL
    ].nunique()
    == EXPECTED_GRIDS
)

top10_per_grid_ok = bool(
    (
        local_top10.groupby(
            GRID_COL
        ).size()
        == LOCAL_TOP_K
    ).all()
)

trajectory_class_grid_counts = (
    local_summary[
        "trajectory_class"
    ]
    .value_counts()
    .reindex(
        TRAJECTORY_CLASSES,
        fill_value=0,
    )
)

expected_trajectory_counts = (
    trajectory[
        "trajectory_class"
    ]
    .value_counts()
    .reindex(
        TRAJECTORY_CLASSES,
        fill_value=0,
    )
)

trajectory_counts_match = bool(
    trajectory_class_grid_counts.equals(
        expected_trajectory_counts
    )
)

all_trajectory_figures_exist = bool(
    len(
        trajectory_figure_paths
    )
    == len(
        TRAJECTORY_CLASSES
    )
    and all(
        path.exists()
        for path
        in trajectory_figure_paths
    )
)

spatial_maps_exist = bool(
    ANOMALY_MAP_PATH.exists()
    and all(
        path.exists()
        for path
        in spatial_map_paths
    )
)

audit_rows = [
    {
        "check": "step23b_audit",
        "status": "PASS",
        "details": "Step-23B v2 audit verified fully PASS.",
    },
    {
        "check": "production_model_hash",
        "status": "PASS",
        "details": EXPECTED_HASHES[
            "22b_production_lightgbm_anomaly_trained_2018_2025.joblib"
        ],
    },
    {
        "check": "trajectory_input_hash",
        "status": "PASS",
        "details": EXPECTED_HASHES[
            "23b_grid_heat_trajectory_classes.csv"
        ],
    },
    {
        "check": "production_rows",
        "status": (
            "PASS"
            if len(
                data
            )
            == EXPECTED_PRODUCTION_ROWS
            else "FAIL"
        ),
        "details": (
            f"actual={len(data)}, expected={EXPECTED_PRODUCTION_ROWS}"
        ),
    },
    {
        "check": "production_years",
        "status": (
            "PASS"
            if years
            == EXPECTED_YEARS
            else "FAIL"
        ),
        "details": str(
            years
        ),
    },
    {
        "check": "unique_grids",
        "status": (
            "PASS"
            if data[
                GRID_COL
            ].nunique()
            == EXPECTED_GRIDS
            else "FAIL"
        ),
        "details": (
            f"actual={data[GRID_COL].nunique()}, expected={EXPECTED_GRIDS}"
        ),
    },
    {
        "check": "frozen_feature_count",
        "status": (
            "PASS"
            if len(
                features
            )
            == EXPECTED_FEATURES
            else "FAIL"
        ),
        "details": (
            f"actual={len(features)}, expected={EXPECTED_FEATURES}"
        ),
    },
    {
        "check": "raw_shap_reconstruction",
        "status": (
            "PASS"
            if max_raw_reconstruction_error
            <= 1e-6
            else "FAIL"
        ),
        "details": (
            f"max_error_c={max_raw_reconstruction_error:.12e}"
        ),
    },
    {
        "check": "centered_shap_reconstruction",
        "status": (
            "PASS"
            if max_centered_reconstruction_error
            <= 1e-6
            else "FAIL"
        ),
        "details": (
            f"max_error_c={max_centered_reconstruction_error:.12e}"
        ),
    },
    {
        "check": "2025_local_grid_count",
        "status": (
            "PASS"
            if local_grid_count_ok
            else "FAIL"
        ),
        "details": (
            f"rows={len(local_summary)}, unique_grids="
            f"{local_summary[GRID_COL].nunique()}"
        ),
    },
    {
        "check": "2025_top10_rows_per_grid",
        "status": (
            "PASS"
            if top10_per_grid_ok
            else "FAIL"
        ),
        "details": (
            f"expected_per_grid={LOCAL_TOP_K}"
        ),
    },
    {
        "check": "2025_local_reconstruction",
        "status": (
            "PASS"
            if local_max_reconstruction_error
            <= 1e-6
            else "FAIL"
        ),
        "details": (
            f"max_error_c={local_max_reconstruction_error:.12e}"
        ),
    },
    {
        "check": "trajectory_class_counts_preserved",
        "status": (
            "PASS"
            if trajectory_counts_match
            else "FAIL"
        ),
        "details": (
            "2025 SHAP grid counts match Step-23B v2 class counts."
        ),
    },
    {
        "check": "trajectory_shap_aggregation_reconstruction",
        "status": (
            "PASS"
            if max(
                trajectory_all_mean_error,
                trajectory_all_mean_abs_error,
                trajectory_2025_mean_error,
                trajectory_2025_mean_abs_error,
            )
            <= 1e-10
            else "FAIL"
        ),
        "details": (
            "max_error="
            f"{max(trajectory_all_mean_error, trajectory_all_mean_abs_error, trajectory_2025_mean_error, trajectory_2025_mean_abs_error):.12e}"
        ),
    },
    {
        "check": "trajectory_conditioned_shap_saved",
        "status": (
            "PASS"
            if (
                TRAJECTORY_ALL_YEARS_PATH.exists()
                and TRAJECTORY_2025_PATH.exists()
                and TRAJECTORY_HEATING_PATH.exists()
                and TRAJECTORY_COOLING_PATH.exists()
            )
            else "FAIL"
        ),
        "details": (
            "All-year, 2025, heating and cooling trajectory SHAP summaries saved."
        ),
    },
    {
        "check": "trajectory_figures_saved",
        "status": (
            "PASS"
            if all_trajectory_figures_exist
            else "FAIL"
        ),
        "details": (
            f"saved={len(trajectory_figure_paths)}, expected={len(TRAJECTORY_CLASSES)}"
        ),
    },
    {
        "check": "trajectory_shap_not_causal_claim",
        "status": "PASS",
        "details": (
            "Trajectory SHAP is explicitly documented as model association, not causation."
        ),
    },
    {
        "check": "real_boundary_loaded",
        "status": "PASS",
        "details": (
            f"{boundary_source}; overlap={boundary_overlap:.4f}"
        ),
    },
    {
        "check": "map_geometry_alignment",
        "status": (
            "PASS"
            if (
                len(
                    map_gdf
                )
                == EXPECTED_GRIDS
                and map_gdf[
                    GRID_COL
                ].nunique()
                == EXPECTED_GRIDS
            )
            else "FAIL"
        ),
        "details": (
            f"map_rows={len(map_gdf)}, expected={EXPECTED_GRIDS}"
        ),
    },
    {
        "check": "report_quality_spatial_pngs_saved",
        "status": (
            "PASS"
            if spatial_maps_exist
            else "FAIL"
        ),
        "details": (
            f"anomaly map + {len(spatial_map_paths)} top-feature maps"
        ),
    },
    {
        "check": "future_predictions_or_future_shap_created",
        "status": "PASS",
        "details": (
            "No. Future SHAP remains deferred until forecasting stages."
        ),
    },
    {
        "check": "new_test_score_created",
        "status": "PASS",
        "details": (
            "No. Step 24 is explainability only."
        ),
    },
]

audit_df = pd.DataFrame(
    audit_rows
)

audit_df.to_csv(
    AUDIT_PATH,
    index=False,
)

print(
    audit_df.to_string(
        index=False
    )
)

if (
    audit_df[
        "status"
    ]
    != "PASS"
).any():
    raise RuntimeError(
        "One or more Step-24 audit checks failed."
    )


# =============================================================================
# 14. REPORT
# =============================================================================

section(
    "14. SAVE STEP-24 REPORT"
)

top_10_text = (
    global_importance.head(
        10
    )
    .to_string(
        index=False
    )
)

trajectory_count_text = (
    local_summary[
        "trajectory_class"
    ]
    .value_counts()
    .reindex(
        TRAJECTORY_CLASSES
    )
    .rename_axis(
        "trajectory_class"
    )
    .reset_index(
        name="grid_count"
    )
    .to_string(
        index=False
    )
)

report_lines = [
    "STEP 24 - SHAP EXPLAINABILITY",
    "=" * 116,
    "",
    "MODEL SCOPE",
    "-" * 116,
    "Production model: LightGBM spatial-anomaly model trained on 2018-2025.",
    "Target: within-year spatial LST anomaly.",
    "SHAP does not explain the annual thermal-level component in this step.",
    "Future 2026-2030 SHAP is deferred until future forecasts exist.",
    "",
    "CENTERED SHAP",
    "-" * 116,
    (
        "Year-centered SHAP contributions were calculated so their row sum "
        "reconstructs the frozen centered spatial-anomaly prediction."
    ),
    (
        "Positive contribution = pushes the grid hotter relative to "
        "Kaduwela's annual thermal level."
    ),
    (
        "Negative contribution = pushes the grid cooler relative to "
        "Kaduwela's annual thermal level."
    ),
    "",
    "TOP 10 GLOBAL FEATURES",
    "-" * 116,
    top_10_text,
    "",
    "TRAJECTORY CLASS COUNTS JOINED TO 2025 SHAP",
    "-" * 116,
    trajectory_count_text,
    "",
    "TRAJECTORY-CONDITIONED SHAP",
    "-" * 116,
    (
        "SHAP patterns are summarized separately for Persistent Hot, "
        "Intensifying, Emerging, Diminishing and Persistent Cool grids."
    ),
    (
        "These are model-based associations with each trajectory class. "
        "They are not causal explanations of how the trajectory formed."
    ),
    "",
    "SHAP RECONSTRUCTION AUDIT",
    "-" * 116,
    year_reconstruction.to_string(
        index=False
    ),
    "",
    "REAL KADUWELA BOUNDARY",
    "-" * 116,
    f"Boundary source: {boundary_source}",
    f"Grid overlap fraction: {boundary_overlap:.4f}",
    "",
    "IMPORTANT",
    "-" * 116,
    (
        "Step 24 explains the final production model. "
        "It does not create a new held-out accuracy result."
    ),
]

REPORT_PATH.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)

print(
    f"Saved: {REPORT_PATH}"
)

print(
    f"Saved: {AUDIT_PATH}"
)


# =============================================================================
# 15. REPRODUCIBILITY HASHES
# =============================================================================

section(
    "15. REPRODUCIBILITY HASHES"
)

hash_files = [
    train_path,
    validation_path,
    test_path,
    FEATURE_LIST_PATH,
    MODEL_PATH,
    TRAJECTORY_PATH,
    TRAJECTORY_AUDIT_PATH,
    GRID_WGS84_PATH,
    GLOBAL_IMPORTANCE_PATH,
    RAW_SHAP_ALL_PATH,
    CENTERED_SHAP_ALL_PATH,
    YEAR_RECON_PATH,
    LOCAL_TOP10_PATH,
    LOCAL_SUMMARY_PATH,
    LOCAL_WITH_TRAJECTORY_PATH,
    TRAJECTORY_ALL_YEARS_PATH,
    TRAJECTORY_2025_PATH,
    TRAJECTORY_HEATING_PATH,
    TRAJECTORY_COOLING_PATH,
    GLOBAL_BAR_PATH,
    GLOBAL_BEESWARM_PATH,
    MAP_GEOJSON_PATH,
    ANOMALY_MAP_PATH,
    *trajectory_figure_paths,
    *spatial_map_paths,
    METHOD_PATH,
    AUDIT_PATH,
    REPORT_PATH,
]

hash_rows = []

for path in hash_files:
    digest = sha256_file(
        path
    )

    hash_rows.append(
        {
            "file": str(
                path
            ),
            "sha256": digest,
        }
    )

    print(
        f"{path.name}: {digest}"
    )

pd.DataFrame(
    hash_rows
).to_csv(
    HASH_PATH,
    index=False,
)

print(
    f"Saved: {HASH_PATH}"
)


# =============================================================================
# FINAL
# =============================================================================

section(
    "STEP 24 COMPLETED SUCCESSFULLY"
)

print(
    "SHAP explainability completed for the final production LightGBM spatial-anomaly model.\n"
    "All 2018-2025 production rows were explained.\n"
    "Year-centered SHAP reconstructs the centered spatial-anomaly prediction.\n"
    f"2025 local explanations were saved for all {EXPECTED_GRIDS:,} grids.\n"
    "Global SHAP importance and beeswarm figures were saved.\n"
    "Trajectory-conditioned SHAP summaries were saved for all five trajectory classes.\n"
    "One trajectory SHAP figure was saved for each trajectory class.\n"
    "Report-quality spatial SHAP maps use the real Kaduwela boundary.\n"
    "No future prediction or future SHAP was created.\n"
    "No new test score was created.\n"
    "\nNEXT: review Step-24 outputs before starting Step 25 forecasting baseline."
)
