from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[4]

TREE_DIR = (
    ROOT
    / "data"
    / "model_ready"
    / "tree"
)

TRAIN_PATH = (
    TREE_DIR
    / "tree_train_2018_2023_ready.csv"
)

VALIDATION_2024_PATH = (
    TREE_DIR
    / "tree_validation_2024_ready.csv"
)

TEST_2025_PATH = (
    TREE_DIR
    / "tree_test_2025_LOCKED_ready.csv"
)

FEATURE_LIST_PATH = (
    ROOT
    / "outputs"
    / "reports"
    / "21_final_feature_list.csv"
)

MODEL_TEMPLATE_PATH = (
    ROOT
    / "models"
    / "final"
    / "21_final_lightgbm_anomaly_trained_2018_2024.joblib"
)

VALIDATION_DIR = (
    ROOT
    / "experiments"
    / "validation"
    / "scenario_validation"
)

OUTPUT_DIR = VALIDATION_DIR / "outputs"
FIGURE_DIR = VALIDATION_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

GRID = "grid_id"
AREA = "area_name"
YEAR = "year"
TARGET = "LST_C"

TARGET_YEARS = [2024, 2025]

EXPECTED_GRIDS = 2194
EXPECTED_FEATURES = 93

EPS = 1e-12


# ============================================================
# HELPERS
# ============================================================

def require_file(path, label):

    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def detect_feature_column(feature_df):

    for col in [
        "feature",
        "feature_name",
        "predictor",
    ]:

        if col in feature_df.columns:

            values = (
                feature_df[col]
                .dropna()
                .astype(str)
                .str.strip()
                .tolist()
            )

            if len(values) == EXPECTED_FEATURES:
                return col

    for col in feature_df.columns:

        values = (
            feature_df[col]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )

        if (
            len(values) == EXPECTED_FEATURES
            and len(set(values))
            == EXPECTED_FEATURES
        ):
            return col

    raise RuntimeError(
        "Could not identify frozen 93-feature column."
    )


def safe_corr(a, b, method="pearson"):

    a = pd.Series(
        np.asarray(a, dtype=float)
    )

    b = pd.Series(
        np.asarray(b, dtype=float)
    )

    if (
        a.nunique() < 2
        or b.nunique() < 2
    ):
        return np.nan

    return float(
        a.corr(
            b,
            method=method,
        )
    )


def sign_agreement(a, b, mask=None):

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    valid = (
        np.isfinite(a)
        & np.isfinite(b)
        & (np.abs(a) > EPS)
        & (np.abs(b) > EPS)
    )

    if mask is not None:
        valid &= np.asarray(
            mask,
            dtype=bool,
        )

    if valid.sum() == 0:
        return np.nan, 0

    agreement = (
        np.sign(a[valid])
        == np.sign(b[valid])
    )

    return (
        float(
            agreement.mean()
            * 100.0
        ),
        int(valid.sum()),
    )


def regression_summary(
    observed,
    predicted,
):

    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    valid = (
        np.isfinite(observed)
        & np.isfinite(predicted)
    )

    observed = observed[valid]
    predicted = predicted[valid]

    mae = mean_absolute_error(
        observed,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            observed,
            predicted,
        )
    )

    r2 = r2_score(
        observed,
        predicted,
    )

    pearson = safe_corr(
        observed,
        predicted,
        method="pearson",
    )

    spearman = safe_corr(
        observed,
        predicted,
        method="spearman",
    )

    sign_pct, sign_n = (
        sign_agreement(
            observed,
            predicted,
        )
    )

    return {
        "n": len(observed),
        "mae_c": float(mae),
        "rmse_c": float(rmse),
        "r2": float(r2),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "sign_agreement_pct": sign_pct,
        "sign_comparison_n": sign_n,
    }


# ============================================================
# COUNTERFACTUAL FEATURE REBUILDERS
# ============================================================

def counterfactual_green(
    X_actual,
    previous_green,
):

    X = X_actual.copy()

    previous_green = np.asarray(
        previous_green,
        dtype=float,
    )

    baseline_green = (
        X_actual["green_mask"]
        .to_numpy(dtype=float)
    )

    X["green_mask"] = previous_green

    if "green_mask_delta1" in X.columns:

        if "green_mask_lag1" not in X.columns:
            raise RuntimeError(
                "green_mask_delta1 exists but "
                "green_mask_lag1 is missing."
            )

        X["green_mask_delta1"] = (
            previous_green
            - X[
                "green_mask_lag1"
            ].to_numpy(dtype=float)
        )

    vegetation_deficit = (
        1.0 - previous_green
    )

    if "vegetation_deficit" in X.columns:
        X[
            "vegetation_deficit"
        ] = vegetation_deficit

    if (
        "ndbi_x_vegetation_deficit"
        in X.columns
    ):
        X[
            "ndbi_x_vegetation_deficit"
        ] = (
            X["NDBI"]
            .to_numpy(dtype=float)
            * vegetation_deficit
        )

    if "ndvi_x_green_mask" in X.columns:
        X[
            "ndvi_x_green_mask"
        ] = (
            X["NDVI"]
            .to_numpy(dtype=float)
            * previous_green
        )

    if (
        "green_mask_nbr_contrast"
        in X.columns
    ):

        if "green_mask_nbr_mean" in X.columns:

            nbr_mean = (
                X[
                    "green_mask_nbr_mean"
                ]
                .to_numpy(dtype=float)
            )

        else:

            nbr_mean = (
                baseline_green
                - X_actual[
                    "green_mask_nbr_contrast"
                ].to_numpy(dtype=float)
            )

        X[
            "green_mask_nbr_contrast"
        ] = (
            previous_green
            - nbr_mean
        )

    return X


def counterfactual_ndbi(
    X_actual,
    previous_ndbi,
):

    X = X_actual.copy()

    previous_ndbi = np.asarray(
        previous_ndbi,
        dtype=float,
    )

    baseline_ndbi = (
        X_actual["NDBI"]
        .to_numpy(dtype=float)
    )

    X["NDBI"] = previous_ndbi

    if "NDBI_delta1" in X.columns:

        if "NDBI_lag1" not in X.columns:
            raise RuntimeError(
                "NDBI_delta1 exists but "
                "NDBI_lag1 is missing."
            )

        X["NDBI_delta1"] = (
            previous_ndbi
            - X[
                "NDBI_lag1"
            ].to_numpy(dtype=float)
        )

    if "ndbi_minus_ndvi" in X.columns:

        X[
            "ndbi_minus_ndvi"
        ] = (
            previous_ndbi
            - X["NDVI"]
            .to_numpy(dtype=float)
        )

    if (
        "ndbi_x_vegetation_deficit"
        in X.columns
    ):

        X[
            "ndbi_x_vegetation_deficit"
        ] = (
            previous_ndbi
            * (
                1.0
                - X["green_mask"]
                .to_numpy(dtype=float)
            )
        )

    if (
        "ndbi_x_building_mask"
        in X.columns
    ):

        X[
            "ndbi_x_building_mask"
        ] = (
            previous_ndbi
            * X["building_mask"]
            .to_numpy(dtype=float)
        )

    if "NDBI_nbr_contrast" in X.columns:

        if "NDBI_nbr_mean" in X.columns:

            nbr_mean = (
                X["NDBI_nbr_mean"]
                .to_numpy(dtype=float)
            )

        else:

            nbr_mean = (
                baseline_ndbi
                - X_actual[
                    "NDBI_nbr_contrast"
                ].to_numpy(dtype=float)
            )

        X["NDBI_nbr_contrast"] = (
            previous_ndbi
            - nbr_mean
        )

    return X


def counterfactual_ndwi(
    X_actual,
    previous_ndwi,
):

    X = X_actual.copy()

    previous_ndwi = np.asarray(
        previous_ndwi,
        dtype=float,
    )

    baseline_ndwi = (
        X_actual["NDWI"]
        .to_numpy(dtype=float)
    )

    X["NDWI"] = previous_ndwi

    if "NDWI_delta1" in X.columns:

        if "NDWI_lag1" not in X.columns:
            raise RuntimeError(
                "NDWI_delta1 exists but "
                "NDWI_lag1 is missing."
            )

        X["NDWI_delta1"] = (
            previous_ndwi
            - X[
                "NDWI_lag1"
            ].to_numpy(dtype=float)
        )

    if "NDWI_nbr_contrast" in X.columns:

        if "NDWI_nbr_mean" in X.columns:

            nbr_mean = (
                X[
                    "NDWI_nbr_mean"
                ]
                .to_numpy(dtype=float)
            )

        else:

            nbr_mean = (
                baseline_ndwi
                - X_actual[
                    "NDWI_nbr_contrast"
                ].to_numpy(dtype=float)
            )

        X["NDWI_nbr_contrast"] = (
            previous_ndwi
            - nbr_mean
        )

    return X


# ============================================================
# LOAD INPUTS
# ============================================================

print("\n============================================================")
print("SCENARIO VALIDATION 05")
print("HISTORICAL PSEUDO-INTERVENTION BACKTEST")
print("============================================================\n")

for path, label in [
    (
        TRAIN_PATH,
        "2018-2023 model-ready training data",
    ),
    (
        VALIDATION_2024_PATH,
        "2024 reserved validation data",
    ),
    (
        TEST_2025_PATH,
        "2025 lockbox data",
    ),
    (
        FEATURE_LIST_PATH,
        "Step-21 frozen feature list",
    ),
    (
        MODEL_TEMPLATE_PATH,
        "Step-21 frozen LightGBM model",
    ),
]:

    require_file(
        path,
        label,
    )


train_df = pd.read_csv(
    TRAIN_PATH,
    dtype={GRID: str},
    low_memory=False,
)

validation_2024 = pd.read_csv(
    VALIDATION_2024_PATH,
    dtype={GRID: str},
    low_memory=False,
)

test_2025 = pd.read_csv(
    TEST_2025_PATH,
    dtype={GRID: str},
    low_memory=False,
)

all_df = pd.concat(
    [
        train_df,
        validation_2024,
        test_2025,
    ],
    ignore_index=True,
)

all_df[GRID] = (
    all_df[GRID]
    .astype(str)
    .str.strip()
)

all_df[YEAR] = pd.to_numeric(
    all_df[YEAR],
    errors="raise",
).astype(int)


# ============================================================
# FROZEN FEATURES / MODEL CONFIG
# ============================================================

feature_df = pd.read_csv(
    FEATURE_LIST_PATH
)

feature_col = detect_feature_column(
    feature_df
)

ordered_features = (
    feature_df[feature_col]
    .dropna()
    .astype(str)
    .str.strip()
    .tolist()
)

if (
    len(ordered_features)
    != EXPECTED_FEATURES
):

    raise RuntimeError(
        f"Expected {EXPECTED_FEATURES} "
        f"frozen features, found "
        f"{len(ordered_features)}."
    )


missing_features = [
    feature
    for feature in ordered_features
    if feature not in all_df.columns
]

if missing_features:

    raise RuntimeError(
        "Historical model-ready data is "
        f"missing features: {missing_features}"
    )


template_model = joblib.load(
    MODEL_TEMPLATE_PATH
)

print(
    f"Frozen predictors : "
    f"{len(ordered_features)}"
)

print(
    f"Historical rows   : "
    f"{len(all_df):,}"
)

print(
    f"Years available   : "
    f"{sorted(all_df[YEAR].unique())}"
)


# ============================================================
# BACKTEST
# ============================================================

total_change_metric_rows = []
scenario_metric_rows = []
detail_rows = []


scenario_definitions = {
    "GREEN": {
        "treatment": "green_mask",
        "builder": counterfactual_green,
    },
    "BUILT_NDBI": {
        "treatment": "NDBI",
        "builder": counterfactual_ndbi,
    },
    "BLUE_NDWI": {
        "treatment": "NDWI",
        "builder": counterfactual_ndwi,
    },
}


for target_year in TARGET_YEARS:

    previous_year = (
        target_year - 1
    )

    print(
        "\n============================================================"
    )
    print(
        f"BACKTEST TARGET YEAR {target_year}"
    )
    print(
        "============================================================"
    )

    training = (
        all_df[
            all_df[YEAR] < target_year
        ]
        .copy()
    )

    target_df = (
        all_df[
            all_df[YEAR] == target_year
        ]
        .copy()
        .sort_values(GRID)
        .reset_index(drop=True)
    )

    previous_df = (
        all_df[
            all_df[YEAR] == previous_year
        ]
        .copy()
        .sort_values(GRID)
        .reset_index(drop=True)
    )

    if len(target_df) != EXPECTED_GRIDS:
        raise RuntimeError(
            f"{target_year}: expected "
            f"{EXPECTED_GRIDS} target grids, "
            f"found {len(target_df)}."
        )

    if len(previous_df) != EXPECTED_GRIDS:
        raise RuntimeError(
            f"{previous_year}: expected "
            f"{EXPECTED_GRIDS} previous grids."
        )

    if not target_df[
        GRID
    ].equals(
        previous_df[GRID]
    ):

        raise RuntimeError(
            f"{target_year}: grid alignment "
            "failed between target and previous year."
        )


    # --------------------------------------------------------
    # Build anomaly target using TRAINING YEARS ONLY
    # --------------------------------------------------------

    training_year_means = (
        training.groupby(
            YEAR
        )[TARGET]
        .transform("mean")
    )

    y_train_anomaly = (
        training[TARGET]
        .to_numpy(dtype=float)
        - training_year_means
        .to_numpy(dtype=float)
    )


    X_train = (
        training[
            ordered_features
        ]
        .apply(
            pd.to_numeric,
            errors="raise",
        )
    )

    X_target = (
        target_df[
            ordered_features
        ]
        .apply(
            pd.to_numeric,
            errors="raise",
        )
        .copy()
    )

    X_previous = (
        previous_df[
            ordered_features
        ]
        .apply(
            pd.to_numeric,
            errors="raise",
        )
        .copy()
    )


    if not np.isfinite(
        X_train.to_numpy(dtype=float)
    ).all():

        raise RuntimeError(
            f"{target_year}: non-finite "
            "training predictors."
        )

    if not np.isfinite(
        X_target.to_numpy(dtype=float)
    ).all():

        raise RuntimeError(
            f"{target_year}: non-finite "
            "target predictors."
        )


    # --------------------------------------------------------
    # Fit frozen LightGBM configuration using only past years
    # --------------------------------------------------------

    model = clone(
        template_model
    )

    model.fit(
        X_train,
        y_train_anomaly,
    )


    # --------------------------------------------------------
    # Actual target-state and previous-state predictions
    # --------------------------------------------------------

    pred_target_raw = np.asarray(
        model.predict(
            X_target
        ),
        dtype=float,
    )

    pred_previous_raw = np.asarray(
        model.predict(
            X_previous
        ),
        dtype=float,
    )


    # Center each year's model output because final model
    # represents within-year spatial anomaly.
    pred_target_centered = (
        pred_target_raw
        - pred_target_raw.mean()
    )

    pred_previous_centered = (
        pred_previous_raw
        - pred_previous_raw.mean()
    )


    # --------------------------------------------------------
    # Observed spatial anomalies
    # --------------------------------------------------------

    actual_target_lst = (
        target_df[TARGET]
        .to_numpy(dtype=float)
    )

    actual_previous_lst = (
        previous_df[TARGET]
        .to_numpy(dtype=float)
    )

    observed_target_anomaly = (
        actual_target_lst
        - actual_target_lst.mean()
    )

    observed_previous_anomaly = (
        actual_previous_lst
        - actual_previous_lst.mean()
    )

    observed_anomaly_change = (
        observed_target_anomaly
        - observed_previous_anomaly
    )


    # --------------------------------------------------------
    # Full spatial-change backtest
    # --------------------------------------------------------

    predicted_total_anomaly_change = (
        pred_target_centered
        - pred_previous_centered
    )

    total_metrics = regression_summary(
        observed_anomaly_change,
        predicted_total_anomaly_change,
    )

    total_change_metric_rows.append(
        {
            "target_year":
                target_year,
            "train_years":
                (
                    f"{int(training[YEAR].min())}"
                    f"-"
                    f"{int(training[YEAR].max())}"
                ),
            **total_metrics,
        }
    )

    print(
        "\nFull observed spatial-change backtest:"
    )

    print(
        f"  MAE       : "
        f"{total_metrics['mae_c']:.4f} °C"
    )

    print(
        f"  RMSE      : "
        f"{total_metrics['rmse_c']:.4f} °C"
    )

    print(
        f"  R²        : "
        f"{total_metrics['r2']:.4f}"
    )

    print(
        f"  Pearson r : "
        f"{total_metrics['pearson_r']:.4f}"
    )

    print(
        f"  Sign agree: "
        f"{total_metrics['sign_agreement_pct']:.2f}%"
    )


    # --------------------------------------------------------
    # Pseudo-intervention family checks
    # --------------------------------------------------------

    for family, definition in (
        scenario_definitions.items()
    ):

        treatment = (
            definition["treatment"]
        )

        builder = (
            definition["builder"]
        )

        current_value = (
            target_df[treatment]
            .to_numpy(dtype=float)
        )

        previous_value = (
            previous_df[treatment]
            .to_numpy(dtype=float)
        )

        actual_treatment_change = (
            current_value
            - previous_value
        )


        # Counterfactual:
        # target-year observed conditions are held fixed,
        # but the scenario treatment is reset to its
        # previous-year value.
        X_counterfactual = builder(
            X_target,
            previous_value,
        )

        if X_counterfactual.isna().any().any():

            raise RuntimeError(
                f"{target_year} {family}: "
                "counterfactual contains missing values."
            )

        if not np.isfinite(
            X_counterfactual.to_numpy(
                dtype=float
            )
        ).all():

            raise RuntimeError(
                f"{target_year} {family}: "
                "counterfactual contains "
                "non-finite values."
            )


        pred_counterfactual_raw = (
            np.asarray(
                model.predict(
                    X_counterfactual
                ),
                dtype=float,
            )
        )


        # Same scenario logic as Step 29:
        # local response = scenario raw anomaly
        # minus baseline raw anomaly.
        #
        # Here:
        # actual state = observed target-year treatment
        # counterfactual = previous-year treatment level.
        modeled_response = (
            pred_target_raw
            - pred_counterfactual_raw
        )


        abs_change = np.abs(
            actual_treatment_change
        )

        strong_threshold = float(
            np.quantile(
                abs_change,
                0.75,
            )
        )

        strong_mask = (
            abs_change
            >= strong_threshold
        )


        all_sign_pct, all_sign_n = (
            sign_agreement(
                observed_anomaly_change,
                modeled_response,
            )
        )

        strong_sign_pct, strong_sign_n = (
            sign_agreement(
                observed_anomaly_change,
                modeled_response,
                mask=strong_mask,
            )
        )


        scenario_metrics = {
            "target_year":
                target_year,
            "scenario_family":
                family,
            "treatment":
                treatment,
            "train_years":
                (
                    f"{int(training[YEAR].min())}"
                    f"-"
                    f"{int(training[YEAR].max())}"
                ),
            "rows":
                len(target_df),
            "mean_treatment_change":
                float(
                    np.mean(
                        actual_treatment_change
                    )
                ),
            "median_abs_treatment_change":
                float(
                    np.median(
                        abs_change
                    )
                ),
            "strong_change_q75_threshold":
                strong_threshold,
            "mean_modeled_response_c":
                float(
                    np.mean(
                        modeled_response
                    )
                ),
            "median_modeled_response_c":
                float(
                    np.median(
                        modeled_response
                    )
                ),
            "mae_vs_observed_anomaly_change_c":
                float(
                    mean_absolute_error(
                        observed_anomaly_change,
                        modeled_response,
                    )
                ),
            "rmse_vs_observed_anomaly_change_c":
                float(
                    np.sqrt(
                        mean_squared_error(
                            observed_anomaly_change,
                            modeled_response,
                        )
                    )
                ),
            "pearson_r":
                safe_corr(
                    observed_anomaly_change,
                    modeled_response,
                    method="pearson",
                ),
            "spearman_rho":
                safe_corr(
                    observed_anomaly_change,
                    modeled_response,
                    method="spearman",
                ),
            "sign_agreement_pct":
                all_sign_pct,
            "sign_comparison_n":
                all_sign_n,
            "strong_change_sign_agreement_pct":
                strong_sign_pct,
            "strong_change_sign_n":
                strong_sign_n,
        }

        scenario_metric_rows.append(
            scenario_metrics
        )


        print(
            f"\n{family} pseudo-intervention:"
        )

        print(
            f"  Mean modeled response : "
            f"{scenario_metrics['mean_modeled_response_c']:+.4f} °C"
        )

        print(
            f"  Pearson r            : "
            f"{scenario_metrics['pearson_r']:.4f}"
        )

        print(
            f"  Spearman rho         : "
            f"{scenario_metrics['spearman_rho']:.4f}"
        )

        print(
            f"  Sign agreement       : "
            f"{scenario_metrics['sign_agreement_pct']:.2f}%"
        )

        print(
            f"  Strong-change sign   : "
            f"{scenario_metrics['strong_change_sign_agreement_pct']:.2f}%"
        )


        # Detailed evidence rows
        for i in range(
            len(target_df)
        ):

            detail_rows.append(
                {
                    "grid_id":
                        target_df.loc[
                            i,
                            GRID,
                        ],
                    "area_name":
                        target_df.loc[
                            i,
                            AREA,
                        ],
                    "target_year":
                        target_year,
                    "previous_year":
                        previous_year,
                    "scenario_family":
                        family,
                    "treatment":
                        treatment,
                    "previous_treatment_value":
                        previous_value[i],
                    "actual_target_treatment_value":
                        current_value[i],
                    "actual_treatment_change":
                        actual_treatment_change[i],
                    "observed_previous_lst_c":
                        actual_previous_lst[i],
                    "observed_target_lst_c":
                        actual_target_lst[i],
                    "observed_raw_lst_change_c":
                        (
                            actual_target_lst[i]
                            - actual_previous_lst[i]
                        ),
                    "observed_spatial_anomaly_change_c":
                        observed_anomaly_change[i],
                    "modeled_pseudo_intervention_response_c":
                        modeled_response[i],
                    "strong_change_q75":
                        bool(
                            strong_mask[i]
                        ),
                }
            )


# ============================================================
# SAVE RESULTS
# ============================================================

total_metrics_df = pd.DataFrame(
    total_change_metric_rows
)

scenario_metrics_df = pd.DataFrame(
    scenario_metric_rows
)

detail_df = pd.DataFrame(
    detail_rows
)


total_metrics_path = (
    OUTPUT_DIR
    / "05_total_spatial_change_backtest_metrics.csv"
)

scenario_metrics_path = (
    OUTPUT_DIR
    / "05_pseudo_intervention_backtest_metrics.csv"
)

detail_path = (
    OUTPUT_DIR
    / "05_pseudo_intervention_grid_results.csv"
)


total_metrics_df.to_csv(
    total_metrics_path,
    index=False,
)

scenario_metrics_df.to_csv(
    scenario_metrics_path,
    index=False,
)

detail_df.to_csv(
    detail_path,
    index=False,
)


# ============================================================
# COMBINED 2024-2025 SUMMARY
# ============================================================

combined_rows = []

for family in scenario_definitions:

    sub = detail_df[
        detail_df[
            "scenario_family"
        ]
        == family
    ].copy()

    observed = (
        sub[
            "observed_spatial_anomaly_change_c"
        ]
        .to_numpy(dtype=float)
    )

    modeled = (
        sub[
            "modeled_pseudo_intervention_response_c"
        ]
        .to_numpy(dtype=float)
    )

    strong = (
        sub["strong_change_q75"]
        .astype(bool)
        .to_numpy()
    )

    sign_pct, sign_n = (
        sign_agreement(
            observed,
            modeled,
        )
    )

    strong_pct, strong_n = (
        sign_agreement(
            observed,
            modeled,
            mask=strong,
        )
    )

    combined_rows.append(
        {
            "scenario_family":
                family,
            "years":
                "2024-2025",
            "grid_year_rows":
                len(sub),
            "mae_vs_observed_spatial_change_c":
                float(
                    mean_absolute_error(
                        observed,
                        modeled,
                    )
                ),
            "rmse_vs_observed_spatial_change_c":
                float(
                    np.sqrt(
                        mean_squared_error(
                            observed,
                            modeled,
                        )
                    )
                ),
            "pearson_r":
                safe_corr(
                    observed,
                    modeled,
                    method="pearson",
                ),
            "spearman_rho":
                safe_corr(
                    observed,
                    modeled,
                    method="spearman",
                ),
            "sign_agreement_pct":
                sign_pct,
            "sign_n":
                sign_n,
            "strong_change_sign_agreement_pct":
                strong_pct,
            "strong_change_sign_n":
                strong_n,
        }
    )


combined_df = pd.DataFrame(
    combined_rows
)

combined_path = (
    OUTPUT_DIR
    / "05_pseudo_intervention_combined_summary.csv"
)

combined_df.to_csv(
    combined_path,
    index=False,
)


# ============================================================
# FIGURES
# ============================================================

for family in scenario_definitions:

    sub = detail_df[
        detail_df[
            "scenario_family"
        ]
        == family
    ]

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ax.scatter(
        sub[
            "observed_spatial_anomaly_change_c"
        ],
        sub[
            "modeled_pseudo_intervention_response_c"
        ],
        alpha=0.35,
        s=12,
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Observed Spatial-Anomaly Change (°C)"
    )

    ax.set_ylabel(
        "Modeled Pseudo-Intervention Response (°C)"
    )

    ax.set_title(
        f"{family} Historical Pseudo-Intervention Backtest"
    )

    fig.tight_layout()

    figure_path = (
        FIGURE_DIR
        / (
            "05_"
            + family.lower()
            + "_pseudo_intervention_backtest.png"
        )
    )

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print("\n============================================================")
print("TOTAL SPATIAL-CHANGE BACKTEST")
print("============================================================")

print(
    total_metrics_df.to_string(
        index=False
    )
)


print("\n============================================================")
print("PSEUDO-INTERVENTION BACKTEST BY YEAR")
print("============================================================")

print(
    scenario_metrics_df.to_string(
        index=False
    )
)


print("\n============================================================")
print("COMBINED 2024-2025 PSEUDO-INTERVENTION SUMMARY")
print("============================================================")

print(
    combined_df.to_string(
        index=False
    )
)


print("\nIMPORTANT INTERPRETATION:")
print(
    "This is a leakage-controlled historical pseudo-intervention "
    "backtest using reserved 2024 and 2025 observations."
)

print(
    "The modeled response isolates one treatment variable while "
    "other target-year predictors are held fixed."
)

print(
    "Observed year-to-year spatial heat change still contains "
    "changes from many environmental factors, so agreement is "
    "supporting consistency evidence, not causal proof."
)


print("\nSaved:")
print(total_metrics_path)
print(scenario_metrics_path)
print(detail_path)
print(combined_path)

print(
    "\nHISTORICAL PSEUDO-INTERVENTION "
    "BACKTEST COMPLETE."
)