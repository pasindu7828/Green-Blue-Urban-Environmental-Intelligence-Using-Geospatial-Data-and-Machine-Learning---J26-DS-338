from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import rasterio


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "part2_candidate_wetland"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "candidate_wetland"
    / "final"
)

PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "part2_candidate_wetland"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "part2_candidate_wetland"
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUTS
# ============================================================

FEATURE_STACK_INPUT = (
    PROCESSED_DIR
    / "feature_stack_10m.tif"
)

STUDY_MASK_INPUT = (
    PROCESSED_DIR
    / "study_mask_10m.tif"
)

FEATURE_METADATA_INPUT = (
    PROCESSED_DIR
    / "feature_stack_metadata.csv"
)

FINAL_MODEL_INPUT = (
    MODEL_DIR
    / "xgboost_final_model.joblib"
)


# ============================================================
# OUTPUTS
# ============================================================

PROBABILITY_OUTPUT = (
    PREDICTION_DIR
    / "12_candidate_wetland_probability_10m.tif"
)

MODEL_POSITIVE_OUTPUT = (
    PREDICTION_DIR
    / "12_model_positive_mask_10m.tif"
)

SUMMARY_OUTPUT = (
    REPORT_DIR
    / "12_candidate_probability_summary.csv"
)

AUDIT_OUTPUT = (
    REPORT_DIR
    / "12_candidate_probability_raster_audit.json"
)


# ============================================================
# SETTINGS
# ============================================================

SELECTED_MODEL = "XGB_03"

FROZEN_THRESHOLD = 0.50

PROBABILITY_NODATA = -9999.0

BINARY_NODATA = 255


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "PART 2 FULL-AREA PROBABILITY PREDICTION"
    )

    print(
        "========================================"
    )

    print(
        f"Frozen model: "
        f"{SELECTED_MODEL}"
    )

    print(
        f"Frozen threshold: "
        f"{FROZEN_THRESHOLD:.2f}"
    )

    # ========================================================
    # 1. LOAD FEATURE METADATA
    # ========================================================

    feature_metadata = pd.read_csv(
        FEATURE_METADATA_INPUT
    )

    feature_names = (
        feature_metadata[
            "feature"
        ]
        .tolist()
    )

    print(
        f"\nExpected feature count: "
        f"{len(feature_names)}"
    )

    # ========================================================
    # 2. LOAD FINAL MODEL
    # ========================================================

    model = joblib.load(
        FINAL_MODEL_INPUT
    )

    print(
        "Final model loaded."
    )

    # ========================================================
    # 3. OPEN RASTERS
    # ========================================================

    with (
        rasterio.open(
            FEATURE_STACK_INPUT
        ) as feature_src,

        rasterio.open(
            STUDY_MASK_INPUT
        ) as mask_src
    ):

        print(
            "\nFeature raster:"
        )

        print(
            f"CRS: {feature_src.crs}"
        )

        print(
            f"Size: "
            f"{feature_src.width} x "
            f"{feature_src.height}"
        )

        print(
            f"Bands: "
            f"{feature_src.count}"
        )

        print(
            f"Resolution: "
            f"{feature_src.res}"
        )

        if (
            feature_src.count
            != len(feature_names)
        ):

            raise RuntimeError(
                "Feature-stack band count does "
                "not match metadata."
            )

        if (
            feature_src.crs
            != mask_src.crs
            or
            feature_src.width
            != mask_src.width
            or
            feature_src.height
            != mask_src.height
            or
            feature_src.transform
            != mask_src.transform
        ):

            raise RuntimeError(
                "Feature stack and study mask "
                "are not spatially aligned."
            )

        # ----------------------------------------------------
        # OUTPUT PROFILE: PROBABILITY
        # ----------------------------------------------------

        probability_profile = (
            feature_src.profile.copy()
        )

        probability_profile.update(
            {
                "count": 1,
                "dtype": "float32",
                "nodata": PROBABILITY_NODATA,
                "compress": "deflate",
                "predictor": 3,
            }
        )

        # ----------------------------------------------------
        # OUTPUT PROFILE: BINARY MODEL-POSITIVE MASK
        # ----------------------------------------------------

        binary_profile = (
            feature_src.profile.copy()
        )

        binary_profile.update(
            {
                "count": 1,
                "dtype": "uint8",
                "nodata": BINARY_NODATA,
                "compress": "deflate",
            }
        )

        valid_pixel_count = 0

        predicted_positive_count = 0

        probability_values = []

        with (
            rasterio.open(
                PROBABILITY_OUTPUT,
                "w",
                **probability_profile,
            ) as probability_dst,

            rasterio.open(
                MODEL_POSITIVE_OUTPUT,
                "w",
                **binary_profile,
            ) as binary_dst
        ):

            # =================================================
            # 4. BLOCK-WISE PREDICTION
            # =================================================

            for _, window in (
                feature_src.block_windows(1)
            ):

                features = (
                    feature_src.read(
                        window=window
                    )
                    .astype(
                        np.float32
                    )
                )

                study_mask = (
                    mask_src.read(
                        1,
                        window=window,
                    )
                )

                rows = features.shape[1]

                cols = features.shape[2]

                flattened = (
                    features
                    .reshape(
                        feature_src.count,
                        -1,
                    )
                    .T
                )

                inside_study = (
                    study_mask
                    .reshape(-1)
                    > 0
                )

                finite_features = (
                    np.isfinite(
                        flattened
                    )
                    .all(
                        axis=1
                    )
                )

                if (
                    feature_src.nodata
                    is not None
                ):

                    not_nodata = (
                        flattened
                        != feature_src.nodata
                    ).all(
                        axis=1
                    )

                else:

                    not_nodata = (
                        np.ones(
                            flattened.shape[0],
                            dtype=bool,
                        )
                    )

                valid = (
                    inside_study
                    &
                    finite_features
                    &
                    not_nodata
                )

                probability_flat = (
                    np.full(
                        flattened.shape[0],
                        PROBABILITY_NODATA,
                        dtype=np.float32,
                    )
                )

                binary_flat = (
                    np.full(
                        flattened.shape[0],
                        BINARY_NODATA,
                        dtype=np.uint8,
                    )
                )

                if (
                    valid.any()
                ):

                    valid_features = (
                        pd.DataFrame(
                            flattened[
                                valid
                            ],
                            columns=feature_names,
                        )
                    )

                    valid_probabilities = (
                        model
                        .predict_proba(
                            valid_features
                        )[:, 1]
                        .astype(
                            np.float32
                        )
                    )

                    if (
                        np.isnan(
                            valid_probabilities
                        ).any()
                        or
                        np.isinf(
                            valid_probabilities
                        ).any()
                    ):

                        raise RuntimeError(
                            "Invalid model probabilities "
                            "were produced."
                        )

                    if (
                        (
                            valid_probabilities < 0
                        ).any()
                        or
                        (
                            valid_probabilities > 1
                        ).any()
                    ):

                        raise RuntimeError(
                            "Probability values outside "
                            "the range [0, 1]."
                        )

                    valid_binary = (
                        valid_probabilities
                        >= FROZEN_THRESHOLD
                    ).astype(
                        np.uint8
                    )

                    probability_flat[
                        valid
                    ] = (
                        valid_probabilities
                    )

                    binary_flat[
                        valid
                    ] = (
                        valid_binary
                    )

                    valid_pixel_count += (
                        len(
                            valid_probabilities
                        )
                    )

                    predicted_positive_count += int(
                        valid_binary.sum()
                    )

                    probability_values.append(
                        valid_probabilities
                    )

                probability_array = (
                    probability_flat
                    .reshape(
                        rows,
                        cols,
                    )
                )

                binary_array = (
                    binary_flat
                    .reshape(
                        rows,
                        cols,
                    )
                )

                probability_dst.write(
                    probability_array,
                    1,
                    window=window,
                )

                binary_dst.write(
                    binary_array,
                    1,
                    window=window,
                )

    # ========================================================
    # 5. SUMMARY STATISTICS
    # ========================================================

    if (
        len(
            probability_values
        )
        == 0
    ):

        raise RuntimeError(
            "No valid pixels were predicted."
        )

    all_probabilities = (
        np.concatenate(
            probability_values
        )
    )

    # 10 m x 10 m pixel = 100 m² = 0.01 ha.
    pixel_area_ha = 0.01

    thresholds = [
        0.10,
        0.25,
        0.50,
        0.70,
        0.80,
        0.90,
    ]

    summary_rows = []

    for threshold in thresholds:

        count = int(
            (
                all_probabilities
                >= threshold
            ).sum()
        )

        summary_rows.append(
            {
                "probability_threshold":
                    threshold,

                "pixel_count":
                    count,

                "area_ha":
                    count
                    *
                    pixel_area_ha,

                "percentage_of_valid_area":
                    (
                        100.0
                        *
                        count
                        /
                        valid_pixel_count
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    # ========================================================
    # 6. AUDIT
    # ========================================================

    audit = {
        "selected_model":
            SELECTED_MODEL,

        "frozen_threshold":
            FROZEN_THRESHOLD,

        "feature_count":
            len(
                feature_names
            ),

        "valid_prediction_pixels":
            int(
                valid_pixel_count
            ),

        "valid_prediction_area_km2":
            float(
                valid_pixel_count
                *
                100
                /
                1_000_000
            ),

        "model_positive_pixels_at_0_50":
            int(
                predicted_positive_count
            ),

        "model_positive_area_ha_at_0_50":
            float(
                predicted_positive_count
                *
                pixel_area_ha
            ),

        "probability_min":
            float(
                np.min(
                    all_probabilities
                )
            ),

        "probability_max":
            float(
                np.max(
                    all_probabilities
                )
            ),

        "probability_mean":
            float(
                np.mean(
                    all_probabilities
                )
            ),

        "probability_median":
            float(
                np.median(
                    all_probabilities
                )
            ),

        "important_note":
            (
                "The 0.50 model-positive mask is not yet "
                "the final candidate-wetland polygon layer. "
                "Known wetlands and likely open-water "
                "confusion have not yet been removed."
            ),
    }

    with open(
        AUDIT_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            audit,
            file,
            indent=4,
        )

    # ========================================================
    # TERMINAL REPORT
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "FULL-AREA PREDICTION SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        f"Valid predicted pixels: "
        f"{valid_pixel_count:,}"
    )

    print(
        f"Predicted valid area: "
        f"{valid_pixel_count * 100 / 1_000_000:.2f} km²"
    )

    print(
        f"\nProbability minimum: "
        f"{np.min(all_probabilities):.6f}"
    )

    print(
        f"Probability maximum: "
        f"{np.max(all_probabilities):.6f}"
    )

    print(
        f"Probability mean: "
        f"{np.mean(all_probabilities):.6f}"
    )

    print(
        f"Probability median: "
        f"{np.median(all_probabilities):.6f}"
    )

    print(
        "\nArea above probability thresholds:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        f"\nModel-positive area "
        f"at frozen threshold 0.50: "
        f"{predicted_positive_count * pixel_area_ha:.2f} ha"
    )

    print(
        "\nOutputs:"
    )

    print(
        PROBABILITY_OUTPUT
    )

    print(
        MODEL_POSITIVE_OUTPUT
    )

    print(
        SUMMARY_OUTPUT
    )

    print(
        AUDIT_OUTPUT
    )

    print(
        "\n========================================"
    )

    print(
        "FULL-AREA PROBABILITY MAPPING: COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        "\nIMPORTANT:"
        "\nThis is a model probability surface."
        "\nThe 0.50 raster is only a model-positive mask."
        "\nDo not yet call every positive pixel a "
        "candidate wetland."
        "\nKnown wetlands, open-water confusion and "
        "minimum-area filtering will be handled "
        "in the next spatial post-processing stage."
    )


if __name__ == "__main__":
    main()