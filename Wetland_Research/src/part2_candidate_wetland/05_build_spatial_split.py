from pathlib import Path

import numpy as np
import pandas as pd


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

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "part2_candidate_wetland"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "part2_candidate_wetland"
)

INTERIM_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUT
# ============================================================

TRAINING_INPUT = (
    PROCESSED_DIR
    / "training_samples_raw.csv"
)

WETLAND_AUDIT_INPUT = (
    INTERIM_DIR
    / "known_wetland_sampling_audit.csv"
)


# ============================================================
# OUTPUTS
# ============================================================

FULL_SPLIT_OUTPUT = (
    SPLIT_DIR
    / "training_samples_spatial_split.csv"
)

TRAIN_OUTPUT = (
    SPLIT_DIR
    / "train_samples.csv"
)

VALIDATION_OUTPUT = (
    SPLIT_DIR
    / "validation_samples.csv"
)

TEST_OUTPUT = (
    SPLIT_DIR
    / "test_samples.csv"
)

GROUP_AUDIT_OUTPUT = (
    INTERIM_DIR
    / "spatial_group_audit.csv"
)

SPLIT_AUDIT_OUTPUT = (
    INTERIM_DIR
    / "spatial_split_audit.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

TARGET_PROPORTIONS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

# Number of deterministic candidate assignments to test.
N_SEARCH_ITERATIONS = 100000

# Permitted deviation from target split proportions.
# Example:
# validation target = 0.15
# accepted range = 0.12 to 0.18.
SPLIT_SIZE_TOLERANCE = 0.03

# Minimum number of directly interpreted 2025 LULC
# reference samples required in validation and test.
MIN_HELDOUT_REFERENCE_COUNTS = {
    "bare_fill_disturbed": 4,
    "built_up": 4,
    "open_water": 4,
    "other_vegetation": 4,
    "paddy_agriculture": 5,
    "wetland_marsh": 5,
}

# Validation and test must each contain supplementary
# evidence from at least this many independent known wetlands.
MIN_HELDOUT_WETLAND_SITES = 3


# ============================================================
# UNION-FIND
#
# Used to join 1 km spatial blocks that contain samples from
# the same known wetland.
#
# This ensures that all supplementary samples belonging to a
# single wetland site remain in one model split.
# ============================================================

class UnionFind:

    def __init__(
        self,
        items,
    ):

        self.parent = {
            item: item
            for item in items
        }

    def find(
        self,
        item,
    ):

        if (
            self.parent[item]
            != item
        ):

            self.parent[item] = (
                self.find(
                    self.parent[item]
                )
            )

        return self.parent[item]

    def union(
        self,
        a,
        b,
    ):

        root_a = self.find(
            a
        )

        root_b = self.find(
            b
        )

        if (
            root_a
            != root_b
        ):

            self.parent[
                root_b
            ] = root_a


# ============================================================
# SPLIT SCORING
# ============================================================

def score_candidate(
    df,
):

    total_n = len(
        df
    )

    global_positive_rate = (
        df[
            "target"
        ].mean()
    )

    # Direct manually interpreted 2025 LULC references.
    direct_reference = df[
        df[
            "sample_source"
        ]
        ==
        "lulc_reference_2025"
    ]

    global_reference_counts = (
        direct_reference[
            "reference_label"
        ]
        .value_counts()
        .to_dict()
    )

    score = 0.0

    for (
        split_name,
        target_prop,
    ) in TARGET_PROPORTIONS.items():

        subset = df[
            df[
                "split"
            ]
            ==
            split_name
        ]

        # Every requested split must exist.
        if (
            len(
                subset
            )
            == 0
        ):

            return np.inf

        # ----------------------------------------------------
        # 1. SPLIT SIZE
        # ----------------------------------------------------

        actual_prop = (
            len(
                subset
            )
            /
            total_n
        )

        # Reject candidates too far from 70 / 15 / 15.
        if (
            abs(
                actual_prop
                -
                target_prop
            )
            >
            SPLIT_SIZE_TOLERANCE
        ):

            return np.inf

        # Prefer candidates closer to the exact target.
        score += (
            abs(
                actual_prop
                -
                target_prop
            )
            *
            30.0
        )

        # ----------------------------------------------------
        # 2. BINARY CLASS BALANCE
        # ----------------------------------------------------

        positives = int(
            (
                subset[
                    "target"
                ]
                == 1
            ).sum()
        )

        negatives = int(
            (
                subset[
                    "target"
                ]
                == 0
            ).sum()
        )

        # Every split must contain both classes.
        if (
            positives == 0
            or
            negatives == 0
        ):

            return np.inf

        positive_rate = (
            positives
            /
            len(
                subset
            )
        )

        # Prefer class proportions similar to the
        # complete labelled dataset.
        score += (
            abs(
                positive_rate
                -
                global_positive_rate
            )
            *
            10.0
        )

        # ----------------------------------------------------
        # 3. DIRECT 2025 REFERENCE REPRESENTATION
        # ----------------------------------------------------

        direct_subset = subset[
            subset[
                "sample_source"
            ]
            ==
            "lulc_reference_2025"
        ]

        for (
            reference_label,
            global_count,
        ) in global_reference_counts.items():

            split_count = int(
                (
                    direct_subset[
                        "reference_label"
                    ]
                    ==
                    reference_label
                ).sum()
            )

            # Every split must contain evidence from every
            # directly interpreted reference class.
            if (
                split_count
                == 0
            ):

                return np.inf

            expected_count = (
                global_count
                *
                target_prop
            )

            relative_deviation = (
                abs(
                    split_count
                    -
                    expected_count
                )
                /
                max(
                    expected_count,
                    1.0,
                )
            )

            score += (
                relative_deviation
                *
                2.0
            )

            # Validation and test require stronger minimum
            # representation of each direct reference class.
            if (
                split_name
                in [
                    "validation",
                    "test",
                ]
            ):

                minimum_required = (
                    MIN_HELDOUT_REFERENCE_COUNTS[
                        reference_label
                    ]
                )

                if (
                    split_count
                    <
                    minimum_required
                ):

                    return np.inf

        # ----------------------------------------------------
        # 4. HELD-OUT KNOWN WETLAND SITES
        # ----------------------------------------------------

        if (
            split_name
            in [
                "validation",
                "test",
            ]
        ):

            known_sites = (
                subset.loc[
                    subset[
                        "sample_source"
                    ]
                    ==
                    "known_wetland_polygon",
                    "wetland_id",
                ]
                .dropna()
                .astype(
                    str
                )
                .nunique()
            )

            if (
                known_sites
                <
                MIN_HELDOUT_WETLAND_SITES
            ):

                return np.inf

    return score


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "PART 2 SPATIAL SPLIT"
    )

    print(
        "========================================"
    )

    # ========================================================
    # LOAD TRAINING DATA
    # ========================================================

    data = pd.read_csv(
        TRAINING_INPUT
    )

    print(
        f"Samples loaded: "
        f"{len(data)}"
    )

    print(
        f"Original 1 km blocks: "
        f"{data['spatial_block_id'].nunique()}"
    )

    print(
        f"Positive samples: "
        f"{int((data['target'] == 1).sum())}"
    )

    print(
        f"Negative samples: "
        f"{int((data['target'] == 0).sum())}"
    )

    # ========================================================
    # 1. CHECK WETLAND SAMPLING FALLBACKS
    # ========================================================

    wetland_audit = pd.read_csv(
        WETLAND_AUDIT_INPUT
    )

    fallback = wetland_audit[
        wetland_audit[
            "inner_buffer_fallback"
        ]
        .astype(
            str
        )
        .str
        .lower()
        ==
        "true"
    ]

    print(
        "\nKnown wetlands requiring "
        "inner-buffer fallback:"
    )

    if (
        len(
            fallback
        )
        == 0
    ):

        print(
            "None"
        )

    else:

        print(
            fallback[
                [
                    "wetland_id",
                    "observation_date",
                    "available_pixels",
                    "sampled_pixels",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # ========================================================
    # 2. BUILD CONNECTED SPATIAL GROUPS
    # ========================================================

    blocks = sorted(
        data[
            "spatial_block_id"
        ]
        .unique()
    )

    uf = UnionFind(
        blocks
    )

    # Samples from a single known wetland must never appear
    # in multiple train/validation/test sets.
    polygon_samples = data[
        data[
            "sample_source"
        ]
        ==
        "known_wetland_polygon"
    ].copy()

    wetland_block_audit = []

    for (
        wetland_id,
        group,
    ) in polygon_samples.groupby(
        "wetland_id"
    ):

        wetland_blocks = sorted(
            group[
                "spatial_block_id"
            ]
            .unique()
        )

        wetland_block_audit.append(
            {
                "wetland_id":
                    wetland_id,

                "spatial_block_count":
                    len(
                        wetland_blocks
                    ),

                "spatial_blocks":
                    "|".join(
                        wetland_blocks
                    ),
            }
        )

        # Merge all blocks containing the same wetland.
        if (
            len(
                wetland_blocks
            )
            > 1
        ):

            first_block = (
                wetland_blocks[
                    0
                ]
            )

            for other_block in (
                wetland_blocks[
                    1:
                ]
            ):

                uf.union(
                    first_block,
                    other_block,
                )

    # ========================================================
    # 3. ASSIGN CONNECTED SPATIAL GROUP IDS
    # ========================================================

    roots = {
        block:
            uf.find(
                block
            )
        for block
        in blocks
    }

    unique_roots = sorted(
        set(
            roots.values()
        )
    )

    root_to_group = {
        root:
            f"G{i:03d}"
        for (
            i,
            root,
        ) in enumerate(
            unique_roots,
            start=1,
        )
    }

    data[
        "spatial_group_id"
    ] = (
        data[
            "spatial_block_id"
        ]
        .map(
            lambda block:
                root_to_group[
                    roots[
                        block
                    ]
                ]
        )
    )

    print(
        f"Connected spatial groups: "
        f"{data['spatial_group_id'].nunique()}"
    )

    wetlands_crossing_blocks = (
        pd.DataFrame(
            wetland_block_audit
        )
    )

    crossing_count = int(
        (
            wetlands_crossing_blocks[
                "spatial_block_count"
            ]
            > 1
        ).sum()
    )

    print(
        f"Known wetlands spanning >1 block: "
        f"{crossing_count}"
    )

    print(
        f"Known wetlands represented: "
        f"{polygon_samples['wetland_id'].nunique()}"
    )

    # ========================================================
    # 4. GROUP-LEVEL SUMMARY
    # ========================================================

    group_summary = (
        data
        .groupby(
            "spatial_group_id"
        )
        .agg(
            sample_count=(
                "sample_id",
                "count",
            ),

            positive_count=(
                "target",
                "sum",
            ),

            spatial_block_count=(
                "spatial_block_id",
                "nunique",
            ),

            wetland_site_count=(
                "wetland_id",
                lambda s:
                    s[
                        s.notna()
                        &
                        (
                            s.astype(
                                str
                            )
                            != ""
                        )
                    ]
                    .nunique(),
            ),
        )
        .reset_index()
    )

    group_summary[
        "negative_count"
    ] = (
        group_summary[
            "sample_count"
        ]
        -
        group_summary[
            "positive_count"
        ]
    )

    group_summary.to_csv(
        GROUP_AUDIT_OUTPUT,
        index=False,
    )

    # ========================================================
    # 5. SEARCH FOR A GOOD GROUPED SPLIT
    # ========================================================

    groups = (
        group_summary[
            "spatial_group_id"
        ]
        .tolist()
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    best_score = np.inf
    best_assignment = None

    split_names = np.array(
        [
            "train",
            "validation",
            "test",
        ]
    )

    split_probabilities = np.array(
        [
            TARGET_PROPORTIONS[
                "train"
            ],
            TARGET_PROPORTIONS[
                "validation"
            ],
            TARGET_PROPORTIONS[
                "test"
            ],
        ]
    )

    # --------------------------------------------------------
    # LARGE-GROUP SAFETY RULE
    #
    # Any connected group larger than the maximum permitted
    # validation/test size cannot sensibly be assigned to a
    # held-out split while satisfying the split-size rule.
    #
    # Therefore such groups remain in training.
    # --------------------------------------------------------

    maximum_heldout_size = (
        len(
            data
        )
        *
        (
            TARGET_PROPORTIONS[
                "validation"
            ]
            +
            SPLIT_SIZE_TOLERANCE
        )
    )

    forced_train_groups = set(
        group_summary.loc[
            group_summary[
                "sample_count"
            ]
            >
            maximum_heldout_size,
            "spatial_group_id",
        ]
    )

    print(
        f"Groups forced to train due to size: "
        f"{len(forced_train_groups)}"
    )

    if (
        len(
            forced_train_groups
        )
        > 0
    ):

        print(
            "Forced train groups:"
        )

        print(
            group_summary[
                group_summary[
                    "spatial_group_id"
                ]
                .isin(
                    forced_train_groups
                )
            ][
                [
                    "spatial_group_id",
                    "sample_count",
                    "positive_count",
                    "negative_count",
                    "wetland_site_count",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # RANDOMIZED DETERMINISTIC SEARCH
    # --------------------------------------------------------

    for _ in range(
        N_SEARCH_ITERATIONS
    ):

        random_splits = (
            rng.choice(
                split_names,
                size=len(
                    groups
                ),
                p=split_probabilities,
            )
        )

        candidate_assignment = {
            group:
                (
                    "train"
                    if (
                        group
                        in forced_train_groups
                    )
                    else
                    random_split
                )
            for (
                group,
                random_split,
            ) in zip(
                groups,
                random_splits,
            )
        }

        candidate = (
            data.copy()
        )

        candidate[
            "split"
        ] = (
            candidate[
                "spatial_group_id"
            ]
            .map(
                candidate_assignment
            )
        )

        candidate_score = (
            score_candidate(
                candidate
            )
        )

        if (
            candidate_score
            <
            best_score
        ):

            best_score = (
                candidate_score
            )

            best_assignment = (
                candidate_assignment.copy()
            )

    # ========================================================
    # APPLY BEST SPLIT
    # ========================================================

    if (
        best_assignment is None
        or
        not np.isfinite(
            best_score
        )
    ):

        raise RuntimeError(
            "Unable to construct a valid spatial split "
            "satisfying the required constraints. "
            "Review the spatial-group distribution or "
            "adjust the documented split constraints."
        )

    data[
        "split"
    ] = (
        data[
            "spatial_group_id"
        ]
        .map(
            best_assignment
        )
    )

    if (
        data[
            "split"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "One or more spatial groups were "
            "not assigned to a split."
        )

    # ========================================================
    # 6. FINAL LEAKAGE CHECKS
    # ========================================================

    block_split_counts = (
        data
        .groupby(
            "spatial_block_id"
        )[
            "split"
        ]
        .nunique()
    )

    block_leakage = int(
        (
            block_split_counts
            > 1
        ).sum()
    )

    wetland_split_counts = (
        data[
            data[
                "sample_source"
            ]
            ==
            "known_wetland_polygon"
        ]
        .groupby(
            "wetland_id"
        )[
            "split"
        ]
        .nunique()
    )

    wetland_leakage = int(
        (
            wetland_split_counts
            > 1
        ).sum()
    )

    group_split_counts = (
        data
        .groupby(
            "spatial_group_id"
        )[
            "split"
        ]
        .nunique()
    )

    group_leakage = int(
        (
            group_split_counts
            > 1
        ).sum()
    )

    if (
        block_leakage != 0
        or
        wetland_leakage != 0
        or
        group_leakage != 0
    ):

        raise RuntimeError(
            "Spatial leakage detected."
        )

    # ========================================================
    # 7. FINAL SPLIT CONSTRAINT VALIDATION
    # ========================================================

    final_constraint_rows = []

    for (
        split_name,
        target_prop,
    ) in TARGET_PROPORTIONS.items():

        subset = data[
            data[
                "split"
            ]
            ==
            split_name
        ]

        actual_prop = (
            len(
                subset
            )
            /
            len(
                data
            )
        )

        size_pass = (
            abs(
                actual_prop
                -
                target_prop
            )
            <=
            SPLIT_SIZE_TOLERANCE
        )

        direct_subset = subset[
            subset[
                "sample_source"
            ]
            ==
            "lulc_reference_2025"
        ]

        known_sites = (
            subset.loc[
                subset[
                    "sample_source"
                ]
                ==
                "known_wetland_polygon",
                "wetland_id",
            ]
            .dropna()
            .astype(
                str
            )
            .nunique()
        )

        if (
            split_name
            in [
                "validation",
                "test",
            ]
        ):

            heldout_reference_pass = True

            for (
                reference_label,
                minimum_count,
            ) in (
                MIN_HELDOUT_REFERENCE_COUNTS
                .items()
            ):

                observed_count = int(
                    (
                        direct_subset[
                            "reference_label"
                        ]
                        ==
                        reference_label
                    ).sum()
                )

                if (
                    observed_count
                    <
                    minimum_count
                ):

                    heldout_reference_pass = (
                        False
                    )

            wetland_site_pass = (
                known_sites
                >=
                MIN_HELDOUT_WETLAND_SITES
            )

        else:

            heldout_reference_pass = True
            wetland_site_pass = True

        final_constraint_rows.append(
            {
                "split":
                    split_name,

                "size_pass":
                    size_pass,

                "reference_minimums_pass":
                    heldout_reference_pass,

                "wetland_site_minimum_pass":
                    wetland_site_pass,
            }
        )

    final_constraint_audit = (
        pd.DataFrame(
            final_constraint_rows
        )
    )

    if not (
        final_constraint_audit[
            [
                "size_pass",
                "reference_minimums_pass",
                "wetland_site_minimum_pass",
            ]
        ]
        .all()
        .all()
    ):

        raise RuntimeError(
            "Best split failed one or more "
            "final constraint checks."
        )

    # ========================================================
    # 8. SAVE SPLITS
    # ========================================================

    data.to_csv(
        FULL_SPLIT_OUTPUT,
        index=False,
    )

    for (
        split_name,
        output_path,
    ) in [
        (
            "train",
            TRAIN_OUTPUT,
        ),
        (
            "validation",
            VALIDATION_OUTPUT,
        ),
        (
            "test",
            TEST_OUTPUT,
        ),
    ]:

        data[
            data[
                "split"
            ]
            ==
            split_name
        ].to_csv(
            output_path,
            index=False,
        )

    # ========================================================
    # 9. AUDIT SUMMARY
    # ========================================================

    audit_rows = []

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        subset = data[
            data[
                "split"
            ]
            ==
            split_name
        ]

        positive_count = int(
            (
                subset[
                    "target"
                ]
                == 1
            ).sum()
        )

        negative_count = int(
            (
                subset[
                    "target"
                ]
                == 0
            ).sum()
        )

        direct_subset = subset[
            subset[
                "sample_source"
            ]
            ==
            "lulc_reference_2025"
        ]

        known_wetland_sites = (
            subset.loc[
                subset[
                    "sample_source"
                ]
                ==
                "known_wetland_polygon",
                "wetland_id",
            ]
            .dropna()
            .astype(
                str
            )
            .nunique()
        )

        audit_rows.append(
            {
                "split":
                    split_name,

                "samples":
                    len(
                        subset
                    ),

                "percentage":
                    (
                        100.0
                        *
                        len(
                            subset
                        )
                        /
                        len(
                            data
                        )
                    ),

                "positive":
                    positive_count,

                "negative":
                    negative_count,

                "positive_rate":
                    (
                        positive_count
                        /
                        len(
                            subset
                        )
                    ),

                "spatial_groups":
                    subset[
                        "spatial_group_id"
                    ]
                    .nunique(),

                "spatial_blocks":
                    subset[
                        "spatial_block_id"
                    ]
                    .nunique(),

                "known_wetland_sites":
                    known_wetland_sites,

                "direct_reference_samples":
                    len(
                        direct_subset
                    ),

                "wetland_marsh_direct":
                    int(
                        (
                            direct_subset[
                                "reference_label"
                            ]
                            ==
                            "wetland_marsh"
                        ).sum()
                    ),

                "paddy_direct":
                    int(
                        (
                            direct_subset[
                                "reference_label"
                            ]
                            ==
                            "paddy_agriculture"
                        ).sum()
                    ),
            }
        )

    audit = pd.DataFrame(
        audit_rows
    )

    audit.to_csv(
        SPLIT_AUDIT_OUTPUT,
        index=False,
    )

    # ========================================================
    # TERMINAL REPORT
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "FINAL SPATIAL SPLIT"
    )

    print(
        "========================================"
    )

    print(
        audit.to_string(
            index=False
        )
    )

    print(
        "\nLeakage checks:"
    )

    print(
        f"Blocks appearing in multiple splits: "
        f"{block_leakage}"
    )

    print(
        f"Connected groups appearing in "
        f"multiple splits: "
        f"{group_leakage}"
    )

    print(
        f"Known wetlands appearing in "
        f"multiple splits: "
        f"{wetland_leakage}"
    )

    print(
        f"\nBest split score: "
        f"{best_score:.6f}"
    )

    print(
        "\nReference-label distribution by split:"
    )

    print(
        pd.crosstab(
            data[
                "reference_label"
            ],
            data[
                "split"
            ],
        )
        .to_string()
    )

    print(
        "\nKnown wetland sites by split:"
    )

    wetland_site_split = (
        data[
            data[
                "sample_source"
            ]
            ==
            "known_wetland_polygon"
        ][
            [
                "wetland_id",
                "split",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "split",
                "wetland_id",
            ]
        )
    )

    print(
        wetland_site_split
        .to_string(
            index=False
        )
    )

    print(
        "\nFinal constraint checks:"
    )

    print(
        final_constraint_audit
        .to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )

    print(
        FULL_SPLIT_OUTPUT
    )

    print(
        TRAIN_OUTPUT
    )

    print(
        VALIDATION_OUTPUT
    )

    print(
        TEST_OUTPUT
    )

    print(
        GROUP_AUDIT_OUTPUT
    )

    print(
        SPLIT_AUDIT_OUTPUT
    )

    print(
        "\n========================================"
    )

    print(
        "PART 2 SPATIAL SPLIT STATUS: PASS"
    )

    print(
        "========================================"
    )

    print(
        "\nIMPORTANT:"
        "\nThe split is spatially grouped."
        "\nNo 1 km spatial block may appear in "
        "more than one split."
        "\nAll supplementary samples from the same "
        "known wetland remain in the same split."
        "\nValidation and test satisfy minimum direct "
        "2025 reference-class requirements."
        "\nThe test set must now remain locked until "
        "final model evaluation."
    )


if __name__ == "__main__":
    main()