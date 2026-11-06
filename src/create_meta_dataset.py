import os
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

META_FEATURES_FILE = "results/meta_features.csv"

ALGORITHM_PERFORMANCE_FILE = (
    "results/algorithm_performance.csv"
)

OUTPUT_FILE = "results/meta_dataset.csv"


# ============================================================
# START MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("ML ALGORITHM RECOMMENDATION SYSTEM")
print("DAY 6 - META-DATASET CREATION")
print("=" * 70)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

print("\nChecking required files...")


if not os.path.exists(META_FEATURES_FILE):

    print(
        "\nERROR: Meta-features file not found:"
    )

    print(
        META_FEATURES_FILE
    )

    print(
        "\nRun Day 4 first."
    )

    exit()


if not os.path.exists(
    ALGORITHM_PERFORMANCE_FILE
):

    print(
        "\nERROR: Algorithm performance "
        "file not found:"
    )

    print(
        ALGORITHM_PERFORMANCE_FILE
    )

    print(
        "\nRun Day 5 first."
    )

    exit()


print(
    "meta_features.csv: OK"
)

print(
    "algorithm_performance.csv: OK"
)


# ============================================================
# LOAD FILES
# ============================================================

print(
    "\nLoading files..."
)


try:

    meta_features = pd.read_csv(
        META_FEATURES_FILE
    )

    algorithm_performance = pd.read_csv(
        ALGORITHM_PERFORMANCE_FILE
    )

except Exception as e:

    print(
        "\nERROR while reading files:"
    )

    print(e)

    exit()


# ============================================================
# DISPLAY INPUT INFORMATION
# ============================================================

print(
    "\nMeta-features shape:"
)

print(
    meta_features.shape
)


print(
    "\nAlgorithm performance shape:"
)

print(
    algorithm_performance.shape
)


# ============================================================
# CHECK dataset_name COLUMN
# ============================================================

if "dataset_name" not in meta_features.columns:

    print(
        "\nERROR: dataset_name missing "
        "from meta_features.csv"
    )

    exit()


if (
    "dataset_name"
    not in algorithm_performance.columns
):

    print(
        "\nERROR: dataset_name missing "
        "from algorithm_performance.csv"
    )

    exit()


# ============================================================
# CHECK DUPLICATES IN META FEATURES
# ============================================================

duplicate_datasets = (
    meta_features[
        "dataset_name"
    ]
    .duplicated()
    .sum()
)


print(
    f"\nDuplicate dataset entries "
    f"in meta_features: "
    f"{duplicate_datasets}"
)


if duplicate_datasets > 0:

    print(
        "\nWARNING:"
    )

    print(
        "Multiple meta-feature rows "
        "found for the same dataset."
    )


# ============================================================
# CHECK DATASET NAMES
# ============================================================

meta_dataset_names = set(
    meta_features[
        "dataset_name"
    ]
)


performance_dataset_names = set(
    algorithm_performance[
        "dataset_name"
    ]
)


print(
    "\nDatasets in meta-features:"
)

print(
    sorted(meta_dataset_names)
)


print(
    "\nDatasets in algorithm performance:"
)

print(
    sorted(performance_dataset_names)
)


# ============================================================
# FIND MISSING DATASETS
# ============================================================

missing_in_meta = (
    performance_dataset_names
    - meta_dataset_names
)


missing_in_performance = (
    meta_dataset_names
    - performance_dataset_names
)


if len(missing_in_meta) > 0:

    print(
        "\nWARNING: These datasets are "
        "missing from meta_features.csv:"
    )

    print(
        sorted(missing_in_meta)
    )


if len(missing_in_performance) > 0:

    print(
        "\nWARNING: These datasets are "
        "missing from algorithm_performance.csv:"
    )

    print(
        sorted(missing_in_performance)
    )


# ============================================================
# MERGE DATASETS
# ============================================================

print(
    "\nMerging datasets..."
)


meta_dataset = pd.merge(

    algorithm_performance,

    meta_features,

    on="dataset_name",

    how="inner",

    suffixes=(
        "_performance",
        "_meta"
    )
)


# ============================================================
# CHECK MERGE RESULT
# ============================================================

print(
    "\nMeta-dataset shape:"
)

print(
    meta_dataset.shape
)


if len(meta_dataset) == 0:

    print(
        "\nERROR: Merge produced "
        "zero rows."
    )

    exit()


# ============================================================
# REORDER COLUMNS
# ============================================================

# We want dataset information first,
# followed by algorithm and performance.

preferred_columns = [

    "dataset_name",

    "algorithm",

    "accuracy_mean",

    "accuracy_std",

    "f1_mean",

    "f1_std",

    "training_time_seconds",

    "n_samples",

    "n_features",

    "n_numerical_features",

    "n_categorical_features",

    "total_missing_values",

    "missing_value_ratio",

    "n_classes",

    "imbalance_ratio",

    "target_entropy",

    "mean_of_feature_means",

    "mean_of_feature_stds",

    "mean_of_feature_medians",

    "mean_of_feature_mins",

    "mean_of_feature_maxs",

    "mean_feature_skewness",

    "std_of_feature_means",

    "std_of_feature_stds",

    "mean_correlation",

    "max_correlation",

    "min_correlation",

    "mean_absolute_correlation",

    "features_per_sample",

    "samples_per_feature"
]


# Keep only columns that actually exist

existing_columns = [

    column

    for column in preferred_columns

    if column in meta_dataset.columns
]


remaining_columns = [

    column

    for column in meta_dataset.columns

    if column not in existing_columns
]


meta_dataset = meta_dataset[
    existing_columns
    + remaining_columns
]


# ============================================================
# SORT DATASET
# ============================================================

meta_dataset = (
    meta_dataset
    .sort_values(
        by=[
            "dataset_name",
            "algorithm"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print(
    "\nChecking missing values..."
)


missing_values = (
    meta_dataset
    .isnull()
    .sum()
)


total_missing = (
    missing_values.sum()
)


print(
    f"Total missing values: "
    f"{total_missing}"
)


if total_missing > 0:

    print(
        "\nColumns containing "
        "missing values:"
    )

    print(
        missing_values[
            missing_values > 0
        ]
    )


# ============================================================
# CHECK NUMBER OF ALGORITHMS
# ============================================================

if "algorithm" in meta_dataset.columns:

    algorithms = (
        meta_dataset[
            "algorithm"
        ]
        .unique()
        .tolist()
    )


    print(
        "\nAlgorithms found:"
    )

    for algorithm in algorithms:

        print(
            f"  - {algorithm}"
        )


# ============================================================
# DATASET ROW COUNTS
# ============================================================

print(
    "\nRows per dataset:"
)


row_counts = (
    meta_dataset[
        "dataset_name"
    ]
    .value_counts()
)


print(
    row_counts
)


# ============================================================
# SAVE META-DATASET
# ============================================================

print(
    "\nSaving meta-dataset..."
)


try:

    meta_dataset.to_csv(

        OUTPUT_FILE,

        index=False
    )

except Exception as e:

    print(
        "\nERROR while saving "
        "meta_dataset.csv:"
    )

    print(e)

    exit()


# ============================================================
# DISPLAY FINAL DATASET
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "META-DATASET CREATED SUCCESSFULLY"
)

print(
    "=" * 70
)


print(
    "\nFinal shape:"
)

print(
    meta_dataset.shape
)


print(
    "\nColumns:"
)

for index, column in enumerate(
    meta_dataset.columns,
    start=1
):

    print(
        f"{index}. {column}"
    )


print(
    "\nFirst 10 rows:"
)

print(
    meta_dataset.head(
        10
    ).to_string(
        index=False
    )
)


# ============================================================
# FINAL FILE
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "OUTPUT FILE:"
)

print(
    OUTPUT_FILE
)


print(
    "\nDay 6 completed successfully!"
)

print(
    "=" * 70
)