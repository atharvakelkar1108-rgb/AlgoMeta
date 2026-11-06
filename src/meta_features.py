import os
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROCESSED_DIR = "data/processed"
CONFIG_FILE = "data/dataset_config.csv"
OUTPUT_FILE = "results/meta_features.csv"


# ============================================================
# CREATE RESULTS FOLDER
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# START MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("ML ALGORITHM RECOMMENDATION SYSTEM")
print("DAY 4 - META-FEATURE EXTRACTION")
print("=" * 70)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not os.path.exists(PROCESSED_DIR):

    print(
        "\nERROR: Processed data folder not found:"
    )

    print(PROCESSED_DIR)

    print(
        "\nRun Day 3 preprocessing first."
    )

    exit()


if not os.path.exists(CONFIG_FILE):

    print(
        "\nERROR: dataset_config.csv not found."
    )

    exit()


# ============================================================
# LOAD CONFIGURATION
# ============================================================

config = pd.read_csv(
    CONFIG_FILE
)


print("\nDatasets found in configuration:")

print(
    config["dataset_name"].tolist()
)


# ============================================================
# FUNCTION TO CALCULATE ENTROPY
# ============================================================

def calculate_entropy(series):

    probabilities = (
        series.value_counts(
            normalize=True
        )
    )

    entropy = 0.0

    for probability in probabilities:

        if probability > 0:

            entropy -= (
                probability
                * np.log2(probability)
            )

    return entropy


# ============================================================
# FUNCTION TO CALCULATE META-FEATURES
# ============================================================

def extract_meta_features(
    dataset_name,
    target_column
):

    print("\n")
    print("=" * 70)

    print(
        f"Processing: {dataset_name}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Find processed dataset
    # --------------------------------------------------------

    input_file = os.path.join(
        PROCESSED_DIR,
        dataset_name +
        "_processed.csv"
    )


    if not os.path.exists(input_file):

        print(
            f"ERROR: Processed dataset not found:"
        )

        print(input_file)

        return None


    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        input_file
    )


    print(
        f"Dataset shape: {df.shape}"
    )


    # --------------------------------------------------------
    # Check target
    # --------------------------------------------------------

    if target_column not in df.columns:

        print(
            f"ERROR: Target column "
            f"'{target_column}' not found."
        )

        return None


    # --------------------------------------------------------
    # Separate X and y
    # --------------------------------------------------------

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]


    # ========================================================
    # BASIC DATASET FEATURES
    # ========================================================

    n_samples = X.shape[0]

    n_features = X.shape[1]


    # ========================================================
    # NUMERICAL / CATEGORICAL FEATURES
    # ========================================================

    numerical_columns = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
    )

    categorical_columns = (
        X.select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        )
        .columns
    )


    n_numerical_features = (
        len(numerical_columns)
    )

    n_categorical_features = (
        len(categorical_columns)
    )


    # ========================================================
    # MISSING VALUES
    # ========================================================

    total_missing_values = (
        X.isnull()
        .sum()
        .sum()
    )


    missing_value_ratio = (
        total_missing_values
        / (n_samples * n_features)
        if n_samples * n_features > 0
        else 0
    )


    # ========================================================
    # TARGET INFORMATION
    # ========================================================

    n_classes = (
        y.nunique()
    )


    class_counts = (
        y.value_counts()
    )


    # --------------------------------------------------------
    # Class imbalance
    # --------------------------------------------------------

    if len(class_counts) > 0:

        majority_class_count = (
            class_counts.max()
        )

        minority_class_count = (
            class_counts.min()
        )


        if minority_class_count > 0:

            imbalance_ratio = (
                majority_class_count
                / minority_class_count
            )

        else:

            imbalance_ratio = 0

    else:

        imbalance_ratio = 0


    # --------------------------------------------------------
    # Target entropy
    # --------------------------------------------------------

    target_entropy = (
        calculate_entropy(y)
    )


    # ========================================================
    # FEATURE STATISTICS
    # ========================================================

    if n_features > 0:

        feature_means = (
            X.mean(
                numeric_only=True
            )
        )

        feature_stds = (
            X.std(
                numeric_only=True
            )
        )

        feature_medians = (
            X.median(
                numeric_only=True
            )
        )

        feature_mins = (
            X.min(
                numeric_only=True
            )
        )

        feature_maxs = (
            X.max(
                numeric_only=True
            )
        )

        feature_skewness = (
            X.skew(
                numeric_only=True
            )
        )


        # ----------------------------------------------------
        # Average statistics
        # ----------------------------------------------------

        mean_of_means = (
            feature_means.mean()
        )

        mean_of_stds = (
            feature_stds.mean()
        )

        mean_of_medians = (
            feature_medians.mean()
        )

        mean_of_mins = (
            feature_mins.mean()
        )

        mean_of_maxs = (
            feature_maxs.mean()
        )

        mean_skewness = (
            feature_skewness.mean()
        )


        # ----------------------------------------------------
        # Feature variation
        # ----------------------------------------------------

        std_of_means = (
            feature_means.std()
        )

        std_of_stds = (
            feature_stds.std()
        )


    else:

        mean_of_means = 0
        mean_of_stds = 0
        mean_of_medians = 0
        mean_of_mins = 0
        mean_of_maxs = 0
        mean_skewness = 0
        std_of_means = 0
        std_of_stds = 0


    # ========================================================
    # CORRELATION FEATURES
    # ========================================================

    if n_features > 1:

        correlation_matrix = (
            X.corr(
                numeric_only=True
            )
        )


        # Get upper triangle
        correlation_values = (
            correlation_matrix
            .where(
                np.triu(
                    np.ones(
                        correlation_matrix.shape
                    ),
                    k=1
                ).astype(bool)
            )
            .stack()
        )


        if len(correlation_values) > 0:

            mean_correlation = (
                correlation_values.mean()
            )

            max_correlation = (
                correlation_values.max()
            )

            min_correlation = (
                correlation_values.min()
            )

            mean_absolute_correlation = (
                correlation_values
                .abs()
                .mean()
            )

        else:

            mean_correlation = 0
            max_correlation = 0
            min_correlation = 0
            mean_absolute_correlation = 0

    else:

        mean_correlation = 0
        max_correlation = 0
        min_correlation = 0
        mean_absolute_correlation = 0


    # ========================================================
    # DIMENSIONALITY FEATURES
    # ========================================================

    if n_samples > 0:

        features_per_sample = (
            n_features
            / n_samples
        )

    else:

        features_per_sample = 0


    if n_features > 0:

        samples_per_feature = (
            n_samples
            / n_features
        )

    else:

        samples_per_feature = 0


    # ========================================================
    # CREATE META-FEATURE RECORD
    # ========================================================

    meta_features = {

        # Dataset identification
        "dataset_name":
            dataset_name,


        # Basic information
        "n_samples":
            n_samples,

        "n_features":
            n_features,


        # Feature types
        "n_numerical_features":
            n_numerical_features,

        "n_categorical_features":
            n_categorical_features,


        # Missing values
        "total_missing_values":
            total_missing_values,

        "missing_value_ratio":
            missing_value_ratio,


        # Target information
        "n_classes":
            n_classes,

        "imbalance_ratio":
            imbalance_ratio,

        "target_entropy":
            target_entropy,


        # Feature statistics
        "mean_of_feature_means":
            mean_of_means,

        "mean_of_feature_stds":
            mean_of_stds,

        "mean_of_feature_medians":
            mean_of_medians,

        "mean_of_feature_mins":
            mean_of_mins,

        "mean_of_feature_maxs":
            mean_of_maxs,

        "mean_feature_skewness":
            mean_skewness,


        # Variation between features
        "std_of_feature_means":
            std_of_means,

        "std_of_feature_stds":
            std_of_stds,


        # Correlation
        "mean_correlation":
            mean_correlation,

        "max_correlation":
            max_correlation,

        "min_correlation":
            min_correlation,

        "mean_absolute_correlation":
            mean_absolute_correlation,


        # Dimensionality
        "features_per_sample":
            features_per_sample,

        "samples_per_feature":
            samples_per_feature
    }


    # ========================================================
    # DISPLAY META-FEATURES
    # ========================================================

    print("\nExtracted meta-features:")

    for key, value in meta_features.items():

        print(
            f"{key}: {value}"
        )


    return meta_features


# ============================================================
# PROCESS ALL DATASETS
# ============================================================

all_meta_features = []


print(
    "\n\nStarting meta-feature extraction..."
)


for _, row in config.iterrows():

    dataset_name = (
        str(
            row["dataset_name"]
        ).strip()
    )


    target_column = (
        str(
            row["target_column"]
        ).strip()
    )


    result = extract_meta_features(
        dataset_name,
        target_column
    )


    if result is not None:

        all_meta_features.append(
            result
        )


# ============================================================
# CREATE META-FEATURE DATAFRAME
# ============================================================

if len(all_meta_features) == 0:

    print(
        "\nERROR: No meta-features were created."
    )

    exit()


meta_features_df = pd.DataFrame(
    all_meta_features
)


# ============================================================
# SAVE META-FEATURES
# ============================================================

meta_features_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n\n"
    + "=" * 70
)

print(
    "META-FEATURE EXTRACTION COMPLETED"
)

print(
    "=" * 70
)


print(
    f"\nDatasets processed: "
    f"{len(meta_features_df)}"
)


print(
    f"Number of meta-features: "
    f"{meta_features_df.shape[1] - 1}"
)


print(
    "\nMeta-feature dataset:"
)


print(
    meta_features_df
)


print(
    "\nSaved to:"
)


print(
    OUTPUT_FILE
)


print(
    "\nDay 4 completed successfully!"
)


print(
    "=" * 70
)