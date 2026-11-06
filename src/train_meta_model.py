# ============================================================
# src/train_meta_model.py
# META-LEARNING MODEL TRAINING
# ============================================================

import os
import joblib
import warnings

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

META_DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "meta_dataset.csv"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

MODEL_FILE = os.path.join(
    RESULTS_DIR,
    "meta_model.joblib"
)

PREDICTIONS_FILE = os.path.join(
    RESULTS_DIR,
    "meta_model_predictions.csv"
)

FEATURE_INFO_FILE = os.path.join(
    RESULTS_DIR,
    "meta_model_info.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "f1_mean"

ALGORITHM_COLUMN = "algorithm"


# ============================================================
# META-FEATURES
# ============================================================

META_FEATURES = [

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


# ============================================================
# START
# ============================================================

print("\n" + "=" * 70)
print("META-LEARNING MODEL TRAINING")
print("=" * 70)


# ============================================================
# CHECK FILE
# ============================================================

if not os.path.exists(META_DATASET_PATH):

    print("\nERROR: meta_dataset.csv not found.")

    print(
        "\nExpected location:"
    )

    print(
        META_DATASET_PATH
    )

    raise SystemExit


# ============================================================
# LOAD META DATASET
# ============================================================

print("\nLoading meta-dataset...")

df = pd.read_csv(
    META_DATASET_PATH
)

print(
    f"Meta-dataset shape: {df.shape}"
)


print(
    f"\nDatasets: "
    f"{df['dataset_name'].nunique()}"
)

print(
    f"Algorithms: "
    f"{df['algorithm'].nunique()}"
)


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

print("\nChecking meta-feature columns...")


# Your create_meta_dataset.py currently creates:
#
# n_samples_performance
# n_features_performance
# n_classes_performance
#
# OR:
#
# n_samples_meta
# n_features_meta
# n_classes_meta
#
# We convert them to:
#
# n_samples
# n_features
# n_classes


COLUMN_ALIASES = {

    "n_samples": [
        "n_samples",
        "n_samples_meta",
        "n_samples_performance"
    ],

    "n_features": [
        "n_features",
        "n_features_meta",
        "n_features_performance"
    ],

    "n_classes": [
        "n_classes",
        "n_classes_meta",
        "n_classes_performance"
    ]
}


for canonical_name, possible_names in COLUMN_ALIASES.items():

    if canonical_name in df.columns:

        continue

    found_column = None

    for column in possible_names:

        if column in df.columns:

            found_column = column
            break

    if found_column is not None:

        print(
            f"Using '{found_column}' "
            f"as '{canonical_name}'"
        )

        df[canonical_name] = df[
            found_column
        ]


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [

    "dataset_name",
    "algorithm",
    TARGET_COLUMN
] + META_FEATURES


missing_columns = [

    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    print("\nERROR: Missing columns:")

    for column in missing_columns:

        print(
            f"  - {column}"
        )

    print("\nAvailable columns:")

    for column in df.columns:

        print(
            f"  - {column}"
        )

    raise SystemExit


print(
    "\nAll required columns found."
)


# ============================================================
# CHECK META FEATURES
# ============================================================

print("\nMeta-features used by the model:")

for feature in META_FEATURES:

    print(
        f"  - {feature}"
    )

print(
    f"\nNumber of meta-features: "
    f"{len(META_FEATURES)}"
)


# ============================================================
# CHECK ALGORITHMS
# ============================================================

algorithms = sorted(
    df[ALGORITHM_COLUMN]
    .dropna()
    .unique()
    .tolist()
)

print("\nAlgorithms:")

for algorithm in algorithms:

    print(
        f"  - {algorithm}"
    )


# ============================================================
# PREPARE X AND Y
# ============================================================

# We use:
#
# dataset meta-features
# +
# algorithm name
#
# to predict:
#
# F1 score


X = df[
    META_FEATURES + [
        ALGORITHM_COLUMN
    ]
].copy()


y = df[
    TARGET_COLUMN
].copy()


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

valid_rows = (

    y.notna()
    &
    X.notna().all(axis=1)
)


X = X[
    valid_rows
].reset_index(
    drop=True
)

y = y[
    valid_rows
].reset_index(
    drop=True
)


print(
    "\nTraining rows:",
    len(X)
)


# ============================================================
# NUMERICAL FEATURES
# ============================================================

numerical_features = META_FEATURES


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    ALGORITHM_COLUMN
]


print(
    "\nCategorical feature:"
)

print(
    categorical_features
)


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "algorithm",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),

        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ============================================================
# MODEL
# ============================================================

model = RandomForestRegressor(

    n_estimators=300,

    random_state=42,

    n_jobs=-1,

    max_features="sqrt"
)


# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline(

    steps=[

        (
            "preprocessing",
            preprocessor
        ),

        (
            "model",
            model
        )
    ]
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print(
    "\nSplitting data..."
)

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42
)


print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# TRAIN
# ============================================================

print(
    "\nTraining meta-learning model..."
)

pipeline.fit(
    X_train,
    y_train
)


print(
    "Training completed."
)


# ============================================================
# PREDICTION
# ============================================================

print(
    "\nGenerating predictions..."
)

y_pred = pipeline.predict(
    X_test
)


# ============================================================
# EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("META-LEARNING MODEL RESULTS")
print("=" * 70)

print(
    f"\nMAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_df = X_test.copy()

prediction_df[
    "actual_performance"
] = y_test.values

prediction_df[
    "predicted_performance"
] = y_pred


try:

    prediction_df.to_csv(

        PREDICTIONS_FILE,

        index=False
    )

    print(
        "\nPredictions saved to:"
    )

    print(
        PREDICTIONS_FILE
    )

except Exception as e:

    print(
        f"\nWarning: Could not save {PREDICTIONS_FILE}: {e}"
    )


# ============================================================
# SAVE TRAINED MODEL
# ============================================================

print(
    "\nSaving trained meta-learning model..."
)

joblib.dump(

    pipeline,

    MODEL_FILE
)


print(
    "Model saved to:"
)

print(
    MODEL_FILE
)


# ============================================================
# SAVE MODEL INFORMATION
# ============================================================

model_info = pd.DataFrame({

    "property": [

        "target",

        "number_of_meta_features",

        "number_of_algorithms",

        "algorithms",

        "training_rows",

        "testing_rows",

        "mae",

        "rmse",

        "r2",

        "model_file"
    ],

    "value": [

        TARGET_COLUMN,

        len(META_FEATURES),

        len(algorithms),

        ", ".join(algorithms),

        len(X_train),

        len(X_test),

        mae,

        rmse,

        r2,

        MODEL_FILE
    ]
})


model_info.to_csv(

    FEATURE_INFO_FILE,

    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("META-LEARNING TRAINING COMPLETED")
print("=" * 70)

print(
    "\nDatasets used:",
    df["dataset_name"].nunique()
)

print(
    "Algorithms used:",
    len(algorithms)
)

print(
    "Meta-features:",
    len(META_FEATURES)
)

print(
    "Training rows:",
    len(X_train)
)

print(
    "Testing rows:",
    len(X_test)
)

print(
    "\nModel:"
)

print(
    MODEL_FILE
)

print(
    "\nReady for Top-3 algorithm recommendation."
)

print("=" * 70)