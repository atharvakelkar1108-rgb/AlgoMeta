# ============================================================
# src/benchmarks.py
# AUTOMATIC ML ALGORITHM BENCHMARKING
# ============================================================

import os
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, f1_score


warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

CONFIG_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset_config.csv"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "algorithm_performance.csv"
)

BEST_OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "best_algorithms.csv"
)


# ============================================================
# LARGE DATASET SETTINGS
# ============================================================

# Expensive algorithms can become extremely slow
# on very large datasets.

MAX_SAMPLES_EXPENSIVE = 50000


# ============================================================
# ALGORITHMS
# ============================================================

def create_algorithms():

    return {

        "Logistic Regression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),

        "Hist Gradient Boosting":
            HistGradientBoostingClassifier(
                max_iter=200,
                random_state=42
            ),

        "Extra Trees":
            ExtraTreesClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                random_state=42
            ),

        "KNN": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                KNeighborsClassifier(
                    n_neighbors=5,
                    n_jobs=-1
                )
            )
        ]),

        "SVM": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                SVC(
                    random_state=42
                )
            )
        ])
    }


# ============================================================
# LOAD DATASET CONFIGURATION
# ============================================================

def load_dataset_config():

    print("\n")
    print("=" * 70)
    print("LOADING DATASET CONFIGURATION")
    print("=" * 70)

    try:
        from src.auto_target import sync_dataset_config
    except ImportError:
        from auto_target import sync_dataset_config

    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    sync_dataset_config(raw_dir=raw_dir, config_file=CONFIG_FILE, verbose=False)

    print("\nConfiguration file:")
    print(CONFIG_FILE)

    if not os.path.exists(CONFIG_FILE):

        print(
            "\nERROR: dataset_config.csv not found."
        )

        return None

    try:

        config = pd.read_csv(
            CONFIG_FILE
        )

    except Exception as e:

        print(
            "\nERROR reading dataset_config.csv:"
        )

        print(e)

        return None

    # ========================================================
    # CHECK REQUIRED COLUMNS
    # ========================================================

    required_columns = [
        "dataset_name",
        "target_column"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in config.columns
    ]

    if missing_columns:

        print(
            "\nERROR: Required columns missing:"
        )

        print(
            missing_columns
        )

        print(
            "\nAvailable columns:"
        )

        print(
            config.columns.tolist()
        )

        return None

    # ========================================================
    # CLEAN CONFIGURATION
    # ========================================================

    config = config[
        required_columns
    ].copy()

    config["dataset_name"] = (
        config["dataset_name"]
        .astype(str)
        .str.strip()
    )

    config["target_column"] = (
        config["target_column"]
        .astype(str)
        .str.strip()
    )

    # Remove empty rows

    config = config[
        (config["dataset_name"] != "")
        &
        (config["target_column"] != "")
    ]

    # Remove duplicate datasets

    config = config.drop_duplicates(
        subset=["dataset_name"],
        keep="last"
    )

    config = config.reset_index(
        drop=True
    )

    print(
        "\nDatasets configured:",
        len(config)
    )

    print("\nDatasets:")

    for _, row in config.iterrows():

        print(
            f"  {row['dataset_name']} "
            f"-> target: {row['target_column']}"
        )

    return config


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(
    dataset_name,
    target_column
):

    print("\n")
    print("=" * 70)
    print(
        f"DATASET: {dataset_name}"
    )
    print("=" * 70)

    processed_file = os.path.join(
        PROCESSED_DIR,
        f"{dataset_name}_processed.csv"
    )

    print("\nLooking for:")
    print(processed_file)

    # ========================================================
    # CHECK FILE
    # ========================================================

    if not os.path.exists(
        processed_file
    ):

        print(
            "\nERROR: Processed file not found."
        )

        print(
            "Expected:"
        )

        print(
            processed_file
        )

        return None

    print(
        "Dataset found."
    )

    # ========================================================
    # READ DATA
    # ========================================================

    try:

        df = pd.read_csv(
            processed_file
        )

    except Exception as e:

        print(
            "\nERROR while reading dataset:"
        )

        print(e)

        return None

    print(
        "\nOriginal processed shape:",
        df.shape
    )

    print(
        "Columns:"
    )

    print(
        df.columns.tolist()
    )

    # ========================================================
    # TARGET CHECK
    # ========================================================

    if target_column not in df.columns:

        print(
            f"\nERROR: Target column "
            f"'{target_column}' not found."
        )

        print(
            "\nAvailable columns:"
        )

        print(
            df.columns.tolist()
        )

        return None

    # ========================================================
    # REMOVE MISSING TARGET
    # ========================================================

    before_rows = len(df)

    df = df.dropna(
        subset=[target_column]
    )

    removed_rows = (
        before_rows
        - len(df)
    )

    if removed_rows > 0:

        print(
            f"\nRemoved {removed_rows} "
            "rows with missing target."
        )

    # ========================================================
    # X / Y
    # ========================================================

    X = df.drop(
        columns=[target_column]
    )

    y = df[
        target_column
    ]

    # ========================================================
    # REMOVE EMPTY COLUMNS
    # ========================================================

    empty_columns = X.columns[
        X.isna().all()
    ]

    if len(empty_columns) > 0:

        print(
            "\nRemoving empty columns:"
        )

        print(
            list(empty_columns)
        )

        X = X.drop(
            columns=empty_columns
        )

    # ========================================================
    # CONVERT FEATURES TO NUMERIC
    # ========================================================

    for column in X.columns:

        if not pd.api.types.is_numeric_dtype(
            X[column]
        ):

            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )

    # ========================================================
    # REPLACE INFINITE VALUES
    # ========================================================

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ========================================================
    # FILL MISSING VALUES
    # ========================================================

    for column in X.columns:

        if X[column].isna().any():

            median_value = X[column].median()

            if pd.isna(median_value):

                median_value = 0

            X[column] = X[column].fillna(
                median_value
            )

    # Final safety

    X = X.fillna(0)

    # ========================================================
    # REMOVE CONSTANT FEATURES
    # ========================================================

    constant_columns = []

    for column in X.columns:

        if X[column].nunique(
            dropna=False
        ) <= 1:

            constant_columns.append(
                column
            )

    if constant_columns:

        print(
            "\nRemoving constant columns:"
        )

        print(
            constant_columns
        )

        X = X.drop(
            columns=constant_columns
        )

    # ========================================================
    # CHECK FEATURES
    # ========================================================

    if X.shape[1] == 0:

        print(
            "\nERROR: No usable features."
        )

        return None

    # ========================================================
    # CHECK TARGET
    # ========================================================

    if y.nunique() < 2:

        print(
            "\nERROR: Target contains "
            "fewer than 2 classes."
        )

        return None

    # ========================================================
    # FINAL INFORMATION
    # ========================================================

    print(
        "\nFinal samples:",
        len(X)
    )

    print(
        "Final features:",
        X.shape[1]
    )

    print(
        "Classes:",
        y.nunique()
    )

    print(
        "Class distribution:"
    )

    print(
        y.value_counts().to_dict()
    )

    return X, y


# ============================================================
# PREPARE TRAIN / TEST DATA
# ============================================================

def prepare_data(
    X,
    y,
    algorithm_name
):

    # ========================================================
    # LARGE DATASET HANDLING
    # ========================================================

    if (
        len(X) > MAX_SAMPLES_EXPENSIVE
        and algorithm_name in [
            "SVM",
            "KNN"
        ]
    ):

        print(
            "\nLarge dataset detected."
        )

        print(
            f"Using {MAX_SAMPLES_EXPENSIVE} "
            f"samples for {algorithm_name}."
        )

        try:

            X, _, y, _ = train_test_split(
                X,
                y,
                train_size=MAX_SAMPLES_EXPENSIVE,
                stratify=y,
                random_state=42
            )

        except ValueError:

            X, _, y, _ = train_test_split(
                X,
                y,
                train_size=MAX_SAMPLES_EXPENSIVE,
                random_state=42
            )

    # ========================================================
    # TRAIN TEST SPLIT
    # ========================================================

    try:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
        )

    except ValueError:

        print(
            "\nStratified split failed."
        )

        print(
            "Using normal random split."
        )

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )
        )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# BENCHMARK ONE ALGORITHM
# ============================================================

def benchmark_algorithm(
    algorithm_name,
    X_train,
    X_test,
    y_train,
    y_test
):

    print(
        f"\nRunning: {algorithm_name}"
    )

    # Create a NEW model every time.
    # This prevents a previous dataset from
    # affecting the next dataset.

    algorithms = create_algorithms()

    algorithm = algorithms[
        algorithm_name
    ]

    start_time = time.time()

    try:

        # ====================================================
        # TRAIN
        # ====================================================

        algorithm.fit(
            X_train,
            y_train
        )

        # ====================================================
        # PREDICT
        # ====================================================

        predictions = algorithm.predict(
            X_test
        )

        # ====================================================
        # METRICS
        # ====================================================

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted"
        )

        elapsed_time = (
            time.time()
            - start_time
        )

        print(
            f"Accuracy: {accuracy:.4f}"
        )

        print(
            f"F1 Score: {f1:.4f}"
        )

        print(
            f"Time: {elapsed_time:.2f} seconds"
        )

        return {
            "accuracy": accuracy,
            "f1": f1,
            "time": elapsed_time
        }

    except Exception as e:

        elapsed_time = (
            time.time()
            - start_time
        )

        print(
            f"\nERROR while running "
            f"{algorithm_name}:"
        )

        print(
            str(e)
        )

        print(
            f"Time before failure: "
            f"{elapsed_time:.2f} seconds"
        )

        return None


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results
):

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    return results_df


# ============================================================
# BEST ALGORITHM
# ============================================================

def create_best_algorithm_summary(
    results_df
):

    print("\n")
    print("=" * 70)
    print(
        "BEST ALGORITHM FOR EACH DATASET"
    )
    print("=" * 70)

    best_results = []

    for dataset_name in sorted(
        results_df[
            "dataset_name"
        ].unique()
    ):

        dataset_results = (
            results_df[
                results_df[
                    "dataset_name"
                ] == dataset_name
            ]
        )

        # Highest F1 first.
        # Accuracy is tie-breaker.

        best_row = (
            dataset_results
            .sort_values(
                by=[
                    "f1_mean",
                    "accuracy_mean"
                ],
                ascending=False
            )
            .iloc[0]
        )

        print(
            f"\n{dataset_name}"
        )

        print(
            "Best algorithm:",
            best_row["algorithm"]
        )

        print(
            f"Accuracy: "
            f"{best_row['accuracy_mean']:.4f}"
        )

        print(
            f"F1 Score: "
            f"{best_row['f1_mean']:.4f}"
        )

        best_results.append({

            "dataset_name":
                dataset_name,

            "best_algorithm":
                best_row[
                    "algorithm"
                ],

            "accuracy":
                best_row[
                    "accuracy_mean"
                ],

            "f1_score":
                best_row[
                    "f1_mean"
                ],

            "training_time_seconds":
                best_row[
                    "training_time_seconds"
                ]
        })

    best_df = pd.DataFrame(
        best_results
    )

    best_df.to_csv(
        BEST_OUTPUT_FILE,
        index=False
    )

    print("\n")
    print(
        "Best algorithm summary saved to:"
    )

    print(
        BEST_OUTPUT_FILE
    )

    return best_df


# ============================================================
# FINAL VALIDATION
# ============================================================

def final_check(
    config,
    results_df
):

    print("\n")
    print("=" * 70)
    print(
        "FINAL BENCHMARK CHECK"
    )
    print("=" * 70)

    expected_datasets = (
        config[
            "dataset_name"
        ].nunique()
    )

    actual_datasets = (
        results_df[
            "dataset_name"
        ].nunique()
    )

    number_of_algorithms = 5

    expected_rows = (
        expected_datasets
        * number_of_algorithms
    )

    actual_rows = len(
        results_df
    )

    print(
        f"\nExpected datasets: "
        f"{expected_datasets}"
    )

    print(
        f"Actual datasets: "
        f"{actual_datasets}"
    )

    print(
        f"Expected benchmark rows: "
        f"{expected_rows}"
    )

    print(
        f"Actual benchmark rows: "
        f"{actual_rows}"
    )

    # ========================================================
    # MISSING DATASETS
    # ========================================================

    configured = set(
        config[
            "dataset_name"
        ]
    )

    benchmarked = set(
        results_df[
            "dataset_name"
        ]
    )

    missing_datasets = (
        configured
        - benchmarked
    )

    if not missing_datasets:

        print(
            "\nSUCCESS: "
            "All datasets were benchmarked."
        )

    else:

        print(
            "\nWARNING: "
            "Some datasets were not benchmarked."
        )

        print(
            "Missing datasets:"
        )

        for dataset in sorted(
            missing_datasets
        ):

            print(
                " -",
                dataset
            )

    # ========================================================
    # INCOMPLETE ALGORITHMS
    # ========================================================

    print("\n")
    print(
        "Algorithm coverage:"
    )

    coverage = (
        results_df
        .groupby("algorithm")
        ["dataset_name"]
        .nunique()
    )

    print(
        coverage.to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(
        "AUTOMATIC ML ALGORITHM BENCHMARKING"
    )
    print("=" * 70)

    print(
        "\nProject root:"
    )

    print(
        PROJECT_ROOT
    )

    print(
        "\nProcessed data directory:"
    )

    print(
        PROCESSED_DIR
    )

    # ========================================================
    # CREATE RESULTS DIRECTORY
    # ========================================================

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # ========================================================
    # LOAD CONFIGURATION
    # ========================================================

    config = load_dataset_config()

    if config is None:

        return

    # ========================================================
    # LOAD ALGORITHMS
    # ========================================================

    algorithms = create_algorithms()

    print("\n")
    print(
        "Algorithms:"
    )

    for algorithm in algorithms:

        print(
            " -",
            algorithm
        )

    # ========================================================
    # RESULTS
    # ========================================================

    results = []

    # ========================================================
    # PROCESS EVERY DATASET
    # ========================================================

    for _, row in config.iterrows():

        dataset_name = (
            row["dataset_name"]
        )

        target_column = (
            row["target_column"]
        )

        # ====================================================
        # LOAD
        # ====================================================

        data = load_dataset(
            dataset_name,
            target_column
        )

        if data is None:

            print(
                f"\nSkipping dataset: "
                f"{dataset_name}"
            )

            continue

        X, y = data

        # ====================================================
        # RUN ALL ALGORITHMS
        # ====================================================

        for algorithm_name in algorithms:

            try:

                (
                    X_train,
                    X_test,
                    y_train,
                    y_test
                ) = prepare_data(
                    X,
                    y,
                    algorithm_name
                )

                metrics = (
                    benchmark_algorithm(
                        algorithm_name,
                        X_train,
                        X_test,
                        y_train,
                        y_test
                    )
                )

                if metrics is None:

                    continue

                results.append({

                    "dataset_name":
                        dataset_name,

                    "algorithm":
                        algorithm_name,

                    "accuracy_mean":
                        metrics[
                            "accuracy"
                        ],

                    "f1_mean":
                        metrics[
                            "f1"
                        ],

                    "training_time_seconds":
                        metrics[
                            "time"
                        ],

                    "n_samples":
                        len(X),

                    "n_features":
                        X.shape[1],

                    "n_classes":
                        y.nunique()
                })

            except Exception as e:

                print(
                    f"\nERROR processing "
                    f"{algorithm_name} "
                    f"for {dataset_name}:"
                )

                print(
                    str(e)
                )

    # ========================================================
    # CHECK RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "BENCHMARKING COMPLETED"
    )
    print("=" * 70)

    if not results:

        print(
            "\nERROR: "
            "No benchmark results generated."
        )

        return

    # ========================================================
    # DATAFRAME
    # ========================================================

    results_df = save_results(
        results
    )

    # ========================================================
    # BASIC SUMMARY
    # ========================================================

    print(
        "\nTotal benchmark results:",
        len(results_df)
    )

    print(
        "Datasets:",
        results_df[
            "dataset_name"
        ].nunique()
    )

    print(
        "Algorithms:",
        results_df[
            "algorithm"
        ].nunique()
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_FILE
    )

    # ========================================================
    # RESULTS TABLE
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "BENCHMARK RESULTS"
    )
    print("=" * 70)

    print(
        results_df[
            [
                "dataset_name",
                "algorithm",
                "accuracy_mean",
                "f1_mean",
                "training_time_seconds"
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # BEST ALGORITHMS
    # ========================================================

    create_best_algorithm_summary(
        results_df
    )

    # ========================================================
    # FINAL CHECK
    # ========================================================

    final_check(
        config,
        results_df
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()