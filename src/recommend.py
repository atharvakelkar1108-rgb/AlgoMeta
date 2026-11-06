import os
import sys
import joblib
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

MODEL_PATH = os.path.join(
    RESULTS_DIR,
    "meta_model.joblib"
)

META_FEATURES_PATH = os.path.join(
    RESULTS_DIR,
    "meta_features.csv"
)

META_DATASET_PATH = os.path.join(
    RESULTS_DIR,
    "meta_dataset.csv"
)


# ============================================================
# ALGORITHMS
# ============================================================

ALGORITHMS = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "KNN",
    "SVM"
]


# ============================================================
# START
# ============================================================

print("\n" + "=" * 70)
print("ML ALGORITHM RECOMMENDATION SYSTEM")
print("=" * 70)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("\nERROR: Trained meta-model not found.")

    print("\nExpected:")
    print(MODEL_PATH)

    print(
        "\nRun first:"
    )

    print(
        "python src/train_meta_model.py"
    )

    raise SystemExit


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained meta-model...")

model = joblib.load(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# GET DATASET NAME
# ============================================================

if len(sys.argv) >= 2:

    dataset_name = " ".join(
        sys.argv[1:]
    )

else:

    dataset_name = input(
        "\nEnter dataset name: "
    ).strip()


if not dataset_name:

    print("\nERROR: Dataset name cannot be empty.")
    raise SystemExit


print(
    f"\nDataset selected: {dataset_name}"
)


# ============================================================
# LOAD META-FEATURES
# ============================================================

if not os.path.exists(
    META_FEATURES_PATH
):

    print(
        "\nERROR: meta_features.csv not found."
    )

    raise SystemExit


meta_features = pd.read_csv(
    META_FEATURES_PATH
)


# ============================================================
# CHECK DATASET
# ============================================================

if "dataset_name" not in meta_features.columns:

    print(
        "\nERROR: dataset_name column missing."
    )

    raise SystemExit


dataset_rows = meta_features[
    meta_features["dataset_name"]
    == dataset_name
].copy()


if len(dataset_rows) == 0:

    print("\nERROR: Dataset not found in meta_features.csv.")

    print("\nAvailable datasets:")

    for name in sorted(
        meta_features["dataset_name"].unique()
    ):

        print(
            "  -",
            name
        )

    print(
        "\nRun preprocessing + meta-feature extraction "
        "for the new dataset first."
    )

    raise SystemExit


# ============================================================
# GET ONE META-FEATURE ROW
# ============================================================

# There should normally be exactly one row
# for each dataset.

meta_row = dataset_rows.iloc[
    0
].copy()


# Remove dataset name

meta_row = meta_row.drop(
    labels=["dataset_name"]
)


# ============================================================
# CREATE FIVE ALGORITHM ROWS
# ============================================================

prediction_rows = []


for algorithm in ALGORITHMS:

    row = meta_row.copy()

    row["algorithm"] = algorithm

    prediction_rows.append(
        row
    )


prediction_df = pd.DataFrame(
    prediction_rows
)


# ============================================================
# REORDER
# ============================================================

meta_columns = [
    column
    for column in meta_features.columns
    if column != "dataset_name"
]


feature_columns = [
    "algorithm"
] + [
    column
    for column in meta_columns
]


prediction_df = prediction_df[
    feature_columns
]


# ============================================================
# PREDICT
# ============================================================

print(
    "\nPredicting algorithm performance..."
)

predictions = model.predict(
    prediction_df
)


# ============================================================
# CREATE RESULT
# ============================================================

recommendations = pd.DataFrame({

    "algorithm":
        ALGORITHMS,

    "predicted_f1":
        predictions
})


# ============================================================
# SORT
# ============================================================

recommendations = recommendations.sort_values(
    by="predicted_f1",
    ascending=False
).reset_index(
    drop=True
)


recommendations[
    "rank"
] = range(
    1,
    len(recommendations) + 1
)


recommendations = recommendations[
    [
        "rank",
        "algorithm",
        "predicted_f1"
    ]
]


# ============================================================
# TOP 3
# ============================================================

top3 = recommendations.head(
    3
)


# ============================================================
# DISPLAY ALL
# ============================================================

print("\n" + "=" * 70)
print("PREDICTED ALGORITHM PERFORMANCE")
print("=" * 70)

print(
    recommendations.to_string(
        index=False
    )
)


# ============================================================
# DISPLAY TOP 3
# ============================================================

print("\n" + "=" * 70)
print("TOP-3 ALGORITHM RECOMMENDATIONS")
print("=" * 70)


for _, row in top3.iterrows():

    print(
        f"\n{int(row['rank'])}. "
        f"{row['algorithm']}"
    )

    print(
        f"   Predicted F1: "
        f"{row['predicted_f1']:.4f}"
    )


# ============================================================
# BEST ALGORITHM
# ============================================================

best = top3.iloc[0]


print("\n" + "=" * 70)
print("BEST RECOMMENDED ALGORITHM")
print("=" * 70)

print(
    f"\n{best['algorithm']}"
)

print(
    f"Predicted F1: "
    f"{best['predicted_f1']:.4f}"
)


# ============================================================
# SAVE RECOMMENDATIONS
# ============================================================

output_file = os.path.join(
    RESULTS_DIR,
    f"{dataset_name}_recommendations.csv"
)

recommendations.to_csv(
    output_file,
    index=False
)


print("\nRecommendations saved to:")
print(output_file)


# ============================================================
# END
# ============================================================

print("\n" + "=" * 70)
print("RECOMMENDATION COMPLETED")
print("=" * 70)