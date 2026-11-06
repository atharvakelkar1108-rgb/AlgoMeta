import os
import pandas as pd


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
CONFIG_FILE = "data/dataset_config.csv"


# ============================================================
# CREATE PROCESSED FOLDER
# ============================================================

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)


# ============================================================
# START MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("ML ALGORITHM RECOMMENDATION SYSTEM")
print("DATASET PREPROCESSING")
print("=" * 70)

print("\nProgram started successfully.")

print("\nCurrent working directory:")
print(os.getcwd())


# ============================================================
# CHECK REQUIRED FILES/FOLDERS
# ============================================================

print("\nChecking project files...")


if not os.path.exists(RAW_DIR):

    print(
        f"ERROR: Raw data folder not found: {RAW_DIR}"
    )

    exit()


if not os.path.exists(CONFIG_FILE):

    print(
        f"ERROR: Configuration file not found: "
        f"{CONFIG_FILE}"
    )

    exit()


try:
    from src.auto_target import sync_dataset_config
except ImportError:
    from auto_target import sync_dataset_config


# Automatically discover target columns for any unconfigured raw datasets
print("\nSynchronizing raw datasets and target configurations...")
sync_dataset_config(raw_dir=RAW_DIR, config_file=CONFIG_FILE)


# ============================================================
# LOAD CONFIGURATION
# ============================================================

print("\nLoading dataset configuration...")


try:

    config = pd.read_csv(
        CONFIG_FILE
    )

except Exception as e:

    print(
        "ERROR while reading "
        "dataset_config.csv"
    )

    print(e)

    exit()


print("\nDataset configuration:")

print(config)


# ============================================================
# CHECK CONFIGURATION COLUMNS
# ============================================================

required_columns = [
    "dataset_name",
    "target_column"
]


for column in required_columns:

    if column not in config.columns:

        print(
            f"\nERROR: Missing column "
            f"'{column}' in dataset_config.csv"
        )

        exit()


# ============================================================
# PREPROCESS ONE DATASET
# ============================================================

def preprocess_dataset(
    dataset_name,
    target_column
):

    print("\n")
    print("=" * 70)
    print(
        f"PROCESSING DATASET: "
        f"{dataset_name}"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Build input path
    # --------------------------------------------------------

    input_file = os.path.join(
        RAW_DIR,
        dataset_name + ".csv"
    )


    print(
        f"\nLooking for dataset:"
    )

    print(
        input_file
    )


    # --------------------------------------------------------
    # Check file exists
    # --------------------------------------------------------

    if not os.path.exists(input_file):

        print(
            f"ERROR: Dataset not found:"
        )

        print(
            input_file
        )

        return False


    print(
        "Dataset found."
    )


    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    try:

        # Bank Marketing dataset uses ;
        if dataset_name == "bank-additional-full":

            df = pd.read_csv(
                input_file,
                sep=";"
            )

        else:

            df = pd.read_csv(
                input_file
            )

    except Exception as e:

        print(
            "ERROR while loading dataset:"
        )

        print(e)

        return False


    # --------------------------------------------------------
    # Display original information
    # --------------------------------------------------------

    print(
        f"\nOriginal shape: "
        f"{df.shape}"
    )


    print(
        "\nOriginal columns:"
    )

    print(
        list(df.columns)
    )


    # --------------------------------------------------------
    # Check target column
    # --------------------------------------------------------

    if target_column not in df.columns:

        print(
            f"\nERROR: Target column "
            f"'{target_column}' not found."
        )

        print(
            "\nAvailable columns:"
        )

        print(
            list(df.columns)
        )

        return False


    print(
        f"\nTarget column: "
        f"{target_column}"
    )


    # ========================================================
    # REMOVE UNNECESSARY ID COLUMNS
    # ========================================================

    columns_to_remove = []


    for column in df.columns:

        if column == target_column:

            continue


        column_lower = (
            str(column)
            .strip()
            .lower()
        )


        # Remove common ID/index columns

        if (
            column_lower == "id"
            or column_lower.endswith("_id")
            or column_lower.startswith("unnamed")
            or column_lower == "customer_id"
        ):

            columns_to_remove.append(
                column
            )


    if len(columns_to_remove) > 0:

        print(
            "\nRemoving ID/unnecessary columns:"
        )

        print(
            columns_to_remove
        )


        df = df.drop(
            columns=columns_to_remove
        )

    else:

        print(
            "\nNo ID columns found."
        )


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    duplicate_count = (
        df.duplicated()
        .sum()
    )


    print(
        f"\nDuplicate rows: "
        f"{duplicate_count}"
    )


    if duplicate_count > 0:

        df = df.drop_duplicates()

        print(
            f"Removed "
            f"{duplicate_count} "
            f"duplicate rows."
        )


    # ========================================================
    # SEPARATE FEATURES AND TARGET
    # ========================================================

    X = df.drop(
        columns=[target_column]
    )


    y = df[target_column]


    print(
        f"\nNumber of samples: "
        f"{len(df)}"
    )


    print(
        f"Number of features: "
        f"{X.shape[1]}"
    )


    # ========================================================
    # IDENTIFY NUMERICAL FEATURES
    # ========================================================

    numerical_columns = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )


    # ========================================================
    # IDENTIFY CATEGORICAL FEATURES
    # ========================================================

    categorical_columns = (
        X.select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        )
        .columns
        .tolist()
    )


    print(
        "\nNumerical columns:"
    )

    print(
        numerical_columns
    )


    print(
        "\nCategorical columns:"
    )

    print(
        categorical_columns
    )


    # ========================================================
    # MISSING VALUES BEFORE PROCESSING
    # ========================================================

    missing_before = (
        X.isnull()
        .sum()
        .sum()
    )


    print(
        f"\nMissing feature values "
        f"before processing: "
        f"{missing_before}"
    )


    # ========================================================
    # NUMERICAL MISSING VALUES
    # ========================================================

    for column in numerical_columns:

        missing_count = (
            X[column]
            .isnull()
            .sum()
        )


        if missing_count > 0:

            median_value = (
                X[column]
                .median()
            )


            X[column] = (
                X[column]
                .fillna(
                    median_value
                )
            )


            print(
                f"Filled {missing_count} "
                f"missing values in "
                f"'{column}' with median "
                f"{median_value}"
            )


    # ========================================================
    # CATEGORICAL MISSING VALUES
    # ========================================================

    for column in categorical_columns:

        missing_count = (
            X[column]
            .isnull()
            .sum()
        )


        if missing_count > 0:

            mode_values = (
                X[column]
                .mode()
            )


            if len(mode_values) > 0:

                mode_value = (
                    mode_values.iloc[0]
                )


                X[column] = (
                    X[column]
                    .fillna(
                        mode_value
                    )
                )


                print(
                    f"Filled {missing_count} "
                    f"missing values in "
                    f"'{column}' with mode "
                    f"'{mode_value}'"
                )

            else:

                X[column] = (
                    X[column]
                    .fillna(
                        "Unknown"
                    )
                )


    # ========================================================
    # ENCODE CATEGORICAL FEATURES
    # ========================================================

    if len(categorical_columns) > 0:

        print(
            "\nEncoding categorical features..."
        )

        low_card_cat = []
        for col in categorical_columns:
            if X[col].nunique() > 50:
                print(
                    f"High-cardinality feature '{col}' "
                    f"({X[col].nunique()} categories) -> Label Encoding"
                )
                X[col] = pd.factorize(X[col])[0]
            else:
                low_card_cat.append(col)

        if len(low_card_cat) > 0:
            X = pd.get_dummies(
                X,
                columns=low_card_cat,
                drop_first=False,
                dtype=int
            )

        print(
            "Categorical encoding completed."
        )

    else:

        print(
            "\nNo categorical features "
            "to encode."
        )


    # ========================================================
    # ENCODE TARGET
    # ========================================================

    print(
        "\nChecking target..."
    )


    if (
        y.dtype == "object"
        or str(y.dtype) == "category"
        or y.dtype == "bool"
    ):

        print(
            "Target is categorical."
        )


        # Remove missing target values
        y = y.dropna()


        unique_values = (
            y.unique()
        )


        target_mapping = {
            value: index
            for index, value
            in enumerate(
                unique_values
            )
        }


        print(
            "\nTarget mapping:"
        )


        for value, number in (
            target_mapping.items()
        ):

            print(
                f"  {value} -> {number}"
            )


        y = y.map(
            target_mapping
        )

    else:

        print(
            "Target is numerical."
        )


    # ========================================================
    # REMOVE ROWS WITH MISSING TARGET
    # ========================================================

    valid_rows = y.notna()


    X = X.loc[
        valid_rows
    ]


    y = y.loc[
        valid_rows
    ]


    # ========================================================
    # COMBINE X AND Y
    # ========================================================

    processed_df = X.copy()


    processed_df[target_column] = y


    # ========================================================
    # FINAL MISSING VALUE CHECK
    # ========================================================

    missing_after = (
        processed_df
        .isnull()
        .sum()
        .sum()
    )


    print(
        f"\nMissing values "
        f"after processing: "
        f"{missing_after}"
    )


    # ========================================================
    # SAVE PROCESSED DATASET
    # ========================================================

    output_file = os.path.join(
        PROCESSED_DIR,
        dataset_name +
        "_processed.csv"
    )


    try:

        processed_df.to_csv(
            output_file,
            index=False
        )

    except Exception as e:

        print(
            "\nERROR while saving:"
        )

        print(e)

        return False


    # ========================================================
    # SUCCESS MESSAGE
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "PROCESSING COMPLETED SUCCESSFULLY"
    )

    print(
        "-" * 70
    )


    print(
        f"Dataset: "
        f"{dataset_name}"
    )


    print(
        f"Final shape: "
        f"{processed_df.shape}"
    )


    print(
        f"Saved to:"
    )


    print(
        output_file
    )


    return True


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print(
        "\nStarting preprocessing..."
    )


    successful = 0
    failed = 0


    # --------------------------------------------------------
    # Process every dataset
    # --------------------------------------------------------

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


        result = preprocess_dataset(
            dataset_name,
            target_column
        )


        if result:

            successful += 1

        else:

            failed += 1


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n\n"
        + "=" * 70
    )


    print(
        "PREPROCESSING SUMMARY"
    )


    print(
        "=" * 70
    )


    print(
        f"Successful datasets: "
        f"{successful}"
    )


    print(
        f"Failed datasets: "
        f"{failed}"
    )


    print(
        "\nProcessed datasets are located at:"
    )


    print(
        PROCESSED_DIR
    )


    print(
        "\nPreprocessing finished."
    )


    print(
        "=" * 70
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()