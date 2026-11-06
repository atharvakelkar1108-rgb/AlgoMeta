import os
import re
import pandas as pd

TARGET_KEYWORDS = [
    "target",
    "label",
    "class",
    "outcome",
    "outcome variable",
    "diagnosis",
    "income",
    "heartdisease",
    "survival_status",
    "cover_type",
    "is_canceled",
    "channel",
    "type",
    "placement_status",
    "risk",
    "species",
    "deposit",
    "y",
    "quality",
    "status",
    "output",
    "response",
    "predict",
    "prediction",
    "price_range",
    "stress_level",
    "stress_index",
    "driver_experience_level",
]

ID_KEYWORDS = ["id", "index", "uuid", "unnamed"]


def is_id_column(col_name, df):
    """Check if a column is likely an ID/index column."""
    col_lower = str(col_name).strip().lower()
    
    # Check if name contains ID keywords
    for id_kw in ID_KEYWORDS:
        if id_kw == col_lower or col_lower.startswith(id_kw + "_") or col_lower.endswith("_" + id_kw):
            return True
            
    # Check if column is sequential index or has 100% unique row values with high row count
    if len(df) > 50 and df[col_name].nunique() == len(df):
        if pd.api.types.is_numeric_dtype(df[col_name]):
            return True
            
    return False


def detect_target_column(df):
    """
    Intelligently detect the target column of a DataFrame.
    Returns the column name string.
    """
    if df is None or df.empty or len(df.columns) == 0:
        return None

    columns = list(df.columns)
    
    # Single column DataFrame
    if len(columns) == 1:
        return columns[0]

    # 1. Check exact match with TARGET_KEYWORDS (searching right-to-left)
    for col in reversed(columns):
        col_clean = str(col).strip().lower()
        if is_id_column(col, df):
            continue
        for kw in TARGET_KEYWORDS:
            if col_clean == kw:
                return col

    # 2. Check whole-word boundary regex match with TARGET_KEYWORDS (searching right-to-left)
    for col in reversed(columns):
        col_clean = str(col).strip().lower()
        if is_id_column(col, df):
            continue
        for kw in TARGET_KEYWORDS:
            # Require exact word match for short keywords (<= 2 chars like 'y'), or regex word boundary match
            if len(kw) <= 2:
                if re.search(r'(?<![a-zA-Z0-9_])' + re.escape(kw) + r'(?![a-zA-Z0-9_])', col_clean):
                    return col
            else:
                if re.search(r'\b' + re.escape(kw) + r'\b', col_clean) or kw in col_clean:
                    return col

    # 3. Search right-to-left for categorical/object or discrete target columns
    n_rows = len(df)
    for col in reversed(columns):
        if is_id_column(col, df):
            continue
        
        # Non-numeric string/object column
        if not pd.api.types.is_numeric_dtype(df[col]):
            return col
            
        # Numeric with low cardinality (likely discrete classes/targets)
        unique_cnt = df[col].nunique()
        if 2 <= unique_cnt <= min(100, max(10, int(n_rows * 0.2))):
            return col

    # 4. Fallback to rightmost non-ID column
    for col in reversed(columns):
        if not is_id_column(col, df):
            return col

    # Ultimate fallback to the very last column
    return columns[-1]


def sync_dataset_config(raw_dir="data/raw", config_file="data/dataset_config.csv", verbose=True):
    """
    Scans raw_dir for CSV datasets and updates config_file with automatically detected target columns.
    Returns a list of newly added (dataset_name, target_column) tuples.
    """
    if not os.path.exists(raw_dir):
        if verbose:
            print(f"[AUTO-CONFIG] Raw directory not found: {raw_dir}")
        return []

    # Read existing config
    existing_config = {}
    if os.path.exists(config_file):
        try:
            cfg_df = pd.read_csv(config_file)
            if "dataset_name" in cfg_df.columns and "target_column" in cfg_df.columns:
                for _, row in cfg_df.iterrows():
                    ds_name = str(row["dataset_name"]).strip()
                    tgt_col = str(row["target_column"]).strip()
                    if ds_name and tgt_col and tgt_col.lower() != "nan":
                        existing_config[ds_name] = tgt_col
        except Exception as e:
            if verbose:
                print(f"[AUTO-CONFIG] Warning reading {config_file}: {e}")

    # Scan raw folder
    raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    newly_added = []
    updated_rows = dict(existing_config)

    for file_name in raw_files:
        dataset_name = os.path.splitext(file_name)[0].strip()
        
        # If dataset not configured or target is missing/invalid
        if dataset_name not in updated_rows:
            file_path = os.path.join(raw_dir, file_name)
            try:
                # Load first few rows to quickly detect target column
                sample_df = pd.read_csv(file_path, nrows=1000)
                detected_target = detect_target_column(sample_df)
                if detected_target:
                    updated_rows[dataset_name] = detected_target
                    newly_added.append((dataset_name, detected_target))
                    if verbose:
                        print(f"[AUTO-CONFIG] Automatically detected target for '{dataset_name}': '{detected_target}'")
            except Exception as e:
                if verbose:
                    print(f"[AUTO-CONFIG] Failed to read {file_name} for target detection: {e}")

    # If new datasets were found, update dataset_config.csv
    if newly_added or not os.path.exists(config_file):
        rows_to_save = [{"dataset_name": name, "target_column": target} for name, target in updated_rows.items()]
        config_df = pd.DataFrame(rows_to_save)
        
        # Ensure target directory exists
        config_dir = os.path.dirname(config_file)
        if config_dir:
            os.makedirs(config_dir, exist_ok=True)
            
        config_df.to_csv(config_file, index=False)
        if verbose:
            print(f"[AUTO-CONFIG] Updated '{config_file}' with {len(newly_added)} new dataset(s). Total configured: {len(config_df)}")

    return newly_added


if __name__ == "__main__":
    print("=" * 70)
    print("AUTOMATIC DATASET TARGET CONFIGURATION SYNC")
    print("=" * 70)
    new_added = sync_dataset_config()
    if not new_added:
        print("All raw datasets are already configured.")
