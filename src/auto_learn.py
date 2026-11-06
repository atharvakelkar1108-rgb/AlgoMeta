import os
import pandas as pd
import numpy as np
import joblib
import subprocess
import sys

from src.auto_target import sync_dataset_config
from src.preprocessing import preprocess_dataset
from src.meta_features import extract_meta_features


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CONFIG_FILE = os.path.join(PROJECT_ROOT, "data", "dataset_config.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

META_FEATURES_FILE = os.path.join(RESULTS_DIR, "meta_features.csv")
ALGORITHM_PERFORMANCE_FILE = os.path.join(RESULTS_DIR, "algorithm_performance.csv")
META_DATASET_FILE = os.path.join(RESULTS_DIR, "meta_dataset.csv")
MODEL_FILE = os.path.join(RESULTS_DIR, "meta_model.joblib")


def save_and_update_knowledge(raw_df, dataset_name, target_column, actual_results_list):
    """
    Saves new dataset session results into raw storage, extracts meta-features,
    appends performance metrics, updates meta_dataset.csv, and retrains meta_model.joblib.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    clean_dataset_name = "".join([c if c.isalnum() or c in ("_", "-") else "_" for c in str(dataset_name).strip()])
    if not clean_dataset_name:
        clean_dataset_name = "user_uploaded_dataset"

    raw_file_path = os.path.join(RAW_DIR, f"{clean_dataset_name}.csv")
    
    # 1. Save raw dataset if not already present
    if not os.path.exists(raw_file_path):
        raw_df.to_csv(raw_file_path, index=False)

    # 2. Sync dataset config
    sync_dataset_config(raw_dir=RAW_DIR, config_file=CONFIG_FILE, verbose=False)

    # 3. Preprocess dataset
    preprocess_dataset(clean_dataset_name, target_column)

    # 4. Extract meta-features for new dataset
    meta_row = extract_meta_features(clean_dataset_name, target_column)

    # Append to meta_features.csv
    if meta_row is not None:
        new_mf_df = pd.DataFrame([meta_row])
        if os.path.exists(META_FEATURES_FILE):
            existing_mf = pd.read_csv(META_FEATURES_FILE)
            # Remove existing row for this dataset if present
            existing_mf = existing_mf[existing_mf["dataset_name"] != clean_dataset_name]
            updated_mf = pd.concat([existing_mf, new_mf_df], ignore_index=True)
        else:
            updated_mf = new_mf_df
        updated_mf.to_csv(META_FEATURES_FILE, index=False)

    # 5. Append results to algorithm_performance.csv
    perf_rows = []
    for item in actual_results_list:
        perf_rows.append({
            "dataset_name": clean_dataset_name,
            "algorithm": item.get("algorithm"),
            "accuracy": item.get("accuracy"),
            "f1_mean": item.get("f1_score"),
            "training_time": item.get("training_time")
        })

    new_perf_df = pd.DataFrame(perf_rows)
    if os.path.exists(ALGORITHM_PERFORMANCE_FILE):
        existing_perf = pd.read_csv(ALGORITHM_PERFORMANCE_FILE)
        existing_perf = existing_perf[existing_perf["dataset_name"] != clean_dataset_name]
        updated_perf = pd.concat([existing_perf, new_perf_df], ignore_index=True)
    else:
        updated_perf = new_perf_df
    updated_perf.to_csv(ALGORITHM_PERFORMANCE_FILE, index=False)

    # 6. Re-create meta_dataset.csv
    try:
        mf = pd.read_csv(META_FEATURES_FILE)
        ap = pd.read_csv(ALGORITHM_PERFORMANCE_FILE)
        meta_ds = pd.merge(mf, ap, on="dataset_name", how="inner")
        meta_ds.to_csv(META_DATASET_FILE, index=False)
    except Exception as e:
        print(f"[AUTO-LEARN] Warning creating meta_dataset: {e}")

    # 7. Retrain meta-model asynchronously or synchronously
    try:
        script_path = os.path.join(PROJECT_ROOT, "src", "train_meta_model.py")
        subprocess.run([sys.executable, script_path], check=True, cwd=PROJECT_ROOT)
        print(f"[AUTO-LEARN] Successfully retrained meta-model with dataset '{clean_dataset_name}'!")
        return True
    except Exception as e:
        print(f"[AUTO-LEARN] Warning retraining meta-model: {e}")
        return False
