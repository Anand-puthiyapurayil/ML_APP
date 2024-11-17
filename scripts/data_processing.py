import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import re
import numpy as np


def create_directories(encoder_dir, scaler_path, processed_data_path):
    print("\n[INFO] Creating necessary directories...")
    os.makedirs(encoder_dir, exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    print("[INFO] Directories created successfully.")


def drop_missing_values(df, feature_columns, target_column=None):
    print("\n[INFO] Dropping rows with missing values...")
    if target_column:
        df = df.dropna(subset=feature_columns + [target_column])
    else:
        df = df.dropna(subset=feature_columns)

    if df.empty:
        raise ValueError("[ERROR] Dataset is empty after dropping missing values.")
    
    print(f"[INFO] Dataset shape after dropping missing values: {df.shape}")
    return df


def handle_target_column(y, task_type, target_column, encoder_dir):
    print("\n[INFO] Handling target column...")

    # Preview the raw target column before any transformations
    print(f"[DEBUG] Raw target column before processing:\n{y.head()}")

    target_encoder = None
    if task_type == "classification" and target_column:
        print(f"[INFO] Encoding target column '{target_column}' for classification...")

        # Ensure all target values are strings
        y = y.astype(str).str.strip()  # Remove leading/trailing whitespaces

        # Handle special cases where the target has complex patterns
        y = y.apply(lambda val: re.sub(r"\s*\|\s*", " | ", val) if isinstance(val, str) else val)

        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        y = pd.Series(y_encoded, index=y.index, name=target_column)  # Maintain alignment with X
        joblib.dump(target_encoder, os.path.join(encoder_dir, "target_encoder.joblib"))
        print(f"[INFO] Target column '{target_column}' encoded successfully.")
    elif task_type in ["regression", "clustering"] and y is not None:
        print("[INFO] Ensuring target column is numeric...")

        # Clean non-numeric characters from the target column
        y = y.apply(lambda val: re.sub(r"[^\d.]", "", str(val)) if isinstance(val, str) else val)
        y = pd.to_numeric(y, errors="coerce").fillna(0)
        print("[INFO] Target column numeric conversion completed.")

    # Preview the target column after transformation
    print(f"[DEBUG] Target column after processing:\n{y.head()}")
    return y, target_encoder



def save_metadata(numeric_columns, categorical_columns, feature_columns, numeric_columns_path, categorical_columns_path, feature_columns_path):
    print("\n[INFO] Saving metadata...")
    joblib.dump(numeric_columns, numeric_columns_path)
    joblib.dump(categorical_columns, categorical_columns_path)
    joblib.dump(feature_columns, feature_columns_path)
    print("[INFO] Metadata saved successfully.")


def clean_numeric_column(value):
    """
    Clean numeric values by removing non-numeric characters.
    """
    if isinstance(value, str):
        value = re.sub(r"[^\d.]", "", value)
    try:
        return float(value)
    except ValueError:
        return np.nan


def preprocess_numeric_columns(X, numeric_columns):
    print("\n[INFO] Cleaning and scaling numeric columns...")
    for col in numeric_columns:
        X[col] = X[col].apply(clean_numeric_column).fillna(0)
    print("[DEBUG] Numeric columns after cleaning:")
    print(X[numeric_columns].head())
    return X


def encode_categorical_columns(X, categorical_columns, encoder_dir):
    print("\n[INFO] Encoding categorical columns...")
    encoders = {}
    for col in categorical_columns:
        encoder_path = os.path.join(encoder_dir, f"{col}_encoder.joblib")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        joblib.dump(le, encoder_path)
        print(f"[INFO] Categorical column '{col}' encoded successfully.")
    print("[DEBUG] Encoded categorical columns preview:")
    print(X[categorical_columns].head())
    return X, encoders


def scale_numeric_columns(X, numeric_columns, scaler_path):
    print("\n[INFO] Scaling numeric columns...")
    scaler = None
    if numeric_columns:
        scaler = StandardScaler()
        X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
        joblib.dump(scaler, scaler_path)
        print("[INFO] Numeric columns scaled successfully.")
        print("[DEBUG] Scaled numeric columns preview:")
        print(X[numeric_columns].head())
    return X, scaler


def preprocess_data(
    df: pd.DataFrame,
    feature_columns: list,
    target_column: str = None,
    task_type: str = "regression",  # "classification", "regression", or "clustering"
    encoder_dir: str = "models/encoders",
    scaler_path: str = "models/scaler/scaler.joblib",
    processed_data_path: str = "data/processed_data.csv",
    dtype_path: str = "models/feature_dtypes.joblib",
    feature_columns_path: str = "models/feature_columns.joblib",
    numeric_columns_path: str = "models/numeric_columns.joblib",
    categorical_columns_path: str = "models/categorical_columns.joblib",
):
    print("\n========== Starting Preprocessing ==========\n")
    create_directories(encoder_dir, scaler_path, processed_data_path)

    df = drop_missing_values(df, feature_columns, target_column)

    print("\n[INFO] Separating features and target...")
    X = df[feature_columns].copy()
    y = None  # Default to None for clustering tasks

    if task_type != "clustering":
        if target_column:
            y = df[target_column].copy()
            print(f"[DEBUG] Target column preview before processing:\n{y.head()}")
            y, target_encoder = handle_target_column(y, task_type, target_column, encoder_dir)
        else:
            raise ValueError("[ERROR] Target column is required for non-clustering tasks.")
    else:
        print("[INFO] Clustering task: No target column required.")
        target_encoder = None

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"[INFO] Numeric columns: {numeric_columns}")
    print(f"[INFO] Categorical columns: {categorical_columns}")

    save_metadata(numeric_columns, categorical_columns, feature_columns, numeric_columns_path, categorical_columns_path, feature_columns_path)

    X = preprocess_numeric_columns(X, numeric_columns)
    X, encoders = encode_categorical_columns(X, categorical_columns, encoder_dir)
    X, scaler = scale_numeric_columns(X, numeric_columns, scaler_path)

    preprocessed_data = X  # Default to X for clustering tasks
    if y is not None:
        preprocessed_data = pd.concat([X, y], axis=1)

    preprocessed_data.to_csv(processed_data_path, index=False)

    print("[INFO] Preprocessed data saved to:", processed_data_path)
    print("\n========== Preprocessing Completed ==========\n")
    return X, y, encoders, scaler, target_encoder, preprocessed_data



def preprocess_data_for_inference(
    input_df: pd.DataFrame,
    encoder_dir: str,
    scaler_path: str,
    task_type: str = "regression",  # "classification", "regression", or "clustering"
    feature_columns_path: str = "models/feature_columns.joblib",
    numeric_columns_path: str = "models/numeric_columns.joblib",
    categorical_columns_path: str = "models/categorical_columns.joblib",
):
    print("\n========== Starting Inference Preprocessing ==========\n")

    # Log raw input data
    print("[DEBUG] Raw inference data before preprocessing:")
    print(input_df.head())

    # Load feature columns and align input data
    if os.path.exists(feature_columns_path):
        feature_columns = joblib.load(feature_columns_path)
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    else:
        raise FileNotFoundError("[ERROR] Feature columns file not found.")

    # Load numeric and categorical columns
    numeric_columns = joblib.load(numeric_columns_path) if os.path.exists(numeric_columns_path) else []
    categorical_columns = joblib.load(categorical_columns_path) if os.path.exists(categorical_columns_path) else []

    # Process numeric columns
    print("[INFO] Processing numeric columns for inference...")
    for col in numeric_columns:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)
    print("[DEBUG] Numeric columns processed for inference:")
    print(input_df[numeric_columns].head())

    # Encode categorical columns
    print("[INFO] Encoding categorical columns for inference...")
    for col in categorical_columns:
        encoder_path = os.path.join(encoder_dir, f"{col}_encoder.joblib")
        if os.path.exists(encoder_path):
            encoder = joblib.load(encoder_path)
            input_df[col] = input_df[col].fillna("unknown").apply(lambda x: x if x in encoder.classes_ else "unknown")
            input_df[col] = encoder.transform(input_df[col])
        else:
            print(f"[WARNING] Encoder for column '{col}' not found. Skipping encoding.")
    print("[DEBUG] Categorical columns processed for inference:")
    print(input_df[categorical_columns].head())

    # Scale numeric columns
    if numeric_columns and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])
        print("[INFO] Numeric columns scaled for inference:")
        print(input_df[numeric_columns].head())

    # Log preprocessed data
    print("[DEBUG] Preprocessed inference data:")
    print(input_df.head())

    print("[INFO] Inference preprocessing completed successfully.")
    print("\n========== Inference Preprocessing Completed ==========\n")
    return input_df