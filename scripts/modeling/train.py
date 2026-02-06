# scripts/modeling/train.py

"""
Full training pipeline for the XGBoost churn prediction model.

Orchestrates data loading, cleaning, preprocessing, model training,
evaluation, and artifact serialization.

"""

import json

import joblib
import pandas as pd
from xgboost import XGBClassifier

from scripts.config import (
    ARTIFACTS_DIR,
    CATEGORICAL_COLS,
    CORRELATION_THRESHOLD,
    RAW_DATA_PATH,
    RANDOM_STATE,
    SMOTENC_CATEGORICAL_INDICES,
    TARGET_COL,
    TEST_SIZE,
    XGBOOST_PARAMS,
    Z_SCORE_THRESHOLD,
)
from scripts.data_prep.cleaning import (
    clean_nulls_and_duplicates,
    drop_highly_correlated_features,
    remove_outliers_zscore,
)
from scripts.data_prep.preprocessing import (
    apply_smotenc,
    build_encoder,
    encode_features,
    encode_target,
    load_raw_data,
    prepare_features,
    scale_features,
    split_data,
)
from scripts.modeling.evaluate import compute_classification_metrics


def train_and_save_model():
    """
    Run the full training pipeline and save artifacts to disk.
    """

    # Load the raw data
    print("Loading raw data...")
    df = load_raw_data(RAW_DATA_PATH)
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # Clean the nulls and duplicates
    print("\nCleaning data...")
    df = clean_nulls_and_duplicates(df)

    # Prepare features (drop Phone_Number, convert Area_Code)
    print("\nPreparing features...")
    df = prepare_features(df)

    # Drop highly correlated features 
    print("\nDropping highly correlated features...")
    df, dropped_features = drop_highly_correlated_features(df, threshold=CORRELATION_THRESHOLD)

    # Remove outliers 
    print("\nRemoving outliers...")
    df = remove_outliers_zscore(df, z_threshold=Z_SCORE_THRESHOLD)
    print(f"  Shape after outlier removal: {df.shape}")

    # Separate target before encoding so encoder doesn't expect Churn at prediction time
    print("\nSeparating target column...")
    y_raw = df[TARGET_COL]
    X_raw = df.drop(columns=TARGET_COL)

    # One-hot encode categorical features
    print("Encoding categorical features...")
    encoder = build_encoder(CATEGORICAL_COLS)
    X_encoded, encoder = encode_features(X_raw, encoder, CATEGORICAL_COLS, fit=True)
    feature_names = list(X_encoded.columns)

    # Label encode the target
    print("Encoding target variable...")
    y_encoded, label_encoder = encode_target(y_raw)

    # Split the data into train/test
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = split_data(X_encoded, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # Scale features
    print("Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Apply SMOTENC to balance classes 
    print("Applying SMOTENC resampling...")
    X_train_resampled, y_train_resampled = apply_smotenc(
        X_train_scaled, y_train,
        categorical_features=SMOTENC_CATEGORICAL_INDICES,
        random_state=RANDOM_STATE,
    )
    print(f"  Resampled train set: {X_train_resampled.shape[0]} samples")
    print(f"  Class distribution: {pd.Series(y_train_resampled).value_counts().to_dict()}")

    # Train XGBoost with tuned hyperparameters 
    print("\nTraining XGBoost model...")
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train_resampled, y_train_resampled)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = compute_classification_metrics(y_test, y_pred, y_proba=y_proba)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # Save model training artifacts
    print("\nSaving artifacts...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    joblib.dump(encoder, ARTIFACTS_DIR / "encoder.joblib")
    joblib.dump(label_encoder, ARTIFACTS_DIR / "label_encoder.joblib")

    with open(ARTIFACTS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    with open(ARTIFACTS_DIR / "dropped_features.json", "w") as f:
        json.dump(dropped_features, f, indent=2)

    print(f"  Artifacts saved to {ARTIFACTS_DIR}/")
    print("\nTraining complete.")

    return model, metrics


# CLI entrypoint
def main():
    """CLI entrypoint."""
    train_and_save_model()


if __name__ == "__main__":
    main()
