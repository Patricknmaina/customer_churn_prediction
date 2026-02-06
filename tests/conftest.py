# tests/conftest.py

"""
Shared test fixtures for the customer churn prediction test suite.
Provides small synthetic DataFrames and mock artifacts for testing.
"""

import json

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Small synthetic DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_churn_df():
    """A small DataFrame mimicking the raw CSV structure (lowercase columns)."""
    return pd.DataFrame({
        "state":                  ["KS", "OH", "NJ", "OH", "OK"],
        "account length":         [128, 107, 137, 84, 75],
        "area code":              [415, 415, 415, 408, 510],
        "phone number":           ["382-4657", "371-7191", "358-1921", "375-9999", "330-6626"],
        "international plan":     ["no", "no", "no", "yes", "yes"],
        "voice mail plan":        ["yes", "yes", "no", "no", "no"],
        "number vmail messages":  [25, 26, 0, 0, 0],
        "total day minutes":      [265.1, 161.6, 243.4, 299.4, 166.7],
        "total day calls":        [110, 123, 114, 71, 113],
        "total day charge":       [45.07, 27.47, 41.38, 50.90, 28.34],
        "total eve minutes":      [197.4, 195.5, 121.2, 61.9, 148.3],
        "total eve calls":        [99, 103, 110, 88, 122],
        "total eve charge":       [16.78, 16.62, 10.30, 5.26, 12.61],
        "total night minutes":    [244.7, 254.4, 162.6, 196.9, 186.9],
        "total night calls":      [91, 103, 104, 89, 121],
        "total night charge":     [11.01, 11.45, 7.32, 8.86, 8.41],
        "total intl minutes":     [10.0, 13.7, 12.2, 6.6, 10.1],
        "total intl calls":       [3, 3, 5, 7, 3],
        "total intl charge":      [2.70, 3.70, 3.29, 1.78, 2.73],
        "customer service calls": [1, 1, 0, 2, 3],
        "churn":                  [False, False, False, True, True],
    })


@pytest.fixture
def cleaned_churn_df(raw_churn_df):
    """DataFrame after clean_nulls_and_duplicates (Title_Case columns)."""
    from scripts.data_prep.cleaning import clean_nulls_and_duplicates
    return clean_nulls_and_duplicates(raw_churn_df.copy())


@pytest.fixture
def prepared_df(cleaned_churn_df):
    """DataFrame after prepare_features (Phone_Number dropped, Area_Code as object)."""
    from scripts.data_prep.preprocessing import prepare_features
    return prepare_features(cleaned_churn_df.copy())


@pytest.fixture
def numeric_df():
    """
    Numeric DataFrame for correlation / outlier tests.
    A and B are perfectly correlated (r=1.0).
    C and D are uncorrelated with A/B and each other.
    """
    rng = np.random.RandomState(0)
    n = 30
    a = np.arange(1, n + 1, dtype=float)
    return pd.DataFrame({
        "A": a,
        "B": a * 2,                           # r(A,B) = 1.0
        "C": rng.normal(50, 10, n),           # uncorrelated
        "D": rng.uniform(0, 100, n),          # uncorrelated
    })


# ---------------------------------------------------------------------------
# Valid customer input dict (Title_Case keys for ChurnPredictor)
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_customer_input():
    """A dict matching the Title_Case keys that ChurnPredictor.preprocess_input expects."""
    return {
        "State": "OH",
        "Account_Length": 100,
        "Area_Code": 415,
        "International_Plan": "no",
        "Voice_Mail_Plan": "yes",
        "Number_Vmail_Messages": 25,
        "Total_Day_Minutes": 0.0,
        "Total_Day_Calls": 110,
        "Total_Day_Charge": 45.07,
        "Total_Eve_Minutes": 0.0,
        "Total_Eve_Calls": 99,
        "Total_Eve_Charge": 16.78,
        "Total_Night_Minutes": 0.0,
        "Total_Night_Calls": 91,
        "Total_Night_Charge": 11.01,
        "Total_Intl_Minutes": 0.0,
        "Total_Intl_Calls": 3,
        "Total_Intl_Charge": 2.70,
        "Customer_Service_Calls": 1,
    }


# ---------------------------------------------------------------------------
# Mock artifacts for ChurnPredictor
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_artifacts(tmp_path):
    """
    Create a minimal set of mock artifacts in tmp_path that ChurnPredictor
    can load. Uses a DecisionTreeClassifier for speed.

    Returns the tmp_path directory.
    """
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier

    categorical_cols = ["State", "Area_Code", "International_Plan", "Voice_Mail_Plan"]
    remaining_cols = [
        "Account_Length", "Number_Vmail_Messages",
        "Total_Day_Calls", "Total_Day_Charge",
        "Total_Eve_Calls", "Total_Eve_Charge",
        "Total_Night_Calls", "Total_Night_Charge",
        "Total_Intl_Calls", "Total_Intl_Charge",
        "Customer_Service_Calls",
    ]

    # Build and fit encoder on tiny training data
    # Area_Code must be object-typed integers (matching preprocess_input behavior)
    tiny_train = pd.DataFrame({
        "State": ["OH", "KS"],
        "Area_Code": pd.array([415, 408], dtype=object),
        "International_Plan": ["no", "yes"],
        "Voice_Mail_Plan": ["yes", "no"],
        "Account_Length": [100, 128],
        "Number_Vmail_Messages": [25, 0],
        "Total_Day_Calls": [110, 100],
        "Total_Day_Charge": [45.07, 30.0],
        "Total_Eve_Calls": [99, 100],
        "Total_Eve_Charge": [16.78, 17.0],
        "Total_Night_Calls": [91, 100],
        "Total_Night_Charge": [11.01, 9.0],
        "Total_Intl_Calls": [3, 4],
        "Total_Intl_Charge": [2.70, 2.80],
        "Customer_Service_Calls": [1, 2],
    })

    encoder = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)
        ],
        remainder="passthrough",
    )
    encoded_array = encoder.fit_transform(tiny_train)

    # Derive feature names
    encoded_columns = encoder.named_transformers_["onehot"].get_feature_names_out(categorical_cols)
    feature_names = list(encoded_columns) + remaining_cols
    n_features = len(feature_names)

    # Fit scaler
    scaler = MinMaxScaler()
    scaler.fit(encoded_array)

    # Fit a trivial model
    model = DecisionTreeClassifier(random_state=42)
    X_dummy = np.zeros((4, n_features))
    y_dummy = np.array([0, 1, 0, 1])
    model.fit(X_dummy, y_dummy)

    # Label encoder
    le = LabelEncoder()
    le.fit(["False", "True"])

    # Dropped features (the correlated minutes columns)
    dropped_features = [
        "Total_Day_Minutes", "Total_Eve_Minutes",
        "Total_Night_Minutes", "Total_Intl_Minutes",
    ]

    # Serialize everything
    joblib.dump(model, tmp_path / "model.joblib")
    joblib.dump(scaler, tmp_path / "scaler.joblib")
    joblib.dump(encoder, tmp_path / "encoder.joblib")
    joblib.dump(le, tmp_path / "label_encoder.joblib")

    with open(tmp_path / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    with open(tmp_path / "dropped_features.json", "w") as f:
        json.dump(dropped_features, f)

    return tmp_path


@pytest.fixture
def predictor(mock_artifacts):
    """A ChurnPredictor instance backed by mock artifacts."""
    from scripts.serving.predict import ChurnPredictor
    return ChurnPredictor(artifacts_dir=mock_artifacts)
