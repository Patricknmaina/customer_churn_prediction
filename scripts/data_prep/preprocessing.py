# scripts/data/preprocessing.py

"""
Functions for loading, encoding, scaling, splitting, and resampling data.

"""

# Imports
from pathlib import Path
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

# Load the raw data
def load_raw_data(path):
    """
    Load the raw churn dataset from a CSV file.

    Parameters:
        path (str or Path): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(path)


# Feature prep
def prepare_features(df):
    """
    Drop the Phone_Number column and convert Area_Code to a categorical type.

    Parameters:
        df (pd.DataFrame): The input DataFrame (after cleaning).

    Returns:
        pd.DataFrame: DataFrame with Phone_Number dropped and Area_Code as object.
    """
    df = df.drop("Phone_Number", axis=1)
    df["Area_Code"] = df["Area_Code"].astype(object)
    return df


# Use one-hot encoder to encode the categorical features
def build_encoder(categorical_cols):
    """
    Create an unfitted ColumnTransformer with OneHotEncoder for categorical columns.

    Parameters:
        categorical_cols (list[str]): Column names to one-hot encode.

    Returns:
        ColumnTransformer: An unfitted transformer.
    """
    return ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)
        ],
        remainder="passthrough",
    )


# Encode the categorical features
def encode_features(df, encoder, categorical_cols, fit=True):
    """
    Apply the ColumnTransformer to one-hot encode categorical features.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        encoder (ColumnTransformer): The transformer to apply.
        categorical_cols (list[str]): The categorical column names.
        fit (bool): If True, fit and transform. If False, transform only.

    Returns:
        tuple: (encoded DataFrame, fitted encoder)
    """
    if fit:
        encoded_array = encoder.fit_transform(df)
    else:
        encoded_array = encoder.transform(df)

    # Build column names: one-hot encoded columns + passthrough columns
    encoded_columns = encoder.named_transformers_["onehot"].get_feature_names_out(categorical_cols)
    remaining_columns = df.drop(columns=categorical_cols).columns
    all_columns = list(encoded_columns) + list(remaining_columns)

    encoded_df = pd.DataFrame(encoded_array, columns=all_columns)
    return encoded_df, encoder


# Encode the target column
def encode_target(series):
    """
    Apply LabelEncoder to the target column.

    Parameters:
        series (pd.Series): The target column (e.g. Churn with 'True'/'False' values).

    Returns:
        tuple: (encoded numpy array, fitted LabelEncoder)
    """
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(series)
    return encoded, label_encoder


# Split the data into train and test sets, applying stratification
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Perform a stratified train/test split.

    Parameters:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target array.
        test_size (float): Fraction of data reserved for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


# Scale the numerical features
def scale_features(X_train, X_test):
    """
    Fit a MinMaxScaler on the training data and transform both train and test sets.

    Parameters:
        X_train: Training feature matrix.
        X_test: Test feature matrix.

    Returns:
        tuple: (X_train_scaled, X_test_scaled, fitted MinMaxScaler)
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# Apply SMOTE to mitigate the class imbalance in the target column
def apply_smotenc(X_train, y_train, categorical_features, random_state=42):
    """
    Apply SMOTENC to oversample the minority class in the training data.

    Parameters:
        X_train (np.ndarray): Scaled training features.
        y_train (np.ndarray or pd.Series): Training labels.
        categorical_features (list[int]): Indices of categorical features in the scaled array.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train_resampled, y_train_resampled)
    """
    smote = SMOTENC(categorical_features=categorical_features, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled
