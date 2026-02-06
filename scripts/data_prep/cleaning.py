# scripts/data/cleaning.py

"""
Functions for cleaning and preparing the raw data.

"""

# Imports
import numpy as np
import pandas as pd
from scipy import stats


def clean_nulls_and_duplicates(df):
    """
    This function cleans a dataframe by checking for, and handling null values and duplicate rows.
    It also standardizes the columns by removing the whitespaces between the words, adding a hyphen for readability and capitalizing the first letter in each word.

    Parameters:
        df(pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A cleaned DataFrame with no duplicate or null values, and standardized columns
    """

    print("Initial shape of the dataset:", df.shape)

    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        print("\nNull values detected in the following columns:")
        print(null_counts[null_counts > 0])

        # Drop the missing values if any
        df = df.dropna(axis=0)
        print("Dropped rows with missing values")
    else:
        print("\nNo null values detected.")

    # Check for duplicate values
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"\nFound {duplicate_count} duplicate rows.")

        # drop the duplicate rows
        df = df.drop_duplicates()
        print("Dropped duplicate rows.")
    else:
        print("\nNo duplicate rows detected.")

    # Standardize the column names
    df.columns = (
        df.columns
        .str.strip()                          # Remove leading/trailing whitespace
        .str.title()                          # Capitalize first letter of each word
        .str.replace(' ', '_', regex=False)   # Replace spaces with hyphens
    )

    print(df.columns)

    print("\n Final shape of data:", df.shape)
    return df


# Function to remove highly correlated features
def drop_highly_correlated_features(df, threshold: float = 0.9, verbose: bool = True):
    """
    This function drops features that are highly correlated with others, beyond a specified threshold

    Parameters:
    -----------
        df(pd.DataFrame): The input dataframe
        threshold: float, default=0.9: Correlation threshold for dropping features
        verbose: bool, default=True: If True, print the names of the dropped features

    Returns:
    --------
        df: A new dataframe with highly correlated features removed.
    """

    # Compute the absolute correlation matrix
    corr_matrix = df.corr(numeric_only=True).abs()

    # Mask the upper triangle (including diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    tri_df = corr_matrix.mask(mask)

    # Identify features to drop
    to_drop = [col for col in tri_df.columns if any(tri_df[col] > threshold)]

    # Print the features that are dropped
    if verbose and to_drop:
        print(f"Dropping {len(to_drop)} highly correlated features (r > {threshold}): {to_drop}")
    elif verbose:
        print("No features exceeded the correlation threshold.")

    # drop the features and return reduced dataframe
    return df.drop(columns=to_drop, axis=1), to_drop


# Function to remove outliers based on Z-score
def remove_outliers_zscore(df, z_threshold=3.0):
    """
    This function returns a new dataframe with rows removed where any numerical column has a Z-score
    exceeding the given threshold

    Parameters:
    -----------
        df(pd.DataFrame): The input dataframe containing both numerical and non-numerical columns.
        z_threshold: float, default=3.0: All rows where |z-score| > z_threshold for any numeric column will be dropped

    Returns:
    --------
        df: A copy of the original DataFrame with outliers removed
    """

    # Separate the numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Compute the Z-scores for each numeric column
    z_scores = np.abs(stats.zscore(df[numeric_cols], nan_policy='omit'))

    # Replace NaN z-scores with 0 before comparison
    z_scores = np.nan_to_num(z_scores, nan=0.0)
    mask = (z_scores <= z_threshold).all(axis=1)

    # Use the mask to filter the original dataframe
    cleaned_df = df.loc[mask].copy()

    return cleaned_df
