# tests/data_prep/test_cleaning.py

"""
Tests for scripts/data_prep/cleaning.py - unit tests for data cleaning functions.
"""

import numpy as np
import pandas as pd

from scripts.data_prep.cleaning import (
    clean_nulls_and_duplicates,
    drop_highly_correlated_features,
    remove_outliers_zscore,
)


# ---------------------------------------------------------------------------
# clean_nulls_and_duplicates
# ---------------------------------------------------------------------------

class TestCleanNullsAndDuplicates:

    def test_column_names_are_standardized(self, raw_churn_df):
        result = clean_nulls_and_duplicates(raw_churn_df.copy())
        for col in result.columns:
            assert "_" in col or col.istitle(), f"Column '{col}' is not Title_Case"
            assert not col.startswith(" ") and not col.endswith(" ")

    def test_expected_columns_present(self, raw_churn_df):
        result = clean_nulls_and_duplicates(raw_churn_df.copy())
        expected = ["State", "Account_Length", "Area_Code", "Phone_Number", "Churn"]
        for col in expected:
            assert col in result.columns

    def test_nulls_are_dropped(self, raw_churn_df):
        df = raw_churn_df.copy()
        df.loc[0, "state"] = np.nan
        result = clean_nulls_and_duplicates(df)
        assert result.isnull().sum().sum() == 0
        assert len(result) == len(raw_churn_df) - 1

    def test_duplicates_are_dropped(self, raw_churn_df):
        df = pd.concat([raw_churn_df, raw_churn_df.iloc[[0]]], ignore_index=True)
        result = clean_nulls_and_duplicates(df)
        assert result.duplicated().sum() == 0
        assert len(result) == len(raw_churn_df)

    def test_clean_data_passes_through_unchanged(self, raw_churn_df):
        result = clean_nulls_and_duplicates(raw_churn_df.copy())
        assert result.shape[0] == 5

    def test_returns_dataframe(self, raw_churn_df):
        result = clean_nulls_and_duplicates(raw_churn_df.copy())
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# drop_highly_correlated_features
# ---------------------------------------------------------------------------

class TestDropHighlyCorrelatedFeatures:

    def test_drops_perfectly_correlated_column(self, numeric_df):
        result_df, dropped = drop_highly_correlated_features(numeric_df, threshold=0.9)
        # Lower-triangle logic: column A has B's corr value below diagonal → A is dropped
        assert "A" in dropped
        assert "A" not in result_df.columns
        assert "B" in result_df.columns
        assert "C" in result_df.columns
        assert "D" in result_df.columns

    def test_no_drop_when_below_threshold(self):
        df = pd.DataFrame({
            "X": [1, 2, 3, 4, 5],
            "Y": [5, 3, 1, 4, 2],
        })
        result_df, dropped = drop_highly_correlated_features(df, threshold=0.99)
        assert dropped == []
        assert list(result_df.columns) == ["X", "Y"]

    def test_returns_tuple_of_df_and_list(self, numeric_df):
        result = drop_highly_correlated_features(numeric_df)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], list)

    def test_verbose_false_suppresses_output(self, numeric_df, capsys):
        drop_highly_correlated_features(numeric_df, threshold=0.9, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# remove_outliers_zscore
# ---------------------------------------------------------------------------

class TestRemoveOutliersZscore:

    def test_removes_outlier_row(self):
        # 50 normal values + 1 extreme outlier → z-score well above 3
        values = np.concatenate([np.ones(50), [1000.0]])
        df = pd.DataFrame({"X": values})
        result = remove_outliers_zscore(df, z_threshold=3.0)
        assert len(result) < len(df)
        assert 1000.0 not in result["X"].values

    def test_no_outliers_returns_all_rows(self):
        df = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = remove_outliers_zscore(df, z_threshold=3.0)
        assert len(result) == len(df)

    def test_only_considers_numeric_columns(self):
        # 20 normal values + 1 extreme outlier, plus a string column
        values = np.concatenate([np.ones(20), [1000.0]])
        names = [f"name_{i}" for i in range(21)]
        df = pd.DataFrame({"name": names, "value": values})
        result = remove_outliers_zscore(df, z_threshold=2.0)
        assert "name" in result.columns
        assert 1000.0 not in result["value"].values

    def test_custom_threshold(self):
        df = pd.DataFrame({"X": np.concatenate([np.ones(50), [10.0]])})
        loose = remove_outliers_zscore(df, z_threshold=10.0)
        tight = remove_outliers_zscore(df, z_threshold=1.0)
        assert len(tight) <= len(loose)

    def test_returns_copy_not_view(self, numeric_df):
        result = remove_outliers_zscore(numeric_df)
        original_val = numeric_df.iloc[0, 0]
        result.iloc[0, 0] = -999
        assert numeric_df.iloc[0, 0] == original_val
