# tests/data_prep/test_preprocessing.py

"""
Tests for scripts/data_prep/preprocessing.py - unit tests for data preprocessing functions.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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


# ---------------------------------------------------------------------------
# load_raw_data
# ---------------------------------------------------------------------------

class TestLoadRawData:

    def test_loads_csv_from_path(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)
        result = load_raw_data(csv_path)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(Exception):
            load_raw_data(tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# prepare_features
# ---------------------------------------------------------------------------

class TestPrepareFeatures:

    def test_drops_phone_number_column(self, cleaned_churn_df):
        assert "Phone_Number" in cleaned_churn_df.columns
        result = prepare_features(cleaned_churn_df.copy())
        assert "Phone_Number" not in result.columns

    def test_converts_area_code_to_object(self, cleaned_churn_df):
        result = prepare_features(cleaned_churn_df.copy())
        assert result["Area_Code"].dtype == object

    def test_preserves_other_columns(self, cleaned_churn_df):
        result = prepare_features(cleaned_churn_df.copy())
        expected_cols = [c for c in cleaned_churn_df.columns if c != "Phone_Number"]
        for col in expected_cols:
            assert col in result.columns
        assert len(result) == len(cleaned_churn_df)


# ---------------------------------------------------------------------------
# build_encoder
# ---------------------------------------------------------------------------

class TestBuildEncoder:

    def test_returns_column_transformer(self):
        result = build_encoder(["State", "Area_Code"])
        assert isinstance(result, ColumnTransformer)

    def test_encoder_has_onehot_transformer(self):
        result = build_encoder(["State"])
        assert result.transformers[0][0] == "onehot"
        assert result.transformers[0][2] == ["State"]

    def test_remainder_is_passthrough(self):
        result = build_encoder(["State"])
        assert result.remainder == "passthrough"


# ---------------------------------------------------------------------------
# encode_features
# ---------------------------------------------------------------------------

class TestEncodeFeatures:

    def test_fit_and_transform(self, prepared_df):
        categorical_cols = ["State", "Area_Code", "International_Plan", "Voice_Mail_Plan"]
        # Drop target before encoding
        df = prepared_df.drop(columns=["Churn"])
        encoder = build_encoder(categorical_cols)
        result_df, fitted_encoder = encode_features(df, encoder, categorical_cols, fit=True)
        assert isinstance(result_df, pd.DataFrame)
        for cat_col in categorical_cols:
            assert cat_col not in result_df.columns
        assert "Total_Day_Charge" in result_df.columns

    def test_transform_only(self, prepared_df):
        categorical_cols = ["State", "Area_Code", "International_Plan", "Voice_Mail_Plan"]
        df = prepared_df.drop(columns=["Churn"])
        encoder = build_encoder(categorical_cols)
        _, fitted = encode_features(df, encoder, categorical_cols, fit=True)
        result_df, _ = encode_features(df, fitted, categorical_cols, fit=False)
        assert result_df.shape[0] == len(df)


# ---------------------------------------------------------------------------
# encode_target
# ---------------------------------------------------------------------------

class TestEncodeTarget:

    def test_encodes_boolean_series(self):
        series = pd.Series([False, True, False, True, False])
        encoded, le = encode_target(series)
        assert isinstance(encoded, np.ndarray)
        assert isinstance(le, LabelEncoder)
        assert set(encoded) == {0, 1}

    def test_label_encoder_can_inverse_transform(self):
        series = pd.Series([True, False, True])
        encoded, le = encode_target(series)
        original = le.inverse_transform(encoded)
        assert list(original) == [True, False, True]


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------

class TestSplitData:

    def test_returns_four_arrays(self):
        X = np.random.rand(100, 5)
        y = np.array([0] * 80 + [1] * 20)
        result = split_data(X, y)
        assert len(result) == 4

    def test_correct_split_proportions(self):
        X = np.random.rand(100, 5)
        y = np.array([0] * 80 + [1] * 20)
        X_train, X_test, _, _ = split_data(X, y, test_size=0.2)
        assert X_test.shape[0] == 20
        assert X_train.shape[0] == 80

    def test_reproducibility_with_random_state(self):
        X = np.random.rand(100, 5)
        y = np.array([0] * 80 + [1] * 20)
        X_train1, _, _, _ = split_data(X, y, random_state=42)
        X_train2, _, _, _ = split_data(X, y, random_state=42)
        np.testing.assert_array_equal(X_train1, X_train2)


# ---------------------------------------------------------------------------
# scale_features
# ---------------------------------------------------------------------------

class TestScaleFeatures:

    def test_returns_scaled_arrays_and_scaler(self):
        X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_test = np.array([[2.0, 3.0]])
        X_tr, X_te, scaler = scale_features(X_train, X_test)
        assert isinstance(scaler, MinMaxScaler)

    def test_train_values_in_zero_one_range(self):
        X_train = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
        X_test = np.array([[2.5, 15.0]])
        X_tr, _, _ = scale_features(X_train, X_test)
        assert X_tr.min() >= 0.0
        assert X_tr.max() <= 1.0

    def test_scaler_fitted_on_train_only(self):
        X_train = np.array([[0.0, 0.0], [10.0, 10.0]])
        X_test = np.array([[5.0, 5.0], [15.0, 15.0]])
        _, X_te, _ = scale_features(X_train, X_test)
        # 15 exceeds train max of 10, so scaled value > 1.0
        assert X_te.max() > 1.0


# ---------------------------------------------------------------------------
# apply_smotenc
# ---------------------------------------------------------------------------

class TestApplySmotenc:

    def test_resamples_minority_class(self):
        rng = np.random.RandomState(42)
        X = rng.rand(100, 5)
        # Make column 0 categorical-like (binary)
        X[:, 0] = rng.choice([0, 1], size=100)
        y = np.array([0] * 90 + [1] * 10)
        X_res, y_res = apply_smotenc(X, y, categorical_features=[0], random_state=42)
        assert len(y_res) > len(y)
        assert np.sum(y_res == 0) == np.sum(y_res == 1)

    def test_preserves_feature_count(self):
        rng = np.random.RandomState(42)
        X = rng.rand(100, 5)
        X[:, 0] = rng.choice([0, 1], size=100)
        y = np.array([0] * 90 + [1] * 10)
        X_res, _ = apply_smotenc(X, y, categorical_features=[0], random_state=42)
        assert X_res.shape[1] == X.shape[1]

    def test_reproducibility(self):
        rng = np.random.RandomState(42)
        X = rng.rand(100, 5)
        X[:, 0] = rng.choice([0, 1], size=100)
        y = np.array([0] * 90 + [1] * 10)
        X_res1, _ = apply_smotenc(X, y, categorical_features=[0], random_state=42)
        X_res2, _ = apply_smotenc(X, y, categorical_features=[0], random_state=42)
        np.testing.assert_array_equal(X_res1, X_res2)
