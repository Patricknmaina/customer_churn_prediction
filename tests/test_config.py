# tests/test_config.py

"""
Tests for scripts/config.py — verify constants have correct types and values.
"""

from pathlib import Path

from scripts.config import (
    ARTIFACTS_DIR,
    CATEGORICAL_COLS,
    CORRELATION_THRESHOLD,
    DATA_DIR,
    DROP_COLS,
    PROJECT_ROOT,
    RANDOM_STATE,
    RISK_THRESHOLDS,
    SMOTENC_CATEGORICAL_INDICES,
    TARGET_COL,
    TEST_SIZE,
    XGBOOST_PARAMS,
    Z_SCORE_THRESHOLD,
)


def test_project_root_exists_and_is_directory():
    assert isinstance(PROJECT_ROOT, Path)
    assert PROJECT_ROOT.is_dir()


def test_data_paths_are_under_project_root():
    assert DATA_DIR == PROJECT_ROOT / "data"


def test_artifacts_dir_is_under_project_root():
    assert ARTIFACTS_DIR == PROJECT_ROOT / "artifacts"


def test_numeric_constants_have_expected_types_and_ranges():
    assert isinstance(RANDOM_STATE, int) and RANDOM_STATE == 42
    assert isinstance(TEST_SIZE, float) and 0 < TEST_SIZE < 1
    assert isinstance(CORRELATION_THRESHOLD, float) and 0 < CORRELATION_THRESHOLD <= 1
    assert isinstance(Z_SCORE_THRESHOLD, float) and Z_SCORE_THRESHOLD > 0


def test_column_definitions():
    assert TARGET_COL == "Churn"
    assert "Phone_Number" in DROP_COLS
    assert CATEGORICAL_COLS == ["State", "Area_Code", "International_Plan", "Voice_Mail_Plan"]


def test_xgboost_params_keys():
    assert isinstance(XGBOOST_PARAMS, dict)
    for key in ("learning_rate", "max_depth", "n_estimators", "subsample", "random_state"):
        assert key in XGBOOST_PARAMS
    assert XGBOOST_PARAMS["learning_rate"] > 0
    assert XGBOOST_PARAMS["max_depth"] > 0
    assert XGBOOST_PARAMS["n_estimators"] > 0
    assert 0 < XGBOOST_PARAMS["subsample"] <= 1


def test_risk_thresholds():
    assert isinstance(RISK_THRESHOLDS, dict)
    assert "low" in RISK_THRESHOLDS and "high" in RISK_THRESHOLDS
    assert 0 < RISK_THRESHOLDS["low"] < RISK_THRESHOLDS["high"] < 1


def test_smotenc_categorical_indices():
    assert isinstance(SMOTENC_CATEGORICAL_INDICES, list)
    assert all(isinstance(i, int) and i >= 0 for i in SMOTENC_CATEGORICAL_INDICES)
