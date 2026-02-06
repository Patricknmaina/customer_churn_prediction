"""
Central configuration constants for the churn prediction project.

"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Data file paths
RAW_DATA_PATH = DATA_DIR / "churn.csv"
CLEANED_DATA_PATH = DATA_DIR / "churn_cleaned.csv"

# Random seed
RANDOM_STATE = 42

# Train/test split
TEST_SIZE = 0.2

# Feature cleaning thresholds
CORRELATION_THRESHOLD = 0.9
Z_SCORE_THRESHOLD = 3.0

# Column definitions
TARGET_COL = "Churn"
DROP_COLS = ["Phone_Number"]
CATEGORICAL_COLS = ["State", "Area_Code", "International_Plan", "Voice_Mail_Plan"]

# SMOTENC categorical feature indices (positions after one-hot encoding + scaling)
SMOTENC_CATEGORICAL_INDICES = [1, 2]

# Tuned XGBoost hyperparameters (from RandomizedSearchCV)
XGBOOST_PARAMS = {
    "learning_rate": 0.059692492785589066,
    "max_depth": 9,
    "n_estimators": 176,
    "subsample": 0.7819846225644594,
    "random_state": RANDOM_STATE,
}

# Prediction risk level thresholds
RISK_THRESHOLDS = {
    "low": 0.3,
    "high": 0.7,
}
