# scripts/serving/predict.py

"""
Inference engine for churn prediction.

Loads the model artifacts and provides a unified prediction interface
shared by both the FastAPI app and the Streamlit dashboard.

"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from scripts.config import CATEGORICAL_COLS, RISK_THRESHOLDS

# Prefixes used by one-hot encoding that should be aggregated
# back to a single original feature name for display.
_ONE_HOT_PREFIXES = {
    "State_": "State",
    "Area_Code_": "Area Code",
    "International_Plan_": "International Plan",
    "Voice_Mail_Plan_": "Voice Mail Plan",
}


class ChurnPredictor:
    """
    Loads trained artifacts and provides churn predictions.
    """

    def __init__(self, artifacts_dir):
        """
        Load model, scaler, encoder, and metadata from the artifacts directory.

        Args:
            artifacts_dir (str or Path): Path to the directory containing serialized artifacts.
        """
        artifacts_dir = Path(artifacts_dir)

        self.model = joblib.load(artifacts_dir / "model.joblib")
        self.scaler = joblib.load(artifacts_dir / "scaler.joblib")
        self.encoder = joblib.load(artifacts_dir / "encoder.joblib")
        self.label_encoder = joblib.load(artifacts_dir / "label_encoder.joblib")

        with open(artifacts_dir / "feature_names.json") as f:
            self.feature_names = json.load(f)

        with open(artifacts_dir / "dropped_features.json") as f:
            self.dropped_features = json.load(f)

        self.explainer = shap.TreeExplainer(self.model)

    def preprocess_input(self, data):
        """
        Transform raw customer data through the same pipeline used during training.

        Accepts data in the same format as the original CSV columns
        (minus Phone_Number and Churn). Correlated features (e.g. Total_Day_Minutes)
        are silently dropped if present.

        Args:
            data (dict): Raw customer features as key-value pairs.

        Returns:
            np.ndarray: Scaled feature array ready for model.predict().
        """
        # Create a single-row DataFrame from the input
        df = pd.DataFrame([data])

        # Convert Area_Code to object type (as done during training)
        if "Area_Code" in df.columns:
            df["Area_Code"] = df["Area_Code"].astype(object)

        # Drop correlated features that the model doesn't use
        cols_to_drop = [col for col in self.dropped_features if col in df.columns]
        df = df.drop(columns=cols_to_drop, errors="ignore")

        # Apply the fitted one-hot encoder
        encoded_array = self.encoder.transform(df)

        # Build column names to match training order
        encoded_columns = self.encoder.named_transformers_["onehot"].get_feature_names_out(CATEGORICAL_COLS)
        remaining_columns = df.drop(columns=CATEGORICAL_COLS).columns
        all_columns = list(encoded_columns) + list(remaining_columns)

        encoded_df = pd.DataFrame(encoded_array, columns=all_columns)

        # Reorder columns to match training feature order
        encoded_df = encoded_df[self.feature_names]

        # Apply the fitted scaler
        scaled = self.scaler.transform(encoded_df)

        return scaled

    def _classify_risk(self, probability):
        """
        Map a churn probability to a risk level string.

        Args:
            probability (float): Churn probability between 0 and 1.

        Returns:
            str: "low", "medium", or "high"
        """
        if probability < RISK_THRESHOLDS["low"]:
            return "low"
        elif probability >= RISK_THRESHOLDS["high"]:
            return "high"
        return "medium"

    def _aggregate_shap_values(self, shap_values):
        """
        Aggregate encoded SHAP values back to original feature names.

        Groups one-hot encoded features (State_*, Area_Code_*, etc.) by summing
        their SHAP contributions. Returns the top 10 contributors sorted by
        absolute contribution.

        Args:
            shap_values (np.ndarray): SHAP values for a single prediction.

        Returns:
            list[dict]: [{"feature": str, "contribution": float}, ...] top 10.
        """
        aggregated = {}

        for feature_name, value in zip(self.feature_names, shap_values):
            # Check if this feature belongs to a one-hot group
            original_name = None
            for prefix, name in _ONE_HOT_PREFIXES.items():
                if feature_name.startswith(prefix):
                    original_name = name
                    break

            if original_name is None:
                # Numeric feature — make display-friendly
                original_name = feature_name.replace("_", " ")

            aggregated[original_name] = aggregated.get(original_name, 0.0) + float(value)

        # Sort by absolute contribution, descending
        sorted_contributions = sorted(
            aggregated.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        return [
            {"feature": name, "contribution": round(contrib, 4)}
            for name, contrib in sorted_contributions[:10]
        ]

    def predict(self, data):
        """
        Predict churn for a single customer.

        Args:
            data (dict): Raw customer features.

        Returns:
            dict: {"churn": bool, "churn_probability": float, "risk_level": str,
                   "feature_contributions": list[dict]}
        """
        scaled = self.preprocess_input(data)

        prediction = self.model.predict(scaled)[0]
        probability = float(self.model.predict_proba(scaled)[0, 1])

        shap_values = self.explainer.shap_values(scaled)
        sv = np.asarray(shap_values[0])
        if sv.ndim == 2:
            sv = sv[:, 1]  # binary classifier: take class-1 (churn) SHAP values
        contributions = self._aggregate_shap_values(sv)

        return {
            "churn": bool(prediction),
            "churn_probability": round(probability, 4),
            "risk_level": self._classify_risk(probability),
            "feature_contributions": contributions,
        }

    def predict_batch(self, data_list):
        """
        Predict churn for multiple customers.

        Args:
            data_list (list[dict]): List of raw customer feature dicts.

        Returns:
            list[dict]: List of prediction results.
        """
        return [self.predict(data) for data in data_list]
