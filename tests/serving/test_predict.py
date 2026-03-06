# tests/serving/test_predict.py

"""
Tests for scripts/serving/predict.py — unit tests for ChurnPredictor class.
"""

import numpy as np
import pytest

from scripts.serving.predict import ChurnPredictor


# ---------------------------------------------------------------------------
# ChurnPredictor.__init__
# ---------------------------------------------------------------------------

class TestChurnPredictorInit:

    def test_loads_all_artifacts(self, mock_artifacts):
        predictor = ChurnPredictor(artifacts_dir=mock_artifacts)
        assert predictor.model is not None
        assert predictor.scaler is not None
        assert predictor.encoder is not None
        assert predictor.label_encoder is not None
        assert isinstance(predictor.feature_names, list) and len(predictor.feature_names) > 0
        assert isinstance(predictor.dropped_features, list)

    def test_raises_on_missing_artifacts(self, tmp_path):
        with pytest.raises(Exception):
            ChurnPredictor(artifacts_dir=tmp_path)


# ---------------------------------------------------------------------------
# _classify_risk
# ---------------------------------------------------------------------------

class TestClassifyRisk:

    def test_low_risk(self, predictor):
        assert predictor._classify_risk(0.0) == "low"
        assert predictor._classify_risk(0.1) == "low"
        assert predictor._classify_risk(0.29) == "low"

    def test_medium_risk(self, predictor):
        assert predictor._classify_risk(0.3) == "medium"
        assert predictor._classify_risk(0.5) == "medium"
        assert predictor._classify_risk(0.69) == "medium"

    def test_high_risk(self, predictor):
        assert predictor._classify_risk(0.7) == "high"
        assert predictor._classify_risk(0.99) == "high"
        assert predictor._classify_risk(1.0) == "high"

    def test_boundary_at_low_threshold(self, predictor):
        # 0.3 is NOT < 0.3, so it should be "medium"
        assert predictor._classify_risk(0.3) == "medium"

    def test_boundary_at_high_threshold(self, predictor):
        # 0.7 is >= 0.7, so it should be "high"
        assert predictor._classify_risk(0.7) == "high"


# ---------------------------------------------------------------------------
# preprocess_input
# ---------------------------------------------------------------------------

class TestPreprocessInput:

    def test_returns_numpy_array(self, predictor, valid_customer_input):
        result = predictor.preprocess_input(valid_customer_input)
        assert isinstance(result, np.ndarray)

    def test_output_shape_matches_feature_count(self, predictor, valid_customer_input):
        result = predictor.preprocess_input(valid_customer_input)
        assert result.shape == (1, len(predictor.feature_names))

    def test_handles_correlated_features(self, predictor, valid_customer_input):
        # Input contains Total_Day_Minutes etc. which should be silently dropped
        result = predictor.preprocess_input(valid_customer_input)
        assert result.shape[1] == len(predictor.feature_names)


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

class TestPredict:

    def test_returns_dict(self, predictor, valid_customer_input):
        result = predictor.predict(valid_customer_input)
        assert isinstance(result, dict)

    def test_has_required_keys(self, predictor, valid_customer_input):
        result = predictor.predict(valid_customer_input)
        assert set(result.keys()) == {"churn", "churn_probability", "risk_level", "feature_contributions"}

    def test_churn_is_bool(self, predictor, valid_customer_input):
        result = predictor.predict(valid_customer_input)
        assert isinstance(result["churn"], bool)

    def test_probability_in_range(self, predictor, valid_customer_input):
        result = predictor.predict(valid_customer_input)
        assert 0.0 <= result["churn_probability"] <= 1.0

    def test_risk_level_is_valid(self, predictor, valid_customer_input):
        result = predictor.predict(valid_customer_input)
        assert result["risk_level"] in ("low", "medium", "high")

    def test_probability_rounded_to_four_decimals(self, predictor, valid_customer_input):
        result = predictor.predict(valid_customer_input)
        prob = result["churn_probability"]
        assert prob == round(prob, 4)


# ---------------------------------------------------------------------------
# predict_batch
# ---------------------------------------------------------------------------

class TestPredictBatch:

    def test_returns_list(self, predictor, valid_customer_input):
        results = predictor.predict_batch([valid_customer_input, valid_customer_input])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_each_element_has_correct_keys(self, predictor, valid_customer_input):
        results = predictor.predict_batch([valid_customer_input])
        for r in results:
            assert set(r.keys()) == {"churn", "churn_probability", "risk_level", "feature_contributions"}

    def test_empty_list(self, predictor):
        results = predictor.predict_batch([])
        assert results == []
