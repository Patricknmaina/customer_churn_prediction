# tests/api/test_main.py

"""
Tests for api/main.py — FastAPI endpoint tests.
"""

from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def valid_payload():
    return {
        "customer_service_calls": 1,
        "total_day_charge": 45.07,
        "international_plan": "no",
        "total_eve_charge": 16.78,
        "total_intl_charge": 2.70,
        "total_intl_calls": 3,
        "total_night_charge": 11.01,
        "voice_mail_plan": "yes",
    }


@pytest.fixture
def mock_predictor():
    pred = MagicMock()
    pred.predict.return_value = {
        "churn": True,
        "churn_probability": 0.85,
        "risk_level": "high",
    }
    pred.predict_batch.return_value = [
        {"churn": True, "churn_probability": 0.85, "risk_level": "high"},
        {"churn": False, "churn_probability": 0.15, "risk_level": "low"},
    ]
    return pred


@pytest.fixture
def client_with_model(mock_predictor):
    from api.main import app
    with TestClient(app) as client:
        import api.main
        original = api.main.predictor
        api.main.predictor = mock_predictor
        yield client
        api.main.predictor = original


@pytest.fixture
def client_without_model():
    from api.main import app
    with TestClient(app) as client:
        import api.main
        original = api.main.predictor
        api.main.predictor = None
        yield client
        api.main.predictor = original


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_health_with_model(self, client_with_model):
        response = client_with_model.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True

    def test_health_without_model(self, client_without_model):
        response = client_without_model.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is False


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

class TestPredictEndpoint:

    def test_predict_success(self, client_with_model, valid_payload):
        response = client_with_model.post("/predict", json=valid_payload)
        assert response.status_code == 200
        body = response.json()
        assert "churn" in body
        assert "churn_probability" in body
        assert "risk_level" in body

    def test_predict_model_not_loaded(self, client_without_model, valid_payload):
        response = client_without_model.post("/predict", json=valid_payload)
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]

    def test_predict_missing_required_field(self, client_with_model):
        response = client_with_model.post("/predict", json={"total_day_charge": 45.07})
        assert response.status_code == 422

    def test_predict_invalid_value(self, client_with_model, valid_payload):
        valid_payload["customer_service_calls"] = -1
        response = client_with_model.post("/predict", json=valid_payload)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /predict/batch
# ---------------------------------------------------------------------------

class TestBatchPredictEndpoint:

    def test_batch_predict_success(self, client_with_model, valid_payload):
        response = client_with_model.post(
            "/predict/batch", json={"customers": [valid_payload, valid_payload]}
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["predictions"]) == 2

    def test_batch_predict_model_not_loaded(self, client_without_model, valid_payload):
        response = client_without_model.post(
            "/predict/batch", json={"customers": [valid_payload]}
        )
        assert response.status_code == 503
