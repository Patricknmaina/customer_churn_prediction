# tests/api/test_schemas.py

"""
Tests for api/schemas.py — Pydantic model validation.
"""

import pytest
from pydantic import ValidationError

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerInput,
    HealthResponse,
    PredictionResponse,
)


# ---------------------------------------------------------------------------
# CustomerInput
# ---------------------------------------------------------------------------

class TestCustomerInput:

    def _required_fields(self):
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

    def test_valid_required_fields_only(self):
        ci = CustomerInput(**self._required_fields())
        assert ci.state == "OH"
        assert ci.account_length == 100
        assert ci.area_code == 415

    def test_valid_all_fields(self):
        fields = self._required_fields()
        fields.update({
            "state": "KS", "account_length": 128, "area_code": 408,
            "number_vmail_messages": 25, "total_day_calls": 110,
            "total_eve_calls": 99, "total_night_calls": 91,
        })
        ci = CustomerInput(**fields)
        assert ci.state == "KS"

    def test_missing_required_field_raises(self):
        fields = self._required_fields()
        del fields["customer_service_calls"]
        with pytest.raises(ValidationError):
            CustomerInput(**fields)

    def test_negative_customer_service_calls_raises(self):
        fields = self._required_fields()
        fields["customer_service_calls"] = -1
        with pytest.raises(ValidationError):
            CustomerInput(**fields)

    def test_negative_charge_raises(self):
        fields = self._required_fields()
        fields["total_day_charge"] = -1.0
        with pytest.raises(ValidationError):
            CustomerInput(**fields)

    def test_to_model_input_returns_dict_with_title_case_keys(self):
        ci = CustomerInput(**self._required_fields())
        d = ci.to_model_input()
        assert isinstance(d, dict)
        assert "Customer_Service_Calls" in d
        assert "State" in d
        assert len(d) == 19

    def test_to_model_input_values_match(self):
        fields = self._required_fields()
        fields["customer_service_calls"] = 3
        fields["total_day_charge"] = 50.0
        ci = CustomerInput(**fields)
        d = ci.to_model_input()
        assert d["Customer_Service_Calls"] == 3
        assert d["Total_Day_Charge"] == 50.0


# ---------------------------------------------------------------------------
# PredictionResponse
# ---------------------------------------------------------------------------

class TestPredictionResponse:

    def test_valid_response(self):
        pr = PredictionResponse(churn=True, churn_probability=0.85, risk_level="high")
        assert pr.churn is True

    def test_probability_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            PredictionResponse(churn=True, churn_probability=1.5, risk_level="high")
        with pytest.raises(ValidationError):
            PredictionResponse(churn=True, churn_probability=-0.1, risk_level="low")


# ---------------------------------------------------------------------------
# Batch models & HealthResponse
# ---------------------------------------------------------------------------

def test_batch_request_with_multiple_customers():
    required = {
        "customer_service_calls": 1, "total_day_charge": 45.0,
        "international_plan": "no", "total_eve_charge": 17.0,
        "total_intl_charge": 2.7, "total_intl_calls": 3,
        "total_night_charge": 11.0, "voice_mail_plan": "yes",
    }
    br = BatchPredictionRequest(customers=[CustomerInput(**required)] * 2)
    assert len(br.customers) == 2


def test_batch_response():
    preds = [
        PredictionResponse(churn=True, churn_probability=0.85, risk_level="high"),
        PredictionResponse(churn=False, churn_probability=0.15, risk_level="low"),
    ]
    resp = BatchPredictionResponse(predictions=preds)
    assert len(resp.predictions) == 2


def test_health_response():
    hr = HealthResponse(status="healthy", model_loaded=True)
    assert hr.status == "healthy"
    assert hr.model_loaded is True
