# api/schemas.py

"""
Pydantic request and response models for the churn prediction API.

"""

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """
    Input schema for churn prediction.

    Required fields are the high-impact features that most influence the
    model's prediction. Optional fields have sensible defaults and can be
    omitted for a simpler request.
    """

    # --- Required: high-impact features ---
    customer_service_calls: int = Field(..., ge=0, description="Number of calls to customer service (strongest predictor)", examples=[1])
    total_day_charge: float = Field(..., ge=0, description="Total day charges ($)", examples=[45.07])
    international_plan: str = Field(..., description="'yes' or 'no'", examples=["no"])
    total_eve_charge: float = Field(..., ge=0, description="Total evening charges ($)", examples=[16.78])
    total_intl_charge: float = Field(..., ge=0, description="Total international charges ($)", examples=[2.70])
    total_intl_calls: int = Field(..., ge=0, description="Total international calls", examples=[3])
    total_night_charge: float = Field(..., ge=0, description="Total night charges ($)", examples=[11.01])
    voice_mail_plan: str = Field(..., description="'yes' or 'no'", examples=["yes"])

    # --- Optional: lower-impact features with defaults ---
    state: str = Field(default="OH", description="US state abbreviation", examples=["KS"])
    account_length: int = Field(default=100, description="Number of days as account holder", examples=[128])
    area_code: int = Field(default=415, description="Area code (408, 415, or 510)", examples=[415])
    number_vmail_messages: int = Field(default=0, ge=0, description="Number of voicemail messages", examples=[25])
    total_day_calls: int = Field(default=100, ge=0, description="Total day calls", examples=[110])
    total_eve_calls: int = Field(default=100, ge=0, description="Total evening calls", examples=[99])
    total_night_calls: int = Field(default=100, ge=0, description="Total night calls", examples=[91])

    # --- Optional: dropped by model (correlated with charges) ---
    total_day_minutes: float = Field(default=0.0, description="Dropped by model - correlated with day charge")
    total_eve_minutes: float = Field(default=0.0, description="Dropped by model - correlated with eve charge")
    total_night_minutes: float = Field(default=0.0, description="Dropped by model - correlated with night charge")
    total_intl_minutes: float = Field(default=0.0, description="Dropped by model - correlated with intl charge")

    def to_model_input(self):
        """
        Convert to a dict with column names matching the training data
        (Title_Case with underscores, as produced by clean_nulls_and_duplicates).
        """
        return {
            "State": self.state,
            "Account_Length": self.account_length,
            "Area_Code": self.area_code,
            "International_Plan": self.international_plan,
            "Voice_Mail_Plan": self.voice_mail_plan,
            "Number_Vmail_Messages": self.number_vmail_messages,
            "Total_Day_Minutes": self.total_day_minutes,
            "Total_Day_Calls": self.total_day_calls,
            "Total_Day_Charge": self.total_day_charge,
            "Total_Eve_Minutes": self.total_eve_minutes,
            "Total_Eve_Calls": self.total_eve_calls,
            "Total_Eve_Charge": self.total_eve_charge,
            "Total_Night_Minutes": self.total_night_minutes,
            "Total_Night_Calls": self.total_night_calls,
            "Total_Night_Charge": self.total_night_charge,
            "Total_Intl_Minutes": self.total_intl_minutes,
            "Total_Intl_Calls": self.total_intl_calls,
            "Total_Intl_Charge": self.total_intl_charge,
            "Customer_Service_Calls": self.customer_service_calls,
        }


class FeatureContribution(BaseModel):
    """
    A single feature's contribution to the churn prediction.
    """

    feature: str
    contribution: float


class PredictionResponse(BaseModel):
    """
    Single customer prediction result with per-feature explanations.
    """

    churn: bool
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str  # "low", "medium", or "high"
    feature_contributions: list[FeatureContribution] = []


class BatchPredictionRequest(BaseModel):
    """
    Batch prediction input.
    """

    customers: list[CustomerInput]


class BatchPredictionResponse(BaseModel):
    """
    Batch prediction output.
    """

    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """
    Health check response.
    """

    status: str
    model_loaded: bool
