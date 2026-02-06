# api/main.py

"""
FastAPI application for serving churn predictions.

Endpoints:
    GET  /health         - Health check
    POST /predict        - Single customer prediction
    POST /predict/batch  - Batch prediction

"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerInput,
    HealthResponse,
    PredictionResponse,
)
from scripts.config import ARTIFACTS_DIR
from scripts.serving.predict import ChurnPredictor

# Global predictor instance, loaded once at startup
predictor: ChurnPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model artifacts on startup."""
    global predictor
    try:
        predictor = ChurnPredictor(artifacts_dir=ARTIFACTS_DIR)
        print(f"Model loaded from {ARTIFACTS_DIR}")
    except FileNotFoundError:
        print(f"WARNING: Artifacts not found at {ARTIFACTS_DIR}. Run the training script first.")
    yield


app = FastAPI(
    title="SyriaTel Customer Churn Prediction API",
    description="Predict whether a customer will churn based on their usage data.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API and model are operational."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerInput):
    """
    Predict churn for a single customer.
        `customer_id`: Unique identifier for the customer
        `features`: Dictionary of feature values (e.g., usage patterns, demographics)
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run the training script first.")

    result = predictor.predict(customer.to_model_input())
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn for multiple customers.
        `customers`: List of customer inputs, each containing a `customer_id` and `features`
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run the training script first.")

    inputs = [c.to_model_input() for c in request.customers]
    results = predictor.predict_batch(inputs)
    return BatchPredictionResponse(
        predictions=[PredictionResponse(**r) for r in results]
    )
