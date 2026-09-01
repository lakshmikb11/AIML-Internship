"""
W4D4: FastAPI Model Serving Endpoint

This API loads the Linear Regression model serialized during W4D3
and exposes endpoints for health checking and prediction.

Model:
    California Housing Linear Regression

Framework:
    FastAPI

Server:
    Uvicorn
"""

from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


# ---------------------------------------------------------
# 1. Application configuration
# ---------------------------------------------------------

MODEL_PATH = Path(
    "output_evidence/w4d3/linear_regression_model.joblib"
)

FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


# ---------------------------------------------------------
# 2. Load the serialized model
# ---------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Serialized model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------
# 3. Create FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="California Housing Model API",
    description="FastAPI endpoint for serving a serialized Linear Regression model.",
    version="1.0.0",
)


# ---------------------------------------------------------
# 4. Define request schema
# ---------------------------------------------------------

class HousingRequest(BaseModel):
    """Input features required by the trained model."""

    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


# ---------------------------------------------------------
# 5. Health check endpoint
# ---------------------------------------------------------

@app.get("/health")
def health_check():
    """Return API and model status."""

    return {
        "status": "healthy",
        "model_loaded": True,
        "model": "Linear Regression",
    }


# ---------------------------------------------------------
# 6. Prediction endpoint
# ---------------------------------------------------------

@app.post("/predict")
def predict(request: HousingRequest):
    """Generate a house-value prediction from input features."""

    input_data = np.array(
        [[
            request.MedInc,
            request.HouseAge,
            request.AveRooms,
            request.AveBedrms,
            request.Population,
            request.AveOccup,
            request.Latitude,
            request.Longitude,
        ]]
    )

    prediction = model.predict(input_data)

    return {
        "prediction": float(prediction[0]),
        "target": "MedHouseVal",
    }


# ---------------------------------------------------------
# 7. Root endpoint
# ---------------------------------------------------------

@app.get("/")
def root():
    """Return basic API information."""

    return {
        "message": "California Housing Model API",
        "endpoints": [
            "/health",
            "/predict",
            "/docs",
        ],
    }