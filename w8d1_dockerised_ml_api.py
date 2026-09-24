"""
W8D1: Dockerised ML API

Production-ready FastAPI service for the California Housing
Linear Regression model.

W8D1 adds Docker, testing, CI/CD, and monitoring around the
existing ML model-serving workflow.
"""

from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path(
    "output_evidence/w4d3/linear_regression_model.joblib"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Serialized model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="California Housing W8D1 ML API",
    description="Dockerised FastAPI service for a California Housing Linear Regression model.",
    version="1.0.0",
)


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


@app.get("/health")
def health_check():
    """Return API and model health status."""

    return {
        "status": "healthy",
        "model_loaded": True,
        "model": "Linear Regression",
    }


@app.post("/predict")
def predict(request: HousingRequest):
    """Generate a house-value prediction."""

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


@app.get("/")
def root():
    """Return basic API information."""

    return {
        "message": "California Housing W8D1 ML API",
        "endpoints": [
            "/health",
            "/predict",
            "/docs",
        ],
    }
