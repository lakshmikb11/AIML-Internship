from fastapi.testclient import TestClient

from w8d1_dockerised_ml_api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict_endpoint():
    payload = {
        "MedInc": 5.0,
        "HouseAge": 20.0,
        "AveRooms": 6.0,
        "AveBedrms": 1.0,
        "Population": 1000.0,
        "AveOccup": 3.0,
        "Latitude": 35.0,
        "Longitude": -119.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["target"] == "MedHouseVal"
