
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import serve
from predict import validate


client = TestClient(serve.app)


# Replace these values with valid encodings from your VALID_VALUES.
VALID_INPUT = {
    "age_group": 10,
    "sex": 1,
    "bmi": 28.5,
    "general_health": 3,
    "exercise_30d": 1,
    "smoking_status": 2,
    "heavy_drinker": 0,
    "heart_disease": 0,
}


def fake_predict(data):
    """Validate input without loading the actual model."""
    validate(pd.DataFrame([data]))

    return pd.DataFrame([{
        "diabetes_probability": 0.2345,
        "predicted_diabetes": 1,
        "prediction": "Higher risk",
    }])


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "200 ok"


def test_predict_success(monkeypatch):
    monkeypatch.setattr(serve, "predict", fake_predict)

    response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    assert response.json()["probability"] == 0.2345
    assert "message" in response.json()
    assert "disclaimer" in response.json()


def test_predict_zero(monkeypatch):
    def fake_zero_prediction(data):
        validate(pd.DataFrame([data]))

        return pd.DataFrame([{
            "diabetes_probability": 0.08,
            "predicted_diabetes": 0,
            "prediction": "Lower risk",
        }])

    monkeypatch.setattr(serve, "predict", fake_zero_prediction)

    response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 200
    assert response.json()["prediction"] == 0


def test_invalid_input(monkeypatch):
    monkeypatch.setattr(serve, "predict", fake_predict)

    invalid_input = VALID_INPUT.copy()
    invalid_input["bmi"] = 9

    response = client.post("/predict", json=invalid_input)

    assert response.status_code == 400


def test_missing_required_field():
    invalid_input = VALID_INPUT.copy()
    del invalid_input["age_group"]

    response = client.post("/predict", json=invalid_input)

    assert response.status_code == 422


def test_invalid_category(monkeypatch):
    monkeypatch.setattr(serve, "predict", fake_predict)

    invalid_input = VALID_INPUT.copy()
    invalid_input["smoking_status"] = 999

    response = client.post("/predict", json=invalid_input)

    assert response.status_code == 400


def test_model_not_found(monkeypatch):
    def missing_model(data):
        raise FileNotFoundError("Model artifact not found")

    monkeypatch.setattr(serve, "predict", missing_model)

    response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 503


def test_prediction_value_error(monkeypatch):
    def invalid_prediction(data):
        raise ValueError("Invalid input")

    monkeypatch.setattr(serve, "predict", invalid_prediction)

    response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 400