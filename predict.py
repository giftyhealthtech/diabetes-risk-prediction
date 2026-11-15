from pathlib import Path

import joblib
import pandas as pd


# MODEL_PATH = Path("diabetes_model.joblib")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diabetes_model.joblib"


def load_model():
    """Load the saved model and its prediction threshold."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}; run train.py first."
        )

    artifact = joblib.load(MODEL_PATH)

    return (
        artifact["model"],
        artifact["threshold"],
        artifact["features"],
    )

VALID_VALUES = {
    "age_group": list(range(1, 14)),
    "sex": [0, 1],
    "general_health": list(range(1, 6)),
    "exercise_30d": [0, 1],
    "smoking_status": [1, 2, 3, 4],
    "heavy_drinker": [0, 1],
    "heart_disease": [0, 1],
}


def validate(data):
    data = data.copy()

    for column, allowed in VALID_VALUES.items():
        if not data[column].isin(allowed).all():
            raise ValueError(f"Invalid value in '{column}'. Allowed: {allowed}")

    # BMI may be missing (the pipeline fills it), but must be realistic if given
    data["bmi"] = pd.to_numeric(data["bmi"], errors="coerce")
    if not data["bmi"].dropna().between(10, 100).all():
        raise ValueError("BMI must be between 10 and 100.")

    # Match the dtype used in training
    data["smoking_status"] = data["smoking_status"].astype(object)
    return data

def predict(data):
    """
    Predict diabetes status for one or more records.

    Expected input: dictionary for one record or DataFrame for multiple records.
    """

    model, threshold, features = load_model()

    if isinstance(data, dict):
        data = pd.DataFrame([data])

    elif not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a dictionary or pandas DataFrame.")

    # Ensure the input contains all required features
    missing_features = set(features) - set(data.columns)

    if missing_features:
        raise ValueError(
            f"Missing required features: {sorted(missing_features)}"
        )

    # Keep the same feature order used during training
    data = validate(data[features])

    # Generate probabilities
    probabilities = model.predict_proba(data)[:, 1]

    # Apply the selected classification threshold
    predictions = (probabilities >= threshold).astype(int)

    results = pd.DataFrame({
        "diabetes_probability": probabilities,
        "predicted_diabetes": predictions,
    })

    results["prediction"] = results["predicted_diabetes"].map({
        0: "Lower risk",
        1: "Higher risk",
    })

    return results


if __name__ == "__main__":

    sample = {
        "age_group": 10,
        "sex": 1,
        "bmi": 28.5,
        "general_health": 3,
        "exercise_30d": 1,
        "smoking_status": 2,
        "heavy_drinker": 0,
        "heart_disease": 0,
    }

    healthy = {"age_group": 2, "sex": 1, "bmi": 22, "general_health": 1,
            "exercise_30d": 1, "smoking_status": 4, "heavy_drinker": 0, "heart_disease": 0}

    high_risk = {"age_group": 11, "sex": 0, "bmi": 36, "general_health": 5,
                "exercise_30d": 0, "smoking_status": 3, "heavy_drinker": 0, "heart_disease": 1}

    result = predict(sample)
    healthy_result = predict(healthy)
    high_risk_result = predict(high_risk)

    print("\nPrediction Result:")
    print(result.to_string(index=False))

    print("\nPrediction Healthy Result:")
    print(healthy_result.to_string(index=False))

    print("\nPrediction high risk Result:")
    print(high_risk_result.to_string(index=False))