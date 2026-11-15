
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from predict import predict


app = FastAPI(
    title="Diabetes Prediction API",
    description="API for predicting self-reported diabetes status using a trained Logistic Regression model.",
    version="1.0.0",
)


class DiabetesInput(BaseModel):
    age_group: int
    sex: int
    bmi: float = Field(gt=0)
    general_health: int
    exercise_30d: int
    smoking_status: int
    heavy_drinker: int
    heart_disease: int


@app.get("/health")
def health_check():
    return {
        "status": "200 ok",
        "message": "Diabetes Prediction API is running"
    }


@app.post("/predict")
def make_prediction(data: DiabetesInput):
    try:
        result = predict(data.model_dump())

        probability = float(result.iloc[0]["diabetes_probability"])
        prediction = int(result.iloc[0]["predicted_diabetes"])

        if prediction == 1:
            message = (
                "The model identified a pattern associated with "
                "self-reported diabetes in the survey data. "
                "This result does not confirm that you have diabetes. "
                "Please visit a healthcare facility for appropriate "
                "blood glucose testing and professional evaluation."
            )
        else:
            message = (
                "The model did not identify a pattern associated with "
                "self-reported diabetes in the survey data. "
                "This does not rule out diabetes. Consider routine "
                "blood glucose testing and consult a healthcare "
                "professional if you have concerns."
            )

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "probability_percentage": f"{probability * 100:.2f}%",
            "message": message,
            "disclaimer": (
                "This tool is for educational purposes only and "
                "is not a medical diagnostic tool."
            )
        }

    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))

    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error))

    
@app.get("/disclaimer")
def disclaimer():
    return {
        "disclaimer": (
            "This application is an educational machine learning project "
            "that predicts self-reported diabetes status from survey data. "
            "It is not a medical diagnostic tool and must not be used to "
            "diagnose diabetes or guide treatment decisions. "
            "Consult a qualified healthcare professional for medical advice."
        )
    }