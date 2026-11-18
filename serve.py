import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from predict import predict


app = FastAPI(
    title="Diabetes Prediction API",
    description="API for predicting self-reported diabetes status using a trained Logistic Regression model.",
    version="1.0.0",
)


class DiabetesInput(BaseModel):
    age_group: int
    sex: int
    bmi: Optional[float] = Field(None, ge=10, le=100)
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
        data_dict = data.model_dump()

        # If BMI is missing, the model's median imputation will fill it
        if data_dict["bmi"] is None:
            data_dict["bmi"] = np.nan

        result = predict(data_dict)

        probability = float(result.iloc[0]["diabetes_probability"])
        prediction = int(result.iloc[0]["predicted_diabetes"])
        chance = f"{probability * 100:.2f}%"

        if prediction == 1:
            diabetes_status = (
                "You may be at risk of diabetes, as your answers are "
                "similar to people who reported diabetes."
            )
            explanation = (
                f"About {chance} of people with answers like yours reported "
                "having diabetes, which is higher than average."
            )
            message = (
                "This result does not confirm that you have diabetes. "
                "Please visit a healthcare facility for appropriate "
                "blood glucose testing and professional evaluation."
            )
        else:
            diabetes_status = (
                "You may not be at high risk of diabetes, as your answers are "
                "not very similar to people who reported diabetes."
            )
            explanation = (
                f"Only about {chance} of people with answers like yours reported "
                "having diabetes, which is low."
            )
            message = (
                "This does not rule out diabetes. Consider routine "
                "blood glucose testing and consult a healthcare "
                "professional if you have concerns."
            )

        return {
            "diabetes_status": diabetes_status,
            "chance_of_diabetes": chance,
            "explanation": explanation,
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