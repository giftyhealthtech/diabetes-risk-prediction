# Diabetes Classification API

A machine learning project that uses selected variables from the **2024 Behavioral Risk Factor Surveillance System (BRFSS)** dataset to classify respondents according to self-reported diabetes status.

The project includes data preprocessing, model training, prediction, a REST API built with FastAPI, automated tests, Docker containerization, and cloud deployment on Render.

**Live API:** https://diabetes-risk-prediction-tmkn.onrender.com  
**API Documentation:** https://diabetes-risk-prediction-tmkn.onrender.com/docs  
**Health Check:** https://diabetes-risk-prediction-tmkn.onrender.com/health

## Problem Statement

Diabetes is common, and many people are not screened until late. This project builds a simple model that uses basic survey answers (age, BMI, general health, lifestyle) to flag respondents whose profile looks like that of people who reported diabetes.

## Project Overview

This project explores how machine learning can identify patterns associated with self-reported diabetes status using selected demographic, health, and lifestyle variables.

The final model is a **Logistic Regression classifier** with a classification threshold of **0.15**.

### Test Performance (threshold = 0.15)

| Metric | Result |
|---|---:|
| ROC-AUC | 0.793 |
| Recall | 0.738 |
| Precision | 0.305 |
| F1-score | 0.432 |

The 0.15 threshold prioritises recall, so the model finds most respondents who reported diabetes but also produces many false alarms (about 7 in 10 flagged respondents did not report diabetes). It is a screening-style filter, not a diagnostic tool.

The model predicts membership in the self-reported diabetes class; it does not predict future diabetes development or provide a medical diagnosis.

## Dataset

The project uses the 2024 Behavioral Risk Factor Surveillance System (BRFSS) survey dataset.

The selected variables are:

| Feature | Description |
|---|---|
| `age_group` | Respondent's age group |
| `sex` | Respondent's sex |
| `bmi` | Body Mass Index |
| `general_health` | Self-reported general health |
| `exercise_30d` | Exercise in the past 30 days |
| `smoking_status` | Smoking status |
| `heavy_drinker` | Heavy drinking classification |
| `heart_disease` | History of coronary heart disease |
| `diabetic` | Binary target variable |

### Target Variable

The original BRFSS variable `DIABETE4` was transformed into a binary target:

- `1`: Self-reported diabetes.
- `0`: No self-reported diabetes.

Respondents reporting pregnancy-only diabetes or prediabetes were excluded from the binary classification dataset.

### Processed Dataset

A reduced dataset containing only the selected features and target variable is provided as:

`diabetes_selected_data.csv`

The original full BRFSS dataset is not included in the repository because of its size.

## EDA Summary

- **Size after cleaning:** 457,670 rows reduced to 383,775 (respondents with prediabetes, pregnancy-only diabetes, or unknown/refused answers removed).
- **Class balance:** 85.1% no diabetes, 14.9% diabetes. The data is imbalanced, so accuracy alone is not a good metric.
- **Missing data:** BMI is missing for 5.4% of respondents, at a similar rate in both classes. It was filled using median imputation.
- **BMI:** Right-skewed (mean 28.5, median 27.4). One extreme value (99.8) looks unrealistic.
- **Strongest patterns:** Diabetes rate rises from 3.3% (excellent health) to 37.4% (poor health). It also rises with age, from 1.4% (18-24) to 24.9% (75-79). It is higher in obese respondents (23.4% vs 7.5% for normal weight), in those with heart disease (36.4% vs 13.4%), and in those who did not exercise (24.6% vs 12.0%).
- **Weak patterns:** Sex and smoking status showed small differences.
- **Surprising result:** Heavy drinkers had a lower diabetes rate (7.1% vs 15.4%). This is probably due to under-reporting or other factors, so it should not be read as "drinking protects against diabetes."
- **BMI and BMI category:** They are strongly correlated (0.83), so only numeric BMI is used.

## Machine Learning Approach

The project follows these steps:

1. Data exploration and preprocessing.
2. Feature selection.
3. 70/15/15 stratified train, validation and test split.
4. Smoking status one-hot encoded.
5. Missing-value imputation using the training data.
6. Model training using Logistic Regression, DecisionTreeClassifier, and RandomForestClassifier.
7. Model evaluation and threshold selection.
8. Final model refitted on training and validation data, then evaluated once on the test set.
9. Saving the trained model as a Joblib artifact.

### Final Model

| Component | Selection |
|---|---|
| Algorithm | Logistic Regression |
| Imputation | Median imputation |
| Classification threshold | 0.15 |
| Model artifact | `diabetes_model.joblib` |

The classification threshold determines how predicted probabilities are converted into binary class predictions.

## Project Structure

```text
diabetes-risk-prediction/
│
├── diabetes_selected_data.csv
├── diabetes_model.joblib
├── diabetes_risk_model.ipynb
├── Dockerfile
├── predict.py
├── prepare_data.py
├── requirements.txt
├── serve.py
├── test_api.py
└── train.py
```

| File | Purpose |
|---|---|
| `diabetes_selected_data.csv` | Reduced dataset containing selected variables |
| `diabetes_model.joblib` | Trained machine learning model |
| `diabetes_risk_model.ipynb` | Exploratory data analysis and experimentation |
| `train.py` | Trains and saves the final model |
| `predict.py` | Loads the model and performs predictions |
| `prepare_data.py` | Extract the selected variable from the original full BRFSS dataset |
| `serve.py` | FastAPI application |
| `test_api.py` | Automated API tests |
| `requirements.txt` | Project dependencies |
| `Dockerfile` | Docker configuration |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/giftyhealthtech/diabetes-risk-prediction.git
cd diabetes-risk-prediction
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the API Locally

Start the FastAPI server:

```bash
uvicorn serve:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks API health |
| POST | `/predict` | Returns a diabetes classification prediction |

### Example Prediction Request

```json
{
  "age_group": 3,
  "sex": 0,
  "bmi": 17.1,
  "general_health": 2,
  "exercise_30d": 0,
  "smoking_status": 2,
  "heavy_drinker": 0,
  "heart_disease": 0
}
```

```
curl -X 'POST' \
  'https://diabetes-risk-prediction-tmkn.onrender.com/predict' \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "age_group": 3,
  "sex": 0,
  "bmi": 17.1,
  "general_health": 2,
  "exercise_30d": 0,
  "smoking_status": 2,
  "heavy_drinker": 0,
  "heart_disease": 0
}
```

Use the interactive `/docs` page to verify the accepted input values and submit a prediction.

### Example Response

```json
{
  "prediction": 0,
  "probability": 0.0183,
  "probability_percentage": "1.83%",
  "message": "This does not rule out diabetes. Consider routine blood glucose testing and consult a healthcare professional if you have concerns.",
  "disclaimer": "This tool is for educational purposes only and is not a medical diagnostic tool."
}
```

The response message may vary depending on the prediction.

## Running Tests

Run the automated API tests with:

```bash
pytest test_api.py -v
```

The test suite covers health checks, successful predictions, invalid inputs, missing fields, and prediction errors.

## Docker

Build the Docker image:

```bash
docker build -t diabetes-prediction-api .
```

Run the container:

```bash
docker run -d --name diabetes-api -p 8000:10000 diabetes-prediction-api
```

Access the API at:

```text
http://localhost:8000/docs
```

## Cloud Deployment

The API is deployed on **Render** using Docker.

Live deployment:

https://diabetes-risk-prediction-tmkn.onrender.com

## Limitations and Disclaimer

- The model is trained on survey data and relies on self-reported diabetes status.
- Predictions represent estimated probabilities of belonging to the survey's diabetes class, not an individual's clinical probability of developing diabetes.
- The model is not a diagnostic tool and must not replace laboratory testing or professional medical evaluation.
- Model performance may vary across populations and settings.
- Key factors like blood sugar, family history, blood pressure, and diet are not included.
- BRFSS covers US adults, so the model may not work well for other populations.
- The model cannot identify early-stage or borderline cases.
- Type 1 and Type 2 are grouped together.

## Next Steps

- Add more features (for example blood pressure, cholesterol, family history) to improve precision.
- Try other models, such as gradient boosting, and compare them with Logistic Regression.
- Handle the BMI outlier and test other ways of filling missing values.
- Add input range checks (for example realistic BMI values) in the API.
- Check performance across age and sex groups.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- FastAPI
- Pydantic
- Pytest
- Docker
- Render

## Author

**Sekinat Oyelami**

Medical Laboratory Scientist | Backend Developer | Healthcare Data Science

Building at the intersection of laboratory science, data, and HealthTech.