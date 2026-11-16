
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# 1. CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "diabetes_selected_data.csv"
MODEL_PATH = BASE_DIR  / "diabetes_model.joblib"


THRESHOLD = 0.15
RANDOM_STATE = 42


# 2. LOAD AND PREPARE DATA

def load_data():

    columns = [
        "DIABETE4",
        "_AGEG5YR",
        "_SEX",
        "_BMI5",
        "GENHLTH",
        "EXERANY2",
        "_SMOKER3",
        "_RFDRHV9",
        "CVDCRHD4",
    ]

    df = pd.read_csv(DATA_PATH, usecols=columns)

    df = df.rename(columns={
        "DIABETE4": "diabetes_status",
        "_AGEG5YR": "age_group",
        "_SEX": "sex",
        "_BMI5": "bmi",
        "GENHLTH": "general_health",
        "EXERANY2": "exercise_30d",
        "_SMOKER3": "smoking_status",
        "_RFDRHV9": "heavy_drinker",
        "CVDCRHD4": "heart_disease",
    })

    # Convert BMI to conventional scale.
    df["bmi"] = df["bmi"] / 100

    # Retain diabetes and non-diabetes respondents.
    df = df[df["diabetes_status"].isin([1, 3])].copy()

    # Create binary target.
    df["diabetic"] = (df["diabetes_status"] == 1).astype(int)

    # Remove invalid survey responses.
    df = df[df["general_health"].isin([1, 2, 3, 4, 5])]
    df = df[df["exercise_30d"].isin([1, 2])]
    df = df[df["heart_disease"].isin([1, 2])]
    df = df[df["age_group"].between(1, 13)]
    df = df[df["smoking_status"].isin([1, 2, 3, 4])]
    df = df[df["heavy_drinker"].isin([1, 2])]

    # Remove original target variable.
    df = df.drop(columns=["diabetes_status"])

    # Recode binary variables.
    df["sex"] = (df["sex"] == 2).astype(int)

    df["exercise_30d"] =  (df["exercise_30d"] == 1).astype(int)

    df["heart_disease"] = (df["heart_disease"] == 1).astype(int)

    df["heavy_drinker"] = (df["heavy_drinker"] == 2).astype(int)

    # Treat multi-category variables as categorical features.
    df['smoking_status'] = df['smoking_status'].astype(object) 

    # Retain ordinal features.
    for column in ["general_health", "age_group"]:
        df[column] = df[column].astype(int)   

    return df


# 3. SPLIT DATA

def split_data(df):

    X = df.drop(columns=["diabetic"])
    y = df["diabetic"]

    # 70% training, 30% temporary.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Divide remaining 30% equally.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# 4. BUILD PREPROCESSING PIPELINE

def build_pipeline():

    preprocessor = ColumnTransformer(
        transformers=[
            ("bmi_num", SimpleImputer(strategy="median"), ["bmi"]),
            ("smoke_encode",
             OneHotEncoder(handle_unknown="ignore", drop="first"),["smoking_status"])
        ],
        remainder="passthrough"
    )

    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight=None,
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


# 5. TRAIN FINAL MODEL

def train_model():

    print("Loading dataset...")

    df = load_data()

    print("Dataset shape:", df.shape)

    X_train, X_val, X_test, y_train, y_val, y_test = (split_data(df))

    # Combine training and validation data.
    X_train_final = pd.concat([X_train, X_val], axis=0)

    y_train_final = pd.concat([y_train, y_val], axis=0)

    print("Training observations:", len(X_train_final))
    print("Test observations:", len(X_test))

    print("\nBuilding model pipeline...")

    pipeline = build_pipeline()

    print("Training Logistic Regression...")

    pipeline.fit(X_train_final, y_train_final)

    print("Training completed.")


    # 6. FINAL TEST EVALUATION

    print("\nEvaluating on untouched test data...")

    y_probability = pipeline.predict_proba(X_test)[:, 1]

    y_prediction = (y_probability >= THRESHOLD).astype(int)

    print("\nClassification threshold:", THRESHOLD)

    print("Accuracy:", round(accuracy_score(y_test, y_prediction), 4))

    print("Precision:", round(precision_score(y_test, y_prediction), 4))

    print("Recall:",round(recall_score(y_test, y_prediction), 4))

    print("F1-score:", round(f1_score(y_test, y_prediction), 4))

    print("ROC-AUC:",round(roc_auc_score(y_test, y_probability), 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_prediction))

    print("\nClassification Report:")
    print(classification_report(y_test, y_prediction, zero_division=0))


    # 7. SAVE MODEL ARTIFACT

    artifact = {
        "model": pipeline,
        "threshold": THRESHOLD,
        "features": X_train.columns.tolist(),
        "model_name": "Logistic Regression",
        "target": "diabetic"
    }

    joblib.dump(artifact, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


# 8. EXECUTION

if __name__ == "__main__":
    train_model()