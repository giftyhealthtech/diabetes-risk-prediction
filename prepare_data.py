import pandas as pd

INPUT_FILE = "brfss_survey_data_2024.csv"
OUTPUT_FILE = "diabetes_selected_data.csv"

selected_columns = [
    "_AGEG5YR",
    "_SEX",
    "_BMI5",
    "GENHLTH",
    "EXERANY2",
    "_SMOKER3",
    "_RFDRHV9",
    "CVDCRHD4",
    "_BMI5CAT",
    "DIABETE4",
]

df = pd.read_csv(INPUT_FILE, usecols=selected_columns)


df.to_csv(OUTPUT_FILE, index=False)

print(f"Processed dataset saved as {OUTPUT_FILE}")
print(f"Dataset shape: {df.shape}")
print(df.head())